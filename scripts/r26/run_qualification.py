#!/usr/bin/env python3
"""Run each GPU phase serially; restore the existing production container."""
from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests
import runtime as rt

HERE = Path(__file__).resolve().parent
PRODUCTION = 'glm53-prod'
PHASES = [
    ('quality-and-mtp-dcp-correctness', 'quality_phase.py', [], 21600),
    ('cache-lifecycle-and-boundaries', 'cache_phase.py', [], 21600),
    ('smoke', 'matrix_phase.py', ['--section', 'smoke'], 1800),
    ('festr-matched-acceptance', 'matrix_phase.py', ['--section', 'priority'], 18000),
    ('complete-topology-matrix', 'matrix_phase.py', ['--section', 'matrix'], 21600),
    ('mixed-agent-scheduling', 'scheduler_phase.py', [], 21600),
    ('batch-dma-and-overlay-controls', 'matrix_phase.py', ['--section', 'tuning'], 10800),
    ('tp2-capacity', 'matrix_phase.py', ['--section', 'tp2'], 7200),
    ('direct-dcp-peer-guard', 'peer_phase.py', [], 7200),
]
ISOLATION_MODE = os.environ.get('GPU_ISOLATION_MODE', 'strict')
_OWNED_GPU_IDENTITIES: dict[int, float] = {}


def interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f'Qualification interrupted by signal {signum}')


def production_idle() -> bool:
    response = requests.get('http://127.0.0.1:5001/metrics', timeout=10)
    response.raise_for_status()
    counts = [float(match.group(1)) for match in re.finditer(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)', response.text, re.MULTILINE)]
    return len(counts) >= 2 and all(value == 0 for value in counts)


def gpu_recovery_health() -> dict:
    command = [
        'nvidia-smi',
        '--query-gpu=index,uuid,gpu_recovery_action',
        '--format=csv,noheader,nounits',
    ]
    try:
        queried = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=True,
        )
        gpus = []
        errors = []
        for line_number, fields in enumerate(csv.reader(queried.stdout.splitlines()), 1):
            fields = [field.strip() for field in fields]
            if len(fields) != 3 or not all(fields):
                errors.append(f'Malformed GPU recovery row {line_number}: {fields!r}')
                continue
            index, uuid, action = fields
            if not index.isdigit():
                errors.append(f'Invalid GPU index on row {line_number}: {index!r}')
                continue
            gpus.append({'index': int(index), 'uuid': uuid, 'recovery_action': action})
        if len(gpus) != 4:
            errors.append(f'Expected 4 readable GPU recovery rows, found {len(gpus)}')
        if {gpu['index'] for gpu in gpus} != {0, 1, 2, 3}:
            errors.append('GPU recovery rows did not contain distinct indices 0,1,2,3')
        if len({gpu['uuid'] for gpu in gpus}) != len(gpus):
            errors.append('GPU recovery rows did not contain distinct UUIDs')
        for gpu in gpus:
            if gpu['recovery_action'] != 'None':
                errors.append(
                    f"GPU {gpu['index']} recovery action is "
                    f"{gpu['recovery_action']!r}, not 'None'"
                )
        return {'healthy': not errors, 'gpus': gpus, 'errors': errors}
    except Exception as error:
        return {
            'healthy': False,
            'gpus': [],
            'errors': [f'GPU recovery query failed: {error!r}'],
        }


def foreign_gpu_processes() -> list[dict]:
    queried = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15, check=True,
    )
    gpu_pids = {int(line.strip()) for line in queried.stdout.splitlines() if line.strip().isdigit()}
    containers = []
    for filter_key in ('name', 'label'):
        subprocess_args = ['docker', 'ps']
        if filter_key == 'name':
            subprocess_args += ['--filter', f'name=^{rt.NAME}$']
        else:
            subprocess_args += ['--filter', 'label=field-lab.battery=r26']
        containers.extend(subprocess.run(
            subprocess_args + ['--format', '{{.ID}}'], capture_output=True, text=True,
            timeout=15, check=True,
        ).stdout.splitlines())
    owned_pids: set[int] = set()
    for container in containers:
        top = subprocess.run(['docker', 'top', container, '-eo', 'pid'],
                             capture_output=True, text=True, timeout=15)
        if top.returncode == 0:
            owned_pids.update(int(line.strip()) for line in top.stdout.splitlines()[1:] if line.strip().isdigit())
    for pid in set(_OWNED_GPU_IDENTITIES) - gpu_pids:
        del _OWNED_GPU_IDENTITIES[pid]
    foreign = []
    for pid in sorted(gpu_pids):
        try:
            process = psutil.Process(pid)
            created_at = process.create_time()
            if process.status() == psutil.STATUS_ZOMBIE:
                continue
            if pid in owned_pids:
                _OWNED_GPU_IDENTITIES[pid] = created_at
            # docker top can disappear before a restarting worker releases CUDA.
            # Only previously verified ownership with the same birth time survives.
            elif _OWNED_GPU_IDENTITIES.get(pid) != created_at:
                _OWNED_GPU_IDENTITIES.pop(pid, None)
                foreign.append({'pid': pid, 'name': process.name(), 'created_at': created_at})
        except psutil.NoSuchProcess:
            _OWNED_GPU_IDENTITIES.pop(pid, None)
    return foreign


def watch_gpu_isolation(stopped: threading.Event) -> None:
    previous: set[int] = set()
    health_warning_recorded = False
    while not stopped.wait(2):
        try:
            gpu_health = gpu_recovery_health()
            foreign = foreign_gpu_processes()
            current = {row['pid'] for row in foreign}
            observed_at = time.time()
            with (rt.ROOT / 'gpu-isolation-events.jsonl').open('a') as handle:
                handle.write(json.dumps({
                    'timestamp': observed_at,
                    'foreign': foreign,
                    'gpu_health': gpu_health,
                    'speed_eligible': gpu_health['healthy'] and not foreign,
                }) + '\n')
            if not gpu_health['healthy']:
                if ISOLATION_MODE == 'strict':
                    rt.save_json('gpu-health-interruption.json', {
                        'timestamp': observed_at,
                        'stage': 'monitor',
                        'gpu_health': gpu_health,
                        'result': 'Run interrupted; GPU recovery state is unhealthy and affected measurements must not qualify.',
                    })
                    rt.note('GPU RECOVERY STATE UNHEALTHY; stopping qualification and restoring production')
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                if not health_warning_recorded:
                    rt.note('GPU RECOVERY STATE UNHEALTHY AND RECORDED; speed measurements are not eligible')
                    health_warning_recorded = True
            else:
                health_warning_recorded = False
            # Functional checks can proceed under recorded background work.
            # Speed comparisons require a clean window and must be rerun when
            # the recorded window overlaps an outside GPU process.
            if current & previous and ISOLATION_MODE == 'strict':
                rt.save_json('gpu-isolation-interruption.json', {'timestamp': observed_at, 'foreign': foreign, 'result': 'Run interrupted; do not qualify affected measurements.'})
                rt.note('GPU ISOLATION LOST; stopping qualification and restoring production')
                os.kill(os.getpid(), signal.SIGINT)
                return
            previous = current
        except Exception as error:
            rt.save_json('gpu-isolation-monitor-error.json', {'error': repr(error), 'timestamp': time.time()})
            rt.note('GPU isolation could not be verified; stopping qualification')
            os.kill(os.getpid(), signal.SIGINT)
            return


def restore_production(production_id: str) -> None:
    deferred_signals: list[int] = []
    previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def defer_interrupt(signum: int, frame: object) -> None:
        deferred_signals.append(signum)

    for signum in previous_handlers:
        signal.signal(signum, defer_interrupt)
    try:
        cleanup_error: BaseException | None = None
        try:
            rt.stop()
        except BaseException as error:
            cleanup_error = error
        if rt._inspect() is not None:
            raise RuntimeError('Owned test container remains; refusing overlapping production restore') from cleanup_error
        # A stop accepted by Docker can outlive an interrupted/timed-out client.
        # Join that transition on the captured identity before starting it again.
        subprocess.run(['docker', 'stop', '--timeout', '60', production_id], check=True, timeout=90)
        subprocess.run(['docker', 'start', production_id], check=True, timeout=60)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                response = requests.get('http://127.0.0.1:5001/health', timeout=5)
                if response.status_code == 200:
                    models = requests.get('http://127.0.0.1:5001/v1/models', timeout=10)
                    models.raise_for_status()
                    rt.save_json('production-restored.json', {'container': PRODUCTION, 'container_id': production_id,
                        'healthy': True, 'models': models.json(), 'timestamp': time.time(),
                        'cleanup_error': repr(cleanup_error) if cleanup_error else None,
                        'deferred_signals': deferred_signals,
                        'policy': 'Original container identity restored, no candidate promotion.'})
                    rt.note('PRODUCTION RESTORED')
                    if cleanup_error is not None:
                        raise cleanup_error
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise RuntimeError('Production did not become healthy after restart')
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    rt.ROOT.mkdir(parents=True, exist_ok=True)
    rt.note('R26 QUALIFICATION START')
    rt.save_json('phase-plan.json', [{'name': name, 'script': script, 'args': args, 'timeout_seconds': timeout} for name, script, args, timeout in PHASES])
    stopped_production = False
    production_id: str | None = None
    completed = []
    guard_stop = threading.Event()
    guard: threading.Thread | None = None
    try:
        idle_deadline = time.monotonic() + 600
        while not production_idle():
            if time.monotonic() >= idle_deadline:
                raise RuntimeError('Production remained busy; refused to interrupt active user requests')
            rt.note('Waiting for production requests to drain')
            time.sleep(10)
        gpu_health = gpu_recovery_health()
        observed_at = time.time()
        rt.save_json('gpu-health-preflight.json', {
            'timestamp': observed_at,
            'gpu_health': gpu_health,
            'speed_eligible': gpu_health['healthy'],
        })
        if not gpu_health['healthy']:
            if ISOLATION_MODE == 'strict':
                rt.save_json('gpu-health-interruption.json', {
                    'timestamp': observed_at,
                    'stage': 'preflight',
                    'gpu_health': gpu_health,
                    'result': 'Run interrupted before pausing production; GPU recovery state is unhealthy.',
                })
                raise RuntimeError(f'Refusing unhealthy GPU recovery state: {gpu_health["errors"]}')
            rt.note('GPU RECOVERY STATE UNHEALTHY AND RECORDED; speed measurements are not eligible')
        original = subprocess.run(['docker', 'inspect', PRODUCTION], capture_output=True, text=True, check=True, timeout=30)
        production = json.loads(original.stdout)[0]
        if not production.get('State', {}).get('Running'):
            raise RuntimeError('Original production container is not running; refusing an ambiguous pause')
        production_id = production['Id']
        rt.save_json('production-before-pause.json', {'container_id': production_id,
            'name': production.get('Name'), 'image': production.get('Image')})
        stopped_production = True
        subprocess.run(['docker', 'stop', '--timeout', '60', production_id], check=True, timeout=90)
        rt.note(f'PRODUCTION PAUSED; GPU qualification begins (isolation={ISOLATION_MODE})')
        time.sleep(5)
        foreign = foreign_gpu_processes()
        if foreign and ISOLATION_MODE == 'strict':
            raise RuntimeError(f'Refusing a contaminated start: {foreign}')
        if foreign:
            rt.note('BACKGROUND GPU WORK RECORDED; overlapping speed samples are not qualified')
        guard = threading.Thread(target=watch_gpu_isolation, args=(guard_stop,), daemon=True)
        guard.start()
        for phase, script, args, timeout in PHASES:
            rt.note('QUALIFICATION PHASE ' + phase)
            rt.save_json('current-phase.json', {'phase': phase, 'script': script, 'started_at': time.time()})
            code = rt.run([sys.executable, str(HERE / script), *args], label='phase-' + phase, timeout=timeout)
            rt.record_gate('phase-execution:' + phase, code == 0, {'returncode': code, 'script': script})
            completed.append({'phase': phase, 'returncode': code})
            rt.save_json('phase-progress.json', completed)
            # Every phase owns the same labelled container; no phase may leave
            # one GPU runtime overlapping the next phase's boot.
            rt.stop()
        rt.save_json('qualification-executed.json', {'all_phases_attempted': True, 'phases': completed, 'finished_at': time.time()})
    except BaseException as error:
        rt.save_json('qualification-interrupted.json', {'error': repr(error), 'completed_phases': completed, 'timestamp': time.time()})
        raise
    finally:
        guard_stop.set()
        try:
            if guard is not None:
                guard.join(timeout=45)
        finally:
            if stopped_production:
                assert production_id is not None
                restore_production(production_id)
            rt.note('R26 QUALIFICATION COORDINATOR EXIT')


if __name__ == '__main__':
    main()
