#!/usr/bin/env python3
"""Re-run speed cells whose windows overlapped outside GPU work, in clean windows."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import run_qualification as guard
import runtime as rt
import steady_metrics

PARENT = Path(os.environ['PARENT_ROOT'])
QUIET_SECONDS = 30
MAX_WAIT_SECONDS = 900


def isolation_rows(root: Path) -> list[dict]:
    path = root / 'gpu-isolation-events.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def covered_window(rows: list[dict], start: float, end: float) -> list[dict] | None:
    window = [row for row in rows if start <= row['timestamp'] <= end]
    if len(window) < 2 or window[0]['timestamp'] - start > 10 or end - window[-1]['timestamp'] > 10:
        return None
    if any(b['timestamp'] - a['timestamp'] > 10 for a, b in zip(window, window[1:])):
        return None
    return window


def contaminated_cells() -> list[dict]:
    rows = isolation_rows(PARENT)
    cells = []
    for command_path in sorted(PARENT.glob('*.bench.command.json')):
        stem = command_path.name[:-len('.bench.command.json')]
        if stem.startswith('smoke'):
            continue
        command = json.loads(command_path.read_text())
        window = covered_window(rows, command['started_at'], command['finished_at'])
        dirty = window is None or any(row['foreign'] or row.get('speed_eligible', True) is not True for row in window)
        needs_verifier_window = stem.startswith('acceptance-')
        if (not dirty and not needs_verifier_window) or command['returncode'] != 0:
            continue
        boot_label = stem.split('-repeat')[0]
        launch_path = PARENT / f'{boot_label}.launch.json'
        if not launch_path.exists():
            raise RuntimeError(f'Cannot reconstruct a required speed rerun: missing {launch_path}')
        cells.append({'result_label': stem, 'boot_label': boot_label, 'launch': json.loads(launch_path.read_text()), 'bench_args': command['args'], 'reason': 'foreign GPU work, unhealthy GPU, or missing coverage' if dirty else 'complete steady-window verifier counters for matched acceptance'})
    return cells


def bench_options(args: list[str]) -> dict:
    options = {}
    for flag, key in (('--concurrency', 'conc'), ('--contexts', 'contexts'), ('--duration', 'duration')):
        if flag in args:
            options[key] = args[args.index(flag) + 1]
    options['duration'] = int(options.get('duration', 30))
    options['prefill'] = '--skip-prefill' not in args
    return options


def wait_for_quiet() -> dict:
    started = time.monotonic()
    quiet_since = None
    while time.monotonic() - started < MAX_WAIT_SECONDS:
        foreign = guard.foreign_gpu_processes()
        now = time.monotonic()
        if foreign or not guard.gpu_recovery_health()['healthy']:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= QUIET_SECONDS:
            return {'waited_seconds': now - started, 'clean_start': True}
        time.sleep(5)
    return {'waited_seconds': time.monotonic() - started, 'clean_start': False}


def cell_clean(result_label: str) -> bool:
    command = json.loads((rt.ROOT / f'{result_label}.bench.command.json').read_text())
    window = covered_window(isolation_rows(rt.ROOT), command['started_at'], command['finished_at'])
    return window is not None and not any(row['foreign'] or row.get('speed_eligible', True) is not True for row in window)


def preserve_attempt(label: str, attempt: int) -> str:
    target = rt.ROOT / 'attempts' / label / f'attempt-{attempt}'
    target.mkdir(parents=True, exist_ok=False)
    for path in rt.ROOT.glob(f'{label}.*'):
        if path.is_file():
            shutil.copyfile(path, target / path.name)
    return str(target)


def resume_state(cells: list[dict]) -> tuple[dict, set[str]]:
    """Reuse only canonical receipts that still satisfy the original contract."""
    path = rt.ROOT / 'clean-rerun-plan.json'
    ledger = json.loads(path.read_text()) if path.exists() else {
        'parent_root': str(PARENT), 'planned': len(cells), 'reruns': [],
    }
    if ledger.get('parent_root') != str(PARENT) or ledger.get('planned') != len(cells):
        raise RuntimeError('Existing clean-rerun ledger belongs to a different plan')
    previous = {cell['result_label'] for group in ledger['reruns']
                for cell in group['cells'] if cell.get('clean')}
    completed: set[str] = set()
    for cell in cells:
        label = cell['result_label']
        if label not in previous:
            continue
        try:
            launch = json.loads((rt.ROOT / f"{cell['boot_label']}.launch.json").read_text())
            expected = dict(cell['launch'])
            if 'model_dir' not in expected or 'draft_dir' not in expected:
                inspected = json.loads((PARENT / f"{cell['boot_label']}.inspect.json").read_text())
                mounts = {mount['Destination']: mount['Source'] for mount in inspected['Mounts']}
                expected.setdefault('model_dir', mounts['/model'])
                expected.setdefault('draft_dir', mounts['/draft-mxfp8'])
            matched = all(launch.get(key) == expected.get(key) for key in (
                'image', 'tp', 'dcp', 'spec', 'cache', 'kv', 'env',
                'extra_args', 'model_dir', 'draft_dir', 'l2_host',
            ))
            command = json.loads((rt.ROOT / f'{label}.bench.command.json').read_text())
            summary = json.loads((rt.ROOT / f'{label}.steady-summary.json').read_text())
            counters_valid = bool(summary.get('all_windows_valid') and summary.get('cells'))
            if launch['spec'] != 'mtp0':
                counters_valid = counters_valid and all(
                    row.get('aggregate_verifier_steps_per_second', 0) > 0
                    for row in summary['cells']
                )
            if (matched and command['returncode'] == 0 and counters_valid
                    and bench_options(command['args']) == bench_options(cell['bench_args'])
                    and cell_clean(label)):
                completed.add(label)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for group in ledger['reruns']:
        for cell in group['cells']:
            if cell.get('clean') and cell['result_label'] not in completed:
                cell['clean'] = False
                cell['resume_validation_failed'] = True
    return ledger, completed


def next_attempt_number(label: str) -> int:
    parent = rt.ROOT / 'attempts' / label
    previous = [int(path.name.removeprefix('attempt-')) for path in parent.glob('attempt-*')
                if path.name.removeprefix('attempt-').isdigit()]
    return max(previous, default=0) + 1


def main() -> None:
    cells = contaminated_cells()
    prior = rt.ROOT / 'clean-rerun-plan.json'
    if prior.exists():
        rt.save_json(f'resume-ledgers/{time.time_ns()}.json', json.loads(prior.read_text()))
    ledger, completed_labels = resume_state(cells)
    attempt_started_at = time.time()
    rt.save_json('clean-rerun-completed.json', {
        'complete': False, 'planned': len(cells), 'clean': len(completed_labels),
        'attempt_started_at': attempt_started_at, 'state': 'running',
    })
    rt.note(f'CLEAN RESUME: preserving {len(completed_labels)}/{len(cells)} verified measurements')
    rt.save_json('clean-rerun-plan.json', ledger)
    by_boot: dict[str, list[dict]] = {}
    for cell in cells:
        if cell['result_label'] in completed_labels:
            continue
        by_boot.setdefault(cell['boot_label'], []).append(cell)
    for boot_label, group in by_boot.items():
        launch = group[0]['launch']
        extra_env = dict(launch['env'])
        if launch.get('l2_host'):
            extra_env['LMCACHE_L2_HOST_DIR'] = launch['l2_host']
        entry = {'boot_label': boot_label, 'image': launch['image'], 'cells': []}
        try:
            entry['quiet_wait'] = wait_for_quiet()
            if not entry['quiet_wait']['clean_start']:
                entry['blocked'] = 'No quiet GPU window; no speed measurement was started'
                rt.record_gate('clean-rerun-resource-availability', False, entry)
                break
            if not rt.boot(boot_label, image=launch['image'], tp=launch['tp'], dcp=launch['dcp'], spec=launch['spec'], cache=launch['cache'], kv=launch['kv'], extra_env=extra_env, extra_args=launch.get('extra_args') or None, model=Path(launch['model_dir']) if launch.get('model_dir') else None):
                entry['booted'] = False
                continue
            entry['booted'] = True
            for cell in group:
                options = bench_options(cell['bench_args'])
                outcome = {'result_label': cell['result_label'], 'attempts': []}
                first_attempt = next_attempt_number(cell['result_label'])
                for attempt in range(first_attempt, first_attempt + 2):
                    wait = wait_for_quiet()
                    if not wait['clean_start']:
                        outcome['attempts'].append({'attempt': attempt, 'quiet_wait': wait, 'executed': False, 'clean': False})
                        break
                    metric_path = rt.ROOT / f"{cell['result_label']}.steady.metrics.jsonl"
                    with steady_metrics.Recorder(rt.BASE_URL, metric_path):
                        ok = rt.bench(cell['result_label'], **options)
                    try:
                        summary = steady_metrics.summarize(rt.ROOT, cell['result_label']) if ok else {}
                        counters_valid = summary.get('all_windows_valid', False)
                        if launch['spec'] != 'mtp0':
                            counters_valid = counters_valid and all(
                                row.get('aggregate_verifier_steps_per_second', 0) > 0
                                for row in summary.get('cells', [])
                            )
                    except Exception as error:
                        counters_valid = False
                        summary = {'error': repr(error)}
                        rt.save_json(f"{cell['result_label']}.steady-summary.json", summary)
                    clean = ok and counters_valid and cell_clean(cell['result_label'])
                    archive = preserve_attempt(cell['result_label'], attempt)
                    outcome['attempts'].append({'attempt': attempt, 'quiet_wait': wait, 'executed': ok, 'counter_windows_valid': counters_valid, 'clean': clean, 'archive': archive})
                    if clean:
                        break
                outcome['clean'] = any(a['clean'] for a in outcome['attempts'])
                rt.record_gate('clean-rerun:' + cell['result_label'], outcome['clean'], outcome)
                entry['cells'].append(outcome)
        except Exception as error:
            entry['error'] = repr(error)
            rt.record_gate('clean-rerun-boot:' + boot_label, False, entry)
        finally:
            rt.stop()
            ledger['reruns'].append(entry)
            rt.save_json('clean-rerun-plan.json', ledger)
    clean_count = len({c['result_label'] for e in ledger['reruns'] for c in e['cells'] if c['clean']})
    complete = clean_count == len(cells)
    if not complete:
        rt.save_json('clean-rerun-completed.json', {
            'complete': False, 'planned': len(cells), 'clean': clean_count,
            'attempt_started_at': attempt_started_at, 'finished_at': time.time(),
        })
        raise SystemExit(2)
    import final_runtime_diagnostics
    final_runtime_diagnostics.main()
    rt.save_json('clean-rerun-completed.json', {
        'complete': True, 'planned': len(cells), 'clean': clean_count,
        'final_diagnostics_attempted': True,
        'attempt_started_at': attempt_started_at, 'finished_at': time.time(),
    })


if __name__ == '__main__':
    main()
