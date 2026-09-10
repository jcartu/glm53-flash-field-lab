#!/usr/bin/env python3
"""Resume only missing R26 measurements after an exclusive GPU window opens."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

R26_IMAGE = 'voipmonitor/vllm@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68'


def foreign_from_production() -> list[dict]:
    """Inspect without stopping other projects or the original production server."""
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20).stdout
    pids = {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}
    top = subprocess.run(['docker', 'top', 'glm53-prod', '-eo', 'pid'],
                         capture_output=True, text=True, timeout=20)
    production = {int(line.strip()) for line in top.stdout.splitlines()[1:] if line.strip().isdigit()} if top.returncode == 0 else set()
    foreign = []
    for pid in sorted(pids - production):
        try:
            process = psutil.Process(pid)
            if process.status() != psutil.STATUS_ZOMBIE:
                foreign.append({'pid': pid, 'name': process.name(), 'created_at': process.create_time()})
        except psutil.NoSuchProcess:
            pass
    return foreign


def wait_for_window(root: Path) -> None:
    quiet_since = None
    previous = None
    while True:
        foreign = foreign_from_production()
        now = time.monotonic()
        if foreign:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        if not foreign and quiet_since is not None and now - quiet_since >= 30:
            return
        identity = tuple(row['pid'] for row in foreign)
        if identity != previous:
            status = {'timestamp': time.time(), 'foreign': foreign,
                      'policy': 'Waiting without stopping other projects; production is left running.'}
            path = root / 'waiting-for-exclusive-gpu.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(path)
            print('CLEAN RESUME WAIT: ' + json.dumps(status), flush=True)
            previous = identity
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path,
                        default=Path('/home/josh/omp-workspace/drock-lmcache/r26-battery'))
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--wait-created', type=float)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if (args.wait_pid is None) != (args.wait_created is None):
        parser.error('--wait-pid and --wait-created must be supplied together')
    parent = args.parent_root.resolve()
    root = parent / 'clean-reruns'
    if not (root / 'clean-rerun-plan.json').is_file():
        raise RuntimeError('No prior clean-run ledger to resume')
    if args.wait_pid is not None and not args.plan_only:
        print('CLEAN RESUME DEPENDENCY QUEUED: waiting for the preceding resume to finish', flush=True)
        from run_qad_battery import wait_for_process
        wait_for_process(args.wait_pid, args.wait_created)
        restored = json.loads((root / 'production-restored.json').read_text())
        if restored.get('healthy') is not True or restored.get('timestamp', 0) < args.wait_created:
            raise RuntimeError('Previous resume did not restore production; refusing a stale GPU handoff')
        prior_completion = json.loads((root / 'clean-rerun-completed.json').read_text())
        if (prior_completion.get('complete') is True
                and prior_completion.get('clean') == prior_completion.get('planned')
                and prior_completion.get('final_diagnostics_attempted') is True
                and prior_completion.get('finished_at', 0) >= args.wait_created):
            print('CLEAN RESUME COMPLETE: predecessor already finished all measurements and closing checks', flush=True)
            return
    os.environ.update({'BATTERY_ROOT': str(root), 'PARENT_ROOT': str(parent),
                       'BATTERY_CONTAINER': 'r26-test', 'BATTERY_PORT': '5002',
                       'BATTERY_IMAGE': R26_IMAGE})
    import clean_rerun_phase as clean
    import run_qualification as coordinator
    cells = clean.contaminated_cells()
    _, completed = clean.resume_state(cells)
    details = {'schema': 'r26-clean-resume/v1', 'parent_root': str(parent),
               'planned': len(cells), 'verified_completed': len(completed),
               'remaining': [c['result_label'] for c in cells if c['result_label'] not in completed],
               'gpu_jobs_started': False, 'other_projects_will_not_be_stopped': True}
    if args.plan_only:
        print(json.dumps(details, indent=2))
        return
    print('CLEAN RESUME QUEUED: waiting for an exclusive GPU window; successful measurements preserved', flush=True)
    wait_for_window(root)
    stamp = str(time.time_ns())
    archive = root / 'resume-history' / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('clean-rerun-plan.json', 'clean-rerun-completed.json', 'phase-plan.json',
                 'phase-progress.json', 'current-phase.json', 'qualification-executed.json',
                 'qualification-interrupted.json', 'production-restored.json'):
        source = root / name
        if source.exists():
            shutil.copy2(source, archive / name)
    (archive / 'resume-plan.json').write_text(json.dumps(details, indent=2) + '\n')
    attempt_started_at = time.time()
    coordinator.rt.save_json('clean-rerun-completed.json', {
        'complete': False, 'planned': len(cells), 'clean': len(completed),
        'attempt_started_at': attempt_started_at, 'state': 'phase-not-yet-started',
    })
    coordinator.PHASES = [('clean-speed-reruns-resume-' + stamp, 'clean_rerun_phase.py', [], 28800)]
    coordinator.main()
    receipt = json.loads((root / 'clean-rerun-completed.json').read_text())
    if (receipt.get('complete') is not True or receipt.get('clean') != receipt.get('planned')
            or receipt.get('finished_at', 0) < attempt_started_at):
        raise SystemExit('Clean resumption remains incomplete; downstream batteries must stay blocked')
    print('CLEAN RESUME COMPLETE: all required measurements and closing probes attempted', flush=True)


if __name__ == '__main__':
    main()
