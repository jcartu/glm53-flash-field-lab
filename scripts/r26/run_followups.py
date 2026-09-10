#!/usr/bin/env python3
"""Wait for the full coordinator, then run the focused native/acceptance rechecks."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-created', type=float, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--phase-plan', type=Path, required=True)
    args = parser.parse_args()
    print('FOLLOWUPS QUEUED: waiting for the full qualification and production restoration', flush=True)
    while True:
        try:
            process = psutil.Process(args.wait_pid)
            if abs(process.create_time() - args.wait_created) > 0.01 or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(10)
    executed_path = args.parent_root / 'qualification-executed.json'
    restored_path = args.parent_root / 'production-restored.json'
    if not executed_path.exists() or not restored_path.exists():
        raise RuntimeError('Upstream qualification did not finish and restore production; followups will not auto-start')
    executed = json.loads(executed_path.read_text())
    restored = json.loads(restored_path.read_text())
    if not executed.get('all_phases_attempted') or not restored.get('healthy'):
        raise RuntimeError('Upstream completion/restoration receipts are not valid')
    if executed.get('finished_at', 0) < args.wait_created or restored.get('timestamp', 0) < executed['finished_at']:
        raise RuntimeError('Upstream receipts are stale; refusing an unrelated GPU launch')
    plan = json.loads(args.phase_plan.read_text())
    if plan.get('schema') != 'r26-followup-plan/v1':
        raise RuntimeError('Unknown followup plan schema')
    phases = []
    for row in plan.get('phases', []):
        script = Path(row['script'])
        if script.name != str(script) or not Path(__file__).with_name(script.name).is_file():
            raise RuntimeError(f'Followup script is unavailable: {script}')
        phases.append((row['name'], script.name, row.get('args', []), int(row['timeout_seconds'])))
    if not phases:
        raise RuntimeError('Followup plan is empty')
    output = args.parent_root / 'followups'
    output.mkdir(exist_ok=False)
    (output / 'executed-followup-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    os.environ['BATTERY_ROOT'] = str(output)
    import run_qualification as coordinator
    coordinator.PHASES = phases
    coordinator.main()
    print('FOCUSED FOLLOWUPS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
