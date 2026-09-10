#!/usr/bin/env python3
"""After the followups restore production, rerun contaminated R26 speed cells cleanly."""
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
    parser.add_argument('--followups-root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    args = parser.parse_args()
    print('CLEAN RERUNS QUEUED: waiting for the followups to finish', flush=True)
    while True:
        try:
            process = psutil.Process(args.wait_pid)
            if abs(process.create_time() - args.wait_created) > 0.01 or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(15)
    executed_path = args.followups_root / 'qualification-executed.json'
    restored_path = args.followups_root / 'production-restored.json'
    if not executed_path.exists() or not restored_path.exists():
        raise RuntimeError('Followups did not complete and restore production; refusing clean reruns')
    executed = json.loads(executed_path.read_text())
    restored = json.loads(restored_path.read_text())
    if not executed.get('all_phases_attempted') or not restored.get('healthy'):
        raise RuntimeError('Followup completion/restoration receipts are invalid')
    if executed.get('finished_at', 0) < args.wait_created or restored.get('timestamp', 0) < executed['finished_at']:
        raise RuntimeError('Followup receipts are stale; refusing an unrelated GPU launch')
    output = args.parent_root / 'clean-reruns'
    output.mkdir(exist_ok=False)
    os.environ['BATTERY_ROOT'] = str(output)
    os.environ['PARENT_ROOT'] = str(args.parent_root)
    import run_qualification as coordinator
    coordinator.PHASES = [('clean-speed-reruns', 'clean_rerun_phase.py', [], 28800)]
    coordinator.main()
    print('CLEAN RERUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
