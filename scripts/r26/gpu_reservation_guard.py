#!/usr/bin/python
"""Keep automation browser jobs off explicitly reserved GLM qualification GPUs."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import psutil
import pynvml as nvml

BROWSER_NAMES = {'chrome', 'chromium', 'chromium-browser', 'headless_shell'}


def automation_owner(pid: int) -> dict | None:
    try:
        process = psutil.Process(pid)
        if process.name() not in BROWSER_NAMES:
            return None
        for candidate in [process, *process.parents()]:
            if candidate.name() not in BROWSER_NAMES:
                continue
            args = candidate.cmdline()
            if any(arg.startswith('--type=') for arg in args):
                continue
            profile = next((arg.split('=', 1)[1] for arg in args
                            if arg.startswith('--user-data-dir=')), None)
            headless = any(arg.startswith('--headless') for arg in args)
            temporary_debug_profile = bool(
                profile and Path(profile).resolve().is_relative_to('/tmp')
                and '--no-sandbox' in args
                and any(arg.startswith('--remote-debugging-') for arg in args)
            )
            if not headless and not temporary_debug_profile:
                continue
            return {'pid': candidate.pid, 'created_at': candidate.create_time(),
                    'gpu_pid': pid, 'gpu_created_at': process.create_time(), 'name': candidate.name(),
                    'profile': profile, 'headless': headless,
                    'temporary_debug_profile': temporary_debug_profile}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return None


def inspect_consumers() -> list[dict]:
    pids = set()
    for index in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        pids.update(row.pid for row in nvml.nvmlDeviceGetComputeRunningProcesses(handle))
        pids.update(row.pid for row in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
    owners = {}
    for pid in pids:
        owner = automation_owner(pid)
        if owner is not None:
            owners[(owner['pid'], owner['created_at'])] = owner
    return list(owners.values())


def stop_verified_owner(owner: dict) -> dict:
    record = {**owner, 'timestamp': time.time(), 'stopped': False}
    try:
        root = psutil.Process(owner['pid'])
        verified = automation_owner(owner['gpu_pid'])
        if (root.create_time() != owner['created_at'] or verified is None
                or verified['pid'] != owner['pid'] or verified['created_at'] != owner['created_at']
                or verified['gpu_created_at'] != owner['gpu_created_at']):
            record['reason'] = 'Identity changed; no signal sent'
            return record
        children = root.children(recursive=True)
        gpu_process = psutil.Process(owner['gpu_pid'])
        if gpu_process.create_time() != owner['gpu_created_at']:
            record['reason'] = 'GPU child identity changed; no signal sent'
            return record
        gpu_process.kill()
        record['gpu_child_released_immediately'] = True
        root.terminate()
        _, alive = psutil.wait_procs([root], timeout=2)
        for process in alive:
            if process.is_running() and process.create_time() == owner['created_at']:
                process.kill()
        # Only descendants captured from this verified headless browser tree.
        for process in children:
            try:
                if process.is_running() and process.name() in BROWSER_NAMES:
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(children, timeout=2)
        for process in remaining:
            try:
                if process.is_running() and process.name() in BROWSER_NAMES:
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        record['stopped'] = True
    except psutil.NoSuchProcess:
        record['stopped'] = True
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--max-hours', type=float, default=30)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    nvml.nvmlInit()
    if args.inspect_only:
        print(json.dumps({'scope': 'Physical-GPU automation browsers only (headless or temporary CDP profiles); never normal desktop profiles or vLLM',
                          'candidates': inspect_consumers(), 'signals_sent': False}, indent=2))
        nvml.nvmlShutdown()
        return
    if not 0 < args.max_hours <= 48:
        parser.error('--max-hours must be in (0,48]')
    args.root.mkdir(parents=True, exist_ok=True)
    release = args.root / 'gpu-reservation-released.json'
    started = time.time()
    deadline = time.monotonic() + args.max_hours * 3600
    stop = False
    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    print('GPU RESERVATION GUARD READY: enforcing explicit user GLM priority; automation GPU browsers only', flush=True)
    while not stop and time.monotonic() < deadline:
        if release.exists() and release.stat().st_mtime >= started:
            break
        try:
            for owner in inspect_consumers():
                record = stop_verified_owner(owner)
                with (args.root / 'gpu-reservation-actions.jsonl').open('a') as output:
                    output.write(json.dumps(record) + '\n')
                print('GPU RESERVATION: ' + json.dumps(record), flush=True)
        except Exception as error:
            print('GPU RESERVATION OBSERVATION ERROR: ' + repr(error), flush=True)
        time.sleep(0.25)
    print('GPU RESERVATION GUARD EXIT: no browser profiles or project files were deleted', flush=True)
    nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
