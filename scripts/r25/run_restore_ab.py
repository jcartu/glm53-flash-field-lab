#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
DRAFT = '/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8'
L2 = '/mnt/2king/lmcache-l2'
NAME = 'r25-restore-ab'
PORT = 5001
PROBE = ROOT / 'replay_one_exact.py'
IMAGES = {
    'R24': 'voipmonitor/vllm@sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd',
    'R25': 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804',
}


def remove() -> None:
    subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(4)


def wait_health() -> None:
    deadline = time.time() + 900
    while time.time() < deadline:
        probe = subprocess.run(
            ['curl', '-fsS', '--max-time', '4', f'http://127.0.0.1:{PORT}/health'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return
        live = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        if NAME not in live:
            raise RuntimeError(f'{NAME} exited during startup')
        time.sleep(10)
    raise RuntimeError(f'{NAME} did not become healthy')


def boot(label: str, image: str) -> None:
    remove()
    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host', '--shm-size', '128g',
        '-v', f'{MODEL}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
        '-v', f'r25-restore-{label.lower()}-runtime:/cache',
        '-v', f'{L2}:/lmcache-l2',
        '-e', 'MODEL=/model', '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=4',
        '-e', 'CACHE_MODE=lmcache', '-e', 'KV_CACHE_QUANT=nvfp4_ds_mla',
        '-e', 'LMCACHE_CHUNK_SIZE=4096', '-e', 'LMCACHE_TARGET_TOKEN_BUDGET=4096',
        '-e', 'LMCACHE_L1_SIZE_GB=64', '-e', 'LMCACHE_L2_ENABLED=1',
        '-e', 'LMCACHE_L2_ROOT=/lmcache-l2',
        '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        '-e', 'MAX_MODEL_LEN=1048576', '-e', 'MAX_NUM_SEQS=32',
        '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
        '-e', 'FAIRNESS_ENGINE=compute_share', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
        '-e', 'GPU_MEMORY_UTILIZATION=0.95',
        '-e', 'SPECULATOR=dflash2', '-e', 'DFLASH_DEPTH=7',
        '-e', 'DFLASH_MODEL=/draft-mxfp8', '-e', 'DFLASH_MODEL_REVISION=',
        image,
    ]
    with (ROOT / f'restore-ab-{label.lower()}.boot.log').open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    wait_health()


def parse_log(text: str) -> dict[str, object]:
    retrieves = [
        {'tokens': int(tokens), 'seconds': float(seconds)}
        for tokens, seconds in re.findall(r'Retrieved (\d+) tokens in ([0-9.]+) seconds', text)
    ]
    prefetch_ms = [
        float(value)
        for value in re.findall(r'Prefetch request completed .*? in ([0-9.]+) ms', text)
    ]
    return {
        'retrieve_events': retrieves,
        'critical_retrieve_seconds': max((row['seconds'] for row in retrieves), default=None),
        'max_retrieved_tokens': max((row['tokens'] for row in retrieves), default=0),
        'prefetch_ms': prefetch_ms,
        'unpinned_fallback_warnings': text.count('Received unpinned CPU tensors'),
    }


rows: dict[str, dict[str, object]] = {}
try:
    print('RESTORE AB START', flush=True)
    for label, image in IMAGES.items():
        print(f'RESTORE AB ARM START {label}', flush=True)
        boot(label, image)
        result_path = ROOT / f'restore-ab-{label.lower()}.json'
        with (ROOT / f'restore-ab-{label.lower()}.probe.log').open('w') as output:
            subprocess.run(
                ['python3', str(PROBE), label, str(result_path), str(PORT)],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        logs = subprocess.run(
            ['docker', 'logs', NAME],
            capture_output=True,
            text=True,
            check=True,
        )
        log_text = logs.stdout + logs.stderr
        (ROOT / f'restore-ab-{label.lower()}.docker.log').write_text(log_text)
        row = json.loads(result_path.read_text())
        row.update(parse_log(log_text))
        rows[label] = row
        print(f'RESTORE AB ARM DONE {label}: {row}', flush=True)
finally:
    remove()

r24 = rows['R24']
r25 = rows['R25']
summary = {
    'method': 'matched exact 1,000,000-token filesystem-L2 restore, private 128 GiB SHM',
    'cache_salt': 'r25-exact-million-v1',
    'arms': rows,
    'comparison': {
        'critical_restore_speedup': (
            float(r24['critical_retrieve_seconds']) / float(r25['critical_retrieve_seconds'])
            if r24['critical_retrieve_seconds'] and r25['critical_retrieve_seconds'] else None
        ),
        'critical_restore_reduction_pct': (
            (1 - float(r25['critical_retrieve_seconds']) / float(r24['critical_retrieve_seconds'])) * 100
            if r24['critical_retrieve_seconds'] and r25['critical_retrieve_seconds'] else None
        ),
        'api_wall_reduction_pct': (1 - float(r25['wall_seconds']) / float(r24['wall_seconds'])) * 100,
        'visible_output_identical': r24['content'] == r25['content'],
    },
}
(ROOT / 'restore-ab-summary.json').write_text(json.dumps(summary, indent=2))
if int(r24['max_retrieved_tokens']) < 999_424 or int(r25['max_retrieved_tokens']) < 999_424:
    raise RuntimeError('matched restore did not retrieve the expected 999,424-token prefix')
if not summary['comparison']['visible_output_identical']:
    raise RuntimeError('R24 and R25 visible outputs differed')
(ROOT / 'RESTORE_AB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print('RESTORE AB DONE', flush=True)
