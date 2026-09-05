#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
IMAGE = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
DRAFT = '/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8'
L2 = '/mnt/2king/lmcache-l2-r25-hot'
NAME = 'r25-hot'
PORT = 5001
HARNESS = ROOT / 'agent_hot_queue.py'


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
            raise RuntimeError(f'{NAME} exited before health')
        time.sleep(10)
    raise RuntimeError(f'{NAME} did not become healthy')


def boot(policy: str) -> None:
    remove()
    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host', '--shm-size', '128g',
        '-v', f'{MODEL}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
        '-v', 'r25-hot-runtime:/cache', '-v', f'{L2}:/lmcache-l2',
        '-e', 'MODEL=/model', '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=4',
        '-e', 'CACHE_MODE=lmcache', '-e', 'KV_CACHE_QUANT=nvfp4_ds_mla',
        '-e', 'LMCACHE_CHUNK_SIZE=4096', '-e', 'LMCACHE_TARGET_TOKEN_BUDGET=4096',
        '-e', 'LMCACHE_L1_SIZE_GB=64', '-e', 'LMCACHE_L2_ENABLED=1',
        '-e', 'LMCACHE_L2_ROOT=/lmcache-l2',
        '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        '-e', 'MAX_MODEL_LEN=1048576', '-e', 'MAX_NUM_SEQS=32',
        '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
        '-e', f'FAIRNESS_ENGINE={policy}', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
        '-e', 'GPU_MEMORY_UTILIZATION=0.95',
        '-e', 'SPECULATOR=dflash2', '-e', 'DFLASH_DEPTH=7',
        '-e', 'DFLASH_MODEL=/draft-mxfp8', '-e', 'DFLASH_MODEL_REVISION=',
        IMAGE,
    ]
    with (ROOT / f'hot-{policy}.boot.log').open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    wait_health()


Path(L2).mkdir(parents=True, exist_ok=True)
try:
    for policy, label in [('none', 'off'), ('compute_share', 'compute-share-0.4')]:
        print(f'HOT POLICY START {label}', flush=True)
        boot(policy)
        with (ROOT / f'hot-{label}.bench.log').open('w') as output:
            subprocess.run(
                [
                    'python3', str(HARNESS),
                    '--base-url', f'http://127.0.0.1:{PORT}',
                    '--model', 'GLM-5.3-Flash-NVFP4',
                    '--policy-label', label,
                    '--concurrencies', '8,16',
                    '--repeats', '2',
                    '--session-context-tokens', '8192',
                    '--turn-decode-tokens', '128',
                    '--incremental-filler-tokens', '220',
                    '--baseline-seconds', '30',
                    '--storm-seconds', '60',
                    '--cold-prompt-tokens', '131072',
                    '--cold-period-seconds', '15',
                    '--output', str(ROOT / f'hot-{label}.json'),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        with (ROOT / f'hot-{label}.docker.log').open('w') as output:
            subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
        with (ROOT / f'hot-{label}.inspect.json').open('w') as output:
            subprocess.run(['docker', 'inspect', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
        print(f'HOT POLICY DONE {label}', flush=True)
finally:
    remove()
(ROOT / 'HOT_QUEUE_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print('HOT QUEUE DONE', flush=True)
