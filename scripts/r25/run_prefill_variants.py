#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
IMAGE = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
NAME = 'r25-prefill'
PORT = 5001
SAMPLER = ROOT / 'prefill_repeat.py'


def remove_server() -> None:
    subprocess.run(
        ['docker', 'rm', '-f', NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
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


def boot(label: str, cache_mode: str, l2_enabled: bool, l2_root: Path | None) -> None:
    remove_server()
    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host',
    ]
    if cache_mode == 'vram':
        args += ['--ipc', 'host', '--shm-size', '32g']
    else:
        args += ['--shm-size', '128g']
    args += [
        '-v', f'{MODEL}:/model:ro',
        '-v', 'r25-runtime-cache:/cache',
        '-e', 'MODEL=/model',
        '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=4',
        '-e', f'CACHE_MODE={cache_mode}', '-e', 'KV_CACHE_QUANT=nvfp4_ds_mla',
        '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        '-e', 'MAX_MODEL_LEN=1048576', '-e', 'MAX_NUM_SEQS=32',
        '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
        '-e', 'FAIRNESS_ENGINE=compute_share', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
        '-e', f"GPU_MEMORY_UTILIZATION={'0.95' if cache_mode == 'lmcache' else '0.93'}",
        '-e', 'SPECULATOR=mtp', '-e', 'MTP_DEPTH=0',
    ]
    if cache_mode == 'lmcache':
        assert l2_root is not None
        l2_root.mkdir(parents=True, exist_ok=True)
        args += [
            '-v', f'{l2_root}:/lmcache-l2',
            '-e', 'LMCACHE_CHUNK_SIZE=4096',
            '-e', 'LMCACHE_TARGET_TOKEN_BUDGET=4096',
            '-e', 'LMCACHE_L1_SIZE_GB=64',
            '-e', f'LMCACHE_L2_ENABLED={1 if l2_enabled else 0}',
            '-e', 'LMCACHE_L2_ROOT=/lmcache-l2',
        ]
    args.append(IMAGE)
    with (ROOT / f'prefill-{label}.boot.log').open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    wait_health()
    with (ROOT / f'prefill-{label}.docker.log').open('w') as output:
        subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)


def sample(label: str) -> None:
    output = ROOT / f'prefill-{label}.json'
    with (ROOT / f'prefill-{label}.bench.log').open('w') as log:
        subprocess.run(
            ['python3', str(SAMPLER), str(PORT), label, str(output), '8'],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1200,
        )
    with (ROOT / f'prefill-{label}.docker.log').open('w') as log:
        subprocess.run(['docker', 'logs', NAME], stdout=log, stderr=subprocess.STDOUT, check=False)


fresh_l2 = Path('/mnt/2king/lmcache-l2-r25-fresh')
if fresh_l2.exists():
    shutil.rmtree(fresh_l2)
fresh_l2.mkdir(parents=True)

arms = [
    ('gpu-only', 'vram', False, None),
    ('lmcache-existing-l2', 'lmcache', True, Path('/mnt/2king/lmcache-l2')),
    ('lmcache-ram-only', 'lmcache', False, Path('/mnt/2king/lmcache-l2-r25-ram')),
    ('lmcache-fresh-l2', 'lmcache', True, fresh_l2),
]
try:
    for arm in arms:
        label, cache_mode, l2_enabled, l2_root = arm
        print(f'PREFILL ARM START {label}', flush=True)
        boot(label, cache_mode, l2_enabled, l2_root)
        sample(label)
        print(f'PREFILL ARM DONE {label}', flush=True)
finally:
    remove_server()
(ROOT / 'PREFILL_VARIANTS_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print('PREFILL VARIANTS DONE', flush=True)
