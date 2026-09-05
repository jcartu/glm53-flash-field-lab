#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
IMAGE = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
NAME = 'r25-gap'
PORT = 5001
BENCH = '/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py'
LABEL = 'dcp2-vram-fp8-mtp3'

env = {
    **os.environ,
    'https_proxy': 'http://127.0.0.1:9',
    'http_proxy': 'http://127.0.0.1:9',
    'no_proxy': 'localhost,127.0.0.1',
    'NO_PROXY': 'localhost,127.0.0.1',
}
subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
args = [
    'docker', 'run', '-d', '--name', NAME, '--init',
    '--gpus', '"device=0,1,2,3"', '--network', 'host', '--ipc', 'host', '--shm-size', '32g',
    '-v', f'{MODEL}:/model:ro', '-v', 'r25-runtime-cache:/cache',
    '-e', 'MODEL=/model', '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
    '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=2',
    '-e', 'CACHE_MODE=vram', '-e', 'KV_CACHE_QUANT=fp8_ds_mla',
    '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
    '-e', 'MAX_MODEL_LEN=1048576', '-e', 'MAX_NUM_SEQS=32',
    '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
    '-e', 'FAIRNESS_ENGINE=compute_share', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
    '-e', 'GPU_MEMORY_UTILIZATION=0.93', '-e', 'DCP_CKV_GATHER=auto',
    '-e', 'SPECULATOR=mtp', '-e', 'MTP_DEPTH=3',
    IMAGE,
]
try:
    print('DCP2 MTP3 START', flush=True)
    with (ROOT / f'{LABEL}.boot.log').open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    deadline = time.time() + 900
    while time.time() < deadline:
        probe = subprocess.run(
            ['curl', '-fsS', '--max-time', '4', f'http://127.0.0.1:{PORT}/health'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            break
        live = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        if NAME not in live:
            raise RuntimeError(f'{NAME} exited before health')
        time.sleep(10)
    else:
        raise RuntimeError(f'{NAME} did not become healthy')
    with (ROOT / f'{LABEL}.bench.log').open('w') as output:
        subprocess.run(
            [
                'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
                '--concurrency', '1,4,8,16', '--contexts', '0,32k', '--duration', '30',
                '--max-tokens', '8192', '--output', str(ROOT / f'{LABEL}.json'),
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            check=True,
            timeout=1200,
        )
    with (ROOT / f'{LABEL}.docker.log').open('w') as output:
        subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
    with (ROOT / f'{LABEL}.inspect.json').open('w') as output:
        subprocess.run(['docker', 'inspect', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
finally:
    subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
(ROOT / 'DCP2_MTP3_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print('DCP2 MTP3 DONE', flush=True)
