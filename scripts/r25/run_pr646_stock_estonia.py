#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
IMAGE = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
DRAFT = '/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8'
BENCH = '/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py'
NAME = 'pr646-stock-quality'
PORT = 5001
PROXY_ENV = {
    **os.environ,
    'https_proxy': 'http://127.0.0.1:9',
    'http_proxy': 'http://127.0.0.1:9',
    'no_proxy': 'localhost,127.0.0.1',
    'NO_PROXY': 'localhost,127.0.0.1',
}
print('PR646 STOCK QUALITY CONTROL START', flush=True)
subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
args = [
    'docker', 'run', '-d', '--name', NAME, '--init',
    '--gpus', '"device=0,1,2,3"', '--network', 'host', '--ipc', 'host', '--shm-size', '32g',
    '-v', f'{MODEL}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
    '-v', 'r25-runtime-cache:/cache',
    '-e', 'MODEL=/model', '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
    '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=1',
    '-e', 'CACHE_MODE=vram', '-e', 'KV_CACHE_QUANT=fp8_ds_mla',
    '-e', 'GLM53_TARGET_BLOCK_SIZE=2048', '-e', 'GLM53_MAMBA_BLOCK_SIZE=2048',
    '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
    '-e', 'MAX_MODEL_LEN=262144', '-e', 'MAX_NUM_SEQS=32',
    '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
    '-e', 'FAIRNESS_ENGINE=compute_share', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
    '-e', 'GPU_MEMORY_UTILIZATION=0.93',
    '-e', 'SPECULATOR=dflash2', '-e', 'DFLASH_DEPTH=7',
    '-e', 'DFLASH_MODEL=/draft-mxfp8', '-e', 'DFLASH_MODEL_REVISION=',
    IMAGE, '--prefix-match-unit', '256',
]
try:
    with (ROOT / 'pr646-stock-estonia.boot.log').open('w') as output:
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
            raise RuntimeError('stock quality server exited before health')
        time.sleep(10)
    else:
        raise RuntimeError('stock quality server did not become healthy')
    for wave in (1, 2):
        with (ROOT / f'pr646-stock-estonia-wave{wave}.log').open('w') as output:
            subprocess.run(
                [
                    'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
                    '--test-profile', 'estonia', '--profile-runs', '6',
                    '--output', str(ROOT / f'pr646-stock-estonia-wave{wave}.json'),
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=PROXY_ENV,
                check=True,
                timeout=3600,
            )
    with (ROOT / 'pr646-stock-estonia.docker.log').open('w') as output:
        subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
finally:
    subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

waves = [
    json.loads((ROOT / f'pr646-stock-estonia-wave{wave}.json').read_text())['selected_summary']
    for wave in (1, 2)
]
summary_path = ROOT / 'pr646-summary.json'
summary = json.loads(summary_path.read_text())
correctness = summary['correctness']
correctness['stock_estonia_waves'] = 2
correctness['stock_estonia_correct'] = sum(wave['correct'] for wave in waves)
correctness['stock_estonia_total'] = sum(wave['attempted'] for wave in waves)
correctness['stock_estonia_errors'] = sum(wave['errors'] for wave in waves)
summary_path.write_text(json.dumps(summary, indent=2))
(ROOT / 'PR646_STOCK_QUALITY_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print(json.dumps(correctness, indent=2), flush=True)
