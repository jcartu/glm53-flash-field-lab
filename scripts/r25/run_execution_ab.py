#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
BENCH = '/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py'
NAME = 'r25-execution-ab'
PORT = 5001
IMAGES = {
    'r25-a': 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804',
    'r24': 'voipmonitor/vllm@sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd',
    'r25-b': 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804',
}
PROXY_ENV = {
    **os.environ,
    'https_proxy': 'http://127.0.0.1:9',
    'http_proxy': 'http://127.0.0.1:9',
    'no_proxy': 'localhost,127.0.0.1',
    'NO_PROXY': 'localhost,127.0.0.1',
}


def remove() -> None:
    subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(4)


def boot(label: str, image: str) -> None:
    remove()
    runtime_volume = 'r24-runtime-cache' if label == 'r24' else 'r25-runtime-cache'
    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host', '--ipc', 'host', '--shm-size', '32g',
        '-v', f'{MODEL}:/model:ro', '-v', f'{runtime_volume}:/cache',
        '-e', 'MODEL=/model', '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=4',
        '-e', 'CACHE_MODE=vram', '-e', 'KV_CACHE_QUANT=fp8_ds_mla',
        '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        '-e', 'MAX_MODEL_LEN=1048576', '-e', 'MAX_NUM_SEQS=32',
        '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
        '-e', 'FAIRNESS_ENGINE=compute_share', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
        '-e', 'GPU_MEMORY_UTILIZATION=0.93', '-e', 'DCP_CKV_GATHER=auto',
        '-e', 'SPECULATOR=mtp', '-e', 'MTP_DEPTH=0',
        image,
    ]
    with (ROOT / f'execution-ab-{label}.boot.log').open('w') as output:
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
            return
        live = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        if NAME not in live:
            raise RuntimeError(f'{label} exited before health')
        time.sleep(10)
    raise RuntimeError(f'{label} did not become healthy')


def headline(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {
        f'C{row["concurrency"]}': float(row['aggregate_tps'])
        for row in data['results']
        if row.get('context_tokens') == 0 and row.get('concurrency') in {1, 8, 16}
    }


rows = {}
try:
    print('EXECUTION AB START', flush=True)
    for label, image in IMAGES.items():
        print(f'EXECUTION AB ARM START {label}', flush=True)
        boot(label, image)
        output_path = ROOT / f'execution-ab-{label}.json'
        with (ROOT / f'execution-ab-{label}.bench.log').open('w') as output:
            subprocess.run(
                [
                    'python3', BENCH, '--port', str(PORT),
                    '--model', 'GLM-5.3-Flash-NVFP4',
                    '--concurrency', '1,8,16', '--contexts', '0',
                    '--skip-prefill', '--duration', '30', '--max-tokens', '8192',
                    '--output', str(output_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=PROXY_ENV,
                check=True,
                timeout=900,
            )
        with (ROOT / f'execution-ab-{label}.docker.log').open('w') as output:
            subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
        rows[label] = headline(output_path)
        print(f'EXECUTION AB ARM DONE {label}: {rows[label]}', flush=True)
finally:
    remove()

r25_mean = {
    metric: statistics.fmean([rows['r25-a'][metric], rows['r25-b'][metric]])
    for metric in ('C1', 'C8', 'C16')
}
deltas = {
    metric: 100 * (r25_mean[metric] / rows['r24'][metric] - 1)
    for metric in r25_mean
}
report = {
    'method': 'R25, R24, R25 bracket on DCP4 FP8 no-spec; identical settings and same live host window',
    'arms': rows,
    'r25_bracket_mean': r25_mean,
    'r25_vs_r24_percent': deltas,
    'within_five_percent_all_cells': all(abs(value) <= 5 for value in deltas.values()),
}
(ROOT / 'execution-ab-summary.json').write_text(json.dumps(report, indent=2))
(ROOT / 'EXECUTION_AB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print(json.dumps(report, indent=2), flush=True)
