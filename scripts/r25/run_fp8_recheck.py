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
NAME = 'r25-fp8-recheck'
PORT = 5001
SPECS = {
    'mtp0': ['SPECULATOR=mtp', 'MTP_DEPTH=0'],
    'mtp3': ['SPECULATOR=mtp', 'MTP_DEPTH=3'],
    'dflash2': ['SPECULATOR=dflash2', 'DFLASH_DEPTH=7', 'DFLASH_MODEL=/draft-mxfp8', 'DFLASH_MODEL_REVISION='],
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


def boot(spec: str) -> None:
    remove()
    envs = [
        'MODEL=/model', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        f'PORT={PORT}', 'TP=4', 'DCP=4', 'CACHE_MODE=vram',
        'KV_CACHE_QUANT=fp8_ds_mla', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        'MAX_MODEL_LEN=1048576', 'MAX_NUM_SEQS=32',
        'MAX_NUM_BATCHED_TOKENS=4096', 'PREFILL_SCHEDULE_INTERVAL=1',
        'FAIRNESS_ENGINE=compute_share', 'PREFILL_COMPUTE_SHARE=0.4',
        'GPU_MEMORY_UTILIZATION=0.93', 'DCP_CKV_GATHER=auto',
        *SPECS[spec],
    ]
    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host', '--ipc', 'host', '--shm-size', '32g',
        '-v', f'{MODEL}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
        '-v', 'r25-runtime-cache:/cache',
    ]
    for item in envs:
        args += ['-e', item]
    args.append(IMAGE)
    with (ROOT / f'recheck-dcp4-vram-fp8-{spec}.boot.log').open('w') as output:
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
            raise RuntimeError(f'{spec} exited before health')
        time.sleep(10)
    raise RuntimeError(f'{spec} did not become healthy')


def headline(path: Path) -> dict[str, float | None]:
    data = json.loads(path.read_text())
    rows = {
        int(row['concurrency']): row
        for row in data['results']
        if row.get('context_tokens') == 0
    }
    return {
        'prefill32': data['prefill']['32768'].get('client_tok_per_sec'),
        **{f'C{concurrency}': rows[concurrency]['aggregate_tps'] for concurrency in (1, 4, 8, 16)},
        'cpu_util_avg_pct': data['hardware_run_summary']['cpu_util_avg_pct'],
        'power_total_avg_w': data['hardware_run_summary']['power_total_avg_w'],
        'pcie_rx_avg_mb_s': data['hardware_run_summary']['pcie_rx_avg_mb_s'],
        'pcie_tx_avg_mb_s': data['hardware_run_summary']['pcie_tx_avg_mb_s'],
    }


summary = {}
try:
    print('FP8 RECHECK START', flush=True)
    for spec in SPECS:
        print(f'FP8 RECHECK ARM START {spec}', flush=True)
        boot(spec)
        path = ROOT / f'recheck-dcp4-vram-fp8-{spec}.json'
        with (ROOT / f'recheck-dcp4-vram-fp8-{spec}.bench.log').open('w') as output:
            subprocess.run(
                [
                    'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
                    '--concurrency', '1,4,8,16', '--contexts', '0,32k', '--duration', '30',
                    '--max-tokens', '8192', '--output', str(path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=PROXY_ENV,
                check=True,
                timeout=1200,
            )
        with (ROOT / f'recheck-dcp4-vram-fp8-{spec}.docker.log').open('w') as output:
            subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
        summary[spec] = headline(path)
        print(f'FP8 RECHECK ARM DONE {spec}: {summary[spec]}', flush=True)
finally:
    remove()

initial = {}
for spec in SPECS:
    initial[spec] = headline(ROOT / f'dcp4-vram-fp8-{spec}.json')
comparison = {}
for spec in SPECS:
    comparison[spec] = {
        metric: 100 * (summary[spec][metric] / initial[spec][metric] - 1)
        for metric in ('prefill32', 'C1', 'C4', 'C8', 'C16')
    }
report = {
    'reason': 'first DCP4 FP8 pass showed low PCIe traffic and elevated host CPU load; packed NVFP4 immediately returned to the R24 range',
    'initial': initial,
    'recheck': summary,
    'recheck_vs_initial_percent': comparison,
}
(ROOT / 'fp8-recheck-summary.json').write_text(json.dumps(report, indent=2))
(ROOT / 'FP8_RECHECK_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print(json.dumps(report, indent=2), flush=True)
