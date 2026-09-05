#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
R25 = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
PATCHED = 'local/glm53-r25-pr646@sha256:70a158cd6825074c8a335b8783c06326132f1cb5a2a167efa47ea136da0e43b4'
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
DRAFT = '/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8'
BENCH = '/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py'
PREFIX = ROOT / 'pr646_prefix_probe.py'
NEEDLE = ROOT / 'pr646_needle_90k.py'
NAME = 'pr646-test'
PORT = 5001
ARMS = [
    {
        'label': 'stock-fine-256',
        'image': R25,
        'target_block': 256,
        'mamba_block': 256,
    },
    {
        'label': 'stock-coarse-2048',
        'image': R25,
        'target_block': 2048,
        'mamba_block': 2048,
    },
    {
        'label': 'pr646-decoupled-2048-256',
        'image': PATCHED,
        'target_block': 2048,
        'mamba_block': 256,
    },
]
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


def boot(arm: dict[str, object]) -> None:
    remove()
    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host', '--ipc', 'host', '--shm-size', '32g',
        '-v', f'{MODEL}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
        '-v', 'r25-runtime-cache:/cache',
        '-e', 'MODEL=/model', '-e', 'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        '-e', f'PORT={PORT}', '-e', 'TP=4', '-e', 'DCP=1',
        '-e', 'CACHE_MODE=vram', '-e', 'KV_CACHE_QUANT=fp8_ds_mla',
        '-e', f"GLM53_TARGET_BLOCK_SIZE={arm['target_block']}",
        '-e', f"GLM53_MAMBA_BLOCK_SIZE={arm['mamba_block']}",
        '-e', 'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        '-e', 'MAX_MODEL_LEN=262144', '-e', 'MAX_NUM_SEQS=32',
        '-e', 'MAX_NUM_BATCHED_TOKENS=4096', '-e', 'PREFILL_SCHEDULE_INTERVAL=1',
        '-e', 'FAIRNESS_ENGINE=compute_share', '-e', 'PREFILL_COMPUTE_SHARE=0.4',
        '-e', 'GPU_MEMORY_UTILIZATION=0.93',
        '-e', 'SPECULATOR=dflash2', '-e', 'DFLASH_DEPTH=7',
        '-e', 'DFLASH_MODEL=/draft-mxfp8', '-e', 'DFLASH_MODEL_REVISION=',
        str(arm['image']), '--prefix-match-unit', '256',
    ]
    label = str(arm['label'])
    with (ROOT / f'pr646-{label}.boot.log').open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    wait_health()


def benchmark(label: str) -> Path:
    path = ROOT / f'pr646-{label}.json'
    with (ROOT / f'pr646-{label}.bench.log').open('w') as output:
        subprocess.run(
            [
                'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
                '--concurrency', '1,4,8,16', '--contexts', '0,16k,32k',
                '--duration', '30', '--max-tokens', '8192', '--output', str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=PROXY_ENV,
            check=True,
            timeout=1800,
        )
    return path


def summarize_bench(path: Path, server_log_path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    benchmark_capacity = None
    for event in data.get('event_log', []):
        match = re.search(r'KV cache budget from vLLM metrics: ([\d,]+) tokens', str(event))
        if match:
            benchmark_capacity = int(match.group(1).replace(',', ''))
            break
    server_log = server_log_path.read_text()
    capacity_match = re.search(r'GPU KV cache size: ([\d,]+) tokens', server_log)
    if not capacity_match:
        raise RuntimeError(f'authoritative KV capacity missing from {server_log_path}')
    capacity = int(capacity_match.group(1).replace(',', ''))
    cells = {
        f"C{row['concurrency']}@{row['context_tokens']}": row['aggregate_tps']
        for row in data['results']
        if row.get('concurrency') in {1, 4, 8, 16}
        and row.get('context_tokens') in {0, 16384, 32768}
    }
    return {
        'kv_capacity_tokens': capacity,
        'benchmark_naive_capacity_tokens': benchmark_capacity,
        'prefill32_tokens_per_second': data['prefill']['32768'].get('client_tok_per_sec'),
        'cells': cells,
        'hardware': data.get('hardware_run_summary', {}),
    }


results = {}
try:
    print('PR646 GPU TEST START', flush=True)
    for arm in ARMS:
        label = str(arm['label'])
        print(f'PR646 ARM START {label}', flush=True)
        boot(arm)
        bench_path = benchmark(label)
        prefix_path = ROOT / f'pr646-prefix-{label}.json'
        with (ROOT / f'pr646-prefix-{label}.log').open('w') as output:
            subprocess.run(
                ['python3', str(PREFIX), label, str(prefix_path), str(PORT)],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        server_log_path = ROOT / f'pr646-{label}.docker.log'
        with server_log_path.open('w') as output:
            subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
        row = summarize_bench(bench_path, server_log_path)
        row['prefix_probe'] = json.loads(prefix_path.read_text())
        results[label] = row
        if label == 'pr646-decoupled-2048-256':
            with (ROOT / 'pr646-needle-90k.log').open('w') as output:
                subprocess.run(
                    ['python3', str(NEEDLE), str(ROOT / 'pr646-needle-90k.json'), str(PORT)],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            with (ROOT / 'pr646-estonia.log').open('w') as output:
                subprocess.run(
                    [
                        'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
                        '--test-profile', 'estonia', '--profile-runs', '6',
                        '--output', str(ROOT / 'pr646-estonia.json'),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=PROXY_ENV,
                    check=True,
                    timeout=3600,
                )
        print(f'PR646 ARM DONE {label}: {row}', flush=True)
finally:
    remove()

fine = results['stock-fine-256']
coarse = results['stock-coarse-2048']
patched = results['pr646-decoupled-2048-256']
common_cells = sorted(set(coarse['cells']) & set(patched['cells']))
throughput_deltas = {
    cell: 100 * (patched['cells'][cell] / coarse['cells'][cell] - 1)
    for cell in common_cells
}
needle = json.loads((ROOT / 'pr646-needle-90k.json').read_text())
estonia = json.loads((ROOT / 'pr646-estonia.json').read_text())['selected_summary']
image_receipt = json.loads((ROOT / 'pr646-image.json').read_text())
summary = {
    'provenance': {
        'pr': 'https://github.com/local-inference-lab/vllm/pull/646',
        'pr_state': 'draft',
        'pr_merge_state_at_test_start': 'DIRTY',
        'base_image': R25,
        'patched_image': PATCHED,
        'patched_image_id': image_receipt['id'],
        'patched_image_layers': image_receipt['layer_count'],
        'rebased_unit_tests': '46 passed',
        'scope': 'TP4 DCP1 DFlash2 K7 FP8 KV only; DCP greater than one not tested',
    },
    'arms': results,
    'comparison': {
        'capacity_gain_vs_fine_256': patched['kv_capacity_tokens'] / fine['kv_capacity_tokens'],
        'capacity_change_vs_stock_coarse': 100 * (patched['kv_capacity_tokens'] / coarse['kv_capacity_tokens'] - 1),
        'fine_prefix_hit_change_vs_stock_coarse': (
            patched['prefix_probe']['repeat_metric_delta']['vllm:prefix_cache_hits_total']
            - coarse['prefix_probe']['repeat_metric_delta']['vllm:prefix_cache_hits_total']
        ),
        'throughput_delta_percent_vs_stock_coarse': throughput_deltas,
        'mean_absolute_throughput_delta_percent': sum(abs(value) for value in throughput_deltas.values()) / len(throughput_deltas),
        'worst_throughput_delta_percent': min(throughput_deltas.values()),
    },
    'correctness': {
        'prefix_outputs_identical_all_arms': all(row['prefix_probe']['outputs_identical'] for row in results.values()),
        'needle_90k_hits': sum(row['hit'] for row in needle['results']),
        'needle_90k_total': len(needle['results']),
        'estonia_correct': estonia['correct'],
        'estonia_total': estonia['attempted'],
        'estonia_errors': estonia['errors'],
    },
}
(ROOT / 'pr646-summary.json').write_text(json.dumps(summary, indent=2))
(ROOT / 'PR646_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print(json.dumps(summary, indent=2), flush=True)
