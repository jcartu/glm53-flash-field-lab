#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
NAME = 'r25-replay'
PORT = 5001
LAUNCHER = Path('/home/josh/omp-workspace/glm53-flash-field-lab/scripts/serve-r25-production.sh')
REPLAY = ROOT / 'replay_exact_1m.py'
PREFIX = ROOT / 'prefix_unique_80k.py'
NEEDLE = ROOT / 'needle_unique_1m.py'
EVICTION = Path('/home/josh/omp-workspace/drock-lmcache/eviction_test.py')
CORRUPTION = Path('/home/josh/omp-workspace/corruption-hunt.py')
BENCH = Path('/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py')
PROXY_ENV = {
    **os.environ,
    'https_proxy': 'http://127.0.0.1:9',
    'http_proxy': 'http://127.0.0.1:9',
    'no_proxy': 'localhost,127.0.0.1',
    'NO_PROXY': 'localhost,127.0.0.1',
}


def capture(name: str) -> None:
    with (ROOT / f'{name}.docker.log').open('w') as output:
        subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)
    with (ROOT / f'{name}.inspect.json').open('w') as output:
        subprocess.run(['docker', 'inspect', NAME], stdout=output, stderr=subprocess.STDOUT, check=False)


def capture_sidecar() -> None:
    top = subprocess.run(
        ['docker', 'top', NAME, '-eo', 'pid,ppid,comm,args'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (ROOT / 'sidecar-processes.txt').write_text(top)
    gpu = subprocess.run(
        [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (ROOT / 'gpu-processes.txt').write_text(gpu)
    gpu_pids = {
        int(line.split(',', 1)[0].strip())
        for line in gpu.splitlines()
        if line.strip()
    }
    process_rows = []
    for line in top.splitlines()[1:]:
        fields = line.split(None, 3)
        if not fields or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        cmdline = fields[3] if len(fields) > 3 else ''
        if 'lmcache' not in cmdline.lower():
            continue
        try:
            environ = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        except OSError:
            environ = []
        visible = next(
            (
                item.split(b'=', 1)[1].decode(errors='replace')
                for item in environ
                if item.startswith(b'CUDA_VISIBLE_DEVICES=')
            ),
            None,
        )
        process_rows.append({
            'pid': pid,
            'command': cmdline,
            'cuda_visible_devices': visible,
            'present_in_nvidia_smi': pid in gpu_pids,
        })
    report = {
        'lmcache_processes': process_rows,
        'lmcache_gpu_process_count': sum(row['present_in_nvidia_smi'] for row in process_rows),
        'verdict': 'pass' if process_rows and not any(row['present_in_nvidia_smi'] for row in process_rows) else 'fail',
    }
    (ROOT / 'sidecar-check.json').write_text(json.dumps(report, indent=2))
    if report['verdict'] != 'pass':
        raise RuntimeError(f'LMCache sidecar check failed: {report}')


env = {
    **os.environ,
    'NAME': NAME,
    'PORT': str(PORT),
}
subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
try:
    print('R25 EXTENDED START', flush=True)
    with (ROOT / 'extended-launch.log').open('w') as output:
        subprocess.run(
            [str(LAUNCHER)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
            timeout=1200,
        )
    capture('extended-ready')
    capture_sidecar()
    prefix_env = {
        **os.environ,
        'TEST_CONTAINER': NAME,
        'PORT': str(PORT),
        'OUT': str(ROOT / 'prefix-unique-80k.json'),
    }
    with (ROOT / 'prefix-unique-80k.log').open('w') as output:
        subprocess.run(
            ['python3', str(PREFIX)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=prefix_env,
            check=True,
            timeout=1800,
        )
    capture('prefix-unique-80k')
    with (ROOT / 'needle-unique-1m.log').open('w') as output:
        subprocess.run(
            ['python3', str(NEEDLE)],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
        )
    replay_env = {
        **os.environ,
        'TEST_CONTAINER': NAME,
        'PORT': str(PORT),
        'OUT': str(ROOT / 'replay-exact-1m.json'),
        'CACHE_SALT': 'r25-exact-million-v1',
    }
    with (ROOT / 'replay-exact-1m.log').open('w') as output:
        subprocess.run(
            ['python3', str(REPLAY)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=replay_env,
            check=True,
            timeout=3600,
        )
    capture('replay-exact-1m')
    eviction_env = {**os.environ, 'TEST_MODEL': 'GLM-5.3-Flash-NVFP4'}
    with (ROOT / 'eviction-r25.log').open('w') as output:
        subprocess.run(
            [
                'python3', str(EVICTION), str(PORT), str(ROOT / 'eviction-r25.json'),
                '--docs', '40', '--words', '12000',
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=eviction_env,
            check=True,
            timeout=3600,
        )
    capture('eviction-r25')
    corruption_env = {**os.environ, 'REASONING_EFFORT': 'low'}
    wave_paths = [ROOT / 'corruption-lmcache-dflash.log']
    for wave in (2, 3):
        path = ROOT / f'corruption-lmcache-dflash-wave{wave}.log'
        with path.open('w') as output:
            subprocess.run(
                ['python3', str(CORRUPTION), str(PORT)],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=corruption_env,
                check=True,
                timeout=1800,
            )
        wave_paths.append(path)
    waves = [json.loads(path.read_text()) for path in wave_paths]
    all_results = [
        {'wave': wave_index + 1, **row}
        for wave_index, wave in enumerate(waves)
        for row in wave['results']
    ]
    clean = sum(row['verdict'].startswith('CLEAN') for row in all_results)
    (ROOT / 'corruption-lmcache-dflash-aggregate.json').write_text(json.dumps({
        'waves': len(waves),
        'clean': clean,
        'total': len(all_results),
        'flagged': len(all_results) - clean,
        'flagged_results': [
            row for row in all_results if not row['verdict'].startswith('CLEAN')
        ],
        'results': all_results,
    }, indent=2))
    with (ROOT / 'lavd-test-repeat.log').open('w') as output:
        subprocess.run(
            [
                'python3', str(BENCH), '--port', str(PORT),
                '--model', 'GLM-5.3-Flash-NVFP4',
                '--test-profile', 'lavd-test', '--profile-runs', '8',
                '--output', str(ROOT / 'lavd-test-repeat.json'),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=PROXY_ENV,
            check=True,
            timeout=3600,
        )
    lavd_waves = [
        json.loads((ROOT / 'lavd-test.json').read_text())['selected_summary'],
        json.loads((ROOT / 'lavd-test-repeat.json').read_text())['selected_summary'],
    ]
    (ROOT / 'lavd-aggregate.json').write_text(json.dumps({
        'waves': 2,
        'attempted': sum(wave['attempted'] for wave in lavd_waves),
        'completed': sum(wave['completed'] for wave in lavd_waves),
        'correct': sum(wave['correct'] for wave in lavd_waves),
        'wrong': sum(wave['wrong'] for wave in lavd_waves),
        'errors': sum(wave['errors'] for wave in lavd_waves),
        'exact': sum(wave['exact'] for wave in lavd_waves),
        'fail': sum(wave['fail'] for wave in lavd_waves),
        'aggregate_gen_tps': statistics.fmean(
            wave['aggregate_gen_tok_s'] for wave in lavd_waves
        ),
        'aggregate_e2e_tps': statistics.fmean(
            wave['aggregate_e2e_tok_s'] for wave in lavd_waves
        ),
    }, indent=2))
    capture_sidecar()
finally:
    capture('extended-final')
    subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
(ROOT / 'EXTENDED_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print('R25 EXTENDED DONE', flush=True)
