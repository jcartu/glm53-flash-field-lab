#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r24-battery')
ROOT.mkdir(parents=True, exist_ok=True)
IMAGE = 'voipmonitor/vllm@sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd'
MODEL = '/mnt/2king/models/GLM-5.3-Flash-NVFP4'
DRAFT = '/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8'
BENCH = '/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py'
COLLISION = '/home/josh/omp-workspace/drock-lmcache/decode_prefill_collision.py'
PREFIX = '/home/josh/omp-workspace/drock-lmcache/prefix_test.py'
CORRUPTION = '/home/josh/omp-workspace/corruption-hunt.py'
TEMPLATE = '/home/josh/omp-workspace/template-probe.py'
NEEDLE = '/home/josh/omp-workspace/probe-1m-needle.py'
NAME = 'r24-test'
PORT = 5001
LOG = ROOT / 'battery.log'

PROXY_ENV = os.environ.copy()
PROXY_ENV.update({
    'https_proxy': 'http://127.0.0.1:9',
    'http_proxy': 'http://127.0.0.1:9',
    'no_proxy': 'localhost,127.0.0.1',
    'NO_PROXY': 'localhost,127.0.0.1',
})


def note(text: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {text}'
    print(line, flush=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def run(args: list[str], *, log: Path | None = None, timeout: int = 1200, env: dict[str, str] | None = None) -> int:
    note('$ ' + ' '.join(args))
    with (log.open('w') if log else subprocess.DEVNULL) as out:
        p = subprocess.run(
            args,
            stdout=out,
            stderr=subprocess.STDOUT if log else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            env=env,
            check=False,
        )
    return p.returncode


def docker_rm() -> None:
    subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(4)


def wait_health(timeout_s: int = 900) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = subprocess.run(
            ['curl', '-fsS', '--max-time', '4', f'http://127.0.0.1:{PORT}/health'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return True
        alive = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        if NAME not in alive:
            return False
        time.sleep(10)
    return False


def spec_env(mode: str) -> list[str]:
    if mode == 'mtp0':
        return ['SPECULATOR=mtp', 'MTP_DEPTH=0']
    if mode == 'mtp3':
        return ['SPECULATOR=mtp', 'MTP_DEPTH=3']
    if mode == 'dflash2':
        return ['SPECULATOR=dflash2', 'DFLASH_DEPTH=7', 'DFLASH_MODEL=/draft-mxfp8', 'DFLASH_MODEL_REVISION=']
    raise ValueError(mode)


def boot(label: str, *, dcp: int, spec: str, cache: str, kv: str, kda: str = 'flashkda', fairness: str = 'compute_share', share: str = '0.4', gather: str = 'auto') -> bool:
    docker_rm()
    shm = '128g' if cache == 'lmcache' else '32g'
    gmu = '0.95' if cache == 'lmcache' else '0.93'
    envs = [
        'MODEL=/model',
        'SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4',
        f'PORT={PORT}', 'TP=4', f'DCP={dcp}',
        f'CACHE_MODE={cache}', f'KV_CACHE_QUANT={kv}',
        'CUDAGRAPH_MODE=FULL_AND_PIECEWISE',
        'MAX_MODEL_LEN=1048576', 'MAX_NUM_SEQS=32',
        'MAX_NUM_BATCHED_TOKENS=4096', 'PREFILL_SCHEDULE_INTERVAL=1',
        f'FAIRNESS_ENGINE={fairness}', f'PREFILL_COMPUTE_SHARE={share}',
        f'GPU_MEMORY_UTILIZATION={gmu}',
        f'DCP_CKV_GATHER={gather}',
    ] + spec_env(spec)
    if kda == 'b12x':
        envs.append('GLM53_KDA_PREFILL_BACKEND=b12x')
    if cache == 'lmcache':
        envs.extend([
            'LMCACHE_CHUNK_SIZE=4096',
            'LMCACHE_TARGET_TOKEN_BUDGET=4096',
            'LMCACHE_L1_SIZE_GB=64',
            'LMCACHE_L2_ENABLED=1',
            'LMCACHE_L2_ROOT=/lmcache-l2',
        ])
    elif cache == 'native':
        envs.append('NATIVE_KV_OFFLOADING_SIZE_GB=64')

    args = [
        'docker', 'run', '-d', '--name', NAME, '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'host', '--ipc', 'host', '--shm-size', shm,
        '-v', f'{MODEL}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
        '-v', 'r24-runtime-cache:/cache', '-v', 'r24-huggingface-cache:/root/.cache/huggingface',
    ]
    if cache == 'lmcache':
        args += ['-v', '/mnt/2king/lmcache-l2:/lmcache-l2']
    for item in envs:
        args += ['-e', item]
    args.append(IMAGE)

    rc = run(args, log=ROOT / f'{label}.boot.log', timeout=180)
    if rc != 0 or not wait_health():
        note(f'BOOT FAIL {label}')
        subprocess.run(['docker', 'logs', NAME], stdout=(ROOT / f'{label}.docker.log').open('w'), stderr=subprocess.STDOUT, check=False)
        return False
    subprocess.run(['docker', 'logs', NAME], stdout=(ROOT / f'{label}.docker.log').open('w'), stderr=subprocess.STDOUT, check=False)
    subprocess.run(['docker', 'inspect', NAME], stdout=(ROOT / f'{label}.inspect.json').open('w'), stderr=subprocess.DEVNULL, check=False)
    note(f'BOOT OK {label}')
    return True


def bench(label: str, *, conc: str = '1,4,8,16', contexts: str = '0,32k', prefill: bool = True) -> bool:
    out_json = ROOT / f'{label}.json'
    args = [
        'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
        '--concurrency', conc, '--contexts', contexts, '--duration', '30',
        '--max-tokens', '8192', '--output', str(out_json),
    ]
    if not prefill:
        args.append('--skip-prefill')
    rc = run(args, log=ROOT / f'{label}.bench.log', timeout=1200, env=PROXY_ENV)
    subprocess.run(['curl', '-fsS', f'http://127.0.0.1:{PORT}/metrics'], stdout=(ROOT / f'{label}.metrics.txt').open('w'), stderr=subprocess.DEVNULL, check=False)
    subprocess.run(['docker', 'logs', NAME], stdout=(ROOT / f'{label}.docker.log').open('w'), stderr=subprocess.STDOUT, check=False)
    note(f'BENCH {"OK" if rc == 0 and out_json.exists() else "FAIL"} {label}')
    return rc == 0 and out_json.exists()


def collision(label: str) -> None:
    run(['python3', COLLISION, '--base-url', f'http://127.0.0.1:{PORT}', '--model', 'GLM-5.3-Flash-NVFP4', '--output', str(ROOT / f'{label}.json')], log=ROOT / f'{label}.log', timeout=700)


def quick_profile(profile: str, runs: int) -> None:
    args = [
        'python3', BENCH, '--port', str(PORT), '--model', 'GLM-5.3-Flash-NVFP4',
        '--test-profile', profile, '--profile-runs', str(runs), '--output', str(ROOT / f'{profile}.json'),
    ]
    run(args, log=ROOT / f'{profile}.log', timeout=3600, env=PROXY_ENV)


configs = [
    dict(label='dcp1-vram-fp8-mtp0', dcp=1, spec='mtp0', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp1-vram-fp8-mtp3', dcp=1, spec='mtp3', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp1-vram-fp8-dflash2', dcp=1, spec='dflash2', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp2-vram-fp8-mtp0', dcp=2, spec='mtp0', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp2-vram-fp8-dflash2', dcp=2, spec='dflash2', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp4-vram-fp8-mtp0', dcp=4, spec='mtp0', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp4-vram-fp8-mtp3', dcp=4, spec='mtp3', cache='vram', kv='fp8_ds_mla'),
    dict(label='dcp4-vram-fp8-dflash2', dcp=4, spec='dflash2', cache='vram', kv='fp8_ds_mla'),
    # dcp4-vram-nvfp4-mtp0 was recorded by the smoke before this script.
    dict(label='dcp4-vram-nvfp4-mtp3', dcp=4, spec='mtp3', cache='vram', kv='nvfp4_ds_mla'),
    dict(label='dcp4-vram-nvfp4-dflash2', dcp=4, spec='dflash2', cache='vram', kv='nvfp4_ds_mla'),
    dict(label='dcp4-lmcache-nvfp4-mtp0', dcp=4, spec='mtp0', cache='lmcache', kv='nvfp4_ds_mla'),
    dict(label='dcp4-lmcache-nvfp4-mtp3', dcp=4, spec='mtp3', cache='lmcache', kv='nvfp4_ds_mla'),
    dict(label='dcp4-lmcache-nvfp4-dflash2', dcp=4, spec='dflash2', cache='lmcache', kv='nvfp4_ds_mla'),
    dict(label='dcp4-vram-nvfp4-mtp0-b12xkda', dcp=4, spec='mtp0', cache='vram', kv='nvfp4_ds_mla', kda='b12x'),
    dict(label='dcp4-vram-nvfp4-mtp0-ranklocal', dcp=4, spec='mtp0', cache='vram', kv='nvfp4_ds_mla', gather='0'),
    dict(label='dcp4-native-nvfp4-mtp0', dcp=4, spec='mtp0', cache='native', kv='nvfp4_ds_mla'),
]

for cfg in configs:
    label = cfg.pop('label')
    if not boot(label, **cfg):
        continue
    bench(label)
    if label == 'dcp1-vram-fp8-mtp0':
        collision('collision-dcp1-fair04')
    if label == 'dcp4-vram-fp8-mtp3':
        run(['python3', CORRUPTION, str(PORT)], log=ROOT / 'corruption-dcp4-mtp3.log', timeout=1800, env={**os.environ, 'REASONING_EFFORT': 'low'})
    if label == 'dcp4-lmcache-nvfp4-mtp0':
        run(['python3', PREFIX, str(PORT), 'r24-lmcache', str(ROOT / 'prefix-lmcache-80k.json'), '--tokens', '80000'], log=ROOT / 'prefix-lmcache-80k.log', timeout=1200, env={**os.environ, 'TEST_CONTAINER': NAME, 'TEST_MODEL': 'GLM-5.3-Flash-NVFP4'})
    if label == 'dcp4-lmcache-nvfp4-dflash2':
        run(['python3', TEMPLATE, str(PORT)], log=ROOT / 'template-probe.log', timeout=900)
        run(['python3', CORRUPTION, str(PORT)], log=ROOT / 'corruption-lmcache-dflash.log', timeout=1800, env={**os.environ, 'REASONING_EFFORT': 'low'})
        quick_profile('estonia', 6)
        quick_profile('lavd-test', 8)
    if label == 'dcp4-native-nvfp4-mtp0':
        run(['python3', PREFIX, str(PORT), 'r24-native', str(ROOT / 'prefix-native-80k.json'), '--tokens', '80000', '--skip-restart'], log=ROOT / 'prefix-native-80k.log', timeout=900, env={**os.environ, 'TEST_CONTAINER': NAME, 'TEST_MODEL': 'GLM-5.3-Flash-NVFP4'})

# Stock scheduler control for collision, fairness off.
if boot('dcp1-vram-fp8-mtp0-fair-off', dcp=1, spec='mtp0', cache='vram', kv='fp8_ds_mla', fairness='none', share='0.4'):
    collision('collision-dcp1-fair-off')

# Finish on DCP4 LMCache no-spec for long-context proof.
if boot('final-dcp4-lmcache-nvfp4-mtp0', dcp=4, spec='mtp0', cache='lmcache', kv='nvfp4_ds_mla'):
    run(['python3', NEEDLE, '0.1', '0.5', '0.9'], log=ROOT / 'needle-1m.log', timeout=3600, env={**os.environ, 'NEEDLE_MODEL': 'GLM-5.3-Flash-NVFP4'})

subprocess.run(['docker', 'rm', '-f', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
(ROOT / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
note('BATTERY DONE')
