#!/usr/bin/env python3
"""Serial, source-pinned runtime helpers for the R26 field qualification."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(os.environ.get('BATTERY_ROOT', '/home/josh/omp-workspace/drock-lmcache/r26-battery'))
NAME = os.environ.get('BATTERY_CONTAINER', 'r26-test')
PORT = int(os.environ.get('BATTERY_PORT', '5002'))
BASE_URL = f'http://127.0.0.1:{PORT}'
MODEL_NAME = 'GLM-5.3-Flash-NVFP4'
IMAGE = os.environ.get('BATTERY_IMAGE', 'voipmonitor/vllm@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68')
R25_IMAGE = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
OVERLAY_IMAGE = 'ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:27fe7a2f1df6d01e824cd24d6b83119edea471f4ec4997aa122d82c670133236'
MODEL = Path(os.environ.get('BATTERY_MODEL_DIR', '/mnt/2king/models/GLM-5.3-Flash-NVFP4'))
# Namespace tag for external cache objects; a different checkpoint must never
# reuse KV objects written by another one.
MODEL_CACHE_TAG = os.environ.get('BATTERY_MODEL_CACHE_TAG', '')
DRAFT = Path('/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8')
L2_HOST_ROOT = Path('/mnt/2king/lmcache-r26-battery')
L2_SHARED = Path(os.environ.get('BATTERY_L2_SHARED', str(L2_HOST_ROOT / 'shared')))
BENCH = Path('/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py')
PROXY_ENV = {**os.environ, 'https_proxy': 'http://127.0.0.1:9', 'http_proxy': 'http://127.0.0.1:9', 'no_proxy': 'localhost,127.0.0.1', 'NO_PROXY': 'localhost,127.0.0.1'}
_CURRENT_LABEL: str | None = None
_HTTP = requests.Session()
_HTTP.trust_env = False


def note(text: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {text}'
    print(line, flush=True)
    with (ROOT / 'battery.log').open('a') as handle:
        handle.write(line + '\n')


def save_json(name: str, data: object) -> Path:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)
    return path


def record_gate(name: str, passed: bool, detail: object) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    entry = {'name': name, 'passed': bool(passed), 'detail': detail, 'timestamp': time.time()}
    with (ROOT / 'gates.jsonl').open('a') as handle:
        handle.write(json.dumps(entry, default=str) + '\n')
    note(f'GATE {"PASS" if passed else "FAIL"} {name}')


def run(args: list[str], *, label: str, timeout: int = 1200, env: dict | None = None) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f'{label}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    note('$ ' + ' '.join(str(arg) for arg in args))
    with path.open('w') as output:
        try:
            child = subprocess.Popen([str(arg) for arg in args], stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                try:
                    parent = psutil.Process(child.pid)
                    processes = parent.children(recursive=True) + [parent]
                except psutil.NoSuchProcess:
                    processes = []
                for process in reversed(processes):
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=10)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.wait(timeout=15)
                if isinstance(error, KeyboardInterrupt):
                    raise
                output.write(f'\nBATTERY TIMEOUT after {timeout}s; command descendants stopped\n')
                code = 124
        except OSError as error:
            output.write(f'\nLAUNCH ERROR: {error}\n')
            code = 127
    finished = time.time()
    save_json(f'{label}.command.json', {'args': args, 'returncode': code, 'started_at': started, 'finished_at': finished, 'elapsed_seconds': finished - started, 'log': str(path)})
    return code


def _inspect() -> dict | None:
    result = subprocess.run(['docker', 'inspect', NAME], capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def metric_snapshot() -> str:
    try:
        response = _HTTP.get(BASE_URL + '/metrics', timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as error:
        return '# METRICS_UNAVAILABLE ' + str(error) + '\n'


def capture(label: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    inspected = _inspect()
    if inspected is not None:
        save_json(f'{label}.inspect.json', inspected)
        with (ROOT / f'{label}.docker.log').open('w') as output:
            subprocess.run(['docker', 'logs', NAME], stdout=output, stderr=subprocess.STDOUT, timeout=30, check=False)
    (ROOT / f'{label}.metrics.txt').write_text(metric_snapshot())
    with (ROOT / f'{label}.gpu.txt').open('w') as output:
        subprocess.run(['nvidia-smi'], stdout=output, stderr=subprocess.STDOUT, timeout=20, check=False)


def stop() -> None:
    global _CURRENT_LABEL
    inspected = _inspect()
    if inspected is None:
        _CURRENT_LABEL = None
        return
    if inspected['Config'].get('Labels', {}).get('field-lab.battery') != 'r26':
        raise RuntimeError(f'Refusing to remove unowned container {NAME}')
    capture_error: BaseException | None = None
    try:
        capture((_CURRENT_LABEL or 'test') + '-final')
    except BaseException as error:
        capture_error = error
        detail = {'label': _CURRENT_LABEL or 'test', 'error': repr(error), 'timestamp': time.time()}
        try:
            save_json(f'{_CURRENT_LABEL or "test"}-capture-error.json', detail)
        except OSError:
            pass
        try:
            print(f'GPU cleanup diagnostic failed; removing the verified container anyway: {error!r}', file=sys.stderr, flush=True)
        except OSError:
            pass
    container_id = inspected['Id']
    removed = subprocess.run(['docker', 'rm', '-f', container_id], capture_output=True, timeout=60)
    if removed.returncode:
        deadline = time.monotonic() + 30
        while True:
            remaining = subprocess.run(
                ['docker', 'ps', '-a', '--no-trunc', '--format', '{{.ID}}', '--filter', f'id={container_id}'],
                capture_output=True, text=True, timeout=20, check=True,
            )
            if container_id not in remaining.stdout.splitlines():
                break
            if time.monotonic() >= deadline:
                raise subprocess.CalledProcessError(removed.returncode, removed.args, removed.stdout, removed.stderr)
            # Another owner of this same verified container may already be
            # removing it during signal cleanup. Wait for that removal; never
            # issue another destructive command or treat a still-live ID as gone.
            time.sleep(0.25)
    _CURRENT_LABEL = None
    time.sleep(3)
    if capture_error is not None and not isinstance(capture_error, Exception):
        raise capture_error


def cleanup_l2_child(path: Path) -> None:
    path = Path(path)
    resolved = path.resolve()
    if path.is_symlink() or resolved.parent != L2_HOST_ROOT.resolve() or not resolved.name.startswith('cache-phase-'):
        raise ValueError(f'Refusing to clean a non-owned cache-phase directory: {path}')
    if not resolved.exists():
        return
    cleanup_code = (
        "from pathlib import Path; import shutil\n"
        "for item in Path('/cleanup').iterdir():\n"
        "    if item.is_symlink() or not item.is_dir(): item.unlink()\n"
        "    else: shutil.rmtree(item)\n"
    )
    code = run(
        ['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
         '-v', f'{resolved}:/cleanup', IMAGE, '-c', cleanup_code],
        label='cleanup-' + resolved.name, timeout=180,
    )
    if code:
        raise RuntimeError(f'Owned cache cleanup failed with exit {code}: {resolved}')
    resolved.rmdir()


def wait_health(timeout: int = 900) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        inspected = _inspect()
        if inspected is None or not inspected['State']['Running']:
            return False
        try:
            response = _HTTP.get(BASE_URL + '/health', timeout=4)
            if response.status_code == 200:
                models = _HTTP.get(BASE_URL + '/v1/models', timeout=4)
                if models.ok and any(row.get('id') == MODEL_NAME for row in models.json().get('data', [])):
                    return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def boot(label: str, *, image: str = IMAGE, tp: int = 4, dcp: int = 4, spec: str = 'mtp0', cache: str = 'vram', kv: str = 'fp8_ds_mla', extra_env: dict[str, str] | None = None, extra_args: list[str] | None = None, model: Path | None = None) -> bool:
    global _CURRENT_LABEL
    stop()
    ROOT.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', PORT)) == 0:
            raise RuntimeError(f'Port {PORT} is occupied by a process outside this test container')
    if tp not in (2, 4) or dcp < 1 or tp % dcp:
        raise ValueError(f'Unsupported physical test topology TP{tp}/DCP{dcp}')
    settings = {
        'MODEL': '/model', 'SERVED_MODEL_NAME': MODEL_NAME, 'HOST': '127.0.0.1', 'PORT': str(PORT),
        'TP': str(tp), 'DCP': str(dcp), 'CACHE_MODE': cache, 'KV_CACHE_QUANT': kv,
        'CUDAGRAPH_MODE': 'FULL_AND_PIECEWISE', 'MAX_MODEL_LEN': '1048576', 'MAX_NUM_SEQS': '32',
        'MAX_NUM_BATCHED_TOKENS': '4096', 'PREFILL_SCHEDULE_INTERVAL': '1',
        'FAIRNESS_ENGINE': 'compute_share', 'PREFILL_COMPUTE_SHARE': '0.4',
        'GPU_MEMORY_UTILIZATION': '0.93', 'DCP_CKV_GATHER': 'auto',
        'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16', 'NCCL_BUFFSIZE': '2097152',
        'OMP_NUM_THREADS': '1', 'VLLM_SERVER_DEV_MODE': '1',
    }
    if spec.startswith('mtp') and spec[3:] in ('0', '3', '5'):
        settings.update({'SPECULATOR': 'mtp', 'MTP_DEPTH': spec[3:]})
    elif spec == 'dflash2':
        settings.update({'SPECULATOR': 'dflash2', 'DFLASH_DEPTH': '7', 'DFLASH_MODEL': '/draft-mxfp8', 'DFLASH_MODEL_REVISION': ''})
    else:
        raise ValueError(f'Unsupported speculator {spec}')
    if cache == 'lmcache':
        settings.update({
            'LMCACHE_CHUNK_SIZE': '4096', 'LMCACHE_TARGET_TOKEN_BUDGET': '4096',
            'LMCACHE_L1_SIZE_GB': '64', 'LMCACHE_L2_ENABLED': '1', 'LMCACHE_L2_ROOT': '/lmcache-l2',
            'LMCACHE_L2_MAX_CAPACITY_GB': '160', 'LMCACHE_MP_HOST': '127.0.0.1',
            'LMCACHE_MP_PORT': '15555', 'LMCACHE_HTTP_PORT': '18085', 'LMCACHE_PROMETHEUS_PORT': '19095',
            'LMCACHE_INSTANCE_ID': 'r26-field-qualification', 'LMCACHE_SHM_NAME': 'r26-field-qualification',
        })
    elif cache == 'native':
        settings['NATIVE_KV_OFFLOADING_SIZE_GB'] = '64'
    elif cache != 'vram':
        raise ValueError(f'Unsupported cache mode {cache}')
    overrides = {key: str(value) for key, value in (extra_env or {}).items()}
    l2_host = Path(overrides.pop('LMCACHE_L2_HOST_DIR', str(L2_SHARED)))
    model_dir = Path(model) if model else MODEL
    if cache == 'lmcache' and MODEL_CACHE_TAG:
        settings['LMCACHE_MODEL_REVISION_ID'] = MODEL_CACHE_TAG
    settings.update(overrides)
    if settings['HOST'] != '127.0.0.1':
        raise ValueError('Development-mode test server must bind loopback')
    if cache == 'lmcache':
        if not l2_host.resolve().is_relative_to(L2_HOST_ROOT.resolve()):
            raise ValueError(f'Test L2 must remain under {L2_HOST_ROOT}')
        l2_host.mkdir(parents=True, exist_ok=True)
    devices = ','.join(str(index) for index in range(tp))
    args = ['docker', 'run', '-d', '--name', NAME, '--label', 'field-lab.battery=r26', '--init',
            '--gpus', f'"device={devices}"', '--network', 'host', '--shm-size', '128g',
            '-v', f'{model_dir}:/model:ro', '-v', f'{DRAFT}:/draft-mxfp8:ro',
            '-v', 'r26-runtime-cache:/cache', '-v', 'r26-huggingface-cache:/root/.cache/huggingface']
    if cache == 'lmcache':
        args += ['-v', f'{l2_host}:/lmcache-l2']
    for key, value in settings.items():
        args += ['-e', f'{key}={value}']
    args += [image, *(extra_args or [])]
    save_json(f'{label}.launch.json', {'label': label, 'image': image, 'tp': tp, 'dcp': dcp, 'spec': spec, 'cache': cache, 'kv': kv, 'env': settings, 'extra_args': extra_args or [], 'l2_host': str(l2_host) if cache == 'lmcache' else None, 'gpus': devices, 'model_dir': str(model_dir), 'draft_dir': str(DRAFT)})
    _CURRENT_LABEL = label
    code = run(args, label=label + '.boot', timeout=180)
    healthy = code == 0 and wait_health()
    capture(label)
    record_gate('boot:' + label, healthy, {'returncode': code, 'image': image, 'launch': str(ROOT / f'{label}.launch.json')})
    if healthy:
        note('BOOT READY ' + label)
    return healthy


def bench(label: str, *, conc: str = '1,4,8,16', contexts: str = '0,32k', duration: int = 30, prefill: bool = True, extra: list[str] | None = None) -> bool:
    output = ROOT / f'{label}.json'
    args = ['python3', str(BENCH), '--port', str(PORT), '--model', MODEL_NAME, '--concurrency', conc,
            '--contexts', contexts, '--duration', str(duration), '--max-tokens', '8192', '--output', str(output)]
    if not prefill:
        args.append('--skip-prefill')
    args.extend(extra or [])
    before = metric_snapshot()
    (ROOT / f'{label}.before.metrics.txt').write_text(before)
    code = run(args, label=label + '.bench', timeout=max(1800, duration * 24 + 600), env=PROXY_ENV)
    capture(label)
    readable = False
    if output.exists():
        try:
            readable = isinstance(json.loads(output.read_text()), dict)
        except (OSError, ValueError):
            pass
    passed = code == 0 and readable
    record_gate('benchmark-execution:' + label, passed, {'returncode': code, 'result': str(output), 'concurrency': conc, 'contexts': contexts, 'duration_seconds': duration})
    return passed


def profile(label: str, profile_name: str, runs: int) -> bool:
    output = ROOT / f'{label}.json'
    code = run(['python3', str(BENCH), '--port', str(PORT), '--model', MODEL_NAME, '--test-profile', profile_name,
                '--profile-runs', str(runs), '--output', str(output)], label=label, timeout=3600, env=PROXY_ENV)
    capture(label)
    passed = code == 0 and output.exists()
    record_gate('profile-execution:' + label, passed, {'returncode': code, 'result': str(output)})
    return passed
