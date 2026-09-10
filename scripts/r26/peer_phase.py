#!/usr/bin/env python3
"""Qualify #599 policy faults, real IPC peers, and direct/collective model paths."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import requests
import runtime as rt

PROBE = Path(__file__).with_name('peer_probe.py')
PROBE_NAME = 'r26-peer-probe'


def remove_probe() -> None:
    inspected = subprocess.run(['docker', 'inspect', PROBE_NAME], capture_output=True, text=True, timeout=20)
    if inspected.returncode:
        return
    container = json.loads(inspected.stdout)[0]
    if container['Config'].get('Labels', {}).get('field-lab.battery') != 'r26':
        raise RuntimeError('Refusing to remove an unrelated peer-probe container')
    subprocess.run(['docker', 'rm', '-f', container['Id']], capture_output=True, timeout=60, check=True)


def isolated_probe(mode: str, *, gpu: bool) -> dict:
    remove_probe()
    args = ['docker', 'run', '--name', PROBE_NAME, '--label', 'field-lab.battery=r26', '--init',
            '--network', 'none', '--shm-size', '1g',
            '-v', f'{PROBE.parent}:/probe:ro', '-v', f'{rt.ROOT}:/results',
            '-e', 'CUDA_VISIBLE_DEVICES=0,1,2,3' if gpu else 'CUDA_VISIBLE_DEVICES=',
            '--entrypoint', 'python']
    if gpu:
        args += ['--gpus', '"device=0,1,2,3"']
    args += [rt.OVERLAY_IMAGE, '/probe/peer_probe.py', '--mode', mode, '--output', f'/results/peer-{mode}.json']
    try:
        code = rt.run(args, label='peer-' + mode, timeout=600 if gpu else 120)
        path = rt.ROOT / f'peer-{mode}.json'
        result = json.loads(path.read_text()) if path.exists() else {'passed': False, 'error': 'Probe produced no receipt'}
        rt.record_gate(f'peer599:{mode}', code == 0 and result.get('passed') is True,
                       {'returncode': code, 'receipt': str(path), 'scope': result.get('scope'), 'gpu_devices_exposed': gpu})
        return result
    finally:
        remove_probe()


def model_path(mode: str) -> dict:
    settings = {'FAIRNESS_ENGINE': 'none'}
    if mode == 'collective':
        settings['VLLM_USE_DIRECT_DCP_A2A'] = '0'
    elif mode == 'forced-direct':
        settings['VLLM_USE_DIRECT_DCP_A2A'] = '1'
    label = 'peer599-model-' + mode
    result = {'mode': mode, 'image': rt.OVERLAY_IMAGE, 'requests': [], 'passed': False}
    try:
        if not rt.boot(label, image=rt.OVERLAY_IMAGE, dcp=4, spec='mtp0', kv='fp8_ds_mla',
                       extra_env=settings, extra_args=['--dcp-comm-backend', 'a2a']):
            result['error'] = 'Model boot failed'
            return result
        for index in range(4):
            expected = f'PEER-CHECK-{index}-7351'
            response = requests.post(rt.BASE_URL + '/v1/chat/completions', json={
                'model': rt.MODEL_NAME,
                'messages': [{'role': 'user', 'content': f'The verification code is {expected}. Return only that exact code.'}],
                'max_tokens': 256, 'temperature': 0, 'seed': 599,
                'chat_template_kwargs': {'reasoning_effort': 'low'},
            }, timeout=120)
            body = response.json()
            content = body.get('choices', [{}])[0].get('message', {}).get('content') or ''
            result['requests'].append({'status': response.status_code, 'expected': expected, 'content': content,
                                       'correct': expected in content, 'raw': body})
        rt.capture(label)
        log = (rt.ROOT / f'{label}.docker.log').read_text(errors='replace')
        result['direct_path_selected'] = 'Using direct symmetric-memory DCP A2A for MLA.' in log
        result['peer_fallback_logged'] = 'Direct CP peer access is unavailable; falling back to collectives.' in log
        result['passed'] = all(row['status'] == 200 and row['correct'] for row in result['requests'])
        if mode == 'collective':
            result['passed'] = result['passed'] and not result['direct_path_selected']
        if mode == 'forced-direct':
            result['passed'] = result['passed'] and result['direct_path_selected']
        return result
    except Exception as error:
        result['error'] = repr(error)
        return result
    finally:
        rt.save_json(label + '.json', result)
        rt.record_gate('peer599:' + mode, result['passed'], result)
        rt.stop()


def main() -> None:
    rt.note('DIRECT DCP PEER QUALIFICATION START')
    rt.stop()
    try:
        isolated_probe('policy', gpu=False)
        physical = isolated_probe('physical', gpu=True)
        model_path('collective')
        model_path('auto')
        if physical.get('passed') is True:
            model_path('forced-direct')
        else:
            rt.record_gate('peer599:forced-direct-safe-prerequisite', False,
                           {'not_run': True, 'reason': 'Never force peer-pointer access without a successful real CUDA IPC matrix.'})
        rt.save_json('peer-phase-completed.json', {'complete': True, 'scope': 'Physical checks apply only to this host; missing-peer/error cases are isolated policy fault injection. Explicit forced-on remains a deliberate operator override.'})
    finally:
        remove_probe()
        rt.stop()


if __name__ == '__main__':
    main()
