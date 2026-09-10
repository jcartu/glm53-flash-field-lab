#!/usr/bin/env python3
"""Run frozen behavior/history cases on pinned checkpoints and restore production."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r26'))
import runtime as rt
import run_qualification as coordinator

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909')
BASE = 'voipmonitor/vllm@sha256:52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f'
R29 = 'localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49'
PUBLISHED = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-published-46aaae8a')
ARMS = [
    ('r281-published', BASE, PUBLISHED, False),
    ('r29-published', R29, PUBLISHED, True),
    ('r29-qad2500', R29, Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD2500-3959f8a0'), True),
    ('r29-tvn1500', R29, Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD-TVN-step1500'), True),
]


def require_inputs():
    for name in ('published', 'qad_step2500'):
        if not json.loads((ROOT / (name + '-verification.json')).read_text()).get('passed'):
            raise RuntimeError('Unverified model control: ' + name)
    tvn = Path('/home/josh/omp-workspace/drock-lmcache/new-drop-review-20260909T083122Z/tvn1500-readiness-result.json')
    if not json.loads(tvn.read_text()).get('serving_smoke_passed'):
        raise RuntimeError('TVN serving readiness gate is missing')
    if not json.loads((ROOT / 'probe-selfcheck.json').read_text()).get('passed'):
        raise RuntimeError('Behavior/fixture probe selfcheck is missing')
    for path in (ROOT / 'behavior-fixtures.json', ROOT / 'history-fixtures/manifest.json'):
        if not path.is_file():
            raise RuntimeError('Missing frozen fixture: ' + str(path))


def phase():
    require_inputs()
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        HERE / 'behavior_probe.py', HERE / 'history_probe.py', ROOT / 'behavior-fixtures.json', ROOT / 'history-fixtures/manifest.json']}
    rt.save_json('comparison-contract.json', {'arms': [(name, image, str(model), behavior) for name, image, model, behavior in ARMS],
        'source_hashes': sources, 'topology': 'TP4/DCP1, no speculation, GPU-local FP8 KV',
        'sampling': 'temperature1, top_p0.95, max reasoning; behavior clear_thinking=false; history explicitly crosses false/true',
        'scope': 'Checkpoint packages as shipped; TVN1500 and QAD2500 input-scale sidecars differ. No pure training-objective attribution. No performance claim from behavior-suite wall time.'})
    progress = []
    for label, image, model, behavior in ARMS:
        record = {'arm': label, 'image': image, 'model': str(model), 'executed': False}
        try:
            for path, expected in sources.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Frozen execution input changed: ' + path)
            if not rt.boot(label, image=image, model=model, tp=4, dcp=1, spec='mtp0', cache='vram',
                extra_env={'MAX_NUM_SEQS': '32', 'MAX_CUDAGRAPH_CAPTURE_SIZE': '32', 'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32'},
                extra_args=['--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
                            '--override-generation-config', '{"temperature":1.0,"top_p":0.95}']):
                raise RuntimeError('Model boot failed')
            if behavior:
                output = rt.ROOT / label / 'behavior'
                code = rt.run([sys.executable, str(HERE / 'behavior_probe.py'), '--base-url', rt.BASE_URL,
                    '--model', rt.MODEL_NAME, '--fixture', str(ROOT / 'behavior-fixtures.json'),
                    '--output-dir', str(output), '--concurrency', '4', '--clear-thinking', 'false'],
                    label=label + '-behavior', timeout=7200)
                record['behavior_returncode'] = code
                record['behavior_summary'] = str(output / 'summary.json')
            if not rt.wait_health(timeout=30):
                raise RuntimeError('Server became unhealthy before history controls')
            output = rt.ROOT / label / 'history'
            code = rt.run([sys.executable, str(HERE / 'history_probe.py'), '--base-url', rt.BASE_URL,
                '--model', rt.MODEL_NAME, '--manifest', str(ROOT / 'history-fixtures/manifest.json'),
                '--output-dir', str(output)], label=label + '-history', timeout=7200)
            record['history_returncode'] = code
            record['history_summary'] = str(output / 'summary.json')
            record['executed'] = True
            record['transport_passed'] = code == 0 and record.get('behavior_returncode', 0) == 0
            rt.record_gate('checkpoint-comparison-execution:' + label, record['transport_passed'], record)
        except Exception as error:
            record['error'] = repr(error)
            rt.record_gate('checkpoint-comparison-execution:' + label, False, record)
        finally:
            progress.append(record)
            rt.save_json('checkpoint-comparison-progress.json', progress)
            rt.stop()
    summary = {'all_arms_attempted': len(progress) == len(ARMS), 'arms': progress,
        'execution_passed': all(row.get('transport_passed') for row in progress)}
    rt.save_json('checkpoint-comparison-summary.json', summary)
    if not summary['execution_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    require_inputs()
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        print('R29 CHECKPOINT COMPARISON: guarded coordinator starting', flush=True)
        coordinator.PHASES = [('checkpoint-comparison', str(Path(__file__).resolve()), ['--phase'], 57600)]
        coordinator.main()
    else:
        raise SystemExit('Usage: checkpoint_comparison.py [--phase]')
