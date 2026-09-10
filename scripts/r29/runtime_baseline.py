#!/usr/bin/env python3
"""Counterbalanced R28.1/R29 GPU-local runtime baseline; restore production."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HELPERS = HERE.parent / 'r26'
sys.path.insert(0, str(HELPERS))
import runtime as rt
import run_qualification as coordinator
import steady_metrics

BASE = 'voipmonitor/vllm@sha256:52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f'
CANDIDATE = 'localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49'
MODEL = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-published-46aaae8a')
EXECUTION = Path('/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909')
EXPECTED_CELLS = {(c, context) for c in (1, 4, 8) for context in (0, 32768)}
ARMS = [('r281-ab', BASE), ('r29-ab', CANDIDATE), ('r29-ba', CANDIDATE), ('r281-ba', BASE)]


def require_inputs() -> None:
    proof = json.loads((EXECUTION / 'published-verification.json').read_text())
    if not proof.get('passed') or proof.get('revision') != '46aaae8a82032f77100f2f03e9cc11b391df3b4d':
        raise RuntimeError('Clean published checkpoint has not passed pinned-file verification')


def phase() -> None:
    require_inputs()
    rt.save_json('experiment-plan.json', {
        'scope': 'Matched runtime throughput baseline, not model quality; no speculation and GPU-local cache.',
        'arms': ARMS, 'model': str(MODEL), 'tp': 4, 'dcp': 1,
        'contexts': [0, 32768], 'concurrency': [1, 4, 8], 'duration_seconds': 30,
        'sampling': {'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'max', 'clear_thinking': False},
        'benchmark_sha256': hashlib.sha256(rt.BENCH.read_bytes()).hexdigest(),
        'request_identity_limit': 'The unchanged benchmark inserts a per-run nonce; workload template and all configuration are matched, not byte-identical generated nonce text.',
    })
    results = []
    for label, image in ARMS:
        record = {'arm': label, 'image': image, 'passed': False}
        try:
            if not rt.boot(label, image=image, model=MODEL, tp=4, dcp=1, spec='mtp0', cache='vram',
                extra_env={'MAX_NUM_SEQS': '32', 'MAX_CUDAGRAPH_CAPTURE_SIZE': '32',
                           'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32'},
                extra_args=['--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
                            '--override-generation-config', '{"temperature":1.0,"top_p":0.95}']):
                raise RuntimeError('Model did not become healthy')
            with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{label}.steady.metrics.jsonl'):
                ran = rt.bench(label, conc='1,4,8', contexts='0,32k', duration=30)
            if not ran:
                raise RuntimeError('Benchmark process failed or produced no readable receipt')
            result = json.loads((rt.ROOT / f'{label}.json').read_text())
            rows = result.get('results', [])
            observed = {(int(row['concurrency']), int(row['context_tokens'])) for row in rows}
            issues = []
            if observed != EXPECTED_CELLS or len(rows) != len(EXPECTED_CELLS):
                issues.append('Measured cell coverage does not match the declared matrix')
            for row in rows:
                if row.get('num_errors', 0) or row.get('underfilled') or row.get('warmup_timed_out') or row.get('capacity_limited') or row.get('aggregate_tps', 0) <= 0 or row.get('measurement_seconds', 0) < 28:
                    issues.append({'cell': [row.get('concurrency'), row.get('context_tokens')],
                        'num_errors': row.get('num_errors'), 'underfilled': row.get('underfilled'),
                        'warmup_timed_out': row.get('warmup_timed_out'), 'capacity_limited': row.get('capacity_limited'),
                        'measurement_seconds': row.get('measurement_seconds')})
            metrics = steady_metrics.summarize(rt.ROOT, label)
            record.update({'passed': not issues, 'issues': issues,
                'benchmark': str(rt.ROOT / f'{label}.json'),
                'counter_windows_valid': metrics['all_windows_valid'],
                'counter_summary': str(rt.ROOT / f'{label}.steady-summary.json')})
            rt.record_gate('runtime-baseline:' + label, record['passed'], record)
        except Exception as error:
            record['error'] = repr(error)
            rt.record_gate('runtime-baseline:' + label, False, record)
        finally:
            results.append(record)
            rt.save_json('runtime-baseline-progress.json', results)
            rt.stop()
    summary = {'complete': len(results) == len(ARMS), 'passed': all(row['passed'] for row in results), 'arms': results}
    rt.save_json('runtime-baseline-summary.json', summary)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    require_inputs()
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        print('R29 RUNTIME BASELINE: guarded coordinator starting', flush=True)
        coordinator.PHASES = [('runtime-baseline', str(Path(__file__).resolve()), ['--phase'], 14400)]
        coordinator.main()
    else:
        raise SystemExit('Usage: runtime_baseline.py [--phase]')
