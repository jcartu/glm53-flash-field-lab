#!/usr/bin/env python3
"""DFlash2 C8 verifier-step collapse confirmation: capture-size 64 versus pin 32."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r26')
import runtime as rt
import run_qualification as coordinator
import steady_metrics

IMAGE = 'localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49'
MODEL = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-published-46aaae8a')
MATRIX_PINS = {'MAX_NUM_SEQS': '32', 'MAX_CUDAGRAPH_CAPTURE_SIZE': '32', 'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32'}
CAPTURE_64_PINS = {'MAX_NUM_SEQS': '32', 'MAX_CUDAGRAPH_CAPTURE_SIZE': '64', 'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32 64'}
EXTRA_ARGS = ['--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
              '--override-generation-config', '{"temperature":1.0,"top_p":0.95}']
EXPECTED_CELLS = {(4, 0), (4, 32768), (8, 0), (8, 32768)}
COLLAPSE_RATIO = 5.0


def refuse_unsafe_start():
    inspected = rt._inspect()
    if inspected is not None:
        raise RuntimeError(f'Test container {rt.NAME} still exists; wait for its campaign to finish')
    for pattern in ('speculative_matrix.py', 'r30_cache_campaign.py', 'cache_lifecycle_probe.py'):
        if subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout.strip():
            raise RuntimeError(f'Another campaign process is running: {pattern}')
    occupied = subprocess.run(
        ['docker', 'ps', '-a', '--filter', 'name=r29-comparison', '--format', '{{.ID}}'],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout.strip()
    if occupied:
        raise RuntimeError('r29-comparison container still exists; GPUs are not free')


def cell_metrics(label):
    result = json.loads((rt.ROOT / f'{label}.json').read_text())
    measured = result.get('results', [])
    coverage = {(row['concurrency'], row['context_tokens']) for row in measured}
    client_valid = (
        coverage == EXPECTED_CELLS
        and len(measured) == len(EXPECTED_CELLS)
        and all(
            row.get('aggregate_tps', 0) > 0 and not row.get('num_errors')
            and not row.get('underfilled') and not row.get('warmup_timed_out')
            and not row.get('capacity_limited')
            and row.get('measurement_seconds', 0) >= 28
            for row in measured
        )
    )
    counter = steady_metrics.summarize(rt.ROOT, label)
    cells = {}
    for row in counter['cells']:
        if not row.get('valid'):
            continue
        cells[(row['concurrency'], row['context_tokens'])] = {
            'tok_s': row.get('output_tokens_per_second'),
            'verifier_steps_s': row.get('aggregate_verifier_steps_per_second'),
        }
    return {
        'client_timing_valid': client_valid,
        'counter_windows_valid': counter['all_windows_valid'],
        'cells': {f'c{key[0]}-ctx{key[1]}': value for key, value in sorted(cells.items())},
    }


def run_arm(arm, extra_env):
    label = f'published-dcp1-dflash2-{arm}'
    record = {'label': label, 'arm': arm, 'extra_env': extra_env, 'complete': False}
    try:
        if not rt.boot(label, image=IMAGE, model=MODEL, tp=4, dcp=1, spec='dflash2', cache='vram',
            extra_env=extra_env, extra_args=EXTRA_ARGS):
            raise RuntimeError('Confirmation arm did not boot')
        with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{label}.steady.metrics.jsonl'):
            ran = rt.bench(label, conc='4,8', contexts='0,32k', duration=30)
        if not ran:
            raise RuntimeError('Benchmark command failed')
        record.update(cell_metrics(label))
        record['complete'] = bool(record['client_timing_valid'] and record['counter_windows_valid'])
        rt.record_gate('dflash2-c8-confirmation:' + label, record['complete'], record)
    except Exception as error:
        record['error'] = repr(error)
        rt.record_gate('dflash2-c8-confirmation:' + label, False, record)
    finally:
        rt.stop()
    return record


def c8_values(record, field):
    return [
        cell[field]
        for key, cell in record.get('cells', {}).items()
        if key.startswith('c8-') and cell.get(field) is not None
    ]


def phase():
    rt.save_json('dflash2-c8-confirmation-plan.json', {
        'image': IMAGE, 'model': str(MODEL), 'topology': 'TP4/DCP1 cache=vram kv=fp8_ds_mla spec=dflash2',
        'arms': {'capture32': MATRIX_PINS, 'capture64': CAPTURE_64_PINS},
        'bench': {'concurrency': '4,8', 'contexts': '0,32k', 'duration_seconds': 30, 'repeats': 1},
        'hypothesis': 'DFlash K7 at C8 pads the verifier decode batch to 8x8=64 tokens, exceeding the matrix-pinned MAX_CUDAGRAPH_CAPTURE_SIZE=32, so no CUDA graph covers the C8 verify step and the verifier step rate collapses; capture size 64 should restore it.',
        'decision_rule': f'hypothesis_supported = both arms complete AND min(capture64 C8 verifier steps/s) >= {COLLAPSE_RATIO} x max(capture32 C8 verifier steps/s); missing speculation counters fail closed.',
        'scope': 'Confirmation on published/DCP1 only; does not re-qualify other matrix arms or checkpoints.',
    })
    control = run_arm('capture32', MATRIX_PINS)
    treated = run_arm('capture64', CAPTURE_64_PINS)
    control_steps = c8_values(control, 'verifier_steps_s')
    treated_steps = c8_values(treated, 'verifier_steps_s')
    ratio = (
        min(treated_steps) / max(control_steps)
        if control_steps and treated_steps and max(control_steps) > 0
        else None
    )
    hypothesis_supported = bool(
        control['complete'] and treated['complete']
        and control_steps and treated_steps
        and ratio is not None and ratio >= COLLAPSE_RATIO
    )
    summary = {
        'arms': {'capture32': control, 'capture64': treated},
        'c8_comparison': {
            'capture32_c8_verifier_steps_s': control_steps,
            'capture64_c8_verifier_steps_s': treated_steps,
            'capture32_c8_tok_s': c8_values(control, 'tok_s'),
            'capture64_c8_tok_s': c8_values(treated, 'tok_s'),
            'min_treated_over_max_control': ratio,
            'threshold': COLLAPSE_RATIO,
        },
        'hypothesis': 'DFlash K7 at C8 pads to 8x8=64 tokens > MAX_CUDAGRAPH_CAPTURE_SIZE=32 (no graph); capture size 64 restores the verifier step rate.',
        'hypothesis_supported': hypothesis_supported,
        'passed': bool(control['complete'] and treated['complete'] and hypothesis_supported),
        'scope': 'One boot and one 30s bench per arm on published/DCP1; confirms or refutes the capture-size mechanism on this host only.',
    }
    rt.save_json('dflash2-c8-confirmation-summary.json', summary)
    rt.record_gate('dflash2-c8-confirmation', summary['passed'], summary)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        refuse_unsafe_start()
        print('DFLASH2 C8 CONFIRMATION: guarded coordinator starting', flush=True)
        coordinator.PHASES = [('dflash2-c8-confirmation', str(Path(__file__).resolve()), ['--phase'], 3600)]
        coordinator.main()
    else:
        raise SystemExit('Usage: dflash2_c8_confirmation.py [--phase]')
