#!/usr/bin/env python3
"""R31 checkpoint comparison: nvidia NVFP4 versus published/QAD2500 plus R31 sanity arms."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path('/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r29')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'r26'))
import runtime as rt
import run_qualification as coordinator
import steady_metrics
import cache_campaign as campaign
import speculative_matrix as matrix

ROOT = campaign.ROOT
R31 = 'localinferencelab/vllm@sha256:da7157d5649a85298635c44b03eeb127837448fa42a45388e48ad9de4a99ff39'
PUBLISHED = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-published-46aaae8a')
QAD2500 = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD2500-3959f8a0')
TVN = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD-TVN-step1500')
READINESS = ROOT / 'nvidia-nvfp4-readiness/readiness.json'
COMMON_ARGS = ['--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
               '--override-generation-config', '{"temperature":1.0,"top_p":0.95}']
MATRIX_PINS = {'MAX_NUM_SEQS': '32', 'MAX_CUDAGRAPH_CAPTURE_SIZE': '32', 'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32'}
DFLASH2_ENV = {'MAX_NUM_SEQS': '32'}
CHECKPOINT_CELLS = {(c, ctx) for c in (1, 4, 8) for ctx in (0, 32768)}
DFLASH2_CELLS = {(c, ctx) for c in (4, 8) for ctx in (0, 32768)}
WITH_TVN = '--with-tvn' in sys.argv


def require_readiness():
    if not READINESS.is_file():
        raise RuntimeError('NVIDIA NVFP4 readiness receipt is missing: ' + str(READINESS))
    readiness = json.loads(READINESS.read_text())
    if readiness.get('loadable_mtp0') is not True:
        raise RuntimeError('NVIDIA NVFP4 is not loadable for mtp0 per the readiness receipt')
    local_dir = readiness.get('local_dir')
    if not local_dir or not Path(local_dir).is_dir():
        raise RuntimeError('NVIDIA NVFP4 local_dir is missing or absent on disk')
    return readiness


def refuse_unsafe_start():
    # No self-name pattern here: this process and its tmux pane carry it.
    for pattern in ('speculative_matrix.py', 'r30_cache_campaign.py', 'r30_dcp4_identity_followup.py'):
        if subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout.strip():
            raise RuntimeError(f'Another campaign process is running: {pattern}')
    occupied = subprocess.run(
        ['docker', 'ps', '-a', '--filter', 'name=r29-comparison', '--format', '{{.ID}}'],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout.strip()
    if occupied:
        raise RuntimeError('r29-comparison container still exists; GPUs are not free')
    require_readiness()


def bench_cells(label, expected):
    result = json.loads((rt.ROOT / f'{label}.json').read_text())
    measured = result.get('results', [])
    coverage = {(row['concurrency'], row['context_tokens']) for row in measured}
    valid = (
        coverage == expected and len(measured) == len(expected)
        and all(
            row.get('aggregate_tps', 0) > 0 and not row.get('num_errors')
            and not row.get('underfilled') and not row.get('warmup_timed_out')
            and not row.get('capacity_limited')
            and row.get('measurement_seconds', 0) >= 28
            for row in measured
        )
    )
    counter = steady_metrics.summarize(rt.ROOT, label)
    cells = {
        f'c{row["concurrency"]}-ctx{row["context_tokens"]}': {
            'tok_s': row.get('output_tokens_per_second'),
            'verifier_steps_s': row.get('aggregate_verifier_steps_per_second'),
        }
        for row in counter['cells'] if row.get('valid')
    }
    return valid, counter['all_windows_valid'], cells


def checkpoint_arm(label, model, cases, sources, extra_env=None):
    record = {'arm': label, 'image': R31, 'model': str(model), 'complete': False}
    try:
        for path, expected in sources.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen execution input changed: ' + path)
        if not rt.boot(label, image=R31, model=model, tp=4, dcp=1, spec='mtp0', cache='vram',
            extra_env=extra_env or MATRIX_PINS, extra_args=COMMON_ARGS):
            raise RuntimeError('Model boot failed')
        record['sentinels'] = matrix.sentinels(label, cases)
        output = rt.ROOT / label / 'behavior'
        code = rt.run([sys.executable, str(HERE / 'behavior_probe.py'), '--base-url', rt.BASE_URL,
            '--model', rt.MODEL_NAME, '--fixture', str(ROOT / 'behavior-fixtures.json'),
            '--output-dir', str(output), '--concurrency', '4', '--clear-thinking', 'false'],
            label=label + '-behavior', timeout=7200)
        record['behavior_returncode'] = code
        record['behavior_summary'] = str(output / 'summary.json')
        summary = json.loads((output / 'summary.json').read_text()) if code == 0 else {}
        rescore_output = output / 'semantic-rescore-v2.json'
        rescore_code = rt.run([sys.executable, str(HERE / 'rescore_behavior.py'),
            '--fixture', str(ROOT / 'behavior-fixtures.json'),
            '--summary', str(output / 'summary.json'), '--output', str(rescore_output)],
            label=label + '-semantic-rescore', timeout=3600)
        record['semantic_rescore'] = str(rescore_output)
        rescore = json.loads(rescore_output.read_text()) if rescore_code == 0 else {}
        record['strict_tasks'] = {'passed': summary.get('task_passed'), 'expected': summary.get('expected'),
            'completed': summary.get('completed')}
        record['semantic_tasks'] = {'correct': rescore.get('semantic_correct'),
            'assessed': rescore.get('semantic_assessed')}
        if not rt.wait_health(timeout=30):
            raise RuntimeError('Server became unhealthy before history controls')
        output = rt.ROOT / label / 'history'
        code = rt.run([sys.executable, str(HERE / 'history_probe.py'), '--base-url', rt.BASE_URL,
            '--model', rt.MODEL_NAME, '--manifest', str(ROOT / 'history-fixtures/manifest.json'),
            '--output-dir', str(output)], label=label + '-history', timeout=7200)
        record['history_returncode'] = code
        record['history_summary'] = str(output / 'summary.json')
        cleared = matrix.reset(rt.BASE_URL)
        rt.save_json(label + '-reset.json', cleared)
        if not cleared['passed']:
            raise RuntimeError('Could not establish a clean benchmark start')
        with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{label}.steady.metrics.jsonl'):
            ran = rt.bench(label, conc='1,4,8', contexts='0,32k', duration=30)
        if not ran:
            raise RuntimeError('Benchmark command failed')
        client_valid, counters_valid, cells = bench_cells(label, CHECKPOINT_CELLS)
        record['bench_cells'] = cells
        record['client_timing_valid'] = client_valid
        record['counter_windows_valid'] = counters_valid
        record['complete'] = bool(
            record['sentinels']['passed'] and record['behavior_returncode'] == 0
            and record['history_returncode'] == 0 and client_valid
        )
        rt.record_gate('r31-comparison:' + label, record['complete'], record)
    except Exception as error:
        record['error'] = repr(error)
        rt.record_gate('r31-comparison:' + label, False, record)
    finally:
        rt.stop()
    return record


def dflash2_sanity_arm():
    label = 'published-dcp1-dflash2'
    record = {'arm': label, 'image': R31, 'model': str(PUBLISHED), 'complete': False,
        'capture_pins': 'R31 launcher defaults (no MAX_CUDAGRAPH_CAPTURE_SIZE/CUDAGRAPH_CAPTURE_SIZES override)'}
    try:
        if not rt.boot(label, image=R31, model=PUBLISHED, tp=4, dcp=1, spec='dflash2', cache='vram',
            extra_env=DFLASH2_ENV, extra_args=COMMON_ARGS):
            raise RuntimeError('DFlash2 sanity arm did not boot')
        with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{label}.steady.metrics.jsonl'):
            ran = rt.bench(label, conc='4,8', contexts='0,32k', duration=30)
        if not ran:
            raise RuntimeError('DFlash2 sanity benchmark failed')
        client_valid, counters_valid, cells = bench_cells(label, DFLASH2_CELLS)
        record.update({'bench_cells': cells, 'client_timing_valid': client_valid,
            'counter_windows_valid': counters_valid,
            'c8_verifier_steps_s': [cell['verifier_steps_s'] for key, cell in cells.items()
                                    if key.startswith('c8-') and cell.get('verifier_steps_s') is not None],
            'complete': bool(client_valid and counters_valid)})
        rt.record_gate('r31-comparison:' + label, record['complete'], record)
    except Exception as error:
        record['error'] = repr(error)
        rt.record_gate('r31-comparison:' + label, False, record)
    finally:
        rt.stop()
    return record


def gate_statuses(record):
    try:
        receipt = json.loads(Path(record['receipt']).read_text())
    except (KeyError, OSError, ValueError, TypeError):
        return {}
    return {
        row.get('name'): row.get('status')
        for row in receipt.get('gates', [])
        if isinstance(row, dict) and row.get('name')
    }


def l2_sanity_arm(run_id):
    info = {'runtime_image': R31, 'candidate_tag': 'r31'}
    disks = campaign.lifecycle(info, run_id, arms=[('r31', R31, 4, 'mtp3')])
    record = {'arm': 'l2-r31-dcp4-mtp3', 'image': R31, 'complete': False}
    if len(disks) == 1:
        row = disks[0]
        record.update({
            'label': row.get('label'), 'complete': row.get('complete'),
            'returncode': row.get('returncode'),
            'failed_gates': row.get('failed_gates'),
            'unavailable_gates': row.get('unavailable_gates'),
            'gate_statuses': gate_statuses(row),
            'lifecycle_record': row,
        })
        record['complete'] = bool(row.get('complete') and row.get('passed') is True)
    rt.record_gate('r31-comparison:l2-r31-dcp4-mtp3', record['complete'], record)
    return record


def phase():
    readiness = require_readiness()
    draft, cases = matrix._checkpoint_inputs()
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        HERE / 'behavior_probe.py', HERE / 'history_probe.py', HERE / 'rescore_behavior.py',
        ROOT / 'behavior-fixtures.json', ROOT / 'history-fixtures/manifest.json']}
    arms = [('nvidia', Path(readiness['local_dir'])), ('published', PUBLISHED), ('qad2500', QAD2500)]
    if WITH_TVN:
        arms.append(('tvn1500', TVN))
    rt.save_json('r31-comparison-plan.json', {
        'image': R31, 'arms': [{'arm': label, 'model': str(model)} for label, model in arms],
        'topology': 'TP4/DCP1 mtp0 cache=vram kv=fp8_ds_mla, matrix capture pins',
        'sampling': 'temperature1, top_p0.95, max reasoning, clear_thinking=false',
        'sanity_arms': [
            'published-dcp1-dflash2 bench c4,8 with R31 launcher default capture sizes (C8 verifier steps/s)',
            'l2-r31-dcp4-mtp3 lifecycle arm via campaign.lifecycle (dedup + cold gates, fresh namespace)',
        ],
        'tvn_included': WITH_TVN,
        'scope': 'Checkpoint packages as shipped on R31; nvidia arm gated on readiness.json loadable_mtp0=true with required_env_first_boot merged into its boot env. NVIDIA speculative arms are excluded by design: the checkpoint ships a layer-45 BF16 MTP predictor incompatible with the quantized config (readiness loader_compatibility); only the mtp0 arm runs. Sanity arms are single-boot observations, not re-qualification.',
    })
    progress = []
    for label, model in arms:
        arm_env = None
        if label == 'nvidia':
            arm_env = {**MATRIX_PINS, **(readiness.get('required_env_first_boot') or {})}
        progress.append(checkpoint_arm(label, model, cases, sources, arm_env))
        rt.save_json('r31-comparison-progress.json', progress)
    run_id = uuid.uuid4().hex[:12]
    progress.append(dflash2_sanity_arm())
    progress.append(l2_sanity_arm(run_id))
    rt.save_json('r31-comparison-progress.json', progress)
    summary = {
        'all_arms_attempted': len(progress) == len(arms) + 2,
        'arms': progress,
        'passed': all(row.get('complete') is True for row in progress),
        'scope': 'Per-arm execution completeness only; no cross-checkpoint promotion verdict.',
    }
    rt.save_json('r31-comparison-summary.json', summary)
    rt.record_gate('r31-checkpoint-comparison', summary['passed'], summary)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        refuse_unsafe_start()
        print('R31 CHECKPOINT COMPARISON: guarded coordinator starting', flush=True)
        coordinator.PHASES = [('r31-checkpoint-comparison', str(Path(__file__).resolve()), ['--phase'], 21600)]
        coordinator.main()
    else:
        raise SystemExit('Usage: r31_checkpoint_comparison.py [--phase] [--with-tvn]')
