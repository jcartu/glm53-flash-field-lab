#!/usr/bin/env python3
"""Separate stock/#64 RAM performance, GPU transport, and bounded L2 qualification."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import re
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r26'))
import runtime as rt
import run_qualification as coordinator
import steady_metrics

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909')
STOCK = 'localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49'
MODEL = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-published-46aaae8a')
COMMON_ARGS = ['--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
               '--override-generation-config', '{"temperature":1.0,"top_p":0.95}']

EXACT_PAYLOAD_DEDUP_GATE = 'replay.exact_payload_dedup'
REQUIRED_CANDIDATE_ARMS = tuple(itertools.product((1, 4), ('mtp0', 'mtp3', 'dflash2')))
REQUIRED_RAM_TAGS = ('stock', 'pr64')
MIXED_RESTORE_ARM = (4, 'mtp3')


def ram_trial_valid(rows):
    expected = {(concurrency, context) for concurrency in (1, 8) for context in (0, 32768)}
    return (
        len(rows) == len(expected)
        and {(row.get('concurrency'), row.get('context_tokens')) for row in rows} == expected
        and all(
            row.get('aggregate_tps', 0) > 0
            and not row.get('num_errors')
            and not row.get('underfilled')
            and not row.get('warmup_timed_out')
            and not row.get('capacity_limited')
            and row.get('measurement_seconds', 0) >= 28
            for row in rows
        )
    )


def lifecycle_arm_passed(record):
    probe_passed = (
        record.get('returncode') == 0
        and record.get('complete') is True
        and record.get('receipt_passed') is True
    )
    if (record.get('dcp'), record.get('spec')) != MIXED_RESTORE_ARM:
        return probe_passed
    mixed = record.get('mixed_restore')
    return bool(
        probe_passed
        and isinstance(mixed, dict)
        and mixed.get('returncode') == 0
        and mixed.get('complete') is True
        and mixed.get('passed') is True
    )


def candidate_coverage(records, candidate_tag='pr64'):
    candidates = [row for row in records if row.get('tag') == candidate_tag]
    observed = [(row.get('dcp'), row.get('spec')) for row in candidates]
    missing = [
        {'dcp': dcp, 'spec': spec}
        for dcp, spec in REQUIRED_CANDIDATE_ARMS
        if observed.count((dcp, spec)) == 0
    ]
    duplicates = [
        {'dcp': dcp, 'spec': spec, 'count': observed.count((dcp, spec))}
        for dcp, spec in REQUIRED_CANDIDATE_ARMS
        if observed.count((dcp, spec)) > 1
    ]
    unexpected = [
        {'dcp': dcp, 'spec': spec}
        for dcp, spec in observed
        if (dcp, spec) not in REQUIRED_CANDIDATE_ARMS
    ]
    return {
        'expected_arm_count': len(REQUIRED_CANDIDATE_ARMS),
        'observed_arm_count': len(candidates),
        'missing_arms': missing,
        'duplicate_arms': duplicates,
        'unexpected_arms': unexpected,
        'complete': (
            len(candidates) == len(REQUIRED_CANDIDATE_ARMS)
            and not missing
            and not duplicates
            and not unexpected
        ),
    }


def assess_stock_negative_control(records):
    arms = [
        row for row in records
        if row.get('tag') == 'stock'
        and (row.get('dcp'), row.get('spec')) == MIXED_RESTORE_ARM
    ]
    assessment = {
        'gate_name': EXACT_PAYLOAD_DEDUP_GATE,
        'expected_status': 'fail',
        'arm_count': len(arms),
        'classification': 'invalid',
        'valid': False,
        'passed': False,
        'inverted': False,
    }
    if len(arms) != 1:
        assessment['reason'] = (
            'required stock DCP4/MTP3 lifecycle arm is missing'
            if not arms else
            'required stock DCP4/MTP3 lifecycle arm is duplicated'
        )
        return assessment

    arm = arms[0]
    matches = arm.get('exact_payload_dedup_gates')
    matches = matches if isinstance(matches, list) else []
    assessment.update({
        'arm': arm.get('label'),
        'returncode': arm.get('returncode'),
        'complete': arm.get('complete') is True,
        'receipt_passed': arm.get('receipt_passed'),
        'fatal_error': arm.get('fatal_error'),
        'matching_gate_count': len(matches),
        'observed_gate': matches[0] if len(matches) == 1 else None,
    })
    if len(matches) == 1:
        assessment['inverted'] = matches[0].get('status') == 'pass'

    if arm.get('complete') is not True or arm.get('returncode') == 2:
        assessment['reason'] = 'stock lifecycle probe aborted or did not complete'
    elif len(matches) != 1:
        assessment['reason'] = (
            'specific exact-payload-dedup gate is missing'
            if not matches else
            'specific exact-payload-dedup gate is duplicated'
        )
    elif matches[0].get('status') == 'pass':
        assessment.update({
            'classification': 'inverted',
            'reason': 'stock unexpectedly passed the exact-payload-dedup gate',
        })
    elif matches[0].get('status') != 'fail':
        assessment['reason'] = (
            'exact-payload-dedup gate did not produce the required fail status'
        )
    elif matches[0].get('passed') is not False:
        assessment['reason'] = 'exact-payload-dedup gate status and passed flag disagree'
    elif matches[0].get('required') is not True:
        assessment['reason'] = 'exact-payload-dedup gate was not exercised as a required gate'
    elif arm.get('fatal_error') is not None:
        assessment['reason'] = 'stock lifecycle receipt contains a fatal error'
    elif arm.get('receipt_passed') is not False:
        assessment['reason'] = 'stock lifecycle result is inconsistent with the required gate failure'
    elif arm.get('returncode') != 1:
        assessment['reason'] = 'stock lifecycle command did not return the expected completed-failure status'
    else:
        assessment.update({
            'classification': 'expected_failure_observed',
            'valid': True,
            'passed': True,
            'reason': 'stock completed and failed the required exact-payload-dedup gate',
        })
    return assessment


def build_campaign_summary(ram, disks, candidate_tag='pr64'):
    coverage = candidate_coverage(disks, candidate_tag)
    candidates = [row for row in disks if row.get('tag') == candidate_tag]
    candidate_passed = bool(
        coverage['complete']
        and all(
            row.get('passed') is True and lifecycle_arm_passed(row)
            for row in candidates
        )
    )
    ram_counts = {
        tag: sum(row.get('tag') == tag for row in ram)
        for tag in REQUIRED_RAM_TAGS
    }
    unexpected_ram = [
        row.get('tag') for row in ram if row.get('tag') not in REQUIRED_RAM_TAGS
    ]
    ram_coverage = {
        'expected_tags': list(REQUIRED_RAM_TAGS),
        'observed_counts': ram_counts,
        'unexpected_tags': unexpected_ram,
        'complete': (
            len(ram) == len(REQUIRED_RAM_TAGS)
            and all(ram_counts[tag] == 1 for tag in REQUIRED_RAM_TAGS)
            and not unexpected_ram
        ),
    }
    ram_passed = bool(
        ram_coverage['complete'] and all(row.get('passed') is True for row in ram)
    )
    control = assess_stock_negative_control(disks)
    stock_arms = [
        row for row in disks
        if row.get('tag') == 'stock'
        and (row.get('dcp'), row.get('spec')) == MIXED_RESTORE_ARM
    ]
    candidate_execution_complete = bool(
        coverage['complete']
        and all(
            row.get('complete') is True
            and (
                (row.get('dcp'), row.get('spec')) != MIXED_RESTORE_ARM
                or (
                    isinstance(row.get('mixed_restore'), dict)
                    and row['mixed_restore'].get('complete') is True
                )
            )
            for row in candidates
        )
    )
    complete = bool(
        ram_coverage['complete']
        and candidate_execution_complete
        and len(stock_arms) == 1
        and stock_arms[0].get('complete') is True
    )
    passed = bool(candidate_passed and ram_passed and control['valid'])
    return {
        'complete': complete,
        'passed': passed,
        'ram_performance': ram,
        'ram_performance_coverage': ram_coverage,
        'ram_performance_passed': ram_passed,
        'lifecycle': disks,
        'candidate_coverage': coverage,
        'candidate_passed': candidate_passed,
        'stock_negative_control': control,
        'dedup_comparison_certified': bool(candidate_passed and control['valid']),
        'scope': 'Actual tested gates only. Failed/unavailable gates are retained; complete execution is not a blanket promotion.',
    }


def artifacts():
    result = json.loads((ROOT / 'lmcache64-candidate-images.json').read_text())
    if result.get('cpu_tests', {}).get('passed') != 205:
        raise RuntimeError('Candidate CPU source qualification is missing')
    if not result['runtime_image'].startswith('sha256:') or not result['validation_image'].startswith('sha256:'):
        raise RuntimeError('Cache images must use immutable local identities')
    draft = json.loads((ROOT / 'draft-verification.json').read_text())
    if not draft.get('passed'):
        raise RuntimeError('Unverified draft')
    rt.DRAFT = Path(draft['path'])
    return result


def gpu_components(info):
    records = []
    for gpu in range(4):
        rt.stop()
        label = f'cuda-block-id-lifetime-gpu{gpu}'
        command = ['docker', 'run', '--name', rt.NAME, '--label', 'field-lab.battery=r26', '--init',
            '--gpus', f'"device={gpu}"', '--network', 'none', '--shm-size', '1g',
            '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
            '--workdir', '/opt/lmcache64-validation', '--entrypoint', '/opt/venv/bin/python',
            info['validation_image'], '-m', 'pytest', '-q',
            'tests/v1/multiprocess/test_paged_transfer_metadata_lifetime.py::test_native_gather_with_delayed_cuda_stream']
        code = rt.run(command, label=label, timeout=900)
        rt.capture(label)
        text = (rt.ROOT / f'{label}.log').read_text()
        passed = code == 0 and re.search(r'\b2 passed\b', text) is not None and re.search(r'\b[1-9][0-9]* skipped\b', text) is None
        records.append({'gpu': gpu, 'passed': passed, 'returncode': code})
        rt.record_gate(label, passed, records[-1])
        rt.stop()
        if not passed:
            raise RuntimeError('Native asynchronous block-ID test failed on a physical GPU')
    parent = rt.ROOT / 'gpu-checkpoint-bytes'
    parent.mkdir(parents=True, exist_ok=False)
    command = ['docker', 'run', '--name', rt.NAME, '--label', 'field-lab.battery=r26', '--init',
        '--gpus', '"device=0,1,2,3"', '--network', 'none', '--shm-size', '1g',
        '-e', 'CUDA_VISIBLE_DEVICES=0,1,2,3', '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
        '-v', f'{HERE / "gpu_checkpoint_roundtrip.py"}:/probe.py:ro',
        '-v', f'{parent}:/component-output', '--entrypoint', '/opt/venv/bin/python',
        info['validation_image'], '/probe.py', '--output-dir', '/component-output/result']
    code = rt.run(command, label='gpu-checkpoint-roundtrip', timeout=1200)
    rt.capture('gpu-checkpoint-roundtrip')
    result_path = parent / 'result/result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    passed = code == 0 and isinstance(result, dict) and result.get('passed') is True
    rt.record_gate('gpu-checkpoint-roundtrip', passed, {'returncode': code, 'result': str(result_path)})
    rt.stop()
    rt.save_json('gpu-component-summary.json', {'native_metadata_lifetime': records, 'checkpoint_roundtrip_passed': passed, 'result': str(result_path)})
    if not passed:
        raise RuntimeError('Four-GPU checkpoint byte roundtrip failed')


def cache_env(label, host_dir, l2):
    return {'LMCACHE_L2_HOST_DIR': str(host_dir), 'LMCACHE_ENABLED': '1',
        'LMCACHE_TRANSFER_MODE': 'engine_driven', 'LMCACHE_L1_SIZE_GB': '8',
        'LMCACHE_L2_ENABLED': '1' if l2 else '0', 'LMCACHE_L2_MAX_CAPACITY_GB': '8',
        'LMCACHE_INSTANCE_ID': label, 'LMCACHE_SHM_NAME': label,
        'MAX_MODEL_LEN': '262144', 'MAX_NUM_SEQS': '32',
        'MAX_CUDAGRAPH_CAPTURE_SIZE': '32', 'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32'}


def ram_performance(info, run_id):
    results = []
    for tag, image in [('stock', STOCK), ('pr64', info['runtime_image'])]:
        label = f'ram-{tag}-mtp3-dcp4'
        host_dir = rt.L2_HOST_ROOT / ('r29-' + run_id) / label
        record = {'label': label, 'tag': tag, 'image': image, 'passed': False}
        try:
            if not rt.boot(label, image=image, model=MODEL, tp=4, dcp=4, spec='mtp3', cache='lmcache',
                extra_env=cache_env(label + '-' + run_id[:8], host_dir, False), extra_args=COMMON_ARGS):
                raise RuntimeError('RAM-only cache arm did not boot')
            trials = []
            for trial in (1, 2):
                name = label + '-trial' + str(trial)
                with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{name}.steady.metrics.jsonl'):
                    code_ok = rt.bench(name, conc='1,8', contexts='0,32k', duration=30)
                if not code_ok:
                    raise RuntimeError('RAM-only benchmark execution failed')
                data = json.loads((rt.ROOT / f'{name}.json').read_text())
                client_timing_valid = ram_trial_valid(data.get('results', []))
                counters = steady_metrics.summarize(rt.ROOT, name)
                trials.append({
                    'label': name,
                    'client_timing_valid': client_timing_valid,
                    'counter_windows_valid': counters['all_windows_valid'],
                })
            client_timing_valid = bool(
                len(trials) == 2
                and all(row['client_timing_valid'] for row in trials)
            )
            record.update({
                'trials': trials,
                'client_timing_valid': client_timing_valid,
                'supplemental_counter_windows_valid': bool(
                    len(trials) == 2
                    and all(row['counter_windows_valid'] for row in trials)
                ),
                'passed': client_timing_valid,
            })
        except Exception as error:
            record['error'] = repr(error)
        finally:
            results.append(record)
            rt.record_gate('cache-ram-performance:' + tag, record['passed'], record)
            rt.save_json('cache-ram-performance-progress.json', results)
            rt.stop()
    return results


def lifecycle(info, run_id, arms=None):
    candidate_tag = info.get('candidate_tag', 'pr64')
    if arms is None:
        arms = [('stock', STOCK, 4, 'mtp3')] + [(candidate_tag, info['runtime_image'], dcp, spec) for dcp, spec in REQUIRED_CANDIDATE_ARMS]
    results = []
    for tag, image, dcp, spec in arms:
        label = f'l2-{tag}-dcp{dcp}-{spec}'
        host_dir = rt.L2_HOST_ROOT / ('r29-' + run_id) / label
        output = rt.ROOT / label
        record = {
            'label': label,
            'tag': tag,
            'image': image,
            'dcp': dcp,
            'spec': spec,
            'complete': False,
            'passed': False,
        }
        halt = False
        try:
            if not rt.boot(label, image=image, model=MODEL, tp=4, dcp=dcp, spec=spec, cache='lmcache',
                extra_env=cache_env(label + '-' + run_id[:8], host_dir, True), extra_args=COMMON_ARGS + ['--enable-prompt-tokens-details', '--enable-per-request-metrics']):
                raise RuntimeError('External-cache arm did not boot')
            code = rt.run([sys.executable, str(HERE / 'cache_lifecycle_probe.py'),
                '--base-url', rt.BASE_URL, '--model', rt.MODEL_NAME, '--container', rt.NAME,
                '--l2-dir', str(host_dir), '--output-dir', str(output), '--context-tokens', '32768', '--growth-turns', '6'],
                label=label + '-probe', timeout=15000)
            path = output / 'receipt.json'
            receipt = json.loads(path.read_text()) if path.exists() else {}
            gates = receipt.get('gates')
            gates = [row for row in gates if isinstance(row, dict)] if isinstance(gates, list) else []
            failed_gate_records = [row for row in gates if row.get('status') == 'fail']
            unavailable_gate_records = [row for row in gates if row.get('status') == 'unavailable']
            record.update({
                'returncode': code,
                'command_succeeded': code == 0,
                'receipt': str(path),
                'complete': receipt.get('complete') is True,
                'receipt_passed': receipt.get('passed') if isinstance(receipt.get('passed'), bool) else None,
                'fatal_error': receipt.get('fatal_error'),
                'failed_gates': [row.get('name') for row in failed_gate_records],
                'unavailable_gates': [row.get('name') for row in unavailable_gate_records],
                'failed_gate_records': failed_gate_records,
                'unavailable_gate_records': unavailable_gate_records,
                'exact_payload_dedup_gates': [
                    row for row in gates if row.get('name') == EXACT_PAYLOAD_DEDUP_GATE
                ],
            })
            if code not in (0, 1) or not record['complete']:
                record['error'] = repr(RuntimeError(
                    'Cache probe aborted; preserve this pilot before any continuation'
                ))
                halt = True
            elif dcp == 4 and spec == 'mtp3':
                mixed_output = rt.ROOT / (label + '-mixed-restore')
                mixed_code = rt.run([sys.executable, str(HERE / 'mixed_restore_probe.py'),
                    '--base-url', rt.BASE_URL, '--model', rt.MODEL_NAME, '--container', rt.NAME,
                    '--manifest', str(ROOT / 'history-fixtures/manifest.json'),
                    '--output-dir', str(mixed_output)], label=label + '-mixed-restore', timeout=7200)
                mixed_path = mixed_output / 'summary.json'
                mixed = json.loads(mixed_path.read_text()) if mixed_path.exists() else {}
                record['mixed_restore'] = {
                    'returncode': mixed_code,
                    'command_succeeded': mixed_code == 0,
                    'summary': str(mixed_path),
                    'complete': mixed.get('complete') is True,
                    'passed': mixed.get('passed') is True,
                    'mixed_source_counts_observed': mixed.get('mixed_source_counts_observed'),
                }
                record['mixed_restore']['qualified'] = bool(
                    record['mixed_restore']['command_succeeded']
                    and record['mixed_restore']['complete']
                    and record['mixed_restore']['passed']
                )
                rt.record_gate(
                    'mixed-restore:' + label,
                    record['mixed_restore']['qualified'],
                    record['mixed_restore'],
                )
            record['passed'] = lifecycle_arm_passed(record)
        except Exception as error:
            record['error'] = repr(error)
            record['passed'] = False
            halt = True
        finally:
            results.append(record)
            rt.record_gate('cache-lifecycle:' + label, record['passed'], record)
            rt.save_json('cache-lifecycle-progress.json', results)
            rt.stop()
        if halt:
            break
    return results


def phase():
    info = artifacts()
    run_id = uuid.uuid4().hex[:12]
    rt.save_json('cache-campaign-plan.json', {'candidate': info, 'run_id': run_id,
        'l1_gib': 8, 'l2_gib': 8,
        'stock_negative_control': f'Stock R29 must complete and fail required gate {EXACT_PAYLOAD_DEDUP_GATE}; this is assessed separately and is not a candidate result.',
        'reclamation_limit': 'PR64 has no reference-counted payload reclamation. Orphan/cancellation/pressure gates remain real gates and may fail.',
        'production_policy': 'Original container restored unchanged; all L2 namespaces are new and bounded.'})
    gpu_components(info)
    ram = ram_performance(info, run_id)
    disks = lifecycle(info, run_id)
    summary = build_campaign_summary(ram, disks)
    rt.save_json('cache-campaign-summary.json', summary)
    rt.record_gate(
        'cache-stock-negative-control',
        summary['stock_negative_control']['valid'],
        summary['stock_negative_control'],
    )
    rt.record_gate('cache-campaign', summary['passed'], {
        'complete': summary['complete'],
        'candidate_passed': summary['candidate_passed'],
        'ram_performance_passed': summary['ram_performance_passed'],
        'negative_control_valid': summary['stock_negative_control']['valid'],
        'negative_control_inverted': summary['stock_negative_control']['inverted'],
    })
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    artifacts()
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        print('R29 CACHE CAMPAIGN: guarded coordinator starting', flush=True)
        coordinator.PHASES = [('cache-campaign', str(Path(__file__).resolve()), ['--phase'], 172800)]
        coordinator.main()
    else:
        raise SystemExit('Usage: cache_campaign.py [--phase]')
