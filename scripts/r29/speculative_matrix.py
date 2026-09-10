#!/usr/bin/env python3
"""Matched R29 serving modes across the three pinned checkpoint packages."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r26'))
import runtime as rt
import run_qualification as coordinator
import steady_metrics
import quality_probes as legacy
from behavior_probe import equal_json, parse_answer
from history_probe import reset

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909')
IMAGE = 'localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49'
MODELS = {
    'published': Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-published-46aaae8a'),
    'qad2500': Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD2500-3959f8a0'),
    'tvn1500': Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD-TVN-step1500'),
}
EXPECTED = {(concurrency, context) for concurrency in (1, 4, 8) for context in (0, 32768)}

BASELINE_ROOT = ROOT / 'runtime-baseline'
BASELINE_IMAGE = 'voipmonitor/vllm@sha256:52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f'
BASELINE_ARMS = (
    ('r281-ab', BASELINE_IMAGE, 'r281'),
    ('r29-ab', IMAGE, 'r29'),
    ('r29-ba', IMAGE, 'r29'),
    ('r281-ba', BASELINE_IMAGE, 'r281'),
)
BASELINE_SAMPLING = {
    'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'max', 'clear_thinking': False,
}
BASELINE_LAUNCH_ENV = {
    'MODEL': '/model',
    'SERVED_MODEL_NAME': 'GLM-5.3-Flash-NVFP4',
    'HOST': '127.0.0.1',
    'PORT': '5002',
    'TP': '4',
    'DCP': '1',
    'CACHE_MODE': 'vram',
    'KV_CACHE_QUANT': 'fp8_ds_mla',
    'CUDAGRAPH_MODE': 'FULL_AND_PIECEWISE',
    'MAX_MODEL_LEN': '1048576',
    'MAX_NUM_SEQS': '32',
    'MAX_NUM_BATCHED_TOKENS': '4096',
    'PREFILL_SCHEDULE_INTERVAL': '1',
    'FAIRNESS_ENGINE': 'compute_share',
    'PREFILL_COMPUTE_SHARE': '0.4',
    'GPU_MEMORY_UTILIZATION': '0.93',
    'DCP_CKV_GATHER': 'auto',
    'NCCL_MIN_NCHANNELS': '16',
    'NCCL_MAX_NCHANNELS': '16',
    'NCCL_BUFFSIZE': '2097152',
    'OMP_NUM_THREADS': '1',
    'VLLM_SERVER_DEV_MODE': '1',
    'SPECULATOR': 'mtp',
    'MTP_DEPTH': '0',
    'MAX_CUDAGRAPH_CAPTURE_SIZE': '32',
    'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32',
}
BASELINE_EXTRA_ARGS = [
    '--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
    '--override-generation-config', '{"temperature":1.0,"top_p":0.95}',
]


class BaselineReuseError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise BaselineReuseError(message)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_number(value, expected):
    return _finite_number(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)

def _same_json(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _same_json(value[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _same_json(item, expected_item)
            for item, expected_item in zip(value, expected))
    return value == expected


def _reject_json_constant(value):
    raise ValueError(f'Non-finite JSON number {value}')


def _record_source(sources, role, path, encoded):
    source = {
        'role': role,
        'path': str(path),
        'sha256': hashlib.sha256(encoded).hexdigest(),
    }
    sources.append(source)
    return source


def _load_json_source(path, role, sources):
    path = Path(path).resolve()
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, ValueError) as error:
        raise BaselineReuseError(f'Cannot verify baseline source {path}: {error}') from error
    _record_source(sources, role, path, encoded)
    return value


def _load_gate_source(path, sources):
    path = Path(path).resolve()
    try:
        encoded = path.read_bytes()
        text = encoded.decode()
        lines = [line for line in text.splitlines() if line.strip()]
        records = [json.loads(line, parse_constant=_reject_json_constant) for line in lines]
    except (OSError, UnicodeError, ValueError) as error:
        raise BaselineReuseError(f'Cannot verify baseline gate source {path}: {error}') from error
    _require(records, 'Baseline gate source is empty')
    _record_source(sources, 'source_arm_gates', path, encoded)
    return records


def _require_source_path(value, expected, field):
    expected = Path(expected).resolve()
    _require(isinstance(value, str) and Path(value).is_absolute()
             and Path(value).resolve() == expected, f'{field} does not identify {expected}')


def _validate_benchmark_receipt(receipt, arm):
    _require(isinstance(receipt, dict), f'{arm} benchmark receipt is not an object')
    metadata = receipt.get('metadata')
    _require(isinstance(metadata, dict), f'{arm} benchmark metadata is missing')
    expected_metadata = {
        'engine': 'vllm',
        'model': 'GLM-5.3-Flash-NVFP4',
        'decode_mode': 'duration',
        'primary_decode_layer': 'sustained_decode',
        'request_count': 0,
        'warmup_request_count': 0,
        'run_burst': False,
        'standalone_prefill': False,
        'prefill_only': False,
        'skip_prefill': False,
        'max_tokens': 8192,
        'ignore_eos': True,
        'concurrency_levels': [1, 4, 8],
        'context_lengths': [0, 32768],
    }
    for field, expected in expected_metadata.items():
        _require(_same_json(metadata.get(field), expected),
                 f'{arm} benchmark metadata {field} does not match the baseline workload')
    _require(_same_number(metadata.get('duration_per_test'), 30.0),
             f'{arm} benchmark duration does not match the baseline workload')

    rows = receipt.get('results')
    _require(isinstance(rows, list) and len(rows) == len(EXPECTED),
             f'{arm} benchmark does not contain exactly six cells')
    rates = {}
    for row in rows:
        _require(isinstance(row, dict), f'{arm} benchmark contains a malformed cell')
        concurrency = row.get('concurrency')
        context = row.get('context_tokens')
        _require(type(concurrency) is int and type(context) is int,
                 f'{arm} benchmark cell identity is malformed')
        cell = (concurrency, context)
        _require(cell in EXPECTED and cell not in rates,
                 f'{arm} benchmark cell coverage is not the exact declared matrix')
        _require(row.get('benchmark_mode') == 'duration',
                 f'{arm} benchmark cell {cell} is not duration-mode')
        _require(_same_number(row.get('request_count_target'), 0),
                 f'{arm} benchmark cell {cell} is not a duration-mode execution')
        _require(_finite_number(row.get('measurement_seconds'))
                 and row['measurement_seconds'] >= 28,
                 f'{arm} benchmark cell {cell} has an invalid measured duration')
        _require(_finite_number(row.get('aggregate_tps')) and row['aggregate_tps'] > 0,
                 f'{arm} benchmark cell {cell} has no valid client throughput')
        _require(row.get('aggregate_source') == 'openai_continuous_usage',
                 f'{arm} benchmark cell {cell} does not use the expected client timing source')
        _require(type(row.get('num_errors')) is int and row['num_errors'] == 0,
                 f'{arm} benchmark cell {cell} recorded request errors')
        for field in ('underfilled', 'warmup_timed_out', 'capacity_limited'):
            _require(row.get(field) is False, f'{arm} benchmark cell {cell} failed {field}')
        rates[cell] = row['aggregate_tps']
    _require(set(rates) == EXPECTED, f'{arm} benchmark cell coverage is incomplete')
    return rates


def _validate_counter_summary(counter, arm, benchmark_path):
    _require(isinstance(counter, dict), f'{arm} counter summary is not an object')
    _require(counter.get('schema') == 'r26-steady-counters/v1'
             and counter.get('label') == arm, f'{arm} counter summary identity is invalid')
    _require_source_path(counter.get('source_benchmark'), benchmark_path,
                         f'{arm} counter summary source')
    all_windows_valid = counter.get('all_windows_valid')
    _require(type(all_windows_valid) is bool,
             f'{arm} counter-window disclosure is missing')
    cells = counter.get('cells')
    _require(isinstance(cells, list) and len(cells) == len(EXPECTED),
             f'{arm} counter-window coverage is incomplete')
    observed = set()
    computed_valid = True
    for row in cells:
        _require(isinstance(row, dict), f'{arm} counter summary contains a malformed cell')
        concurrency = row.get('concurrency')
        context = row.get('context_tokens')
        _require(type(concurrency) is int and type(context) is int,
                 f'{arm} counter cell identity is malformed')
        cell = (concurrency, context)
        _require(cell in EXPECTED and cell not in observed,
                 f'{arm} counter-window coverage is not the exact declared matrix')
        _require(type(row.get('valid')) is bool,
                 f'{arm} counter cell {cell} has no validity disclosure')
        observed.add(cell)
        computed_valid = computed_valid and row['valid']
    _require(observed == EXPECTED and computed_valid is all_windows_valid,
             f'{arm} aggregate counter-window disclosure is inconsistent')
    return all_windows_valid


def verify_baseline_reuse(baseline_root=BASELINE_ROOT, *, benchmark_path=None):
    baseline_root = Path(baseline_root).resolve()
    benchmark_path = Path(benchmark_path or rt.BENCH).resolve()
    sources = []
    try:
        benchmark_bytes = benchmark_path.read_bytes()
    except OSError as error:
        raise BaselineReuseError(f'Cannot verify benchmark program {benchmark_path}: {error}') from error
    benchmark_source = _record_source(
        sources, 'current_benchmark_program', benchmark_path, benchmark_bytes)

    plan = _load_json_source(
        baseline_root / 'experiment-plan.json', 'experiment_configuration', sources)
    expected_arms = [[arm, image] for arm, image, _ in BASELINE_ARMS]
    _require(isinstance(plan, dict), 'Baseline experiment configuration is not an object')
    _require(_same_json(plan.get('arms'), expected_arms),
             'Baseline arms are not the required ABBA sequence')
    _require(plan.get('model') == str(MODELS['published']),
             'Baseline model is not the published checkpoint')
    _require(type(plan.get('tp')) is int and plan['tp'] == 4
             and type(plan.get('dcp')) is int and plan['dcp'] == 1,
             'Baseline topology is not TP4/DCP1')
    _require(_same_json(plan.get('contexts'), [0, 32768])
             and _same_json(plan.get('concurrency'), [1, 4, 8]),
             'Baseline matrix does not contain the expected cells')
    _require(type(plan.get('duration_seconds')) is int and plan['duration_seconds'] == 30,
             'Baseline configured duration is not 30 seconds')
    _require(_same_json(plan.get('sampling'), BASELINE_SAMPLING),
             'Baseline sampling configuration does not match speculative matrix')
    _require(plan.get('benchmark_sha256') == benchmark_source['sha256'],
             'Baseline benchmark program identity is stale')

    summary = _load_json_source(
        baseline_root / 'runtime-baseline-summary.json', 'runtime_baseline_summary', sources)
    _require(isinstance(summary, dict) and summary.get('complete') is True
             and summary.get('passed') is True,
             'Runtime baseline summary is incomplete or failed')
    summary_arms = summary.get('arms')
    _require(isinstance(summary_arms, list) and len(summary_arms) == len(BASELINE_ARMS),
             'Runtime baseline summary does not contain all four arms')

    gates = _load_gate_source(baseline_root / 'gates.jsonl', sources)
    _require(all(isinstance(gate, dict) for gate in gates),
             'Baseline gate source contains a malformed record')
    required_gate_names = [
        name for arm, _, _ in BASELINE_ARMS
        for name in (f'boot:{arm}', f'benchmark-execution:{arm}', f'runtime-baseline:{arm}')
    ]
    required_name_set = set(required_gate_names)
    arm_gates = [gate for gate in gates if gate.get('name') in required_name_set]
    _require([gate.get('name') for gate in arm_gates] == required_gate_names,
             'Recorded source arm gates are missing, duplicated, or not in ABBA order')
    _require(all(_finite_number(gate.get('timestamp')) for gate in arm_gates)
             and all(left['timestamp'] < right['timestamp']
                     for left, right in zip(arm_gates, arm_gates[1:])),
             'Recorded source arm gate chronology is invalid')
    gates_by_name = {gate['name']: gate for gate in arm_gates}

    arm_rates = {}
    arm_receipts = []
    supplemental_counter_validity = {}
    for index, (arm, image, family) in enumerate(BASELINE_ARMS):
        summary_arm = summary_arms[index]
        _require(isinstance(summary_arm, dict) and summary_arm.get('arm') == arm
                 and summary_arm.get('image') == image and summary_arm.get('passed') is True
                 and summary_arm.get('issues') == [],
                 f'Runtime baseline summary arm {arm} is stale or failed')
        benchmark_file = (baseline_root / f'{arm}.json').resolve()
        launch_file = (baseline_root / f'{arm}.launch.json').resolve()
        command_file = (baseline_root / f'{arm}.bench.command.json').resolve()
        counter_file = (baseline_root / f'{arm}.steady-summary.json').resolve()
        _require_source_path(summary_arm.get('benchmark'), benchmark_file,
                             f'Runtime baseline summary benchmark for {arm}')
        _require_source_path(summary_arm.get('counter_summary'), counter_file,
                             f'Runtime baseline summary counter source for {arm}')
        _require(type(summary_arm.get('counter_windows_valid')) is bool,
                 f'Runtime baseline summary lacks counter disclosure for {arm}')

        boot_gate = gates_by_name[f'boot:{arm}']
        execution_gate = gates_by_name[f'benchmark-execution:{arm}']
        source_gate = gates_by_name[f'runtime-baseline:{arm}']
        for gate in (boot_gate, execution_gate, source_gate):
            _require(gate.get('passed') is True and isinstance(gate.get('detail'), dict),
                     f'Recorded source gate {gate.get("name")} failed')
        boot_detail = boot_gate['detail']
        _require(type(boot_detail.get('returncode')) is int
                 and boot_detail['returncode'] == 0 and boot_detail.get('image') == image,
                 f'Boot gate identity or execution failed for {arm}')
        _require_source_path(boot_detail.get('launch'), launch_file,
                             f'Boot gate launch source for {arm}')
        execution_detail = execution_gate['detail']
        _require(type(execution_detail.get('returncode')) is int
                 and execution_detail['returncode'] == 0
                 and execution_detail.get('concurrency') == '1,4,8'
                 and execution_detail.get('contexts') == '0,32k'
                 and type(execution_detail.get('duration_seconds')) is int
                 and execution_detail['duration_seconds'] == 30,
                 f'Benchmark execution gate configuration failed for {arm}')
        _require_source_path(execution_detail.get('result'), benchmark_file,
                             f'Benchmark execution result for {arm}')
        source_detail = source_gate['detail']
        _require(source_detail.get('arm') == arm and source_detail.get('image') == image
                 and source_detail.get('passed') is True and source_detail.get('issues') == [],
                 f'Runtime baseline source gate failed for {arm}')
        _require_source_path(source_detail.get('benchmark'), benchmark_file,
                             f'Runtime baseline source receipt for {arm}')
        _require_source_path(source_detail.get('counter_summary'), counter_file,
                             f'Runtime baseline source counter receipt for {arm}')
        _require(source_detail.get('counter_windows_valid')
                 is summary_arm['counter_windows_valid'],
                 f'Counter-window disclosures disagree for {arm}')

        launch = _load_json_source(launch_file, f'{arm}_launch_configuration', sources)
        _require(isinstance(launch, dict) and launch.get('label') == arm
                 and launch.get('image') == image,
                 f'Launch identity is invalid for {arm}')
        _require(type(launch.get('tp')) is int and launch['tp'] == 4
                 and type(launch.get('dcp')) is int and launch['dcp'] == 1
                 and launch.get('spec') == 'mtp0' and launch.get('cache') == 'vram'
                 and launch.get('kv') == 'fp8_ds_mla',
                 f'Launch topology or serving mode is invalid for {arm}')
        _require(_same_json(launch.get('env'), BASELINE_LAUNCH_ENV)
                 and _same_json(launch.get('extra_args'), BASELINE_EXTRA_ARGS),
                 f'Launch workload configuration is invalid for {arm}')
        _require(launch.get('gpus') == '0,1,2,3' and launch.get('l2_host') is None,
                 f'Launch GPU or cache ownership is invalid for {arm}')
        _require_source_path(launch.get('model_dir'), MODELS['published'],
                             f'Launch model for {arm}')

        command = _load_json_source(command_file, f'{arm}_benchmark_execution', sources)
        expected_command = [
            'python3', str(benchmark_path), '--port', '5002',
            '--model', 'GLM-5.3-Flash-NVFP4', '--concurrency', '1,4,8',
            '--contexts', '0,32k', '--duration', '30', '--max-tokens', '8192',
            '--output', str(benchmark_file),
        ]
        _require(isinstance(command, dict) and command.get('args') == expected_command
                 and type(command.get('returncode')) is int and command['returncode'] == 0,
                 f'Benchmark command receipt is stale or failed for {arm}')
        _require(_finite_number(command.get('started_at'))
                 and _finite_number(command.get('finished_at'))
                 and command['finished_at'] >= command['started_at']
                 and _finite_number(command.get('elapsed_seconds'))
                 and command['elapsed_seconds'] >= 0,
                 f'Benchmark command timing receipt is invalid for {arm}')

        benchmark = _load_json_source(benchmark_file, f'{arm}_raw_benchmark', sources)
        rates = _validate_benchmark_receipt(benchmark, arm)
        counter = _load_json_source(counter_file, f'{arm}_supplemental_counters', sources)
        counter_valid = _validate_counter_summary(counter, arm, benchmark_file)
        _require(counter_valid is summary_arm['counter_windows_valid'],
                 f'Counter-window source does not match summary for {arm}')
        arm_rates[arm] = rates
        supplemental_counter_validity[arm] = counter_valid
        arm_receipts.append({
            'arm': arm,
            'image': image,
            'image_family': family,
            'client_timing_valid': True,
            'supplemental_counter_windows_valid': counter_valid,
        })

    comparison = _load_json_source(
        baseline_root / 'paired-runtime-comparison.json', 'paired_runtime_comparison', sources)
    _require(isinstance(comparison, dict), 'Paired runtime comparison is not an object')
    comparison_rows = comparison.get('rows')
    _require(isinstance(comparison_rows, list) and len(comparison_rows) == len(EXPECTED),
             'Paired runtime comparison does not contain exactly six cells')
    indexed_comparison = {}
    for row in comparison_rows:
        _require(isinstance(row, dict), 'Paired runtime comparison contains a malformed row')
        concurrency = row.get('concurrency')
        context = row.get('context')
        _require(type(concurrency) is int and type(context) is int,
                 'Paired runtime comparison cell identity is malformed')
        cell = (concurrency, context)
        _require(cell in EXPECTED and cell not in indexed_comparison,
                 'Paired runtime comparison coverage is not the exact declared matrix')
        indexed_comparison[cell] = row
    _require(set(indexed_comparison) == EXPECTED,
             'Paired runtime comparison cell coverage is incomplete')

    repeat_arms = {
        'r281': ['r281-ab', 'r281-ba'],
        'r29': ['r29-ab', 'r29-ba'],
    }
    derived_comparison = []
    for concurrency, context in sorted(EXPECTED):
        cell = (concurrency, context)
        r281_repeats = [arm_rates[arm][cell] for arm in repeat_arms['r281']]
        r29_repeats = [arm_rates[arm][cell] for arm in repeat_arms['r29']]
        r281_mean = sum(r281_repeats) / 2
        r29_mean = sum(r29_repeats) / 2
        change_percent = (r29_mean / r281_mean - 1.0) * 100
        row = indexed_comparison[cell]
        for field, expected in (
            ('r281_repeats', r281_repeats),
            ('r29_repeats', r29_repeats),
        ):
            values = row.get(field)
            _require(isinstance(values, list) and len(values) == 2
                     and all(_same_number(value, expected_value)
                             for value, expected_value in zip(values, expected)),
                     f'Paired runtime comparison {field} is stale for cell {cell}')
        for field, expected in (
            ('r281_mean', r281_mean),
            ('r29_mean', r29_mean),
            ('change_percent', change_percent),
        ):
            _require(_same_number(row.get(field), expected),
                     f'Paired runtime comparison {field} is stale for cell {cell}')
        derived_comparison.append({
            'concurrency': concurrency,
            'context': context,
            'r281_repeats': r281_repeats,
            'r29_repeats': r29_repeats,
            'r281_mean': r281_mean,
            'r29_mean': r29_mean,
            'change_percent': change_percent,
        })

    comparison_source = next(
        source for source in sources if source['role'] == 'paired_runtime_comparison')
    return {
        'schema': 'r29-verified-baseline-reuse/v1',
        'client_timing_valid': True,
        'supplemental_counter_windows_valid': all(supplemental_counter_validity.values()),
        'supplemental_counter_windows_by_arm': supplemental_counter_validity,
        'comparison_source': comparison_source,
        'repeat_arms': repeat_arms,
        'arms': arm_receipts,
        'derived_comparison': derived_comparison,
        'sources': sources,
    }


def _checkpoint_inputs():
    draft = json.loads((ROOT / 'draft-verification.json').read_text())
    if not draft.get('passed'):
        raise RuntimeError('Draft artifact is not verified')
    rt.DRAFT = Path(draft['path'])
    for name in ('published', 'qad_step2500'):
        if not json.loads((ROOT / (name + '-verification.json')).read_text()).get('passed'):
            raise RuntimeError('Unverified checkpoint control')
    tvn_proof = Path('/home/josh/omp-workspace/drock-lmcache/new-drop-review-20260909T083122Z/tvn1500-readiness-result.json')
    if not json.loads(tvn_proof.read_text()).get('serving_smoke_passed'):
        raise RuntimeError('TVN checkpoint has not passed serving readiness')
    manifest = json.loads((ROOT / 'history-fixtures/manifest.json').read_text())
    selected = {}
    for size in (16384, 131072):
        entry = next(item for item in manifest['cases'] if item['kind'] == 'visible_length_control' and item['nominal_filler_tokens'] == size)
        encoded = (ROOT / 'history-fixtures' / entry['path']).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != entry['sha256']:
            raise RuntimeError('Sentinel fixture changed')
        selected[size] = json.loads(encoded)
    return draft, selected


def inputs():
    verify_baseline_reuse()
    return _checkpoint_inputs()


def sentinels(label, cases):
    rows = []
    for concurrency in (1, 4, 8):
        case = cases[131072 if concurrency == 8 else 16384]
        def one(index):
            messages = copy.deepcopy(case['messages'])
            observations = [{'id': f'client-{index}-check-{item:02}', 'ok': (item + index) % 3 != 0,
                             'weight': ((item + 1) * (index + 3)) % 13 + 1} for item in range(12)]
            messages[0]['content'] += f' This is independent client {index} at concurrency {concurrency}.'
            messages[-1]['content'] = json.dumps({'rows': observations}, separators=(',', ':'))
            expected = {'passed': sorted(row['id'] for row in observations if row['ok']),
                        'failed': sorted(row['id'] for row in observations if not row['ok']),
                        'score': sum(row['weight'] for row in observations if row['ok'])}
            body = {'model': rt.MODEL_NAME, 'messages': messages, 'max_tokens': 2048,
                'temperature': 0.0, 'top_p': 1.0, 'seed': 260913 + index,
                'cache_salt': f'r29-mode-sentinel-c{concurrency}-{index}',
                'chat_template_kwargs': {'reasoning_effort': 'high', 'clear_thinking': False}}
            call = legacy.post_json(rt.BASE_URL, '/v1/chat/completions', body, 900)
            name = f'{label}-sentinel-c{concurrency}-{index}'
            rt.save_json(name + '.json', call)
            message, error = legacy.response_message(call)
            record = {'concurrency': concurrency, 'index': index, 'expected': expected, 'passed': False, 'runtime_error': bool(error)}
            if error:
                record['error'] = error
                return record
            content, reasoning = legacy.message_text(message)
            choice = call['response']['choices'][0]
            record.update({'finish_reason': choice.get('finish_reason'), 'usage': call['response'].get('usage'),
                'visible_answer': content, 'reasoning_chars': len(reasoning)})
            try:
                record['passed'] = equal_json(parse_answer(content), expected) and choice.get('finish_reason') != 'length' and not message.get('tool_calls')
            except (ValueError, TypeError) as error:
                record['parse_error'] = str(error)
            return record
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows.extend(pool.map(one, range(concurrency)))
    report = {'passed': all(row['passed'] for row in rows), 'rows': rows,
        'scope': 'Deterministic T0/high-reasoning functional sentinels with client-specific input and expected output; C8 uses unique cache salts on a 128K nominal visible document. This is distinct from T1/max-reasoning throughput.'}
    rt.save_json(label + '-sentinels.json', report)
    rt.record_gate('mode-sentinels:' + label, report['passed'], report)
    return report


def phase():
    baseline_reuse = verify_baseline_reuse()
    draft, cases = _checkpoint_inputs()
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    rt.save_json('speculative-matrix-plan.json', {'models': {key: str(value) for key, value in MODELS.items()},
        'image': IMAGE, 'draft': draft, 'dcp': [1, 4], 'modes': ['mtp0', 'mtp3', 'dflash2'],
        'measured_repeats': 2, 'concurrency': [1, 4, 8], 'contexts': [0, 32768], 'duration_seconds': 30,
        'sampling': {'temperature': 1.0, 'top_p': .95, 'reasoning_effort': 'max', 'clear_thinking': False},
        'scope': 'Two warmed measured matrices within one boot per configuration, except published/no-spec/DCP1 reuses the earlier independent-boot ABBA timing evidence. Not all configurations have independent-boot repeats.'})
    progress = []
    for group_index, (dcp, spec) in enumerate(itertools.product((1, 4), ('mtp0', 'mtp3', 'dflash2'))):
        model_order = list(MODELS) if group_index % 2 == 0 else list(reversed(MODELS))
        for variant in model_order:
            label = f'{variant}-dcp{dcp}-{spec}'
            record = {'label': label, 'variant': variant, 'dcp': dcp, 'spec': spec, 'complete': False}
            try:
                if not rt.boot(label, image=IMAGE, model=MODELS[variant], tp=4, dcp=dcp, spec=spec, cache='vram',
                    extra_env={'MAX_NUM_SEQS': '32', 'MAX_CUDAGRAPH_CAPTURE_SIZE': '32', 'CUDAGRAPH_CAPTURE_SIZES': '1 2 4 8 16 32'},
                    extra_args=['--default-chat-template-kwargs', '{"reasoning_effort":"max","clear_thinking":false}',
                                '--override-generation-config', '{"temperature":1.0,"top_p":0.95}']):
                    raise RuntimeError('Mode did not boot')
                record['sentinels_passed'] = sentinels(label, cases)['passed']
                cleared = reset(rt.BASE_URL)
                rt.save_json(label + '-reset.json', cleared)
                if not cleared['passed']:
                    raise RuntimeError('Could not establish a clean benchmark start')
                if variant == 'published' and dcp == 1 and spec == 'mtp0':
                    record['timing_reused_from'] = baseline_reuse['comparison_source']['path']
                    record['timing_reuse'] = baseline_reuse
                    record['timing_valid'] = baseline_reuse['client_timing_valid']
                    record['counter_windows_valid'] = baseline_reuse['supplemental_counter_windows_valid']
                else:
                    trials = []
                    for trial in (1, 2):
                        name = label + '-trial' + str(trial)
                        with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{name}.steady.metrics.jsonl'):
                            ran = rt.bench(name, conc='1,4,8', contexts='0,32k', duration=30)
                        if not ran:
                            raise RuntimeError('Benchmark command failed')
                        result = json.loads((rt.ROOT / f'{name}.json').read_text())
                        measured = result.get('results', [])
                        coverage = {(row['concurrency'], row['context_tokens']) for row in measured}
                        valid = coverage == EXPECTED and len(measured) == len(EXPECTED) and all(
                            row.get('aggregate_tps', 0) > 0 and not row.get('num_errors') and not row.get('underfilled')
                            and not row.get('warmup_timed_out') and not row.get('capacity_limited')
                            and row.get('measurement_seconds', 0) >= 28 for row in measured)
                        counter = steady_metrics.summarize(rt.ROOT, name)
                        trials.append({'label': name, 'client_timing_valid': valid, 'counter_windows_valid': counter['all_windows_valid']})
                    record['trials'] = trials
                    record['timing_valid'] = all(row['client_timing_valid'] for row in trials)
                record['complete'] = True
                rt.record_gate('speculative-matrix:' + label, record['sentinels_passed'] and record['timing_valid'], record)
            except Exception as error:
                record['error'] = repr(error)
                rt.record_gate('speculative-matrix:' + label, False, record)
            finally:
                progress.append(record)
                rt.save_json('speculative-matrix-progress.json', progress)
                rt.stop()
    result = {'all_configurations_attempted': len(progress) == 18, 'configurations': progress,
        'passed': all(row['complete'] and row.get('sentinels_passed') and row.get('timing_valid') for row in progress)}
    rt.save_json('speculative-matrix-summary.json', result)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        inputs()
        print('R29 SPECULATIVE MATRIX: guarded coordinator starting', flush=True)
        coordinator.PHASES = [('speculative-matrix', str(Path(__file__).resolve()), ['--phase'], 86400)]
        coordinator.main()
    else:
        raise SystemExit('Usage: speculative_matrix.py [--phase]')
