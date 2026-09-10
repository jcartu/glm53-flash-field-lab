#!/usr/bin/env python3
"""Reuse final R27 evidence and execute only phases whose canonical reports are non-final."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from run_qad_battery import wait_for_process
import run_r27_battery as battery

HERE = Path(__file__).resolve().parent
PHASES = battery.PHASES
STOCK = battery.STOCK
REPORT_FINALIZERS = {
    'r27-scheduler-controls': ('r27_scheduler_phase.py', ['--reanalyze-existing'], 1800),
}

LOG_MTIME_SLACK_SECONDS = 5.0
PRODUCTION_CONTAINER = 'glm53-prod'
PRODUCTION_BASE_URL = 'http://127.0.0.1:5001'
LIVE_VERIFICATION_SCOPE = (
    'live HTTP health and /v1/models verification only; '
    'no container restart and no GPU work'
)


def read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'Invalid preserved R27 evidence: {path}') from error

def file_identity(path: Path) -> dict:
    path = path.resolve()
    return {
        'path': str(path),
        'exists': path.is_file(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
    }


def write_new_json(path: Path, value: object) -> dict:
    """Persist an immutable run-specific receipt and return its identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite continuation receipt: {path}')
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise RuntimeError(f'Stale temporary continuation receipt: {temporary}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
    return file_identity(path)


def root_artifact_reference(root: Path, value: object) -> dict:
    """Describe a referenced artifact without following references outside the evidence root."""
    result = {'path': str(value) if isinstance(value, str) else None,
              'inside_root': False, 'exists': False}
    if not isinstance(value, str):
        return result
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return result
    result.update({'path': str(resolved), 'inside_root': True, 'exists': resolved.is_file()})
    if resolved.is_file():
        stat = resolved.stat()
        result.update({
            'mtime': stat.st_mtime,
            'size_bytes': stat.st_size,
            'sha256': hashlib.sha256(resolved.read_bytes()).hexdigest(),
        })
    return result


def referenced_log_integrity(root: Path, command_path: Path, command: dict) -> dict:
    finished_at = command.get('finished_at')
    reference = root_artifact_reference(root, command.get('log'))
    issues = []
    if not reference['inside_root']:
        issues.append('referenced_log_is_outside_evidence_root')
    elif not reference['exists']:
        issues.append('referenced_log_is_missing')
    if isinstance(finished_at, (int, float)) and reference.get('mtime') is not None:
        if reference['mtime'] > float(finished_at) + LOG_MTIME_SLACK_SECONDS:
            issues.append('referenced_log_is_newer_than_command_completion')
    elif not isinstance(finished_at, (int, float)):
        issues.append('command_finished_at_is_missing')
    return {
        'command_receipt': str(command_path),
        'referenced_log': reference,
        'command_finished_at': finished_at,
        'mtime_slack_seconds': LOG_MTIME_SLACK_SECONDS,
        'issues': issues,
        'conflict': bool(issues),
        'metadata_caveat': (
            'Filesystem mtime chronology is diagnostic because copying can change mtimes; '
            'a conflict is reported only beyond the recorded slack and does not invalidate '
            'intact canonical measurement receipts.'
        ),
    }


def recorded_invocations(root: Path) -> list[dict]:
    """Recover actual phase return codes without inferring them from report contents."""
    phase_names = {phase for phase, *_rest in PHASES}
    candidates: dict[str, tuple[float, dict]] = {}

    def remember(rows: object, source: Path, field: str, recorded_at: float) -> None:
        if rows is None:
            return
        if not isinstance(rows, list):
            raise RuntimeError(f'Invalid phase invocation ledger in {source}')
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f'Invalid phase invocation row in {source}')
            phase = row.get('phase')
            if phase not in phase_names:
                continue
            code = row.get('returncode')
            if isinstance(code, bool) or not isinstance(code, int):
                raise RuntimeError(f'Invalid return code for {phase} in {source}')
            recovered = {'phase': phase, 'returncode': code,
                         'recovered_from': f'{source}:{field}'}
            if phase not in candidates or recorded_at >= candidates[phase][0]:
                candidates[phase] = (recorded_at, recovered)

    ledgers = (
        ('qualification-executed.json', 'phases', 'finished_at'),
        ('qualification-interrupted.json', 'completed_phases', 'timestamp'),
    )
    for filename, field, time_field in ledgers:
        path = root / filename
        report = read_json(path)
        if report is None:
            continue
        if not isinstance(report, dict):
            raise RuntimeError(f'Invalid preserved R27 evidence: {path}')
        observed_time = report.get(time_field)
        recorded_at = (float(observed_time) if isinstance(observed_time, (int, float))
                       else path.stat().st_mtime)
        remember(report.get(field), path, field, recorded_at)

    progress_path = root / 'phase-progress.json'
    progress = read_json(progress_path)
    if progress is not None:
        remember(progress, progress_path, 'root', progress_path.stat().st_mtime)

    completion_path = root / 'r27-qualification-completed.json'
    completion = read_json(completion_path)
    if completion is not None:
        if not isinstance(completion, dict):
            raise RuntimeError(f'Invalid preserved R27 evidence: {completion_path}')
        observed_time = completion.get('finished_at')
        recorded_at = (float(observed_time) if isinstance(observed_time, (int, float))
                       else completion_path.stat().st_mtime)
        for field in ('phase_execution', 'preserved_completed_phases',
                      'remaining_phase_execution', 'recorded_phase_execution'):
            remember(completion.get(field), completion_path, field, recorded_at)

    for phase, script, arguments, _timeout in PHASES:
        path = root / f'phase-{phase}.command.json'
        command = read_json(path)
        if command is None:
            continue
        if not isinstance(command, dict):
            raise RuntimeError(f'Invalid preserved R27 evidence: {path}')
        argv = command.get('args')
        expected = [script, *arguments]
        if (not isinstance(argv, list) or len(argv) < 2
                or not isinstance(argv[1], str)
                or [Path(argv[1]).name, *argv[2:]] != expected):
            raise RuntimeError(f'Phase invocation receipt does not match the R27 plan: {path}')
        code = command.get('returncode')
        if isinstance(code, bool) or not isinstance(code, int):
            raise RuntimeError(f'Invalid return code for {phase} in {path}')
        finished_at = command.get('finished_at')
        recorded_at = (float(finished_at) if isinstance(finished_at, (int, float))
                       and not isinstance(finished_at, bool) else path.stat().st_mtime)
        candidates[phase] = (
            recorded_at,
            {
                'phase': phase,
                'returncode': code,
                'invocation_receipt': str(path),
                'invocation_log_integrity': referenced_log_integrity(root, path, command),
                'recovered_from': f'{path}:returncode',
            },
        )

    return [candidates[phase][1] for phase, *_rest in PHASES if phase in candidates]

def cache_receipt_integrity(root: Path) -> dict:
    """Separate overwritten launcher metadata from intact cache measurements."""
    summary_path = root / 'r27-cache-summary.json'
    summary = read_json(summary_path)
    phase_command_path = root / 'phase-r27-cache-lifecycle.command.json'
    phase_command = read_json(phase_command_path)
    if not isinstance(summary, dict) or not isinstance(summary.get('lifecycles'), list):
        return {
            'status': 'cache_summary_unavailable',
            'summary': root_artifact_reference(root, str(summary_path)),
            'conflicts': [],
        }
    phase_finished_at = (
        phase_command.get('finished_at') if isinstance(phase_command, dict) else None
    )
    conflicts = []
    for row in summary['lifecycles']:
        if not isinstance(row, dict) or not isinstance(row.get('label'), str):
            continue
        label = row['label']
        launch_path = root / f'{label}.launch.json'
        boot_path = root / f'{label}.boot.command.json'
        launch = read_json(launch_path)
        boot_receipt = read_json(boot_path)
        issues = []
        configuration_hosts = set()
        configuration_rows = {}
        configurations = row.get('configurations')
        if isinstance(configurations, dict):
            for stage, embedded in configurations.items():
                if not isinstance(embedded, dict):
                    continue
                reference = root_artifact_reference(root, embedded.get('receipt'))
                config = (
                    read_json(Path(reference['path']))
                    if reference['inside_root'] and reference['exists'] else None
                )
                host = (
                    config.get('expected', {}).get('l2_host')
                    if isinstance(config, dict)
                    and isinstance(config.get('expected'), dict) else None
                )
                if isinstance(host, str):
                    configuration_hosts.add(host)
                configuration_rows[stage] = {
                    'receipt': reference,
                    'recorded_returncode': embedded.get('returncode'),
                    'recorded_passed': embedded.get('passed'),
                    'recorded_l2_host': host,
                }

        launch_host = None
        launch_instance = None
        if isinstance(launch, dict):
            launch_host = launch.get('l2_host')
            launch_env = launch.get('env')
            if isinstance(launch_env, dict):
                launch_instance = launch_env.get('LMCACHE_INSTANCE_ID')
            if configuration_hosts and launch_host not in configuration_hosts:
                issues.append('launch_l2_host_conflicts_with_intact_config_receipts')
            prompt = row.get('prompt')
            measured_identity = prompt.get('identity') if isinstance(prompt, dict) else None
            if (isinstance(measured_identity, str) and isinstance(launch_instance, str)
                    and launch_instance != measured_identity):
                issues.append('launch_instance_conflicts_with_measured_prompt_identity')
        elif launch is not None:
            issues.append('launch_receipt_is_not_an_object')

        if isinstance(boot_receipt, dict):
            boot_started_at = boot_receipt.get('started_at')
            if (isinstance(boot_started_at, (int, float))
                    and not isinstance(boot_started_at, bool)
                    and isinstance(phase_finished_at, (int, float))
                    and not isinstance(phase_finished_at, bool)
                    and boot_started_at > phase_finished_at + LOG_MTIME_SLACK_SECONDS):
                issues.append('boot_receipt_started_after_recorded_phase_finished')
            boot_args = boot_receipt.get('args')
            if isinstance(boot_args, list) and isinstance(launch_host, str):
                if f'{launch_host}:/lmcache-l2' not in boot_args:
                    issues.append('boot_receipt_does_not_reference_current_launch_l2_host')
                if (isinstance(launch_instance, str)
                        and f'LMCACHE_INSTANCE_ID={launch_instance}' not in boot_args):
                    issues.append('boot_receipt_does_not_reference_current_launch_instance')
        elif boot_receipt is not None:
            issues.append('boot_receipt_is_not_an_object')

        if not issues:
            continue
        request_rows = {}
        requests = row.get('requests')
        if isinstance(requests, dict):
            for stage, embedded in requests.items():
                if not isinstance(embedded, dict):
                    continue
                request_rows[stage] = {
                    'receipt': root_artifact_reference(root, embedded.get('receipt')),
                    'recorded_returncode': embedded.get('returncode'),
                    'recorded_checks': embedded.get('checks'),
                    'recorded_answer_evidence': embedded.get('answer_evidence'),
                }
        conflicts.append({
            'label': label,
            'issues': issues,
            'conflicting_receipts': {
                'launch': root_artifact_reference(root, str(launch_path)),
                'boot_command': root_artifact_reference(root, str(boot_path)),
                'launch_l2_host': launch_host,
                'launch_instance_id': launch_instance,
                'boot_started_at': (
                    boot_receipt.get('started_at') if isinstance(boot_receipt, dict) else None
                ),
                'recorded_phase_finished_at': phase_finished_at,
            },
            'intact_measurement_cross_references': {
                'canonical_summary': root_artifact_reference(root, str(summary_path)),
                'completed_cleanup': root_artifact_reference(
                    root, str(root / f'{label}-cleanup.json')
                ),
                'configurations': configuration_rows,
                'requests': request_rows,
                'recorded_measurement_passed': row.get('passed'),
                'recorded_measurement_failure_preserved': row.get('passed') is False,
            },
            'scope': (
                'Launcher/boot provenance conflicts are annotated separately. '
                'The embedded summary, referenced configuration snapshots, and per-request '
                'measurements retain their recorded outcomes and are not rewritten or invalidated.'
            ),
        })
    return {
        'status': 'conflicts_detected' if conflicts else 'no_conflicts_detected',
        'summary': root_artifact_reference(root, str(summary_path)),
        'phase_command': root_artifact_reference(root, str(phase_command_path)),
        'conflicts': conflicts,
    }


def evidence_integrity(root: Path, invocations: list[dict]) -> dict:
    invocation_log_conflicts = [
        {
            'phase': row['phase'],
            'integrity': row['invocation_log_integrity'],
        }
        for row in invocations
        if row.get('invocation_log_integrity', {}).get('conflict') is True
    ]
    cache_integrity = cache_receipt_integrity(root)
    cache_conflicts = cache_integrity.get('conflicts', [])
    return {
        'schema': 'r27-continuation-evidence-integrity/v1',
        'status': (
            'conflicts_detected'
            if invocation_log_conflicts or cache_conflicts else 'no_conflicts_detected'
        ),
        'invocation_log_conflicts': invocation_log_conflicts,
        'cache_launcher_conflicts': cache_integrity,
        'measurement_policy': (
            'Receipt conflicts are provenance findings, not permission to repair raw artifacts, '
            'guess missing originals, or change completed native measurement outcomes.'
        ),
    }


def continuation_plan(root: Path) -> dict:
    root = root.resolve()
    invocations = recorded_invocations(root)
    finality = battery.phase_report_finality(root, {'phases': invocations})
    integrity = evidence_integrity(root, invocations)
    by_phase = {row['phase']: row for row in invocations}
    completed = []
    finalizers = []
    remaining = []
    for phase_entry in PHASES:
        phase = phase_entry[0]
        check = finality['phases'][phase]
        if not check['final']:
            if phase in REPORT_FINALIZERS:
                script, arguments, timeout = REPORT_FINALIZERS[phase]
                finalizers.append((phase, script, arguments, timeout))
            else:
                remaining.append(phase_entry)
            continue
        invocation = by_phase[phase]
        completed.append({
            'phase': phase,
            'returncode': invocation['returncode'],
            'receipt': check['report'],
            'schema': check['schema'],
            'invocation_receipt': invocation.get('invocation_receipt'),
            'invocation_recovered_from': invocation['recovered_from'],
            'invocation_log_integrity': invocation.get('invocation_log_integrity'),
            'preserved_interrupted_evidence': True,
        })
    return {
        'schema': 'r27-continuation-plan/v1',
        'root': str(root),
        'recorded_invocations': invocations,
        'completed_receipts': completed,
        'remaining_phases': remaining,
        'report_finalizers': finalizers,
        'all_phase_reports_final': finality['all_final'],
        'phase_report_finality': finality,
        'evidence_integrity': integrity,
        'note': ('Return codes come from recorded invocations, not report verdicts. '
                 'Measured failures remain terminal findings when the canonical phase contract is final; '
                 'scheduler summary repair reanalyzes preserved evidence before any GPU phase is considered.'),
        'gpu_jobs_started': False,
    }


def completed_receipts(root: Path) -> list[dict]:
    return continuation_plan(root)['completed_receipts']

def coordinator_batches(phases: list[tuple]) -> list[list[tuple]]:
    """Place a restoration/check boundary after cache work before any later GPU phase."""
    if not phases:
        return []
    cache_index = next(
        (index for index, row in enumerate(phases) if row[0] == 'r27-cache-lifecycle'),
        None,
    )
    if cache_index is None or cache_index == len(phases) - 1:
        return [phases]
    return [phases[:cache_index + 1], phases[cache_index + 1:]]


def build_continuation_execution_plan(
        root: Path, state: dict, continuation_id: str, planned_at: float) -> dict:
    base_plan = battery.execution_plan()
    if not all(row.get('exists') for row in base_plan.get('scripts', {}).values()):
        raise RuntimeError('A planned R27 continuation module is missing')
    finalizer_scripts = {}
    for phase, (script, _arguments, _timeout) in REPORT_FINALIZERS.items():
        identity = file_identity(HERE / script)
        if not identity['exists']:
            raise RuntimeError(f'Report finalizer is missing: {script}')
        finalizer_scripts[phase] = identity
    source_verification = {
        'required': bool(state['remaining_phases']),
        'performed': False,
        'verified_at': None,
        'manifest': None,
        'scope': 'No source verification is substituted for a resumed GPU phase.',
    }
    if state['remaining_phases']:
        verified_manifest = battery.verify_sources()
        source_verification.update({
            'performed': True,
            'verified_at': time.time(),
            'manifest': verified_manifest,
        })
    return {
        'schema': 'r27-continuation-execution-plan/v1',
        'continuation_id': continuation_id,
        'planned_at': planned_at,
        'root': str(root.resolve()),
        'gpu_phase_resume_planned': bool(state['remaining_phases']),
        'remaining_phases': state['remaining_phases'],
        'coordinator_batches': coordinator_batches(state['remaining_phases']),
        'report_finalizers': state['report_finalizers'],
        'base_execution_plan': base_plan,
        'continuation_runner': file_identity(Path(__file__)),
        'report_finalizer_scripts': finalizer_scripts,
        'downstream_qad_wrapper': file_identity(HERE / 'run_qad_after_r27.py'),
        'verified_sources': source_verification,
        'preserved_original_execution_plan': file_identity(root / 'execution-plan.json'),
        'preserved_source_manifest': file_identity(root / 'source-manifest.json'),
        'policy': (
            'This continuation-specific plan does not replace execution-plan.json. '
            'Every resumed GPU phase requires verified pinned sources and the recorded script hashes.'
        ),
    }


def run_report_finalizer(
        root: Path, entry: tuple, expected_script: dict, continuation_id: str) -> dict:
    phase, script, arguments, timeout = entry
    observed_script = file_identity(HERE / script)
    if observed_script != expected_script:
        raise RuntimeError(f'Report finalizer changed after continuation planning: {script}')
    command = [sys.executable, str(HERE / script), *arguments]
    label = f'report-finalizer-{phase}-{continuation_id}'
    log_path = root / f'{label}.log'
    receipt_path = root / f'{label}.command.json'
    if log_path.exists() or receipt_path.exists():
        raise RuntimeError(f'Refusing to overwrite prior report-finalizer evidence: {label}')
    started = time.time()
    env = {
        **os.environ,
        'BATTERY_ROOT': str(root),
        'BATTERY_CONTAINER': 'r27-test',
        'BATTERY_PORT': '5002',
    }
    execution_error = None
    try:
        with log_path.open('x') as output:
            result = subprocess.run(
                command, cwd=HERE, env=env, stdout=output, stderr=subprocess.STDOUT,
                check=False, timeout=timeout,
            )
        returncode = result.returncode
    except subprocess.TimeoutExpired as error:
        returncode = 124
        execution_error = f'{type(error).__name__}: {error}'
    except Exception as error:
        returncode = 127
        execution_error = f'{type(error).__name__}: {error}'
    finished = time.time()
    receipt = {
        'phase': phase,
        'args': command,
        'script': observed_script,
        'returncode': returncode,
        'started_at': started,
        'finished_at': finished,
        'elapsed_seconds': finished - started,
        'log': str(log_path),
        'execution_error': execution_error,
        'execution_scope': 'derived report reanalysis from preserved evidence; no GPU work',
    }
    receipt_identity = write_new_json(receipt_path, receipt)
    return {
        'phase': phase,
        'returncode': returncode,
        'script': observed_script,
        'receipt': receipt_identity,
        'log': root_artifact_reference(root, str(log_path)),
    }


def model_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        return []
    return sorted({
        row['id'] for row in payload['data']
        if isinstance(row, dict) and isinstance(row.get('id'), str)
    })


def record_cache_orphan_cleanup_plan(root: Path, continuation_id: str) -> dict:
    """Record exact receipt-proven cleanup candidates without deleting anything."""
    import r27_cache_phase as cache_recovery

    details = cache_recovery.plan()
    summary = cache_recovery.resume_summary(root, details)
    cleanup_plan = cache_recovery.orphan_cleanup_plan(
        root, details, summary, continuation_id
    )
    cleanup_plan.update({
        'planned_by': str(Path(__file__).resolve()),
        'filesystem_deletion_executed': False,
        'execution_authorized': False,
    })
    path = root / f'r27-cache-orphan-cleanup-plan-{continuation_id}.json'
    identity = write_new_json(path, cleanup_plan)
    return {
        'receipt': identity,
        'candidate_count': len(cleanup_plan['candidates']),
        'safe_to_execute': cleanup_plan['safe_to_execute'],
        'filesystem_deletion_executed': False,
    }


def verify_live_production(
        root: Path, restored: dict, continuation_id: str, verification_not_before: float) -> dict:
    """Verify live production without changing its preserved restoration timestamp."""
    receipt_path = root / f'production-live-verification-{continuation_id}.json'
    expected_models = model_ids(restored.get('models'))
    receipt = {
        'schema': 'r27-production-live-verification/v1',
        'continuation_id': continuation_id,
        'verification_started_at': time.time(),
        'verified_at': None,
        'verification_not_before': verification_not_before,
        'scope': LIVE_VERIFICATION_SCOPE,
        'container': restored.get('container'),
        'original_restoration_receipt': str(root / 'production-restored.json'),
        'original_restoration_timestamp': restored.get('timestamp'),
        'restart_performed': False,
        'gpu_jobs_started': False,
        'health_status': None,
        'models_status': None,
        'expected_model_ids': expected_models,
        'observed_model_ids': [],
        'models': None,
        'healthy': False,
        'error': None,
    }
    try:
        with requests.Session() as session:
            session.trust_env = False
            health = session.get(PRODUCTION_BASE_URL + '/health', timeout=5)
            receipt['health_status'] = health.status_code
            models = session.get(PRODUCTION_BASE_URL + '/v1/models', timeout=10)
            receipt['models_status'] = models.status_code
            models.raise_for_status()
            payload = models.json()
            receipt['models'] = payload
            receipt['observed_model_ids'] = model_ids(payload)
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    receipt['verified_at'] = time.time()
    receipt['healthy'] = (
        receipt['container'] == PRODUCTION_CONTAINER
        and receipt['health_status'] == 200
        and receipt['models_status'] == 200
        and bool(expected_models)
        and receipt['observed_model_ids'] == expected_models
        and receipt['verified_at'] >= verification_not_before
    )
    identity = write_new_json(receipt_path, receipt)
    if not receipt['healthy']:
        raise RuntimeError(
            f'Fresh no-restart production verification failed; full QAD remains gated: {receipt_path}'
        )
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-created', type=float, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.plan_only:
        print(json.dumps(continuation_plan(args.root), indent=2))
        return

    print('R27 CONTINUATION QUEUED: waiting for the preceding R27 work to finish', flush=True)
    wait_for_process(args.wait_pid, args.wait_created)
    started = time.time()
    continuation_id = (
        time.strftime('%Y%m%dT%H%M%S', time.gmtime(started)) + f'-{os.getpid()}'
    )
    state = continuation_plan(args.root)
    preserved = state['completed_receipts']
    restored_path = args.root / 'production-restored.json'
    restored = read_json(restored_path)
    if not isinstance(restored, dict) or restored.get('healthy') is not True:
        raise RuntimeError('R27 evidence does not record healthy restored production')
    handoff_restoration = {
        'receipt': file_identity(restored_path),
        'timestamp': restored.get('timestamp'),
        'container': restored.get('container'),
    }
    if (state['remaining_phases']
            and (not isinstance(restored.get('timestamp'), (int, float))
                 or restored['timestamp'] < args.wait_created)):
        raise RuntimeError('Preceding R27 GPU work did not restore production')

    execution_plan = build_continuation_execution_plan(
        args.root, state, continuation_id, started
    )
    plan_path = args.root / f'continuation-execution-plan-{continuation_id}.json'
    execution_plan_identity = write_new_json(plan_path, execution_plan)
    finalized_now = [
        run_report_finalizer(
            args.root,
            entry,
            execution_plan['report_finalizer_scripts'][entry[0]],
            continuation_id,
        )
        for entry in state['report_finalizers']
    ]
    state = continuation_plan(args.root)
    executed_now = []
    gpu_jobs_started = False
    if state['remaining_phases']:
        verified_sources = execution_plan['verified_sources']
        if (execution_plan.get('gpu_phase_resume_planned') is not True
                or verified_sources.get('performed') is not True
                or not isinstance(verified_sources.get('manifest'), dict)):
            raise RuntimeError('Continuation source verification is absent; no resumed GPU phase')
        os.environ.update({
            'BATTERY_ROOT': str(args.root), 'BATTERY_CONTAINER': 'r27-test',
            'BATTERY_PORT': '5002', 'BATTERY_IMAGE': STOCK,
            'BATTERY_MODEL_DIR': '/mnt/2king/models/GLM-5.3-Flash-NVFP4',
            'BATTERY_MODEL_CACHE_TAG': 'r27-published-target',
            'BATTERY_L2_SHARED': '/mnt/2king/lmcache-r26-battery/cache-phase-r27-shared',
        })
        import run_qualification as coordinator
        planned_batches = execution_plan['coordinator_batches']
        if ([row[0] for batch in planned_batches for row in batch]
                != [row[0] for row in state['remaining_phases']]):
            raise RuntimeError('Continuation coordinator batches changed after planning')
        gpu_jobs_started = True
        execution_path = args.root / 'qualification-executed.json'
        for batch_index, batch in enumerate(planned_batches):
            coordinator.PHASES = batch
            coordinator.main()
            execution = read_json(execution_path)
            expected = [row[0] for row in batch]
            if (not isinstance(execution, dict)
                    or execution.get('all_phases_attempted') is not True
                    or not isinstance(execution.get('phases'), list)
                    or [row.get('phase') for row in execution['phases']] != expected):
                raise RuntimeError(
                    f'Continuation execution receipt is incomplete: {execution_path}'
                )
            batch_receipt = write_new_json(
                args.root / f'continuation-phase-batch-{continuation_id}-{batch_index}.json',
                {
                    'schema': 'r27-continuation-phase-batch/v1',
                    'continuation_id': continuation_id,
                    'batch_index': batch_index,
                    'planned_phases': batch,
                    'coordinator_execution_receipt': file_identity(execution_path),
                    'phase_execution': execution['phases'],
                    'finished_at': time.time(),
                },
            )
            executed_now.extend([
                {**row, 'continuation_batch_receipt': batch_receipt}
                for row in execution['phases']
            ])
            cache_rows = [
                row for row in execution['phases']
                if row.get('phase') == 'r27-cache-lifecycle'
            ]
            if cache_rows and cache_rows[0].get('returncode') != 0:
                raise RuntimeError(
                    'R27 cache recovery exited nonzero after its recorded teardown; '
                    'no later GPU batch will start'
                )

    final_state = continuation_plan(args.root)
    finality = final_state['phase_report_finality']
    orphan_cleanup = record_cache_orphan_cleanup_plan(args.root, continuation_id)
    restored_after = read_json(restored_path)
    live_verification = None
    if gpu_jobs_started:
        production_healthy = (
            isinstance(restored_after, dict)
            and restored_after.get('healthy') is True
            and isinstance(restored_after.get('timestamp'), (int, float))
            and restored_after['timestamp'] >= started
        )
        continuation_scope = 'resumed-gpu-phases'
    else:
        if not isinstance(restored_after, dict):
            raise RuntimeError('Original production restoration receipt became unreadable')
        if file_identity(restored_path) != handoff_restoration['receipt']:
            raise RuntimeError(
                'No-GPU continuation observed a changed restoration receipt; full QAD remains gated'
            )
        live_verification = verify_live_production(
            args.root, restored_after, continuation_id, started
        )
        production_healthy = True
        continuation_scope = 'no-gpu-finalization'
    recorded = final_state['recorded_invocations']
    receipt = {
        'schema': 'r27-qualification-completed/v1',
        'continuation_id': continuation_id,
        'continuation_scope': continuation_scope,
        'started_at': started,
        'finished_at': time.time(),
        'all_phases_attempted': len(recorded) == len(PHASES),
        'preserved_completed_phases': preserved,
        'report_finalization_execution': finalized_now,
        'report_finalizer_scripts': execution_plan['report_finalizer_scripts'],
        'remaining_phase_execution': executed_now,
        'gpu_jobs_started': gpu_jobs_started,
        'recorded_phase_execution': recorded,
        'all_phase_reports_final': finality['all_final'],
        'phase_report_finality': finality,
        'pending_report_finalizers': final_state['report_finalizers'],
        'evidence_integrity': final_state['evidence_integrity'],
        'continuation_execution_plan': execution_plan_identity,
        'cache_orphan_cleanup_plan': orphan_cleanup,
        'handoff_restoration': handoff_restoration,
        'resulting_restoration': (
            file_identity(restored_path) if isinstance(restored_after, dict) else None
        ),
        'live_production_verification': live_verification,
        'production_healthy': production_healthy,
        'qualification_is_not_promotion': True,
        'note': (
            'Completion preserves measured failures. Provenance conflicts are annotations only; '
            'no raw launch, log, request, configuration, or original plan receipt is repaired.'
        ),
    }
    temporary = args.root / 'r27-qualification-completed.json.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(args.root / 'r27-qualification-completed.json')
    if not finality['all_final']:
        raise SystemExit('R27 continuation evidence is incomplete; full QAD remains gated')
    if not production_healthy:
        raise SystemExit('R27 continuation did not leave verified healthy production; full QAD remains gated')
    print('R27 BATTERY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
