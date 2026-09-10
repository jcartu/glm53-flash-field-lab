#!/usr/bin/env python3
"""Keep the full original QAD runbook, but wait for the R27 decision battery first."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from run_qad_battery import wait_for_process

HERE = Path(__file__).resolve().parent
R26_IMAGE = 'voipmonitor/vllm@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68'
PRODUCTION_CONTAINER = 'glm53-prod'
LIVE_VERIFICATION_SCOPE = (
    'live HTTP health and /v1/models verification only; '
    'no container restart and no GPU work'
)

def model_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        return []
    return sorted({
        row['id'] for row in payload['data']
        if isinstance(row, dict) and isinstance(row.get('id'), str)
    })


def load_identity_json(
        root: Path, identity: object, filename_prefix: str) -> tuple[dict | None, str | None]:
    if not isinstance(identity, dict):
        return None, 'receipt identity is not an object'
    value = identity.get('path')
    expected_hash = identity.get('sha256')
    if not isinstance(value, str) or not isinstance(expected_hash, str):
        return None, 'receipt identity lacks path or sha256'
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
    except OSError as error:
        return None, f'cannot resolve receipt: {error}'
    expected_name = (
        resolved.name == filename_prefix
        if filename_prefix.endswith('.json')
        else resolved.name.startswith(filename_prefix)
    )
    if path.is_symlink() or resolved.parent != root.resolve() or not expected_name:
        return None, 'receipt is not an expected direct child of the R27 root'
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        return None, f'cannot read receipt: {error}'
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        return None, 'receipt sha256 does not match completion provenance'
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f'invalid receipt JSON: {error}'
    if not isinstance(value, dict):
        return None, 'receipt JSON is not an object'
    return value, None


def valid_no_gpu_handoff(
        root: Path, complete: dict, restored: dict, wait_created: float) -> tuple[bool, list[str]]:
    """Allow a stale restoration time only after a fresh, explicit no-GPU verification."""
    issues = []
    if complete.get('continuation_scope') != 'no-gpu-finalization':
        issues.append('completion is not an explicit no-GPU finalization')
    if complete.get('gpu_jobs_started') is not False:
        issues.append('completion does not prove that zero GPU jobs started')
    if complete.get('remaining_phase_execution') != []:
        issues.append('completion records resumed phase execution')

    continuation_id = complete.get('continuation_id')
    if not isinstance(continuation_id, str):
        issues.append('completion lacks a continuation ID')
    plan, plan_error = load_identity_json(
        root, complete.get('continuation_execution_plan'),
        'continuation-execution-plan-',
    )
    if plan_error:
        issues.append(f'continuation plan: {plan_error}')
    else:
        verified_sources = plan.get('verified_sources')
        base_plan = plan.get('base_execution_plan')
        scripts = (
            base_plan.get('scripts') if isinstance(base_plan, dict) else None
        )
        script_hashes_recorded = (
            isinstance(scripts, dict)
            and bool(scripts)
            and all(
                isinstance(identity, dict)
                and identity.get('exists') is True
                and isinstance(identity.get('sha256'), str)
                for identity in scripts.values()
            )
        )
        finalizer_scripts = plan.get('report_finalizer_scripts')
        finalizer_hashes_recorded = (
            isinstance(finalizer_scripts, dict)
            and bool(finalizer_scripts)
            and all(
                isinstance(identity, dict)
                and identity.get('exists') is True
                and isinstance(identity.get('sha256'), str)
                for identity in finalizer_scripts.values()
            )
        )
        if (plan.get('schema') != 'r27-continuation-execution-plan/v1'
                or plan.get('continuation_id') != continuation_id
                or plan.get('gpu_phase_resume_planned') is not False
                or plan.get('remaining_phases') != []
                or not script_hashes_recorded
                or not finalizer_hashes_recorded
                or complete.get('report_finalizer_scripts')
                != plan.get('report_finalizer_scripts')
                or not isinstance(verified_sources, dict)
                or verified_sources.get('performed') is not False
                or verified_sources.get('required') is not False):
            issues.append('continuation plan is not a no-GPU plan')

    verification, verification_error = load_identity_json(
        root, complete.get('live_production_verification'),
        'production-live-verification-',
    )
    if verification_error:
        issues.append(f'live production verification: {verification_error}')
    else:
        verified_at = verification.get('verified_at')
        started_at = complete.get('started_at')
        finished_at = complete.get('finished_at')
        expected_models = verification.get('expected_model_ids')
        observed_models = verification.get('observed_model_ids')
        if (verification.get('schema') != 'r27-production-live-verification/v1'
                or verification.get('continuation_id') != continuation_id
                or verification.get('scope') != LIVE_VERIFICATION_SCOPE
                or verification.get('restart_performed') is not False
                or verification.get('verification_not_before') != started_at
                or verification.get('gpu_jobs_started') is not False
                or verification.get('healthy') is not True
                or verification.get('container') != PRODUCTION_CONTAINER
                or verification.get('container') != restored.get('container')):
            issues.append('live production verification scope or health fields are invalid')
        if (not isinstance(verified_at, (int, float)) or isinstance(verified_at, bool)
                or not isinstance(started_at, (int, float)) or isinstance(started_at, bool)
                or not isinstance(finished_at, (int, float)) or isinstance(finished_at, bool)
                or verified_at < max(wait_created, started_at)
                or verified_at > finished_at):
            issues.append('live production verification time is outside the continuation boundary')
        if (verification.get('original_restoration_timestamp') != restored.get('timestamp')
                or verification.get('original_restoration_receipt')
                != str(root / 'production-restored.json')):
            issues.append('live verification does not preserve the original restoration receipt')
        if (not isinstance(expected_models, list) or not expected_models
                or expected_models != observed_models
                or observed_models != model_ids(verification.get('models'))
                or expected_models != model_ids(restored.get('models'))):
            issues.append('live model inventory does not match the restored production inventory')

    handoff = complete.get('handoff_restoration')
    if not isinstance(handoff, dict):
        issues.append('completion lacks preserved handoff restoration provenance')
    else:
        handoff_receipt, handoff_error = load_identity_json(
            root, handoff.get('receipt'), 'production-restored.json'
        )
        if handoff_error:
            issues.append(f'handoff restoration: {handoff_error}')
        elif handoff_receipt != restored:
            issues.append('hashed handoff restoration does not match the live root receipt')
        if (handoff.get('timestamp') != restored.get('timestamp')
                or handoff.get('container') != restored.get('container')):
            issues.append('handoff restoration fields differ from the preserved receipt')
    return not issues, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-created', type=float, required=True)
    parser.add_argument('--r27-root', type=Path, required=True)
    parser.add_argument('--r26-root', type=Path, required=True)
    parser.add_argument('--r26-wait-pid', type=int, required=True)
    parser.add_argument('--r26-wait-created', type=float, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-tag', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true')
    actions.add_argument('--handoff-check-only', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, str(HERE / 'run_qad_battery.py'),
               '--wait-pid', str(args.r26_wait_pid), '--wait-created', str(args.r26_wait_created),
               '--upstream-root', str(args.r26_root), '--model-dir', str(args.model_dir),
               '--model-tag', args.model_tag, '--revision', args.revision,
               '--output-root', str(args.output_root)]
    if args.plan_only:
        print(json.dumps({'schema': 'qad-after-r27-plan/v1', 'original_full_runbook': command,
                          'checkpoint_comparison_image': R26_IMAGE,
                          'r27_completion_root': str(args.r27_root), 'gpu_jobs_started': False}, indent=2))
        return
    print('QAD BATTERY QUEUED: waiting for complete R27 decision battery and restoration', flush=True)
    wait_for_process(args.wait_pid, args.wait_created)
    r27_root = args.r27_root.resolve()
    complete = json.loads((r27_root / 'r27-qualification-completed.json').read_text())
    restored = json.loads((r27_root / 'production-restored.json').read_text())
    if not isinstance(complete, dict) or not isinstance(restored, dict):
        raise RuntimeError('R27 completion/restoration receipt is invalid; no QAD GPU launch')
    finished_at = complete.get('finished_at')
    restored_at = restored.get('timestamp')
    base_complete = (
        complete.get('all_phases_attempted') is True
        and complete.get('all_phase_reports_final') is True
        and complete.get('production_healthy') is True
        and isinstance(finished_at, (int, float))
        and not isinstance(finished_at, bool)
        and finished_at >= args.wait_created
        and restored.get('healthy') is True
    )
    restoration_is_fresh = (
        isinstance(restored_at, (int, float))
        and not isinstance(restored_at, bool)
        and restored_at >= args.wait_created
    )
    no_gpu_handoff_valid = False
    no_gpu_issues: list[str] = []
    if base_complete and not restoration_is_fresh:
        no_gpu_handoff_valid, no_gpu_issues = valid_no_gpu_handoff(
            r27_root, complete, restored, args.wait_created
        )
    if not base_complete or not (restoration_is_fresh or no_gpu_handoff_valid):
        detail = '; '.join(no_gpu_issues) if no_gpu_issues else 'stale or incomplete receipts'
        raise RuntimeError(
            f'R27 completion/restoration is incomplete ({detail}); no QAD GPU launch'
        )
    if args.handoff_check_only:
        print(json.dumps({
            'schema': 'qad-after-r27-handoff-check/v1',
            'r27_completion_root': str(r27_root),
            'handoff_mode': (
                'fresh-restoration' if restoration_is_fresh
                else 'fresh-no-gpu-live-verification'
            ),
            'completion_finished_at': finished_at,
            'restoration_timestamp_preserved': restored_at,
            'gpu_jobs_started': False,
        }, indent=2))
        return
    # Runtime and checkpoint changes stay separate. Preserve the complete R26
    # runbook and its R25/overlay controls rather than silently changing images.
    os.environ.update({'BATTERY_IMAGE': R26_IMAGE, 'BATTERY_CONTAINER': 'r26-test',
                       'BATTERY_PORT': '5002'})
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
