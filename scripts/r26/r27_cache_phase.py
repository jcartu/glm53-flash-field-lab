#!/usr/bin/env python3
"""Requalify R27 cache tiers and the previously failing native boot point."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cache_phase as cache
import runtime as rt
from r27_config import IMAGES, boot, ensure_scope

CONFIG_STAGES = ('before', 'after-apc', 'before-restart', 'after-restart')


def plan() -> dict:
    lifecycles = []
    for arm in ('stock', 'patched'):
        for kv in ('fp8_ds_mla', 'nvfp4_ds_mla'):
            lifecycles.append({'arm': arm, 'kv': kv, 'spec': 'mtp0', 'target': cache.TARGET_80K})
        lifecycles.append({'arm': arm, 'kv': 'nvfp4_ds_mla', 'spec': 'dflash2', 'target': cache.TARGET_1M})
    return {
        'schema': 'r27-cache-plan/v1', 'lifecycles': lifecycles,
        'native': [{'arm': arm, 'kv': kv, 'gmu': '0.93', 'loader': 'image default'}
                   for arm in ('stock', 'patched') for kv in ('fp8_ds_mla', 'nvfp4_ds_mla')],
        'tier_order': ['cold', 'APC', 'local reset then L1', 'full container restart then L2'],
        'semantic_scope': 'Complete visible reference recall and visible-answer equality; NOT equality of all KV bytes or reasoning text.',
        'native_scope': 'Matched loader/allocator boot and 80K one-token canary; not an offload pressure or durability claim.',
        'probe_sha256': hashlib.sha256(Path(cache.PROBE).read_bytes()).hexdigest(),
    }


def resume_summary(root: Path, details: dict) -> dict:
    """Preserve every terminal case receipt after an interrupted run."""
    plan_path = root / 'r27-cache-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != details:
        raise RuntimeError('Existing R27 cache plan is incompatible; use a fresh root')
    path = root / 'r27-cache-summary.json'
    if not path.exists():
        return {'schema': 'r27-cache-summary/v1', 'plan': details, 'lifecycles': [],
                'native': [], 'all_cases_attempted': False}
    summary = json.loads(path.read_text())
    if (summary.get('schema') != 'r27-cache-summary/v1'
            or summary.get('plan') != details
            or not isinstance(summary.get('lifecycles'), list)
            or not isinstance(summary.get('native'), list)):
        raise RuntimeError('Existing R27 cache summary belongs to a different plan; use a fresh root')
    for section in ('lifecycles', 'native'):
        planned = details[section]
        seen = []
        for row in summary[section]:
            if not isinstance(row, dict) or row.get('attempted') is not True:
                raise RuntimeError(f'Existing R27 cache summary contains a non-terminal {section} row')
            case = row.get('case')
            if case not in planned or case in seen:
                raise RuntimeError('Existing R27 cache summary belongs to a different plan; use a fresh root')
            seen.append(case)
    complete = all(
        len(summary[section]) == len(details[section])
        for section in ('lifecycles', 'native')
    )
    marker = summary.get('all_cases_attempted')
    if not isinstance(marker, bool) or (marker and not complete):
        raise RuntimeError('Existing R27 cache summary has inconsistent completion accounting')
    return summary

def lifecycle_label(case: dict) -> str:
    shape = 'fp8' if case['kv'] == 'fp8_ds_mla' else 'nvfp4'
    return f"r27-cache-{case['arm']}-{case['spec']}-{shape}-{case['target']}"


def read_receipt(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, f'{type(error).__name__}: {error}'
    if not isinstance(value, dict):
        return None, 'receipt is not a JSON object'
    return value, None

def receipt_identity(path: Path) -> dict:
    resolved = path.resolve()
    return {
        'path': str(resolved),
        'sha256': hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def proof_receipts_unchanged(candidate: dict) -> bool:
    for proof in candidate.get('proofs', []):
        for name, identity in proof.items():
            if not name.endswith('_receipt'):
                continue
            if not isinstance(identity, dict):
                return False
            path = Path(identity.get('path', ''))
            expected = identity.get('sha256')
            try:
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                return False
            if observed != expected:
                return False
    return True


def option_value(arguments: object, option: str) -> str | None:
    if not isinstance(arguments, list):
        return None
    for index, value in enumerate(arguments[:-1]):
        if value == option and isinstance(arguments[index + 1], str):
            return arguments[index + 1]
    return None


def owned_l2_identity(value: object, label: str) -> dict:
    result = {
        'recorded_path': value if isinstance(value, str) else None,
        'resolved_path': None,
        'safe_path_check': False,
        'exists': False,
        'run_id': None,
        'expected_instance_id': None,
    }
    if not isinstance(value, str):
        return result
    path = Path(value)
    try:
        resolved = path.resolve(strict=False)
        l2_root = rt.L2_HOST_ROOT.resolve()
    except OSError:
        return result
    prefix = 'cache-phase-'
    suffix = '-' + label
    name = resolved.name
    run_id = (
        name[len(prefix):-len(suffix)]
        if name.startswith(prefix) and name.endswith(suffix) else ''
    )
    safe = (
        path.is_absolute()
        and str(path) == str(resolved)
        and not path.is_symlink()
        and resolved.parent == l2_root
        and bool(run_id)
    )
    result.update({
        'resolved_path': str(resolved),
        'safe_path_check': safe,
        'exists': resolved.exists() if safe else False,
        'run_id': run_id or None,
        'expected_instance_id': f'{label}-{run_id}' if run_id else None,
    })
    return result


def orphan_cleanup_plan(root: Path, details: dict, summary: dict, cleanup_run_id: str) -> dict:
    """Discover only exact L2 paths corroborated by this root's own receipts."""
    proven: dict[str, dict] = {}
    rejected = []

    def accept(path_info: dict, proof: dict) -> None:
        path = path_info['resolved_path']
        record = proven.setdefault(path, {
            'label': label,
            'path': path,
            'run_id': path_info['run_id'],
            'expected_instance_id': path_info['expected_instance_id'],
            'exists': path_info['exists'],
            'safe_path_check': True,
            'proofs': [],
        })
        record['proofs'].append(proof)

    for case in details['lifecycles']:
        label = lifecycle_label(case)
        launch_path = root / f'{label}.launch.json'
        if launch_path.exists():
            launch, launch_error = read_receipt(launch_path)
            errors = [launch_error] if launch_error else []
            launch_host = launch.get('l2_host') if launch else None
            path_info = owned_l2_identity(launch_host, label)
            launch_env = launch.get('env') if launch else None
            instance = (
                launch_env.get('LMCACHE_INSTANCE_ID')
                if isinstance(launch_env, dict) else None
            )
            if launch and (
                    launch.get('label') != label
                    or launch.get('image') != IMAGES[case['arm']]
                    or launch.get('cache') != 'lmcache'
                    or launch.get('kv') != case['kv']
                    or launch.get('spec') != case['spec']):
                errors.append('launch fields do not match the planned lifecycle')
            if not path_info['safe_path_check']:
                errors.append('launch L2 path is not a direct owned cache-phase child')
            if instance != path_info['expected_instance_id']:
                errors.append('launch instance ID does not match its owned L2 run ID')
            if (isinstance(launch_env, dict)
                    and launch_env.get('LMCACHE_SHM_NAME') != instance):
                errors.append('launch shared-memory name does not match its instance ID')
            boot_path = root / f'{label}.boot.command.json'
            boot, boot_error = read_receipt(boot_path)
            if boot_error:
                errors.append(f'boot receipt: {boot_error}')
            boot_args = boot.get('args') if boot else None
            if (not isinstance(boot_args, list)
                    or f'{path_info["resolved_path"]}:/lmcache-l2' not in boot_args
                    or f'LMCACHE_INSTANCE_ID={instance}' not in boot_args
                    or IMAGES[case['arm']] not in boot_args
                    or isinstance(boot.get('returncode') if boot else None, bool)
                    or not isinstance(boot.get('returncode') if boot else None, int)):
                errors.append('boot command does not corroborate the launch-owned L2 path')
            if errors:
                rejected.append({
                    'label': label,
                    'receipt': str(launch_path),
                    'recorded_path': launch_host,
                    'path_exists': path_info['exists'],
                    'errors': errors,
                })
            else:
                accept(path_info, {
                    'kind': 'launch-and-boot',
                    'launch_receipt': receipt_identity(launch_path),
                    'boot_receipt': receipt_identity(boot_path),
                })

        for stage in CONFIG_STAGES:
            config_path = root / f'{label}-config-{stage}.json'
            if not config_path.exists():
                continue
            config, config_error = read_receipt(config_path)
            errors = [config_error] if config_error else []
            expected = config.get('expected') if config else None
            host = expected.get('l2_host') if isinstance(expected, dict) else None
            path_info = owned_l2_identity(host, label)
            if not path_info['safe_path_check']:
                errors.append('configuration L2 path is not a direct owned cache-phase child')
            if (not isinstance(expected, dict)
                    or expected.get('image') != IMAGES[case['arm']]
                    or expected.get('mode') != 'l2-on'):
                errors.append('configuration expectation does not match the planned lifecycle')
            mounts = (
                config.get('container_config', {}).get('mounts', [])
                if isinstance(config, dict)
                and isinstance(config.get('container_config'), dict) else []
            )
            if not any(
                    isinstance(mount, dict)
                    and mount.get('Destination') == '/lmcache-l2'
                    and mount.get('Source') == path_info['resolved_path']
                    for mount in mounts):
                errors.append('configuration container mount does not corroborate the L2 path')
            command_path = root / f'{label}-config-{stage}.command.json'
            command, command_error = read_receipt(command_path)
            if command_error:
                errors.append(f'configuration command receipt: {command_error}')
            if (not command
                    or command.get('returncode') != 0
                    or option_value(command.get('args'), '--expected-l2-host')
                    != path_info['resolved_path']):
                errors.append('configuration command does not corroborate the L2 path')
            if errors:
                rejected.append({
                    'label': label,
                    'receipt': str(config_path),
                    'recorded_path': host,
                    'path_exists': path_info['exists'],
                    'errors': errors,
                })
            else:
                accept(path_info, {
                    'kind': 'configuration-and-command',
                    'stage': stage,
                    'configuration_receipt': receipt_identity(config_path),
                    'command_receipt': receipt_identity(command_path),
                })

    completed_cases = [
        row.get('case') for row in summary.get('lifecycles', [])
        if isinstance(row, dict)
    ]
    candidates = sorted(
        (row for row in proven.values() if row['exists']),
        key=lambda row: row['path'],
    )
    eligible_candidates = []
    for candidate in candidates:
        candidate['canonical_case_completed'] = any(
            lifecycle_label(case) == candidate['label'] for case in completed_cases
        )
        ownership_ended = False
        cleanup_names = (
            f"{candidate['label']}-cleanup.json",
            f"{candidate['label']}-cleanup-{candidate['run_id']}.json",
        )
        for cleanup_name in cleanup_names:
            cleanup_path = root / cleanup_name
            if not cleanup_path.exists():
                continue
            cleanup, cleanup_error = read_receipt(cleanup_path)
            before = cleanup.get('before') if cleanup else None
            cleaned_path = before.get('path') if isinstance(before, dict) else None
            if cleanup_error:
                rejected.append({
                    'label': candidate['label'],
                    'receipt': str(cleanup_path),
                    'recorded_path': candidate['path'],
                    'path_exists': True,
                    'errors': [f'cleanup receipt: {cleanup_error}'],
                })
                ownership_ended = True
            elif cleaned_path == candidate['path']:
                candidate['proofs'].append({
                    'kind': 'prior-cleanup-outcome',
                    'cleanup_receipt': receipt_identity(cleanup_path),
                })
                if cleanup.get('removed') is True:
                    rejected.append({
                        'label': candidate['label'],
                        'receipt': str(cleanup_path),
                        'recorded_path': candidate['path'],
                        'path_exists': True,
                        'errors': [
                            'receipt says this owned path was removed; its recreation is not proven owned'
                        ],
                    })
                    ownership_ended = True
        if not ownership_ended:
            eligible_candidates.append(candidate)
    candidates = eligible_candidates
    return {
        'schema': 'r27-cache-orphan-cleanup-plan/v1',
        'cleanup_run_id': cleanup_run_id,
        'root': str(root.resolve()),
        'created_at': time.time(),
        'missing_lifecycle_labels': [
            lifecycle_label(case) for case in details['lifecycles']
            if case not in completed_cases
        ],
        'discovery_scope': (
            'Exact per-label launch+boot and configuration+command receipts in this evidence root; '
            'no cache-phase-* directory glob or unscoped filesystem sweep.'
        ),
        'cleanup_helper': 'runtime.cleanup_l2_child',
        'candidates': candidates,
        'rejected_receipts': rejected,
        'safe_to_execute': not rejected,
        'unrecoverable_scope': (
            'A directory whose exact ownership receipts were overwritten or lost is intentionally '
            'not discovered or deleted by this plan.'
        ),
    }


def stop_outcome(phase: cache.CachePhase, label: str) -> dict:
    before = len(phase.checks)
    call_error = None
    try:
        phase.stop(label)
    except Exception as error:
        call_error = f'{type(error).__name__}: {error}'
    emitted = phase.checks[before:]
    failures = [row for row in emitted if row.get('passed') is not True]
    return {
        'attempted': True,
        'passed': call_error is None and not failures,
        'error': call_error,
        'recorded_failure_gates': failures,
    }


def record_gate_safely(name: str, passed: bool, detail: object) -> dict:
    try:
        rt.record_gate(name, passed, detail)
        return {'recorded': True, 'error': None}
    except Exception as error:
        return {'recorded': False, 'error': f'{type(error).__name__}: {error}'}


def cleanup_proven_orphans(
        phase: cache.CachePhase, cleanup_plan: dict, plan_receipt: Path) -> dict:
    result = {
        'schema': 'r27-cache-orphan-cleanup-run/v1',
        'cleanup_run_id': phase.run_id,
        'started_at': time.time(),
        'plan_receipt': str(plan_receipt),
        'precleanup_teardown': None,
        'cleanups': [],
        'passed': False,
    }
    if cleanup_plan.get('safe_to_execute') is not True:
        result['error'] = 'Receipt conflicts prevent scoped orphan cleanup'
    else:
        result['precleanup_teardown'] = stop_outcome(phase, 'orphan-cleanup-precondition')
        if result['precleanup_teardown']['passed']:
            for index, candidate in enumerate(cleanup_plan['candidates']):
                path = Path(candidate['path'])
                receipt = {
                    'candidate': candidate,
                    'attempted': False,
                    'before': cache.CachePhase.inventory(path),
                    'removed': False,
                    'error': None,
                    'cleanup_helper': 'runtime.cleanup_l2_child',
                }
                command_path = rt.ROOT / f'cleanup-{path.name}.command.json'
                log_path = rt.ROOT / f'cleanup-{path.name}.log'
                try:
                    if not proof_receipts_unchanged(candidate):
                        raise RuntimeError(
                            'Ownership receipts changed after cleanup planning; refusing deletion'
                        )
                    if command_path.exists() or log_path.exists():
                        raise RuntimeError(
                            'Existing cleanup execution receipt requires manual archival; '
                            'refusing a blind retry'
                        )
                    receipt['attempted'] = True
                    if path.exists():
                        rt.cleanup_l2_child(path)
                    receipt['removed'] = not path.exists()
                except Exception as error:
                    receipt['error'] = f'{type(error).__name__}: {error}'
                receipt_path = save_new_receipt(
                    f'r27-cache-orphan-cleanup-{phase.run_id}-{index}.json', receipt
                )
                receipt['receipt'] = str(receipt_path)
                receipt['gate'] = record_gate_safely(
                    f'r27-cache:orphan-cleanup:{phase.run_id}:{index}',
                    receipt['removed'],
                    receipt,
                )
                result['cleanups'].append(receipt)
                if not receipt['removed']:
                    break
        else:
            result['error'] = 'Test-container teardown failed before cleanup'
    result['finished_at'] = time.time()
    result['passed'] = (
        cleanup_plan.get('safe_to_execute') is True
        and isinstance(result.get('precleanup_teardown'), dict)
        and result['precleanup_teardown'].get('passed') is True
        and all(row.get('removed') is True for row in result['cleanups'])
        and len(result['cleanups']) == len(cleanup_plan['candidates'])
    )
    result_path = save_new_receipt(
        f'r27-cache-orphan-cleanup-run-{phase.run_id}.json', result
    )
    result['receipt'] = str(result_path)
    return result


def save_new_receipt(name: str, value: object) -> Path:
    path = rt.ROOT / name
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite cache recovery receipt: {path}')
    return rt.save_json(name, value)


def persist_terminal_row(
        phase: cache.CachePhase, row: dict, *, host: Path | None = None) -> None:
    """Persist measured state before teardown, then append fail-closed cleanup outcomes."""
    label = row['label']
    row.setdefault('passed', False)
    row['measurement_terminal'] = {
        'attempted': row.get('attempted') is True,
        'passed': row['passed'] is True,
        'persisted_before_teardown': True,
    }
    rt.save_json(label + '-result.json', row)
    row['measurement_gate'] = record_gate_safely(
        label + ':measurement', row['passed'] is True, row
    )
    row['teardown'] = stop_outcome(phase, label)
    if host is None:
        cleanup = {'required': False, 'attempted': False, 'passed': True, 'receipt': None}
    elif row['teardown']['passed']:
        cleanup_label = f'{label}-cleanup-{phase.run_id}'
        cleanup_attempted = False
        try:
            cleanup_receipt_path = rt.ROOT / f'{cleanup_label}.json'
            if cleanup_receipt_path.exists():
                raise RuntimeError(
                    f'Refusing to overwrite prior lifecycle cleanup receipt: {cleanup_receipt_path}'
                )
            cleanup_attempted = True
            passed = phase.cleanup_owned_l2(host, cleanup_label)
            cleanup = {
                'required': True,
                'attempted': True,
                'passed': passed,
                'receipt': str(rt.ROOT / f'{cleanup_label}.json'),
                'error': None,
            }
        except Exception as error:
            cleanup = {
                'required': True,
                'attempted': cleanup_attempted,
                'passed': False,
                'receipt': str(rt.ROOT / f'{cleanup_label}.json'),
                'error': f'{type(error).__name__}: {error}',
            }
    else:
        cleanup_label = f'{label}-cleanup-skipped-{phase.run_id}'
        cleanup = {
            'required': True,
            'attempted': False,
            'passed': False,
            'receipt': str(rt.ROOT / f'{cleanup_label}.json'),
            'error': 'Cleanup skipped because the test container teardown failed',
        }
        save_new_receipt(f'{cleanup_label}.json', cleanup)
    row['l2_cleanup'] = cleanup
    row['terminal_isolation_safe'] = (
        row['teardown']['passed'] is True and cleanup['passed'] is True
    )
    row['terminal_gate'] = record_gate_safely(
        label + ':terminal-isolation', row['terminal_isolation_safe'], row
    )
    rt.save_json(label + '-result.json', row)


def lifecycle(phase: cache.CachePhase, case: dict) -> dict:
    arm, kv, spec, target = (case[key] for key in ('arm', 'kv', 'spec', 'target'))
    label = lifecycle_label(case)
    host = phase.owned_l2(label)
    namespace = label + '-' + phase.run_id
    row = {'label': label, 'case': case, 'image': IMAGES[arm], 'booted': False,
           'requests': {}, 'configurations': {}, 'checks': {}, 'attempted': True}
    try:
        row['booted'] = boot(label, arm=arm, spec=spec, dcp=4, cache='lmcache', kv=kv,
                            extra_env={'LMCACHE_L2_HOST_DIR': str(host),
                                       'LMCACHE_INSTANCE_ID': namespace, 'LMCACHE_SHM_NAME': namespace})
        if not row['booted']:
            row['failure_class'] = 'boot_failed'
            return row
        ok, prompt, metadata = phase.prepare_prompt(label, target_tokens=target,
                                                    identity=namespace, kind='reference')
        row['prompt'] = metadata
        if not ok:
            row['failure_class'] = 'prompt_preparation'
            return row
        request_kwargs = dict(prompt=prompt, cache_salt=namespace,
                              max_tokens=cache.SEMANTIC_MAX_TOKENS, deadline=1800,
                              expected_tokens=target, expected_reference=metadata['reference_code'])

        def request(stage: str) -> dict:
            code, data, path = phase.request(label + '-' + stage, **request_kwargs)
            row['requests'][stage] = {'receipt': str(path), 'returncode': code,
                                      'summary': phase.request_summary(data),
                                      'checks': data.get('checks'),
                                      'answer_evidence': data.get('answer_evidence')}
            return data

        def snapshot(stage: str) -> dict:
            code, data, path = phase.config_snapshot(label + '-config-' + stage,
                                                     image=IMAGES[arm], mode='l2-on', l2_host=host)
            row['configurations'][stage] = {'receipt': str(path), 'returncode': code,
                                            'passed': data.get('passed')}
            return data

        snapshot('before')
        cold = request('cold')
        time.sleep(cache.STORE_DRAIN_SECONDS)
        apc = request('apc')
        after_apc = snapshot('after-apc')
        reset_code, reset_data, reset_path = phase.reset_local(label + '-reset-local')
        row['local_reset'] = {'returncode': reset_code, 'passed': reset_data.get('passed'),
                              'receipt': str(reset_path)}
        warm = request('l1')
        time.sleep(cache.STORE_DRAIN_SECONDS)
        before_restart = snapshot('before-restart')
        row['restarted'] = phase.restart(label)
        restarted = request('l2') if row['restarted'] else {}
        after_restart = snapshot('after-restart') if row['restarted'] else {}
        verdict = phase.lifecycle_verdict(target, 'reference', cold, warm, restarted)
        row['verdict'] = verdict
        apc_stats = phase.request_summary(apc).get('cache_stats') or {}
        row['checks'] = {
            **verdict['checks'],
            'apc_complete_visible_answer': apc.get('passed') is True,
            'apc_native_hit_observed': apc_stats.get('num_vllm_cached_tokens', 0) > 0,
            'local_reset_succeeded': reset_code == 0 and reset_data.get('passed') is True,
            'actual_configuration_verified': all(s.get('passed') is True for s in row['configurations'].values()),
        }
        l1_before = phase.metrics_matching(after_apc, r'(l0_l1|l1).*(load|retrieve|read)')
        l1_after = phase.metrics_matching(before_restart, r'(l0_l1|l1).*(load|retrieve|read)')
        l2_before = phase.metrics_matching(after_apc, r'l2.*(prefetch|load).*(hit|completed|chunks)')
        l2_after = phase.metrics_matching(before_restart, r'l2.*(prefetch|load).*(hit|completed|chunks)')
        row['l1_load_delta'] = {key: l1_after.get(key, 0) - l1_before.get(key, 0)
                                for key in l1_after.keys() | l1_before.keys()}
        row['l2_load_delta_during_l1'] = {key: l2_after.get(key, 0) - l2_before.get(key, 0)
                                         for key in l2_after.keys() | l2_before.keys()}
        row['checks']['l1_restore_path_observed'] = (
            any(value > 0 for value in row['l1_load_delta'].values())
            and not any(value > 0 for value in row['l2_load_delta_during_l1'].values()))
        l2_load = phase.metrics_matching(after_restart, r'l2.*(prefetch|load).*(hit|completed|chunks)')
        row['checks']['restart_l2_path_observed'] = any(value > 0 for value in l2_load.values())
        row['passed'] = all(row['checks'].values())
        return row
    except Exception as error:
        row['error'] = repr(error)
        row['passed'] = False
        return row
    finally:
        persist_terminal_row(phase, row, host=host)




def native_boot(phase: cache.CachePhase, case: dict) -> dict:
    arm, kv = case['arm'], case['kv']
    label = f"r27-native-{arm}-{'fp8' if kv == 'fp8_ds_mla' else 'nvfp4'}"
    row = {'label': label, 'case': case, 'image': IMAGES[arm], 'attempted': True,
           'scope': 'Default-loader boot plus exact 80K prompt, one-token response only',
           'booted': False, 'canary_passed': False}
    try:
        row['booted'] = boot(label, arm=arm, spec='mtp0', dcp=4, cache='native', kv=kv)
        if not row['booted']:
            log_path = rt.ROOT / (label + '.docker.log')
            text = log_path.read_text(errors='replace') if log_path.exists() else ''
            row['failure_class'] = 'native_boot_oom' if 'out of memory' in text.lower() else 'native_boot_failed'
            row['log_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            return row
        ok, prompt, metadata = phase.prepare_prompt(label, target_tokens=cache.TARGET_80K,
                                                    identity=phase.run_id + '-' + label, kind='period')
        row['prompt'] = metadata
        if not ok:
            row['failure_class'] = 'prompt_preparation'
            return row
        code, data, path = phase.request(label + '-canary', prompt=prompt,
                                         cache_salt=phase.run_id + '-' + label,
                                         max_tokens=1, deadline=900, expected_tokens=cache.TARGET_80K,
                                         ignore_eos=True)
        row['canary_passed'] = code == 0 and data.get('passed') is True
        row['canary_receipt'] = str(path)
        return row
    except Exception as error:
        row['error'] = repr(error)
        return row
    finally:
        row['passed'] = row['booted'] and row['canary_passed']
        persist_terminal_row(phase, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true')
    actions.add_argument('--orphan-cleanup-plan-only', action='store_true')
    actions.add_argument('--cleanup-proven-orphans', action='store_true')
    args = parser.parse_args()
    details = plan()
    if args.plan_only:
        print(json.dumps(details, indent=2))
        return
    summary = resume_summary(rt.ROOT, details)
    missing_lifecycles = [
        case for case in details['lifecycles']
        if not any(row['case'] == case for row in summary['lifecycles'])
    ]
    missing_native = [
        case for case in details['native']
        if not any(row['case'] == case for row in summary['native'])
    ]
    recovery_requested = (
        args.orphan_cleanup_plan_only
        or args.cleanup_proven_orphans
        or bool(missing_lifecycles)
        or bool(missing_native)
    )
    if not recovery_requested:
        if summary['all_cases_attempted'] is not True:
            summary['all_cases_attempted'] = True
            rt.save_json('r27-cache-summary.json', summary)
        return

    ensure_scope()
    phase = cache.CachePhase()
    cleanup_details = orphan_cleanup_plan(rt.ROOT, details, summary, phase.run_id)
    cleanup_plan_name = f'r27-cache-orphan-cleanup-plan-{phase.run_id}.json'
    cleanup_plan_receipt = save_new_receipt(cleanup_plan_name, cleanup_details)
    if args.orphan_cleanup_plan_only:
        print(json.dumps({
            **cleanup_details,
            'plan_receipt': str(cleanup_plan_receipt),
            'filesystem_deletion_executed': False,
        }, indent=2))
        return

    cleanup_result = cleanup_proven_orphans(
        phase, cleanup_details, cleanup_plan_receipt
    )
    if not cleanup_result['passed']:
        raise RuntimeError(
            f"Scoped orphan cleanup failed; no GPU case will start: {cleanup_result['receipt']}"
        )
    if args.cleanup_proven_orphans:
        print(json.dumps(cleanup_result, indent=2))
        return

    summary.setdefault('orphan_cleanup_runs', []).append({
        'plan_receipt': str(cleanup_plan_receipt),
        'execution_receipt': cleanup_result['receipt'],
        'passed': cleanup_result['passed'],
    })
    rt.save_json('r27-cache-summary.json', summary)
    plan_path = rt.ROOT / 'r27-cache-plan.json'
    if not plan_path.exists():
        rt.save_json('r27-cache-plan.json', details)

    for case in missing_lifecycles:
        row = lifecycle(phase, case)
        summary['lifecycles'].append(row)
        summary['all_cases_attempted'] = (
            len(summary['lifecycles']) == len(details['lifecycles'])
            and len(summary['native']) == len(details['native'])
        )
        rt.save_json('r27-cache-summary.json', summary)
        if row.get('terminal_isolation_safe') is not True:
            raise RuntimeError(
                f"{row['label']} teardown/cleanup failed; no later GPU case will start"
            )
    for case in missing_native:
        row = native_boot(phase, case)
        summary['native'].append(row)
        summary['all_cases_attempted'] = (
            len(summary['lifecycles']) == len(details['lifecycles'])
            and len(summary['native']) == len(details['native'])
        )
        rt.save_json('r27-cache-summary.json', summary)
        if row.get('terminal_isolation_safe') is not True:
            raise RuntimeError(
                f"{row['label']} teardown failed; no later GPU case will start"
            )


if __name__ == '__main__':
    main()
