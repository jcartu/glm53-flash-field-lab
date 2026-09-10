#!/usr/bin/env python3
"""Matched R27 execution, natural acceptance, and answer-integrity probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import quality_phase as quality
import realistic_acceptance_phase as natural
import runtime as rt
import steady_metrics
from r27_config import IMAGES, boot, ensure_scope

from r27_boundary_phase import FINE_GEOMETRY_ENV
MODES = (
    ('nospec-dcp1', 'mtp0', 1, 'fp8_ds_mla'),
    ('mtp3-dcp1', 'mtp3', 1, 'fp8_ds_mla'),
    ('mtp3-dcp4', 'mtp3', 4, 'fp8_ds_mla'),
    ('dflash-dcp4', 'dflash2', 4, 'nvfp4_ds_mla'),
)


GEOMETRY_MODES = (
    ('default', 'dflash2', 4, 'nvfp4_ds_mla'),
    ('fine-pages-256', 'dflash2', 4, 'nvfp4_ds_mla'),
)


def plan(section: str) -> dict:
    speed = [dict(arm=arm, mode=mode, spec=spec, dcp=dcp, kv=kv,
                  concurrency=[1, 8], contexts=[0, 32768], seconds=60)
             for mode, spec, dcp, kv in MODES for arm in ('stock', 'patched')]
    quality_arms = [quality.arm_manifest(arm) for arm in quality_plan()]
    return {
        'schema': 'r27-workload-plan/v1', 'section': section,
        'speed': speed if section == 'speed' else [],
        'geometry': [dict(arm='patched', mode=mode, spec=spec, dcp=dcp, kv=kv,
                          environment=dict(FINE_GEOMETRY_ENV) if mode == 'fine-pages-256' else {},
                          concurrency=[1, 8], contexts=[0, 32768], seconds=60)
                     for mode, spec, dcp, kv in GEOMETRY_MODES] if section == 'geometry' else [],
        'natural': [{'arm': arm, 'requests': 32, 'dcp': 4, 'spec': 'mtp3',
                     'kv': 'fp8_ds_mla', 'draft_head': 'nvfp4'}
                    for arm in ('stock', 'patched')] if section == 'natural' else [],
        'quality': quality_arms if section == 'quality' else [],
        'sources': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('runtime.py', 'steady_metrics.py', 'quality_probes.py',
                                 'realistic_acceptance_probe.py')},
        'benchmark_sha256': hashlib.sha256(rt.BENCH.read_bytes()).hexdigest(),
        'speed_scope': 'Same frozen client workloads and settings as R26. Matched release means use matching context/concurrency/config only; workload output and verifier counters share interior windows.',
        'geometry_scope': 'Same patched image and settings; both target and recurrent pages change from 2048 to 256. This is a coupled geometry tradeoff, not an isolated #676 effect or reproduction of the public recurrent-only penalty.',
        'quality_scope': 'Low-reasoning and template-default profiles remain separate. Small semantic tallies are observations, not equivalence gates.',
    }


def quality_plan() -> tuple[quality.Arm, ...]:
    result = []
    for candidate in ('stock', 'patched'):
        for spec in ('mtp3', 'dflash2'):
            label = f'r27-quality-{candidate}-{spec}'
            result.append(quality.Arm(
                label=label, image=IMAGES[candidate], dcp=4, spec=spec,
                cache='vram', kv='fp8_ds_mla', proposal_head='nvfp4' if spec == 'mtp3' else 'not MTP',
                role=f'R27 {candidate} matched answer-integrity control; published target checkpoint',
                probes=(
                    quality.ProbePlan(label + '-long', 'long', 3600,
                                      ('--waves', '3', '--max-tokens', '8192', '--seed-base', '26090500')),
                    quality.ProbePlan(label + '-tool-order', 'tool-order', 1200),
                    quality.ProbePlan(label + '-sampling', 'sampling', 2400),
                ),
                profiles=(quality.ProfilePlan(label + '-estonia-low', 'estonia', 24),
                          quality.ProfilePlan(label + '-lavd-low', 'lavd-test', 24)),
            ))
    return tuple(result)


def speed(geometry_only: bool = False) -> dict:
    import clean_rerun_phase as isolation
    prefix = 'geometry' if geometry_only else 'speed'
    modes = GEOMETRY_MODES if geometry_only else MODES
    arms = ('patched',) if geometry_only else ('stock', 'patched')
    output_name = f'r27-{prefix}-summary.json'
    summary = {'schema': f'r27-{prefix}/v1', 'cells': [], 'all_cells_attempted': False}
    for mode, spec, dcp, kv in modes:
        for arm in arms:
            label = f'r27-{prefix}-{mode}-{arm}'
            row = {'label': label, 'arm': arm, 'image': IMAGES[arm],
                   'spec': spec, 'dcp': dcp, 'kv': kv, 'booted': False,
                   'executed': False, 'speed_eligible': False}
            try:
                row['quiet_before_boot'] = isolation.wait_for_quiet()
                if not row['quiet_before_boot']['clean_start']:
                    row['unavailable'] = 'No exclusive GPU window; no speed launch'
                    continue
                row['booted'] = boot(label, arm=arm, spec=spec, dcp=dcp, kv=kv,
                                     extra_env=dict(FINE_GEOMETRY_ENV) if mode == 'fine-pages-256' else None)
                if not row['booted']:
                    continue
                row['quiet_before_bench'] = isolation.wait_for_quiet()
                if not row['quiet_before_bench']['clean_start']:
                    row['unavailable'] = 'Exclusive GPU window lost after boot'
                    continue
                with steady_metrics.Recorder(rt.BASE_URL, rt.ROOT / f'{label}.steady.metrics.jsonl'):
                    row['executed'] = rt.bench(label, conc='1,8', contexts='0,32k', duration=60)
                if row['executed']:
                    counters = steady_metrics.summarize(rt.ROOT, label)
                    windows_valid = counters.get('all_windows_valid', False)
                    if spec != 'mtp0':
                        windows_valid = windows_valid and all(
                            cell.get('aggregate_verifier_steps_per_second', 0) > 0
                            for cell in counters.get('cells', []))
                    row['counter_windows_valid'] = bool(windows_valid)
                    row['exclusive_gpu_window'] = isolation.cell_clean(label)
                    row['speed_eligible'] = bool(windows_valid and row['exclusive_gpu_window'])
                    row['steady_summary'] = str(rt.ROOT / f'{label}.steady-summary.json')
            except Exception as error:
                row['error'] = repr(error)
            finally:
                rt.stop()
                rt.record_gate('r27:speed:' + label, row['speed_eligible'], row)
                summary['cells'].append(row)
                rt.save_json(output_name, summary)
    summary['all_cells_attempted'] = len(summary['cells']) == len(modes) * len(arms)
    summary['all_speed_windows_eligible'] = all(row['speed_eligible'] for row in summary['cells'])
    rt.save_json(output_name, summary)
    return summary


def natural_acceptance() -> dict:
    import clean_rerun_phase as isolation
    probe = natural.load_probe()
    summary = {'schema': 'r27-natural-acceptance/v1', 'arms': [],
               'input_manifest': probe.input_manifest(rt.MODEL_NAME, probe.DEFAULT_SEEDS),
               'all_arms_attempted': False}
    for candidate in ('stock', 'patched'):
        arm = natural.Arm('r27-' + candidate, IMAGES[candidate], 'nvfp4',
                          f'R27 {candidate}, unchanged published target and normal prompt slice')
        start = time.time()
        result = natural.run_arm(rt, probe, arm, 240)
        end = time.time()
        samples = isolation.covered_window(isolation.isolation_rows(rt.ROOT), start, end)
        result['exclusive_gpu_window'] = samples is not None and not any(row['foreign'] or row.get('speed_eligible', True) is not True for row in samples)
        summary['arms'].append(result)
        rt.save_json('r27-natural-acceptance-summary.json', summary)
    summary['all_arms_attempted'] = len(summary['arms']) == 2
    rt.save_json('r27-natural-acceptance-summary.json', summary)
    return summary


def answer_quality() -> dict:
    summary = {'schema': 'r27-answer-quality/v1', 'arms': [], 'all_arms_attempted': False}
    for arm in quality_plan():
        quality.run_arm(rt, arm)
        row = {'arm': quality.arm_manifest(arm),
               'receipts': [str(rt.ROOT / f'{item.label}.json') for item in (*arm.probes, *arm.profiles)],
               'gates': [gate for gate in quality._LOCAL_GATES if arm.label in gate['name']]}
        # The unchanged benchmark profile uses the checkpoint's template-default
        # reasoning; do not pool it with explicit-low results above.
        if arm.spec == 'dflash2':
            label = arm.label + '-lavd-template-default'
            try:
                if boot(label, arm='stock' if arm.image == IMAGES['stock'] else 'patched',
                        spec=arm.spec, dcp=4, kv=arm.kv):
                    row['template_default_profile_executed'] = rt.profile(label, 'lavd-test', 8)
                    row['template_default_profile'] = str(rt.ROOT / f'{label}.json')
                else:
                    row['template_default_profile_executed'] = False
            finally:
                rt.stop()
        summary['arms'].append(row)
        rt.save_json('r27-answer-quality-summary.json', summary)
    summary['all_arms_attempted'] = len(summary['arms']) == len(quality_plan())
    rt.save_json('r27-answer-quality-summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=('speed', 'geometry', 'natural', 'quality'), required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    details = plan(args.section)
    if args.plan_only:
        print(json.dumps(details, indent=2))
        return
    ensure_scope()
    rt.save_json('r27-workload-plan-' + args.section + '.json', details)
    try:
        if args.section == 'geometry':
            speed(geometry_only=True)
        else:
            {'speed': speed, 'natural': natural_acceptance, 'quality': answer_quality}[args.section]()
    finally:
        rt.stop()


if __name__ == '__main__':
    main()
