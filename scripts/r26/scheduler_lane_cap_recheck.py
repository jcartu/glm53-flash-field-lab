#!/usr/bin/env python3
"""Retry only scheduler groups skipped by the incorrect exact-lane assertion."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import scheduler_recheck_phase as scheduler
import runtime as rt


def selected_groups(parent: Path) -> tuple:
    original = parent / 'followups' / 'scheduler-recheck'
    names = set()
    for group in scheduler.BOOT_GROUPS:
        discovery, error = scheduler.read_json_file(original / f'{group.name}-api-discovery.json')
        if discovery is None:
            raise RuntimeError(f'Cannot classify prior scheduler group {group.name}: {error}')
        if (discovery.get('schema') == 'overlay-r26'
                and discovery.get('passed') is False
                and not discovery.get('boot_structure_mismatches')
                and isinstance(discovery.get('effective_lane_budget'), int)
                and discovery['effective_lane_budget'] >= 1):
            names.add(group.name)
    if any(group.cache_mode == 'lmcache' and group.name in names for group in scheduler.BOOT_GROUPS):
        # Repeat writer and reader together; do not rely on an old cache prime.
        names.update(group.name for group in scheduler.BOOT_GROUPS if group.cache_mode == 'lmcache')
    chosen = [group for group in scheduler.BOOT_GROUPS if group.name in names]
    template = next(group for group in scheduler.BOOT_GROUPS if group.name == 'overlay-factor-bt16384-default-dma')
    plans = tuple(replace(plan, name='overlay-factor-bt32768-four-lane-diagnostic') for plan in template.plans)
    chosen.append(replace(template, name='overlay-factor-bt32768-four-lane-diagnostic', batch_tokens=32768,
                          factor_role='four-effective-lane-diagnostic', plans=plans))
    return tuple(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    parent = Path(os.environ.get('PARENT_ROOT', '/home/josh/omp-workspace/drock-lmcache/r26-battery'))
    groups = selected_groups(parent)
    report = {
        'schema': 'r26-scheduler-lane-cap-recheck/v1',
        'previous_root': str(parent / 'followups' / 'scheduler-recheck'),
        'reason': 'Explicit lane counts are upper bounds. The shipped resolver may clip them to the scheduled-token/cache-block budget; a positive clipped count is not an invalid API state.',
        'groups': [scheduler.group_to_json(group) for group in groups],
        'extra_diagnostic': '32768-token budget requests four effective lanes under the observed 8192-token global block size; actual readback and capacity remain authoritative.',
        'results': {},
    }
    if args.plan_only:
        import json
        print(json.dumps({'groups': [group.name for group in groups], 'plans': sum(len(group.plans) for group in groups), 'gpu_jobs_started': False}, indent=2))
        return
    state = scheduler.PhaseState()
    rt.save_json('scheduler-lane-cap-recheck.json', report)
    try:
        for group in groups:
            scheduler.boot_group(state, group)
            report['results'] = state.invocations
            report['api_discoveries'] = state.api_discoveries
            report['runtime_facts'] = state.runtime_facts
            report['errors'] = state.errors
            report['result_files'] = {
                plan.name: str(rt.ROOT / scheduler.OUTPUT_DIR / f'{plan.name}.json')
                for selected in groups for plan in selected.plans if plan.name in state.invocations
            }
            rt.save_json('scheduler-lane-cap-recheck.json', report)
        report['all_groups_attempted'] = True
        report['requested_plan_count'] = sum(len(group.plans) for group in groups)
        report['recorded_plan_count'] = len(state.invocations)
        rt.save_json('scheduler-lane-cap-recheck.json', report)
    finally:
        rt.stop()


if __name__ == '__main__':
    main()
