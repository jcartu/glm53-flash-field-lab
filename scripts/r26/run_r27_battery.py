#!/usr/bin/env python3
"""Run the pinned R27 decision battery only after R26 completes and restores service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from run_qad_battery import wait_for_process

HERE = Path(__file__).resolve().parent
LAB = Path('/home/josh/omp-workspace/drock-lmcache')
SOURCE = LAB / 'r27-source-20260906'
STOCK = 'voipmonitor/vllm@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512'
PHASES = [
    ('r27-boundary-checkpoints', 'r27_boundary_phase.py', [], 43200),
    ('r27-matched-speed', 'r27_workload_phase.py', ['--section', 'speed'], 10800),
    ('r27-page-geometry-cost', 'r27_workload_phase.py', ['--section', 'geometry'], 7200),
    ('r27-scheduler-controls', 'r27_scheduler_phase.py', [], 14400),
    ('r27-natural-acceptance', 'r27_workload_phase.py', ['--section', 'natural'], 7200),
    ('r27-cache-lifecycle', 'r27_cache_phase.py', [], 14400),
    ('r27-answer-integrity', 'r27_workload_phase.py', ['--section', 'quality'], 14400),
]
HELPERS = ('r27_config.py', 'r27_boundary_probe.py', 'runtime.py', 'run_qualification.py',
           'steady_metrics.py', 'quality_probes.py', 'realistic_acceptance_probe.py',
           'cache_phase.py', 'cache_probe.py', 'cache_config_probe.py', 'agent_workload_recheck.py',
           'quality_phase.py', 'realistic_acceptance_phase.py', 'clean_rerun_phase.py',
           'run_qad_battery.py')


def execution_plan() -> dict:
    names = sorted({entry[1] for entry in PHASES} | set(HELPERS))
    return {'schema': 'r27-execution-plan/v1', 'phases': PHASES,
            'source_manifest': str(SOURCE / 'manifest.json'),
            'scripts': {name: {'exists': (HERE / name).is_file(),
                               'sha256': hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                               if (HERE / name).is_file() else None} for name in names},
            'qad_scope_retained': 'Full original R26-image checkpoint runbook remains separate after this decision pass.',
            'promotion': False}


def verify_sources() -> dict:
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    if manifest.get('schema') != 'r27-source-snapshot/v1' or len(manifest.get('images', [])) != 3:
        raise RuntimeError('Incomplete R27 source snapshot')
    for arm in manifest['images']:
        inspect = json.loads(subprocess.run(['docker', 'image', 'inspect', arm['image']],
                                             capture_output=True, text=True, check=True, timeout=30).stdout)[0]
        for item in arm['files']:
            observed = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
            if observed != item['sha256']:
                raise RuntimeError(f'Changed source snapshot: {item["path"]}')
        arm['local_image_id'] = inspect['Id']
    return manifest


def phase_report_finality(root: Path, executed: dict) -> dict:
    contracts = {
        'r27-boundary-checkpoints': ('r27-boundary-phase-summary.json', 'r27-boundary-phase-summary/v1', None),
        'r27-matched-speed': ('r27-speed-summary.json', 'r27-speed/v1', 'all_cells_attempted'),
        'r27-page-geometry-cost': ('r27-geometry-summary.json', 'r27-geometry/v1', 'all_cells_attempted'),
        'r27-scheduler-controls': ('r27-scheduler/phase-summary.json', 'glm-r27-scheduler-phase/v1', None),
        'r27-natural-acceptance': ('r27-natural-acceptance-summary.json', 'r27-natural-acceptance/v1', 'all_arms_attempted'),
        'r27-cache-lifecycle': ('r27-cache-summary.json', 'r27-cache-summary/v1', 'all_cases_attempted'),
        'r27-answer-integrity': ('r27-answer-quality-summary.json', 'r27-answer-quality/v1', 'all_arms_attempted'),
    }
    invocations = {row['phase']: row for row in executed.get('phases', [])}
    checks = {}
    for phase, (filename, schema, marker) in contracts.items():
        path = root / filename
        row = {'report': str(path), 'final': False}
        try:
            report = json.loads(path.read_text())
            code = invocations.get(phase, {}).get('returncode')
            complete = report.get('schema') == schema and code in (0, 1)
            if marker:
                complete = complete and report.get(marker) is True
            if phase == 'r27-boundary-checkpoints':
                counts = report.get('case_accounting', {})
                complete = complete and report.get('source_verified') is True
                complete = complete and len(report.get('scalar_results', [])) == 3
                complete = complete and counts.get('planned_http', 0) > 0 and (
                    counts.get('measured_http', 0) + counts.get('unattempted_http', 0)
                    == counts.get('planned_http')
                )
            elif phase == 'r27-scheduler-controls':
                complete = complete and report.get('observed', {}).get('all_boot_cases_attempted') is True
                complete = complete and not report.get('findings', {}).get('harness_failures')
            elif phase in ('r27-matched-speed', 'r27-page-geometry-cost'):
                complete = complete and bool(report.get('cells')) and all(
                    not cell.get('unavailable') and not cell.get('error')
                    and (not cell.get('executed') or cell.get('speed_eligible') is True)
                    for cell in report['cells']
                )
            row.update({'final': bool(complete), 'returncode': code, 'schema': report.get('schema')})
        except (OSError, ValueError, KeyError, TypeError) as error:
            row['error'] = repr(error)
        checks[phase] = row
    return {'all_final': all(row['final'] for row in checks.values()), 'phases': checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--wait-created', type=float)
    parser.add_argument('--upstream-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=LAB / 'r27-battery')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(execution_plan(), indent=2))
        return
    if args.wait_pid is None or args.wait_created is None or args.upstream_root is None:
        parser.error('execution requires --wait-pid, --wait-created, and --upstream-root')
    if args.output_root.resolve().parent != LAB or not args.output_root.name.startswith('r27-'):
        raise RuntimeError('R27 requires its own output root')
    print('R27 BATTERY QUEUED: waiting for complete R26 clean reruns and restoration', flush=True)
    wait_for_process(args.wait_pid, args.wait_created)
    complete = json.loads((args.upstream_root / 'clean-rerun-completed.json').read_text())
    restored = json.loads((args.upstream_root / 'production-restored.json').read_text())
    if (complete.get('complete') is not True or complete.get('clean') != complete.get('planned')
            or complete.get('final_diagnostics_attempted') is not True
            or complete.get('finished_at', 0) < args.wait_created
            or restored.get('healthy') is not True or restored.get('timestamp', 0) < args.wait_created):
        raise RuntimeError('R26 completion/restoration boundary is not valid; no R27 GPU launch')
    details = execution_plan()
    if not all(item['exists'] for item in details['scripts'].values()):
        raise RuntimeError('A planned R27 module is missing')
    sources = verify_sources()
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / 'source-manifest.json').write_text(json.dumps(sources, indent=2) + '\n')
    (args.output_root / 'execution-plan.json').write_text(json.dumps(details, indent=2) + '\n')
    os.environ.update({
        'BATTERY_ROOT': str(args.output_root), 'BATTERY_CONTAINER': 'r27-test',
        'BATTERY_PORT': '5002', 'BATTERY_IMAGE': STOCK,
        'BATTERY_MODEL_DIR': '/mnt/2king/models/GLM-5.3-Flash-NVFP4',
        'BATTERY_MODEL_CACHE_TAG': 'r27-published-target',
        'BATTERY_L2_SHARED': '/mnt/2king/lmcache-r26-battery/cache-phase-r27-shared',
        'PARENT_ROOT': str(args.upstream_root.parent),
    })
    import run_qualification as coordinator
    coordinator.PHASES = PHASES
    started = time.time()
    coordinator.main()
    executed = json.loads((args.output_root / 'qualification-executed.json').read_text())
    restored_r27 = json.loads((args.output_root / 'production-restored.json').read_text())
    attempted = executed.get('all_phases_attempted') is True and len(executed.get('phases', [])) == len(PHASES)
    finality = phase_report_finality(args.output_root, executed)
    receipt = {'schema': 'r27-qualification-completed/v1', 'started_at': started,
               'finished_at': time.time(), 'all_phases_attempted': attempted,
               'phase_execution': executed.get('phases', []),
               'all_phase_reports_final': finality['all_final'],
               'phase_report_finality': finality,
               'production_healthy': restored_r27.get('healthy') is True,
               'qualification_is_not_promotion': True,
               'note': 'Completion means all phase attempts are preserved; failing release behavior remains failed and must be reported.'}
    (args.output_root / 'r27-qualification-completed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not finality['all_final']:
        raise SystemExit('R27 phase evidence is incomplete; full QAD remains gated')
    print('R27 BATTERY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
