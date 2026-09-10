#!/usr/bin/env python3
"""Stock R30 disk-cache lifecycle campaign; stock R29 is the negative control."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

SCRIPTS = Path('/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r29')
sys.path.insert(0, str(SCRIPTS))
import cache_campaign as campaign

ROOT = campaign.ROOT
PRIOR = ROOT / 'cache-campaign'
FROZEN = ROOT / 'source-qualified-r30/manifest.json'
PREFLIGHT = ROOT / 'r30-cache-campaign/preflight/launcher-contract.json'
rt = campaign.rt

R30_IMAGE = 'localinferencelab/vllm@sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131'
RAM_PERFORMANCE_SCOPE = (
    'Reused PR64-overlay RAM timing (stock R29 versus PR64 overlay, tags stock/pr64) is '
    'supporting evidence only: both images run LMCache source a3a230c8f655749a8d220aabebea5deec4c66497, '
    'which R30 ships natively, but no R30 RAM measurement was performed in this campaign.'
)


def prior_evidence():
    sources = []

    def load(path):
        encoded = path.read_bytes()
        sources.append({'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest()})
        return json.loads(encoded)

    pr64 = campaign.artifacts()
    info = {
        'runtime_image': R30_IMAGE,
        'candidate_tag': 'r30',
        'base_image': R30_IMAGE,
        'preflight': str(PREFLIGHT),
    }
    frozen = load(FROZEN)
    for row in frozen['sources']:
        path = Path(row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f'R30 campaign source changed after freeze: {path.name}')
    components = load(PRIOR / 'gpu-component-summary.json')
    native = components['native_metadata_lifetime']
    if (len(native) != 4 or {row['gpu'] for row in native} != {0, 1, 2, 3}
            or not all(row['passed'] is True and row['returncode'] == 0 for row in native)
            or components['checkpoint_roundtrip_passed'] is not True):
        raise RuntimeError('Prior native GPU qualification was incomplete or failed')
    byte_result = load(PRIOR / 'gpu-checkpoint-bytes/result/result.json')
    if byte_result['passed'] is not True or byte_result['gpu_count'] != 4:
        raise RuntimeError('Prior all-rank byte proof failed')
    reports = byte_result['reports']
    if len(reports) != 2 or {row['schema_version'] for row in reports} != {1, 2}:
        raise RuntimeError('Prior byte proof lacks both manifest schemas')
    for report in reports:
        stages = report['stages']
        if set(stages) != {'ram_first', 'ram_second', 'native_fs_restart'}:
            raise RuntimeError('Prior byte proof lacks required restore stages')
        for rows in stages.values():
            if (len(rows) != 24 or {row['rank'] for row in rows} != {0, 1, 2, 3}
                    or not all(row['matched'] is True for row in rows)):
                raise RuntimeError('Prior byte comparison coverage or equality failed')
    ram = load(PRIOR / 'cache-ram-performance-progress.json')
    if len(ram) != 2 or {row['tag'] for row in ram} != {'stock', 'pr64'}:
        raise RuntimeError('Prior RAM comparison lacks both images')
    for row in ram:
        # The reused RAM rows belong to the PR64-overlay measurement, not to R30.
        expected_image = campaign.STOCK if row['tag'] == 'stock' else pr64['runtime_image']
        label = f'ram-{row["tag"]}-mtp3-dcp4'
        if row['label'] != label or row['image'] != expected_image or row['passed'] is not True:
            raise RuntimeError('Prior RAM arm identity or outcome is invalid')
        if len(row['trials']) != 2:
            raise RuntimeError('Prior RAM arm lacks both repeats')
        for trial, record in enumerate(row['trials'], 1):
            if record['label'] != f'{label}-trial{trial}' or record['client_timing_valid'] is not True:
                raise RuntimeError('Prior RAM repeat failed')
            raw = load(PRIOR / f'{label}-trial{trial}.json')
            if not campaign.ram_trial_valid(raw['results']):
                raise RuntimeError('Prior raw RAM timing is not valid')
    return info, ram, {
        'sources': sources,
        'pr64_overlay_image': pr64['runtime_image'],
        'scope': (
            'Reuse completed native GPU, schema-v1/v2 byte, and 16-cell RAM evidence from the '
            'PR64-overlay campaign as supporting data for R30, whose native LMCache commit '
            'a3a230c8f655749a8d220aabebea5deec4c66497 equals the overlay source_commit. '
            'Do not reuse the aborted lifecycle pilot as a negative control.'
        ),
        'observer_change': 'Use explicitly observed external-hit counter deltas only across one successful isolated request in one counter epoch; missing or contaminated counters remain unavailable.',
        'ram_performance_scope': RAM_PERFORMANCE_SCOPE,
    }


def refuse_unsafe_start():
    inspected = rt._inspect()
    if inspected is not None:
        raise RuntimeError(f'Test container {rt.NAME} still exists; wait for its campaign to finish')
    if subprocess.run(['pgrep', '-f', 'speculative_matrix.py'], capture_output=True, text=True).stdout.strip():
        raise RuntimeError('Speculative matrix is still running; GPUs are not free')
    occupied = subprocess.run(
        ['docker', 'ps', '-a', '--filter', 'name=r29-comparison', '--format', '{{.ID}}'],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout.strip()
    if occupied:
        raise RuntimeError('r29-comparison container still exists; GPUs are not free')


def phase():
    info, ram, proof = prior_evidence()
    rt.save_json('reused-component-and-ram-evidence.json', proof)
    run_id = uuid.uuid4().hex[:12]
    rt.save_json('cache-campaign-plan.json', {
        'candidate': info, 'run_id': run_id, 'l1_gib': 8, 'l2_gib': 8,
        'scope': 'Lifecycle-only stock R30 candidate versus stock R29 negative control in fresh namespaces.',
        'stock_negative_control': campaign.EXACT_PAYLOAD_DEDUP_GATE,
        'stock_control_expectation': 'Stock R29 must complete and FAIL required gate replay.exact_payload_dedup; that completed failure certifies the gate detects missing identical-payload dedup.',
        'candidate_expectation': 'Stock R30 arms dcp{1,4} x {mtp0,mtp3,dflash2} must complete and pass every gate.',
        'mixed_restore': 'Mixed-restore probe runs on the dcp4/mtp3 arms.',
        'ram_performance_scope': RAM_PERFORMANCE_SCOPE,
        'reclamation_limit': 'R30 dedups identical payloads; orphan/cancellation/pressure gates remain real gates and may fail.',
        'production_policy': 'Original container restored unchanged; all L2 namespaces are new and bounded.',
    })
    disks = campaign.lifecycle(info, run_id)
    summary = campaign.build_campaign_summary(ram, disks, candidate_tag='r30')
    summary['ram_performance_scope'] = RAM_PERFORMANCE_SCOPE
    summary['reused_component_and_ram_evidence'] = str(rt.ROOT / 'reused-component-and-ram-evidence.json')
    rt.save_json('cache-campaign-summary.json', summary)
    rt.record_gate('cache-stock-negative-control', summary['stock_negative_control']['valid'], summary['stock_negative_control'])
    rt.record_gate('cache-campaign', summary['passed'], summary)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        prior_evidence()
        refuse_unsafe_start()
        print('R30 CACHE LIFECYCLE CAMPAIGN: guarded coordinator starting', flush=True)
        campaign.coordinator.PHASES = [('r30-cache-lifecycle', str(Path(__file__).resolve()), ['--phase'], 172800)]
        campaign.coordinator.main()
    else:
        raise SystemExit('Usage: r30_cache_campaign.py [--phase]')
