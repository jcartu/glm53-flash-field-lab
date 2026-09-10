#!/usr/bin/env python3
"""Continue unexecuted disk-cache arms, preserving completed GPU/RAM evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import uuid

SCRIPTS = Path('/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r29')
sys.path.insert(0, str(SCRIPTS))
import cache_campaign as campaign

ROOT = campaign.ROOT
PRIOR = ROOT / 'cache-campaign'
rt = campaign.rt


def prior_evidence():
    sources = []

    def load(path):
        encoded = path.read_bytes()
        sources.append({'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest()})
        return json.loads(encoded)

    info = campaign.artifacts()
    frozen = load(ROOT / 'source-qualified-before-gpu/manifest.json')
    for name in ('cache_campaign.py', 'gpu_checkpoint_roundtrip.py'):
        path = SCRIPTS / name
        expected = next(row for row in frozen['sources'] if row['path'] == str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise RuntimeError(f'Reused GPU/RAM experiment source changed: {name}')
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
        expected_image = campaign.STOCK if row['tag'] == 'stock' else info['runtime_image']
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
        'scope': 'Reuse completed native GPU, schema-v1/v2 byte, and 16-cell RAM evidence. Do not reuse the aborted lifecycle pilot as a negative control.',
        'observer_change': 'Use explicitly observed external-hit counter deltas only across one successful isolated request in one counter epoch; missing or contaminated counters remain unavailable.',
    }


def phase():
    info, ram, proof = prior_evidence()
    rt.save_json('reused-component-and-ram-evidence.json', proof)
    run_id = uuid.uuid4().hex[:12]
    rt.save_json('cache-campaign-plan.json', {
        'candidate': info, 'run_id': run_id, 'l1_gib': 8, 'l2_gib': 8,
        'scope': 'Lifecycle-only continuation in fresh namespaces after observer correction.',
        'stock_negative_control': campaign.EXACT_PAYLOAD_DEDUP_GATE,
        'reclamation_limit': 'PR64 has no reference-counted payload reclamation; failures remain required gates.',
        'production_policy': 'Original container restored unchanged; no promotion.',
    })
    disks = campaign.lifecycle(info, run_id)
    summary = campaign.build_campaign_summary(ram, disks)
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
        if rt._inspect() is not None:
            raise RuntimeError('Another test container exists; wait for its campaign to finish')
        print('R29 CACHE LIFECYCLE CONTINUATION: guarded coordinator starting', flush=True)
        campaign.coordinator.PHASES = [('cache-lifecycle', str(Path(__file__).resolve()), ['--phase'], 172800)]
        campaign.coordinator.main()
    else:
        raise SystemExit('Usage: resume_cache_lifecycle.py [--phase]')
