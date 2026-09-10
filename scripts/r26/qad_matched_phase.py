#!/usr/bin/env python3
"""Same image, same rig, same prompts: published NVFP4 weights versus the QAD checkpoint."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import quality_phase as quality
import runtime as rt

PUBLISHED = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4')
CANDIDATE = rt.MODEL
ACCEPTANCE = Path(__file__).with_name('acceptance_probe.py')
CONFIGS = [
    ('dcp1-dflash', dict(dcp=1, spec='dflash2', kv='fp8_ds_mla'), {}),
    ('dcp4-mtp3', dict(dcp=4, spec='mtp3', kv='fp8_ds_mla'), {'VLLM_GLM53_MTP_DRAFT_HEAD': 'nvfp4'}),
]


def main() -> None:
    if CANDIDATE.resolve() == PUBLISHED.resolve():
        raise SystemExit('BATTERY_MODEL_DIR must point at the candidate checkpoint, not the published weights')
    ledger = {'published': str(PUBLISHED), 'candidate': str(CANDIDATE), 'image': rt.IMAGE, 'cells': []}
    for config_name, config, head_env in CONFIGS:
        # Alternate the order per config so neither checkpoint always runs first.
        order = [('published', PUBLISHED), ('candidate', CANDIDATE)]
        if config_name.startswith('dcp4'):
            order.reverse()
        for weights_name, weights in order:
            label = f'qad-matched-{config_name}-{weights_name}'
            cell = {'label': label, 'config': config_name, 'weights': weights_name, 'model_dir': str(weights), 'receipts': []}
            try:
                if not rt.boot(label, image=rt.IMAGE, extra_env=head_env or None, model=weights, **config):
                    cell['booted'] = False
                    continue
                cell['booted'] = True
                for profile_name in ('estonia', 'lavd-test'):
                    plan = quality.ProfilePlan(f'{label}-{profile_name}', profile_name, 24)
                    quality.run_profile(rt, plan)
                    cell['receipts'].append(plan.label + '.json')
                default_label = f'{label}-lavd-template-default'
                rt.profile(default_label, 'lavd-test', 8)
                cell['receipts'].append(default_label + '.json')
                cell['reasoning_scopes'] = {
                    'verified_profiles': 'explicit low; matched across checkpoints',
                    'lavd_template_default': 'same unchanged checkpoint template default; never pooled with low',
                }
                long_plan = quality.ProbePlan(f'{label}-long', 'long', 3600, ('--waves', '3', '--max-tokens', '8192', '--seed-base', '26090800'))
                quality.run_probe(rt, long_plan)
                cell['receipts'].append(long_plan.label + '.json')
                if config['spec'] != 'mtp0':
                    code = rt.run([sys.executable, str(ACCEPTANCE), '--label', label, '--contexts', '0,32768', '--concurrency', '1,8', '--duration', '30', '--repeats', '2'], label=label + '-acceptance-run', timeout=1800)
                    cell['receipts'].append(label + '-acceptance.json')
                    cell['acceptance_returncode'] = code
            except Exception as error:
                cell['error'] = repr(error)
                rt.record_gate('qad-matched:' + label, False, cell)
            finally:
                rt.capture(label)
                rt.stop()
                ledger['cells'].append(cell)
                rt.save_json('qad-matched-ledger.json', ledger)
    rt.save_json('qad-matched-completed.json', {'complete': True, 'cells': len(ledger['cells']), 'note': 'Same R26 image and prompts; only the mounted checkpoint differs. Quality tallies are observations with small samples, not fidelity proofs.'})


if __name__ == '__main__':
    main()
