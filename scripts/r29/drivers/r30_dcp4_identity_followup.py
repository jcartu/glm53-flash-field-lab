#!/usr/bin/env python3
"""R30 DCP4 identity-gate follow-up with stock R29 controls on the same arms."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import uuid

SCRIPTS = Path('/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r29')
sys.path.insert(0, str(SCRIPTS))
import cache_campaign as campaign

ROOT = campaign.ROOT
sys.path.insert(0, str(ROOT))
import r30_cache_campaign as base

rt = campaign.rt
R30_IMAGE = base.R30_IMAGE
IDENTITY_GATES = ('replay.correct_and_identical', 'restart.correct_and_identical')
FOLLOWUP_ARMS = [
    ('stock', campaign.STOCK, 4, 'mtp0'),
    ('stock', campaign.STOCK, 4, 'dflash2'),
    ('r30', R30_IMAGE, 4, 'mtp0'),
]


def refuse_unsafe_start():
    base.refuse_unsafe_start()
    # The own driver name must not be matched: this process's and the tmux
    # pane's command lines contain it and would abort the guard instantly.
    for pattern in ('r30_cache_campaign.py',):
        if subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout.strip():
            raise RuntimeError(f'Another campaign process is running: {pattern}')


def gate_status(record, name):
    try:
        receipt = json.loads(Path(record['receipt']).read_text())
    except (KeyError, OSError, ValueError, TypeError):
        return None
    rows = [
        row for row in receipt.get('gates', [])
        if isinstance(row, dict) and row.get('name') == name
    ]
    return rows[0].get('status') if len(rows) == 1 else None


def identity_statuses(record):
    return {name: gate_status(record, name) for name in IDENTITY_GATES}


def gates_pass(statuses):
    return all(value == 'pass' for value in statuses.values())


def gates_fail(statuses):
    return any(value == 'fail' for value in statuses.values())


def classify(arm_rows):
    stock = [row for row in arm_rows if row['tag'] == 'stock']
    r30 = [row for row in arm_rows if row['tag'] == 'r30']
    if len(stock) != 2 or len(r30) != 1:
        return 'mixed'
    r30_statuses = r30[0]['identity_gate_statuses']
    if gates_pass(r30_statuses):
        return 'not_reproduced'
    if all(gates_pass(row['identity_gate_statuses']) for row in stock) and gates_fail(r30_statuses):
        return 'r30_specific'
    if any(gates_fail(row['identity_gate_statuses']) for row in stock):
        return 'dcp4_property_both_images'
    return 'mixed'


def phase():
    info, ram, proof = base.prior_evidence()
    rt.save_json('reused-component-and-ram-evidence.json', proof)
    run_id = uuid.uuid4().hex[:12]
    rt.save_json('identity-followup-plan.json', {
        'candidate': info, 'run_id': run_id,
        'arms': [{'tag': tag, 'image': image, 'dcp': dcp, 'spec': spec} for tag, image, dcp, spec in FOLLOWUP_ARMS],
        'scope': 'Diagnostic follow-up: stock R29 controls for dcp4-mtp0/dflash2 versus the R30 dcp4-mtp0 identity-gate repeat. Not a candidate promotion; build_campaign_summary is intentionally not used.',
        'identity_gates': list(IDENTITY_GATES),
        'classification_rule': "r30_specific if both stock arms pass both identity gates and the r30 repeat fails one; dcp4_property_both_images if a stock arm also fails one; not_reproduced if the r30 repeat passes both; mixed otherwise (including unavailable or missing gates).",
        'production_policy': 'Original container restored unchanged; fresh L2 namespaces.',
    })
    disks = campaign.lifecycle(info, run_id, arms=FOLLOWUP_ARMS)
    arm_rows = []
    for record in disks:
        arm_rows.append({
            'label': record.get('label'),
            'tag': record.get('tag'),
            'image': record.get('image'),
            'dcp': record.get('dcp'),
            'spec': record.get('spec'),
            'complete': record.get('complete'),
            'returncode': record.get('returncode'),
            'failed_gates': record.get('failed_gates'),
            'unavailable_gates': record.get('unavailable_gates'),
            'receipt': record.get('receipt'),
            'identity_gate_statuses': identity_statuses(record),
        })
    classification = classify(arm_rows)
    decisive = bool(
        len(arm_rows) == 3 and all(row['complete'] is True for row in arm_rows)
        and classification != 'mixed'
    )
    summary = {
        'arms': arm_rows,
        'classification': classification,
        'decisive': decisive,
        'identity_gates': list(IDENTITY_GATES),
        'scope': 'Diagnostic only; no candidate promotion and no build_campaign_summary verdict.',
        'prior_evidence': str(rt.ROOT / 'reused-component-and-ram-evidence.json'),
    }
    rt.save_json('identity-followup-summary.json', summary)
    rt.record_gate('r30-dcp4-identity-followup', decisive, summary)
    if not decisive:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--phase']:
        phase()
    elif not sys.argv[1:]:
        base.prior_evidence()
        refuse_unsafe_start()
        print('R30 DCP4 IDENTITY FOLLOW-UP: guarded coordinator starting', flush=True)
        campaign.coordinator.PHASES = [('r30-dcp4-identity-followup', str(Path(__file__).resolve()), ['--phase'], 14400)]
        campaign.coordinator.main()
    else:
        raise SystemExit('Usage: r30_dcp4_identity_followup.py [--phase]')
