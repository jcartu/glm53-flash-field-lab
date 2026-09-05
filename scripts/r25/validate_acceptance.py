#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
EXPECTED_CONFIGS = [
    'dcp1-vram-fp8-mtp0',
    'dcp1-vram-fp8-mtp3',
    'dcp1-vram-fp8-dflash2',
    'dcp2-vram-fp8-mtp0',
    'dcp2-vram-fp8-mtp3',
    'dcp2-vram-fp8-dflash2',
    'dcp4-vram-fp8-mtp0',
    'dcp4-vram-fp8-mtp3',
    'dcp4-vram-fp8-dflash2',
    'dcp4-vram-nvfp4-mtp0',
    'dcp4-vram-nvfp4-mtp3',
    'dcp4-vram-nvfp4-dflash2',
    'dcp4-lmcache-nvfp4-mtp0',
    'dcp4-lmcache-nvfp4-mtp3',
    'dcp4-lmcache-nvfp4-dflash2',
    'dcp4-vram-nvfp4-mtp0-b12xkda',
    'dcp4-vram-nvfp4-mtp0-ranklocal',
    'dcp4-native-nvfp4-mtp0',
]
checks: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: object) -> None:
    checks.append({'name': name, 'passed': bool(passed), 'detail': detail})


def load_json(name: str) -> dict:
    path = ROOT / name
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        check(f'file:{name}', False, str(error))
        return {}


for stem in EXPECTED_CONFIGS:
    data = load_json(f'{stem}.json')
    headline = {
        int(row.get('concurrency'))
        for row in data.get('results', [])
        if row.get('context_tokens') == 0 and row.get('aggregate_tps')
    }
    check(f'matrix:{stem}', headline == {1, 4, 8, 16}, sorted(headline))
execution_ab = load_json('execution-ab-summary.json')
check(
    'execution:matched-r25-r24-r25-bracket',
    execution_ab.get('within_five_percent_all_cells') is True,
    execution_ab.get('r25_vs_r24_percent'),
)
fp8_recheck = load_json('fp8-recheck-summary.json')
check(
    'execution:dcp4-fp8-recheck',
    set(fp8_recheck.get('recheck', {})) == {'mtp0', 'mtp3', 'dflash2'}
    and all(
        all(fp8_recheck['recheck'][spec].get(metric) for metric in ('C1', 'C4', 'C8', 'C16'))
        for spec in ('mtp0', 'mtp3', 'dflash2')
    ),
    fp8_recheck.get('recheck'),
)



for filename in ('collision-dcp1-fair04.json', 'collision-dcp1-fair-off.json'):
    data = load_json(filename)
    check(f'collision:{filename}', data.get('summary', {}).get('runs') == 3, data.get('summary', {}).get('runs'))

prefix = load_json('prefix-unique-80k.json')
prefix_labels = {row.get('label') for row in prefix.get('results', [])}
check(
    'lmcache:unique-80k-cold-warm-restart',
    prefix.get('target_prompt_tokens') == 80_000
    and prefix_labels == {'cold', 'warm', 'restart'}
    and all(row.get('prompt_tokens') == 80_000 for row in prefix.get('results', [])),
    {'target': prefix.get('target_prompt_tokens'), 'labels': sorted(str(label) for label in prefix_labels)},
)
shared_prefix = load_json('prefix-lmcache-80k.json')
check(
    'lmcache:shared-l2-cross-version-80k',
    set(shared_prefix.get('phases', {})) >= {'cold', 'warm', 'restart'},
    shared_prefix.get('phases'),
)
native = load_json('prefix-native-80k.json')
check(
    'native:80k-cold-warm',
    set(native.get('phases', {})) >= {'cold', 'warm'},
    native.get('phases'),
)

mtp_corruption = (ROOT / 'corruption-dcp4-mtp3.log').read_text() if (ROOT / 'corruption-dcp4-mtp3.log').exists() else ''
check('corruption:dcp4-mtp3', '8/8 CLEAN, 0 flagged' in mtp_corruption, mtp_corruption[-200:])
lmcache_corruption = load_json('corruption-lmcache-dflash-aggregate.json')
check(
    'corruption:lmcache-dflash-three-waves',
    lmcache_corruption.get('waves') == 3
    and lmcache_corruption.get('total') == 24
    and lmcache_corruption.get('clean', 0) + lmcache_corruption.get('flagged', 0) == 24,
    {
        'clean': lmcache_corruption.get('clean'),
        'flagged': lmcache_corruption.get('flagged'),
        'flagged_results': lmcache_corruption.get('flagged_results'),
    },
)
check(
    'corruption:lmcache-dflash-all-clean',
    lmcache_corruption.get('flagged') == 0,
    lmcache_corruption.get('flagged_results'),
)

template = (ROOT / 'template-probe.log').read_text() if (ROOT / 'template-probe.log').exists() else ''
check('template:tool-order', '"verdict": "PASS"' in template, template[-300:])
unique_needles = load_json('needle-unique-1m.json')
check(
    'needle:unique-1m-depths',
    len(unique_needles.get('results', [])) == 3
    and sum(row.get('hit', False) for row in unique_needles.get('results', [])) == 3,
    unique_needles,
)
shared_needle = (ROOT / 'needle-1m.log').read_text() if (ROOT / 'needle-1m.log').exists() else ''
check('needle:shared-l2-cross-version', 'SUMMARY: 3/3 needle hits' in shared_needle, shared_needle[-300:])

estonia = load_json('estonia.json').get('selected_summary', {})
check('quality:estonia', estonia.get('correct') == 6 and estonia.get('errors') == 0, estonia)
lavd = load_json('lavd-aggregate.json')
check(
    'quality:lavd',
    lavd.get('attempted') == 16
    and lavd.get('exact') == 16
    and lavd.get('errors') == 0,
    lavd,
)

replay = load_json('replay-exact-1m.json')
replay_summary = replay.get('summary', {})
check(
    'lmcache:exact-1m-replay',
    replay.get('target_prompt_tokens') == 1_000_000
    and replay_summary.get('hit_count') == 9
    and replay_summary.get('exact_output_count') == 9,
    replay_summary,
)
eviction = load_json('eviction-r25.json')
check(
    'lmcache:eviction-byte-identity',
    eviction.get('identical_reply_count') == 40 and eviction.get('mismatch_count') == 0,
    eviction,
)
sidecar = load_json('sidecar-check.json')
check('lmcache:cpu-only-sidecar', sidecar.get('verdict') == 'pass', sidecar)
restore = load_json('restore-ab-summary.json')
arms = restore.get('arms', {})
check(
    'lmcache:matched-r24-r25-restore',
    arms.get('R24', {}).get('max_retrieved_tokens', 0) >= 999_424
    and arms.get('R25', {}).get('max_retrieved_tokens', 0) >= 999_424
    and restore.get('comparison', {}).get('visible_output_identical') is True,
    restore,
)
check(
    'lmcache:r25-no-unpinned-fallback',
    arms.get('R25', {}).get('unpinned_fallback_warnings') == 0,
    arms.get('R25', {}).get('unpinned_fallback_warnings'),
)

for label in ('gpu-only', 'lmcache-existing-l2', 'lmcache-ram-only', 'lmcache-fresh-l2'):
    data = load_json(f'prefill-{label}.json')
    check(f'prefill:{label}', len(data.get('runs', [])) == 8, len(data.get('runs', [])))

for label in ('off', 'compute-share-0.4'):
    data = load_json(f'hot-{label}.json')
    cells = data.get('cells', [])
    complete = (
        len(cells) == 8
        and all(cell.get('summary', {}).get('hot_turns_offered', 0) > 0 for cell in cells)
        and {cell.get('scenario') for cell in cells} == {'baseline', 'periodic-cold'}
        and {cell.get('concurrency') for cell in cells} == {8, 16}
    )
    check(f'hot-queue:{label}', complete, {'cells': len(cells)})
unit_log = (ROOT / 'pr646-unit-tests.log').read_text() if (ROOT / 'pr646-unit-tests.log').exists() else ''
check('pr646:rebased-unit-tests', '46 passed' in unit_log, unit_log[-300:])
pr646 = load_json('pr646-summary.json')
pr646_correctness = pr646.get('correctness', {})
check(
    'pr646:three-arm-field-test',
    set(pr646.get('arms', {})) == {
        'stock-fine-256',
        'stock-coarse-2048',
        'pr646-decoupled-2048-256',
    }
    and pr646.get('comparison', {}).get('capacity_gain_vs_fine_256', 0) > 2.5
    and abs(pr646.get('comparison', {}).get('capacity_change_vs_stock_coarse', 100)) < 0.1
    and pr646.get('comparison', {}).get('mean_absolute_throughput_delta_percent', 100) < 6
    and pr646_correctness.get('prefix_outputs_identical_all_arms') is True
    and pr646_correctness.get('needle_90k_hits') == 4
    and pr646_correctness.get('estonia_total') == 12
    and pr646_correctness.get('estonia_errors') == 0
    and pr646_correctness.get('stock_estonia_total') == 12
    and pr646_correctness.get('stock_estonia_errors') == 0,
    {
        'arms': sorted(pr646.get('arms', {})),
        'correctness': pr646_correctness,
        'comparison': pr646.get('comparison'),
    },
)
check(
    'pr646:concurrent-estonia-parity',
    pr646_correctness.get('estonia_correct')
    == pr646_correctness.get('stock_estonia_correct'),
    {
        'patched': (
            pr646_correctness.get('estonia_correct'),
            pr646_correctness.get('estonia_total'),
        ),
        'stock': (
            pr646_correctness.get('stock_estonia_correct'),
            pr646_correctness.get('stock_estonia_total'),
        ),
    },
)


failed = [item for item in checks if not item['passed']]
report = {
    'passed': not failed,
    'checks_passed': len(checks) - len(failed),
    'checks_total': len(checks),
    'failed': failed,
    'checks': checks,
}
(ROOT / 'acceptance.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if failed:
    raise SystemExit(1)
