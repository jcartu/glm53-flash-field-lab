#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
R24_ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r24-battery')


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def collision(path: Path) -> dict[str, Any]:
    data = load(path)
    summary = data.get('clean_summary') or data['summary']
    gap_p95 = [
        float(run['decode']['inter_chunk_gap_seconds']['p95']) * 1000
        for run in data['runs']
    ]
    return {
        'runs': summary['runs'],
        'baseline_decode_tps': summary['baseline_decode_tokens_per_second']['mean'],
        'collision_decode_tps': summary['collision_decode_tokens_per_second']['mean'],
        'slowdown_percent': summary['decode_slowdown_percent']['mean'],
        'cold_prefill_tps': summary['prefill_tokens_per_second']['mean'],
        'decode_gap_p95_ms': statistics.fmean(gap_p95),
        'decode_gap_max_ms': summary['max_decode_inter_chunk_gap_seconds']['max'] * 1000,
    }


def corruption(path: Path) -> dict[str, Any]:
    text = path.read_text()
    match = re.search(r'"summary": "(\d+)/(\d+) CLEAN, (\d+) flagged"', text)
    return {
        'clean': int(match.group(1)) if match else 0,
        'total': int(match.group(2)) if match else 0,
        'flagged': int(match.group(3)) if match else None,
    }


def needle(path: Path) -> dict[str, Any]:
    text = path.read_text()
    rows = []
    for match in re.finditer(
        r'depth=(\d+)% prompt_tokens=(\d+) completion=(\d+) wall=([0-9.]+)s needle=(\w+)',
        text,
    ):
        rows.append({
            'depth_percent': int(match.group(1)),
            'prompt_tokens': int(match.group(2)),
            'completion_tokens': int(match.group(3)),
            'wall_seconds': float(match.group(4)),
            'verdict': match.group(5),
        })
    return {
        'results': rows,
        'hits': sum(row['verdict'] == 'HIT' for row in rows),
        'total': len(rows),
    }


r25_configs = load(ROOT / 'summary.json')
r24_report = load(R24_ROOT / 'report-data.json')
r24_configs = r24_report['configs']
r24_by = {row['name']: row for row in r24_configs}
r25_by = {row['name']: row for row in r25_configs}
matched = []
for name in sorted(set(r24_by) & set(r25_by)):
    old = r24_by[name]
    new = r25_by[name]
    deltas = {}
    for metric in ('prefill32', 'C1', 'C4', 'C8', 'C16'):
        if old.get(metric) and new.get(metric) is not None:
            deltas[metric] = 100 * (new[metric] / old[metric] - 1)
    matched.append({'name': name, 'r24': old, 'r25': new, 'delta_percent': deltas})

prefill = {
    label: load(ROOT / f'prefill-{label}.json')
    for label in ('gpu-only', 'lmcache-existing-l2', 'lmcache-ram-only', 'lmcache-fresh-l2')
}
quality = {}
selected = load(ROOT / 'estonia.json')['selected_summary']
quality['estonia'] = {
    'attempted': selected['attempted'],
    'completed': selected['completed'],
    'correct': selected['correct'],
    'wrong': selected['wrong'],
    'errors': selected['errors'],
    'exact': selected['exact'],
    'aggregate_gen_tps': selected['aggregate_gen_tok_s'],
    'aggregate_e2e_tps': selected['aggregate_e2e_tok_s'],
}
quality['lavd'] = load(ROOT / 'lavd-aggregate.json')
unique_needles = load(ROOT / 'needle-unique-1m.json')

report = {
    'provenance': load(ROOT / 'provenance.json'),
    'configs': r25_configs,
    'matched_r24_r25': matched,
    'execution_ab': load(ROOT / 'execution-ab-summary.json'),
    'fp8_recheck': load(ROOT / 'fp8-recheck-summary.json'),
    'fairness': {
        'compute_share_0.4': collision(ROOT / 'collision-dcp1-fair04.json'),
        'off': collision(ROOT / 'collision-dcp1-fair-off.json'),
    },
    'prefix_80k': {
        'lmcache_unique': load(ROOT / 'prefix-unique-80k.json'),
        'shared_l2_cross_version_probe': load(ROOT / 'prefix-lmcache-80k.json'),
        'native': load(ROOT / 'prefix-native-80k.json'),
    },
    'exact_replay_1m': load(ROOT / 'replay-exact-1m.json'),
    'restore_ab': load(ROOT / 'restore-ab-summary.json'),
    'eviction': load(ROOT / 'eviction-r25.json'),
    'prefill_variants': prefill,
    'sidecar': load(ROOT / 'sidecar-check.json'),
    'needles': {
        'results': unique_needles['results'],
        'hits': sum(row['hit'] for row in unique_needles['results']),
        'total': len(unique_needles['results']),
        'shared_l2_cross_version_probe': needle(ROOT / 'needle-1m.log'),
    },
    'corruption': {
        'dcp4_mtp3': corruption(ROOT / 'corruption-dcp4-mtp3.log'),
        'lmcache_dflash': load(ROOT / 'corruption-lmcache-dflash-aggregate.json'),
    },
    'template': {
        'pass': '"verdict": "PASS"' in (ROOT / 'template-probe.log').read_text(),
    },
    'pr646': load(ROOT / 'pr646-summary.json'),
    'quality': quality,
    'hot_queue': load(ROOT / 'hot-queue-summary.json'),
    'r24_special': r24_report['special'],
}
(ROOT / 'report-data.json').write_text(json.dumps(report, indent=2))
print(json.dumps({
    'configs': len(r25_configs),
    'matched_configs': len(matched),
    'needle_hits': report['needles']['hits'],
    'restore_comparison': report['restore_ab']['comparison'],
    'hot_comparisons': report['hot_queue']['comparisons'],
}, indent=2))
