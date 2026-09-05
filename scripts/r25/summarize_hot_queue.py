#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
FILES = {
    'off': ROOT / 'hot-off.json',
    'compute-share-0.4': ROOT / 'hot-compute-share-0.4.json',
}


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
for policy, path in FILES.items():
    data = json.loads(path.read_text())
    for cell in data['cells']:
        grouped[(policy, int(cell['concurrency']), cell['scenario'])].append(cell)

rows = []
for (policy, concurrency, scenario), cells in sorted(grouped.items()):
    summaries = [cell['summary'] for cell in cells]
    duration = float(cells[0]['measurement_seconds'])
    rows.append({
        'policy': policy,
        'concurrency': concurrency,
        'scenario': scenario,
        'repeats': len(cells),
        'measurement_seconds': duration,
        'hot_turns_finished_per_second': mean([
            summary['hot_turns_finished_within_window'] / duration
            for summary in summaries
        ]),
        'hot_output_tokens_per_second': mean([
            summary['hot_output_tokens_per_second'] for summary in summaries
        ]),
        'hot_ttft_p95_seconds': mean([
            summary['hot_ttft_seconds']['p95'] for summary in summaries
        ]),
        'hot_ttft_while_cold_pending_p95_seconds': mean([
            summary['hot_ttft_while_cold_pending_seconds']['p95']
            for summary in summaries
        ]),
        'hot_ttft_no_cold_pending_p95_seconds': mean([
            summary['hot_ttft_when_no_cold_pending_seconds']['p95']
            for summary in summaries
        ]),
        'late_hot_turns': mean([
            summary['late_hot_turns'] for summary in summaries
        ]),
        'incremental_prefill_tokens_median': mean([
            summary['small_incremental_prefill_tokens']['median']
            for summary in summaries
        ]),
        'max_waiting_requests': mean([
            summary['scheduler']['max_waiting'] for summary in summaries
        ]),
        'mean_waiting_requests': mean([
            summary['scheduler']['mean_waiting'] for summary in summaries
        ]),
        'mean_cold_requests_pending': mean([
            summary['scheduler']['mean_cold_requests_pending']
            for summary in summaries
        ]),
        'cold_requests_finished_within_window': mean([
            summary['cold_requests_finished_within_window']
            for summary in summaries
        ]),
        'cold_drain_seconds_after_window': mean([
            summary['cold_drain_seconds_after_window'] for summary in summaries
        ]),
        'total_useful_tokens_per_second': mean([
            summary['total_useful_tokens_per_second'] for summary in summaries
        ]),
    })

lookup = {
    (row['policy'], row['concurrency'], row['scenario']): row
    for row in rows
}
comparisons = []
for policy in FILES:
    for concurrency in (8, 16):
        baseline = lookup[(policy, concurrency, 'baseline')]
        storm = lookup[(policy, concurrency, 'periodic-cold')]
        comparisons.append({
            'policy': policy,
            'concurrency': concurrency,
            'hot_turn_rate_retention_percent': (
                100 * storm['hot_turns_finished_per_second'] / baseline['hot_turns_finished_per_second']
            ),
            'hot_output_retention_percent': (
                100 * storm['hot_output_tokens_per_second'] / baseline['hot_output_tokens_per_second']
            ),
            'hot_ttft_p95_multiplier': (
                storm['hot_ttft_p95_seconds'] / baseline['hot_ttft_p95_seconds']
            ),
            'contended_vs_clear_hot_ttft_p95_multiplier': (
                storm['hot_ttft_while_cold_pending_p95_seconds']
                / storm['hot_ttft_no_cold_pending_p95_seconds']
                if storm['hot_ttft_no_cold_pending_p95_seconds']
                and storm['hot_ttft_while_cold_pending_p95_seconds']
                else None
            ),
            'max_waiting_requests': storm['max_waiting_requests'],
            'cold_drain_seconds_after_window': storm['cold_drain_seconds_after_window'],
        })

report = {
    'question': 'Do cyclic agent-session small prefills queue behind periodic large cold prefills?',
    'scheduler_scope': 'R25 compute_share protects decode-vs-prefill execution share; it does not prioritize or interleave queued prefill jobs.',
    'rows': rows,
    'comparisons': comparisons,
}
(ROOT / 'hot-queue-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
