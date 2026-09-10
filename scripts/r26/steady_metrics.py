#!/usr/bin/env python3
"""Read-only counter sampling around the unchanged sustained-decode benchmark."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

COUNTERS = {
    'generated_tokens': 'vllm:generation_tokens_total',
    'verifier_request_steps': 'vllm:spec_decode_num_drafts_total',
    'proposed_tokens': 'vllm:spec_decode_num_draft_tokens_total',
    'accepted_tokens': 'vllm:spec_decode_num_accepted_tokens_total',
}
EVENT = re.compile(r'^(\d\d:\d\d:\d\d) cell (start|done) C=(\d+) ctx=([^ ]+)')


def parse_counters(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    names = {value: key for key, value in COUNTERS.items()}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in names:
            key = names[name]
            values[key] = values.get(key, 0.0) + float(line.split()[-1])
    return values


class Recorder:
    def __init__(self, base_url: str, output: Path):
        self.base_url = base_url
        self.output = output
        self.finished = threading.Event()
        self.thread: threading.Thread | None = None

    def _collect(self) -> None:
        session = requests.Session()
        session.trust_env = False
        try:
            with self.output.open('w') as output:
                while not self.finished.is_set():
                    start = time.time()
                    try:
                        response = session.get(self.base_url + '/metrics', timeout=3)
                        response.raise_for_status()
                        end = time.time()
                        row = {'timestamp': (start + end) / 2, 'request_started_at': start,
                               'request_finished_at': end, 'counters': parse_counters(response.text)}
                    except Exception as error:
                        row = {'timestamp': time.time(), 'error': repr(error)}
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    self.finished.wait(0.5)
        finally:
            session.close()

    def __enter__(self) -> 'Recorder':
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def context_tokens(text: str) -> int:
    text = text.lower().replace(',', '')
    return int(float(text[:-1]) * 1024) if text.endswith('k') else int(text)


def summarize(root: Path, label: str) -> dict:
    benchmark = json.loads((root / f'{label}.json').read_text())
    command = json.loads((root / f'{label}.bench.command.json').read_text())
    samples = [json.loads(line) for line in (root / f'{label}.steady.metrics.jsonl').read_text().splitlines() if line.strip()]
    zone = ZoneInfo('Europe/Berlin')
    day = datetime.fromtimestamp(command['started_at'], zone).date()
    previous = command['started_at'] - 2
    open_cells: dict[tuple[int, int], float] = {}
    cells: dict[tuple[int, int], tuple[float, float]] = {}
    for event in benchmark.get('event_log', []):
        match = EVENT.match(event)
        if match is None:
            continue
        clock, kind, concurrency, context = match.groups()
        instant = datetime.combine(day, datetime.strptime(clock, '%H:%M:%S').time(), zone)
        while instant.timestamp() < previous - 2:
            instant += timedelta(days=1)
            day = instant.date()
        timestamp = instant.timestamp()
        previous = timestamp
        key = (int(concurrency), context_tokens(context))
        if kind == 'start':
            open_cells[key] = timestamp
        elif key in open_cells:
            # A short warmup may have the same key; the final matching cell
            # replaces it, rather than becoming an extra measurement.
            cells[key] = (open_cells.pop(key), timestamp)
    results = []
    for result in benchmark.get('results', []):
        key = (int(result['concurrency']), int(result['context_tokens']))
        row = {'concurrency': key[0], 'context_tokens': key[1], 'valid': False}
        if key not in cells:
            row['error'] = 'No matching start/done events'
            results.append(row)
            continue
        cell_start, cell_end = cells[key]
        warmup = float(result.get('warmup_duration', 0))
        duration = float(result.get('measurement_wall_seconds', result.get('measurement_seconds', 0)))
        # Event timestamps are only second-resolution. Exclude two seconds
        # at each boundary, selecting only the interior steady-decode window.
        start = cell_start + warmup + 2
        end = min(cell_start + warmup + duration - 2, cell_end - 1)
        selected = [sample for sample in samples if start <= sample['timestamp'] <= end]
        if len(selected) < 3 or any('error' in sample for sample in selected):
            row['error'] = 'Insufficient successful metric coverage in the steady interval'
            results.append(row)
            continue
        if selected[0]['timestamp'] - start > 2 or end - selected[-1]['timestamp'] > 2 or any(b['timestamp'] - a['timestamp'] > 2 for a, b in zip(selected, selected[1:])):
            row['error'] = 'Metric coverage gap'
            results.append(row)
            continue
        first, last = selected[0], selected[-1]
        elapsed = last['timestamp'] - first['timestamp']
        common = first['counters'].keys() & last['counters'].keys()
        delta = {name: last['counters'][name] - first['counters'][name] for name in common}
        if elapsed <= 0 or any(value < 0 for value in delta.values()):
            row['error'] = 'Invalid or reset counter window'
            results.append(row)
            continue
        row.update({'valid': True, 'started_at': first['timestamp'], 'finished_at': last['timestamp'],
                    'seconds': elapsed, 'counter_delta': delta,
                    'output_tokens_per_second': delta.get('generated_tokens', 0) / elapsed})
        steps = delta.get('verifier_request_steps', 0)
        proposed = delta.get('proposed_tokens', 0)
        if steps > 0 and proposed > 0:
            row.update({'aggregate_verifier_steps_per_second': steps / elapsed,
                        'acceptance_fraction': delta['accepted_tokens'] / proposed,
                        'accepted_draft_tokens_per_step': delta['accepted_tokens'] / steps,
                        'emitted_tokens_per_verifier_step': delta['generated_tokens'] / steps})
        else:
            row['speculation_counters_applicable'] = False
        results.append(row)
    summary = {'schema': 'r26-steady-counters/v1', 'label': label,
               'method': 'Read-only 0.5s Prometheus sampling during the unchanged benchmark; two-second guards inside second-resolution cell events plus measured warmup.',
               'step_units': 'Aggregate per-request speculative verifier steps, not physical batched GPU kernel launches.',
               'source_benchmark': str(root / f'{label}.json'), 'source_samples': str(root / f'{label}.steady.metrics.jsonl'),
               'cells': results, 'all_windows_valid': bool(results) and all(row['valid'] for row in results)}
    path = root / f'{label}.steady-summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary
