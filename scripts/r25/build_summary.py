#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
CONFIG_PATTERN = re.compile(r'^dcp[124]-(?:vram|lmcache|native)-(?:fp8|nvfp4)-(?:mtp0|mtp3|dflash2)(?:-b12xkda|-ranklocal)?$')
rows = []
for path in sorted(ROOT.glob('*.json')):
    if not CONFIG_PATTERN.match(path.stem):
        continue
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    results = data.get('results')
    if not isinstance(results, list):
        continue
    headline = {}
    c1_accept = None
    c1_server_gen = None
    for result in results:
        if result.get('context_tokens') != 0:
            continue
        concurrency = result.get('concurrency')
        if concurrency in {1, 4, 8, 16}:
            headline[f'C{concurrency}'] = result.get('aggregate_tps')
        if concurrency == 1:
            accept = result.get('server_spec_accept_rate')
            c1_accept = accept * 100 if isinstance(accept, (int, float)) else None
            c1_server_gen = result.get('server_gen_throughput')
    prefill = data.get('prefill', {}).get('32768', {})
    prefill32 = prefill.get('client_tok_per_sec') or prefill.get('tok_per_sec')
    kv_tokens = None
    for event in data.get('event_log', []):
        match = re.search(r'KV cache budget from vLLM metrics: ([\d,]+) tokens', str(event))
        if match:
            kv_tokens = int(match.group(1).replace(',', ''))
            break
    rows.append({
        'name': path.stem,
        'prefill32': prefill32,
        **{key: headline.get(key) for key in ('C1', 'C4', 'C8', 'C16')},
        'kv_tokens': kv_tokens,
        'c1_spec_accept_percent': c1_accept,
        'c1_server_gen_tps': c1_server_gen,
    })
(ROOT / 'summary.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
