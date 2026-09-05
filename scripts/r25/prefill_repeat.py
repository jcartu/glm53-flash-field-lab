#!/usr/bin/env python3
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PORT = int(sys.argv[1])
LABEL = sys.argv[2]
OUT = Path(sys.argv[3])
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MODEL = 'GLM-5.3-Flash-NVFP4'
PARA = (
    'Distributed systems must preserve correctness while components fail independently. '
    'Replication, consensus, backpressure, idempotency, and observability turn partial '
    'failure into behavior operators can understand. '
)
prompt = (PARA * (202_500 // len(PARA) + 1))[:202_500] + '\n\nSummarize in one word.'
rows = []
for index in range(RUNS):
    body = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 1,
        'temperature': 0.0,
        'reasoning_effort': 'low',
        'cache_salt': f'{LABEL}-{uuid.uuid4()}',
    }).encode()
    request = urllib.request.Request(
        f'http://127.0.0.1:{PORT}/v1/chat/completions',
        data=body,
        headers={'Content-Type': 'application/json'},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    wall = time.perf_counter() - start
    tokens = result['usage']['prompt_tokens']
    row = {
        'run': index + 1,
        'prompt_tokens': tokens,
        'seconds': wall,
        'tokens_per_second': tokens / wall,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
summary = {
    'label': LABEL,
    'runs': rows,
    'median_tokens_per_second': statistics.median(row['tokens_per_second'] for row in rows),
    'aggregate_tokens_per_second': sum(row['prompt_tokens'] for row in rows) / sum(row['seconds'] for row in rows),
}
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
