#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

MODEL = 'GLM-5.3-Flash-NVFP4'
PORT = 5001
OUT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery/needle-unique-1m.json')
URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'
PARA = (
    'The history of computational linguistics intertwines with formal grammar, '
    'statistical methods, and neural architectures. Early systems used handcrafted '
    'rules, while modern approaches learn representations from large corpora. '
)
base_document = PARA * (5_950_000 // len(PARA))
rows = []
for depth_percent in (10, 50, 90):
    identity = f'R25-UNIQUE-DEPTH-{depth_percent}-25001'
    code = f'R25-NEEDLE-{depth_percent}-SILVER'
    fact = f'\nThe archive reference for specimen Sigma-{depth_percent} is {code}.\n'
    prefix = f'Independent R25 cold needle document {identity}.\n'
    position = int(len(base_document) * depth_percent / 100)
    content = (
        prefix
        + base_document[:position]
        + fact
        + base_document[position:]
        + f'\nWhat is the archive reference for specimen Sigma-{depth_percent}? Reply with only the reference.'
    )
    payload = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': 256,
        'temperature': 0.0,
        'reasoning_effort': 'low',
        'cache_salt': f'r25-unique-needle-{depth_percent}-v1',
    }).encode()
    request = urllib.request.Request(URL, data=payload, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1500) as response:
        result = json.loads(response.read())
    wall = time.perf_counter() - started
    message = result['choices'][0]['message']
    visible = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    row = {
        'depth_percent': depth_percent,
        'identity': identity,
        'needle_code': code,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'hit': code in visible + reasoning,
        'content': visible,
        'reasoning_content': reasoning,
    }
    rows.append(row)
    OUT.write_text(json.dumps({'results': rows}, indent=2))
    print(json.dumps(row), flush=True)
if sum(row['hit'] for row in rows) != 3:
    raise RuntimeError('one or more unique million-token needles missed')
