#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

TARGET_TOKENS = 90_000
MODEL = 'GLM-5.3-Flash-NVFP4'
CODE = 'VIOLET-COMET-64690'
OUT = Path(sys.argv[1])
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5001
TOKENIZE_URL = f'http://127.0.0.1:{PORT}/tokenize'
CHAT_URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def count(content: str) -> int:
    return int(post(TOKENIZE_URL, {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': content}],
    }, timeout=300)['count'])


prefix = 'PR646 exact 90K cached-needle probe.\nBEGIN MANUSCRIPT\n'
fact = f'\nThe observatory registry code is {CODE}.\n'
suffix = '\nEND MANUSCRIPT\nWhat is the observatory registry code? Reply with only the code.'
base = count(prefix + fact + suffix)
repetitions = TARGET_TOKENS - base
for _ in range(6):
    split = int(repetitions * 0.90)
    content = prefix + (' x' * split) + fact + (' x' * (repetitions - split)) + suffix
    observed = count(content)
    if observed == TARGET_TOKENS:
        break
    repetitions += TARGET_TOKENS - observed
else:
    raise RuntimeError(f'could not calibrate exact 90K prompt: {observed}')

payload = {
    'model': MODEL,
    'messages': [{'role': 'user', 'content': content}],
    'max_tokens': 256,
    'temperature': 0.0,
    'reasoning_effort': 'low',
    'cache_salt': 'pr646-needle-90k-v1',
}
rows = []
for index in range(4):
    started = time.perf_counter()
    result = post(CHAT_URL, payload)
    wall = time.perf_counter() - started
    message = result['choices'][0]['message']
    visible = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    row = {
        'run': index + 1,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'hit': CODE in visible + reasoning,
        'content': visible,
        'reasoning_content': reasoning,
    }
    rows.append(row)
    OUT.write_text(json.dumps({'target_prompt_tokens': TARGET_TOKENS, 'results': rows}, indent=2))
    print(json.dumps(row), flush=True)
    time.sleep(3)
if any(row['prompt_tokens'] != TARGET_TOKENS for row in rows):
    raise RuntimeError('server prompt count did not equal 90,000')
if sum(row['hit'] for row in rows) != 4:
    raise RuntimeError('one or more 90K needle runs missed')
