#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path


TARGET_TOKENS = 1_000_000
CODE = 'SILVER-CEDAR-25001'
MODEL = 'GLM-5.3-Flash-NVFP4'
LABEL = sys.argv[1]
OUT = Path(sys.argv[2])
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 5001

TOKENIZE_URL = f'http://127.0.0.1:{PORT}/tokenize'
suffix = (
    f'\nThe museum catalog reference for specimen Sigma-9 is {CODE}. '
    'Reply with only that reference.'
)


def prompt_token_count(chat_messages: list[dict[str, str]]) -> int:
    payload = json.dumps({
        'model': MODEL,
        'messages': chat_messages,
    }).encode()
    request = urllib.request.Request(
        TOKENIZE_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return int(json.loads(response.read())['count'])


fixed = prompt_token_count([{'role': 'user', 'content': suffix}])
filler_count = TARGET_TOKENS - fixed
for _ in range(4):
    content = ' a' * filler_count + suffix
    messages = [{'role': 'user', 'content': content}]
    count = prompt_token_count(messages)
    if count == TARGET_TOKENS:
        break
    filler_count += TARGET_TOKENS - count
else:
    raise RuntimeError(f'could not calibrate exact prompt: {count}')

body = json.dumps({
    'model': MODEL,
    'messages': messages,
    'max_tokens': 64,
    'temperature': 0.0,
    'reasoning_effort': 'low',
    'cache_salt': 'r25-exact-million-v1',
}).encode()
request = urllib.request.Request(
    f'http://127.0.0.1:{PORT}/v1/chat/completions',
    data=body,
    headers={'Content-Type': 'application/json'},
)
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=1500) as response:
    result = json.loads(response.read())
wall = time.perf_counter() - started
message = result['choices'][0]['message']
visible = message.get('content') or ''
reasoning = message.get('reasoning_content') or ''
row = {
    'label': LABEL,
    'wall_seconds': wall,
    'local_prompt_tokens': count,
    'server_prompt_tokens': result['usage']['prompt_tokens'],
    'completion_tokens': result['usage']['completion_tokens'],
    'needle_hit': CODE in visible + reasoning,
    'content': visible,
    'reasoning_content': reasoning,
}
OUT.write_text(json.dumps(row, indent=2))
print(json.dumps(row), flush=True)
if count != TARGET_TOKENS or row['server_prompt_tokens'] != TARGET_TOKENS:
    raise RuntimeError('prompt was not exactly one million tokens')
if not row['needle_hit']:
    raise RuntimeError('catalog reference missing from response')
