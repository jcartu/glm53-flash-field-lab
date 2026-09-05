#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


TARGET_TOKENS = 1_000_000
CODE = 'SILVER-CEDAR-25001'
MODEL = 'GLM-5.3-Flash-NVFP4'
PORT = int(os.environ.get('PORT', '5001'))
CONTAINER = os.environ.get('TEST_CONTAINER', 'r25-replay')
OUT = Path(os.environ.get(
    'OUT',
    '/home/josh/omp-workspace/drock-lmcache/r25-battery/replay-exact-1m.json',
))
CACHE_SALT = os.environ.get('CACHE_SALT', 'r25-exact-million-v1')
URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'
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


messages = [{'role': 'user', 'content': suffix}]
fixed_tokens = prompt_token_count(messages)
filler_count = TARGET_TOKENS - fixed_tokens
for _ in range(4):
    content = ' a' * filler_count + suffix
    messages = [{'role': 'user', 'content': content}]
    token_count = prompt_token_count(messages)
    if token_count == TARGET_TOKENS:
        break
    filler_count += TARGET_TOKENS - token_count
else:
    raise RuntimeError(f'could not calibrate prompt: {token_count} tokens')

body = json.dumps({
    'model': MODEL,
    'messages': messages,
    'max_tokens': 64,
    'temperature': 0.0,
    'reasoning_effort': 'low',
    'cache_salt': CACHE_SALT,
}).encode()
rows: list[dict[str, object]] = []


def persist() -> None:
    answers = [str(row['answer']) for row in rows]
    warm = [float(row['wall_seconds']) for row in rows if str(row['label']).startswith('warm-')]
    restart_steady = [
        float(row['wall_seconds'])
        for row in rows
        if str(row['label']) in {'restart-2', 'restart-3'}
    ]
    report = {
        'target_prompt_tokens': TARGET_TOKENS,
        'local_prompt_tokens': token_count,
        'cache_salt': CACHE_SALT,
        'needle_code': CODE,
        'results': rows,
        'summary': {
            'hit_count': sum(bool(row['hit']) for row in rows),
            'exact_output_count': sum(answer == answers[0] for answer in answers) if answers else 0,
            'warm_median_seconds': statistics.median(warm) if warm else None,
            'restart_first_seconds': next((row['wall_seconds'] for row in rows if row['label'] == 'restart-1'), None),
            'restart_steady_median_seconds': statistics.median(restart_steady) if restart_steady else None,
        },
    }
    OUT.write_text(json.dumps(report, indent=2))


def request(label: str) -> None:
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1500) as response:
        result = json.loads(response.read())
    wall = time.perf_counter() - started
    message = result['choices'][0]['message']
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    full_answer = content + reasoning
    row = {
        'label': label,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'hit': CODE in full_answer,
        'content': content,
        'reasoning_content': reasoning,
        'answer': content,
        'finish_reason': result['choices'][0]['finish_reason'],
    }
    rows.append(row)
    persist()
    print(
        f"{label}: {wall:.3f}s prompt={row['prompt_tokens']} "
        f"completion={row['completion_tokens']} hit={row['hit']}",
        flush=True,
    )


request('cold')
time.sleep(15)
for index in range(1, 6):
    request(f'warm-{index}')
time.sleep(5)
subprocess.run(
    ['docker', 'restart', CONTAINER],
    stdout=subprocess.DEVNULL,
    check=True,
    timeout=900,
)
for _ in range(120):
    probe = subprocess.run(
        ['curl', '-fsS', '--max-time', '4', f'http://127.0.0.1:{PORT}/health'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        break
    time.sleep(10)
else:
    raise RuntimeError('server did not become healthy after restart')
for index in range(1, 4):
    request(f'restart-{index}')

if any(int(row['prompt_tokens']) != TARGET_TOKENS for row in rows):
    raise RuntimeError('server token count did not match exact one-million-token target')
if any(not bool(row['hit']) for row in rows):
    raise RuntimeError('one or more replay answers missed the catalog reference')
if len({str(row['answer']) for row in rows}) != 1:
    raise RuntimeError('one or more replay answers differed byte-for-byte')
