#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

TARGET_TOKENS = 80_000
MODEL = 'GLM-5.3-Flash-NVFP4'
PORT = int(os.environ.get('PORT', '5001'))
CONTAINER = os.environ.get('TEST_CONTAINER', 'r25-replay')
OUT = Path(os.environ.get(
    'OUT',
    '/home/josh/omp-workspace/drock-lmcache/r25-battery/prefix-unique-80k.json',
))
IDENTITY = 'R25-UNIQUE-80K-SILVER-25001'
CACHE_SALT = 'r25-unique-80k-v1'
TOKENIZE_URL = f'http://127.0.0.1:{PORT}/tokenize'
CHAT_URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'


def token_count(content: str) -> int:
    payload = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': content}],
    }).encode()
    request = urllib.request.Request(
        TOKENIZE_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return int(json.loads(response.read())['count'])


prefix = f'Unique R25 LMCache persistence probe {IDENTITY}.\nBEGIN\n'
suffix = '\nEND\nReply with one period.'
base = token_count(prefix + suffix)
repetitions = TARGET_TOKENS - base
for _ in range(6):
    content = prefix + (' x' * repetitions) + suffix
    observed = token_count(content)
    if observed == TARGET_TOKENS:
        break
    repetitions += TARGET_TOKENS - observed
else:
    raise RuntimeError(f'could not calibrate exact 80K prompt: {observed}')

body = json.dumps({
    'model': MODEL,
    'messages': [{'role': 'user', 'content': content}],
    'max_tokens': 1,
    'temperature': 0.0,
    'reasoning_effort': 'low',
    'ignore_eos': True,
    'cache_salt': CACHE_SALT,
}).encode()
rows = []


def request(label: str) -> None:
    req = urllib.request.Request(CHAT_URL, data=body, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.loads(response.read())
    wall = time.perf_counter() - started
    row = {
        'label': label,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
    }
    rows.append(row)
    OUT.write_text(json.dumps({
        'identity': IDENTITY,
        'cache_salt': CACHE_SALT,
        'target_prompt_tokens': TARGET_TOKENS,
        'results': rows,
    }, indent=2))
    print(f'{label}: {wall:.3f}s prompt={row["prompt_tokens"]}', flush=True)


request('cold')
time.sleep(10)
request('warm')
time.sleep(3)
subprocess.run(['docker', 'restart', CONTAINER], stdout=subprocess.DEVNULL, check=True, timeout=900)
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
request('restart')
if any(row['prompt_tokens'] != TARGET_TOKENS for row in rows):
    raise RuntimeError('server prompt count did not equal 80,000')
