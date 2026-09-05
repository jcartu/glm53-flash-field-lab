#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

TARGET_TOKENS = 52_445
MODEL = 'GLM-5.3-Flash-NVFP4'
CODE = 'COPPER-LANTERN-64625'
LABEL = sys.argv[1]
OUT = Path(sys.argv[2])
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 5001
TOKENIZE_URL = f'http://127.0.0.1:{PORT}/tokenize'
CHAT_URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'
METRICS_URL = f'http://127.0.0.1:{PORT}/metrics'


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def token_count(content: str) -> int:
    return int(post_json(TOKENIZE_URL, {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': content}],
    })['count'])


def metrics() -> dict[str, float]:
    with urllib.request.urlopen(METRICS_URL, timeout=30) as response:
        text = response.read().decode()
    wanted = {
        'vllm:prefix_cache_hits_total',
        'vllm:prompt_tokens_by_source_total',
    }
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(None, 1)[0]
        if name not in wanted:
            continue
        key = name
        if name == 'vllm:prompt_tokens_by_source_total':
            if 'source="local_cache_hit"' in line:
                key += ':local_cache_hit'
            elif 'source="local_compute"' in line:
                key += ':local_compute'
            else:
                continue
        result[key] = result.get(key, 0.0) + float(line.rsplit(None, 1)[1])
    return result


prefix = f'PR646 shared-prefix probe {LABEL}.\nBEGIN DOCUMENT\n'
suffix = (
    f'\nEND DOCUMENT\nThe catalog code is {CODE}. '
    'Reply with only that catalog code.'
)
base_tokens = token_count(prefix + suffix)
repetitions = TARGET_TOKENS - base_tokens
for _ in range(6):
    content = prefix + (' x' * repetitions) + suffix
    observed = token_count(content)
    if observed == TARGET_TOKENS:
        break
    repetitions += TARGET_TOKENS - observed
else:
    raise RuntimeError(f'could not build exact {TARGET_TOKENS}-token prompt: {observed}')

payload = {
    'model': MODEL,
    'messages': [{'role': 'user', 'content': content}],
    'max_tokens': 64,
    'temperature': 0.0,
    'reasoning_effort': 'low',
    'cache_salt': f'pr646-{LABEL}-v1',
}


def request(label: str) -> dict:
    started = time.perf_counter()
    result = post_json(CHAT_URL, payload, timeout=900)
    wall = time.perf_counter() - started
    message = result['choices'][0]['message']
    visible = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    return {
        'label': label,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'hit': CODE in visible + reasoning,
        'content': visible,
        'reasoning_content': reasoning,
    }


before = metrics()
cold = request('cold')
time.sleep(5)
mid = metrics()
repeat = request('repeat')
after = metrics()
metric_keys = set(before) | set(mid) | set(after)
report = {
    'label': LABEL,
    'target_prompt_tokens': TARGET_TOKENS,
    'results': [cold, repeat],
    'cold_metric_delta': {key: mid.get(key, 0.0) - before.get(key, 0.0) for key in metric_keys},
    'repeat_metric_delta': {key: after.get(key, 0.0) - mid.get(key, 0.0) for key in metric_keys},
    'outputs_identical': cold['content'] == repeat['content'],
}
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
if cold['prompt_tokens'] != TARGET_TOKENS or repeat['prompt_tokens'] != TARGET_TOKENS:
    raise RuntimeError('server prompt token count differs from target')
if not cold['hit'] or not repeat['hit'] or not report['outputs_identical']:
    raise RuntimeError('prefix replay output correctness failed')
