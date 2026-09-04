#!/usr/bin/env python3
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

CODE = 'SILVER-CEDAR-90025'
FACT = f'\n\nThe museum catalog lists specimen Sigma-9 under catalog reference {CODE}.\n\n'
QUESTION = 'According to the manuscript, what catalog reference belongs to specimen Sigma-9? Answer with only the reference.'
PARA = (
    'The history of computational linguistics intertwines with the development '
    'of formal grammar, statistical methods, and neural architectures. Early '
    'systems relied on handcrafted rules, while modern approaches learn '
    'representations from vast corpora. Each era brought new abstractions: '
    'from context-free grammars to embedding spaces, from n-gram statistics '
    'to attention mechanisms. Understanding this progression clarifies why '
    'current models balance symbolic structure with distributed semantics. '
)
reps = 5_950_000 // len(PARA)
doc = PARA * reps
prompt = doc[: len(doc) // 2] + FACT + doc[len(doc) // 2 :] + '\n\n' + QUESTION
body = json.dumps({
    'model': 'GLM-5.3-Flash-NVFP4',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 256,
    'temperature': 0.0,
    'reasoning_effort': 'low',
}).encode()
URL = os.environ.get('URL', 'http://127.0.0.1:5001/v1/chat/completions')
OUT = Path(os.environ.get(
    'OUT',
    '/home/josh/omp-workspace/drock-lmcache/r24-battery/replay-r25-1m.json',
))
CONTAINER = os.environ.get('CONTAINER', 'r24-replay')
RUN_LABEL = os.environ.get('RUN_LABEL', 'R25')
rows = []


def request(label: str) -> None:
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    start = time.time()
    with urllib.request.urlopen(req, timeout=1500) as response:
        result = json.loads(response.read())
    message = result['choices'][0]['message']
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    row = {
        'label': label,
        'wall_seconds': time.time() - start,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'hit': CODE in content + reasoning,
        'content': content,
        'finish_reason': result['choices'][0]['finish_reason'],
    }
    rows.append(row)
    OUT.write_text(json.dumps({
        'run_label': RUN_LABEL,
        'needle_code': CODE,
        'results': rows,
    }, indent=2))
    print(f"{label}: {row['wall_seconds']:.3f}s tokens={row['prompt_tokens']} hit={row['hit']}", flush=True)


request('cold')
for i in range(1, 6):
    request(f'warm-{i}')
subprocess.run(['docker', 'restart', CONTAINER], stdout=subprocess.DEVNULL, check=True, timeout=900)
for _ in range(90):
    probe = subprocess.run(['curl', '-fsS', '--max-time', '4', 'http://127.0.0.1:5001/health'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        break
    time.sleep(10)
for i in range(1, 4):
    request(f'restart-{i}')
