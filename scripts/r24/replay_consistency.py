#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.request
from pathlib import Path

CODE = 'ZEBRA-QUASAR-82413'
NEEDLE = f'\n\nIMPORTANT REGISTRY NOTE: The activation code for the Andromeda relay station is {CODE}. Memorize it exactly.\n\n'
QUESTION = 'What is the activation code for the Andromeda relay station? Answer with just the code.'
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
pos = len(doc) // 2
prompt = doc[:pos] + NEEDLE + doc[pos:] + '\n\n' + QUESTION
body = json.dumps({
    'model': 'GLM-5.3-Flash-NVFP4',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 512,
    'temperature': 0.0,
    'reasoning_effort': 'low',
}).encode()
URL = 'http://127.0.0.1:5001/v1/chat/completions'
OUT = Path('/home/josh/omp-workspace/drock-lmcache/r24-battery/replay-consistency-1m.json')
rows = []


def one(label: str) -> None:
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1500) as response:
        result = json.loads(response.read())
    wall = time.time() - t0
    message = result['choices'][0]['message']
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    row = {
        'label': label,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'hit': CODE in (content + reasoning),
        'content': content,
        'reasoning_content': reasoning,
        'finish_reason': result['choices'][0]['finish_reason'],
    }
    rows.append(row)
    OUT.write_text(json.dumps({'needle_code': CODE, 'results': rows}, indent=2))
    print(f"{label}: {wall:.2f}s tokens={row['prompt_tokens']} hit={row['hit']} answer={content[:100]!r}", flush=True)


one('cold')
for i in range(1, 5):
    one(f'warm-{i}')
subprocess.run(['docker', 'restart', 'r24-replay'], stdout=subprocess.DEVNULL, check=True, timeout=900)
for _ in range(90):
    check = subprocess.run(['curl', '-fsS', '--max-time', '4', 'http://127.0.0.1:5001/health'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode == 0:
        break
    time.sleep(10)
for i in range(1, 4):
    one(f'restart-{i}')
