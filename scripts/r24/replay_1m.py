#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.request

from pathlib import Path
import sys

sys.path.insert(0, '/home/josh/omp-workspace')
from importlib.machinery import SourceFileLoader
needle = SourceFileLoader('needle_probe', '/home/josh/omp-workspace/probe-1m-needle.py').load_module()

MODEL = 'GLM-5.3-Flash-NVFP4'
URL = 'http://127.0.0.1:5001/v1/chat/completions'
CONTAINER = 'r24-replay'
OUT = Path('/home/josh/omp-workspace/drock-lmcache/r24-battery/replay-1m.json')

prompt = needle.build_prompt(0.5)
body = json.dumps({
    'model': MODEL,
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 512,
    'temperature': 0.0,
    'reasoning_effort': 'low',
}).encode()


def request(label: str) -> dict:
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1500) as response:
        result = json.loads(response.read())
    wall = time.time() - t0
    message = result['choices'][0]['message']
    answer = (message.get('content') or '') + ' ' + (message.get('reasoning_content') or '')
    row = {
        'label': label,
        'wall_seconds': wall,
        'prompt_tokens': result['usage']['prompt_tokens'],
        'completion_tokens': result['usage']['completion_tokens'],
        'needle_hit': needle.NEEDLE_CODE in answer,
        'finish_reason': result['choices'][0]['finish_reason'],
    }
    print(json.dumps(row), flush=True)
    return row


rows = [request('cold'), request('warm')]
subprocess.run(['docker', 'restart', CONTAINER], check=True, timeout=900, stdout=subprocess.DEVNULL)
for _ in range(90):
    probe = subprocess.run(['curl', '-fsS', '--max-time', '4', 'http://127.0.0.1:5001/health'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        break
    time.sleep(10)
rows.append(request('restart'))
OUT.write_text(json.dumps({'results': rows}, indent=2))
