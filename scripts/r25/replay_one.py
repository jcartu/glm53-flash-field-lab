#!/usr/bin/env python3
import json
import sys
import time
import urllib.request

CODE = sys.argv[1]
OUT = sys.argv[2]
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
request = urllib.request.Request('http://127.0.0.1:5001/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
start = time.time()
with urllib.request.urlopen(request, timeout=1500) as response:
    result = json.loads(response.read())
message = result['choices'][0]['message']
row = {
    'wall_seconds': time.time() - start,
    'prompt_tokens': result['usage']['prompt_tokens'],
    'completion_tokens': result['usage']['completion_tokens'],
    'hit': CODE in ((message.get('content') or '') + (message.get('reasoning_content') or '')),
}
open(OUT, 'w').write(json.dumps(row, indent=2))
print(json.dumps(row))
