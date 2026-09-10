#!/usr/bin/env python3
"""Materialize reasoning-retention and length-preserving control histories."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

WORDS = ('archive parser scheduler cache worker request response trace review module boundary state '
         'buffer version audit fixture result caller allocation page record policy branch function '
         'syntax schema signal context register pointer layout frame source checkpoint token queue '
         'order index complete pending stable current previous immutable local remote validation '
         'read write merge update preserve compare derive measured expected observed').split()


def render(template, messages, clear):
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message.get('reasoning') is not None:
            message['reasoning_content'] = message['reasoning']
        for call in message.get('tool_calls', []):
            arguments = call['function']['arguments']
            if isinstance(arguments, str):
                call['function']['arguments'] = json.loads(arguments)
    return template.render(messages=normalized, tools=None, add_generation_prompt=True,
                           reasoning_effort='max', clear_thinking=clear)


def make_history(filler, rows, kind):
    split = len(filler) // 2
    parts = [filler[:split], filler[split:]]
    messages = [{'role': 'system', 'content': 'You are a software maintenance agent. Historical planning is not authoritative for the current checks. Use the latest tool result for the requested calculation. Return only the requested JSON in the visible answer.'}]
    for index, part in enumerate(parts):
        messages.append({'role': 'user', 'content': f'Review archived maintenance notes, section {index + 1}.'})
        if kind == 'reasoning_history':
            messages.append({'role': 'assistant', 'reasoning': part,
                             'content': 'The historical review is complete; the current checks will arrive separately.'})
        else:
            messages.append({'role': 'assistant', 'reasoning': 'I will retain the archived text as a reference document.',
                             'content': 'Archived reference, not current check results:\n' + part})
    messages.append({'role': 'user', 'content': 'Use read_checks for the current run. From its rows return exactly JSON {"passed":sorted ids with ok=true,"failed":sorted ids with ok=false,"score":sum of weights for passed rows}. Historical numbers must not override these current rows.'})
    messages.append({'role': 'assistant', 'reasoning': 'I need the current tool rows, then must classify every row and sum only passing weights. This is the active unfinished tool round.',
        'content': '', 'tool_calls': [{'id': 'current_checks', 'type': 'function',
        'function': {'name': 'read_checks', 'arguments': '{"run":"current"}'}}]})
    messages.append({'role': 'tool', 'tool_call_id': 'current_checks', 'content': json.dumps({'rows': rows}, separators=(',', ':'))})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sizes', default='16384,131072,524288,819200')
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(',')]
    if not sizes or min(sizes) < 1024 or max(sizes) > 900000:
        parser.error('History sizes must be between 1024 and 900000 tokens')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    template_text = args.template.read_text()
    template = _compile_jinja_template(template_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in (26090931, 26090932):
        rng = random.Random(seed)
        paragraphs = []
        # Deliberately varied archived engineering text, rather than a repeated-token stressor.
        for index in range(max(sizes) // 14 + 1000):
            words = ' '.join(rng.choice(WORDS) for _ in range(20))
            paragraphs.append(f'Archived review {index:06}: {words}. Prior tentative score {rng.randrange(10, 999)} is superseded by future check results.\n')
        full_ids = tokenizer.encode(''.join(paragraphs), add_special_tokens=False).ids
        if len(full_ids) < max(sizes):
            raise RuntimeError('Filler did not cover the requested context range')
        rows = [{'id': f'check-{i:02}', 'ok': bool(rng.getrandbits(1)), 'weight': rng.randrange(1, 10)} for i in range(12)]
        rows[0]['ok'], rows[1]['ok'] = True, False
        expected = {'passed': sorted(row['id'] for row in rows if row['ok']),
                    'failed': sorted(row['id'] for row in rows if not row['ok']),
                    'score': sum(row['weight'] for row in rows if row['ok'])}
        for nominal in sizes:
            filler = tokenizer.decode(full_ids[:nominal], skip_special_tokens=False)
            for kind in ('reasoning_history', 'visible_length_control'):
                messages = make_history(filler, rows, kind)
                lengths = {str(clear).lower(): len(tokenizer.encode(render(template, messages, clear), add_special_tokens=False).ids) for clear in (False, True)}
                if max(lengths.values()) + 4096 > 1048576:
                    raise RuntimeError('History and output reserve exceed the common context limit')
                identity = f'{kind}-{nominal}-{seed}'
                record = {'schema': 'r29-history-case/v1', 'id': identity, 'kind': kind, 'nominal_filler_tokens': nominal,
                          'seed': seed, 'messages': messages, 'expected': expected, 'offline_rendered_tokens': lengths}
                path = args.output_dir / (identity + '.json')
                encoded = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
                path.write_bytes(encoded)
                cases.append({'id': identity, 'path': path.name, 'sha256': hashlib.sha256(encoded).hexdigest(),
                              'kind': kind, 'nominal_filler_tokens': nominal, 'seed': seed, 'offline_rendered_tokens': lengths})
    manifest = {'schema': 'r29-history-fixtures/v1', 'cases': cases,
        'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        'template_sha256': hashlib.sha256(template_text.encode()).hexdigest(),
        'requests_per_arm': len(cases) * 2,
        'scope': 'Fixed synthetic coding-tool histories, clear_thinking false/true. Visible-length controls retain similarly long input after reasoning is cleared. Role placement differs, so this is a diagnostic control, not a pure causal proof or the private reported reproducer.'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': len(cases), 'requests_per_arm': manifest['requests_per_arm'], 'sizes': sizes}, indent=2))


if __name__ == '__main__':
    main()
