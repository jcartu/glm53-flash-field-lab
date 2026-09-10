#!/usr/bin/env python3
"""Replay fixed long histories with explicit reasoning-retention controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'r26'))
import quality_probes as legacy
from behavior_probe import equal_json, parse_answer, save


def reset(base_url):
    attempts = []
    for delay in (0, .2, .5, 1, 2, 3):
        time.sleep(delay)
        call = legacy.post_json(base_url, '/reset_prefix_cache', {'reset_running_requests': False}, 30)
        attempts.append(call)
        if call.get('ok') and isinstance(call.get('response'), dict) and call['response'].get('success') is True:
            return {'passed': True, 'attempts': attempts}
    return {'passed': False, 'attempts': attempts}


def run(args):
    manifest = json.loads(args.manifest.read_text())
    rows = []
    started = time.time()
    for position, item in enumerate(manifest['cases']):
        path = (args.manifest.parent / item['path']).resolve()
        if not path.is_relative_to(args.manifest.parent.resolve()):
            raise RuntimeError('Unsafe fixture path')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise RuntimeError('Frozen history hash mismatch: ' + item['id'])
        case = json.loads(content)
        for clear in ((False, True) if position % 2 == 0 else (True, False)):
            label = case['id'] + '-clear' + str(int(clear))
            row = {'id': label, 'case': case['id'], 'kind': case['kind'],
                'nominal_filler_tokens': case['nominal_filler_tokens'], 'clear_thinking': clear,
                'offline_rendered_tokens': case['offline_rendered_tokens'][str(clear).lower()],
                'semantic_passed': False, 'runtime_error': False, 'budget_exhausted': False}
            reset_record = reset(args.base_url)
            save(args.output_dir / 'resets' / (label + '.json'), reset_record)
            if not reset_record['passed']:
                row.update({'runtime_error': True, 'error': 'Prefix cache reset failed; refused to label request cold'})
                rows.append(row)
                save(args.output_dir / 'summary.json', {'complete': False, 'rows': rows, 'error': row['error']})
                return 1
            body = {'model': args.model, 'messages': case['messages'], 'max_tokens': 4096,
                'temperature': 1.0, 'top_p': 0.95, 'seed': case['seed'],
                'cache_salt': 'r29-history-' + case['id'],
                'chat_template_kwargs': {'reasoning_effort': 'max', 'clear_thinking': clear}}
            row['request_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            call = legacy.post_json(args.base_url, '/v1/chat/completions', body, args.timeout)
            save(args.output_dir / 'requests' / (label + '.json'), call)
            message, error = legacy.response_message(call)
            if error:
                row.update({'runtime_error': True, 'error': error})
            else:
                choice = call['response']['choices'][0]
                visible, reasoning = legacy.message_text(message)
                usage = call['response'].get('usage') or {}
                row.update({'finish_reason': choice.get('finish_reason'), 'prompt_tokens': usage.get('prompt_tokens'),
                    'completion_tokens': usage.get('completion_tokens'), 'elapsed_seconds': call['elapsed_seconds'],
                    'visible_answer': visible, 'reasoning_chars': len(reasoning),
                    'visible_cjk_chars': sum('\u4e00' <= char <= '\u9fff' for char in visible),
                    'reasoning_cjk_chars': sum('\u4e00' <= char <= '\u9fff' for char in reasoning),
                    'repetition_evidence': legacy.repetition_evidence(visible + '\n' + reasoning),
                    'unexpected_tool_calls': bool(message.get('tool_calls'))})
                row['budget_exhausted'] = choice.get('finish_reason') == 'length'
                try:
                    parsed = parse_answer(visible)
                    row['semantic_passed'] = equal_json(parsed, case['expected']) and not row['budget_exhausted'] and not row['unexpected_tool_calls']
                    row['parsed_answer'] = parsed
                except (ValueError, TypeError) as error:
                    row['answer_parse_error'] = str(error)
                if isinstance(row.get('prompt_tokens'), int):
                    row['tokenization_difference'] = row['prompt_tokens'] - row['offline_rendered_tokens']
                else:
                    row['runtime_error'] = True
                    row['error'] = 'No server prompt-token accounting'
            rows.append(row)
            save(args.output_dir / 'progress.json', {'completed': len(rows), 'expected': manifest['requests_per_arm'], 'rows': rows})
            print(json.dumps({key: row.get(key) for key in ['id', 'prompt_tokens', 'semantic_passed', 'runtime_error', 'budget_exhausted']}), flush=True)
    summary = {'schema': 'r29-history-results/v1', 'complete': len(rows) == manifest['requests_per_arm'],
        'expected': manifest['requests_per_arm'], 'completed': len(rows),
        'semantic_passed': sum(row['semantic_passed'] for row in rows),
        'runtime_errors': sum(row['runtime_error'] for row in rows),
        'budget_exhausted': sum(row['budget_exhausted'] for row in rows),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'started_at': started, 'finished_at': time.time(), 'rows': rows,
        'scope': manifest['scope'] + ' Actual prompt_tokens, not the nominal original conversation size, govern context-length interpretation.'}
    save(args.output_dir / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}), flush=True)
    return int(not summary['complete'] or summary['runtime_errors'] > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:5002')
    parser.add_argument('--model', default='GLM-5.3-Flash-NVFP4')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.port != 5002:
        parser.error('This mutating cold-cache probe is restricted to the dedicated loopback5002 test endpoint')
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
