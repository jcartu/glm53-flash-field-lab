#!/usr/bin/env python3
"""Matched sustained MTP measurements from complete Prometheus counter windows."""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp
import runtime as rt

COUNTERS = {
    'generated': 'vllm:generation_tokens_total',
    'draft_steps': 'vllm:spec_decode_num_drafts_total',
    'proposed': 'vllm:spec_decode_num_draft_tokens_total',
    'accepted': 'vllm:spec_decode_num_accepted_tokens_total',
}
TEXT = (
    'A distributed storage service keeps a journal of completed writes. Each record has a sequence number, '
    'a checksum and an owner. Readers validate the checksum before serving a record. The service measures '
    'latency separately from throughput and does not treat a successful request as proof of data integrity. '
)


def counters(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        metric = line.split('{', 1)[0].split()[0]
        for key, expected in COUNTERS.items():
            if metric == expected:
                value = float(line.split()[-1])
                result[key] = result.get(key, 0.0) + value
    return result


async def get_metrics(session: aiohttp.ClientSession) -> str:
    async with session.get(rt.BASE_URL + '/metrics') as response:
        response.raise_for_status()
        return await response.text()


async def prompt_ids(session: aiohttp.ClientSession, context: int) -> list[int]:
    text = ('Continue the following technical explanation in detail.\n' + TEXT * (1800 if context else 2)
            + '\nExplain practical design choices and tradeoffs in a numbered sequence:')
    async with session.post(rt.BASE_URL + '/tokenize', json={'model': rt.MODEL_NAME, 'prompt': text}) as response:
        response.raise_for_status()
        payload = await response.json()
    ids = payload.get('tokens')
    if not isinstance(ids, list) or not ids or not all(isinstance(token, int) for token in ids):
        raise RuntimeError('Server tokenizer did not return token IDs')
    if context:
        if len(ids) < context:
            raise RuntimeError(f'Token construction too short: {len(ids)} < {context}')
        ids = ids[:context]
    return ids


async def measure(session: aiohttp.ClientSession, label: str, context: int, concurrency: int, duration: float, repeat: int) -> dict:
    ids = await prompt_ids(session, context)
    first_tokens = [asyncio.Event() for _ in range(concurrency)]
    streams = [{'index': index, 'chunks': 0, 'characters': 0, 'first_token_at': None, 'ended': False, 'error': None} for index in range(concurrency)]
    tasks: list[asyncio.Task] = []
    sample = {'context_tokens_requested': context, 'input_tokens': len(ids), 'concurrency': concurrency,
              'repeat': repeat, 'duration_requested_seconds': duration, 'sampling': {'temperature': 0, 'seed': 260905, 'ignore_eos': True, 'max_tokens': 32768},
              'endpoint': '/v1/completions', 'streams': streams, 'passed': False}
    async def stream(index: int) -> None:
        payload = {'model': rt.MODEL_NAME, 'prompt': ids, 'max_tokens': 32768, 'temperature': 0,
                   'seed': 260905, 'ignore_eos': True, 'stream': True, 'cache_salt': f'r26-acceptance-{context}-{concurrency}-{repeat}'}
        try:
            async with session.post(rt.BASE_URL + '/v1/completions', json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:1000]}')
                async for raw in response.content:
                    line = raw.decode('utf-8').strip()
                    if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                        continue
                    event = json.loads(line[5:])
                    if event.get('error'):
                        raise RuntimeError(str(event['error']))
                    for choice in event.get('choices', []):
                        text = choice.get('text', '')
                        if text:
                            streams[index]['chunks'] += 1
                            streams[index]['characters'] += len(text)
                            if not first_tokens[index].is_set():
                                streams[index]['first_token_at'] = time.time()
                                first_tokens[index].set()
                        if choice.get('finish_reason') is not None:
                            streams[index]['finish_reason'] = choice['finish_reason']
                streams[index]['ended'] = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            streams[index]['error'] = repr(error)
            first_tokens[index].set()
    try:
        tasks = [asyncio.create_task(stream(index)) for index in range(concurrency)]
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in first_tokens)), timeout=240)
        await asyncio.sleep(5)
        if any(task.done() for task in tasks):
            raise RuntimeError('A stream ended before the steady measurement window')
        before_text = await get_metrics(session)
        started = time.monotonic()
        await asyncio.sleep(duration)
        after_text = await get_metrics(session)
        elapsed = time.monotonic() - started
        if any(task.done() for task in tasks):
            raise RuntimeError('A stream ended during the sustained window; result is not a full-concurrency sample')
        before, after = counters(before_text), counters(after_text)
        if set(before) != set(COUNTERS) or set(after) != set(COUNTERS):
            raise RuntimeError(f'Missing speculative counters: before={before}, after={after}')
        delta = {key: after[key] - before[key] for key in COUNTERS}
        if any(value < 0 for value in delta.values()) or delta['draft_steps'] <= 0 or delta['proposed'] <= 0:
            raise RuntimeError(f'Invalid counter window: {delta}')
        fraction = delta['accepted'] / delta['proposed']
        if not 0 <= fraction <= 1:
            raise RuntimeError(f'Invalid acceptance fraction: {fraction}')
        sample.update({'passed': True, 'elapsed_seconds': elapsed, 'counter_before': before, 'counter_after': after,
                       'counter_delta': delta, 'output_tokens_per_second': delta['generated'] / elapsed,
                       'aggregate_verifier_steps_per_second': delta['draft_steps'] / elapsed,
                       'acceptance_fraction': fraction, 'accepted_draft_tokens_per_step': delta['accepted'] / delta['draft_steps'],
                       'emitted_tokens_per_verifier_step': delta['generated'] / delta['draft_steps'],
                       'step_metric_note': 'Aggregate per-request speculative draft/verifier steps, not physical batched GPU kernel launches.'})
        stem = f'{label}-acceptance-c{concurrency}-ctx{context}-repeat{repeat}'
        (rt.ROOT / f'{stem}.before.metrics.txt').write_text(before_text)
        (rt.ROOT / f'{stem}.after.metrics.txt').write_text(after_text)
    except Exception as error:
        sample['error'] = repr(error)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Cancellation is intentional for this performance-only sustained probe.
        await asyncio.sleep(3)
    return sample


async def execute(args: argparse.Namespace) -> dict:
    result = {'label': args.label, 'measurement': 'Matched fixed-prompt completion streams, 5s decode warmup, whole-window server counter deltas',
              'quality_claim': False, 'sample_count': 0, 'samples': [], 'passed': False}
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        for repeat in range(1, args.repeats + 1):
            for context in args.contexts:
                for concurrency in args.concurrency:
                    rt.note(f'ACCEPTANCE WINDOW {args.label} ctx={context} c={concurrency} repeat={repeat}')
                    sample = await measure(session, args.label, context, concurrency, args.duration, repeat)
                    result['samples'].append(sample)
                    result['sample_count'] = len(result['samples'])
                    rt.save_json(args.label + '-acceptance.json', result)
    result['passed'] = bool(result['samples']) and all(sample['passed'] for sample in result['samples'])
    rt.save_json(args.label + '-acceptance.json', result)
    rt.record_gate('acceptance-window:' + args.label, result['passed'], {'samples': result['sample_count'], 'failed': sum(not sample['passed'] for sample in result['samples'])})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--contexts', default='0,32768', type=lambda text: [int(value) for value in text.split(',')])
    parser.add_argument('--concurrency', default='1,8', type=lambda text: [int(value) for value in text.split(',')])
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--repeats', type=int, default=2)
    result = asyncio.run(execute(parser.parse_args()))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
