#!/usr/bin/env python3
"""Measure cyclic agent turns while periodic large cold prefills contend.

Each hot session is primed once, then repeatedly performs a short decode and
immediately submits the next turn with a small incremental user message. The
storm arm injects unique 128K cold prompts on a fixed cadence. The resulting
hot-turn TTFT directly measures whether those small prefills queue behind the
large cold work that iSource described.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


async def iter_sse(response: httpx.Response):
    async for line in response.aiter_lines():
        if not line.startswith('data: '):
            continue
        payload = line[6:]
        if payload == '[DONE]':
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def parse_metrics(text: str) -> dict[str, float]:
    wanted = {
        'vllm:num_requests_running',
        'vllm:num_requests_waiting',
        'vllm:prompt_tokens_total',
        'vllm:generation_tokens_total',
        'vllm:scheduler_compute_seconds_total',
    }
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        base = line.split('{', 1)[0].split(None, 1)[0]
        if base not in wanted:
            continue
        key = base
        if base == 'vllm:scheduler_compute_seconds_total':
            if 'class="decode"' in line:
                key += ':decode'
            elif 'class="prefill"' in line:
                key += ':prefill'
            else:
                continue
        try:
            result[key] = result.get(key, 0.0) + float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return result


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    response = await client.get(f'{base_url}/metrics')
    response.raise_for_status()
    return parse_metrics(response.text)


async def wait_idle(client: httpx.AsyncClient, base_url: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metrics = await scrape_metrics(client, base_url)
        if (
            metrics.get('vllm:num_requests_running', 0.0) == 0.0
            and metrics.get('vllm:num_requests_waiting', 0.0) == 0.0
        ):
            return
        await asyncio.sleep(0.5)
    raise RuntimeError('server did not return to idle')


async def tokenize_count(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    content: str,
) -> int:
    response = await client.post(
        f'{base_url}/tokenize',
        json={
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
        },
    )
    response.raise_for_status()
    return int(response.json()['count'])


async def exact_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    target_tokens: int,
    identity: str,
) -> str:
    prefix = (
        'Agent-session scheduler probe. This text is independent filler.\n'
        f'TRAFFIC-ID: {identity}\n'
        'BEGIN CONTEXT\n'
    )
    suffix = '\nEND CONTEXT\nAnalyze the context and continue with a detailed technical explanation.'
    base_count = await tokenize_count(client, base_url, model, prefix + suffix)
    repetitions = max(target_tokens - base_count, 0)
    for _ in range(6):
        content = prefix + (' x' * repetitions) + suffix
        observed = await tokenize_count(client, base_url, model, content)
        if observed == target_tokens:
            return content
        repetitions = max(repetitions + target_tokens - observed, 0)
    raise RuntimeError(f'cannot construct exact {target_tokens}-token prompt')


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    offered_at: float,
) -> dict[str, Any]:
    started = time.monotonic()
    first_token_at: float | None = None
    chunk_times: list[float] = []
    usage: dict[str, Any] = {}
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    async with client.stream(
        'POST', f'{base_url}/v1/chat/completions', json=payload
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors='replace')
            raise RuntimeError(f'HTTP {response.status_code}: {body[:1000]}')
        async for event in iter_sse(response):
            now = time.monotonic()
            if event.get('usage'):
                usage = event['usage']
            choices = event.get('choices') or []
            if not choices:
                continue
            delta = choices[0].get('delta') or {}
            visible = delta.get('content') or ''
            reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
            if visible:
                visible_parts.append(visible)
            if reasoning:
                reasoning_parts.append(reasoning)
            if visible or reasoning:
                chunk_times.append(now)
                if first_token_at is None:
                    first_token_at = now
    finished = time.monotonic()
    first_token_at = first_token_at or finished
    gaps = [current - previous for previous, current in zip(chunk_times, chunk_times[1:])]
    return {
        'offered_at': offered_at,
        'request_started_at': started,
        'first_token_at': first_token_at,
        'finished_at': finished,
        'ttft_seconds': first_token_at - offered_at,
        'request_seconds': finished - offered_at,
        'prompt_tokens': int(usage.get('prompt_tokens') or 0),
        'completion_tokens': int(usage.get('completion_tokens') or 0),
        'decode_gap_p95_seconds': percentile(gaps, 0.95),
        'decode_gap_max_seconds': max(gaps) if gaps else None,
        'content': ''.join(visible_parts),
        'reasoning_content': ''.join(reasoning_parts),
    }


async def prime_session(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    identity: str,
    session_index: int,
) -> dict[str, Any]:
    content = await exact_prompt(
        client,
        args.base_url,
        args.model,
        args.session_context_tokens,
        identity,
    )
    messages = [{'role': 'user', 'content': content}]
    payload = {
        'model': args.model,
        'messages': messages,
        'stream': True,
        'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
        'max_tokens': args.prime_decode_tokens,
        'temperature': 0,
        'seed': 0,
        'ignore_eos': True,
        'reasoning_effort': 'low',
        'cache_salt': f'{identity}-salt',
    }
    offered = time.monotonic()
    result = await stream_chat(client, args.base_url, payload, offered)
    assistant = result['content'] or 'Acknowledged.'
    messages.append({'role': 'assistant', 'content': assistant})
    return {
        'session_index': session_index,
        'identity': identity,
        'cache_salt': f'{identity}-salt',
        'messages': messages,
        'previous_prompt_tokens': result['prompt_tokens'],
        'prime': result,
        'turns': [],
    }


async def session_loop(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    state: dict[str, Any],
    start_at: float,
    end_at: float,
) -> None:
    delay = start_at - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    turn_index = 0
    while time.monotonic() < end_at:
        turn_index += 1
        small_user = (
            f'Agent turn {turn_index}. Continue from the prior answer and preserve all context. '
            + (' detail' * args.incremental_filler_tokens)
        )
        state['messages'].append({'role': 'user', 'content': small_user})
        payload = {
            'model': args.model,
            'messages': state['messages'],
            'stream': True,
            'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
            'max_tokens': args.turn_decode_tokens,
            'temperature': 0,
            'seed': 0,
            'ignore_eos': True,
            'reasoning_effort': 'low',
            'cache_salt': state['cache_salt'],
        }
        offered = time.monotonic()
        result = await stream_chat(client, args.base_url, payload, offered)
        result['session_index'] = state['session_index']
        result['turn_index'] = turn_index
        result['incremental_prompt_tokens'] = max(
            0,
            result['prompt_tokens'] - state['previous_prompt_tokens'],
        )
        state['previous_prompt_tokens'] = result['prompt_tokens']
        state['turns'].append(result)
        assistant = result['content'] or 'Acknowledged.'
        state['messages'].append({'role': 'assistant', 'content': assistant})


async def cold_request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    content: str,
    identity: str,
    offered_at: float,
) -> dict[str, Any]:
    delay = offered_at - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': content}],
        'stream': True,
        'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
        'max_tokens': 1,
        'temperature': 0,
        'seed': 0,
        'ignore_eos': True,
        'reasoning_effort': 'low',
        'cache_salt': f'{identity}-salt',
    }
    result = await stream_chat(client, args.base_url, payload, offered_at)
    result['identity'] = identity
    return result


async def metrics_sampler(
    client: httpx.AsyncClient,
    base_url: str,
    stop: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    while not stop.is_set():
        try:
            samples.append({'time': time.monotonic(), **await scrape_metrics(client, base_url)})
        except Exception as error:
            samples.append({'time': time.monotonic(), 'error': str(error)})
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass


def summarize(
    states: list[dict[str, Any]],
    cold_rows: list[dict[str, Any]],
    metric_samples: list[dict[str, Any]],
    start_at: float,
    end_at: float,
) -> dict[str, Any]:
    turns = [turn for state in states for turn in state['turns']]
    cold_intervals = [
        (float(row['offered_at']), float(row['first_token_at']))
        for row in cold_rows
    ]
    for turn in turns:
        offered = float(turn['offered_at'])
        turn['cold_backlog_at_offer'] = sum(
            cold_start <= offered < cold_end
            for cold_start, cold_end in cold_intervals
        )
    turns_within_window = [
        turn for turn in turns if float(turn['finished_at']) <= end_at
    ]
    late_turns = [
        turn for turn in turns if float(turn['finished_at']) > end_at
    ]
    contended_turns = [
        turn for turn in turns if int(turn['cold_backlog_at_offer']) > 0
    ]
    clear_turns = [
        turn for turn in turns if int(turn['cold_backlog_at_offer']) == 0
    ]
    ttfts = [float(turn['ttft_seconds']) for turn in turns]
    contended_ttfts = [float(turn['ttft_seconds']) for turn in contended_turns]
    clear_ttfts = [float(turn['ttft_seconds']) for turn in clear_turns]
    request_times = [float(turn['request_seconds']) for turn in turns]
    incremental = [int(turn['incremental_prompt_tokens']) for turn in turns]
    gaps = [
        float(turn['decode_gap_p95_seconds'])
        for turn in turns
        if turn['decode_gap_p95_seconds'] is not None
    ]
    per_session = [len(state['turns']) for state in states]
    window_samples = [
        sample for sample in metric_samples
        if start_at <= sample['time'] <= end_at
    ]
    cold_ttfts = [float(row['ttft_seconds']) for row in cold_rows]
    cold_within_window = [
        row for row in cold_rows if float(row['first_token_at']) <= end_at
    ]
    average_cold_backlog = sum(
        max(0.0, min(end_at, cold_end) - max(start_at, cold_start))
        for cold_start, cold_end in cold_intervals
    ) / (end_at - start_at)
    cold_drain_seconds = max(
        (
            max(0.0, float(row['finished_at']) - end_at)
            for row in cold_rows
        ),
        default=0.0,
    )
    within_hot_tokens = sum(
        int(turn['completion_tokens']) for turn in turns_within_window
    )
    within_cold_tokens = sum(
        int(row['prompt_tokens']) for row in cold_within_window
    )
    return {
        'hot_turns_offered': len(turns),
        'hot_turns_finished_within_window': len(turns_within_window),
        'late_hot_turns': len(late_turns),
        'hot_completion_tokens_all_offered_turns': sum(
            int(turn['completion_tokens']) for turn in turns
        ),
        'hot_completion_tokens_finished_within_window': within_hot_tokens,
        'hot_output_tokens_per_second': within_hot_tokens / (end_at - start_at),
        'total_useful_tokens_per_second': (
            within_hot_tokens + within_cold_tokens
        ) / (end_at - start_at),
        'turns_per_session': {
            'min': min(per_session),
            'mean': statistics.fmean(per_session),
            'max': max(per_session),
        },
        'small_incremental_prefill_tokens': {
            'median': statistics.median(incremental) if incremental else None,
            'p95': percentile([float(value) for value in incremental], 0.95),
        },
        'hot_ttft_seconds': {
            'median': statistics.median(ttfts) if ttfts else None,
            'p95': percentile(ttfts, 0.95),
            'p99': percentile(ttfts, 0.99),
            'max': max(ttfts) if ttfts else None,
        },
        'hot_ttft_when_no_cold_pending_seconds': {
            'count': len(clear_ttfts),
            'median': statistics.median(clear_ttfts) if clear_ttfts else None,
            'p95': percentile(clear_ttfts, 0.95),
            'max': max(clear_ttfts) if clear_ttfts else None,
        },
        'hot_ttft_while_cold_pending_seconds': {
            'count': len(contended_ttfts),
            'median': statistics.median(contended_ttfts) if contended_ttfts else None,
            'p95': percentile(contended_ttfts, 0.95),
            'max': max(contended_ttfts) if contended_ttfts else None,
        },
        'cold_backlog_at_hot_offer': {
            'mean': statistics.fmean(
                int(turn['cold_backlog_at_offer']) for turn in turns
            ) if turns else 0.0,
            'max': max(
                (int(turn['cold_backlog_at_offer']) for turn in turns),
                default=0,
            ),
        },
        'hot_request_seconds': {
            'median': statistics.median(request_times) if request_times else None,
            'p95': percentile(request_times, 0.95),
            'max': max(request_times) if request_times else None,
        },
        'hot_decode_gap_p95_seconds': {
            'median': statistics.median(gaps) if gaps else None,
            'p95': percentile(gaps, 0.95),
            'max': max(gaps) if gaps else None,
        },
        'scheduler': {
            'max_running': max(
                (
                    sample.get('vllm:num_requests_running', 0.0)
                    for sample in window_samples
                ),
                default=0.0,
            ),
            'max_waiting': max(
                (
                    sample.get('vllm:num_requests_waiting', 0.0)
                    for sample in window_samples
                ),
                default=0.0,
            ),
            'mean_waiting': statistics.fmean(
                sample.get('vllm:num_requests_waiting', 0.0)
                for sample in window_samples
            ) if window_samples else 0.0,
            'mean_cold_requests_pending': average_cold_backlog,
        },
        'cold_requests_offered': len(cold_rows),
        'cold_requests_finished_within_window': len(cold_within_window),
        'cold_prompt_tokens_finished_within_window': within_cold_tokens,
        'cold_drain_seconds_after_window': cold_drain_seconds,
        'cold_ttft_seconds': {
            'median': statistics.median(cold_ttfts) if cold_ttfts else None,
            'p95': percentile(cold_ttfts, 0.95),
            'max': max(cold_ttfts) if cold_ttfts else None,
        },
    }


async def run_scenario(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    concurrency: int,
    repeat: int,
    storm: bool,
) -> dict[str, Any]:
    scenario = 'periodic-cold' if storm else 'baseline'
    identity = f'hot-{args.policy_label}-c{concurrency}-r{repeat}-{scenario}-{uuid.uuid4().hex[:8]}'
    print(f'START {identity}', flush=True)
    states = await asyncio.gather(*[
        prime_session(client, args, f'{identity}-session-{index}', index)
        for index in range(concurrency)
    ])
    await wait_idle(client, args.base_url)
    await asyncio.sleep(3.0)
    duration = args.storm_seconds if storm else args.baseline_seconds
    cold_offsets = (
        [float(value) for value in range(0, int(duration), args.cold_period_seconds)]
        if storm else []
    )
    cold_contents = await asyncio.gather(*[
        exact_prompt(
            client,
            args.base_url,
            args.model,
            args.cold_prompt_tokens,
            f'{identity}-cold-{index}',
        )
        for index in range(len(cold_offsets))
    ])
    start_at = time.monotonic() + 1.0
    end_at = start_at + duration
    metric_stop = asyncio.Event()
    metric_samples: list[dict[str, Any]] = []
    sampler = asyncio.create_task(metrics_sampler(client, args.base_url, metric_stop, metric_samples))
    session_tasks = [
        asyncio.create_task(session_loop(client, args, state, start_at, end_at))
        for state in states
    ]
    cold_tasks = [
        asyncio.create_task(cold_request(
            client,
            args,
            content,
            f'{identity}-cold-{index}',
            start_at + offset,
        ))
        for index, (offset, content) in enumerate(zip(cold_offsets, cold_contents))
    ]
    await asyncio.gather(*session_tasks)
    cold_rows: list[dict[str, Any]] = []
    if cold_tasks:
        cold_rows = list(await asyncio.gather(*cold_tasks))
    metric_stop.set()
    await sampler
    finished_at = time.monotonic()
    await wait_idle(client, args.base_url)
    result = {
        'policy': args.policy_label,
        'concurrency': concurrency,
        'repeat': repeat,
        'scenario': scenario,
        'identity': identity,
        'measurement_seconds': duration,
        'actual_seconds_until_all_offered_work_finished': finished_at - start_at,
        'cold_prompt_tokens': args.cold_prompt_tokens,
        'cold_offsets_seconds': cold_offsets,
        'summary': summarize(states, cold_rows, metric_samples, start_at, end_at),
        'sessions': states,
        'cold_requests': cold_rows,
        'metric_samples': metric_samples,
    }
    summary = result['summary']
    print(
        f"DONE {identity} turns={summary['hot_turns_finished_within_window']} "
        f"hot_p95={summary['hot_ttft_seconds']['p95']:.3f}s "
        f"waiting={summary['scheduler']['max_waiting']:.0f}",
        flush=True,
    )
    return result


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    timeout = httpx.Timeout(None, connect=30.0)
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for repeat in range(1, args.repeats + 1):
            for concurrency in args.concurrencies:
                for storm in (False, True):
                    cell = await run_scenario(client, args, concurrency, repeat, storm)
                    cells.append(cell)
                    args.output.write_text(json.dumps({
                        'metadata': vars(args) | {'output': str(args.output)},
                        'cells': cells,
                    }, indent=2, default=str))
                    await asyncio.sleep(args.cooldown_seconds)
    return {
        'metadata': vars(args) | {'output': str(args.output)},
        'cells': cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:5001')
    parser.add_argument('--model', default='GLM-5.3-Flash-NVFP4')
    parser.add_argument('--policy-label', required=True)
    parser.add_argument('--concurrencies', default='8,16')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--session-context-tokens', type=int, default=8192)
    parser.add_argument('--prime-decode-tokens', type=int, default=32)
    parser.add_argument('--turn-decode-tokens', type=int, default=128)
    parser.add_argument('--incremental-filler-tokens', type=int, default=220)
    parser.add_argument('--baseline-seconds', type=float, default=30.0)
    parser.add_argument('--storm-seconds', type=float, default=60.0)
    parser.add_argument('--cold-prompt-tokens', type=int, default=131072)
    parser.add_argument('--cold-period-seconds', type=int, default=15)
    parser.add_argument('--cooldown-seconds', type=float, default=3.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.concurrencies = [int(value) for value in args.concurrencies.split(',')]
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f'output={args.output}', flush=True)


if __name__ == '__main__':
    main()
