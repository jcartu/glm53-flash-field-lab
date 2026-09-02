#!/usr/bin/env python3
"""Measure decode degradation while a cold long prefill collides with it.

The default request geometry follows the GLM-5.3-Flash JJ scheduler-fairness
qualification: begin a long decode from a 4,094-token prompt, inject a cold
65,535-token prefill after 2,048 output tokens, and measure both requests over
the collision window.  Results use per-request continuous OpenAI usage for
decode throughput and Prometheus only for scheduler/counter corroboration.
"""

from __future__ import annotations

import argparse
import os
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
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def parse_prometheus(text: str) -> dict[str, float]:
    wanted = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    }
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name not in wanted or name in result:
            continue
        try:
            result[name] = float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return result


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    response = await client.get(f"{base_url}/metrics")
    response.raise_for_status()
    return parse_prometheus(response.text)


async def tokenize_count(
    client: httpx.AsyncClient, base_url: str, model: str, content: str
) -> int:
    response = await client.post(
        f"{base_url}/tokenize",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
    )
    response.raise_for_status()
    return int(response.json()["count"])


async def exact_prompt(
    client: httpx.AsyncClient, base_url: str, model: str, target: int
) -> str:
    """Build deterministic chat content whose templated token count is exact."""
    prefix = "Scheduler collision benchmark. Continue producing detailed text."

    async def count_for(repetitions: int) -> int:
        return await tokenize_count(
            client, base_url, model, prefix + (" x" * repetitions)
        )

    low, high = 0, max(32, target * 2)
    while await count_for(high) < target:
        high *= 2
    while low <= high:
        middle = (low + high) // 2
        count = await count_for(middle)
        if count == target:
            return prefix + (" x" * middle)
        if count < target:
            low = middle + 1
        else:
            high = middle - 1

    # Tokenization is usually linear for the repeated token. Search the small
    # boundary neighborhood in case a merge makes the binary-search crossing
    # skip the exact count.
    for repetitions in range(max(0, high - 32), low + 33):
        content = prefix + (" x" * repetitions)
        if await tokenize_count(client, base_url, model, content) == target:
            return content
    raise RuntimeError(f"could not construct an exact {target}-token chat prompt")


async def iter_sse(response: httpx.Response):
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


async def decode_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    content: str,
    max_tokens: int,
    trigger_tokens: int,
    trigger: asyncio.Event,
    cancel: asyncio.Event,
    state: dict[str, Any],
    salt: str,
) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "cache_salt": salt,
    }
    started = time.monotonic()
    state.update(started_at=started, usage_samples=[], chunk_times=[])
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise RuntimeError(f"decode HTTP {response.status_code}: {body[:1000]}")
        last_usage = 0
        async for event in iter_sse(response):
            now = time.monotonic()
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                current = int(usage["completion_tokens"])
                if current > last_usage:
                    state["usage_samples"].append((now, current))
                    state["completion_tokens"] = current
                    last_usage = current
                    if current >= trigger_tokens and not trigger.is_set():
                        state["trigger_at"] = now
                        state["trigger_count"] = current
                        trigger.set()
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning") or delta.get("reasoning_content") or delta.get("content"):
                    state["chunk_times"].append(now)
            if cancel.is_set():
                state["cancelled_at"] = now
                return
    state["finished_at"] = time.monotonic()


async def prefill_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    content: str,
    salt: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "cache_salt": salt,
    }
    started = time.monotonic()
    first_token = None
    usage: dict[str, Any] = {}
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise RuntimeError(f"prefill HTTP {response.status_code}: {body[:1000]}")
        async for event in iter_sse(response):
            now = time.monotonic()
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if (
                    first_token is None
                    and (delta.get("reasoning") or delta.get("reasoning_content") or delta.get("content"))
                ):
                    first_token = now
    finished = time.monotonic()
    first_token = first_token or finished
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft = first_token - started
    return {
        "started_at": started,
        "first_token_at": first_token,
        "finished_at": finished,
        "ttft_seconds": ttft,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": prompt_tokens / ttft if prompt_tokens and ttft else None,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def token_count_at(samples: list[tuple[float, int]], timestamp: float) -> int:
    count = 0
    for sample_time, sample_count in samples:
        if sample_time > timestamp:
            break
        count = sample_count
    return count


def rate_over_last_tokens(
    samples: list[tuple[float, int]], end_count: int, token_window: int
) -> float | None:
    if len(samples) < 2:
        return None
    end_time = next((t for t, c in reversed(samples) if c <= end_count), samples[-1][0])
    target = max(0, end_count - token_window)
    start_time, start_count = samples[0]
    for sample_time, sample_count in samples:
        if sample_count >= target:
            start_time, start_count = sample_time, sample_count
            break
    elapsed = end_time - start_time
    return (end_count - start_count) / elapsed if elapsed > 0 else None


async def wait_for_count(
    state: dict[str, Any], target: int, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if int(state.get("completion_tokens") or 0) >= target:
            return True
        if state.get("finished_at"):
            return False
        await asyncio.sleep(0.05)
    return False


async def metrics_sampler(
    client: httpx.AsyncClient,
    base_url: str,
    stop: asyncio.Event,
    output: list[dict[str, Any]],
) -> None:
    while not stop.is_set():
        try:
            output.append({"time": time.monotonic(), **await scrape_metrics(client, base_url)})
        except Exception as error:  # Metrics are corroboration, not the primary signal.
            output.append({"time": time.monotonic(), "error": str(error)})
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass


async def wait_for_idle(
    client: httpx.AsyncClient, base_url: str, timeout_seconds: float = 60.0
) -> dict[str, float]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, float] = {}
    while time.monotonic() < deadline:
        last = await scrape_metrics(client, base_url)
        if (
            last.get("vllm:num_requests_running", 0.0) == 0.0
            and last.get("vllm:num_requests_waiting", 0.0) == 0.0
        ):
            return last
        await asyncio.sleep(1.0)
    raise RuntimeError(f"server did not become idle before test: {last}")


async def one_run(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    decode_content: str,
    prefill_content: str,
    run_number: int,
) -> dict[str, Any]:
    idle_metrics = await wait_for_idle(client, args.base_url)
    run_id = f"collision-{run_number}-{uuid.uuid4().hex}"
    trigger = asyncio.Event()
    cancel = asyncio.Event()
    metrics_stop = asyncio.Event()
    state: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    sampler = asyncio.create_task(
        metrics_sampler(client, args.base_url, metrics_stop, metrics)
    )
    decoder = asyncio.create_task(
        decode_stream(
            client,
            args.base_url,
            args.model,
            decode_content,
            args.decode_max_tokens,
            args.trigger_output_tokens,
            trigger,
            cancel,
            state,
            f"{run_id}-decode",
        )
    )
    try:
        await asyncio.wait_for(trigger.wait(), timeout=args.trigger_timeout)
        prefill = await prefill_request(
            client,
            args.base_url,
            args.model,
            prefill_content,
            os.environ.get("WARM_SALT","") or f"{run_id}-prefill",
        )
        collision_start = prefill["started_at"]
        collision_end = prefill["first_token_at"]
        end_count = token_count_at(state["usage_samples"], collision_end)
        recovery_target = end_count + args.recovery_tokens
        recovered = await wait_for_count(state, recovery_target, args.recovery_timeout)
    finally:
        cancel.set()
        try:
            await asyncio.wait_for(decoder, timeout=10.0)
        except asyncio.TimeoutError:
            decoder.cancel()
        metrics_stop.set()
        await sampler

    samples = state["usage_samples"]
    start_count = token_count_at(samples, collision_start)
    end_count = token_count_at(samples, collision_end)
    collision_seconds = collision_end - collision_start
    collision_tokens = max(0, end_count - start_count)
    collision_tps = collision_tokens / collision_seconds if collision_seconds > 0 else None
    baseline_tps = rate_over_last_tokens(
        samples, start_count, args.baseline_window_tokens
    )
    slowdown = (
        100.0 * (1.0 - collision_tps / baseline_tps)
        if baseline_tps and collision_tps is not None
        else None
    )

    post_end_count = int(state.get("completion_tokens") or end_count)
    post_samples = [(t, c) for t, c in samples if t >= collision_end]
    recovery_tps = None
    if post_samples and post_end_count > end_count:
        recovery_seconds = post_samples[-1][0] - collision_end
        if recovery_seconds > 0:
            recovery_tps = (post_end_count - end_count) / recovery_seconds

    chunk_times = state["chunk_times"]
    gaps = [
        current - previous
        for previous, current in zip(chunk_times, chunk_times[1:])
        if collision_start <= current <= collision_end
    ]
    metric_window = [
        row for row in metrics if collision_start <= row["time"] <= collision_end
    ]
    running = [row.get("vllm:num_requests_running", 0.0) for row in metric_window]
    waiting = [row.get("vllm:num_requests_waiting", 0.0) for row in metric_window]
    prompt_counter_delta = None
    generation_counter_delta = None
    if len(metric_window) >= 2:
        prompt_counter_delta = metric_window[-1].get("vllm:prompt_tokens_total", 0.0) - metric_window[0].get(
            "vllm:prompt_tokens_total", 0.0
        )
        generation_counter_delta = metric_window[-1].get(
            "vllm:generation_tokens_total", 0.0
        ) - metric_window[0].get("vllm:generation_tokens_total", 0.0)

    return {
        "run": run_number,
        "run_id": run_id,
        "idle_metrics_before": idle_metrics,
        "decode": {
            "trigger_count": int(state.get("trigger_count") or 0),
            "baseline_window_tokens": args.baseline_window_tokens,
            "baseline_tokens_per_second": baseline_tps,
            "collision_start_count": start_count,
            "collision_end_count": end_count,
            "collision_tokens": collision_tokens,
            "collision_seconds": collision_seconds,
            "collision_tokens_per_second": collision_tps,
            "slowdown_percent": slowdown,
            "recovery_target_tokens": args.recovery_tokens,
            "recovery_reached": recovered,
            "recovery_tokens_per_second": recovery_tps,
            "inter_chunk_gap_seconds": summarize(gaps),
        },
        "prefill": prefill,
        "scheduler": {
            "samples": len(metric_window),
            "running_requests": summarize(running),
            "waiting_requests": summarize(waiting),
            "prompt_tokens_counter_delta": prompt_counter_delta,
            "generation_tokens_counter_delta": generation_counter_delta,
            "external_traffic_suspected": bool(running and max(running) > 2),
        },
    }


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(None, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        decode_content = await exact_prompt(
            client, args.base_url, args.model, args.decode_prompt_tokens
        )
        prefill_content = await exact_prompt(
            client, args.base_url, args.model, args.prefill_prompt_tokens
        )
        verified_decode = await tokenize_count(
            client, args.base_url, args.model, decode_content
        )
        verified_prefill = await tokenize_count(
            client, args.base_url, args.model, prefill_content
        )
        runs = []
        for run_number in range(1, args.repeats + 1):
            run = await one_run(
                client, args, decode_content, prefill_content, run_number
            )
            runs.append(run)
            decode = run["decode"]
            prefill = run["prefill"]
            print(
                f"run={run_number} baseline={decode['baseline_tokens_per_second']:.2f} "
                f"collision={decode['collision_tokens_per_second']:.2f} "
                f"slowdown={decode['slowdown_percent']:.2f}% "
                f"prefill={prefill['prompt_tokens_per_second']:.2f} tok/s "
                f"ttft={prefill['ttft_seconds']:.3f}s "
                f"max_gap={decode['inter_chunk_gap_seconds']['max']:.3f}s "
                f"contaminated={run['scheduler']['external_traffic_suspected']}"
            )

    def build_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        baseline = [run["decode"]["baseline_tokens_per_second"] for run in selected]
        collision = [run["decode"]["collision_tokens_per_second"] for run in selected]
        slowdown = [run["decode"]["slowdown_percent"] for run in selected]
        prefill_tps = [run["prefill"]["prompt_tokens_per_second"] for run in selected]
        max_gaps = [run["decode"]["inter_chunk_gap_seconds"]["max"] for run in selected]
        return {
            "runs": len(selected),
            "baseline_decode_tokens_per_second": summarize(baseline),
            "collision_decode_tokens_per_second": summarize(collision),
            "decode_slowdown_percent": summarize(slowdown),
            "prefill_tokens_per_second": summarize(prefill_tps),
            "max_decode_inter_chunk_gap_seconds": summarize(max_gaps),
        }

    clean_runs = [
        run for run in runs if not run["scheduler"]["external_traffic_suspected"]
    ]
    return {
        "metadata": {
            "base_url": args.base_url,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decode_prompt_tokens": verified_decode,
            "prefill_prompt_tokens": verified_prefill,
            "trigger_output_tokens": args.trigger_output_tokens,
            "decode_max_tokens": args.decode_max_tokens,
            "repeats": args.repeats,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "unique_cache_salt_per_request": True,
        },
        "summary": build_summary(runs),
        "clean_summary": build_summary(clean_runs),
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--decode-prompt-tokens", type=int, default=4094)
    parser.add_argument("--prefill-prompt-tokens", type=int, default=65535)
    parser.add_argument("--trigger-output-tokens", type=int, default=2048)
    parser.add_argument("--decode-max-tokens", type=int, default=8192)
    parser.add_argument("--baseline-window-tokens", type=int, default=512)
    parser.add_argument("--recovery-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trigger-timeout", type=float, default=180.0)
    parser.add_argument("--recovery-timeout", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={output}")


if __name__ == "__main__":
    main()
