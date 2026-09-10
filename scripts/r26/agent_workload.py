#!/usr/bin/env python3
"""Replay deterministic cyclic-agent traffic with contending cold prefills.

The user-level offered schedule is deterministic: each synthetic session
replays growing agent turns at fixed offsets, with one in-flight turn per
session and client-side wait charged to TTFT. Policy and image A/B arms receive
identical prompts and offered arrivals. Every request remains in the JSON,
including late, failed, timed-out, and drain-censored work.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


NATIVE_POLICY_FIELDS = (
    "prefill_compute_share",
    "prefill_compute_half_life",
    "max_parallel_prefills",
    "prefill_interleave_policy",
    "decode_reservoir_low_watermark",
)
LEGACY_POLICY_FIELDS = (
    "fairness_engine",
    "prefill_compute_share",
    "max_num_prefill_tokens_per_step",
    "max_num_partial_prefills",
    "decode_prefill_min_decode_steps",
    "decode_prefill_max_wait_ms",
)
POLICY_PATH = "/prefill_fairness"
METRIC_NAMES = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:scheduler_compute_seconds_total",
    "vllm:scheduler_prefill_compute_share",
    "vllm:scheduler_compute_pressure",
    "vllm:scheduler_local_prefill_backlog_tokens",
    "vllm:kv_cache_usage_perc",
    "vllm:cache_config_info",
}
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class PromptArtifact:
    content: str
    target_tokens: int
    observed_tokens: int
    sha256: str
    characters: int


@dataclass(frozen=True)
class RequestSpec:
    kind: str
    scheduled_offset_seconds: float
    request_key: str
    session_index: int | None
    turn_index: int | None
    cold_index: int | None
    target_prompt_tokens: int | None
    payload: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_at(origin_unix: float, offset_seconds: float) -> str:
    return datetime.fromtimestamp(
        origin_unix + offset_seconds, tz=timezone.utc
    ).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def linear_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0.0:
        return None
    return sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / denominator


def decode_label(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def parse_labels(metric: str) -> dict[str, str]:
    if "{" not in metric:
        return {}
    label_text = metric.split("{", 1)[1].rsplit("}", 1)[0]
    return {match.group(1): decode_label(match.group(2)) for match in LABEL_RE.finditer(label_text)}


def parse_metrics(text: str) -> dict[str, Any]:
    values: dict[str, float] = {}
    cache_configs: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pieces = line.rsplit(None, 1)
        if len(pieces) != 2:
            continue
        metric, raw_value = pieces
        name = metric.split("{", 1)[0]
        if name not in METRIC_NAMES:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = parse_labels(metric)
        if name == "vllm:cache_config_info":
            cache_configs.append({"labels": labels, "value": value})
            continue
        key = name
        metric_class = labels.get("class")
        if name in {
            "vllm:scheduler_compute_seconds_total",
            "vllm:scheduler_compute_pressure",
        } and metric_class:
            key = f"{name}:{metric_class}"
        values[key] = values.get(key, 0.0) + value
    return {"values": values, "cache_configs": cache_configs}


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    response = await client.get(f"{base_url}/metrics", timeout=10.0)
    response.raise_for_status()
    parsed = parse_metrics(response.text)
    parsed["raw"] = response.text
    parsed["observed_at_utc"] = utc_now()
    return parsed


async def wait_idle(
    client: httpx.AsyncClient,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = await scrape_metrics(client, base_url)
        values = last["values"]
        if (
            values.get("vllm:num_requests_running", 0.0) == 0.0
            and values.get("vllm:num_requests_waiting", 0.0) == 0.0
        ):
            return last
        await asyncio.sleep(0.5)
    raise RuntimeError(f"server did not return to idle; last metrics={last}")


def detect_policy_schema(config: object) -> str:
    if not isinstance(config, dict):
        return "unknown"
    if all(field in config for field in NATIVE_POLICY_FIELDS):
        return "native"
    if "fairness_engine" in config:
        return "legacy"
    return "unknown"


async def get_policy(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    observed = utc_now()
    try:
        response = await client.get(f"{base_url}{POLICY_PATH}", timeout=15.0)
        text = response.text
        try:
            body: object = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None
        return {
            "observed_at_utc": observed,
            "status_code": response.status_code,
            "body": body,
            "body_text": text,
            "schema": detect_policy_schema(body) if response.status_code == 200 else "unavailable",
        }
    except Exception as error:
        return {
            "observed_at_utc": observed,
            "status_code": None,
            "body": None,
            "body_text": "",
            "schema": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }


async def post_policy(
    client: httpx.AsyncClient,
    base_url: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    observed = utc_now()
    try:
        response = await client.post(
            f"{base_url}{POLICY_PATH}", json=config, timeout=15.0
        )
        text = response.text
        try:
            body: object = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None
        return {
            "observed_at_utc": observed,
            "status_code": response.status_code,
            "request": config,
            "body": body,
            "body_text": text,
        }
    except Exception as error:
        return {
            "observed_at_utc": observed,
            "status_code": None,
            "request": config,
            "body": None,
            "body_text": "",
            "error": f"{type(error).__name__}: {error}",
        }


def values_equal(left: object, right: object) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def policy_mismatches(
    actual: object,
    expected: dict[str, Any] | None,
) -> dict[str, dict[str, object]]:
    if expected is None:
        return {}
    if not isinstance(actual, dict):
        return {"config": {"expected": expected, "actual": actual}}
    mismatches: dict[str, dict[str, object]] = {}
    for field, wanted in expected.items():
        observed = actual.get(field)
        if not values_equal(observed, wanted):
            mismatches[field] = {"expected": wanted, "actual": observed}
    return mismatches


async def tokenize_count(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    content: str,
) -> int:
    async with asyncio.timeout(args.tokenize_timeout_seconds):
        response = await client.post(
            f"{args.base_url}/tokenize",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
            },
        )
        response.raise_for_status()
        body = response.json()
    count = int(body["count"])
    if count <= 0:
        raise RuntimeError(f"tokenizer returned invalid count {count}")
    return count


async def exact_prompt(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    target_tokens: int,
    identity: str,
    purpose: str,
) -> PromptArtifact:
    prefix = (
        "Deterministic R26 mixed-agent scheduler qualification input.\n"
        f"TRACE-ID: {identity}\n"
        f"PURPOSE: {purpose}\n"
        "BEGIN UNIQUE CONTEXT\n"
    )
    suffix = (
        "\nEND UNIQUE CONTEXT\n"
        "Preserve the context and produce the requested technical continuation."
    )
    base_count = await tokenize_count(client, args, prefix + suffix)
    repetitions = max(target_tokens - base_count, 0)
    for _ in range(8):
        content = prefix + (" x" * repetitions) + suffix
        observed = await tokenize_count(client, args, content)
        if observed == target_tokens:
            return PromptArtifact(
                content=content,
                target_tokens=target_tokens,
                observed_tokens=observed,
                sha256=sha256_text(content),
                characters=len(content),
            )
        repetitions = max(repetitions + target_tokens - observed, 0)
    raise RuntimeError(
        f"cannot construct exact {target_tokens}-token prompt for {identity}; "
        f"last observed={observed}"
    )


def synthetic_user(turn_index: int, filler_tokens: int) -> str:
    return (
        f"Agent turn {turn_index}: inspect the accumulated state, choose the next "
        "implementation action, and explain it precisely."
        + (" detail" * filler_tokens)
    )


def synthetic_assistant(turn_index: int, filler_tokens: int) -> str:
    return (
        f"Synthetic prior agent result {turn_index}; state and tool observations retained."
        + (" result" * filler_tokens)
    )


def turn_messages(
    base_prompt: str,
    turn_index: int,
    incremental_filler_tokens: int,
    synthetic_assistant_tokens: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "user", "content": base_prompt}]
    messages.append(
        {
            "role": "assistant",
            "content": synthetic_assistant(0, synthetic_assistant_tokens),
        }
    )
    for prior_turn in range(1, turn_index):
        messages.append(
            {
                "role": "user",
                "content": synthetic_user(prior_turn, incremental_filler_tokens),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": synthetic_assistant(prior_turn, synthetic_assistant_tokens),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": synthetic_user(turn_index, incremental_filler_tokens),
        }
    )
    return messages


def chat_payload(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    cache_salt: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "reasoning_effort": "low",
        "cache_salt": cache_salt,
    }


def payload_shape_hash(payload: dict[str, Any]) -> str:
    shape = {key: value for key, value in payload.items() if key != "cache_salt"}
    return sha256_text(canonical_json(shape))


def initialize_request_row(
    spec: RequestSpec,
    origin_unix: float,
) -> dict[str, Any]:
    return {
        "request_key": spec.request_key,
        "kind": spec.kind,
        "session_index": spec.session_index,
        "turn_index": spec.turn_index,
        "cold_index": spec.cold_index,
        "target_prompt_tokens": spec.target_prompt_tokens,
        "scheduled_offset_seconds": spec.scheduled_offset_seconds,
        "scheduled_at_utc": utc_at(origin_unix, spec.scheduled_offset_seconds),
        "status": "scheduled",
        "payload_shape_sha256": payload_shape_hash(spec.payload),
        "cache_salt_sha256": sha256_text(str(spec.payload["cache_salt"])),
        "message_count": len(spec.payload["messages"]),
        "max_tokens": int(spec.payload["max_tokens"]),
        "request_started_offset_seconds": None,
        "client_queue_seconds": None,
        "first_token_offset_seconds": None,
        "finished_offset_seconds": None,
        "ttft_seconds": None,
        "turn_end_to_end_seconds": None,
        "http_active_seconds": None,
        "http_status_code": None,
        "done_received": False,
        "sse_data_events": 0,
        "malformed_sse_events": [],
        "response_ids": [],
        "finish_reasons": [],
        "usage": {},
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "chunk_offsets_seconds": [],
        "decode_chunk_gaps_seconds": [],
        "content": "",
        "reasoning_content": "",
        "content_sha256": sha256_text(""),
        "reasoning_content_sha256": sha256_text(""),
    }


async def execute_stream(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    payload: dict[str, Any],
    row: dict[str, Any],
    origin_monotonic: float,
    origin_unix: float,
) -> None:
    started = time.monotonic()
    started_offset = started - origin_monotonic
    row["request_started_offset_seconds"] = started_offset
    row["request_started_at_utc"] = utc_at(origin_unix, started_offset)
    row["client_queue_seconds"] = (
        started_offset - float(row["scheduled_offset_seconds"])
    )
    row["status"] = "in_flight"
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    response_ids: set[str] = set()
    chunk_offsets: list[float] = []
    usage: dict[str, Any] = {}
    first_token_offset: float | None = None
    last_usage_completion_tokens = 0
    try:
        async with client.stream(
            "POST", f"{args.base_url}/v1/chat/completions", json=payload
        ) as response:
            row["http_status_code"] = response.status_code
            row["response_header_request_id"] = response.headers.get("x-request-id")
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                row["error_response_body"] = body
                raise RuntimeError(f"HTTP {response.status_code}: {body[:1000]}")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_text = line[6:]
                if payload_text == "[DONE]":
                    row["done_received"] = True
                    break
                try:
                    event = json.loads(payload_text)
                except json.JSONDecodeError as error:
                    row["malformed_sse_events"].append(
                        {"payload": payload_text[:1000], "error": str(error)}
                    )
                    raise RuntimeError(f"malformed SSE JSON: {error}") from error
                row["sse_data_events"] += 1
                event_id = event.get("id")
                if isinstance(event_id, str) and event_id:
                    response_ids.add(event_id)
                event_usage = event.get("usage")
                usage_advanced = False
                if isinstance(event_usage, dict):
                    usage = event_usage
                    current_usage_completion = int(
                        event_usage.get("completion_tokens") or 0
                    )
                    if current_usage_completion > last_usage_completion_tokens:
                        offset = time.monotonic() - origin_monotonic
                        chunk_offsets.append(offset)
                        if first_token_offset is None:
                            first_token_offset = offset
                        last_usage_completion_tokens = current_usage_completion
                        usage_advanced = True
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    row["finish_reasons"].append(finish_reason)
                delta = choice.get("delta") or {}
                visible = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if visible:
                    visible_parts.append(str(visible))
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                if (visible or reasoning) and not usage_advanced:
                    offset = time.monotonic() - origin_monotonic
                    chunk_offsets.append(offset)
                    if first_token_offset is None:
                        first_token_offset = offset
        if not row["done_received"]:
            raise RuntimeError("stream ended without [DONE]")
    finally:
        finished_offset = time.monotonic() - origin_monotonic
        row["finished_offset_seconds"] = finished_offset
        row["finished_at_utc"] = utc_at(origin_unix, finished_offset)
        row["response_ids"] = sorted(response_ids)
        row["usage"] = usage
        row["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
        row["completion_tokens"] = int(usage.get("completion_tokens") or 0)
        row["chunk_offsets_seconds"] = chunk_offsets
        row["decode_chunk_gaps_seconds"] = [
            current - previous for previous, current in zip(chunk_offsets, chunk_offsets[1:])
        ]
        row["content"] = "".join(visible_parts)
        row["reasoning_content"] = "".join(reasoning_parts)
        row["content_sha256"] = sha256_text(row["content"])
        row["reasoning_content_sha256"] = sha256_text(row["reasoning_content"])
        if first_token_offset is not None:
            row["first_token_offset_seconds"] = first_token_offset
            row["first_token_at_utc"] = utc_at(origin_unix, first_token_offset)
            row["ttft_seconds"] = (
                first_token_offset - float(row["scheduled_offset_seconds"])
            )
        row["turn_end_to_end_seconds"] = (
            finished_offset - float(row["scheduled_offset_seconds"])
        )
        row["http_active_seconds"] = finished_offset - started_offset


async def perform_scheduled_request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    spec: RequestSpec,
    origin_monotonic: float,
    origin_unix: float,
    row: dict[str, Any],
    session_lock: asyncio.Lock | None = None,
) -> dict[str, Any]:
    acquired = False
    try:
        delay = origin_monotonic + spec.scheduled_offset_seconds - time.monotonic()
        if delay > 0.0:
            await asyncio.sleep(delay)
        if session_lock is not None:
            row["status"] = "client_queued"
            await session_lock.acquire()
            acquired = True
        async with asyncio.timeout(args.request_timeout_seconds):
            await execute_stream(
                client,
                args,
                spec.payload,
                row,
                origin_monotonic,
                origin_unix,
            )
        row["status"] = "completed"
    except asyncio.CancelledError:
        row["status"] = "censored"
        row["censored_offset_seconds"] = time.monotonic() - origin_monotonic
        row["censored_at_utc"] = utc_at(origin_unix, row["censored_offset_seconds"])
        raise
    except TimeoutError as error:
        row["status"] = "timed_out"
        row["error"] = f"{type(error).__name__}: request timeout"
    except Exception as error:
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        if acquired:
            session_lock.release()
    return row


def request_integrity(row: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if row.get("status") != "completed":
        problems.append(f"status={row.get('status')}")
    if row.get("http_status_code") != 200:
        problems.append(f"http_status={row.get('http_status_code')}")
    if not row.get("done_received"):
        problems.append("missing_done")
    if row.get("malformed_sse_events"):
        problems.append("malformed_sse")
    if len(row.get("response_ids") or []) > 1:
        problems.append("response_id_changed")
    if int(row.get("prompt_tokens") or 0) <= 0:
        problems.append("missing_prompt_usage")
    if int(row.get("completion_tokens") or 0) <= 0:
        problems.append("missing_completion_usage")
    if int(row.get("sse_data_events") or 0) <= 0:
        problems.append("missing_sse_data")
    if row.get("first_token_offset_seconds") is None:
        problems.append("missing_first_token_timestamp")
    if row.get("finished_offset_seconds") is None:
        problems.append("missing_finish_timestamp")
    return not problems, problems


def profile_duration(args: argparse.Namespace, profile: str) -> float:
    durations = {
        "baseline": args.baseline_seconds,
        "periodic-128k": args.periodic_seconds,
        "short-prefill-heavy": args.short_heavy_seconds,
        "analytics-200k-burst": args.analytics_seconds,
    }
    try:
        return float(durations[profile])
    except KeyError as error:
        raise ValueError(f"unknown profile {profile!r}") from error


def cold_trace(
    args: argparse.Namespace,
    profile: str,
    duration: float,
) -> list[tuple[float, int, str]]:
    if profile == "baseline":
        return []
    if profile == "periodic-128k":
        rows: list[tuple[float, int, str]] = []
        offset = 0.0
        while offset < duration:
            rows.append((offset, args.periodic_cold_tokens, "periodic-unique-128k"))
            offset += args.cold_period_seconds
        return rows
    if profile == "short-prefill-heavy":
        rows = []
        offset = 0.0
        index = 0
        while offset < duration:
            tokens = args.short_prefill_tokens[index % len(args.short_prefill_tokens)]
            rows.append((offset, tokens, "short-prefill"))
            index += 1
            offset += args.short_prefill_period_seconds
        return rows
    if profile == "analytics-200k-burst":
        return [
            (offset, args.analytics_cold_tokens, "analytics-200k")
            for offset in args.analytics_offsets_seconds
            if offset < duration
        ]
    raise ValueError(f"unknown profile {profile!r}")


def hot_offsets(
    concurrency: int,
    duration: float,
    period_seconds: float,
) -> list[tuple[float, int, int]]:
    rows: list[tuple[float, int, int]] = []
    for session_index in range(concurrency):
        offset = period_seconds * session_index / concurrency
        turn_index = 1
        while offset < duration:
            rows.append((offset, session_index, turn_index))
            turn_index += 1
            offset += period_seconds
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))


async def build_prompt_artifacts(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    trace_id: str,
    concurrency: int,
    cold_rows: list[tuple[float, int, str]],
) -> tuple[list[PromptArtifact], list[PromptArtifact]]:
    semaphore = asyncio.Semaphore(args.tokenize_concurrency)

    async def build(target: int, identity: str, purpose: str) -> PromptArtifact:
        async with semaphore:
            return await exact_prompt(client, args, target, identity, purpose)

    sessions = await asyncio.gather(
        *[
            build(
                args.session_context_tokens,
                f"{trace_id}:session:{index:03d}",
                "hot cyclic coding-agent session state",
            )
            for index in range(concurrency)
        ]
    )
    colds = await asyncio.gather(
        *[
            build(
                target,
                f"{trace_id}:cold:{index:03d}",
                purpose,
            )
            for index, (_, target, purpose) in enumerate(cold_rows)
        ]
    )
    return list(sessions), list(colds)


async def prime_sessions(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    trace_id: str,
    session_prompts: list[PromptArtifact],
) -> list[dict[str, Any]]:
    origin_monotonic = time.monotonic()
    origin_unix = time.time()
    specs: list[RequestSpec] = []
    for session_index, artifact in enumerate(session_prompts):
        salt = f"{args.cache_namespace}:{trace_id}:session:{session_index:03d}"
        payload = chat_payload(
            args,
            [{"role": "user", "content": artifact.content}],
            salt,
            args.prime_decode_tokens,
        )
        specs.append(
            RequestSpec(
                kind="prime",
                scheduled_offset_seconds=0.0,
                request_key=f"prime-session-{session_index:03d}",
                session_index=session_index,
                turn_index=0,
                cold_index=None,
                target_prompt_tokens=artifact.target_tokens,
                payload=payload,
            )
        )
    rows = [initialize_request_row(spec, origin_unix) for spec in specs]
    await asyncio.gather(
        *[
            perform_scheduled_request(
                client, args, spec, origin_monotonic, origin_unix, row
            )
            for spec, row in zip(specs, rows)
        ]
    )
    return rows


def make_request_specs(
    args: argparse.Namespace,
    trace_id: str,
    concurrency: int,
    duration: float,
    session_prompts: list[PromptArtifact],
    cold_trace_rows: list[tuple[float, int, str]],
    cold_prompts: list[PromptArtifact],
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for offset, session_index, turn_index in hot_offsets(
        concurrency, duration, args.hot_turn_period_seconds
    ):
        artifact = session_prompts[session_index]
        salt = f"{args.cache_namespace}:{trace_id}:session:{session_index:03d}"
        messages = turn_messages(
            artifact.content,
            turn_index,
            args.incremental_filler_tokens,
            args.synthetic_assistant_tokens,
        )
        specs.append(
            RequestSpec(
                kind="hot_turn",
                scheduled_offset_seconds=offset,
                request_key=f"hot-s{session_index:03d}-t{turn_index:03d}",
                session_index=session_index,
                turn_index=turn_index,
                cold_index=None,
                target_prompt_tokens=None,
                payload=chat_payload(args, messages, salt, args.turn_decode_tokens),
            )
        )
    for cold_index, ((offset, target, _), artifact) in enumerate(
        zip(cold_trace_rows, cold_prompts)
    ):
        salt = f"{args.cache_namespace}:{trace_id}:cold:{cold_index:03d}"
        specs.append(
            RequestSpec(
                kind="cold_prefill",
                scheduled_offset_seconds=offset,
                request_key=f"cold-{cold_index:03d}",
                session_index=None,
                turn_index=None,
                cold_index=cold_index,
                target_prompt_tokens=target,
                payload=chat_payload(
                    args,
                    [{"role": "user", "content": artifact.content}],
                    salt,
                    args.cold_decode_tokens,
                ),
            )
        )
    return sorted(
        specs,
        key=lambda spec: (
            spec.scheduled_offset_seconds,
            0 if spec.kind == "cold_prefill" else 1,
            spec.request_key,
        ),
    )


def build_trace_manifest(
    args: argparse.Namespace,
    profile: str,
    concurrency: int,
    repeat: int,
    duration: float,
    session_prompts: list[PromptArtifact],
    cold_prompts: list[PromptArtifact],
    specs: list[RequestSpec],
) -> dict[str, Any]:
    requests = [
        {
            "kind": spec.kind,
            "scheduled_offset_seconds": spec.scheduled_offset_seconds,
            "request_key": spec.request_key,
            "session_index": spec.session_index,
            "turn_index": spec.turn_index,
            "cold_index": spec.cold_index,
            "target_prompt_tokens": spec.target_prompt_tokens,
            "payload_shape_sha256": payload_shape_hash(spec.payload),
        }
        for spec in specs
    ]
    stable = {
        "trace_seed": args.trace_seed,
        "profile": profile,
        "concurrency": concurrency,
        "repeat": repeat,
        "duration_seconds": duration,
        "session_context_tokens": args.session_context_tokens,
        "prime_decode_tokens": args.prime_decode_tokens,
        "turn_decode_tokens": args.turn_decode_tokens,
        "incremental_filler_tokens": args.incremental_filler_tokens,
        "synthetic_assistant_tokens": args.synthetic_assistant_tokens,
        "hot_turn_period_seconds": args.hot_turn_period_seconds,
        "session_prompts": [
            {
                "target_tokens": artifact.target_tokens,
                "observed_tokens": artifact.observed_tokens,
                "sha256": artifact.sha256,
                "characters": artifact.characters,
            }
            for artifact in session_prompts
        ],
        "cold_prompts": [
            {
                "target_tokens": artifact.target_tokens,
                "observed_tokens": artifact.observed_tokens,
                "sha256": artifact.sha256,
                "characters": artifact.characters,
            }
            for artifact in cold_prompts
        ],
        "requests": requests,
    }
    return {
        **stable,
        "trace_hash": sha256_text(canonical_json(stable)),
        "cache_namespace": args.cache_namespace,
        "cache_namespace_excluded_from_trace_hash": True,
    }


async def telemetry_sampler(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    origin_monotonic: float,
    stop: asyncio.Event,
    metric_samples: list[dict[str, Any]],
    policy_samples: list[dict[str, Any]],
) -> None:
    next_policy = 0.0
    while not stop.is_set():
        offset = time.monotonic() - origin_monotonic
        try:
            snapshot = await scrape_metrics(client, args.base_url)
            metric_samples.append(
                {
                    "offset_seconds": offset,
                    "observed_at_utc": snapshot["observed_at_utc"],
                    **snapshot["values"],
                }
            )
        except Exception as error:
            metric_samples.append(
                {
                    "offset_seconds": offset,
                    "observed_at_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if offset >= next_policy:
            sample = await get_policy(client, args.base_url)
            sample["offset_seconds"] = time.monotonic() - origin_monotonic
            policy_samples.append(sample)
            next_policy = offset + args.policy_sample_seconds
        try:
            await asyncio.wait_for(stop.wait(), timeout=args.metric_sample_seconds)
        except asyncio.TimeoutError:
            pass


async def execute_policy_updates(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    origin_monotonic: float,
    updates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    for index, update in enumerate(updates):
        offset = float(update["offset_seconds"])
        delay = origin_monotonic + offset - time.monotonic()
        if delay > 0.0:
            await asyncio.sleep(delay)
        try:
            metrics = await scrape_metrics(client, args.base_url)
            values = metrics["values"]
        except Exception as error:
            values = {}
            metrics = {"error": f"{type(error).__name__}: {error}"}
        before = await get_policy(client, args.base_url)
        posted = await post_policy(client, args.base_url, dict(update["config"]))
        after = await get_policy(client, args.base_url)
        actual_after = after.get("body")
        mismatches = policy_mismatches(actual_after, dict(update["config"]))
        running = float(values.get("vllm:num_requests_running", 0.0))
        waiting = float(values.get("vllm:num_requests_waiting", 0.0))
        backlog = float(values.get("vllm:scheduler_local_prefill_backlog_tokens", 0.0))
        post_body = posted.get("body")
        applied = (
            posted.get("status_code") == 200
            and isinstance(post_body, dict)
            and post_body.get("applied") is True
        )
        before_body = before.get("body") if isinstance(before.get("body"), dict) else None
        after_body = after.get("body") if isinstance(after.get("body"), dict) else None
        observations.append(
            {
                "index": index,
                "scheduled_offset_seconds": offset,
                "actual_offset_seconds": time.monotonic() - origin_monotonic,
                "config": update["config"],
                "metrics_before": metrics,
                "running_requests_before": running,
                "waiting_requests_before": waiting,
                "local_prefill_backlog_tokens_before": backlog,
                "active_work_before": running + waiting > 0.0 or backlog > 0.0,
                "before": before,
                "post": posted,
                "after": after,
                "applied": applied,
                "readback_mismatches": mismatches,
                "readback_matches": not mismatches,
                "configured_state_changed": before_body != after_body,
            }
        )


def metric_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    key: str,
) -> float | None:
    before_value = before.get(key)
    after_value = after.get(key)
    if before_value is None or after_value is None:
        return None
    delta = float(after_value) - float(before_value)
    return delta if delta >= 0.0 else None


def model_work_split(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    decode = metric_delta(before, after, "vllm:scheduler_compute_seconds_total:decode")
    prefill = metric_delta(before, after, "vllm:scheduler_compute_seconds_total:prefill")
    total = decode + prefill if decode is not None and prefill is not None else None
    return {
        "decode_compute_seconds": decode,
        "prefill_compute_seconds": prefill,
        "total_compute_seconds": total,
        "decode_fraction": decode / total if total and decode is not None else None,
        "prefill_fraction": prefill / total if total and prefill is not None else None,
        "prompt_tokens": metric_delta(before, after, "vllm:prompt_tokens_total"),
        "generation_tokens": metric_delta(before, after, "vllm:generation_tokens_total"),
    }


def nearest_sample_values(
    samples: list[dict[str, Any]],
    target_offset: float,
) -> dict[str, Any]:
    valid = [
        sample
        for sample in samples
        if "error" not in sample
        and float(sample.get("offset_seconds", math.inf)) <= target_offset
    ]
    if not valid:
        return {}
    return max(valid, key=lambda row: float(row["offset_seconds"]))


def queue_summary(
    samples: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    window = [
        sample
        for sample in samples
        if 0.0 <= float(sample.get("offset_seconds", -1.0)) <= duration
        and "error" not in sample
    ]

    def summarize_key(key: str) -> dict[str, Any]:
        points = [
            (float(sample["offset_seconds"]), float(sample[key]))
            for sample in window
            if key in sample
        ]
        values = [point[1] for point in points]
        return {
            "samples": len(values),
            "start": values[0] if values else None,
            "end": values[-1] if values else None,
            "growth": values[-1] - values[0] if values else None,
            "slope_per_second": linear_slope(points),
            "mean": statistics.fmean(values) if values else None,
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }

    return {
        "running_requests": summarize_key("vllm:num_requests_running"),
        "waiting_requests": summarize_key("vllm:num_requests_waiting"),
        "local_prefill_backlog_tokens": summarize_key(
            "vllm:scheduler_local_prefill_backlog_tokens"
        ),
        "sample_errors": sum("error" in sample for sample in samples),
        "raw_sample_count": len(samples),
        "measurement_sample_count": len(window),
    }


def policy_telemetry_summary(samples: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    configs = [
        sample["body"]
        for sample in samples
        if 0.0 <= float(sample.get("offset_seconds", -1.0)) <= duration
        and sample.get("status_code") == 200
        and isinstance(sample.get("body"), dict)
    ]
    shares = [
        float(config["effective_prefill_compute_share"])
        for config in configs
        if isinstance(config.get("effective_prefill_compute_share"), (int, float))
    ]
    policies = [
        str(config["effective_prefill_interleave_policy"])
        for config in configs
        if config.get("effective_prefill_interleave_policy") is not None
    ]
    decode_pressures = [
        float(config["decode_pressure"])
        for config in configs
        if isinstance(config.get("decode_pressure"), (int, float))
    ]
    prefill_pressures = [
        float(config["prefill_pressure"])
        for config in configs
        if isinstance(config.get("prefill_pressure"), (int, float))
    ]
    return {
        "successful_samples": len(configs),
        "effective_prefill_compute_share": {
            **distribution(shares),
            "min": min(shares) if shares else None,
            "distinct_rounded_6dp": sorted({round(value, 6) for value in shares}),
        },
        "effective_prefill_interleave_policies": sorted(set(policies)),
        "decode_pressure": distribution(decode_pressures),
        "prefill_pressure": distribution(prefill_pressures),
    }


def annotate_incremental_prompt_tokens(
    rows: list[dict[str, Any]],
    primes: list[dict[str, Any]],
) -> None:
    previous_by_session = {
        int(row["session_index"]): int(row.get("prompt_tokens") or 0)
        for row in primes
        if row.get("status") == "completed"
        and row.get("session_index") is not None
        and int(row.get("prompt_tokens") or 0) > 0
    }
    hot_by_session: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "hot_turn" or row.get("session_index") is None:
            continue
        hot_by_session.setdefault(int(row["session_index"]), []).append(row)
    for session_index, session_rows in hot_by_session.items():
        previous = previous_by_session.get(session_index)
        for row in sorted(session_rows, key=lambda item: int(item["turn_index"])):
            current = (
                int(row.get("prompt_tokens") or 0)
                if row.get("status") == "completed"
                else 0
            )
            row["previous_prompt_tokens"] = previous
            row["incremental_prompt_tokens"] = (
                max(0, current - previous)
                if current > 0 and previous is not None
                else None
            )
            previous = current if current > 0 else None


def summarize_requests(
    rows: list[dict[str, Any]],
    concurrency: int,
    duration: float,
    metric_samples: list[dict[str, Any]],
    metrics_start: dict[str, Any],
    metrics_end: dict[str, Any],
) -> dict[str, Any]:
    hot = [row for row in rows if row["kind"] == "hot_turn"]
    cold = [row for row in rows if row["kind"] == "cold_prefill"]
    completed_hot = [row for row in hot if row["status"] == "completed"]
    completed_cold = [row for row in cold if row["status"] == "completed"]
    hot_within = [
        row
        for row in completed_hot
        if float(row.get("finished_offset_seconds") or math.inf) <= duration
    ]
    cold_within = [
        row
        for row in completed_cold
        if float(row.get("finished_offset_seconds") or math.inf) <= duration
    ]
    incomplete_at_deadline = [
        row
        for row in rows
        if row["status"] != "completed"
        or float(row.get("finished_offset_seconds") or math.inf) > duration
    ]
    unresolved = [row for row in rows if row["status"] != "completed"]
    hot_incomplete_at_deadline = [
        row
        for row in hot
        if row["status"] != "completed"
        or float(row.get("finished_offset_seconds") or math.inf) > duration
    ]
    cold_incomplete_at_deadline = [
        row
        for row in cold
        if row["status"] != "completed"
        or float(row.get("finished_offset_seconds") or math.inf) > duration
    ]
    hot_censored_lower_bounds = [
        max(0.0, duration - float(row["scheduled_offset_seconds"]))
        for row in hot_incomplete_at_deadline
    ]
    cold_censored_lower_bounds = [
        max(0.0, duration - float(row["scheduled_offset_seconds"]))
        for row in cold_incomplete_at_deadline
    ]
    hot_ttft = [float(row["ttft_seconds"]) for row in completed_hot if row["ttft_seconds"] is not None]
    hot_e2e = [
        float(row["turn_end_to_end_seconds"])
        for row in completed_hot
        if row["turn_end_to_end_seconds"] is not None
    ]
    hot_client_queue = [
        float(row["client_queue_seconds"])
        for row in completed_hot
        if row.get("client_queue_seconds") is not None
    ]
    cold_ttft = [float(row["ttft_seconds"]) for row in completed_cold if row["ttft_seconds"] is not None]
    cold_e2e = [
        float(row["turn_end_to_end_seconds"])
        for row in completed_cold
        if row["turn_end_to_end_seconds"] is not None
    ]
    decode_gaps = [
        float(gap)
        for row in completed_hot
        for gap in row.get("decode_chunk_gaps_seconds", [])
    ]
    incremental_prompt_tokens = [
        float(row["incremental_prompt_tokens"])
        for row in completed_hot
        if row.get("incremental_prompt_tokens") is not None
    ]
    per_session_offered = [
        sum(row["session_index"] == session for row in hot) for session in range(concurrency)
    ]
    per_session_completed = [
        sum(row["session_index"] == session for row in hot_within)
        for session in range(concurrency)
    ]
    measurement_values = nearest_sample_values(metric_samples, duration)
    start_values = metrics_start.get("values", {})
    end_values = metrics_end.get("values", {})
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1

    def work(rows_to_count: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "requests": len(rows_to_count),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows_to_count),
            "completion_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in rows_to_count
            ),
        }

    completed_within = hot_within + cold_within
    completed_eventually = completed_hot + completed_cold
    return {
        "absolute_counts": {
            "requests_offered": len(rows),
            "request_statuses_after_drain": statuses,
            "hot_turns_offered": len(hot),
            "hot_turns_completed_within_window": len(hot_within),
            "hot_turns_completed_after_window": len(completed_hot) - len(hot_within),
            "hot_turns_completed_total": len(completed_hot),
            "cold_requests_offered": len(cold),
            "cold_requests_completed_within_window": len(cold_within),
            "cold_requests_completed_after_window": len(completed_cold) - len(cold_within),
            "cold_requests_completed_total": len(completed_cold),
            "incomplete_or_censored_at_measurement_end": len(incomplete_at_deadline),
            "unresolved_after_drain": len(unresolved),
        },
        "completed_work": {
            "within_measurement_window": work(completed_within),
            "eventually_before_drain_cutoff": work(completed_eventually),
        },
        "normalization": {
            "hot_turns_completed_per_measurement_second": len(hot_within) / duration,
            "hot_turns_completed_per_session_second": (
                len(hot_within) / duration / concurrency
            ),
            "hot_turn_completion_fraction_by_deadline": (
                len(hot_within) / len(hot) if hot else None
            ),
            "cold_completion_fraction_by_deadline": (
                len(cold_within) / len(cold) if cold else None
            ),
            "all_requests_completed_per_measurement_second": (
                len(completed_within) / duration
            ),
            "prompt_tokens_in_completed_work_per_second": (
                sum(int(row.get("prompt_tokens") or 0) for row in completed_within)
                / duration
            ),
            "completion_tokens_in_completed_work_per_second": (
                sum(int(row.get("completion_tokens") or 0) for row in completed_within)
                / duration
            ),
        },
        "hot_ttft_seconds": distribution(hot_ttft),
        "hot_turn_end_to_end_seconds": distribution(hot_e2e),
        "hot_client_queue_before_submit_seconds": distribution(hot_client_queue),
        "hot_incremental_prefill_tokens": distribution(incremental_prompt_tokens),
        "hot_decode_chunk_gap_seconds": distribution(decode_gaps),
        "cold_ttft_seconds": distribution(cold_ttft),
        "cold_end_to_end_seconds": distribution(cold_e2e),
        "hot_incomplete_ttft_or_e2e_lower_bound_seconds": distribution(
            hot_censored_lower_bounds
        ),
        "cold_incomplete_ttft_or_e2e_lower_bound_seconds": distribution(
            cold_censored_lower_bounds
        ),
        "per_session": {
            "offered": {
                "min": min(per_session_offered),
                "p50": percentile([float(value) for value in per_session_offered], 0.5),
                "max": max(per_session_offered),
            },
            "completed_within_window": {
                "min": min(per_session_completed),
                "p50": percentile([float(value) for value in per_session_completed], 0.5),
                "max": max(per_session_completed),
            },
        },
        "queue_depths_and_growth": queue_summary(metric_samples, duration),
        "model_work_split_measurement_window": model_work_split(
            start_values, measurement_values
        ),
        "model_work_split_until_drain": model_work_split(start_values, end_values),
    }


def make_gate(name: str, passed: bool, detail: object) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


async def run_scenario(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    profile: str,
    concurrency: int,
    repeat: int,
) -> dict[str, Any]:
    duration = profile_duration(args, profile)
    trace_id = f"{args.trace_seed}:{profile}:c{concurrency}:r{repeat}"
    print(f"START {trace_id} policy={args.policy_label}", flush=True)
    boundary_before = await get_policy(client, args.base_url)
    boundary_post = await post_policy(client, args.base_url, args.expected_policy)
    boundary_after = await get_policy(client, args.base_url)
    boundary_after_config = boundary_after.get("body")
    boundary_mismatches = policy_mismatches(
        boundary_after_config, args.expected_policy
    )
    boundary_post_body = boundary_post.get("body")
    boundary_applied = (
        boundary_post.get("status_code") == 200
        and isinstance(boundary_post_body, dict)
        and boundary_post_body.get("applied") is True
        and boundary_after.get("schema") == args.expected_api_schema
        and not boundary_mismatches
    )
    boundary_receipt = {
        "before": boundary_before,
        "post": boundary_post,
        "after": boundary_after,
        "expected_schema": args.expected_api_schema,
        "expected_policy": args.expected_policy,
        "readback_mismatches": boundary_mismatches,
        "applied_and_read_back": boundary_applied,
    }
    if not boundary_applied:
        return {
            "status": "failed",
            "candidate": args.candidate_label,
            "policy": args.policy_label,
            "profile": profile,
            "concurrency": concurrency,
            "repeat": repeat,
            "measurement_seconds": duration,
            "api": {"cell_boundary_apply": boundary_receipt},
            "primes": [],
            "requests": [],
            "gates": [
                make_gate(
                    "policy_applied_and_read_back_at_cell_boundary",
                    False,
                    boundary_receipt,
                )
            ],
        }
    cold_trace_rows = cold_trace(args, profile, duration)
    session_prompts, cold_prompts = await build_prompt_artifacts(
        client, args, trace_id, concurrency, cold_trace_rows
    )
    primes = await prime_sessions(client, args, trace_id, session_prompts)
    prime_integrity = [request_integrity(row) for row in primes]
    if not all(passed for passed, _ in prime_integrity):
        return {
            "status": "failed",
            "candidate": args.candidate_label,
            "policy": args.policy_label,
            "profile": profile,
            "concurrency": concurrency,
            "repeat": repeat,
            "measurement_seconds": duration,
            "primes": primes,
            "requests": [],
            "gates": [
                make_gate(
                    "all_hot_sessions_primed",
                    False,
                    {
                        row["request_key"]: problems
                        for row, (_, problems) in zip(primes, prime_integrity)
                        if problems
                    },
                )
            ],
        }
    await wait_idle(client, args.base_url, args.idle_timeout_seconds)
    await asyncio.sleep(args.cooldown_seconds)

    specs = make_request_specs(
        args,
        trace_id,
        concurrency,
        duration,
        session_prompts,
        cold_trace_rows,
        cold_prompts,
    )
    trace = build_trace_manifest(
        args,
        profile,
        concurrency,
        repeat,
        duration,
        session_prompts,
        cold_prompts,
        specs,
    )
    initial_policy = await get_policy(client, args.base_url)
    initial_mismatches = policy_mismatches(
        initial_policy.get("body"), args.expected_policy
    )
    metrics_start = await scrape_metrics(client, args.base_url)

    origin_monotonic = time.monotonic() + args.start_delay_seconds
    origin_unix = time.time() + (origin_monotonic - time.monotonic())
    rows = [initialize_request_row(spec, origin_unix) for spec in specs]
    metric_samples: list[dict[str, Any]] = []
    policy_samples: list[dict[str, Any]] = []
    update_observations: list[dict[str, Any]] = []
    sampler_stop = asyncio.Event()
    sampler = asyncio.create_task(
        telemetry_sampler(
            client,
            args,
            origin_monotonic,
            sampler_stop,
            metric_samples,
            policy_samples,
        )
    )
    updater = asyncio.create_task(
        execute_policy_updates(
            client,
            args,
            origin_monotonic,
            args.policy_updates,
            update_observations,
        )
    )
    session_locks = {
        session_index: asyncio.Lock() for session_index in range(concurrency)
    }
    request_tasks = [
        asyncio.create_task(
            perform_scheduled_request(
                client,
                args,
                spec,
                origin_monotonic,
                origin_unix,
                row,
                (
                    session_locks[int(spec.session_index)]
                    if spec.kind == "hot_turn" and spec.session_index is not None
                    else None
                ),
            )
        )
        for spec, row in zip(specs, rows)
    ]
    drain_deadline = origin_monotonic + duration + args.drain_timeout_seconds
    remaining = max(0.0, drain_deadline - time.monotonic())
    _, pending = await asyncio.wait(request_tasks, timeout=remaining)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await updater
    sampler_stop.set()
    await sampler

    idle_after = True
    idle_error: str | None = None
    try:
        await wait_idle(client, args.base_url, args.idle_timeout_seconds)
    except Exception as error:
        idle_after = False
        idle_error = f"{type(error).__name__}: {error}"
    metrics_end = await scrape_metrics(client, args.base_url)
    final_policy = await get_policy(client, args.base_url)

    annotate_incremental_prompt_tokens(rows, primes)
    integrity = {row["request_key"]: request_integrity(row) for row in rows}
    bad_integrity = {
        key: problems for key, (passed, problems) in integrity.items() if not passed
    }
    summary = summarize_requests(
        rows,
        concurrency,
        duration,
        metric_samples,
        metrics_start,
        metrics_end,
    )
    policy_summary = policy_telemetry_summary(policy_samples, duration)
    start_values = metrics_start.get("values", {})
    required_metric_keys = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:scheduler_compute_seconds_total:decode",
        "vllm:scheduler_compute_seconds_total:prefill",
    }
    if args.expected_api_schema == "native":
        required_metric_keys.update(
            {
                "vllm:scheduler_prefill_compute_share",
                "vllm:scheduler_local_prefill_backlog_tokens",
            }
        )
    if args.expected_policy.get("prefill_compute_share") == "auto":
        required_metric_keys.update(
            {
                "vllm:scheduler_compute_pressure:decode",
                "vllm:scheduler_compute_pressure:prefill",
            }
        )
    missing_metrics = sorted(required_metric_keys - set(start_values))
    cache_capacities = sorted(
        {
            int(config["labels"]["kv_cache_size_tokens"])
            for config in metrics_start.get("cache_configs", [])
            if config.get("labels", {}).get("kv_cache_size_tokens", "").isdigit()
        }
    )
    work_split_until_drain = summary["model_work_split_until_drain"]
    work_split_observed = (
        work_split_until_drain["decode_compute_seconds"] is not None
        and work_split_until_drain["prefill_compute_seconds"] is not None
        and work_split_until_drain["total_compute_seconds"] is not None
        and float(work_split_until_drain["total_compute_seconds"]) > 0.0
    )
    update_failures = [
        observation
        for observation in update_observations
        if not (
            observation["active_work_before"]
            and observation["applied"]
            and observation["readback_matches"]
            and observation["configured_state_changed"]
        )
    ]
    gates = [
        make_gate(
            "policy_applied_and_read_back_at_cell_boundary",
            boundary_applied,
            boundary_receipt,
        ),
        make_gate(
            "all_hot_sessions_primed",
            True,
            {"sessions": len(primes)},
        ),
        make_gate(
            "policy_schema_and_readback_match",
            initial_policy.get("schema") == args.expected_api_schema
            and not initial_mismatches,
            {
                "expected_schema": args.expected_api_schema,
                "observed_schema": initial_policy.get("schema"),
                "mismatches": initial_mismatches,
                "readback": initial_policy,
            },
        ),
        make_gate(
            "entire_deterministic_trace_offered",
            len(rows) == len(specs),
            {"planned": len(specs), "observed": len(rows)},
        ),
        make_gate(
            "no_lost_or_corrupt_responses",
            not bad_integrity,
            {
                "offered": len(rows),
                "completed_and_valid": len(rows) - len(bad_integrity),
                "failures": bad_integrity,
            },
        ),
        make_gate(
            "required_scheduler_metrics_present",
            not missing_metrics and bool(cache_capacities),
            {
                "missing": missing_metrics,
                "kv_cache_size_tokens": cache_capacities,
            },
        ),
        make_gate(
            "model_prefill_decode_work_split_observed",
            work_split_observed,
            work_split_until_drain,
        ),
        make_gate(
            "server_idle_after_drain",
            idle_after,
            {"error": idle_error},
        ),
    ]
    if args.expected_api_schema == "native":
        configured_policies = {
            str(args.expected_policy["prefill_interleave_policy"]),
            *{
                str(update["config"]["prefill_interleave_policy"])
                for update in args.policy_updates
            },
        }
        allowed_effective_policies: set[str] = set()
        for configured in configured_policies:
            if configured == "decode-aware":
                allowed_effective_policies.update(
                    {"round-robin", "shortest-remaining"}
                )
            else:
                allowed_effective_policies.add(configured)
        observed_effective_policies = set(
            policy_summary["effective_prefill_interleave_policies"]
        )
        gates.append(
            make_gate(
                "effective_interleave_policy_read_back",
                bool(observed_effective_policies)
                and observed_effective_policies <= allowed_effective_policies,
                {
                    "configured": sorted(configured_policies),
                    "allowed_effective": sorted(allowed_effective_policies),
                    "observed_effective": sorted(observed_effective_policies),
                },
            )
        )
    if args.expected_policy and args.expected_policy.get("prefill_compute_share") == "auto":
        share_stats = policy_summary["effective_prefill_compute_share"]
        share_min = share_stats["min"]
        share_max = share_stats["max"]
        shares_in_bounds = (
            share_min is not None
            and share_max is not None
            and 0.2 <= float(share_min) <= float(share_max) <= 0.8
        )
        gates.append(
            make_gate(
                "automatic_compute_share_telemetry_in_bounds",
                shares_in_bounds,
                share_stats,
            )
        )
    if args.policy_updates:
        gates.append(
            make_gate(
                "live_policy_updates_applied_during_active_work",
                len(update_observations) == len(args.policy_updates)
                and not update_failures,
                {
                    "planned": len(args.policy_updates),
                    "observed": len(update_observations),
                    "failures": update_failures,
                },
            )
        )
    status = "complete" if all(gate["passed"] for gate in gates) else "failed"
    result = {
        "status": status,
        "candidate": args.candidate_label,
        "policy": args.policy_label,
        "profile": profile,
        "concurrency": concurrency,
        "repeat": repeat,
        "measurement_seconds": duration,
        "drain_timeout_seconds": args.drain_timeout_seconds,
        "trace": trace,
        "api": {
            "cell_boundary_apply": boundary_receipt,
            "initial_policy": initial_policy,
            "final_policy": final_policy,
            "samples": policy_samples,
            "summary": policy_summary,
            "live_updates": update_observations,
        },
        "metrics": {
            "start": metrics_start,
            "end": metrics_end,
            "samples": metric_samples,
        },
        "summary": summary,
        "primes": primes,
        "requests": rows,
        "gates": gates,
    }
    counts = summary["absolute_counts"]
    hot_p95 = summary["hot_ttft_seconds"]["p95"]
    print(
        f"DONE {trace_id} status={status} "
        f"hot={counts['hot_turns_completed_within_window']}/"
        f"{counts['hot_turns_offered']} hot_p95={hot_p95}",
        flush=True,
    )
    return result


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
    )
    timeout = httpx.Timeout(None, connect=args.connect_timeout_seconds)
    cells: list[dict[str, Any]] = []
    metadata = jsonable_args(args)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for repeat in range(1, args.repeats + 1):
            for concurrency in args.concurrencies:
                for profile in args.profiles:
                    try:
                        cell = await run_scenario(
                            client, args, profile, concurrency, repeat
                        )
                    except Exception as error:
                        cell = {
                            "status": "error",
                            "candidate": args.candidate_label,
                            "policy": args.policy_label,
                            "profile": profile,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                            "gates": [
                                make_gate(
                                    "scenario_completed",
                                    False,
                                    f"{type(error).__name__}: {error}",
                                )
                            ],
                        }
                    cells.append(cell)
                    atomic_write_json(
                        args.output,
                        {"metadata": metadata, "cells": cells},
                    )
                    await asyncio.sleep(args.between_scenarios_seconds)
    return {"metadata": metadata, "cells": cells}


def comma_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def comma_ints(value: str) -> list[int]:
    return [int(part) for part in comma_strings(value)]


def comma_floats(value: str) -> list[float]:
    return [float(part) for part in comma_strings(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5002")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--expected-api-schema", choices=("native", "legacy"), required=True)
    parser.add_argument("--expected-policy-json", default="{}")
    parser.add_argument("--policy-updates-json", default="[]")
    parser.add_argument("--profiles", default="baseline,periodic-128k")
    parser.add_argument("--concurrencies", default="8,16")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--trace-seed", default="r26-mixed-agent-v1")
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--session-context-tokens", type=int, default=8192)
    parser.add_argument("--prime-decode-tokens", type=int, default=32)
    parser.add_argument("--turn-decode-tokens", type=int, default=128)
    parser.add_argument("--cold-decode-tokens", type=int, default=1)
    parser.add_argument("--incremental-filler-tokens", type=int, default=220)
    parser.add_argument("--synthetic-assistant-tokens", type=int, default=96)
    parser.add_argument("--hot-turn-period-seconds", type=float, default=6.0)
    parser.add_argument("--baseline-seconds", type=float, default=30.0)
    parser.add_argument("--periodic-seconds", type=float, default=60.0)
    parser.add_argument("--short-heavy-seconds", type=float, default=60.0)
    parser.add_argument("--analytics-seconds", type=float, default=90.0)
    parser.add_argument("--periodic-cold-tokens", type=int, default=131072)
    parser.add_argument("--cold-period-seconds", type=float, default=15.0)
    parser.add_argument("--short-prefill-tokens", default="2048,4096,8192")
    parser.add_argument("--short-prefill-period-seconds", type=float, default=1.5)
    parser.add_argument("--analytics-cold-tokens", type=int, default=204800)
    parser.add_argument("--analytics-offsets-seconds", default="0,1,2")
    parser.add_argument("--drain-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--tokenize-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--between-scenarios-seconds", type=float, default=2.0)
    parser.add_argument("--start-delay-seconds", type=float, default=1.0)
    parser.add_argument("--metric-sample-seconds", type=float, default=0.25)
    parser.add_argument("--policy-sample-seconds", type=float, default=1.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-connections", type=int, default=256)
    parser.add_argument("--max-keepalive-connections", type=int, default=128)
    parser.add_argument("--tokenize-concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.profiles = comma_strings(args.profiles)
    args.concurrencies = comma_ints(args.concurrencies)
    args.short_prefill_tokens = comma_ints(args.short_prefill_tokens)
    args.analytics_offsets_seconds = comma_floats(args.analytics_offsets_seconds)
    args.expected_policy = json.loads(args.expected_policy_json)
    args.policy_updates = json.loads(args.policy_updates_json)
    allowed_profiles = {
        "baseline",
        "periodic-128k",
        "short-prefill-heavy",
        "analytics-200k-burst",
    }
    unknown_profiles = set(args.profiles) - allowed_profiles
    if unknown_profiles:
        parser.error(f"unknown profiles: {sorted(unknown_profiles)}")
    if not args.concurrencies or any(value <= 0 for value in args.concurrencies):
        parser.error("concurrencies must be positive")
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    if not args.short_prefill_tokens or any(value <= 0 for value in args.short_prefill_tokens):
        parser.error("short prefill token sizes must be positive")
    if args.hot_turn_period_seconds <= 0.0 or args.short_prefill_period_seconds <= 0.0:
        parser.error("arrival periods must be positive")
    if not isinstance(args.expected_policy, dict):
        parser.error("expected policy JSON must be an object")
    if not isinstance(args.policy_updates, list):
        parser.error("policy updates JSON must be an array")
    for update in args.policy_updates:
        if not isinstance(update, dict) or set(update) != {"offset_seconds", "config"}:
            parser.error("each policy update needs exactly offset_seconds and config")
        if not isinstance(update["config"], dict):
            parser.error("each policy update config must be an object")
        if set(update["config"]) != set(NATIVE_POLICY_FIELDS):
            parser.error("each live update must contain all five native policy fields")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    atomic_write_json(args.output, result)
    print(f"output={args.output}", flush=True)
    if any(cell.get("status") != "complete" for cell in result["cells"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
