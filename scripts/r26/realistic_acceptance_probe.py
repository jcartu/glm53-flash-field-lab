#!/usr/bin/env python3
"""Measure MTP acceptance on fixed, bounded, natural chat completions."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import aiohttp

SCHEMA = "r26-realistic-acceptance/v1"
DEFAULT_MODEL = "GLM-5.3-Flash-NVFP4"
DEFAULT_SEEDS = (260905, 260906)
MAX_TOKENS = 8192
TEMPERATURE = 0.2
TOP_P = 0.95
REASONING_EFFORT = "low"
MTP_DEPTH = 3

COUNTERS = {
    "generated": "vllm:generation_tokens_total",
    "draft_steps": "vllm:spec_decode_num_drafts_total",
    "proposed": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}
ACTIVITY_GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
}
METRIC_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{.*\})?\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
SYSTEM_MESSAGE = (
    "Complete the bounded task directly and accurately. Produce a visible final answer, "
    "follow every requested output constraint, and stop naturally when the answer is complete."
)
PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "id": "coding-python-collapse-ranges",
        "domain": "coding",
        "title": "Small typed Python implementation",
        "user": (
            "Implement a Python 3 function `collapse_ranges(numbers: list[int]) -> list[tuple[int, int]]`. "
            "It must ignore duplicate inputs, sort the values, and return maximal inclusive ranges of "
            "consecutive integers. It must not mutate the input and must use no external packages or I/O. "
            "Give exactly one fenced Python code block followed by an explanation of at most 120 words. "
            "Do not execute the code."
        ),
        "bound": "one code block plus at most 120 explanatory words",
    },
    {
        "id": "coding-typescript-retry-review",
        "domain": "coding",
        "title": "TypeScript bug fix and explanation",
        "user": (
            "A TypeScript retry loop currently retries every failure, including invalid user input. Write a "
            "replacement `async function withRetry<T>(operation: () => Promise<T>, shouldRetry: "
            "(error: unknown) => boolean, attempts: number): Promise<T>` that validates `attempts >= 1`, "
            "returns immediately on success, retries only when `shouldRetry` is true, and rethrows the last "
            "error otherwise. Use no dependencies. Give one fenced TypeScript block and no more than five "
            "bullets explaining edge cases. Do not execute the code."
        ),
        "bound": "one code block plus at most five bullets",
    },
    {
        "id": "technical-wal-checkpoint",
        "domain": "technical_prose",
        "title": "Storage durability explanation",
        "user": (
            "Explain to a backend engineer how a write-ahead log and periodic checkpoints work together "
            "during crash recovery. Cover ordering, durability acknowledgement, replay, checkpoint atomicity, "
            "and one operational trade-off. Use four short paragraphs, include one concrete crash example, "
            "and stay under 450 words."
        ),
        "bound": "four short paragraphs under 450 words",
    },
    {
        "id": "technical-tcp-backpressure",
        "domain": "technical_prose",
        "title": "Network backpressure explanation",
        "user": (
            "Describe TCP backpressure for an application developer whose producer is faster than its "
            "consumer. Distinguish the application buffer, socket send buffer, receiver window, and congestion "
            "control without claiming they are the same mechanism. Use exactly six numbered points and finish "
            "with a two-sentence practical example. Keep the whole response under 500 words."
        ),
        "bound": "six numbered points and a two-sentence example under 500 words",
    },
    {
        "id": "reasoning-inventory-arithmetic",
        "domain": "reasoning_math",
        "title": "Verifiable inventory arithmetic",
        "user": (
            "A warehouse receives 18 crates with 24 filters in each crate. It ships three eighths of all "
            "received filters in the morning, then ships 57 more filters in the afternoon. How many filters "
            "remain? Show the arithmetic in at most four lines and end with exactly `FINAL: <integer> filters`."
        ),
        "bound": "at most four arithmetic lines plus one final line",
        "reference": "FINAL: 213 filters",
    },
    {
        "id": "reasoning-batch-duration",
        "domain": "reasoning_math",
        "title": "Verifiable rate and duration arithmetic",
        "user": (
            "A service must process 1,260 records. For the first 4 minutes, three workers each process 72 "
            "records per minute. One worker then stops, and the two remaining workers continue at the same "
            "individual rate. Assuming continuous processing and no overlap or setup delay, how long from the "
            "start until all records are processed? Give a concise derivation and end with exactly "
            "`FINAL: <minutes> minutes <seconds> seconds`."
        ),
        "bound": "concise derivation plus one final line",
        "reference": "FINAL: 6 minutes 45 seconds",
    },
    {
        "id": "structured-service-json",
        "domain": "structured_format",
        "title": "Constrained JSON transformation",
        "user": (
            "Convert these facts into a JSON object with exactly one top-level key `services`, whose value is "
            "an array sorted by service name. Every array item must have exactly the keys `name`, `owner`, and "
            "`healthy` in that order. Facts: api is owned by Core and healthy; billing is owned by Ledger and "
            "not healthy; search is owned by Discovery and healthy; worker is owned by Core and healthy. "
            "Return valid JSON only, with no code fence or commentary."
        ),
        "bound": "one JSON object with four bounded records",
    },
    {
        "id": "structured-maintenance-yaml",
        "domain": "structured_format",
        "title": "Constrained YAML transformation",
        "user": (
            "Write a YAML document for two maintenance windows. The top-level key must be `windows`. Each item "
            "must contain `service`, `start_utc`, `duration_minutes`, and `rollback_owner` in that order. Use "
            "these facts without alteration: search, 2026-09-08T01:00:00Z, 30, Mira; billing, "
            "2026-09-08T02:15:00Z, 45, Theo. Preserve the listed item order. Return YAML only, with no document "
            "marker, code fence, anchors, or commentary."
        ),
        "bound": "one YAML document with two bounded records",
    },
)


def epoch_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def request_payload(model: str, prompt: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt["user"]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": seed,
        "reasoning_effort": REASONING_EFFORT,
        "stream": False,
    }


def input_manifest(model: str, seeds: Iterable[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    fixed_seeds = list(seeds)
    prompts = [dict(prompt) for prompt in PROMPTS]
    requests = [
        {
            "seed": seed,
            "prompt_id": prompt["id"],
            "payload": request_payload(model, prompt, seed),
        }
        for seed in fixed_seeds
        for prompt in PROMPTS
    ]
    fingerprint_material = {
        "prompts": prompts,
        "seeds": fixed_seeds,
        "requests": requests,
        "execution": {
            "c1": "one completed request per exact counter window, in prompt order",
            "c8": "the same eight prompt payloads submitted together in one exact counter window",
            "order": "for each seed, all C1 prompts followed by the matched C8 group",
        },
    }
    return {
        "name": "r26-realistic-natural-chat-v1",
        "prompt_count": len(prompts),
        "domain_counts": dict(sorted(Counter(prompt["domain"] for prompt in prompts).items())),
        "prompts": prompts,
        "request_settings": {
            "endpoint": "/v1/chat/completions",
            "model": model,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "reasoning_effort": REASONING_EFFORT,
            "seed_values": fixed_seeds,
            "stream": False,
            "natural_eos": True,
            "ignore_eos_field_present": False,
        },
        "execution": fingerprint_material["execution"],
        "request_payload_fingerprint_sha256": sha256_text(canonical_json(requests)),
        "input_set_sha256": sha256_text(canonical_json(fingerprint_material)),
    }


def parse_metric_body(raw_body: str) -> dict[str, Any]:
    wanted = {**COUNTERS, **ACTIVITY_GAUGES}
    by_metric = {metric: key for key, metric in wanted.items()}
    totals: dict[str, float] = {}
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in wanted}
    errors: list[str] = []
    for raw_line in raw_body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        metric_name, raw_value = match.groups()
        key = by_metric.get(metric_name)
        if key is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"could not parse {metric_name} value {raw_value!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"non-finite {metric_name} value {raw_value!r}")
            continue
        totals[key] = totals.get(key, 0.0) + value
        samples[key].append({"metric": metric_name, "value": value, "raw_line": raw_line})
    return {
        "counters": {key: totals[key] for key in COUNTERS if key in totals},
        "activity": {key: totals[key] for key in ACTIVITY_GAUGES if key in totals},
        "samples": samples,
        "parse_errors": errors,
    }


async def scrape_metrics(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/metrics"
    started_at = time.time()
    monotonic_started = time.monotonic()
    status: int | None = None
    headers: dict[str, str] = {}
    raw_body = ""
    errors: list[str] = []
    try:
        async with session.get(url) as response:
            status = response.status
            headers = dict(response.headers.items())
            raw_body = (await response.read()).decode("utf-8", errors="replace")
            if status != 200:
                errors.append(f"metrics HTTP status {status}")
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    parsed = parse_metric_body(raw_body)
    errors.extend(parsed["parse_errors"])
    finished_at = time.time()
    return {
        "url": url,
        "started_at": started_at,
        "started_at_iso": epoch_iso(started_at),
        "finished_at": finished_at,
        "finished_at_iso": epoch_iso(finished_at),
        "elapsed_seconds": time.monotonic() - monotonic_started,
        "status": status,
        "response_headers": headers,
        "raw_body": raw_body,
        "raw_body_sha256": sha256_text(raw_body),
        "counters": parsed["counters"],
        "activity": parsed["activity"],
        "counter_samples": {key: parsed["samples"][key] for key in COUNTERS},
        "activity_samples": {key: parsed["samples"][key] for key in ACTIVITY_GAUGES},
        "errors": errors,
    }


def numeric_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or int(value) != value:
        return None
    return int(value)


def classify_request(record: dict[str, Any]) -> None:
    errors = record["errors"]
    parsed = record.get("response")
    status = record.get("http_status")
    finish_reason: str | None = None
    visible_output = ""
    reasoning_chars = 0
    usage: dict[str, Any] | None = None

    if status != 200:
        errors.append(f"chat HTTP status {status}")
    if not isinstance(parsed, dict):
        errors.append("response JSON root is not an object")
    else:
        if parsed.get("error") is not None:
            errors.append(f"API error object: {parsed['error']!r}")
        raw_usage = parsed.get("usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage
        else:
            errors.append("response has no usage object")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            errors.append("response has no first choice object")
        else:
            choice = choices[0]
            raw_finish_reason = choice.get("finish_reason")
            if raw_finish_reason is None:
                errors.append("first choice has no finish_reason")
            else:
                finish_reason = str(raw_finish_reason)
            message = choice.get("message")
            if not isinstance(message, dict):
                errors.append("first choice has no message object")
            else:
                content = message.get("content")
                if isinstance(content, str):
                    visible_output = content
                elif content is not None:
                    errors.append("message.content is neither text nor null")
                reasoning = message.get("reasoning")
                if reasoning is None:
                    reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str):
                    reasoning_chars = len(reasoning)

    completion_tokens = numeric_int(usage.get("completion_tokens")) if usage else None
    budget_limit_reached = finish_reason == "length"
    requested_output_cap_reached = budget_limit_reached and (
        completion_tokens is None or completion_tokens >= MAX_TOKENS
    )
    has_visible_output = bool(visible_output.strip())
    if errors:
        classification = "runtime_error"
        final_completion: str | None = None
    elif finish_reason == "length":
        classification = (
            "budget_limited_with_visible_partial"
            if has_visible_output
            else "budget_limited_no_visible_final"
        )
        final_completion = None
    elif finish_reason == "stop" and has_visible_output:
        classification = "completed_visible_final"
        final_completion = visible_output
    elif finish_reason == "stop":
        classification = "completed_no_visible_final"
        final_completion = None
    elif has_visible_output:
        classification = "other_finish_with_visible_output"
        final_completion = None
    else:
        classification = "incomplete_no_visible_final"
        final_completion = None

    record.update(
        {
            "finish_reason": finish_reason,
            "usage": usage,
            "error": "; ".join(errors) if errors else None,
            "classification": classification,
            "budget_limit_reached": budget_limit_reached,
            "requested_output_cap_reached": requested_output_cap_reached,
            "visible_output": visible_output,
            "final_completion": final_completion,
            "final_completion_source": "choices[0].message.content only",
            "hidden_reasoning_used_as_final": False,
            "visible_characters": len(visible_output),
            "reasoning_characters": reasoning_chars,
        }
    )


async def post_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = request_payload(model, prompt, seed)
    raw_request = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    started_at = time.time()
    monotonic_started = time.monotonic()
    record: dict[str, Any] = {
        "prompt_id": prompt["id"],
        "domain": prompt["domain"],
        "seed": seed,
        "started_at": started_at,
        "started_at_iso": epoch_iso(started_at),
        "url": url,
        "request": payload,
        "raw_request": raw_request,
        "raw_request_sha256": sha256_text(raw_request),
        "http_status": None,
        "response_headers": {},
        "raw_response": "",
        "raw_response_sha256": sha256_text(""),
        "response": None,
        "errors": [],
    }
    try:
        async with session.post(
            url,
            data=raw_request.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ) as response:
            raw_response = (await response.read()).decode("utf-8", errors="replace")
            record["http_status"] = response.status
            record["response_headers"] = dict(response.headers.items())
            record["raw_response"] = raw_response
            record["raw_response_sha256"] = sha256_text(raw_response)
            try:
                record["response"] = json.loads(raw_response)
            except json.JSONDecodeError as error:
                record["errors"].append(
                    f"JSONDecodeError at {error.pos}: {error.msg}"
                )
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    finished_at = time.time()
    elapsed = time.monotonic() - monotonic_started
    record.update(
        {
            "finished_at": finished_at,
            "finished_at_iso": epoch_iso(finished_at),
            "elapsed": elapsed,
            "elapsed_seconds": elapsed,
            "elapsed_scope": (
                "client end-to-end request time including queueing, prefill, decode, and response transfer; "
                "not pure decode time"
            ),
        }
    )
    classify_request(record)
    return record


def counter_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float | None]:
    return {
        key: after[key] - before[key] if key in before and key in after else None
        for key in COUNTERS
    }


def group_integrity(
    concurrency: int,
    requests: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
    delta: dict[str, float | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_issues: list[str] = []
    count_issues: list[str] = []
    if len(requests) != concurrency:
        runtime_issues.append(
            f"expected {concurrency} request records, observed {len(requests)}"
        )
    for index, request in enumerate(requests):
        if request["errors"]:
            runtime_issues.append(
                f"request {index} ({request['prompt_id']}): " + "; ".join(request["errors"])
            )
        if request["classification"] == "runtime_error":
            runtime_issues.append(
                f"request {index} ({request['prompt_id']}) classified runtime_error"
            )

    for side, snapshot in (("before", before), ("after", after)):
        for error in snapshot["errors"]:
            count_issues.append(f"{side} metrics: {error}")
        missing_counters = sorted(set(COUNTERS) - set(snapshot["counters"]))
        if missing_counters:
            count_issues.append(f"{side} metrics missing counters {missing_counters}")
        missing_activity = sorted(set(ACTIVITY_GAUGES) - set(snapshot["activity"]))
        if missing_activity:
            count_issues.append(f"{side} metrics missing activity gauges {missing_activity}")
        for name in ACTIVITY_GAUGES:
            value = snapshot["activity"].get(name)
            if value is not None and value != 0:
                count_issues.append(f"{side} activity gauge {name}={value}, expected idle zero")

    if all(delta[key] is not None for key in COUNTERS):
        numeric = {key: float(delta[key]) for key in COUNTERS}
        for key, value in numeric.items():
            if value < 0 or not math.isfinite(value):
                count_issues.append(f"invalid {key} counter delta {value}")
        if numeric["generated"] <= 0:
            count_issues.append("generated counter delta is not positive")
        if numeric["draft_steps"] <= 0:
            count_issues.append("draft_steps counter delta is not positive")
        if numeric["proposed"] <= 0:
            count_issues.append("proposed counter delta is not positive")
        if numeric["accepted"] > numeric["proposed"]:
            count_issues.append("accepted counter delta exceeds proposed counter delta")
        if numeric["draft_steps"] > 0 and numeric["proposed"] > MTP_DEPTH * numeric["draft_steps"]:
            count_issues.append(
                "proposed counter delta exceeds MTP3 depth times draft-step delta"
            )

        completion_tokens = [
            numeric_int(request["usage"].get("completion_tokens"))
            if isinstance(request.get("usage"), dict)
            else None
            for request in requests
        ]
        if any(value is None for value in completion_tokens):
            count_issues.append("one or more requests lack integral usage.completion_tokens")
        else:
            usage_total = sum(value for value in completion_tokens if value is not None)
            if numeric["generated"] != usage_total:
                count_issues.append(
                    f"generated counter delta {numeric['generated']} != response completion-token total {usage_total}"
                )
    else:
        missing_delta = [key for key, value in delta.items() if value is None]
        count_issues.append(f"counter delta unavailable for {missing_delta}")

    return (
        {"passed": not runtime_issues, "issues": runtime_issues},
        {"passed": not count_issues, "issues": count_issues},
    )


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


async def measure_group(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    seed: int,
    group_id: str,
    domain: str,
) -> dict[str, Any]:
    concurrency = len(prompts)
    started_at = time.time()
    before = await scrape_metrics(session, args.base_url)
    tasks = [
        asyncio.create_task(post_chat(session, args.base_url, args.model, prompt, seed))
        for prompt in prompts
    ]
    requests = await asyncio.gather(*tasks)
    after = await scrape_metrics(session, args.base_url)
    finished_at = time.time()
    delta = counter_delta(before["counters"], after["counters"])
    runtime_integrity, count_integrity = group_integrity(
        concurrency, requests, before, after, delta
    )
    accepted = delta["accepted"]
    proposed = delta["proposed"]
    draft_steps = delta["draft_steps"]
    generated = delta["generated"]
    passed = runtime_integrity["passed"] and count_integrity["passed"]
    return {
        "group_id": group_id,
        "concurrency": concurrency,
        "domain": domain,
        "seed": seed,
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "started_at": started_at,
        "started_at_iso": epoch_iso(started_at),
        "finished_at": finished_at,
        "finished_at_iso": epoch_iso(finished_at),
        "elapsed_seconds": finished_at - started_at,
        "elapsed_scope": (
            "whole counter window including metrics scrapes and end-to-end requests; request elapsed values "
            "include prefill and are not pure decode or GPU-kernel timing"
        ),
        "counter_before": before["counters"],
        "counter_after": after["counters"],
        "counter_delta": delta,
        "counter_evidence": {"before": before, "after": after},
        "accepted_tokens": accepted,
        "proposed_tokens": proposed,
        "acceptance_fraction": ratio(accepted, proposed),
        "accepted_draft_tokens_per_step": ratio(accepted, draft_steps),
        "emitted_tokens_per_verifier_step": ratio(generated, draft_steps),
        "step_metric_note": (
            "draft_steps is the server's aggregate logical speculative draft/verifier-step counter across "
            "requests, not a count of physical batched GPU kernel launches"
        ),
        "requests": requests,
        "runtime_integrity": runtime_integrity,
        "count_integrity": count_integrity,
        "measurement_eligible": passed,
        "passed": passed,
    }


def failed_group(
    group_id: str,
    prompts: list[dict[str, Any]],
    seed: int,
    domain: str,
    error: BaseException,
) -> dict[str, Any]:
    now = time.time()
    issue = f"{type(error).__name__}: {error}"
    return {
        "group_id": group_id,
        "concurrency": len(prompts),
        "domain": domain,
        "seed": seed,
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "started_at": now,
        "started_at_iso": epoch_iso(now),
        "finished_at": now,
        "finished_at_iso": epoch_iso(now),
        "elapsed_seconds": 0.0,
        "counter_before": {},
        "counter_after": {},
        "counter_delta": {key: None for key in COUNTERS},
        "counter_evidence": {},
        "accepted_tokens": None,
        "proposed_tokens": None,
        "acceptance_fraction": None,
        "accepted_draft_tokens_per_step": None,
        "emitted_tokens_per_verifier_step": None,
        "step_metric_note": (
            "draft_steps would denote aggregate logical speculative steps, not GPU kernel launches"
        ),
        "requests": [],
        "runtime_integrity": {"passed": False, "issues": [issue]},
        "count_integrity": {"passed": False, "issues": [issue]},
        "measurement_eligible": False,
        "passed": False,
    }


def aggregate_groups(
    groups: list[dict[str, Any]],
    include: Callable[[dict[str, Any]], bool],
    label: str,
) -> dict[str, Any]:
    selected = [group for group in groups if include(group)]
    eligible = [group for group in selected if group.get("measurement_eligible") is True]
    excluded = [group["group_id"] for group in selected if group.get("measurement_eligible") is not True]
    totals = {key: 0.0 for key in COUNTERS}
    for group in eligible:
        for key in COUNTERS:
            totals[key] += float(group["counter_delta"][key])
    return {
        "label": label,
        "weighting": (
            "token-weighted speculative acceptance: sum(accepted draft tokens) / sum(proposed draft "
            "tokens), never an arithmetic mean of per-group percentages"
        ),
        "selected_group_count": len(selected),
        "eligible_group_count": len(eligible),
        "excluded_group_ids": excluded,
        "eligible_request_count": sum(len(group["requests"]) for group in eligible),
        "counter_delta_totals": totals,
        "accepted_tokens": totals["accepted"],
        "proposed_tokens": totals["proposed"],
        "token_weighted_pooled_acceptance_fraction": ratio(
            totals["accepted"], totals["proposed"]
        ),
        "accepted_draft_tokens_per_step": ratio(
            totals["accepted"], totals["draft_steps"]
        ),
        "emitted_tokens_per_verifier_step": ratio(
            totals["generated"], totals["draft_steps"]
        ),
    }


def request_trace_issues(
    groups: list[dict[str, Any]], seeds: list[int], model: str = DEFAULT_MODEL
) -> list[str]:
    issues: list[str] = []
    prompts_by_id = {prompt["id"]: prompt for prompt in PROMPTS}
    all_prompt_ids = [prompt["id"] for prompt in PROMPTS]
    observed_pairs: Counter[tuple[int, str]] = Counter()
    for group in groups:
        concurrency = group.get("concurrency")
        prompt_ids = group.get("prompt_ids")
        seed = group.get("seed")
        if concurrency == 1 and (
            not isinstance(prompt_ids, list) or len(prompt_ids) != 1
        ):
            issues.append(f"{group.get('group_id')}: C1 must contain exactly one prompt")
        elif concurrency == 8 and prompt_ids != all_prompt_ids:
            issues.append(f"{group.get('group_id')}: C8 prompt order/set is not the fixed input set")
        elif concurrency not in (1, 8):
            issues.append(f"{group.get('group_id')}: unsupported concurrency {concurrency!r}")
        for request in group.get("requests", []):
            prompt_id = request.get("prompt_id")
            request_seed = request.get("seed")
            prompt = prompts_by_id.get(prompt_id)
            if prompt is None or request_seed not in seeds:
                issues.append(
                    f"{group.get('group_id')}: unknown prompt/seed pair {prompt_id!r}/{request_seed!r}"
                )
                continue
            expected_payload = request_payload(model, prompt, request_seed)
            expected_raw = json.dumps(
                expected_payload, separators=(",", ":"), ensure_ascii=False
            )
            if request.get("request") != expected_payload:
                issues.append(
                    f"{group.get('group_id')}: request payload drift for {prompt_id}/{request_seed}"
                )
            if request.get("raw_request") != expected_raw:
                issues.append(
                    f"{group.get('group_id')}: raw request drift for {prompt_id}/{request_seed}"
                )
            if request_seed != seed:
                issues.append(
                    f"{group.get('group_id')}: request seed {request_seed} != group seed {seed}"
                )
            observed_pairs[(request_seed, prompt_id)] += 1
    for seed in seeds:
        for prompt_id in all_prompt_ids:
            observed = observed_pairs[(seed, prompt_id)]
            if observed != 2:
                issues.append(
                    f"prompt/seed {prompt_id}/{seed} observed {observed} times; expected matched C1+C8 pair"
                )
    return issues


def build_summary(
    groups: list[dict[str, Any]],
    seeds: list[int],
    config_ok: bool,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    expected_groups = len(seeds) * (len(PROMPTS) + 1)
    expected_requests = len(seeds) * len(PROMPTS) * 2
    invalid_groups = [
        {
            "group_id": group["group_id"],
            "runtime_issues": group["runtime_integrity"]["issues"],
            "count_issues": group["count_integrity"]["issues"],
        }
        for group in groups
        if group.get("measurement_eligible") is not True
    ]
    actual_requests = sum(len(group.get("requests", [])) for group in groups)
    integrity_issues: list[str] = []
    if not config_ok:
        integrity_issues.append("launch configuration integrity failed")
    if len(groups) != expected_groups:
        integrity_issues.append(f"expected {expected_groups} groups, observed {len(groups)}")
    if actual_requests != expected_requests:
        integrity_issues.append(
            f"expected {expected_requests} request records, observed {actual_requests}"
        )
    if invalid_groups:
        integrity_issues.append(f"{len(invalid_groups)} groups failed runtime/count integrity")
    integrity_issues.extend(request_trace_issues(groups, seeds, model))

    domains = sorted({prompt["domain"] for prompt in PROMPTS})
    classifications = Counter(
        request.get("classification", "missing")
        for group in groups
        for request in group.get("requests", [])
    )
    return {
        "runtime_count_integrity": {
            "passed": not integrity_issues,
            "gate_scope": (
                "runtime completion, exact request accounting, launch matching, idle counter windows, and "
                "server-counter consistency only"
            ),
            "issues": integrity_issues,
            "invalid_groups": invalid_groups,
            "expected_groups": expected_groups,
            "observed_groups": len(groups),
            "expected_requests": expected_requests,
            "observed_requests": actual_requests,
        },
        "response_classifications": dict(sorted(classifications.items())),
        "all_measurements": aggregate_groups(
            groups,
            lambda group: True,
            "all C1 and C8 measurement tokens across both fixed seeds",
        ),
        "by_concurrency": {
            str(concurrency): aggregate_groups(
                groups,
                lambda group, concurrency=concurrency: group["concurrency"] == concurrency,
                f"C{concurrency} measurement tokens across both fixed seeds",
            )
            for concurrency in (1, 8)
        },
        "c1_by_domain": {
            domain: aggregate_groups(
                groups,
                lambda group, domain=domain: group["concurrency"] == 1
                and group["domain"] == domain,
                f"C1 {domain} measurement tokens across its two prompts and both fixed seeds",
            )
            for domain in domains
        },
        "claim_limits": {
            "semantic_quality_proof": False,
            "statistical_equivalence_test": False,
            "performance_kernel_benchmark": False,
            "interpretation": (
                "Acceptance is workload- and token-weighted descriptive evidence for this fixed chat slice. "
                "Visible-final and budget classifications are retained, but no semantic-quality gate is made."
            ),
        },
    }


def configuration_integrity(
    launch: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    env = launch.get("env") if isinstance(launch.get("env"), dict) else {}
    expected = {
        "image": args.image,
        "tp": 4,
        "dcp": 4,
        "spec": "mtp3",
        "cache": "vram",
        "kv": "fp8_ds_mla",
        "GPU_MEMORY_UTILIZATION": "0.93",
        "VLLM_GLM53_MTP_DRAFT_HEAD": args.expected_draft_head,
        "MODEL": "/model",
    }
    observed = {
        "image": launch.get("image"),
        "tp": launch.get("tp"),
        "dcp": launch.get("dcp"),
        "spec": launch.get("spec"),
        "cache": launch.get("cache"),
        "kv": launch.get("kv"),
        "GPU_MEMORY_UTILIZATION": env.get("GPU_MEMORY_UTILIZATION"),
        "VLLM_GLM53_MTP_DRAFT_HEAD": env.get("VLLM_GLM53_MTP_DRAFT_HEAD"),
        "MODEL": env.get("MODEL"),
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    return {
        "passed": not mismatches,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
        "weights": {
            "target_host_path": args.target_weights,
            "target_container_path": "/model",
            "weights_changed_by_phase": False,
        },
    }


async def execute(
    args: argparse.Namespace,
    launch: dict[str, Any],
    config_provenance: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.time()
    inputs = input_manifest(args.model, args.seeds)
    config_check = configuration_integrity(launch, args)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "suite": "realistic_acceptance",
        "arm": args.arm,
        "image": args.image,
        "config": launch,
        "config_provenance": config_provenance,
        "configuration_integrity": config_check,
        "input_set": inputs,
        "started_at": started_at,
        "started_at_iso": epoch_iso(started_at),
        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "groups": [],
        "summary": build_summary([], args.seeds, config_check["passed"], args.model),
        "measurement_scope": (
            "Natural-completion chat acceptance follow-up; not the sustained repeated-text ceiling probe"
        ),
        "semantic_quality_claim": False,
        "statistical_equivalence_claim": False,
        "finished_at": None,
        "finished_at_iso": None,
    }
    write_atomic(args.output, receipt)
    if not config_check["passed"]:
        finished_at = time.time()
        receipt["finished_at"] = finished_at
        receipt["finished_at_iso"] = epoch_iso(finished_at)
        receipt["summary"] = build_summary([], args.seeds, False, args.model)
        write_atomic(args.output, receipt)
        return receipt

    timeout = aiohttp.ClientTimeout(
        total=args.request_timeout,
        connect=min(30.0, args.request_timeout),
        sock_read=args.request_timeout,
    )
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, trust_env=False
    ) as session:
        for seed in args.seeds:
            for prompt in PROMPTS:
                group_id = f"seed-{seed}-c1-{prompt['id']}"
                try:
                    group = await measure_group(
                        session, args, [prompt], seed, group_id, prompt["domain"]
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    group = failed_group(
                        group_id, [prompt], seed, prompt["domain"], error
                    )
                receipt["groups"].append(group)
                receipt["summary"] = build_summary(
                    receipt["groups"], args.seeds, True, args.model
                )
                write_atomic(args.output, receipt)

            mixed_prompts = list(PROMPTS)
            group_id = f"seed-{seed}-c8-all-prompts"
            try:
                group = await measure_group(
                    session, args, mixed_prompts, seed, group_id, "mixed"
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                group = failed_group(group_id, mixed_prompts, seed, "mixed", error)
            receipt["groups"].append(group)
            receipt["summary"] = build_summary(
                receipt["groups"], args.seeds, True, args.model
            )
            write_atomic(args.output, receipt)

    finished_at = time.time()
    receipt["finished_at"] = finished_at
    receipt["finished_at_iso"] = epoch_iso(finished_at)
    receipt["summary"] = build_summary(
        receipt["groups"], args.seeds, True, args.model
    )
    write_atomic(args.output, receipt)
    return receipt


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("at least two seeds are required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-inputs", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--image")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-draft-head", choices=("bf16", "nvfp4"))
    parser.add_argument("--target-weights", default="/mnt/2king/models/GLM-5.3-Flash-NVFP4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:5002")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", type=parse_seeds, default=list(DEFAULT_SEEDS))
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if not args.print_inputs:
        missing = [
            option
            for option, value in (
                ("--arm", args.arm),
                ("--image", args.image),
                ("--config", args.config),
                ("--expected-draft-head", args.expected_draft_head),
                ("--output", args.output),
            )
            if value is None
        ]
        if missing:
            parser.error("required for a measurement run: " + ", ".join(missing))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_inputs:
        print(json.dumps(input_manifest(args.model, args.seeds), indent=2, ensure_ascii=False))
        return 0
    assert args.config is not None
    assert args.output is not None
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing raw receipt: {args.output}")
    try:
        launch = json.loads(args.config.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read launch config {args.config}: {error}") from error
    if not isinstance(launch, dict):
        raise SystemExit(f"launch config root is not an object: {args.config}")
    config_provenance = {
        "path": str(args.config.resolve()),
        "sha256": sha256_file(args.config),
        "probe_script": str(Path(__file__).resolve()),
        "probe_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt = asyncio.run(execute(args, launch, config_provenance))
    integrity = receipt["summary"]["runtime_count_integrity"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arm": args.arm,
                "groups": len(receipt["groups"]),
                "requests": integrity["observed_requests"],
                "runtime_count_integrity_passed": integrity["passed"],
                "response_classifications": receipt["summary"]["response_classifications"],
            },
            sort_keys=True,
        )
    )
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
