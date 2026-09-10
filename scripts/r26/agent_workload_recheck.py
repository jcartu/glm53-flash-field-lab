#!/usr/bin/env python3
"""Run the deterministic mixed-agent trace against the shipped R26 scheduler APIs.

Prompt construction, offered-arrival scheduling, streaming request capture, and
latency/work-split summaries are deliberately reused from agent_workload.py.
This module replaces only the obsolete fairness-API contract and adds explicit
lane, non-progress, completion, and external-memory evidence.

With --resume the existing output receipt is validated and its terminal cells
(those that finished the measurement window, including measured qualification
failures) are reused verbatim; only missing scenarios are executed, and corrupt,
duplicate, or configuration-mismatched receipts fail closed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

try:
    from . import agent_workload as base
except ImportError:
    import agent_workload as base  # type: ignore[no-redef]


POLICY_PATH = "/prefill_fairness"
OVERLAY_CONFIGURED_FIELDS = (
    "prefill_compute_share",
    "prefill_compute_half_life",
    "max_parallel_prefills",
    "prefill_policy",
    "decode_refill_target",
)
OVERLAY_GET_FIELDS = (
    *OVERLAY_CONFIGURED_FIELDS,
    "effective_max_parallel_prefills",
    "effective_decode_refill_target",
)
OVERLAY_MUTABLE_FIELDS = (
    "prefill_compute_share",
    "prefill_compute_half_life",
)
LEGACY_FIELDS = (
    "fairness_engine",
    "prefill_compute_share",
    "max_num_prefill_tokens_per_step",
    "max_num_partial_prefills",
    "decode_prefill_min_decode_steps",
    "decode_prefill_max_wait_ms",
)
METRIC_NAMES = {
    *base.METRIC_NAMES,
    "vllm:num_requests_waiting_by_reason",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_by_source_total",
    "vllm:prompt_tokens_cached_total",
}


def detect_policy_schema(config: object) -> str:
    if not isinstance(config, dict):
        return "unknown"
    if all(field in config for field in OVERLAY_GET_FIELDS):
        return "overlay-r26"
    if all(field in config for field in LEGACY_FIELDS):
        return "legacy-r26"
    return "unknown"


def response_config(body: object) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(body, dict):
        return None, None
    nested = body.get("config")
    if isinstance(nested, dict):
        return nested, "body.config"
    known = set(OVERLAY_GET_FIELDS) | set(LEGACY_FIELDS)
    if known.intersection(body):
        return body, "body"
    return None, None


def response_shape(body: object) -> dict[str, Any]:
    config, location = response_config(body)
    return {
        "body_type": type(body).__name__,
        "top_level_keys": sorted(body) if isinstance(body, dict) else [],
        "config_location": location,
        "config_keys": sorted(config) if config is not None else [],
        "applied_marker_present": isinstance(body, dict) and "applied" in body,
        "applied_marker": body.get("applied") if isinstance(body, dict) else None,
    }


def parse_metrics(text: str) -> dict[str, Any]:
    values: dict[str, float] = {}
    cache_configs: list[dict[str, Any]] = []
    labelled = {
        "vllm:scheduler_compute_seconds_total": "class",
        "vllm:scheduler_compute_pressure": "class",
        "vllm:num_requests_waiting_by_reason": "reason",
        "vllm:prompt_tokens_by_source_total": "source",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
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
        labels = base.parse_labels(metric)
        if name == "vllm:cache_config_info":
            cache_configs.append({"labels": labels, "value": value})
            continue
        key = name
        label_name = labelled.get(name)
        if label_name and labels.get(label_name):
            key = f"{name}:{labels[label_name]}"
        values[key] = values.get(key, 0.0) + value
    return {"values": values, "cache_configs": cache_configs}


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    response = await client.get(f"{base_url}/metrics", timeout=10.0)
    response.raise_for_status()
    parsed = parse_metrics(response.text)
    parsed["raw"] = response.text
    parsed["observed_at_utc"] = base.utc_now()
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


async def get_policy(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    observed = base.utc_now()
    try:
        response = await client.get(f"{base_url}{POLICY_PATH}", timeout=15.0)
        text = response.text
        try:
            body: object = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None
        return {
            "observed_at_utc": observed,
            "method": "GET",
            "url": f"{base_url}{POLICY_PATH}",
            "request": None,
            "status_code": response.status_code,
            "body": body,
            "body_text": text,
            "schema": (
                detect_policy_schema(body)
                if response.status_code == 200
                else "unavailable"
            ),
            "response_shape": response_shape(body),
        }
    except Exception as error:
        return {
            "observed_at_utc": observed,
            "method": "GET",
            "url": f"{base_url}{POLICY_PATH}",
            "request": None,
            "status_code": None,
            "body": None,
            "body_text": "",
            "schema": "unavailable",
            "response_shape": response_shape(None),
            "error": f"{type(error).__name__}: {error}",
        }


async def post_policy(
    client: httpx.AsyncClient,
    base_url: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    observed = base.utc_now()
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
            "method": "POST",
            "url": f"{base_url}{POLICY_PATH}",
            "request": config,
            "status_code": response.status_code,
            "body": body,
            "body_text": text,
            "response_shape": response_shape(body),
        }
    except Exception as error:
        return {
            "observed_at_utc": observed,
            "method": "POST",
            "url": f"{base_url}{POLICY_PATH}",
            "request": config,
            "status_code": None,
            "body": None,
            "body_text": "",
            "response_shape": response_shape(None),
            "error": f"{type(error).__name__}: {error}",
        }


def mutable_payload(schema: str, expected: dict[str, Any]) -> dict[str, Any]:
    if schema == "overlay-r26":
        return {field: expected[field] for field in OVERLAY_MUTABLE_FIELDS}
    if schema == "legacy-r26":
        return {field: expected[field] for field in LEGACY_FIELDS}
    raise ValueError(f"unsupported API schema {schema!r}")


def policy_mismatches(
    actual: object,
    expected: dict[str, Any],
) -> dict[str, dict[str, object]]:
    return base.policy_mismatches(actual, expected)


def successful_readback(
    posted: dict[str, Any],
    after: dict[str, Any],
    expected_schema: str,
    expected: dict[str, Any],
) -> tuple[bool, dict[str, dict[str, object]]]:
    mismatches = policy_mismatches(after.get("body"), expected)
    status = posted.get("status_code")
    body = posted.get("body")
    explicit_applied = body.get("applied") if isinstance(body, dict) else None
    passed = (
        isinstance(status, int)
        and 200 <= status < 300
        and explicit_applied is not False
        and after.get("schema") == expected_schema
        and not mismatches
    )
    return passed, mismatches


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
                    "observed_at_utc": base.utc_now(),
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
        config = dict(update["config"])
        posted = await post_policy(client, args.base_url, config)
        after = await get_policy(client, args.base_url)
        accepted, mismatches = successful_readback(
            posted, after, args.expected_api_schema, config
        )
        running = float(values.get("vllm:num_requests_running", 0.0))
        waiting = float(values.get("vllm:num_requests_waiting", 0.0))
        backlog = float(
            values.get("vllm:scheduler_local_prefill_backlog_tokens", 0.0)
        )
        before_body = before.get("body") if isinstance(before.get("body"), dict) else {}
        after_body = after.get("body") if isinstance(after.get("body"), dict) else {}
        before_mutable = {
            field: before_body.get(field) for field in OVERLAY_MUTABLE_FIELDS
        }
        after_mutable = {
            field: after_body.get(field) for field in OVERLAY_MUTABLE_FIELDS
        }
        observations.append(
            {
                "index": index,
                "scheduled_offset_seconds": offset,
                "actual_offset_seconds": time.monotonic() - origin_monotonic,
                "config": config,
                "allowed_mutable_fields": list(OVERLAY_MUTABLE_FIELDS),
                "metrics_before": metrics,
                "running_requests_before": running,
                "waiting_requests_before": waiting,
                "local_prefill_backlog_tokens_before": backlog,
                "active_work_before": running + waiting > 0.0 or backlog > 0.0,
                "before": before,
                "post": posted,
                "after": after,
                "accepted_and_read_back": accepted,
                "readback_mismatches": mismatches,
                "readback_matches": not mismatches,
                "configured_state_changed": before_mutable != after_mutable,
                "success_does_not_require_applied_marker": True,
            }
        )


def numeric_distribution(values: list[float]) -> dict[str, Any]:
    return {
        **base.distribution(values),
        "min": min(values) if values else None,
        "distinct": sorted(set(values)),
    }


def policy_telemetry_summary(
    samples: list[dict[str, Any]], duration: float
) -> dict[str, Any]:
    configs = [
        sample["body"]
        for sample in samples
        if 0.0 <= float(sample.get("offset_seconds", -1.0)) <= duration
        and sample.get("status_code") == 200
        and isinstance(sample.get("body"), dict)
    ]

    def numbers(field: str) -> list[float]:
        return [
            float(config[field])
            for config in configs
            if isinstance(config.get(field), (int, float))
            and not isinstance(config.get(field), bool)
        ]

    def distinct(field: str) -> list[Any]:
        encoded = {
            json.dumps(config[field], sort_keys=True)
            for config in configs
            if field in config
        }
        return [json.loads(value) for value in sorted(encoded)]

    return {
        "successful_samples": len(configs),
        "effective_prefill_compute_share": numeric_distribution(
            numbers("effective_prefill_compute_share")
        ),
        "decode_pressure": numeric_distribution(numbers("decode_pressure")),
        "prefill_pressure": numeric_distribution(numbers("prefill_pressure")),
        "configured_max_parallel_prefills": distinct("max_parallel_prefills"),
        "effective_max_parallel_prefills": numeric_distribution(
            numbers("effective_max_parallel_prefills")
        ),
        "configured_prefill_policies": distinct("prefill_policy"),
        "configured_decode_refill_targets": distinct("decode_refill_target"),
        "effective_decode_refill_target": numeric_distribution(
            numbers("effective_decode_refill_target")
        ),
        "runnable_decodes": numeric_distribution(numbers("runnable_decodes")),
        "raw_samples_field": "api.samples",
    }


def prefill_progress_summary(
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    duration: float,
    metrics_start: dict[str, Any],
    measurement_values: dict[str, Any],
) -> dict[str, Any]:
    valid = sorted(
        (
            sample
            for sample in samples
            if "error" not in sample
            and 0.0 <= float(sample.get("offset_seconds", -1.0)) <= duration
        ),
        key=lambda sample: float(sample["offset_seconds"]),
    )
    intervals: list[dict[str, Any]] = []
    for previous, current in zip(valid, valid[1:]):
        start = float(previous["offset_seconds"])
        end = float(current["offset_seconds"])
        backlog = max(
            float(previous.get("vllm:scheduler_local_prefill_backlog_tokens", 0.0)),
            float(current.get("vllm:scheduler_local_prefill_backlog_tokens", 0.0)),
        )
        waiting = max(
            float(previous.get("vllm:num_requests_waiting", 0.0)),
            float(current.get("vllm:num_requests_waiting", 0.0)),
        )
        before_compute = previous.get("vllm:scheduler_compute_seconds_total:prefill")
        after_compute = current.get("vllm:scheduler_compute_seconds_total:prefill")
        compute_delta = (
            max(0.0, float(after_compute) - float(before_compute))
            if before_compute is not None and after_compute is not None
            else None
        )
        under_pressure = backlog > 0.0 or waiting > 0.0
        intervals.append(
            {
                "start_offset_seconds": start,
                "end_offset_seconds": end,
                "seconds": max(0.0, end - start),
                "backlog_tokens": backlog,
                "waiting_requests": waiting,
                "under_prefill_pressure": under_pressure,
                "prefill_compute_seconds_delta": compute_delta,
                "progress_observed": compute_delta is not None and compute_delta > 0.0,
            }
        )
    pressure = [row for row in intervals if row["under_prefill_pressure"]]
    nonprogress = [row for row in pressure if row["progress_observed"] is False]
    progress = [row for row in pressure if row["progress_observed"] is True]
    start_values = metrics_start.get("values", {})
    compute_delta = base.metric_delta(
        start_values,
        measurement_values,
        "vllm:scheduler_compute_seconds_total:prefill",
    )
    cold = [row for row in rows if row.get("kind") == "cold_prefill"]
    cold_first_tokens_total = sum(
        row.get("first_token_offset_seconds") is not None for row in cold
    )
    cold_first_tokens_by_deadline = sum(
        row.get("first_token_offset_seconds") is not None
        and float(row["first_token_offset_seconds"]) <= duration
        for row in cold
    )
    pressure_observed = bool(pressure)
    zero_quantum_suspected = (
        bool(cold)
        and pressure_observed
        and compute_delta is not None
        and compute_delta <= 0.0
        and cold_first_tokens_by_deadline == 0
    )
    return {
        "metric": "vllm:scheduler_compute_seconds_total:prefill",
        "cold_requests_offered": len(cold),
        "cold_requests_with_first_token_by_measurement_end": cold_first_tokens_by_deadline,
        "cold_requests_with_first_token_eventually": cold_first_tokens_total,
        "prefill_compute_seconds_measurement_delta": compute_delta,
        "pressure_intervals": len(pressure),
        "progress_intervals": len(progress),
        "nonprogress_intervals": len(nonprogress),
        "nonprogress_seconds": sum(float(row["seconds"]) for row in nonprogress),
        "longest_sampled_nonprogress_seconds": max(
            (float(row["seconds"]) for row in nonprogress), default=0.0
        ),
        "pressure_observed": pressure_observed,
        "zero_prefill_quantum_suspected": zero_quantum_suspected,
        "interpretation": (
            "suspected zero-prefill-quantum: queued/backlogged cold work had neither "
            "prefill compute-counter progress nor a first token"
            if zero_quantum_suspected
            else "no whole-window zero-prefill-quantum signature observed"
        ),
        "raw_intervals": intervals,
    }


def stalled_request_summary(rows: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    stalled: list[dict[str, Any]] = []
    for row in rows:
        finish = row.get("finished_offset_seconds")
        if row.get("status") == "completed" and finish is not None and float(finish) <= duration:
            continue
        stalled.append(
            {
                "request_key": row.get("request_key"),
                "kind": row.get("kind"),
                "status_after_drain": row.get("status"),
                "scheduled_offset_seconds": row.get("scheduled_offset_seconds"),
                "first_token_offset_seconds": row.get("first_token_offset_seconds"),
                "finished_offset_seconds": finish,
                "minimum_wait_at_measurement_end_seconds": max(
                    0.0, duration - float(row.get("scheduled_offset_seconds") or 0.0)
                ),
                "error": row.get("error"),
            }
        )
    return {
        "count_at_measurement_end": len(stalled),
        "requests": stalled,
        "status_is_not_interpreted_as_model_corruption": True,
    }


def completion_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    for row in rows:
        reasons = {str(reason) for reason in row.get("finish_reasons", [])}
        if row.get("status") == "completed" and "length" in reasons:
            classification = "protocol_complete_budget_limited"
        elif row.get("status") == "completed":
            classification = "protocol_complete"
        elif row.get("status") in {"censored", "timed_out", "client_queued", "in_flight"}:
            classification = "incomplete_or_timeout"
        else:
            classification = "protocol_or_service_failure"
        counts[classification] = counts.get(classification, 0) + 1
        events.append(
            {
                "request_key": row.get("request_key"),
                "classification": classification,
                "finish_reasons": sorted(reasons),
                "status": row.get("status"),
            }
        )
    return {
        "counts": counts,
        "events": events,
        "model_answer_correctness_assessed": False,
        "text_equality_used_as_kv_byte_equality": False,
    }


def memory_transfer_evidence(
    metrics_start: dict[str, Any], metrics_end: dict[str, Any]
) -> dict[str, Any]:
    before = metrics_start.get("values", {})
    after = metrics_end.get("values", {})
    keys = {
        "external_prefix_queries_tokens": "vllm:external_prefix_cache_queries_total",
        "external_prefix_hits_tokens": "vllm:external_prefix_cache_hits_total",
        "prompt_tokens_external_kv_transfer": (
            "vllm:prompt_tokens_by_source_total:external_kv_transfer"
        ),
        "prompt_tokens_local_cache_hit": (
            "vllm:prompt_tokens_by_source_total:local_cache_hit"
        ),
        "prompt_tokens_local_compute": (
            "vllm:prompt_tokens_by_source_total:local_compute"
        ),
    }
    deltas = {
        label: base.metric_delta(before, after, metric) for label, metric in keys.items()
    }
    external_hits = deltas["external_prefix_hits_tokens"]
    transferred = deltas["prompt_tokens_external_kv_transfer"]
    hit_observed = external_hits is not None and external_hits > 0.0
    transfer_observed = transferred is not None and transferred > 0.0
    queries = deltas["external_prefix_queries_tokens"]
    if transfer_observed:
        classification = "external_kv_transfer_counter_advanced"
    elif hit_observed:
        classification = "external_cache_hit_without_transfer_counter_evidence"
    else:
        classification = "no_external_kv_transfer_evidence"
    return {
        "counter_deltas": deltas,
        "external_prefix_hit_observed": hit_observed,
        "external_memory_transfer_observed": transfer_observed,
        "transfer_evidence_basis": (
            "vllm:prompt_tokens_by_source_total{source='external_kv_transfer'}"
        ),
        "queries_without_external_hits": bool(
            queries is not None
            and queries > 0.0
            and not hit_observed
        ),
        "classification": classification,
        "failed_retrieval_is_not_classified_as_corrupted_memory": True,
        "text_output_is_not_kv_byte_evidence": True,
    }


async def run_scenario(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    profile: str,
    concurrency: int,
    repeat: int,
) -> dict[str, Any]:
    duration = base.profile_duration(args, profile)
    trace_id = f"{args.trace_seed}:{profile}:c{concurrency}:r{repeat}"
    print(f"START {trace_id} policy={args.policy_label}", flush=True)
    post_payload = mutable_payload(args.expected_api_schema, args.expected_policy)
    boundary_before = await get_policy(client, args.base_url)
    boundary_post = await post_policy(client, args.base_url, post_payload)
    boundary_after = await get_policy(client, args.base_url)
    boundary_passed, boundary_mismatches = successful_readback(
        boundary_post,
        boundary_after,
        args.expected_api_schema,
        args.expected_policy,
    )
    boundary_receipt = {
        "before": boundary_before,
        "post": boundary_post,
        "after": boundary_after,
        "expected_schema": args.expected_api_schema,
        "expected_configured_readback": args.expected_policy,
        "posted_mutable_config": post_payload,
        "readback_mismatches": boundary_mismatches,
        "accepted_and_read_back": boundary_passed,
        "success_basis": "2xx POST plus independent GET readback; applied marker not required",
    }
    if not boundary_passed:
        return {
            "status": "failed",
            "failure_class": "harness_or_scheduler_api_contract",
            "candidate": args.candidate_label,
            "policy": args.policy_label,
            "profile": profile,
            "concurrency": concurrency,
            "repeat": repeat,
            "measurement_seconds": duration,
            "api": {"cell_boundary_apply": boundary_receipt},
            "primes": [],
            "requests": [],
            "raw_request_events_field": "requests",
            "gates": [
                base.make_gate(
                    "policy_accepted_and_read_back_at_cell_boundary",
                    False,
                    boundary_receipt,
                )
            ],
        }

    cold_trace_rows = base.cold_trace(args, profile, duration)
    session_prompts, cold_prompts = await base.build_prompt_artifacts(
        client, args, trace_id, concurrency, cold_trace_rows
    )
    primes = await base.prime_sessions(client, args, trace_id, session_prompts)
    prime_integrity = [base.request_integrity(row) for row in primes]
    if not all(passed for passed, _ in prime_integrity):
        return {
            "status": "failed",
            "failure_class": "request_protocol_or_service",
            "candidate": args.candidate_label,
            "policy": args.policy_label,
            "profile": profile,
            "concurrency": concurrency,
            "repeat": repeat,
            "measurement_seconds": duration,
            "api": {"cell_boundary_apply": boundary_receipt},
            "primes": primes,
            "requests": [],
            "raw_request_events_field": "requests",
            "gates": [
                base.make_gate(
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

    specs = base.make_request_specs(
        args,
        trace_id,
        concurrency,
        duration,
        session_prompts,
        cold_trace_rows,
        cold_prompts,
    )
    trace = base.build_trace_manifest(
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
    rows = [base.initialize_request_row(spec, origin_unix) for spec in specs]
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
            base.perform_scheduled_request(
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

    base.annotate_incremental_prompt_tokens(rows, primes)
    integrity = {row["request_key"]: base.request_integrity(row) for row in rows}
    bad_integrity = {
        key: problems for key, (passed, problems) in integrity.items() if not passed
    }
    summary = base.summarize_requests(
        rows,
        concurrency,
        duration,
        metric_samples,
        metrics_start,
        metrics_end,
    )
    policy_summary = policy_telemetry_summary(policy_samples, duration)
    measurement_values = base.nearest_sample_values(metric_samples, duration)
    progress = prefill_progress_summary(
        metric_samples,
        rows,
        duration,
        metrics_start,
        measurement_values,
    )
    stalls = stalled_request_summary(rows, duration)
    completions = completion_classification(rows)
    transfer = memory_transfer_evidence(metrics_start, metrics_end)
    summary["prefill_progress_and_nonprogress"] = progress
    summary["stalled_requests"] = stalls
    summary["completion_classification"] = completions
    summary["memory_transfer_evidence"] = transfer

    start_values = metrics_start.get("values", {})
    required_metric_keys = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:scheduler_compute_seconds_total:decode",
        "vllm:scheduler_compute_seconds_total:prefill",
    }
    if args.expected_api_schema == "overlay-r26":
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
    if args.cache_mode == "lmcache":
        required_metric_keys.update(
            {
                "vllm:external_prefix_cache_queries_total",
                "vllm:external_prefix_cache_hits_total",
                "vllm:prompt_tokens_by_source_total:external_kv_transfer",
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
    work_split = summary["model_work_split_until_drain"]
    work_split_observed = (
        work_split["decode_compute_seconds"] is not None
        and work_split["prefill_compute_seconds"] is not None
        and work_split["total_compute_seconds"] is not None
        and float(work_split["total_compute_seconds"]) > 0.0
    )
    queue_telemetry = summary["queue_depths_and_growth"]
    measurement_metric_samples = int(queue_telemetry["measurement_sample_count"])
    measurement_policy_samples = int(policy_summary["successful_samples"])
    update_failures = [
        observation
        for observation in update_observations
        if not (
            observation["active_work_before"]
            and observation["accepted_and_read_back"]
            and observation["configured_state_changed"]
        )
    ]
    gates = [
        base.make_gate(
            "policy_accepted_and_read_back_at_cell_boundary",
            boundary_passed,
            boundary_receipt,
        ),
        base.make_gate("all_hot_sessions_primed", True, {"sessions": len(primes)}),
        base.make_gate(
            "policy_schema_and_configured_readback_match",
            initial_policy.get("schema") == args.expected_api_schema
            and not initial_mismatches,
            {
                "expected_schema": args.expected_api_schema,
                "observed_schema": initial_policy.get("schema"),
                "mismatches": initial_mismatches,
                "readback": initial_policy,
            },
        ),
        base.make_gate(
            "entire_deterministic_trace_offered",
            len(rows) == len(specs),
            {"planned": len(specs), "observed": len(rows)},
        ),
        base.make_gate(
            "all_stream_protocol_responses_complete",
            not bad_integrity,
            {
                "offered": len(rows),
                "completed_and_valid": len(rows) - len(bad_integrity),
                "failures": bad_integrity,
                "model_answer_correctness_assessed": False,
            },
        ),
        base.make_gate(
            "required_scheduler_metrics_present",
            not missing_metrics and bool(cache_capacities),
            {
                "missing": missing_metrics,
                "kv_cache_size_tokens": cache_capacities,
            },
        ),
        base.make_gate(
            "measurement_telemetry_sampled",
            measurement_metric_samples > 0 and measurement_policy_samples > 0,
            {
                "metric_samples": measurement_metric_samples,
                "metric_sample_errors": queue_telemetry["sample_errors"],
                "policy_samples": measurement_policy_samples,
            },
        ),
        base.make_gate(
            "model_prefill_decode_work_split_observed",
            work_split_observed,
            work_split,
        ),
        base.make_gate(
            "no_zero_prefill_quantum_signature",
            profile == "baseline" or not progress["zero_prefill_quantum_suspected"],
            {"applicable": profile != "baseline", **progress},
        ),
        base.make_gate(
            "server_idle_after_drain", idle_after, {"error": idle_error}
        ),
    ]
    if args.expected_api_schema == "overlay-r26":
        structural_mismatches = policy_mismatches(
            initial_policy.get("body"),
            {
                field: args.expected_policy[field]
                for field in (
                    "max_parallel_prefills",
                    "prefill_policy",
                    "decode_refill_target",
                )
            },
        )
        effective_lanes = policy_summary["effective_max_parallel_prefills"]
        effective_refill = policy_summary["effective_decode_refill_target"]
        gates.append(
            base.make_gate(
                "effective_lane_budget_and_refill_captured",
                not structural_mismatches
                and effective_lanes["count"] > 0
                and float(effective_lanes["min"]) >= 1.0
                and effective_refill["count"] > 0
                and float(effective_refill["min"]) >= 1.0,
                {
                    "structural_mismatches": structural_mismatches,
                    "effective_max_parallel_prefills": effective_lanes,
                    "effective_decode_refill_target": effective_refill,
                },
            )
        )
    if args.expected_policy.get("prefill_compute_share") == "auto":
        share_stats = policy_summary["effective_prefill_compute_share"]
        share_min = share_stats["min"]
        share_max = share_stats["max"]
        gates.append(
            base.make_gate(
                "automatic_compute_share_telemetry_in_bounds",
                share_min is not None
                and share_max is not None
                and 0.2 <= float(share_min) <= float(share_max) <= 0.8,
                share_stats,
            )
        )
    if args.policy_updates:
        gates.append(
            base.make_gate(
                "mutable_compute_fields_updated_during_active_work",
                len(update_observations) == len(args.policy_updates)
                and not update_failures,
                {
                    "allowed_fields": list(OVERLAY_MUTABLE_FIELDS),
                    "planned": len(args.policy_updates),
                    "observed": len(update_observations),
                    "failures": update_failures,
                },
            )
        )
    if args.memory_role == "reader":
        gates.append(
            base.make_gate(
                "external_memory_transfer_observed_for_matched_reader",
                transfer["external_memory_transfer_observed"],
                transfer,
            )
        )
    status = "complete" if all(gate["passed"] for gate in gates) else "failed"
    failed_gate_names = [gate["name"] for gate in gates if not gate["passed"]]
    failure_class = None
    if failed_gate_names:
        if any("policy" in name or "metric" in name for name in failed_gate_names):
            failure_class = "harness_or_scheduler_capability"
        elif any("stream_protocol" in name or "idle" in name for name in failed_gate_names):
            failure_class = "request_protocol_or_service"
        elif "external_memory_transfer_observed_for_matched_reader" in failed_gate_names:
            failure_class = "memory_transfer_not_observed_not_corruption"
        else:
            failure_class = "scheduler_behavior_qualification"
    result = {
        "status": status,
        "failure_class": failure_class,
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
        "raw_request_events_field": "requests",
        "answer_assessment": {
            "model_correctness_assessed": False,
            "budget_limited_finish_is_protocol_complete": True,
            "text_equality_used_as_kv_byte_equality": False,
        },
        "gates": gates,
    }
    counts = summary["absolute_counts"]
    print(
        f"DONE {trace_id} status={status} "
        f"hot={counts['hot_turns_completed_within_window']}/"
        f"{counts['hot_turns_offered']} "
        f"hot_p95={summary['hot_ttft_seconds']['p95']}",
        flush=True,
    )
    return result


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = base.jsonable_args(args)
    result.pop("expected_policy", None)
    result.pop("policy_updates", None)
    # Resume mechanics are provenance, not measurement configuration.
    result.pop("resume", None)
    return result


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **jsonable_args(args),
        "api_contract": {
            "overlay_get_fields": list(OVERLAY_GET_FIELDS),
            "overlay_post_fields_only": list(OVERLAY_MUTABLE_FIELDS),
            "legacy_fields": list(LEGACY_FIELDS),
            "post_success_basis": "HTTP 2xx and independent GET readback",
        },
        "raw_request_events_field": "cells[].requests",
        "deterministic_trace_implementation": str(Path(base.__file__).resolve()),
    }


def cell_identity(cell: dict[str, Any]) -> tuple[str, int, int] | None:
    try:
        return (
            str(cell.get("profile")),
            int(cell.get("concurrency")),
            int(cell.get("repeat")),
        )
    except (TypeError, ValueError):
        return None


def identity_record(key: tuple[str, int, int]) -> dict[str, Any]:
    return {"profile": key[0], "concurrency": key[1], "repeat": key[2]}


def cell_terminal(cell: dict[str, Any]) -> bool:
    """True for a cell that finished its measurement window and evaluated gates.

    Boundary/prime failures and harness exceptions never measured anything;
    resume re-executes those and reuses terminal cells verbatim. Keep in sync
    with scheduler_recheck_phase.py.
    """
    return (
        cell.get("status") in ("complete", "failed")
        and isinstance(cell.get("trace"), dict)
        and isinstance(cell["trace"].get("trace_hash"), str)
        and isinstance(cell.get("gates"), list)
        and bool(cell["gates"])
        and isinstance(cell.get("summary"), dict)
        and isinstance(cell.get("requests"), list)
    )


def scenario_grid(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    return [
        (profile, concurrency, repeat)
        for repeat in range(1, args.repeats + 1)
        for concurrency in args.concurrencies
        for profile in args.profiles
    ]


def resume_metadata_mismatches(
    saved: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    # The output path legitimately moves between roots and the resume block is
    # bookkeeping added by this tool; every other metadata field must match.
    exclude = {"output", "resume"}
    mismatches: dict[str, dict[str, Any]] = {}
    for key in sorted((set(saved) | set(expected)) - exclude):
        if key not in saved or key not in expected or saved[key] != expected[key]:
            mismatches[key] = {
                "saved": saved.get(key, "<absent-from-saved-receipt>"),
                "current": expected.get(key, "<absent-from-current-arguments>"),
            }
    return mismatches


def load_resume_state(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the existing output receipt and plan the resumed scenario set.

    Fails closed on corrupt, duplicate, or configuration-mismatched receipts.
    Terminal cells are reused verbatim; only missing scenarios re-run.
    """
    path: Path = args.output
    if not path.is_file():
        raise RuntimeError(
            f"--resume requires an existing output receipt; {path} is missing"
        )
    try:
        saved = json.loads(path.read_text())
    except Exception as error:
        raise RuntimeError(
            f"--resume receipt {path} is corrupt: {type(error).__name__}: {error}"
        ) from error
    if (
        not isinstance(saved, dict)
        or not isinstance(saved.get("metadata"), dict)
        or not isinstance(saved.get("cells"), list)
    ):
        raise RuntimeError(
            f"--resume receipt {path} must contain a metadata object and a cells list"
        )
    saved_metadata = saved["metadata"]
    mismatches = resume_metadata_mismatches(saved_metadata, build_metadata(args))
    if mismatches:
        raise RuntimeError(
            f"--resume receipt {path} does not match the requested configuration: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    grid = scenario_grid(args)
    grid_set = set(grid)
    reused: dict[tuple[str, int, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int], int] = {}
    problems: list[str] = []
    for index, cell in enumerate(saved["cells"]):
        if not isinstance(cell, dict):
            problems.append(f"cells[{index}] is not an object")
            continue
        key = cell_identity(cell)
        if key is None or key not in grid_set:
            problems.append(
                f"cells[{index}] identity {key} is outside the requested scenario grid"
            )
            continue
        if key in seen:
            problems.append(
                f"cells[{index}] duplicates cells[{seen[key]}] for {key}"
            )
            continue
        seen[key] = index
        if cell_terminal(cell):
            reused[key] = cell
        else:
            dropped.append(
                {
                    "index": index,
                    **identity_record(key),
                    "status": cell.get("status"),
                    "failure_class": cell.get("failure_class"),
                }
            )
    if problems:
        raise RuntimeError(f"--resume receipt {path} failed validation: {problems}")
    missing = [key for key in grid if key not in reused]
    provenance = {
        "enabled": True,
        "resumed_at": base.utc_now(),
        "receipt": str(path),
        "prior_output_path": saved_metadata.get("output"),
        "reused_cells": [
            {
                **identity_record(key),
                "status": reused[key].get("status"),
                "failure_class": reused[key].get("failure_class"),
                "trace_hash": reused[key]["trace"]["trace_hash"],
            }
            for key in grid
            if key in reused
        ],
        "reexecuted_cells": [],
        "dropped_nonterminal_cells": dropped,
        "prior_resume": saved_metadata.get("resume"),
    }
    return {
        "grid": grid,
        "reused": reused,
        "missing": missing,
        "provenance": provenance,
    }


def merged_grid_cells(
    grid: list[tuple[str, int, int]],
    reused: dict[tuple[str, int, int], dict[str, Any]],
    executed: dict[tuple[str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        reused[key] if key in reused else executed[key]
        for key in grid
        if key in reused or key in executed
    ]


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
    )
    timeout = httpx.Timeout(None, connect=args.connect_timeout_seconds)
    resume = load_resume_state(args) if args.resume else None
    metadata = build_metadata(args)
    if resume is not None:
        metadata["resume"] = resume["provenance"]
    reused = resume["reused"] if resume is not None else {}
    executed: dict[tuple[str, int, int], dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for repeat in range(1, args.repeats + 1):
            for concurrency in args.concurrencies:
                for profile in args.profiles:
                    key = (profile, concurrency, repeat)
                    if key in reused:
                        continue
                    try:
                        cell = await run_scenario(
                            client, args, profile, concurrency, repeat
                        )
                    except Exception as error:
                        cell = {
                            "status": "error",
                            "failure_class": "harness_exception",
                            "candidate": args.candidate_label,
                            "policy": args.policy_label,
                            "profile": profile,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                            "requests": [],
                            "raw_request_events_field": "requests",
                            "gates": [
                                base.make_gate(
                                    "scenario_completed",
                                    False,
                                    f"{type(error).__name__}: {error}",
                                )
                            ],
                        }
                    if resume is not None:
                        executed[key] = cell
                        resume["provenance"]["reexecuted_cells"].append(
                            {
                                **identity_record(key),
                                "status": cell.get("status"),
                                "failure_class": cell.get("failure_class"),
                            }
                        )
                        snapshot: list[dict[str, Any]] = merged_grid_cells(
                            resume["grid"], reused, executed
                        )
                    else:
                        cells.append(cell)
                        snapshot = cells
                    base.atomic_write_json(
                        args.output, {"metadata": metadata, "cells": snapshot}
                    )
                    await asyncio.sleep(args.between_scenarios_seconds)
    final_cells = (
        merged_grid_cells(resume["grid"], reused, executed)
        if resume is not None
        else cells
    )
    return {"metadata": metadata, "cells": final_cells}


def compute_config_error(config: dict[str, Any]) -> str | None:
    share = config.get("prefill_compute_share")
    valid_share = share in (None, "auto") or (
        isinstance(share, (int, float))
        and not isinstance(share, bool)
        and 0.0 < float(share) < 1.0
    )
    if not valid_share:
        return "prefill_compute_share must be null, 'auto', or strictly between zero and one"
    half_life = config.get("prefill_compute_half_life")
    valid_half_life = half_life in (None, "smooth", "responsive") or (
        isinstance(half_life, (int, float))
        and not isinstance(half_life, bool)
        and math.isfinite(float(half_life))
        and float(half_life) > 0.0
    )
    if not valid_half_life:
        return "prefill_compute_half_life has an invalid value"
    if half_life is not None and share != "auto":
        return "prefill_compute_half_life requires prefill_compute_share='auto'"
    return None


def expected_policy_error(schema: str, config: dict[str, Any]) -> str | None:
    if schema == "legacy-r26":
        engine = config.get("fairness_engine")
        share = config.get("prefill_compute_share")
        if engine not in (None, "compute_share"):
            return "recheck legacy controls allow only off or compute_share"
        if engine is None and share is not None:
            return "legacy off requires a null compute share"
        if engine == "compute_share" and not (
            isinstance(share, (int, float))
            and not isinstance(share, bool)
            and 0.0 < float(share) < 1.0
        ):
            return "legacy compute_share requires a numeric share strictly between zero and one"
        for field in set(LEGACY_FIELDS) - {"fairness_engine", "prefill_compute_share"}:
            value = config.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return f"{field} must be a non-negative integer"
        return None
    compute_error = compute_config_error(config)
    if compute_error is not None:
        return compute_error
    lanes = config.get("max_parallel_prefills")
    if not (
        lanes == "auto"
        or (isinstance(lanes, int) and not isinstance(lanes, bool) and lanes >= 1)
    ):
        return "max_parallel_prefills must be an integer >=1 or 'auto'"
    prefill_policy = config.get("prefill_policy")
    if prefill_policy not in {"round-robin", "decode-aware"}:
        return "prefill_policy must be round-robin or decode-aware"
    refill = config.get("decode_refill_target")
    if not (
        refill == "auto"
        or (isinstance(refill, int) and not isinstance(refill, bool) and refill >= 1)
    ):
        return "decode_refill_target must be a positive integer or 'auto'"
    if lanes == 1 and prefill_policy != "round-robin":
        return "single-lane mode requires round-robin"
    if prefill_policy != "decode-aware" and refill != "auto":
        return "round-robin requires decode_refill_target='auto'"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5002")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument(
        "--expected-api-schema",
        choices=("overlay-r26", "legacy-r26"),
        required=True,
    )
    parser.add_argument("--expected-policy-json", required=True)
    parser.add_argument("--policy-updates-json", default="[]")
    parser.add_argument("--profiles", default="baseline,periodic-128k")
    parser.add_argument("--concurrencies", default="8,16")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--trace-seed", default="r26-mixed-agent-recheck-v2")
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--cache-mode", choices=("vram", "lmcache"), default="vram")
    parser.add_argument(
        "--memory-role", choices=("none", "writer", "reader"), default="none"
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse terminal scenario cells already recorded in --output and run "
            "only the missing ones; corrupt, duplicate, or configuration-"
            "mismatched receipts fail closed"
        ),
    )
    args = parser.parse_args(argv)
    args.profiles = base.comma_strings(args.profiles)
    args.concurrencies = base.comma_ints(args.concurrencies)
    args.short_prefill_tokens = base.comma_ints(args.short_prefill_tokens)
    args.analytics_offsets_seconds = base.comma_floats(args.analytics_offsets_seconds)
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
    if not args.base_url.startswith("http://127.0.0.1:"):
        parser.error("development API must use loopback http://127.0.0.1:<port>")
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
    expected_fields = (
        set(OVERLAY_CONFIGURED_FIELDS)
        if args.expected_api_schema == "overlay-r26"
        else set(LEGACY_FIELDS)
    )
    if set(args.expected_policy) != expected_fields:
        parser.error(
            f"expected policy fields for {args.expected_api_schema}: "
            f"{sorted(expected_fields)}"
        )
    policy_error = expected_policy_error(
        args.expected_api_schema, args.expected_policy
    )
    if policy_error is not None:
        parser.error(policy_error)
    if not isinstance(args.policy_updates, list):
        parser.error("policy updates JSON must be an array")
    if args.expected_api_schema != "overlay-r26" and args.policy_updates:
        parser.error("live-update coverage is only defined for the overlay API")
    for update in args.policy_updates:
        if not isinstance(update, dict) or set(update) != {"offset_seconds", "config"}:
            parser.error("each policy update needs exactly offset_seconds and config")
        if not isinstance(update["config"], dict):
            parser.error("each policy update config must be an object")
        if set(update["config"]) != set(OVERLAY_MUTABLE_FIELDS):
            parser.error("each live update must contain exactly the two mutable fields")
        update_error = compute_config_error(update["config"])
        if update_error is not None:
            parser.error(f"invalid live update: {update_error}")
        offset = update["offset_seconds"]
        if (
            not isinstance(offset, (int, float))
            or isinstance(offset, bool)
            or float(offset) < 0.0
            or any(
                float(offset) >= base.profile_duration(args, profile)
                for profile in args.profiles
            )
        ):
            parser.error("each live update offset must fall within every selected profile")
    if args.memory_role != "none" and args.cache_mode != "lmcache":
        parser.error("memory writer/reader roles require --cache-mode lmcache")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    base.atomic_write_json(args.output, result)
    print(f"output={args.output}", flush=True)
    if any(cell.get("status") != "complete" for cell in result["cells"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
