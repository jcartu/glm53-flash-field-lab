#!/usr/bin/env python3
"""Raw OpenAI-API quality probes for the R26 qualification phase.

The executable records complete request/response bodies and classifies transport or
schema failures separately from wrong answers and repetition.  It never treats a
quality failure as a successful answer, but leaves release policy to
``quality_phase.py`` so every arm can finish and be compared.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA = "r26-quality-probe/v1"
DEFAULT_BENCH = Path("/home/josh/qwen-project/llm-inference-bench/llm_decode_bench.py")
CODEWORD = "PINE-SABLE-7723"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_city_temp",
            "description": "Get the current temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_population",
            "description": "Get the current population for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

# The first entry is the exact topic flagged by the R25 23/24 receipt.  Four
# Apollo variants make every three-wave run contain twelve Apollo-family
# observations while retaining four unrelated corruption sentinels.
LONG_TOPICS = [
    "the design of the Apollo guidance computer",
    "Apollo Guidance Computer rope memory and erasable memory",
    "the Apollo Guidance Computer executive and restart protection",
    "Apollo guidance, navigation, and the DSKY interface",
    "how a modern CPU pipeline works",
    "how the TLS 1.3 handshake works",
    "the biochemistry of photosynthesis",
    "the evolution of compiler optimization",
]

METRIC_NAMES = {
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_by_source_total",
}
METRIC_LINE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_receipt(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def decode_json_body(raw_body: str) -> tuple[object | None, str]:
    try:
        return json.loads(raw_body), ""
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            parsed, decode_error = decode_json_body(raw_body)
            status = int(response.status)
            return {
                "ok": 200 <= status < 300 and not decode_error,
                "url": url,
                "request": payload,
                "status": status,
                "response_headers": dict(response.headers.items()),
                "raw_body": raw_body,
                "response": parsed,
                "error": decode_error,
                "elapsed_seconds": time.monotonic() - started,
            }
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        parsed, decode_error = decode_json_body(raw_body)
        return {
            "ok": False,
            "url": url,
            "request": payload,
            "status": int(exc.code),
            "response_headers": dict(exc.headers.items()) if exc.headers else {},
            "raw_body": raw_body,
            "response": parsed,
            "error": f"HTTPError: {exc}; {decode_error}".strip("; "),
            "elapsed_seconds": time.monotonic() - started,
        }
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        return {
            "ok": False,
            "url": url,
            "request": payload,
            "status": None,
            "response_headers": {},
            "raw_body": "",
            "response": None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        }


def post_empty(base_url: str, path: str, timeout: float = 60.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    started = time.monotonic()
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            parsed, decode_error = decode_json_body(raw_body) if raw_body else (None, "")
            status = int(response.status)
            return {
                "ok": 200 <= status < 300,
                "url": url,
                "status": status,
                "response_headers": dict(response.headers.items()),
                "raw_body": raw_body,
                "response": parsed,
                "decode_note": decode_error,
                "error": "",
                "elapsed_seconds": time.monotonic() - started,
            }
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "url": url,
            "status": int(exc.code),
            "response_headers": dict(exc.headers.items()) if exc.headers else {},
            "raw_body": raw_body,
            "response": None,
            "decode_note": "",
            "error": f"HTTPError: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        }
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        return {
            "ok": False,
            "url": url,
            "status": None,
            "response_headers": {},
            "raw_body": "",
            "response": None,
            "decode_note": "",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        }


def parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for match in LABEL_RE.finditer(raw):
        value = bytes(match.group(2), "utf-8").decode("unicode_escape")
        labels[match.group(1)] = value
    return labels


def selected_metrics(raw: str) -> dict[str, float]:
    values: dict[str, float] = {
        "prefix_cache_queries": 0.0,
        "prefix_cache_hits": 0.0,
        "external_prefix_cache_queries": 0.0,
        "external_prefix_cache_hits": 0.0,
        "prompt_tokens": 0.0,
        "prompt_tokens_cached": 0.0,
        "local_compute": 0.0,
        "local_cache_hit": 0.0,
        "external_kv_transfer": 0.0,
    }
    direct_names = {
        "vllm:prefix_cache_queries_total": "prefix_cache_queries",
        "vllm:prefix_cache_hits_total": "prefix_cache_hits",
        "vllm:external_prefix_cache_queries_total": "external_prefix_cache_queries",
        "vllm:external_prefix_cache_hits_total": "external_prefix_cache_hits",
        "vllm:prompt_tokens_total": "prompt_tokens",
        "vllm:prompt_tokens_cached_total": "prompt_tokens_cached",
    }
    for line in raw.splitlines():
        match = METRIC_LINE_RE.match(line.strip())
        if not match or match.group(1) not in METRIC_NAMES:
            continue
        name, raw_labels, raw_value = match.groups()
        value = float(raw_value)
        if name in direct_names:
            values[direct_names[name]] += value
            continue
        labels = parse_labels(raw_labels or "")
        source = labels.get("source")
        if source in {"local_compute", "local_cache_hit", "external_kv_transfer"}:
            values[source] += value
    return values


def get_metrics(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/metrics"
    started = time.monotonic()
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
            return {
                "ok": 200 <= status < 300,
                "url": url,
                "status": status,
                "raw_body": raw_body,
                "selected": selected_metrics(raw_body),
                "error": "",
                "elapsed_seconds": time.monotonic() - started,
            }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raw_body = ""
        status: int | None = None
        if isinstance(exc, urllib.error.HTTPError):
            status = int(exc.code)
            raw_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "url": url,
            "status": status,
            "raw_body": raw_body,
            "selected": {},
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        }


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    before_values = before.get("selected") or {}
    after_values = after.get("selected") or {}
    return {
        key: float(after_values.get(key, 0.0)) - float(before_values.get(key, 0.0))
        for key in sorted(set(before_values) | set(after_values))
    }


def response_message(call: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if not call.get("ok"):
        return None, str(call.get("error") or f"HTTP {call.get('status')}")
    response = call.get("response")
    if not isinstance(response, dict):
        return None, "response JSON is not an object"
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "response has no choices"
    first = choices[0]
    if not isinstance(first, dict):
        return None, "first choice is not an object"
    message = first.get("message")
    if not isinstance(message, dict):
        return None, "first choice has no message object"
    return message, ""


def message_text(message: dict[str, Any]) -> tuple[str, str]:
    content = message.get("content")
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    return (content if isinstance(content, str) else "", reasoning if isinstance(reasoning, str) else "")


def repetition_evidence(text: str) -> dict[str, Any] | None:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    shingles: dict[str, int] = {}
    for offset in range(0, max(len(normalized) - 60, 0), 30):
        shingle = normalized[offset : offset + 60]
        shingles[shingle] = shingles.get(shingle, 0) + 1
        if shingles[shingle] >= 6:
            return {
                "detector": "60-character shingle at 30-character stride",
                "threshold": 6,
                "count": shingles[shingle],
                "shingle": shingle,
            }
    lines: dict[str, int] = {}
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip().lower()
        if len(line) < 24:
            continue
        lines[line] = lines.get(line, 0) + 1
        if lines[line] >= 5:
            return {
                "detector": "normalized repeated line",
                "threshold": 5,
                "count": lines[line],
                "line": line[:240],
            }
    return None


def classify_completion(call: dict[str, Any]) -> dict[str, Any]:
    message, schema_error = response_message(call)
    if schema_error:
        return {"category": "runtime_error", "reason": schema_error, "repetition": None}
    assert message is not None
    content, reasoning = message_text(message)
    combined = f"{content}\n{reasoning}"
    repetition = repetition_evidence(combined)
    if repetition:
        return {"category": "repetition", "reason": repetition["detector"], "repetition": repetition}
    if not content.strip():
        return {
            "category": "wrong_answer",
            "reason": "empty visible content",
            "repetition": None,
        }
    cjk = sum(1 for character in combined if "\u4e00" <= character <= "\u9fff")
    if len(combined) > 500 and cjk / len(combined) > 0.02:
        return {
            "category": "wrong_answer",
            "reason": f"English-prompt CJK drift ({cjk / len(combined):.1%})",
            "repetition": None,
        }
    return {"category": "clean", "reason": "", "repetition": None}


def load_bench(path: Path) -> tuple[ModuleType, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("_r26_quality_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {
        "path": str(path),
        "sha256": file_sha256(path),
        "version": str(getattr(module, "VERSION", "")),
        "reused_symbols": [
            "decode_builtin_test_profile_prompt",
            "extract_final_answer",
            "score_completion_profile",
        ],
    }


def build_reference() -> str:
    lines = [
        "This is an immutable synthetic archive. Read it, ignore decorative entries, and answer only the final question."
    ]
    vocabulary = (
        "amber cobalt delta ember fern granite harbor indigo juniper kestrel lunar maple "
        "nickel ochre prairie quartz river saffron timber umber violet willow xenon yarrow zephyr"
    ).split()
    for index in range(1200):
        words = " ".join(vocabulary[(index + step * 3) % len(vocabulary)] for step in range(18))
        line = f"Archive line {index:04d}: {words}; checksum {index * 7919 % 104729:06d}."
        if index == 600:
            line += f" AUTHORITATIVE ANSWERCODE: {CODEWORD}."
        lines.append(line)
    return "\n".join(lines)


def normalized_suite_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(requests))
    for request in normalized:
        request["cache_salt"] = "<per-execution-cache-salt>"
    return normalized


def make_chat_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    cache_salt: str,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 1.0,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "reasoning_effort": "low",
        "cache_salt": cache_salt,
    }


def score_cache_response(
    call: dict[str, Any],
    *,
    verifier: str,
    bench: ModuleType,
    hotel_profile: dict[str, Any],
) -> dict[str, Any]:
    base = classify_completion(call)
    message, schema_error = response_message(call)
    if schema_error:
        return {**base, "correct": False, "verifier": verifier, "verifier_detail": schema_error}
    assert message is not None
    content, reasoning = message_text(message)
    if verifier == "codeword_exact":
        correct = CODEWORD.casefold() in content.casefold()
        detail: object = {"expected_codeword": CODEWORD, "found": correct}
    elif verifier == "benchmark:hotel-lights:numeric_exact":
        final_answer = bench.extract_final_answer(content)
        score = bench.score_completion_profile(
            profile=hotel_profile,
            final_answer=final_answer,
            content_text=content,
            output_text=f"{reasoning}\n{content}",
            regex="",
            source="final_answer",
        )
        correct = score.get("correct") is True
        detail = score
    else:
        raise ValueError(f"unknown verifier {verifier}")
    if base["category"] == "clean" and not correct:
        base = {"category": "wrong_answer", "reason": "answer verifier failed", "repetition": None}
    response = call.get("response") or {}
    first_choice = ((response.get("choices") or [{}])[0]) if isinstance(response, dict) else {}
    return {
        **base,
        "correct": bool(correct),
        "verifier": verifier,
        "verifier_detail": detail,
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "content": content,
        "reasoning": reasoning,
    }


def run_cache_suite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started_at = utc_now()
    bench, verifier_source = load_bench(Path(args.bench_path))
    hotel_prompt, _, hotel_profile = bench.decode_builtin_test_profile_prompt("hotel-lights")
    reference = build_reference()
    needle_prompt = (
        f"{reference}\n\nRepeat the exact authoritative ANSWERCODE from the archive. "
        "Answer with the codeword only."
    )
    hotel_long_prompt = (
        f"{reference}\n\nThe archive above is padding for a prefix-cache correctness check. "
        "Now solve this independently:\n\n"
        f"{hotel_prompt}"
    )
    cache_salt = args.cache_salt or f"r26-quality-{uuid.uuid4().hex}"
    cold_payload = make_chat_payload(
        model=args.model,
        prompt=needle_prompt,
        max_tokens=2048,
        seed=26090501,
        cache_salt=cache_salt,
    )
    exact_payload = json.loads(json.dumps(cold_payload))
    partial_payload = make_chat_payload(
        model=args.model,
        prompt=hotel_long_prompt,
        max_tokens=4096,
        seed=26090502,
        cache_salt=cache_salt,
    )
    planned = [
        ("cold", cold_payload, "codeword_exact"),
        ("exact_replay", exact_payload, "codeword_exact"),
        ("shared_prefix_new_suffix", partial_payload, "benchmark:hotel-lights:numeric_exact"),
    ]
    resets: list[dict[str, Any]] = []
    if args.reset_local_between:
        initial_reset = post_empty(args.base_url, "/reset_prefix_cache")
        resets.append({"before_phase": "cold", **initial_reset})

    snapshots = [get_metrics(args.base_url)]
    outcomes: list[dict[str, Any]] = []
    for index, (phase, payload, verifier) in enumerate(planned):
        call = post_json(args.base_url, "/v1/chat/completions", payload, args.timeout)
        assessment = score_cache_response(
            call,
            verifier=verifier,
            bench=bench,
            hotel_profile=hotel_profile,
        )
        outcomes.append(
            {
                "phase": phase,
                "cache_expectation": "cold compute" if phase == "cold" else "cache-hit eligible",
                "assessment": assessment,
                "raw_api": call,
            }
        )
        snapshots.append(get_metrics(args.base_url))
        if args.reset_local_between and index < len(planned) - 1:
            time.sleep(args.store_wait_seconds)
            reset = post_empty(args.base_url, "/reset_prefix_cache")
            resets.append({"before_phase": planned[index + 1][0], **reset})

    phase_deltas = [
        {"phase": planned[index][0], **metric_delta(snapshots[index], snapshots[index + 1])}
        for index in range(len(planned))
    ]
    categories = {name: 0 for name in ("clean", "wrong_answer", "repetition", "runtime_error")}
    for outcome in outcomes:
        categories[outcome["assessment"]["category"]] += 1
    metrics_errors = sum(not snapshot.get("ok", False) for snapshot in snapshots)
    reset_errors = sum(not reset.get("ok", False) for reset in resets)
    correct = sum(outcome["assessment"].get("correct") is True for outcome in outcomes)
    replay_delta = phase_deltas[1]
    partial_delta = phase_deltas[2]
    local_hit_observed = (
        replay_delta.get("local_cache_hit", 0.0) > 0
        and partial_delta.get("local_cache_hit", 0.0) > 0
    )
    external_hit_observed = (
        replay_delta.get("external_kv_transfer", 0.0) > 0
        and partial_delta.get("external_kv_transfer", 0.0) > 0
    )
    receipt = {
        "schema": SCHEMA,
        "suite": "cache_correctness",
        "label": args.label,
        "started_at": started_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "model": args.model,
        "request_policy": {
            "reasoning_effort": "low",
            "cold_then_exact_replay_then_shared_prefix_new_suffix": True,
            "reset_local_between": bool(args.reset_local_between),
            "external_cache_expected": bool(args.external_cache_expected),
            "store_wait_seconds": args.store_wait_seconds,
            "cache_salt": cache_salt,
            "weights_changed": False,
        },
        "prompt_evidence": {
            "shared_reference_chars": len(reference),
            "shared_reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            "needle_prompt_sha256": hashlib.sha256(needle_prompt.encode()).hexdigest(),
            "partial_prompt_sha256": hashlib.sha256(hotel_long_prompt.encode()).hexdigest(),
            "codeword": CODEWORD,
        },
        "verifier_source": verifier_source,
        "suite_fingerprint": canonical_hash(
            normalized_suite_requests([payload for _, payload, _ in planned])
        ),
        "resets": resets,
        "metric_snapshots": snapshots,
        "cache_effects": {
            "phase_deltas": phase_deltas,
            "cold_compute_observed": phase_deltas[0].get("local_compute", 0.0) > 0,
            "local_hit_observed_on_both_replays": local_hit_observed,
            "external_hit_observed_on_both_replays": external_hit_observed,
        },
        "outcomes": outcomes,
        "summary": {
            "requested": len(outcomes),
            "completed": len(outcomes) - categories["runtime_error"],
            "correct": correct,
            "wrong_answers": categories["wrong_answer"],
            "repetitions": categories["repetition"],
            "runtime_errors": categories["runtime_error"],
            "metrics_errors": metrics_errors,
            "reset_errors": reset_errors,
            "categories": categories,
        },
        "finished_at": utc_now(),
    }
    runtime_failure = categories["runtime_error"] > 0 or metrics_errors > 0 or reset_errors > 0
    return receipt, int(runtime_failure)


def build_long_prompt(topic: str) -> str:
    return (
        f"Write a detailed, structured technical essay about {topic}. "
        "Cover origins, key mechanisms, important figures or components, and lasting impact. "
        "Use clear section headings. "
        + ("Filler context for length. " * 1500)
    )


async def run_long_wave(
    args: argparse.Namespace,
    *,
    wave: int,
    cache_salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    before = await asyncio.to_thread(get_metrics, args.base_url)
    tasks = []
    payloads: list[dict[str, Any]] = []
    for topic_index, topic in enumerate(LONG_TOPICS):
        payload = make_chat_payload(
            model=args.model,
            prompt=build_long_prompt(topic),
            max_tokens=args.max_tokens,
            seed=args.seed_base + wave * len(LONG_TOPICS) + topic_index,
            cache_salt=cache_salt,
            temperature=0.7,
            top_k=-1,
            top_p=0.95,
        )
        payloads.append(payload)
        tasks.append(
            asyncio.to_thread(
                post_json,
                args.base_url,
                "/v1/chat/completions",
                payload,
                args.timeout,
            )
        )
    calls = await asyncio.gather(*tasks)
    after = await asyncio.to_thread(get_metrics, args.base_url)
    results: list[dict[str, Any]] = []
    for topic_index, (topic, payload, call) in enumerate(zip(LONG_TOPICS, payloads, calls)):
        assessment = classify_completion(call)
        message, schema_error = response_message(call)
        content = ""
        reasoning = ""
        if message is not None:
            content, reasoning = message_text(message)
        response = call.get("response") or {}
        first_choice = ((response.get("choices") or [{}])[0]) if isinstance(response, dict) else {}
        usage = response.get("usage") if isinstance(response, dict) else {}
        results.append(
            {
                "wave": wave + 1,
                "topic_index": topic_index,
                "topic": topic,
                "apollo_family": topic.lower().startswith("apollo") or "apollo" in topic.lower(),
                "seed": payload["seed"],
                "assessment": assessment,
                "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
                "usage": usage if isinstance(usage, dict) else {},
                "visible_chars": len(content),
                "reasoning_chars": len(reasoning),
                "schema_error": schema_error,
                "raw_api": call,
            }
        )
    return results, before, after


async def run_long_suite_async(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started_at = utc_now()
    if args.waves != 3:
        raise ValueError("the R26 corruption qualification is fixed at exactly three waves")
    cache_salt = args.cache_salt or f"r26-quality-long-{uuid.uuid4().hex}"
    all_results: list[dict[str, Any]] = []
    wave_metrics: list[dict[str, Any]] = []
    planned_payloads: list[dict[str, Any]] = []
    for wave in range(args.waves):
        for topic_index, topic in enumerate(LONG_TOPICS):
            planned_payloads.append(
                make_chat_payload(
                    model=args.model,
                    prompt=build_long_prompt(topic),
                    max_tokens=args.max_tokens,
                    seed=args.seed_base + wave * len(LONG_TOPICS) + topic_index,
                    cache_salt=cache_salt,
                    temperature=0.7,
                    top_k=-1,
                    top_p=0.95,
                )
            )
        results, before, after = await run_long_wave(
            args,
            wave=wave,
            cache_salt=cache_salt,
        )
        all_results.extend(results)
        wave_metrics.append(
            {
                "wave": wave + 1,
                "cache_expectation": "cold" if wave == 0 else "exact-prompt replay eligible",
                "before": before,
                "after": after,
                "delta": metric_delta(before, after),
            }
        )
    categories = {name: 0 for name in ("clean", "wrong_answer", "repetition", "runtime_error")}
    for result in all_results:
        categories[result["assessment"]["category"]] += 1
    metrics_errors = sum(
        not item[side].get("ok", False)
        for item in wave_metrics
        for side in ("before", "after")
    )
    receipt = {
        "schema": SCHEMA,
        "suite": "long_generation_corruption",
        "label": args.label,
        "started_at": started_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "model": args.model,
        "method": {
            "derived_from": "/home/josh/omp-workspace/corruption-hunt.py",
            "waves": args.waves,
            "wave_size": len(LONG_TOPICS),
            "concurrency_per_wave": len(LONG_TOPICS),
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
            "top_k": -1,
            "top_p": 0.95,
            "reasoning_effort": "low",
            "seed_base": args.seed_base,
            "cache_salt": cache_salt,
            "apollo_observations": args.waves * sum("apollo" in topic.lower() for topic in LONG_TOPICS),
            "weights_changed": False,
        },
        "topics": LONG_TOPICS,
        "suite_fingerprint": canonical_hash(normalized_suite_requests(planned_payloads)),
        "wave_metrics": wave_metrics,
        "results": all_results,
        "summary": {
            "requested": args.waves * len(LONG_TOPICS),
            "completed": len(all_results) - categories["runtime_error"],
            "clean": categories["clean"],
            "wrong_answers": categories["wrong_answer"],
            "repetitions": categories["repetition"],
            "runtime_errors": categories["runtime_error"],
            "metrics_errors": metrics_errors,
            "apollo_requested": sum(result["apollo_family"] for result in all_results),
            "apollo_repetitions": sum(
                result["apollo_family"] and result["assessment"]["category"] == "repetition"
                for result in all_results
            ),
            "categories": categories,
        },
        "finished_at": utc_now(),
    }
    return receipt, int(categories["runtime_error"] > 0 or metrics_errors > 0)


def run_long_suite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return asyncio.run(run_long_suite_async(args))


async def run_sampling_setting(
    args: argparse.Namespace,
    *,
    setting_index: int,
    setting: dict[str, Any],
    cache_salt: str,
) -> list[dict[str, Any]]:
    support = ["41", "42", "43", "44"]
    prompt = (
        "What is 6 multiplied by 7? Select exactly one answer from 41, 42, 43, or 44. "
        "Return only the selected number."
    )
    payloads = []
    tasks = []
    for run_index in range(args.runs_per_setting):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": setting["temperature"],
            "top_k": setting["top_k"],
            "top_p": setting["top_p"],
            "seed": args.seed_base + setting_index * 100 + run_index,
            "reasoning_effort": "low",
            "cache_salt": cache_salt,
            "structured_outputs": {"choice": support},
            "logprobs": True,
            "top_logprobs": 5,
        }
        payloads.append(payload)
        tasks.append(
            asyncio.to_thread(
                post_json,
                args.base_url,
                "/v1/chat/completions",
                payload,
                args.timeout,
            )
        )
    calls = await asyncio.gather(*tasks)
    results: list[dict[str, Any]] = []
    for run_index, (payload, call) in enumerate(zip(payloads, calls)):
        assessment = classify_completion(call)
        message, schema_error = response_message(call)
        content = ""
        reasoning = ""
        if message is not None:
            content, reasoning = message_text(message)
        stripped = content.strip().strip("`\"'").strip()
        valid_support = stripped in support
        correct = stripped == "42"
        if assessment["category"] == "clean" and not valid_support:
            assessment = {
                "category": "wrong_answer",
                "reason": "output is outside declared structured choice support",
                "repetition": None,
            }
        elif assessment["category"] == "clean" and not correct:
            assessment = {
                "category": "wrong_answer",
                "reason": f"arithmetic verifier expected 42, received {stripped!r}",
                "repetition": None,
            }
        results.append(
            {
                "setting_index": setting_index,
                "run_index": run_index,
                "setting": setting,
                "seed": payload["seed"],
                "parsed_choice": stripped,
                "valid_support": valid_support,
                "correct": correct,
                "schema_error": schema_error,
                "assessment": assessment,
                "content": content,
                "reasoning": reasoning,
                "raw_api": call,
            }
        )
    return results


async def run_sampling_suite_async(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started_at = utc_now()
    settings = [
        {"name": "greedy-control", "temperature": 0.0, "top_k": -1, "top_p": 1.0},
        {"name": "low-temperature-tight", "temperature": 0.2, "top_k": 5, "top_p": 0.8},
        {"name": "representative", "temperature": 0.7, "top_k": 20, "top_p": 0.9},
        {"name": "high-temperature-wide", "temperature": 1.0, "top_k": 50, "top_p": 0.95},
    ]
    cache_salt = args.cache_salt or f"r26-quality-sampling-{uuid.uuid4().hex}"
    all_results: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(settings):
        all_results.extend(
            await run_sampling_setting(
                args,
                setting_index=setting_index,
                setting=setting,
                cache_salt=cache_salt,
            )
        )
    categories = {name: 0 for name in ("clean", "wrong_answer", "repetition", "runtime_error")}
    for result in all_results:
        categories[result["assessment"]["category"]] += 1
    per_setting = []
    for setting_index, setting in enumerate(settings):
        rows = [result for result in all_results if result["setting_index"] == setting_index]
        per_setting.append(
            {
                **setting,
                "requested": len(rows),
                "runtime_errors": sum(row["assessment"]["category"] == "runtime_error" for row in rows),
                "valid_support": sum(row["valid_support"] for row in rows),
                "correct": sum(row["correct"] for row in rows),
                "observed_choices": sorted({row["parsed_choice"] for row in rows}),
            }
        )
    normalized_requests = [
        {**result["raw_api"]["request"], "cache_salt": "<per-execution-cache-salt>"}
        for result in all_results
    ]
    receipt = {
        "schema": SCHEMA,
        "suite": "dflash_constrained_sampling",
        "label": args.label,
        "started_at": started_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "model": args.model,
        "claim_scope": (
            "This observes API acceptance, structured-choice support validity, and a real arithmetic "
            "verifier at representative temperature/top-k/top-p settings. These samples do not establish "
            "distribution equivalence."
        ),
        "support": ["41", "42", "43", "44"],
        "expected_answer": "42",
        "request_policy": {
            "runs_per_setting": args.runs_per_setting,
            "reasoning_effort": "low",
            "structured_outputs": {"choice": ["41", "42", "43", "44"]},
            "logprobs": True,
            "top_logprobs": 5,
            "cache_salt": cache_salt,
            "weights_changed": False,
        },
        "settings": per_setting,
        "suite_fingerprint": canonical_hash(normalized_requests),
        "results": all_results,
        "summary": {
            "requested": len(all_results),
            "completed": len(all_results) - categories["runtime_error"],
            "valid_support": sum(result["valid_support"] for result in all_results),
            "correct": sum(result["correct"] for result in all_results),
            "wrong_answers": categories["wrong_answer"],
            "repetitions": categories["repetition"],
            "runtime_errors": categories["runtime_error"],
            "categories": categories,
        },
        "finished_at": utc_now(),
    }
    return receipt, int(categories["runtime_error"] > 0)


def run_sampling_suite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return asyncio.run(run_sampling_suite_async(args))


async def run_profile_calls(
    args: argparse.Namespace,
    payloads: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(payload: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(
                post_json,
                args.base_url,
                "/v1/chat/completions",
                payload,
                args.timeout,
            )

    return await asyncio.gather(*(one(payload) for payload in payloads))


async def run_profile_suite_async(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started_at = utc_now()
    bench, verifier_source = load_bench(Path(args.bench_path))
    prompt, prompt_source, profile = bench.decode_builtin_test_profile_prompt(args.profile_name)
    concurrency = args.concurrency or int(profile.get("default_concurrency") or 1)
    if concurrency <= 0:
        raise ValueError("profile concurrency must be positive")
    default_max_tokens = int(profile.get("default_max_tokens") or 0)
    cache_salt = args.cache_salt or f"r26-quality-profile-{uuid.uuid4().hex}"
    payloads: list[dict[str, Any]] = []
    for run_index in range(args.runs):
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": args.seed_base + run_index,
            "reasoning_effort": "low",
            "cache_salt": cache_salt,
        }
        if default_max_tokens > 0:
            payload["max_tokens"] = default_max_tokens
        payloads.append(payload)

    before_metrics = await asyncio.to_thread(get_metrics, args.base_url)
    prefill_scout_enabled = not bool(profile.get("default_no_prefill_scout", False))
    scout: dict[str, Any] | None = None
    scout_runtime_error = 0
    if prefill_scout_enabled:
        scout_payload = json.loads(json.dumps(payloads[0]))
        scout_payload["max_tokens"] = 1
        scout = await asyncio.to_thread(
            post_json,
            args.base_url,
            "/v1/chat/completions",
            scout_payload,
            args.timeout,
        )
        _, scout_schema_error = response_message(scout)
        scout_runtime_error = int(bool(scout_schema_error))
    calls = await run_profile_calls(args, payloads, concurrency)
    after_metrics = await asyncio.to_thread(get_metrics, args.base_url)
    results: list[dict[str, Any]] = []
    for run_index, (payload, call) in enumerate(zip(payloads, calls)):
        assessment = classify_completion(call)
        message, schema_error = response_message(call)
        content = ""
        reasoning = ""
        if message is not None:
            content, reasoning = message_text(message)
        output_text = f"{reasoning}{content}"
        final_answer = bench.extract_final_answer(content or output_text)
        score: dict[str, Any] = {
            "correct": False,
            "score_label": "fail",
            "score_detail": schema_error or "response unavailable",
            "parsed_answer": "",
        }
        if not schema_error:
            score = bench.score_completion_profile(
                profile=profile,
                final_answer=final_answer,
                content_text=content,
                output_text=output_text,
                regex=str(profile.get("correct_regex") or ""),
                source=str(profile.get("score_source") or "final_answer"),
            )
            if assessment["category"] == "clean" and score.get("correct") is not True:
                assessment = {
                    "category": "wrong_answer",
                    "reason": "profile answer verifier failed",
                    "repetition": None,
                }
        response = call.get("response") or {}
        first_choice = ((response.get("choices") or [{}])[0]) if isinstance(response, dict) else {}
        results.append(
            {
                "run_index": run_index,
                "seed": payload["seed"],
                "assessment": assessment,
                "verifier": score,
                "finish_reason": (
                    first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
                ),
                "final_answer": final_answer,
                "visible_chars": len(content),
                "reasoning_chars": len(reasoning),
                "schema_error": schema_error,
                "raw_api": call,
            }
        )

    categories = {name: 0 for name in ("clean", "wrong_answer", "repetition", "runtime_error")}
    for result in results:
        categories[result["assessment"]["category"]] += 1
    runtime_errors = categories["runtime_error"]
    completed = len(results) - runtime_errors
    scored = sum(
        result["assessment"]["category"] != "runtime_error"
        and isinstance(result["verifier"].get("correct"), bool)
        for result in results
    )
    correct = sum(
        result["assessment"]["category"] != "runtime_error"
        and result["verifier"].get("correct") is True
        for result in results
    )
    verifier_wrong = sum(
        result["assessment"]["category"] != "runtime_error"
        and result["verifier"].get("correct") is False
        for result in results
    )
    score_counts: dict[str, int] = {}
    for result in results:
        label = str(result["verifier"].get("score_label") or "")
        if label:
            score_counts[label] = score_counts.get(label, 0) + 1
    metrics_errors = int(not before_metrics.get("ok", False)) + int(
        not after_metrics.get("ok", False)
    )
    selected_summary = {
        "attempted": args.runs,
        "completed": completed,
        "errors": runtime_errors,
        "scored": scored,
        "correct": correct,
        "wrong": verifier_wrong,
        "wrong_answers": categories["wrong_answer"],
        "repetitions": categories["repetition"],
        "score_counts": score_counts,
        "exact": score_counts.get("exact", 0),
        "near": score_counts.get("near", 0),
        "fail": score_counts.get("fail", 0),
        "score_available": scored == completed,
        "correct_rate": correct / completed if completed else None,
    }
    metadata = {
        "mode": "quality_profile_raw_api",
        "model": args.model,
        "test_profile": args.profile_name,
        "test_profile_description": profile.get("description"),
        "prompt_source": prompt_source,
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_tokens": default_max_tokens if default_max_tokens > 0 else None,
        "max_tokens_omitted": default_max_tokens <= 0,
        "fixed_concurrency": concurrency,
        "requested_runs": args.runs,
        "concurrency_levels_requested": [concurrency],
        "min_results": args.runs,
        "probe_waves": 1,
        "auto_stop": False,
        "correct_regex": profile.get("correct_regex") or "",
        "score_source": profile.get("score_source") or "final_answer",
        "profile_scorer": profile.get("scorer") or "regex",
        "expected_answer": (
            profile.get("expected_number")
            if profile.get("expected_number") is not None
            else (
                f"{profile.get('expected_count')}, {profile.get('expected_hours')}"
                if profile.get("expected_count") is not None
                else ""
            )
        ),
        "approx_tolerance": profile.get("approx_tolerance"),
        "dataset_rows": profile.get("dataset_rows"),
        "dataset_sha256": profile.get("dataset_sha256"),
        "prefill_scout": prefill_scout_enabled,
        "temperature": None,
        "top_p": None,
        "reasoning_effort": "low",
        "seed_base": args.seed_base,
        "cache_salt": cache_salt,
    }
    receipt = {
        "schema": SCHEMA,
        "suite": "verified_quality_profile",
        "label": args.label,
        "started_at": started_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "metadata": metadata,
        "verifier_source": verifier_source,
        "suite_fingerprint": canonical_hash(normalized_suite_requests(payloads)),
        "metrics": {
            "before": before_metrics,
            "after": after_metrics,
            "delta": metric_delta(before_metrics, after_metrics),
            "errors": metrics_errors,
        },
        "prefill_scout": {
            "enabled": prefill_scout_enabled,
            "max_tokens": 1 if prefill_scout_enabled else None,
            "runtime_errors": scout_runtime_error,
            "raw_api": scout,
        },
        "selected_summary": selected_summary,
        "wrong_runs": [
            {
                "run_index": result["run_index"],
                "seed": result["seed"],
                "assessment": result["assessment"],
                "verifier": result["verifier"],
                "finish_reason": result["finish_reason"],
                "final_answer": result["final_answer"],
                "schema_error": result["schema_error"],
            }
            for result in results
            if result["verifier"].get("correct") is not True
            or result["assessment"]["category"] == "repetition"
        ],
        "results": results,
        "summary": {
            "requested": args.runs,
            "completed": completed,
            "runtime_errors": runtime_errors,
            "prefill_scout_runtime_errors": scout_runtime_error,
            "wrong_answers": categories["wrong_answer"],
            "repetitions": categories["repetition"],
            "metrics_errors": metrics_errors,
        },
        "finished_at": utc_now(),
    }
    return receipt, int(
        runtime_errors > 0 or scout_runtime_error > 0 or metrics_errors > 0
    )


def run_profile_suite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return asyncio.run(run_profile_suite_async(args))


def tool_call_arguments_valid(tool_call: dict[str, Any]) -> tuple[bool, object]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return False, "missing function object"
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return False, "arguments are not a JSON string"
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return False, f"invalid argument JSON: {exc}"
    valid = isinstance(parsed, dict) and str(parsed.get("city", "")).casefold() == "tokyo"
    return valid, parsed


def run_tool_order_suite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started_at = utc_now()
    first_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use both tools for Tokyo: get the temperature and the population. "
                    "Call them together in one turn."
                ),
            }
        ],
        "tools": TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "max_tokens": 600,
        "temperature": 0.0,
        "seed": 26090521,
        "reasoning_effort": "low",
    }
    first = post_json(args.base_url, "/v1/chat/completions", first_payload, args.timeout)
    first_message, first_schema_error = response_message(first)
    runtime_errors = int(bool(first_schema_error))
    second: dict[str, Any] | None = None
    semantic = {
        "exact_expected_calls": False,
        "arguments_valid": False,
        "result_submission_reversed": False,
        "rendered_prompt_exposed": False,
        "rendered_result_order_matches_call_order": False,
        "population_cited": False,
        "temperature_cited": False,
        "answer_uses_both_results": False,
    }
    call_rows: list[dict[str, Any]] = []
    submitted_results: list[dict[str, Any]] = []
    first_call_order: list[str] = []
    second_content = ""
    second_reasoning = ""
    prompt_text = ""

    if first_message is not None:
        tool_calls = first_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        expected_names = {"get_city_temp", "get_city_population"}
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                call_rows.append({"valid": False, "detail": "tool call is not an object", "raw": tool_call})
                continue
            function = tool_call.get("function") or {}
            name = function.get("name") if isinstance(function, dict) else None
            arguments_valid, detail = tool_call_arguments_valid(tool_call)
            valid = (
                arguments_valid
                and tool_call.get("type") == "function"
                and isinstance(tool_call.get("id"), str)
                and bool(tool_call.get("id"))
            )
            first_call_order.append(str(name or ""))
            call_rows.append(
                {
                    "id": tool_call.get("id"),
                    "name": name,
                    "valid": valid,
                    "detail": detail,
                    "raw": tool_call,
                }
            )
        semantic["exact_expected_calls"] = (
            len(call_rows) == 2
            and {row.get("name") for row in call_rows} == expected_names
            and len({row.get("id") for row in call_rows}) == 2
        )
        semantic["arguments_valid"] = bool(call_rows) and all(row.get("valid") for row in call_rows)

        if semantic["exact_expected_calls"]:
            fixtures = {
                "get_city_temp": {
                    "city": "Tokyo",
                    "temp_c": 31,
                    "fixture_receipt": "TEMPERATURE_RESULT_R26",
                },
                "get_city_population": {
                    "city": "Tokyo",
                    "population": 13960000,
                    "fixture_receipt": "POPULATION_RESULT_R26",
                },
            }
            messages = list(first_payload["messages"])
            messages.append(first_message)
            for row in reversed(call_rows):
                fixture = fixtures[str(row["name"])]
                tool_message = {
                    "role": "tool",
                    "tool_call_id": row["id"],
                    "content": json.dumps(fixture, separators=(",", ":")),
                }
                messages.append(tool_message)
                submitted_results.append(tool_message)
            semantic["result_submission_reversed"] = [
                next(row["name"] for row in call_rows if row["id"] == result["tool_call_id"])
                for result in submitted_results
            ] == list(reversed(first_call_order))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Which is bigger, Tokyo's population in millions or its temperature in Celsius? "
                        "State both numbers received from the tools."
                    ),
                }
            )
            second_payload = {
                "model": args.model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "none",
                "parallel_tool_calls": True,
                "max_tokens": 600,
                "temperature": 0.0,
                "seed": 26090522,
                "reasoning_effort": "low",
                "return_prompt_text": True,
            }
            second = post_json(args.base_url, "/v1/chat/completions", second_payload, args.timeout)
            second_message, second_schema_error = response_message(second)
            if second_schema_error:
                runtime_errors += 1
            elif second_message is not None:
                second_content, second_reasoning = message_text(second_message)
                response = second.get("response")
                if isinstance(response, dict) and isinstance(response.get("prompt_text"), str):
                    prompt_text = response["prompt_text"]
                semantic["rendered_prompt_exposed"] = bool(prompt_text)
                tags = {
                    "get_city_temp": "TEMPERATURE_RESULT_R26",
                    "get_city_population": "POPULATION_RESULT_R26",
                }
                tag_positions = {name: prompt_text.find(tag) for name, tag in tags.items()}
                semantic["rendered_result_order_matches_call_order"] = (
                    bool(prompt_text)
                    and all(position >= 0 for position in tag_positions.values())
                    and sorted(first_call_order, key=lambda name: tag_positions[name]) == first_call_order
                )
                answer = second_content
                semantic["population_cited"] = bool(
                    re.search(r"(?:13[.,]96\s*(?:million|m)|13[ ,]?960[ ,]?000)", answer, re.I)
                )
                semantic["temperature_cited"] = bool(
                    re.search(r"\b31\b", answer)
                    and re.search(r"(?:celsius|degree|temperature|°\s*c)", answer, re.I)
                )
                semantic["answer_uses_both_results"] = (
                    semantic["population_cited"] and semantic["temperature_cited"]
                )

    parser_correct = all(semantic.values())
    receipt = {
        "schema": SCHEMA,
        "suite": "tool_result_reordering",
        "label": args.label,
        "started_at": started_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "model": args.model,
        "method": {
            "tool_results": "deterministic fixtures supplied in reverse of the model's actual call order",
            "fixtures_are_external_calls": False,
            "reasoning_effort": "low",
            "temperature": 0.0,
            "return_prompt_text": True,
            "wording_pinned": False,
            "weights_changed": False,
        },
        "first_call_order": first_call_order,
        "parsed_tool_calls": call_rows,
        "submitted_tool_results": submitted_results,
        "semantic_checks": semantic,
        "rendered_prompt": prompt_text,
        "second_content": second_content,
        "second_reasoning": second_reasoning,
        "raw_api": {"first": first, "second": second},
        "summary": {
            "requested_turns": 2,
            "completed_turns": 2 - runtime_errors,
            "runtime_errors": runtime_errors,
            "parser_and_reordering_correct": parser_correct,
            "wrong_answers": int(runtime_errors == 0 and not parser_correct),
            "repetitions": 0,
        },
        "finished_at": utc_now(),
    }
    return receipt, int(runtime_errors > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5002")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cache-salt", default="")
    subparsers = parser.add_subparsers(dest="suite", required=True)

    cache = subparsers.add_parser("cache", help="cold/replay/shared-prefix answer verification")
    cache.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH)
    cache.add_argument("--reset-local-between", action="store_true")
    cache.add_argument("--external-cache-expected", action="store_true")
    cache.add_argument("--store-wait-seconds", type=float, default=8.0)

    long_parser = subparsers.add_parser("long", help="three waves of eight long generations")
    long_parser.add_argument("--waves", type=int, default=3)
    long_parser.add_argument("--max-tokens", type=int, default=8192)
    long_parser.add_argument("--seed-base", type=int, default=26090500)

    sampling = subparsers.add_parser("sampling", help="DFlash constrained-sampling support probe")
    sampling.add_argument("--runs-per-setting", type=int, default=6)
    sampling.add_argument("--seed-base", type=int, default=26090600)

    profile = subparsers.add_parser(
        "profile",
        help="raw API output with the benchmark's built-in prompt and verifier",
    )
    profile.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH)
    profile.add_argument("--profile-name", required=True)
    profile.add_argument("--runs", type=int, default=24)
    profile.add_argument("--concurrency", type=int, default=0)
    profile.add_argument("--seed-base", type=int, default=26090700)

    subparsers.add_parser("tool-order", help="parallel tool parsing and reversed result ordering")
    return parser.parse_args(argv)


def run_selected(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.suite == "cache":
        return run_cache_suite(args)
    if args.suite == "long":
        return run_long_suite(args)
    if args.suite == "sampling":
        return run_sampling_suite(args)
    if args.suite == "profile":
        return run_profile_suite(args)
    if args.suite == "tool-order":
        return run_tool_order_suite(args)
    raise ValueError(f"unknown suite {args.suite}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt, status = run_selected(args)
    except Exception as exc:
        receipt = {
            "schema": SCHEMA,
            "suite": args.suite,
            "label": args.label,
            "status": "harness_error",
            "started_at": utc_now(),
            "summary": {
                "runtime_errors": 1,
                "wrong_answers": 0,
                "repetitions": 0,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "finished_at": utc_now(),
        }
        status = 2
    save_receipt(args.output, receipt)
    print(json.dumps({"output": str(args.output), "summary": receipt.get("summary", {})}, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
