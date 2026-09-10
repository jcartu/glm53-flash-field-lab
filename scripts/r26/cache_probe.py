#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL = "GLM-5.3-Flash-NVFP4"
CHAT_PATH = "/v1/chat/completions"
TOKENIZE_PATH = "/tokenize"
PROBE_SCHEMA = "r26-cache-probe/v2"
CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "low"}
UNIQUE_PREAMBLE_ROWS = 128
SEMANTIC_MAX_TOKENS = 512
NEEDLE_MAX_TOKENS = 1024
WORDS = (
    "atlas binder cobalt delta ember flint granite harbor ivory jasper kilo "
    "lunar mantle nectar opal pivot quartz raster sulfur tundra umber vector "
    "willow xenon yonder zephyr anchor bishop cipher dredger falcon gypsum"
).split()


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def json_safe(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex(), "bytes_length": len(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def http_json(
    port: int,
    path: str,
    *,
    payload: object | None,
    timeout: int,
    label: str,
) -> dict[str, Any]:
    raw_request = (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        if payload is not None
        else b""
    )
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": re.sub(r"[^A-Za-z0-9_.:-]", "-", label)[:200],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=raw_request if payload is not None or path == "/reset_prefix_cache" else None,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_response = response.read()
            status = int(response.status)
            response_headers = dict(response.headers.items())
        error = None
    except urllib.error.HTTPError as caught:
        raw_response = caught.read()
        status = int(caught.code)
        response_headers = dict(caught.headers.items())
        error = f"HTTPError: {caught}"
    except (OSError, TimeoutError) as caught:
        return {
            "label": label,
            "ok": False,
            "status": None,
            "request": payload,
            "request_body_sha256": hashlib.sha256(raw_request).hexdigest(),
            "response_headers": {},
            "raw_body": "",
            "raw_body_sha256": None,
            "parsed": None,
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(caught).__name__}: {caught}",
        }
    body_text = raw_response.decode(errors="replace")
    try:
        parsed: object | None = json.loads(body_text)
    except json.JSONDecodeError:
        parsed = None
    return {
        "label": label,
        "ok": 200 <= status < 300 and parsed is not None,
        "status": status,
        "request": payload,
        "request_body_sha256": hashlib.sha256(raw_request).hexdigest(),
        "response_headers": response_headers,
        "raw_body": body_text,
        "raw_body_sha256": hashlib.sha256(raw_response).hexdigest(),
        "parsed": parsed,
        "wall_seconds": time.perf_counter() - started,
        "error": error,
    }


def tokenize(
    port: int,
    model: str,
    content: str,
    timeout: int = 600,
    *,
    include_token_ids: bool = False,
) -> dict[str, Any]:
    result = http_json(
        port,
        TOKENIZE_PATH,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        },
        timeout=timeout,
        label="tokenize",
    )
    parsed = result.get("parsed")
    if (
        not result["ok"]
        or not isinstance(parsed, dict)
        or not isinstance(parsed.get("count"), int)
    ):
        raise RuntimeError(
            f"tokenization failed: {result.get('error') or result.get('raw_body')}"
        )
    count = int(parsed["count"])
    response: dict[str, Any] = {
        "count": count,
        "wall_seconds": result["wall_seconds"],
    }
    if include_token_ids:
        token_ids = parsed.get("tokens")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != count
            or not all(isinstance(token_id, int) for token_id in token_ids)
        ):
            raise RuntimeError(
                "tokenizer did not return an exact token-id sequence matching count"
            )
        response["token_ids"] = token_ids
    return response


def unique_preamble(identity: str) -> str:
    """Build an identity-specific prefix long enough to cover one 4K cache chunk.

    The identity is deliberately not repeated on every row.  The prior 256-row
    form repeated a long identity beside every 64-hex digest and itself consumed
    about 18.5K tokens.  These 128 independently derived rows retain a unique
    first cache block while leaving ample room inside a 16,384-token prompt.
    Exact coverage and fit are still established by the serving tokenizer in
    ``calibrate_prompt`` and ``calibrate_needle_prompt``.
    """
    rows = [f"UNIQUE-CACHE-FIRST-BLOCK {identity} BEGIN"]
    for index in range(UNIQUE_PREAMBLE_ROWS):
        digest = hashlib.sha256(f"{identity}:{index}".encode()).hexdigest()
        rows.append(f"row-{index:04d} {digest}")
    rows.append(f"UNIQUE-CACHE-FIRST-BLOCK {identity} END")
    return "\n".join(rows) + "\n"


def reference_code(identity: str) -> str:
    return "R26-" + hashlib.sha256(identity.encode()).hexdigest()[:20].upper()


def calibrate_prompt(
    port: int,
    model: str,
    target_tokens: int,
    identity: str,
    kind: str,
) -> tuple[str, dict[str, Any]]:
    preamble = unique_preamble(identity)
    if kind == "period":
        code = None
        suffix = "\nEND OF CACHE PROBE. Reply with one period."
    elif kind == "reference":
        code = reference_code(identity)
        suffix = (
            f"\nThe museum catalog reference for specimen Sigma is {code}. "
            "Reply with only that reference."
        )
    elif kind == "prefix":
        code = None
        suffix = "\nEND OF SHARED CACHE PREFIX."
    else:
        raise ValueError(f"unknown prompt kind: {kind}")
    preamble_tokenization = tokenize(
        port, model, preamble, include_token_ids=True
    )
    preamble_tokens = preamble_tokenization["count"]
    first_block_token_ids = preamble_tokenization["token_ids"][:4096]
    if len(first_block_token_ids) != 4096:
        raise RuntimeError(
            f"unique preamble is only {preamble_tokens} tokens; first cache chunk is not wholly unique"
        )
    first_block_token_ids_sha256 = hashlib.sha256(
        json.dumps(first_block_token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    fixed = tokenize(port, model, preamble + suffix)["count"]
    filler_count = target_tokens - fixed
    if filler_count < 0:
        raise RuntimeError(
            f"target {target_tokens} is smaller than fixed unique prompt ({fixed} tokens)"
        )
    observed = fixed
    content = ""
    attempts = []
    for _ in range(8):
        content = preamble + (" a" * filler_count) + suffix
        observed = tokenize(port, model, content)["count"]
        attempts.append({"filler_count": filler_count, "observed_tokens": observed})
        if observed == target_tokens:
            break
        filler_count += target_tokens - observed
        if filler_count < 0:
            raise RuntimeError("prompt calibration produced a negative filler count")
    else:
        raise RuntimeError(
            f"could not calibrate {target_tokens}-token prompt after {len(attempts)} attempts; got {observed}"
        )
    metadata = {
        "schema": PROBE_SCHEMA,
        "identity": identity,
        "kind": kind,
        "reference_code": code,
        "target_tokens": target_tokens,
        "observed_tokens": observed,
        "preamble_tokens": preamble_tokens,
        "preamble_rows": UNIQUE_PREAMBLE_ROWS,
        "unique_first_block_token_span": len(first_block_token_ids),
        "first_block_token_ids_sha256": first_block_token_ids_sha256,
        "preamble_chars": len(preamble),
        "preamble_sha256": hashlib.sha256(preamble.encode()).hexdigest(),
        "prompt_chars": len(content),
        "prompt_utf8_bytes": len(content.encode()),
        "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "calibration_attempts": attempts,
        "tokenizer_method": {
            "endpoint": TOKENIZE_PATH,
            "messages_api": True,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "exact_server_token_count_required": True,
            "exact_first_block_token_ids_required": True,
        },
    }
    return content, metadata


def cache_stats(record: dict[str, Any]) -> dict[str, int] | None:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    transfer = parsed.get("kv_transfer_params")
    if not isinstance(transfer, dict):
        return None
    stats = transfer.get("cached_token_stats")
    if not isinstance(stats, dict):
        return None
    required = (
        "num_vllm_cached_tokens",
        "num_lmcache_cached_tokens",
        "num_lmcache_extra_cached_tokens",
    )
    if not all(isinstance(stats.get(key), int) and stats[key] >= 0 for key in required):
        return None
    return {key: int(stats[key]) for key in required}


def response_summary(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "content": "",
            "reasoning_content": "",
            "visible_output_sha256": None,
            "full_output_sha256": None,
            "visible_output_utf8_bytes": None,
            "reasoning_output_utf8_bytes": None,
            "finish_reason": None,
            "response_protocol_complete": False,
            "generation_budget_reached": False,
            "budget_limited_empty_final": False,
            "visible_final_complete": False,
            "visible_final_status": "protocol_response_unavailable",
            "cache_stats": None,
        }
    usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    raw_content = message.get("content")
    raw_reasoning = message.get("reasoning_content")
    if raw_reasoning is None:
        raw_reasoning = message.get("reasoning")
    content = raw_content if isinstance(raw_content, str) else ""
    reasoning = raw_reasoning if isinstance(raw_reasoning, str) else ""
    finish_reason = choice.get("finish_reason")
    completion_tokens = usage.get("completion_tokens")
    response_protocol_complete = bool(
        choice
        and message
        and "content" in message
        and isinstance(finish_reason, str)
        and isinstance(usage.get("completion_tokens"), int)
    )
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    max_tokens = request.get("max_tokens")
    generation_budget_reached = bool(
        finish_reason == "length"
        or (
            isinstance(completion_tokens, int)
            and isinstance(max_tokens, int)
            and max_tokens > 0
            and completion_tokens >= max_tokens
            and finish_reason != "stop"
        )
    )
    visible_nonempty = bool(content.strip())
    visible_final_complete = visible_nonempty and finish_reason == "stop"
    budget_limited_empty_final = generation_budget_reached and not visible_nonempty
    if budget_limited_empty_final:
        visible_final_status = "budget_limited_empty_final"
    elif generation_budget_reached:
        visible_final_status = "budget_limited_partial_visible_final"
    elif visible_final_complete:
        visible_final_status = "complete_visible_final"
    elif visible_nonempty:
        visible_final_status = "incomplete_visible_final"
    else:
        visible_final_status = "empty_visible_final"
    full_output_bytes = json.dumps(
        {"content": content, "reasoning_content": reasoning},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "content": content,
        "reasoning_content": reasoning,
        "visible_output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "full_output_sha256": hashlib.sha256(full_output_bytes).hexdigest(),
        "visible_output_utf8_bytes": len(content.encode()),
        "reasoning_output_utf8_bytes": len(reasoning.encode()),
        "finish_reason": finish_reason,
        "response_protocol_complete": response_protocol_complete,
        "generation_budget_reached": generation_budget_reached,
        "budget_limited_empty_final": budget_limited_empty_final,
        "visible_final_complete": visible_final_complete,
        "visible_final_status": visible_final_status,
        "cache_stats": cache_stats(record),
    }


def semantic_answer_evidence(
    record: dict[str, Any], expected: str
) -> dict[str, Any]:
    """Classify a semantic answer using the user-visible final channel only."""
    summary = record.get("summary")
    if not isinstance(summary, dict):
        summary = response_summary(record)
    visible = str(summary.get("content") or "")
    reasoning = str(summary.get("reasoning_content") or "")
    visible_contains = expected in visible
    visible_exact = visible.strip() == expected
    reasoning_contains = expected in reasoning
    if record.get("status") is None:
        outcome = "client_transport_or_harness_failure"
    elif record.get("ok") is not True:
        outcome = "api_protocol_or_response_failure"
    elif summary.get("response_protocol_complete") is not True:
        outcome = "api_protocol_or_response_failure"
    elif summary.get("generation_budget_reached") is True:
        outcome = (
            "budget_limited_empty_final"
            if summary.get("budget_limited_empty_final") is True
            else "budget_limited_partial_visible_final"
        )
    elif (
        summary.get("visible_final_complete") is not True
        and reasoning_contains
        and not visible_contains
    ):
        outcome = "reasoning_only_match_not_retrieval"
    elif summary.get("visible_final_complete") is not True:
        outcome = "incomplete_visible_final"
    elif visible_exact:
        outcome = "correct_visible_answer"
    elif reasoning_contains and not visible_contains:
        outcome = "reasoning_only_match_not_retrieval"
    else:
        outcome = "model_wrong_visible_answer"
    return {
        "expected": expected,
        "visible_contains_expected": visible_contains,
        "visible_exact_match": visible_exact,
        "reasoning_contains_expected_diagnostic": reasoning_contains,
        "response_protocol_complete": summary.get("response_protocol_complete") is True,
        "visible_final_complete": summary.get("visible_final_complete") is True,
        "generation_budget_reached": summary.get("generation_budget_reached") is True,
        "budget_limited_empty_final": summary.get("budget_limited_empty_final") is True,
        "accepted_visible_recall": bool(
            record.get("ok") is True
            and summary.get("response_protocol_complete") is True
            and summary.get("visible_final_complete") is True
            and visible_exact
        ),
        "outcome": outcome,
    }


def chat_request(
    port: int,
    model: str,
    content: str,
    *,
    cache_salt: str,
    label: str,
    max_tokens: int,
    timeout: int,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "cache_salt": cache_salt,
        "kv_transfer_params": {"cached_token_stats": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    record = http_json(
        port,
        CHAT_PATH,
        payload=payload,
        timeout=timeout,
        label=label,
    )
    record["summary"] = response_summary(record)
    record["prompt_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    record["prompt_utf8_bytes"] = len(content.encode())
    record["method"] = {
        "schema": PROBE_SCHEMA,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "max_tokens": max_tokens,
        "output_identity": "visible final content UTF-8 bytes",
        "full_channel_hash_scope": "diagnostic content+reasoning JSON only",
    }
    return record


def reset_local_prefix_cache(port: int, label: str) -> dict[str, Any]:
    return http_json(
        port,
        "/reset_prefix_cache",
        payload=None,
        timeout=120,
        label=label,
    )


def reset_succeeded(record: dict[str, Any]) -> bool:
    parsed = record.get("parsed")
    return bool(
        record.get("ok")
        and isinstance(parsed, dict)
        and parsed.get("success") is True
    )


def command_prepare(args: argparse.Namespace) -> int:
    content, metadata = calibrate_prompt(
        args.port, args.model, args.target_tokens, args.identity, args.kind
    )
    args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_out.write_text(content)
    metadata["prompt_path"] = str(args.prompt_out)
    save_json(args.out, metadata)
    print(json.dumps(metadata, indent=2))
    return 0


def command_request(args: argparse.Namespace) -> int:
    if (
        args.expected_reference is not None
        and args.max_tokens < SEMANTIC_MAX_TOKENS
    ):
        raise ValueError(
            "reference probes require at least "
            f"{SEMANTIC_MAX_TOKENS} output tokens; no automatic budget retry is used"
        )
    content = args.prompt.read_text()
    record = chat_request(
        args.port,
        args.model,
        content,
        cache_salt=args.cache_salt,
        label=args.label,
        max_tokens=args.max_tokens,
        timeout=args.deadline,
        ignore_eos=args.ignore_eos,
    )
    record["schema"] = PROBE_SCHEMA
    record["expected_prompt_tokens"] = args.expected_tokens
    record["expected_reference"] = args.expected_reference
    summary = record["summary"]
    checks = {
        "http_success": bool(record["ok"]),
        "response_protocol_complete": summary.get("response_protocol_complete")
        is True,
        "prompt_tokens": args.expected_tokens is None
        or summary["prompt_tokens"] == args.expected_tokens,
    }
    if args.expected_reference is not None:
        answer = semantic_answer_evidence(record, args.expected_reference)
        record["answer_evidence"] = answer
        checks.update(
            {
                "reference_present": answer["visible_contains_expected"],
                "reference_exact_visible_answer": answer["visible_exact_match"],
                "complete_visible_answer": answer["visible_final_complete"],
                "visible_recall_accepted": answer["accepted_visible_recall"],
            }
        )
    record["checks"] = checks
    record["passed"] = all(checks.values())
    save_json(args.out, record)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "label": args.label,
                "passed": record["passed"],
                "wall_seconds": record["wall_seconds"],
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0 if record["passed"] else 1


def command_reset(args: argparse.Namespace) -> int:
    result = reset_local_prefix_cache(args.port, args.label)
    result["passed"] = reset_succeeded(result)
    save_json(args.out, result)
    print(json.dumps({"out": str(args.out), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 1


def calibrate_needle_prompt(
    port: int,
    model: str,
    target_tokens: int,
    identity: str,
    depth_percent: int,
    code: str,
) -> tuple[str, dict[str, Any]]:
    preamble = unique_preamble(identity)
    fact = f"\nThe archive reference for specimen Sigma-{depth_percent} is {code}.\n"
    question = (
        f"\nWhat is the archive reference for specimen Sigma-{depth_percent}? "
        "Reply with only the reference."
    )
    preamble_tokenization = tokenize(
        port, model, preamble, include_token_ids=True
    )
    preamble_tokens = preamble_tokenization["count"]
    first_block_token_ids = preamble_tokenization["token_ids"][:4096]
    if len(first_block_token_ids) != 4096:
        raise RuntimeError(
            "needle preamble did not cover the first 4096-token cache chunk"
        )
    first_block_token_ids_sha256 = hashlib.sha256(
        json.dumps(first_block_token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    fixed = tokenize(port, model, preamble + fact + question)["count"]
    filler_total = target_tokens - fixed
    if filler_total <= 0:
        raise RuntimeError("needle target is too small for its fixed content")
    before_count = int(filler_total * depth_percent / 100)
    after_count = filler_total - before_count
    attempts = []
    for _ in range(8):
        content = preamble + (" a" * before_count) + fact + (" a" * after_count) + question
        observed = tokenize(port, model, content)["count"]
        attempts.append(
            {
                "before_filler": before_count,
                "after_filler": after_count,
                "observed_tokens": observed,
            }
        )
        if observed == target_tokens:
            break
        after_count += target_tokens - observed
        if after_count < 0:
            raise RuntimeError("needle calibration produced negative trailing filler")
    else:
        raise RuntimeError(f"could not calibrate needle prompt; observed {observed}")
    fact_prefix_tokens = tokenize(
        port, model, preamble + (" a" * before_count) + fact
    )["count"]
    return content, {
        "schema": PROBE_SCHEMA,
        "identity": identity,
        "depth_percent_requested": depth_percent,
        "depth_percent_observed": 100.0 * fact_prefix_tokens / target_tokens,
        "target_tokens": target_tokens,
        "observed_tokens": observed,
        "preamble_tokens": preamble_tokens,
        "preamble_rows": UNIQUE_PREAMBLE_ROWS,
        "unique_first_block_token_span": len(first_block_token_ids),
        "first_block_token_ids_sha256": first_block_token_ids_sha256,
        "preamble_sha256": hashlib.sha256(preamble.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "needle_code": code,
        "calibration_attempts": attempts,
        "tokenizer_method": {
            "endpoint": TOKENIZE_PATH,
            "messages_api": True,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "exact_server_token_count_required": True,
            "exact_first_block_token_ids_required": True,
        },
    }


def command_needles(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "target_tokens": args.target_tokens,
        "suite_identity": args.suite_identity,
        "depths": list(args.depths),
        "method": {
            "oracle": "exact code in a naturally completed user-visible final answer",
            "reasoning_channel_matches_are_diagnostic_only": True,
            "max_tokens": NEEDLE_MAX_TOKENS,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "cold_requests_only": True,
        },
        "results": [],
    }
    save_json(args.out, report)
    for depth in args.depths:
        identity = f"{args.suite_identity}-depth-{depth}"
        code = "NEEDLE-" + hashlib.sha256(identity.encode()).hexdigest()[:18].upper()
        try:
            content, metadata = calibrate_needle_prompt(
                args.port,
                args.model,
                args.target_tokens,
                identity,
                depth,
                code,
            )
            request = chat_request(
                args.port,
                args.model,
                content,
                cache_salt=f"{args.suite_identity}-needle-{depth}",
                label=f"needle-depth-{depth}",
                max_tokens=NEEDLE_MAX_TOKENS,
                timeout=args.deadline,
            )
            answer = semantic_answer_evidence(request, code)
            row = {
                **metadata,
                "request": request,
                "answer_evidence": answer,
                "visible_recall_hit": answer["accepted_visible_recall"],
                "reasoning_only_match_diagnostic": bool(
                    answer["reasoning_contains_expected_diagnostic"]
                    and not answer["visible_contains_expected"]
                ),
                "cold_external_miss": (
                    request["summary"]["cache_stats"] is not None
                    and request["summary"]["cache_stats"]["num_lmcache_cached_tokens"] == 0
                ),
            }
        except Exception as error:
            row = {
                "identity": identity,
                "depth_percent_requested": depth,
                "needle_code": code,
                "visible_recall_hit": False,
                "reasoning_only_match_diagnostic": False,
                "cold_external_miss": False,
                "outcome": "harness_failure",
                "error": f"{type(error).__name__}: {error}",
            }
        report["results"].append(row)
        save_json(args.out, report)
        print(json.dumps({k: v for k, v in row.items() if k != "request"}), flush=True)
    report["checks"] = {
        "all_depths_attempted": len(report["results"]) == len(args.depths),
        "all_exact_target": all(
            row.get("observed_tokens") == args.target_tokens for row in report["results"]
        ),
        "all_first_blocks_unique": all(
            int(row.get("preamble_tokens", 0)) >= 4096
            and row.get("unique_first_block_token_span") == 4096
            for row in report["results"]
        )
        and len(
            {
                row.get("first_block_token_ids_sha256")
                for row in report["results"]
                if row.get("first_block_token_ids_sha256")
            }
        )
        == len(args.depths),
        "all_cold_misses": all(row.get("cold_external_miss") is True for row in report["results"]),
        "all_needles_visible_recall": all(
            row.get("visible_recall_hit") is True for row in report["results"]
        ),
        "all_visible_finals_complete": all(
            row.get("answer_evidence", {}).get("visible_final_complete") is True
            for row in report["results"]
        ),
    }
    report["passed"] = all(report["checks"].values())
    save_json(args.out, report)
    return 0 if report["passed"] else 1


def make_document(seed: int, n_words: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def document_request(
    port: int,
    model: str,
    suite_identity: str,
    *,
    seed: int,
    n_words: int,
    word_index: int,
    label: str,
    salt: str,
    timeout: int,
) -> dict[str, Any]:
    identity = f"{suite_identity}-document-{seed}"
    words = make_document(seed, n_words).split()
    # The marker is inserted at a random offset only to distribute the fact
    # through each document.  The question names the marker directly; it does
    # not ask the model to count random words or infer an ordinal position.
    marker_digest = hashlib.sha256(
        f"marker:{identity}:{word_index}".encode()
    ).hexdigest()
    value_digest = hashlib.sha256(
        f"value:{identity}:{word_index}".encode()
    ).hexdigest()
    marker = f"MARKER-{marker_digest[:16].upper()}"
    answer = f"VALUE-{value_digest[:16].upper()}"
    words.insert(word_index, f"FACT[{marker}]={answer}.")
    document = " ".join(words)
    prompt = (
        unique_preamble(identity)
        + "Document:\n"
        + document
        + f"\n\nQuestion: What exact value is assigned to {marker}? "
        + "Reply with that value only."
    )
    request = chat_request(
        port,
        model,
        prompt,
        cache_salt=salt,
        label=label,
        max_tokens=SEMANTIC_MAX_TOKENS,
        timeout=timeout,
    )
    summary = request["summary"]
    evidence = semantic_answer_evidence(request, answer)
    stats = summary.get("cache_stats") or {}
    return {
        "seed": seed,
        "marker_insertion_word_offset": word_index,
        "marker": marker,
        "expected_value": answer,
        "answer_evidence": evidence,
        "visible_ground_truth_match": evidence["accepted_visible_recall"],
        "visible_final": summary["content"],
        "visible_final_sha256": summary["visible_output_sha256"],
        "full_output_sha256_diagnostic": summary["full_output_sha256"],
        "finish_reason": summary.get("finish_reason"),
        "visible_final_complete": summary.get("visible_final_complete"),
        "generation_budget_reached": summary.get("generation_budget_reached"),
        "external_hit_tokens": stats.get("num_lmcache_cached_tokens"),
        "first_block_identity": identity,
        "first_block_sha256": hashlib.sha256(unique_preamble(identity).encode()).hexdigest(),
        "request": request,
    }


def command_eviction(args: argparse.Namespace) -> int:
    rng = random.Random(20260905)
    base_seed = int(hashlib.sha256(args.suite_identity.encode()).hexdigest()[:8], 16)
    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "suite_identity": args.suite_identity,
        "docs": args.docs,
        "words_per_doc": args.words,
        "method": {
            "oracle": "direct retrieval of a unique labelled marker value",
            "marker_offset_is_not_queried": True,
            "ground_truth_channel": "user-visible final only",
            "natural_completion_required": True,
            "exact_visible_byte_identity_checked_separately": True,
            "full_content_plus_reasoning_hash_is_diagnostic_only": True,
            "max_tokens": SEMANTIC_MAX_TOKENS,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "churn_answer_quality_required": False,
            "interpretation": (
                "A wrong visible reply is a model retrieval failure, not by itself proof "
                "of corrupted KV memory."
            ),
        },
        "pass1": [],
        "churn": [],
        "pass2": [],
        "failures": [],
    }
    save_json(args.out, report)
    selections: list[int] = [rng.randrange(1, args.words - 1) for _ in range(args.docs)]
    for index in range(args.docs):
        try:
            row = document_request(
                args.port,
                args.model,
                args.suite_identity,
                seed=base_seed + index,
                n_words=args.words,
                word_index=selections[index],
                label=f"eviction-pass1-{index:02d}",
                salt=f"{args.suite_identity}-doc-{index:02d}",
                timeout=args.deadline,
            )
            row["index"] = index
            report["pass1"].append(row)
        except Exception as error:
            report["failures"].append(
                {"stage": "pass1", "index": index, "error": f"{type(error).__name__}: {error}"}
            )
        save_json(args.out, report)
    time.sleep(args.store_drain_seconds)
    for index in reversed(range(args.docs)):
        churn_seed = base_seed + 100_000 + index
        try:
            churn = document_request(
                args.port,
                args.model,
                args.suite_identity + "-churn",
                seed=churn_seed,
                n_words=args.words,
                word_index=5,
                label=f"eviction-churn-{index:02d}",
                salt=f"{args.suite_identity}-churn-{index:02d}",
                timeout=args.deadline,
            )
            churn["index"] = index
            report["churn"].append(churn)
        except Exception as error:
            report["failures"].append(
                {"stage": "churn", "index": index, "error": f"{type(error).__name__}: {error}"}
            )
        try:
            replay = document_request(
                args.port,
                args.model,
                args.suite_identity,
                seed=base_seed + index,
                n_words=args.words,
                word_index=selections[index],
                label=f"eviction-pass2-{index:02d}",
                salt=f"{args.suite_identity}-doc-{index:02d}",
                timeout=args.deadline,
            )
            replay["index"] = index
            report["pass2"].append(replay)
        except Exception as error:
            report["failures"].append(
                {"stage": "pass2", "index": index, "error": f"{type(error).__name__}: {error}"}
            )
        save_json(args.out, report)
        print(
            json.dumps(
                {
                    "completed_pass1": len(report["pass1"]),
                    "completed_churn": len(report["churn"]),
                    "completed_pass2": len(report["pass2"]),
                    "failures": len(report["failures"]),
                }
            ),
            flush=True,
        )
    first = {row["index"]: row for row in report["pass1"]}
    second = {row["index"]: row for row in report["pass2"]}
    comparisons = []
    for index in range(args.docs):
        first_row = first.get(index)
        second_row = second.get(index)
        both_completed = first_row is not None and second_row is not None
        visible_byte_identical = bool(
            both_completed
            and first_row["visible_final"].encode()
            == second_row["visible_final"].encode()
        )
        full_byte_identical = bool(
            both_completed
            and first_row.get("full_output_sha256_diagnostic") is not None
            and first_row.get("full_output_sha256_diagnostic")
            == second_row.get("full_output_sha256_diagnostic")
        )
        first_correct = bool(first_row and first_row["visible_ground_truth_match"])
        second_correct = bool(second_row and second_row["visible_ground_truth_match"])
        first_natural = bool(first_row and first_row["visible_final_complete"])
        second_natural = bool(second_row and second_row["visible_final_complete"])
        budget_limited = bool(
            (first_row and first_row["generation_budget_reached"])
            or (second_row and second_row["generation_budget_reached"])
        )
        if not both_completed:
            outcome = "harness_request_incomplete"
        elif budget_limited:
            outcome = "budget_limited_incomplete_answer"
        elif not first_natural or not second_natural:
            outcome = "incomplete_visible_answer"
        elif first_correct and second_correct and visible_byte_identical:
            outcome = "stable_correct_visible_answer"
        elif first_correct and second_correct:
            outcome = "correct_visible_answers_with_byte_variation"
        elif first_correct and not second_correct:
            outcome = "replay_wrong_after_correct_control_not_kv_corruption_proof"
        elif not first_correct and second_correct:
            outcome = "cold_model_failure_then_correct_replay"
        else:
            outcome = "model_retrieval_failure_both_passes"
        comparisons.append(
            {
                "index": index,
                "both_completed": both_completed,
                "exact_visible_byte_identity": visible_byte_identical,
                "full_channel_byte_identity_diagnostic": full_byte_identical,
                "ground_truth_pass1_visible": first_correct,
                "ground_truth_pass2_visible": second_correct,
                "pass1_visible_final_complete": first_natural,
                "pass2_visible_final_complete": second_natural,
                "budget_limited": budget_limited,
                "outcome": outcome,
                "pass1_finish_reason": first_row.get("finish_reason") if first_row else None,
                "pass2_finish_reason": second_row.get("finish_reason") if second_row else None,
                "pass2_external_hit_tokens": (
                    second_row.get("external_hit_tokens") if second_row else None
                ),
                "pass1_visible_sha256": (
                    first_row.get("visible_final_sha256") if first_row else None
                ),
                "pass2_visible_sha256": (
                    second_row.get("visible_final_sha256") if second_row else None
                ),
                "pass1_full_sha256_diagnostic": (
                    first_row.get("full_output_sha256_diagnostic") if first_row else None
                ),
                "pass2_full_sha256_diagnostic": (
                    second_row.get("full_output_sha256_diagnostic") if second_row else None
                ),
            }
        )
    report["comparisons"] = comparisons
    all_requests = [*report["pass1"], *report["churn"], *report["pass2"]]
    report["outcome_counts"] = {
        outcome: sum(row["outcome"] == outcome for row in comparisons)
        for outcome in sorted({row["outcome"] for row in comparisons})
    }
    report["checks"] = {
        "all_pass1_completed": len(report["pass1"]) == args.docs,
        "all_churn_completed": len(report["churn"]) == args.docs,
        "all_pass2_completed": len(report["pass2"]) == args.docs,
        "all_http_requests_succeeded": all(
            row.get("request", {}).get("ok") is True for row in all_requests
        ),
        "all_response_protocols_complete": all(
            row.get("request", {})
            .get("summary", {})
            .get("response_protocol_complete")
            is True
            for row in all_requests
        ),
        "all_target_replies_finished_naturally": all(
            row["pass1_visible_final_complete"]
            and row["pass2_visible_final_complete"]
            for row in comparisons
        ),
        "no_target_reply_was_budget_limited": all(
            not row["budget_limited"] for row in comparisons
        ),
        "all_target_visible_outputs_byte_identical": all(
            row["exact_visible_byte_identity"] for row in comparisons
        ),
        "all_target_visible_ground_truth_correct": all(
            row["ground_truth_pass1_visible"]
            and row["ground_truth_pass2_visible"]
            for row in comparisons
        ),
        "no_request_failures": not report["failures"],
    }
    report["passed"] = all(report["checks"].values())
    save_json(args.out, report)
    return 0 if report["passed"] else 1


def command_prefill(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "label": args.label,
        "suite_identity": args.suite_identity,
        "target_tokens": args.target_tokens,
        "runs_requested": args.runs,
        "method": {
            "scope": "one-token forced-generation prefill latency only",
            "recall_or_answer_quality_evaluated": False,
            "max_tokens": 1,
            "ignore_eos": True,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        },
        "runs": [],
        "failures": [],
    }
    save_json(args.out, report)
    for index in range(args.runs):
        identity = f"{args.suite_identity}-{args.label}-run-{index:02d}"
        try:
            content, prompt_meta = calibrate_prompt(
                args.port,
                args.model,
                args.target_tokens,
                identity,
                "period",
            )
            request = chat_request(
                args.port,
                args.model,
                content,
                cache_salt=f"{identity}-salt",
                label=f"prefill-{args.label}-{index:02d}",
                max_tokens=1,
                timeout=args.deadline,
                ignore_eos=True,
            )
            summary = request["summary"]
            wall = float(request["wall_seconds"])
            row = {
                "run": index + 1,
                "prompt": prompt_meta,
                "request": request,
                "tokens_per_second": args.target_tokens / wall if wall > 0 else None,
                "cold_external_miss": summary["cache_stats"] is not None
                and summary["cache_stats"]["num_lmcache_cached_tokens"] == 0,
            }
            report["runs"].append(row)
        except Exception as error:
            report["failures"].append(
                {"run": index + 1, "error": f"{type(error).__name__}: {error}"}
            )
        save_json(args.out, report)
        print(
            json.dumps(
                {
                    "run": index + 1,
                    "completed": len(report["runs"]),
                    "failures": len(report["failures"]),
                }
            ),
            flush=True,
        )
    speeds = [float(row["tokens_per_second"]) for row in report["runs"] if row["tokens_per_second"]]
    report["summary"] = {
        "median_tokens_per_second": statistics.median(speeds) if speeds else None,
        "steady_median_tokens_per_second": statistics.median(speeds[1:])
        if len(speeds) > 1
        else None,
        "aggregate_tokens_per_second": (
            sum(args.target_tokens for _ in report["runs"])
            / sum(float(row["request"]["wall_seconds"]) for row in report["runs"])
            if report["runs"]
            else None
        ),
        "first_request_seconds": report["runs"][0]["request"]["wall_seconds"]
        if report["runs"]
        else None,
    }
    report["checks"] = {
        "all_runs_completed": len(report["runs"]) == args.runs,
        "all_http_requests_succeeded": all(
            row["request"].get("ok") is True for row in report["runs"]
        ),
        "all_response_protocols_complete": all(
            row["request"]["summary"].get("response_protocol_complete") is True
            for row in report["runs"]
        ),
        "all_exact_tokens": all(
            row["request"]["summary"]["prompt_tokens"] == args.target_tokens
            for row in report["runs"]
        ),
        "all_first_blocks_unique": len(
            {
                row["prompt"]["first_block_token_ids_sha256"]
                for row in report["runs"]
            }
        )
        == args.runs
        and all(
            row["prompt"]["preamble_tokens"] >= 4096
            and row["prompt"]["unique_first_block_token_span"] == 4096
            for row in report["runs"]
        ),
        "all_external_cold_misses": all(row["cold_external_miss"] for row in report["runs"]),
        "no_failures": not report["failures"],
    }
    report["passed"] = all(report["checks"].values())
    save_json(args.out, report)
    return 0 if report["passed"] else 1


class EventCollector:
    def __init__(self, endpoint: str, replay_endpoint: str, topic: str) -> None:
        self.available = False
        self.error: str | None = None
        self.live: list[dict[str, Any]] = []
        self.replayed: list[dict[str, Any]] = []
        self._zmq: Any = None
        self._msgpack: Any = None
        self._context: Any = None
        self._sub: Any = None
        self._replay: Any = None
        try:
            import msgpack
            import zmq

            self._zmq = zmq
            self._msgpack = msgpack
            self._context = zmq.Context()
            self._sub = self._context.socket(zmq.SUB)
            self._sub.setsockopt(zmq.RCVHWM, 100_000)
            self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
            self._sub.connect(endpoint)
            self._replay = self._context.socket(zmq.DEALER)
            self._replay.connect(replay_endpoint)
            self.available = True
            time.sleep(0.5)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def _decode(self, payload: bytes) -> object:
        assert self._msgpack is not None
        return json_safe(
            self._msgpack.unpackb(payload, raw=False, strict_map_key=False)
        )

    def drain(self, stage: str, max_seconds: float = 4.0, idle_seconds: float = 0.5) -> None:
        if not self.available:
            return
        deadline = time.monotonic() + max_seconds
        idle_deadline = time.monotonic() + idle_seconds
        while time.monotonic() < deadline and time.monotonic() < idle_deadline:
            assert self._sub is not None
            if not self._sub.poll(100):
                continue
            frames = self._sub.recv_multipart()
            if len(frames) != 3:
                self.live.append(
                    {"stage": stage, "decode_error": f"expected 3 frames, got {len(frames)}"}
                )
                continue
            topic, seq_bytes, payload = frames
            try:
                row = {
                    "stage": stage,
                    "topic": topic.decode(errors="replace"),
                    "seq": int.from_bytes(seq_bytes, "big"),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "batch": self._decode(payload),
                }
            except Exception as error:
                row = {
                    "stage": stage,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "decode_error": f"{type(error).__name__}: {error}",
                }
            self.live.append(row)
            idle_deadline = time.monotonic() + idle_seconds

    def request_replay(self, start_seq: int = 0, timeout_seconds: float = 10.0) -> None:
        if not self.available:
            return
        assert self._replay is not None
        # ROUTER sends a stream of replies. REQ accepts only the first reply;
        # DEALER preserves the multipart envelope and receives the full stream.
        self._replay.send_multipart([b"", start_seq.to_bytes(8, "big")])
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not self._replay.poll(250):
                continue
            frames = self._replay.recv_multipart()
            if len(frames) != 4 or frames[0] != b"":
                self.error = f"replay expected delimiter plus 3 frames, got {len(frames)}"
                return
            _, topic, seq_bytes, payload = frames
            if seq_bytes == (-1).to_bytes(8, "big", signed=True):
                if payload:
                    self.error = "replay end marker carried an unexpected payload"
                return
            try:
                self.replayed.append(
                    {
                        "topic": topic.decode(errors="replace"),
                        "seq": int.from_bytes(seq_bytes, "big"),
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "batch": self._decode(payload),
                    }
                )
            except Exception as error:
                self.replayed.append(
                    {
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "decode_error": f"{type(error).__name__}: {error}",
                    }
                )
        self.error = f"replay did not terminate within {timeout_seconds}s"

    def close(self) -> None:
        for socket in (self._sub, self._replay):
            if socket is not None:
                socket.close(linger=0)
        if self._context is not None:
            self._context.term()


def batch_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    batch = row.get("batch")
    if isinstance(batch, list) and len(batch) >= 2 and isinstance(batch[1], list):
        events = batch[1]
    elif isinstance(batch, dict) and isinstance(batch.get("events"), list):
        events = batch["events"]
    else:
        return []
    return [event for event in events if isinstance(event, dict)]


def event_kind(event: dict[str, Any]) -> str | None:
    for key in ("type", "kind", "tag"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return None


def event_evidence(collector: EventCollector) -> dict[str, Any]:
    live_events = [event for row in collector.live for event in batch_events(row)]
    replay_events = [event for row in collector.replayed for event in batch_events(row)]
    stored = [event for event in replay_events if event_kind(event) == "BlockStored"]
    if not stored:
        stored = [event for event in live_events if event_kind(event) == "BlockStored"]
    stage_event_counts: dict[str, int] = {}
    stage_stored_counts: dict[str, int] = {}
    for row in collector.live:
        stage = str(row.get("stage") or "unknown")
        events = batch_events(row)
        stage_event_counts[stage] = stage_event_counts.get(stage, 0) + len(events)
        stage_stored_counts[stage] = stage_stored_counts.get(stage, 0) + sum(
            event_kind(event) == "BlockStored" for event in events
        )
    cached_prefix_replay_published = (
        stage_stored_counts.get("serial-warm", 0) > 0
    )
    shaped = [
        event
        for event in stored
        if isinstance(event.get("block_size"), int) and int(event["block_size"]) > 0
    ]
    truthful_rows = []
    for event in shaped:
        hashes = event.get("block_hashes")
        tokens = event.get("token_ids")
        block_size = int(event["block_size"])
        hashes_count = len(hashes) if isinstance(hashes, list) else 0
        tokens_count = len(tokens) if isinstance(tokens, list) else 0
        truthful_rows.append(
            {
                "hashes": hashes_count,
                "tokens": tokens_count,
                "block_size": block_size,
                "group_idx": event.get("group_idx"),
                "kind": event.get("kv_cache_spec_kind"),
                "has_skipped_context_fields": any(
                    key in event
                    for key in (
                        "skipped_parent_block_hash",
                        "skipped_token_ids",
                        "skipped_extra_keys",
                    )
                ),
                "nonempty_exact_run": hashes_count > 0
                and tokens_count == hashes_count * block_size,
            }
        )
    replay_seqs = [row.get("seq") for row in collector.replayed if isinstance(row.get("seq"), int)]
    replay_contiguous = bool(replay_seqs) and replay_seqs == list(
        range(replay_seqs[0], replay_seqs[0] + len(replay_seqs))
    )
    live_by_seq = {
        row["seq"]: row.get("payload_sha256")
        for row in collector.live
        if isinstance(row.get("seq"), int)
    }
    replay_by_seq = {
        row["seq"]: row.get("payload_sha256")
        for row in collector.replayed
        if isinstance(row.get("seq"), int)
    }
    overlap = set(live_by_seq) & set(replay_by_seq)
    return {
        "collector_available": collector.available,
        "collector_error": collector.error,
        "live_batches": len(collector.live),
        "replay_batches": len(collector.replayed),
        "live_events": len(live_events),
        "replay_events": len(replay_events),
        "block_stored_events": len(stored),
        "shaped_block_stored_events": len(shaped),
        "live_stage_event_counts": stage_event_counts,
        "live_stage_block_stored_counts": stage_stored_counts,
        "cached_prefix_replay_published": cached_prefix_replay_published,
        "skipped_context_observed": any(
            row["has_skipped_context_fields"] for row in truthful_rows
        ),
        "truthful_rows": truthful_rows,
        "replay_sequences": replay_seqs,
        "replay_contiguous": replay_contiguous,
        "live_replay_overlap": len(overlap),
        "live_replay_overlap_identical": bool(overlap)
        and all(live_by_seq[seq] == replay_by_seq[seq] for seq in overlap),
        "truthful_sparse_event_shape": bool(truthful_rows)
        and all(
            row["nonempty_exact_run"] and row["group_idx"] is not None
            for row in truthful_rows
        ),
    }


def stats_hit(record: dict[str, Any]) -> int | None:
    stats = record.get("summary", {}).get("cache_stats")
    if not isinstance(stats, dict):
        return None
    value = stats.get("num_lmcache_cached_tokens")
    return int(value) if isinstance(value, int) else None


def command_hitmiss(args: argparse.Namespace) -> int:
    collector = EventCollector(args.event_endpoint, args.replay_endpoint, args.event_topic)
    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "suite_identity": args.suite_identity,
        "target_tokens": args.target_tokens,
        "event_endpoint": args.event_endpoint,
        "replay_endpoint": args.replay_endpoint,
        "event_topic": args.event_topic,
        "method": {
            "semantic_max_tokens": SEMANTIC_MAX_TOKENS,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "semantic_ground_truth_channel": "user-visible final only",
            "visible_output_identity": "exact visible final UTF-8 bytes",
            "full_output_identity": "diagnostic content+reasoning JSON only",
            "period_churn_scope": "one-token latency/cache-pressure request; no recall or quality claim",
        },
        "scenarios": {},
        "failures": [],
    }
    save_json(args.out, report)

    def stage_request(
        stage: str,
        content: str,
        salt: str,
        *,
        max_tokens: int = SEMANTIC_MAX_TOKENS,
        expected: str | None = None,
    ) -> dict[str, Any]:
        request = chat_request(
            args.port,
            args.model,
            content,
            cache_salt=salt,
            label=f"{args.suite_identity}-{stage}",
            max_tokens=max_tokens,
            timeout=args.deadline,
        )
        if expected is not None:
            request["semantic_answer"] = semantic_answer_evidence(request, expected)
        time.sleep(args.event_settle_seconds)
        collector.drain(stage)
        return request

    def execute_scenario(
        name: str, operation: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            result = operation()
        except Exception as error:
            failure = {
                "stage": name,
                "error": f"{type(error).__name__}: {error}",
            }
            report["failures"].append(failure)
            result = {"attempted": True, **failure}
        report["scenarios"][name] = result
        save_json(args.out, report)
        return result

    def serial_scenario() -> dict[str, Any]:
        prompt, prompt_meta = calibrate_prompt(
            args.port,
            args.model,
            args.target_tokens,
            f"{args.suite_identity}-serial",
            "reference",
        )
        salt = f"{args.suite_identity}-serial"
        reference = str(prompt_meta["reference_code"])
        cold = stage_request("serial-cold", prompt, salt, expected=reference)
        time.sleep(args.store_drain_seconds)
        reset = reset_local_prefix_cache(args.port, "serial-reset-local")
        warm = stage_request("serial-warm", prompt, salt, expected=reference)
        return {
            "attempted": True,
            "prompt": prompt_meta,
            "cold": cold,
            "reset": reset,
            "warm": warm,
        }

    def divergent_scenario() -> dict[str, Any]:
        shared_base, base_meta = calibrate_prompt(
            args.port,
            args.model,
            args.target_tokens,
            f"{args.suite_identity}-divergent-base",
            "prefix",
        )
        first_prompt = (
            shared_base
            + "\nDivergent continuation value: ALPHA. Reply with exactly ALPHA."
        )
        second_prompt = (
            shared_base
            + "\nDivergent continuation value: BRAVO. Reply with exactly BRAVO."
        )
        salt = f"{args.suite_identity}-divergent"
        first = stage_request(
            "divergent-alpha", first_prompt, salt, expected="ALPHA"
        )
        time.sleep(args.store_drain_seconds)
        reset = reset_local_prefix_cache(args.port, "divergent-reset-local")
        second = stage_request(
            "divergent-bravo", second_prompt, salt, expected="BRAVO"
        )
        return {
            "attempted": True,
            "shared_base": base_meta,
            "shared_base_sha256": hashlib.sha256(shared_base.encode()).hexdigest(),
            "first": first,
            "reset": reset,
            "second": second,
        }

    def concurrent_scenario() -> dict[str, Any]:
        shared_base, base_meta = calibrate_prompt(
            args.port,
            args.model,
            args.target_tokens,
            f"{args.suite_identity}-concurrent-base",
            "prefix",
        )
        salt = f"{args.suite_identity}-concurrent"
        prime = stage_request(
            "concurrent-prime",
            shared_base
            + "\nConcurrent continuation value: PRIME. Reply with exactly PRIME.",
            salt,
            expected="PRIME",
        )
        time.sleep(args.store_drain_seconds)
        reset = reset_local_prefix_cache(args.port, "concurrent-reset-local")
        barrier = __import__("threading").Barrier(2)

        def concurrent_call(index: int) -> dict[str, Any]:
            barrier.wait(timeout=30)
            expected = f"BRANCH-{index}"
            request = chat_request(
                args.port,
                args.model,
                shared_base
                + f"\nConcurrent continuation value: {expected}. "
                + f"Reply with exactly {expected}.",
                cache_salt=salt,
                label=f"{args.suite_identity}-concurrent-{index}",
                max_tokens=SEMANTIC_MAX_TOKENS,
                timeout=args.deadline,
            )
            request["semantic_answer"] = semantic_answer_evidence(request, expected)
            return request

        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                index: executor.submit(concurrent_call, index)
                for index in (1, 2)
            }
            for index, future in futures.items():
                try:
                    rows.append(future.result())
                except Exception as error:
                    failure = {
                        "stage": "concurrent_shared_prefix",
                        "branch": index,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    report["failures"].append(failure)
                    rows.append({"label": f"concurrent-{index}", "ok": False, **failure})
        time.sleep(args.event_settle_seconds)
        collector.drain("concurrent-branches")
        return {
            "attempted": True,
            "shared_base": base_meta,
            "prime": prime,
            "reset": reset,
            "concurrent": rows,
        }

    def eviction_scenario() -> dict[str, Any]:
        target_prompt, target_meta = calibrate_prompt(
            args.port,
            args.model,
            args.target_tokens,
            f"{args.suite_identity}-eviction-target",
            "reference",
        )
        salt = f"{args.suite_identity}-eviction-target"
        reference = str(target_meta["reference_code"])
        first = stage_request(
            "eviction-target-cold", target_prompt, salt, expected=reference
        )
        time.sleep(args.store_drain_seconds)
        warm_reset = reset_local_prefix_cache(args.port, "eviction-target-reset1")
        warm = stage_request(
            "eviction-target-warm", target_prompt, salt, expected=reference
        )
        scenario: dict[str, Any] = {
            "attempted": True,
            "target_prompt": target_meta,
            "first": first,
            "warm_reset": warm_reset,
            "warm": warm,
            "churn": [],
        }
        report["scenarios"]["eviction_then_reuse"] = scenario
        save_json(args.out, report)
        for index in range(args.churn_docs):
            try:
                churn_prompt, churn_meta = calibrate_prompt(
                    args.port,
                    args.model,
                    args.target_tokens,
                    f"{args.suite_identity}-eviction-churn-{index:02d}",
                    "period",
                )
                churn_request = stage_request(
                    f"eviction-churn-{index:02d}",
                    churn_prompt,
                    f"{args.suite_identity}-eviction-churn-{index:02d}",
                    max_tokens=1,
                )
                row = {"index": index, "prompt": churn_meta, "request": churn_request}
            except Exception as error:
                failure = {
                    "stage": "eviction_then_reuse",
                    "churn_index": index,
                    "error": f"{type(error).__name__}: {error}",
                }
                report["failures"].append(failure)
                row = {"index": index, "attempted": True, **failure}
            scenario["churn"].append(row)
            save_json(args.out, report)
        time.sleep(args.eviction_settle_seconds)
        reuse_reset = reset_local_prefix_cache(args.port, "eviction-target-reset2")
        reuse = stage_request(
            "eviction-target-reuse", target_prompt, salt, expected=reference
        )
        scenario.update({"reuse_reset": reuse_reset, "reuse": reuse})
        return scenario

    serial = execute_scenario("serial_shared_prefix", serial_scenario)
    divergent = execute_scenario("divergent_continuation", divergent_scenario)
    concurrent_result = execute_scenario(
        "concurrent_shared_prefix", concurrent_scenario
    )
    eviction = execute_scenario("eviction_then_reuse", eviction_scenario)

    try:
        collector.drain("final")
        collector.request_replay(0)
    except Exception as error:
        report["failures"].append(
            {"stage": "event-collector", "error": f"{type(error).__name__}: {error}"}
        )
    event_report = event_evidence(collector)
    report["events"] = {
        "summary": event_report,
        "live": collector.live,
        "replayed": collector.replayed,
    }
    collector.close()

    serial_cold_hit = stats_hit(serial.get("cold", {}))
    serial_warm_hit = stats_hit(serial.get("warm", {}))
    divergent_hit = stats_hit(divergent.get("second", {}))
    concurrent_hits = [
        stats_hit(row) for row in concurrent_result.get("concurrent", [])
    ]
    eviction_warm_hit = stats_hit(eviction.get("warm", {}))
    eviction_reuse_hit = stats_hit(eviction.get("reuse", {}))
    serial_rows = [serial.get("cold", {}), serial.get("warm", {})]
    divergent_rows = [divergent.get("first", {}), divergent.get("second", {})]
    concurrent_rows = [
        concurrent_result.get("prime", {}),
        *concurrent_result.get("concurrent", []),
    ]
    eviction_rows = [
        eviction.get("first", {}),
        eviction.get("warm", {}),
        eviction.get("reuse", {}),
    ]

    def accepted_visible_answer(row: dict[str, Any]) -> bool:
        return row.get("semantic_answer", {}).get("accepted_visible_recall") is True

    def output_hashes(
        rows: list[dict[str, Any]], field: str
    ) -> list[str | None]:
        return [
            row.get("summary", {}).get(field)
            if isinstance(row.get("summary"), dict)
            else None
            for row in rows
        ]
    def exact_visible_byte_identity(rows: list[dict[str, Any]]) -> bool:
        outputs = [
            row.get("summary", {}).get("content")
            if isinstance(row.get("summary"), dict)
            else None
            for row in rows
        ]
        return bool(
            outputs
            and all(isinstance(output, str) for output in outputs)
            and all(
                output.encode() == outputs[0].encode()
                for output in outputs[1:]
            )
        )


    serial_visible_hashes = output_hashes(serial_rows, "visible_output_sha256")
    serial_full_hashes = output_hashes(serial_rows, "full_output_sha256")
    target_visible_hashes = output_hashes(eviction_rows, "visible_output_sha256")
    target_full_hashes = output_hashes(eviction_rows, "full_output_sha256")
    serial_visible_byte_equal = exact_visible_byte_identity(serial_rows)
    target_visible_byte_equal = exact_visible_byte_identity(eviction_rows)
    report["semantic_answer_evidence"] = {
        "serial": [row.get("semantic_answer") for row in serial_rows],
        "divergent": [row.get("semantic_answer") for row in divergent_rows],
        "concurrent": [row.get("semantic_answer") for row in concurrent_rows],
        "eviction": [row.get("semantic_answer") for row in eviction_rows],
    }
    report["output_identity_diagnostics"] = {
        "visible_final": {
            "scope": "exact user-visible final UTF-8 bytes; required for answer identity",
            "serial_sha256": serial_visible_hashes,
            "serial_byte_equal": serial_visible_byte_equal,
            "eviction_sha256": target_visible_hashes,
            "eviction_byte_equal": target_visible_byte_equal,
        },
        "full_channels": {
            "scope": "content+reasoning canonical JSON; diagnostic only, never a recall oracle",
            "serial_sha256": serial_full_hashes,
            "serial_byte_equal_diagnostic": (
                serial_full_hashes[0] is not None
                and len(set(serial_full_hashes)) == 1
            ),
            "eviction_sha256": target_full_hashes,
            "eviction_byte_equal_diagnostic": (
                target_full_hashes[0] is not None
                and len(set(target_full_hashes)) == 1
            ),
        },
    }
    report["checks"] = {
        "all_scenarios_attempted": set(report["scenarios"])
        == {
            "serial_shared_prefix",
            "divergent_continuation",
            "concurrent_shared_prefix",
            "eviction_then_reuse",
        },
        "all_local_resets_succeeded": all(
            reset_succeeded(record)
            for record in (
                serial.get("reset", {}),
                divergent.get("reset", {}),
                concurrent_result.get("reset", {}),
                eviction.get("warm_reset", {}),
                eviction.get("reuse_reset", {}),
            )
        ),
        "serial_cold_miss": serial_cold_hit == 0,
        "serial_warm_hit": isinstance(serial_warm_hit, int)
        and serial_warm_hit > 0,
        "serial_visible_answers_complete_and_correct": all(
            accepted_visible_answer(row) for row in serial_rows
        ),
        "serial_visible_output_byte_equal": serial_visible_byte_equal,
        "divergent_continuation_partial_hit": isinstance(divergent_hit, int)
        and divergent_hit > 0
        and divergent_hit
        < int(
            divergent.get("second", {})
            .get("summary", {})
            .get("prompt_tokens")
            or 0
        ),
        "divergent_visible_answers_complete_and_correct": all(
            accepted_visible_answer(row) for row in divergent_rows
        ),
        "concurrent_shared_prefix_hits": len(concurrent_hits) == 2
        and all(
            isinstance(value, int) and value > 0 for value in concurrent_hits
        ),
        "concurrent_visible_answers_complete_and_correct": len(concurrent_rows) == 3
        and all(accepted_visible_answer(row) for row in concurrent_rows),
        "eviction_warm_control_hit": isinstance(eviction_warm_hit, int)
        and eviction_warm_hit > 0,
        "eviction_churn_complete": len(eviction.get("churn", []))
        == args.churn_docs,
        "eviction_churn_requests_succeeded": all(
            row.get("request", {}).get("ok") is True
            and row.get("request", {})
            .get("summary", {})
            .get("response_protocol_complete")
            is True
            for row in eviction.get("churn", [])
        ),
        "eviction_then_reuse_miss": eviction_reuse_hit == 0,
        "eviction_visible_answers_complete_and_correct": len(eviction_rows) == 3
        and all(accepted_visible_answer(row) for row in eviction_rows),
        "eviction_recompute_visible_output_byte_equal": target_visible_byte_equal,
        "event_live_and_replay": event_report["live_batches"] > 0
        and event_report["replay_batches"] > 0
        and event_report["replay_contiguous"],
        "event_replay_payload_identity": event_report[
            "live_replay_overlap_identical"
        ],
        "truthful_sparse_event_shape": event_report[
            "truthful_sparse_event_shape"
        ],
        "cached_prefix_replay_event": event_report[
            "cached_prefix_replay_published"
        ],
        "sparse_skipped_context_event": event_report[
            "skipped_context_observed"
        ],
        "no_runner_failures": not report["failures"],
    }
    report["passed"] = all(report["checks"].values())
    report["restart_control"] = {
        "prompt_identity": f"{args.suite_identity}-eviction-target",
        "prompt_kind": "reference",
        "prompt_sha256": eviction.get("target_prompt", {}).get("prompt_sha256"),
        "cache_salt": f"{args.suite_identity}-eviction-target",
        "target_tokens": args.target_tokens,
        "reference_code": reference_code(
            f"{args.suite_identity}-eviction-target"
        ),
        "baseline_visible_output_sha256": eviction.get("first", {})
        .get("summary", {})
        .get("visible_output_sha256"),
        "baseline_full_output_sha256_diagnostic": eviction.get("first", {})
        .get("summary", {})
        .get("full_output_sha256"),
    }
    save_json(args.out, report)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "passed": report["passed"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


def filesystem_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        vfs = os.statvfs(path)
    except OSError as error:
        return {"path": str(path), "error": f"{type(error).__name__}: {error}"}
    major = os.major(stat.st_dev)
    minor = os.minor(stat.st_dev)
    sys_device = Path(f"/sys/dev/block/{major}:{minor}")
    resolved = None
    rotational = None
    model = None
    try:
        resolved_path = sys_device.resolve()
        resolved = str(resolved_path)
        for candidate in (resolved_path, *resolved_path.parents):
            rotational_path = candidate / "queue" / "rotational"
            if rotational_path.exists():
                rotational = rotational_path.read_text().strip() == "1"
                model_path = candidate / "device" / "model"
                if model_path.exists():
                    model = model_path.read_text().strip()
                break
    except OSError:
        pass
    mount = None
    try:
        best = (0, None)
        target = str(path.resolve())
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            if "-" not in fields or len(fields) < 10:
                continue
            mount_point = fields[4].replace("\\040", " ")
            if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
                if len(mount_point) > best[0]:
                    marker = fields.index("-")
                    best = (
                        len(mount_point),
                        {
                            "mount_point": mount_point,
                            "filesystem_type": fields[marker + 1],
                            "source": fields[marker + 2],
                        },
                    )
        mount = best[1]
    except OSError:
        pass
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "st_dev": stat.st_dev,
        "device_major": major,
        "device_minor": minor,
        "sys_device": resolved,
        "rotational": rotational,
        "device_model": model,
        "mount": mount,
        "capacity_bytes": vfs.f_blocks * vfs.f_frsize,
        "available_bytes": vfs.f_bavail * vfs.f_frsize,
        "used_fraction": 1.0 - (vfs.f_bavail / vfs.f_blocks) if vfs.f_blocks else None,
    }


def command_storage(args: argparse.Namespace) -> int:
    paths = [Path(value) for value in args.paths]
    rows = [filesystem_info(path) for path in paths]
    valid = [row for row in rows if "st_dev" in row]
    same_device = len(valid) == len(rows) and len({row["st_dev"] for row in valid}) == 1
    non_rotational = (
        len(valid) == len(rows)
        and all(row.get("rotational") is False for row in valid)
    )
    nvme_named = all(
        "nvme" in str(row.get("sys_device") or "").lower()
        or "nvme" in str((row.get("mount") or {}).get("source") or "").lower()
        or "nvme" in str(row.get("device_model") or "").lower()
        for row in valid
    )
    report = {
        "paths": rows,
        "same_filesystem_device": same_device,
        "non_rotational": non_rotational,
        "nvme_named_in_observed_device_metadata": nvme_named,
        "throttling_performed": False,
        "limitations": [
            "This records the real host filesystem and block-device metadata only; it does not throttle or load the system disk.",
            "The field host is an NVMe observation. It does not reproduce or justify a SATA latency claim.",
            "API wall and connector-reported timer segments are retained separately; neither alone attributes scheduler waiting, compute contention, and storage latency.",
        ],
        "passed": len(valid) == len(rows)
        and same_device
        and non_rotational
        and nvme_named,
    }
    save_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def source_candidate(path: Path, claims: dict[str, str]) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        return {
            "path": str(path),
            "exists": False,
            "error": f"{type(error).__name__}: {error}",
            "claims": {name: False for name in claims},
        }
    text = raw.decode(errors="replace")
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "claims": {
            name: bool(re.search(pattern, text, re.MULTILINE | re.DOTALL))
            for name, pattern in claims.items()
        },
    }


def command_source_audit(args: argparse.Namespace) -> int:
    source = args.source_root
    candidates = [
        source_candidate(
            source / "tests/v1/gpu_connector/test_blocks_first_cs_kv_format.py",
            {
                "packed_fused_shape": r"\[NB, NH, BS, 2 \* HS\]",
                "bit_exact_tensor_equality": r"torch\.equal\(original, recovered\)",
                "cpu_fallback_scope": r"targets the CPU handle-mode path",
            },
        ),
        source_candidate(
            source / "tests/v1/multiprocess/test_cb_plan_executor_gpu.py",
            {
                "gpu_required": r"CUDA is not available",
                "packed_and_split": r"fused-packed.*un-fused split-K/V",
                "bit_exact_reference": r"torch\.equal\(paged_ref\[layer\], paged_new\[layer\]\)",
            },
        ),
        source_candidate(
            source / "tests/v1/multiprocess/test_engine_driven_transfer.py",
            {
                "mocked_mq_scope": r"with_mocked_mq",
                "shm_tensor_equality": r"assert torch\.equal\(",
            },
        ),
        source_candidate(
            source / ".buildkite/k3_tests/multiprocess/scripts/run-dsv4-flash-tp.sh",
            {
                "real_model_external_retrieve": r"LMCache retrieve run",
                "text_comparison_only": r"cmp -s \"\$OUT_A\" \"\$OUT_B\"",
                "explicit_generated_output_scope": r"outputs are identical",
            },
        ),
    ]
    observed = all(
        row.get("exists")
        and all(value is True for value in row.get("claims", {}).values())
        for row in candidates
    )
    report = {
        "source_root": str(source),
        "source_revision_evidence": source_candidate(
            source / ".git/refs/heads/integration/glm53-upstream-consolidation",
            {"commit_sha": r"^[0-9a-f]{40}\s*$"},
        ),
        "candidates": candidates,
        "full_byte_level_external_kv_transfer_test_available": False,
        "invoked": False,
        "supported_shapes_observed": [
            {
                "scope": "CPU fallback handle-mode D2H+H2D",
                "shape": "3 layers x 16 blocks x 4 heads x 128 tokens x fused 2*64 content",
                "equality": "torch.equal",
                "external_storage": False,
            },
            {
                "scope": "GPU CB plan executor, not an external-store round trip",
                "shape": "fused-packed HND and split K/V; bf16; 4 layers; 8 tokens/chunk",
                "equality": "torch.equal against sequential GPU reference",
                "external_storage": False,
            },
            {
                "scope": "Engine-driven SHM protocol with mocked message queue",
                "shape": "small float32 CPU tensors",
                "equality": "torch.equal",
                "external_storage": False,
            },
            {
                "scope": "Real DeepSeek-V4-Flash LMCache L1 retrieve",
                "shape": "sparse-MLA/indexer groups at TP4",
                "equality": "generated text cmp only",
                "external_storage": True,
            },
        ],
        "limitation": (
            "The shipped source evidence inspected here does not combine a real GLM-5.3 GPU KV buffer, "
            "an external L1/L2 store-and-retrieve, and byte/bit comparison of the retrieved KV tensors. "
            "Generated-text equality and answer equality are correctness evidence but are not byte-level KV validation."
        ),
        "passed": observed,
    }
    save_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused R26 cache qualification probes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--port", type=int, required=True)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--target-tokens", type=int, required=True)
    prepare.add_argument("--identity", required=True)
    prepare.add_argument("--kind", choices=("period", "reference", "prefix"), required=True)
    prepare.add_argument("--prompt-out", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.set_defaults(handler=command_prepare)

    request = subparsers.add_parser("request")
    request.add_argument("--port", type=int, required=True)
    request.add_argument("--model", default=DEFAULT_MODEL)
    request.add_argument("--prompt", type=Path, required=True)
    request.add_argument("--cache-salt", required=True)
    request.add_argument("--label", required=True)
    request.add_argument("--max-tokens", type=int, required=True)
    request.add_argument("--deadline", type=int, default=1500)
    request.add_argument("--expected-tokens", type=int)
    request.add_argument("--expected-reference")
    request.add_argument("--ignore-eos", action="store_true")
    request.add_argument("--out", type=Path, required=True)
    request.set_defaults(handler=command_request)

    reset = subparsers.add_parser("reset-local")
    reset.add_argument("--port", type=int, required=True)
    reset.add_argument("--label", required=True)
    reset.add_argument("--out", type=Path, required=True)
    reset.set_defaults(handler=command_reset)

    needles = subparsers.add_parser("needles")
    needles.add_argument("--port", type=int, required=True)
    needles.add_argument("--model", default=DEFAULT_MODEL)
    needles.add_argument("--suite-identity", required=True)
    needles.add_argument("--target-tokens", type=int, default=990_000)
    needles.add_argument("--depths", type=int, nargs="+", default=(10, 50, 90))
    needles.add_argument("--deadline", type=int, default=1800)
    needles.add_argument("--out", type=Path, required=True)
    needles.set_defaults(handler=command_needles)

    eviction = subparsers.add_parser("eviction")
    eviction.add_argument("--port", type=int, required=True)
    eviction.add_argument("--model", default=DEFAULT_MODEL)
    eviction.add_argument("--suite-identity", required=True)
    eviction.add_argument("--docs", type=int, default=40)
    eviction.add_argument("--words", type=int, default=12_000)
    eviction.add_argument("--deadline", type=int, default=900)
    eviction.add_argument("--store-drain-seconds", type=float, default=10.0)
    eviction.add_argument("--out", type=Path, required=True)
    eviction.set_defaults(handler=command_eviction)

    prefill = subparsers.add_parser("prefill")
    prefill.add_argument("--port", type=int, required=True)
    prefill.add_argument("--model", default=DEFAULT_MODEL)
    prefill.add_argument("--suite-identity", required=True)
    prefill.add_argument("--label", required=True)
    prefill.add_argument("--target-tokens", type=int, default=33_000)
    prefill.add_argument("--runs", type=int, default=8)
    prefill.add_argument("--deadline", type=int, default=900)
    prefill.add_argument("--out", type=Path, required=True)
    prefill.set_defaults(handler=command_prefill)

    hitmiss = subparsers.add_parser("hitmiss")
    hitmiss.add_argument("--port", type=int, required=True)
    hitmiss.add_argument("--model", default=DEFAULT_MODEL)
    hitmiss.add_argument("--suite-identity", required=True)
    hitmiss.add_argument("--target-tokens", type=int, default=16_384)
    hitmiss.add_argument("--churn-docs", type=int, default=40)
    hitmiss.add_argument("--deadline", type=int, default=900)
    hitmiss.add_argument("--store-drain-seconds", type=float, default=8.0)
    hitmiss.add_argument("--eviction-settle-seconds", type=float, default=12.0)
    hitmiss.add_argument("--event-settle-seconds", type=float, default=0.5)
    hitmiss.add_argument("--event-endpoint", required=True)
    hitmiss.add_argument("--replay-endpoint", required=True)
    hitmiss.add_argument("--event-topic", required=True)
    hitmiss.add_argument("--out", type=Path, required=True)
    hitmiss.set_defaults(handler=command_hitmiss)

    storage = subparsers.add_parser("storage")
    storage.add_argument("--paths", nargs="+", required=True)
    storage.add_argument("--out", type=Path, required=True)
    storage.set_defaults(handler=command_storage)

    source_audit = subparsers.add_parser("source-audit")
    source_audit.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/josh/omp-workspace/drock-lmcache/LMCache"),
    )
    source_audit.add_argument("--out", type=Path, required=True)
    source_audit.set_defaults(handler=command_source_audit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return_code = int(args.handler(args))
    except Exception as error:
        out = getattr(args, "out", None)
        failure = {
            "schema": PROBE_SCHEMA,
            "failure_class": "probe_harness_exception",
            "passed": False,
            "fatal_error": f"{type(error).__name__}: {error}",
            "command": getattr(args, "command", None),
        }
        if isinstance(out, Path):
            save_json(out, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return_code = 1
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
