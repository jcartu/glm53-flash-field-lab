#!/usr/bin/env python3
"""Source-linked scalar and HTTP boundary probes for the focused R27 qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA = "r27-boundary-probe/v2"
CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "low"}
BOUNDARY_TARGETS = (255, 256, 257, 2047, 2048, 2049, 8191, 8192, 8193)
BOUNDARY_NEGATIVE_TARGETS = (257, 2049, 8193)
SHARED_PREFIX_TARGETS = (4095, 4097, 8191, 8193)
TOOL_PREFIX_TARGET = 8193
AGENT_PREFIX_TARGET = 4097
MAX_TOKENS = 96
MATCH_UNIT = 256
DEFAULT_TARGET_PAGE = 2048
DEFAULT_RECURRENT_PAGE = 2048
FINE_TARGET_PAGE = 256
FINE_RECURRENT_PAGE = 256
FINE_GEOMETRY_ENV = {
    "GLM53_TARGET_BLOCK_SIZE": str(FINE_TARGET_PAGE),
    "GLM53_MAMBA_BLOCK_SIZE": str(FINE_RECURRENT_PAGE),
}
GEOMETRY_LAUNCHER_CONTRACT = {
    "mechanism": (
        "serve-glm53-flash.sh resolves GLM53_TARGET_BLOCK_SIZE and "
        "GLM53_MAMBA_BLOCK_SIZE environment variables and exports "
        "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE / VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE "
        "for the engine"
    ),
    "vram_defaults": (
        "GPU-only (vram) serving defaults the target page to 2048 and the mamba "
        "page to auto, which resolves equal to the resolved target page"
    ),
    "constraint": (
        "an explicit mamba page must be a multiple of 64 and of the target page, "
        "so the 256-token recurrent diagnostic sets both GLM53_TARGET_BLOCK_SIZE "
        "and GLM53_MAMBA_BLOCK_SIZE to 256"
    ),
    "launcher_source": "/usr/local/bin/serve-glm53-flash.sh",
    "launcher_sha256": (
        "0c35ba35e24e5915b8da1ea2e1cae19e84ed1156057dd059800150d33f41ff18"
    ),
    "launcher_evidence_lines": "resolution 126-133, validation 134-146, exports 173-174",
    "image_label_target_page": (
        "local-inference.cache.target-block-size: 2048 for GPU-only serving; "
        "LMCache derives a DCP-aligned per-rank page"
    ),
    "image_label_recurrent_page": (
        "local-inference.cache.recurrent-block-size: equal to the resolved target page"
    ),
    "realized_authority": (
        "vllm:cache_config_info block_size / mamba_block_size / prefix_match_unit "
        "labels are authoritative at runtime; the probe reads them before every suite"
    ),
}
BOUNDARY_SOURCE_PATH = "/opt/glm53-flash/vllm/vllm/v1/worker/gpu/boundary_checkpoint.py"
MANAGER_SOURCE_PATH = "/opt/glm53-flash/vllm/vllm/v1/core/single_type_kv_cache_manager.py"
LAUNCHER_SOURCE_PATH = "/usr/local/bin/serve-glm53-flash.sh"
LMCACHE_LAUNCHER_SOURCE_PATH = (
    "/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_workspace",
            "description": "Return deterministic evidence for one requested workspace step.",
            "parameters": {
                "type": "object",
                "properties": {"step": {"type": "integer"}},
                "required": ["step"],
            },
        },
    }
]
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
METRICS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached_total",
    "local_prefix_queries": "vllm:prefix_cache_queries_total",
    "local_prefix_hits": "vllm:prefix_cache_hits_total",
    "external_prefix_queries": "vllm:external_prefix_cache_queries_total",
    "external_prefix_hits": "vllm:external_prefix_cache_hits_total",
}
SOURCE_METRIC = "vllm:prompt_tokens_by_source_total"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def source_provenance(manifest_path: Path, source_arm: str) -> dict[str, Any]:
    manifest = read_object(manifest_path)
    entries = manifest.get("images")
    selected = next(
        (entry for entry in entries or [] if isinstance(entry, dict) and entry.get("arm") == source_arm),
        None,
    )
    if selected is None:
        raise ValueError(f"source arm {source_arm!r} is absent from {manifest_path}")
    files: list[dict[str, Any]] = []
    for item in selected.get("files") or []:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        expected = item.get("sha256")
        try:
            actual = file_sha256(path)
            error = None
        except OSError as caught:
            actual = None
            error = f"{type(caught).__name__}: {caught}"
        files.append(
            {
                "container_path": item.get("container_path"),
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches_manifest": actual == expected,
                "error": error,
            }
        )
    relevant = {
        row["container_path"]: row
        for row in files
        if row["container_path"]
        in {
            BOUNDARY_SOURCE_PATH,
            MANAGER_SOURCE_PATH,
            LAUNCHER_SOURCE_PATH,
            LMCACHE_LAUNCHER_SOURCE_PATH,
        }
    }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_schema": manifest.get("schema"),
        "source_arm": source_arm,
        "image": selected.get("image"),
        "files": files,
        "relevant_files": relevant,
        "all_manifest_files_verified": bool(files) and all(row["matches_manifest"] for row in files),
        "inherited_source_lock_is_not_patch_evidence": True,
    }


def _cache_helpers() -> tuple[Any, Any, Any]:
    try:
        from .cache_probe import http_json, response_summary, semantic_answer_evidence
    except (ImportError, TypeError):
        from cache_probe import http_json, response_summary, semantic_answer_evidence
    return http_json, response_summary, semantic_answer_evidence


def loopback_port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("R27 probes require a loopback HTTP endpoint")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain a path, query, or fragment")
    return parsed.port or 80


def raw_get(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
            headers = dict(response.headers.items())
        error = None
    except urllib.error.HTTPError as caught:
        raw = caught.read()
        status = int(caught.code)
        headers = dict(caught.headers.items())
        error = f"HTTPError: {caught}"
    except (OSError, TimeoutError) as caught:
        return {
            "status": None,
            "headers": {},
            "raw_body": "",
            "raw_body_sha256": None,
            "parsed": None,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(caught).__name__}: {caught}",
        }
    text = raw.decode(errors="replace")
    try:
        parsed: object | None = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return {
        "status": status,
        "headers": headers,
        "raw_body": text,
        "raw_body_sha256": digest(raw),
        "parsed": parsed,
        "elapsed_seconds": time.perf_counter() - started,
        "error": error,
    }


def decode_label(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def parse_labels(metric: str) -> dict[str, str]:
    return {match.group(1): decode_label(match.group(2)) for match in LABEL_RE.finditer(metric)}


def parse_metrics(raw: str) -> dict[str, Any]:
    wanted = {name: key for key, name in METRICS.items()}
    values: dict[str, float] = {}
    cache_configs: list[dict[str, str]] = []
    for line in raw.splitlines():
        pieces = line.rsplit(None, 1)
        if not line or line.startswith("#") or len(pieces) != 2:
            continue
        metric, raw_value = pieces
        name = metric.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if name in wanted:
            key = wanted[name]
        elif name == SOURCE_METRIC:
            source = parse_labels(metric).get("source")
            key = f"prompt_source_{source}" if source else ""
        elif name == "vllm:cache_config_info":
            cache_configs.append(parse_labels(metric))
            continue
        else:
            continue
        if key:
            values[key] = values.get(key, 0.0) + value
    return {"values": values, "cache_configs": cache_configs}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    keys = sorted(set(before["values"]) | set(after["values"]) | set(METRICS))
    result: dict[str, float | None] = {}
    for key in keys:
        old = before["values"].get(key)
        new = after["values"].get(key)
        result[key] = new - old if old is not None and new is not None and new >= old else None
    return result


def tokenize_messages(
    port: int,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    timeout: int,
    label: str,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    http_json, _, _ = _cache_helpers()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }
    if tools is not None:
        payload["tools"] = tools
    record = http_json(port, "/tokenize", payload=payload, timeout=timeout, label=label)
    parsed = record.get("parsed")
    token_ids = parsed.get("tokens") if isinstance(parsed, dict) else None
    count = parsed.get("count") if isinstance(parsed, dict) else None
    if (
        record.get("ok") is not True
        or not isinstance(count, int)
        or not isinstance(token_ids, list)
        or len(token_ids) != count
        or not all(isinstance(token, int) for token in token_ids)
    ):
        raise RuntimeError(f"tokenization failed: {record.get('error') or record.get('raw_body')}")
    return {
        "count": count,
        "token_ids": token_ids,
        "token_ids_sha256": digest(canonical(token_ids)),
        "messages_sha256": digest(canonical(messages)),
        "payload_sha256": digest(canonical(payload)),
        "http": record,
    }


def instruction_messages(target: int, filler: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "R27 shared instruction boundary. Preserve every preceding instruction exactly. "
                "For each user request, put only its requested marker in the visible final answer."
            ),
        },
        {
            "role": "developer",
            "content": (
                f"Deterministic shared developer preamble r27-prefix-{target}. "
                + " datum" * filler
                + " End of shared developer instructions. Never copy a marker from an earlier turn."
            ),
        },
    ]


def calibrate_instruction_prefix(
    port: int, model: str, target: int, timeout: int
) -> dict[str, Any]:
    filler = max(1, target - 80)
    attempts: list[dict[str, Any]] = []
    for attempt in range(16):
        messages = instruction_messages(target, filler)
        tokenized = tokenize_messages(
            port,
            model,
            messages,
            tools=TOOLS,
            timeout=timeout,
            label=f"instruction-{target}-cal-{attempt}",
            add_generation_prompt=False,
        )
        attempts.append(
            {
                "attempt": attempt,
                "filler_words": filler,
                "observed_tokens": tokenized["count"],
                "token_ids_sha256": tokenized["token_ids_sha256"],
                "http": tokenized["http"],
            }
        )
        if tokenized["count"] == target:
            return {"messages": messages, "tokenization": tokenized, "attempts": attempts}
        filler += target - tokenized["count"]
        if filler < 1:
            break
    raise RuntimeError(f"could not calibrate exact {target}-token instruction preamble")


def boundary_messages(target: int, filler: int, marker: str, *, divergent: bool = False) -> list[dict[str, Any]]:
    words = ["datum"] * filler
    if divergent and words:
        words[max(0, len(words) * 3 // 4 - 1)] = "vector"
    content = (
        f"R27 boundary fixture {target}. "
        + " ".join(words)
        + f"\nIgnore the filler. Reply with exactly {marker} and no other visible text."
    )
    return [
        {"role": "system", "content": "Follow the final marker instruction exactly."},
        {"role": "developer", "content": f"Deterministic boundary canary {target}."},
        {"role": "user", "content": content},
    ]


def calibrate_boundary_prompt(
    port: int, model: str, target: int, marker: str, timeout: int
) -> dict[str, Any]:
    filler = max(1, target - 50)
    attempts: list[dict[str, Any]] = []
    for attempt in range(16):
        messages = boundary_messages(target, filler, marker)
        tokenized = tokenize_messages(
            port,
            model,
            messages,
            tools=None,
            timeout=timeout,
            label=f"boundary-{target}-cal-{attempt}",
        )
        attempts.append(
            {
                "attempt": attempt,
                "filler_words": filler,
                "observed_tokens": tokenized["count"],
                "token_ids_sha256": tokenized["token_ids_sha256"],
                "http": tokenized["http"],
            }
        )
        if tokenized["count"] == target:
            return {
                "messages": messages,
                "filler_words": filler,
                "tokenization": tokenized,
                "attempts": attempts,
            }
        filler += target - tokenized["count"]
        if filler < 1:
            break
    raise RuntimeError(f"could not calibrate exact {target}-token boundary prompt")


def planned_case_ids(suites: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    if "shared" in suites:
        for target in SHARED_PREFIX_TARGETS:
            result.extend(
                f"shared-{target}-{step}"
                for step in ("cold", "exact-replay", "different-user", "divergent-prefix")
            )
            if target == TOOL_PREFIX_TARGET:
                result.extend((f"shared-{target}-tool-seed", f"shared-{target}-tool-append"))
        result.extend(f"agent-turn-{turn}" for turn in range(4))
    if "boundary" in suites:
        for target in BOUNDARY_TARGETS:
            result.extend((f"boundary-{target}-cold", f"boundary-{target}-exact-replay"))
            if target in BOUNDARY_NEGATIVE_TARGETS:
                result.append(f"boundary-{target}-divergent-prefix")
    return result


def planned_request_count(suites: tuple[str, ...]) -> int:
    return len(planned_case_ids(suites))


def geometry_plan(dcp: int, spec: str, geometry: str) -> dict[str, Any]:
    fine = geometry == "fine-pages-256"
    target_page = FINE_TARGET_PAGE if fine else DEFAULT_TARGET_PAGE
    recurrent = FINE_RECURRENT_PAGE if fine else DEFAULT_RECURRENT_PAGE
    policy = "request-boundaries" if dcp == 1 and spec in {"mtp0", "mtp3"} else "aligned"
    return {
        "name": geometry,
        "public_hash_block_tokens": MATCH_UNIT,
        "hash_unit_note": (
            "planned hash-unit hypothesis; the realized vllm:cache_config_info "
            "prefix_match_unit (or block_size when prefix_match_unit is unset) is "
            "authoritative for the oracle"
        ),
        "target_page_tokens_per_dcp_rank": target_page,
        "effective_attention_page_tokens": target_page * dcp,
        "recurrent_page_tokens": recurrent,
        "checkpoint_policy_expectation": policy,
        "policy_source": (
            "source.lock runtime.vllm.recurrent-checkpoint-policy: auto selects "
            "request-boundary checkpoints for DCP1 no-speculation and MTP; DCP4, "
            "DFlash2, and external-cache serving use aligned retention"
        ),
        "request_boundary_replay_rule": (
            "exact endpoint checkpoints let a follow-up request reuse every token "
            "of the exact rendered common prefix; the engine reports those cached "
            "prompt tokens"
        ),
        "aligned_hybrid_rule": (
            "floor the exact common prefix to the realized hash unit, materialize "
            "recurrent state at recurrent-page boundaries, and tolerate one "
            "hash-unit EAGLE rewind below the recurrent floor"
        ),
        "fine_geometry_environment": dict(FINE_GEOMETRY_ENV) if fine else {},
        "fine_geometry_is_separate_diagnostic": fine,
        "launcher_geometry_contract": GEOMETRY_LAUNCHER_CONTRACT,
    }


def realized_geometry(metrics: dict[str, Any], planned: dict[str, Any]) -> dict[str, Any]:
    labels = metrics.get("cache_configs") or []
    merged: dict[str, str] = {}
    for row in labels:
        if isinstance(row, dict):
            merged.update({str(key): str(value) for key, value in row.items()})

    def positive_int(raw: str | None) -> int | None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    observed_mamba = positive_int(merged.get("mamba_block_size"))
    observed_block = positive_int(merged.get("block_size"))
    observed_match_unit = positive_int(merged.get("prefix_match_unit"))
    hash_unit = (
        observed_match_unit
        or observed_block
        or int(planned["public_hash_block_tokens"])
    )
    recurrent = observed_mamba or int(planned["recurrent_page_tokens"])
    return {
        "metric": "vllm:cache_config_info",
        "raw_label_sets": labels,
        "merged_labels": merged,
        "public_block_size_tokens": merged.get("block_size"),
        "prefix_match_unit_tokens": merged.get("prefix_match_unit"),
        "mamba_block_size_tokens": merged.get("mamba_block_size"),
        "mamba_cache_mode": merged.get("mamba_cache_mode"),
        "user_specified_block_size": merged.get("user_specified_block_size"),
        "user_specified_mamba_block_size": merged.get(
            "user_specified_mamba_block_size"
        ),
        "expected_target_page_tokens": int(
            planned["target_page_tokens_per_dcp_rank"]
        ),
        "expected_mamba_block_size_tokens": int(planned["recurrent_page_tokens"]),
        "attention_block_size_matches_plan": (
            observed_block == int(planned["target_page_tokens_per_dcp_rank"])
        ),
        "mamba_geometry_matches_plan": (
            observed_mamba == int(planned["recurrent_page_tokens"])
        ),
        "effective_hash_unit_tokens": hash_unit,
        "effective_recurrent_page_tokens": recurrent,
        "hash_unit_source": (
            "prefix_match_unit"
            if observed_match_unit
            else ("block_size" if observed_block else "planned")
        ),
        "recurrent_source": "mamba_block_size" if observed_mamba else "planned",
        "target_page_realization_source": (
            "launcher VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE export; vram default 2048, "
            "fine diagnostic sets GLM53_TARGET_BLOCK_SIZE=256"
        ),
        "realized_claim_available": bool(labels) and observed_mamba is not None,
    }


def maximum_reuse(lcp: int, current_tokens: int, geometry: dict[str, Any]) -> int:
    """Correctness ceiling: the most cache reuse the exact rendered prefix justifies.

    Request-boundary checkpoints (DCP1 no-speculation and MTP) publish exact
    endpoints, so every token of the exact common prefix may be reused. Aligned
    hybrid retention can never legitimately reuse beyond the hash-aligned floor
    of the exact common prefix.
    """
    if geometry["checkpoint_policy_expectation"] == "request-boundaries":
        return max(0, min(lcp, current_tokens))
    hash_unit = max(1, int(geometry["public_hash_block_tokens"]))
    return lcp - lcp % hash_unit


def aligned_recurrent_floor(bound: int, geometry: dict[str, Any]) -> int:
    recurrent = max(1, int(geometry["recurrent_page_tokens"]))
    return bound - bound % recurrent


def case_oracle(
    *,
    case_kind: str,
    token_ids: list[int],
    history: list[tuple[str, list[int]]],
    geometry: dict[str, Any],
    instruction_tokens: int | None,
) -> dict[str, Any]:
    comparisons = [
        {"case_id": case_id, "common_prefix_tokens": common_prefix(prior, token_ids)}
        for case_id, prior in history
    ]
    best = max(comparisons, key=lambda row: row["common_prefix_tokens"], default=None)
    lcp = int(best["common_prefix_tokens"]) if best else 0
    upper = maximum_reuse(lcp, len(token_ids), geometry)
    negative = case_kind == "divergent-prefix"
    hash_unit = max(1, int(geometry["public_hash_block_tokens"]))
    if not history:
        lower = upper = 0
        basis = "cold unique namespace"
    elif negative:
        lower = 0
        basis = "negative safety oracle: never reuse beyond the exact rendered common prefix"
    elif geometry["checkpoint_policy_expectation"] == "request-boundaries":
        expected = min(lcp, len(token_ids))
        if case_kind in {"different-user", "tool-seed"} and instruction_tokens is not None:
            expected = min(instruction_tokens, upper)
        lower = max(0, expected - hash_unit)
        basis = (
            "leading system+developer instruction checkpoint"
            if case_kind in {"different-user", "tool-seed"}
            else "exact endpoint checkpoint replay with one hash-unit attention tolerance"
        )
    else:
        recurrent_floor = aligned_recurrent_floor(upper, geometry)
        lower = max(0, recurrent_floor - hash_unit)
        basis = (
            "aligned hybrid retention: hash-aligned lookup floor with recurrent-page "
            "materialization and one hash-unit EAGLE rewind tolerance"
        )
    return {
        "case_kind": case_kind,
        "history_common_prefixes": comparisons,
        "best_reference_case_id": best.get("case_id") if best else None,
        "exact_rendered_common_prefix_tokens": lcp,
        "current_prompt_tokens": len(token_ids),
        "checkpoint_policy_expectation": geometry["checkpoint_policy_expectation"],
        "hash_unit_tokens_used": hash_unit,
        "recurrent_page_tokens_used": int(geometry["recurrent_page_tokens"]),
        "expected_min_cached_tokens": lower,
        "expected_max_cached_tokens": upper,
        "reduced_hits_within_window_are_quality_observations_not_correctness_failures": True,
        "negative_prefix_oracle": negative,
        "basis": basis,
    }


def cache_observation(
    response: dict[str, Any], before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    parsed = response.get("parsed")
    transfer = parsed.get("kv_transfer_params") if isinstance(parsed, dict) else None
    stats = transfer.get("cached_token_stats") if isinstance(transfer, dict) else None
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    usage_cached = details.get("cached_tokens") if isinstance(details, dict) else None
    vllm_cached: int | None = None
    lmcache_cached: int | None = None
    extra_cached: int | None = None
    if isinstance(stats, dict):
        vllm_cached = (
            stats["num_vllm_cached_tokens"]
            if isinstance(stats.get("num_vllm_cached_tokens"), int)
            and stats["num_vllm_cached_tokens"] >= 0
            else None
        )
        lmcache_cached = (
            stats["num_lmcache_cached_tokens"]
            if isinstance(stats.get("num_lmcache_cached_tokens"), int)
            and stats["num_lmcache_cached_tokens"] >= 0
            else None
        )
        extra_cached = (
            stats["num_lmcache_extra_cached_tokens"]
            if isinstance(stats.get("num_lmcache_extra_cached_tokens"), int)
            and stats["num_lmcache_extra_cached_tokens"] >= 0
            else None
        )
    response_cached: int | None = None
    response_source: str | None = None
    if vllm_cached is not None and extra_cached is not None:
        response_cached = vllm_cached + extra_cached
        response_source = "kv_transfer_params.cached_token_stats vLLM+extra union"
    elif vllm_cached is not None:
        response_cached = vllm_cached
        response_source = "kv_transfer_params.cached_token_stats.num_vllm_cached_tokens"
    if response_cached is None and isinstance(usage_cached, int) and usage_cached >= 0:
        response_cached = usage_cached
        response_source = "usage.prompt_tokens_details.cached_tokens"
    delta = metric_delta(before, after)
    raw_metric = delta.get("prompt_tokens_cached")
    metric_cached = (
        int(raw_metric)
        if isinstance(raw_metric, float) and raw_metric.is_integer()
        else raw_metric
    )
    # The metric counter is vLLM-attributed only; the response-side vLLM
    # component is what must reconcile. LMCache-extra tokens are optional
    # transfer telemetry and cannot be required to appear in that counter.
    reconcile_basis = (
        vllm_cached
        if vllm_cached is not None
        else (usage_cached if isinstance(usage_cached, int) else None)
    )
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    serialized_counter_verified = (
        isinstance(prompt_tokens, int) and prompt_tokens > 0
        and delta.get("prompt_tokens") == prompt_tokens
        and isinstance(metric_cached, int) and 0 <= metric_cached <= prompt_tokens
        and delta.get("external_prefix_hits") == 0
        and delta.get("prompt_source_external_kv_transfer") == 0
    )
    observed_cached = response_cached
    observed_source = response_source
    if observed_cached is None and serialized_counter_verified:
        observed_cached = metric_cached
        observed_source = "serialized vLLM prompt-token counter delta (exact request token accounting)"
    observed = [
        value
        for value in (response_cached, usage_cached, metric_cached)
        if isinstance(value, int | float)
    ]
    return {
        "raw_cached_token_stats": stats,
        "response_cached_tokens": response_cached,
        "response_cached_source": response_source,
        "observed_cached_tokens": observed_cached,
        "observed_cached_source": observed_source,
        "serialized_counter_verified": serialized_counter_verified,
        "vllm_response_cached_tokens": vllm_cached,
        "lmcache_cached_tokens": lmcache_cached,
        "lmcache_extra_cached_tokens": extra_cached,
        "usage_cached_tokens": usage_cached,
        "metric_cached_tokens": metric_cached,
        "metric_delta": delta,
        "reconciliation_supported": (
            reconcile_basis is not None and metric_cached is not None
        ),
        "reconciled": (
            reconcile_basis == metric_cached
            if reconcile_basis is not None and metric_cached is not None
            else None
        ),
        "all_exposed_counts_agree": len(set(observed)) <= 1,
    }


def execute_chat_case(
    *,
    base_url: str,
    port: int,
    model: str,
    case_id: str,
    case_kind: str,
    messages: list[dict[str, Any]],
    marker: str,
    cache_salt: str,
    tools: list[dict[str, Any]] | None,
    history: list[tuple[str, list[int]]],
    geometry: dict[str, Any],
    instruction_tokens: int | None,
    timeout: int,
    linkage: dict[str, str],
) -> dict[str, Any]:
    http_json, response_summary, semantic_answer_evidence = _cache_helpers()
    started = utc_now()
    tokenized = tokenize_messages(
        port,
        model,
        messages,
        tools=tools,
        timeout=timeout,
        label=case_id + "-tokenize",
    )
    oracle = case_oracle(
        case_kind=case_kind,
        token_ids=tokenized["token_ids"],
        history=history,
        geometry=geometry,
        instruction_tokens=instruction_tokens,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 0,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "cache_salt": cache_salt,
        "kv_transfer_params": {"cached_token_stats": True},
        "return_token_ids": True,
    }
    if tools is not None:
        payload.update({"tools": tools, "tool_choice": "none"})
    before_http = raw_get(base_url, "/metrics", min(timeout, 30))
    before_raw = before_http.get("raw_body") or ""
    response = http_json(
        port,
        "/v1/chat/completions",
        payload=payload,
        timeout=timeout,
        label=case_id,
    )
    after_http = raw_get(base_url, "/metrics", min(timeout, 30))
    after_raw = after_http.get("raw_body") or ""
    summary = response_summary(response)
    response["summary"] = summary
    answer = semantic_answer_evidence(response, marker)
    observation = cache_observation(response, parse_metrics(before_raw), parse_metrics(after_raw))
    # Prometheus counters may publish asynchronously just after the response.
    # One bounded rescrape disambiguates a real response/metric disagreement
    # from a scrape landing between the response and its counter update.
    reconciliation_retry: dict[str, Any] = {"used": False, "after_retry": None}
    if (
        (observation["reconciliation_supported"] is True and observation["reconciled"] is False)
        or (observation["response_cached_tokens"] is None and not observation["serialized_counter_verified"])
    ):
        time.sleep(1.0)
        retry_after_http = raw_get(base_url, "/metrics", min(timeout, 30))
        observation = cache_observation(
            response,
            parse_metrics(before_raw),
            parse_metrics(retry_after_http.get("raw_body") or ""),
        )
        reconciliation_retry = {"used": True, "after_retry": retry_after_http}
    cached = observation["observed_cached_tokens"]
    oracle_passed = (
        isinstance(cached, int)
        and int(oracle["expected_min_cached_tokens"]) <= cached <= int(oracle["expected_max_cached_tokens"])
    )
    usage_matches = summary.get("prompt_tokens") == tokenized["count"]
    parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else {}
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    returned_prompt_ids = parsed.get("prompt_token_ids")
    generated_ids = choices[0].get("token_ids") if choices and isinstance(choices[0], dict) else None
    token_ids_verified = (
        isinstance(returned_prompt_ids, list)
        and returned_prompt_ids == tokenized["token_ids"]
        and isinstance(generated_ids, list)
        and all(isinstance(token, int) for token in generated_ids)
        and len(generated_ids) == summary.get("completion_tokens")
    )
    emitted_sequence = returned_prompt_ids + generated_ids if token_ids_verified else []
    return {
        "case_id": case_id,
        "suite": "boundary" if case_id.startswith("boundary-") else "shared",
        "attempt_state": "measured",
        "started_at": started,
        "finished_at": utc_now(),
        "linkage": linkage,
        "cache_salt": cache_salt,
        "cache_salt_sha256": digest(cache_salt),
        "marker": marker,
        "messages": messages,
        "messages_sha256": tokenized["messages_sha256"],
        "tokenizer": tokenized,
        "prompt_token_ids": tokenized["token_ids"],
        "prompt_token_ids_sha256": tokenized["token_ids_sha256"],
        "generated_token_ids": generated_ids,
        "emitted_sequence_token_ids": emitted_sequence,
        "emitted_sequence_sha256": digest(canonical(emitted_sequence)),
        "history_scope": "Exact returned prompt plus generated IDs; next requests may reuse prior assistant tokens. The final sampled token need not have been forwarded.",
        "request_payload_sha256": digest(canonical(payload)),
        "raw_request_body": canonical(payload),
        "raw_request_body_sha256": digest(canonical(payload)),
        "response": response,
        "answer_evidence": answer,
        "cache_observation": observation,
        "cache_oracle": {**oracle, "supported": isinstance(cached, int) and token_ids_verified,
                         "passed": oracle_passed if isinstance(cached, int) and token_ids_verified else None},
        "metrics": {
            "before": before_http,
            "after": after_http,
            "reconciliation_retry": reconciliation_retry,
        },
        "integrity": {
            "http_protocol_complete": response.get("ok") is True and summary.get("response_protocol_complete") is True,
            "prompt_usage_matches_exact_tokenizer_render": usage_matches,
            "returned_token_ids_match_render_and_usage": token_ids_verified,
            "visible_marker_correct": answer.get("accepted_visible_recall") is True,
            "cache_count_exposed": isinstance(cached, int),
            "cache_oracle_passed": oracle_passed if isinstance(cached, int) else None,
            "exposed_cache_counts_reconciled": observation["reconciled"],
        },
    }


def unattempted(case_id: str, reason: str, detail: object | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "suite": "boundary" if case_id.startswith("boundary-") else "shared",
        "attempt_state": "unattempted",
        "started_at": None,
        "finished_at": utc_now(),
        "error": reason,
        "detail": detail,
    }


def append_case(report: dict[str, Any], output: Path, row: dict[str, Any]) -> None:
    report["cases"].append(row)
    report["progress"] = {
        "planned": len(report["planned_case_ids"]),
        "recorded": len(report["cases"]),
        "measured": sum(case.get("attempt_state") == "measured" for case in report["cases"]),
        "unattempted": sum(case.get("attempt_state") == "unattempted" for case in report["cases"]),
        "last_updated_at": utc_now(),
    }
    atomic_json(output, report)


def safe_case(
    report: dict[str, Any], output: Path, case_id: str, operation: Any
) -> dict[str, Any]:
    try:
        row = operation()
    except Exception as error:
        row = unattempted(
            case_id,
            f"{type(error).__name__}: {error}",
            {"traceback": traceback.format_exc()},
        )
    append_case(report, output, row)
    return row


def run_shared_suite(
    report: dict[str, Any], output: Path, args: argparse.Namespace, geometry: dict[str, Any], linkage: dict[str, str]
) -> None:
    port = loopback_port(args.base_url)
    for target in SHARED_PREFIX_TARGETS:
        planned = [
            f"shared-{target}-cold",
            f"shared-{target}-exact-replay",
            f"shared-{target}-different-user",
            f"shared-{target}-divergent-prefix",
        ]
        if target == TOOL_PREFIX_TARGET:
            planned.extend((f"shared-{target}-tool-seed", f"shared-{target}-tool-append"))
        try:
            calibrated = calibrate_instruction_prefix(port, args.model, target, args.request_timeout)
        except Exception as error:
            detail = {"traceback": traceback.format_exc()}
            for case_id in planned:
                append_case(report, output, unattempted(case_id, f"instruction calibration failed: {error}", detail))
            continue
        report["instruction_preambles"][str(target)] = calibrated
        atomic_json(output, report)
        base = calibrated["messages"]
        instruction_tokens = calibrated["tokenization"]["count"]
        salt_identity = args.arm + ':' + args.spec + ':' + str(args.dcp) + ':' + args.kv + ':' + str(target)
        salt = f"r27-shared-{digest(salt_identity + ':' + linkage['config_sha256'])[:32]}"
        history: list[tuple[str, list[int]]] = []

        def run(messages: list[dict[str, Any]], marker: str, case_id: str, kind: str) -> dict[str, Any]:
            row = safe_case(
                report,
                output,
                case_id,
                lambda: execute_chat_case(
                    base_url=args.base_url,
                    port=port,
                    model=args.model,
                    case_id=case_id,
                    case_kind=kind,
                    messages=messages,
                    marker=marker,
                    cache_salt=salt,
                    tools=TOOLS,
                    history=history,
                    geometry=geometry,
                    instruction_tokens=instruction_tokens,
                    timeout=args.request_timeout,
                    linkage=linkage,
                ),
            )
            if row.get("attempt_state") == "measured" and row.get("integrity", {}).get("returned_token_ids_match_render_and_usage"):
                history.append((case_id, list(row["emitted_sequence_token_ids"])))
            return row

        marker_a = f"R27_SHARED_{target}_A"
        cold_messages = base + [{"role": "user", "content": f"Reply with exactly {marker_a}."}]
        run(cold_messages, marker_a, f"shared-{target}-cold", "cold")
        run(cold_messages, marker_a, f"shared-{target}-exact-replay", "exact-replay")
        marker_b = f"R27_SHARED_{target}_B"
        changed = base + [{"role": "user", "content": f"This is a different continuation. Reply with exactly {marker_b}."}]
        run(changed, marker_b, f"shared-{target}-different-user", "different-user")

        if target == TOOL_PREFIX_TARGET:
            call_id = f"call_shared_{target}_0"
            marker_c = f"R27_SHARED_{target}_TOOL_A"
            tool_seed = base + [
                {"role": "user", "content": "Inspect deterministic workspace step zero."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "inspect_workspace", "arguments": canonical({"step": 0})},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "EVIDENCE shared-step-zero 7f3f2f85"},
                {"role": "user", "content": f"Use that tool evidence and reply with exactly {marker_c}."},
            ]
            seed_row = run(tool_seed, marker_c, f"shared-{target}-tool-seed", "tool-seed")
            actual = ""
            if seed_row.get("attempt_state") == "measured":
                actual = str(seed_row.get("response", {}).get("summary", {}).get("content") or "")
            marker_d = f"R27_SHARED_{target}_TOOL_B"
            call_id_2 = f"call_shared_{target}_1"
            tool_append = tool_seed + [
                {"role": "assistant", "content": actual},
                {"role": "user", "content": "Inspect deterministic workspace step one."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id_2,
                            "type": "function",
                            "function": {"name": "inspect_workspace", "arguments": canonical({"step": 1})},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id_2, "content": "EVIDENCE shared-step-one 60f21190"},
                {"role": "user", "content": f"Continue the append-only transcript and reply with exactly {marker_d}."},
            ]
            run(tool_append, marker_d, f"shared-{target}-tool-append", "tool-append")

        marker_n = f"R27_SHARED_{target}_NEG"
        divergent = [
            {"role": "system", "content": f"DIVERGENT R27 prefix {target}; do not reuse another instruction state."},
            {"role": "developer", "content": "Independent negative-control instructions."},
            {"role": "user", "content": f"Reply with exactly {marker_n}."},
        ]
        run(divergent, marker_n, f"shared-{target}-divergent-prefix", "divergent-prefix")

    agent_ids = [f"agent-turn-{turn}" for turn in range(4)]
    try:
        calibrated = calibrate_instruction_prefix(port, args.model, AGENT_PREFIX_TARGET, args.request_timeout)
    except Exception as error:
        for case_id in agent_ids:
            append_case(
                report,
                output,
                unattempted(case_id, f"agent instruction calibration failed: {error}", {"traceback": traceback.format_exc()}),
            )
        return
    base = calibrated["messages"]
    agent_identity = args.arm + ':' + args.spec + ':' + str(args.dcp) + ':' + args.kv
    salt = f"r27-agent-{digest(agent_identity + ':' + linkage['config_sha256'])[:32]}"
    # The agent transcript lives in its own cache-salt namespace, so its oracle
    # history starts empty: the shared-target prompts cannot be visible to it
    # and must not seed turn-zero reuse expectations.
    agent_history: list[tuple[str, list[int]]] = []
    messages = base + [{"role": "user", "content": "Begin the append-only workspace transcript."}]
    for turn in range(4):
        marker = f"R27_AGENT_TURN_{turn}"
        current = messages + [{"role": "user", "content": f"Reply with exactly {marker}."}]
        case_id = f"agent-turn-{turn}"
        row = safe_case(
            report,
            output,
            case_id,
            lambda current=current, marker=marker, case_id=case_id: execute_chat_case(
                base_url=args.base_url,
                port=port,
                model=args.model,
                case_id=case_id,
                case_kind="cold" if turn == 0 else "tool-append",
                messages=current,
                marker=marker,
                cache_salt=salt,
                tools=TOOLS,
                history=agent_history,
                geometry=geometry,
                instruction_tokens=calibrated["tokenization"]["count"],
                timeout=args.request_timeout,
                linkage=linkage,
            ),
        )
        actual = ""
        if row.get("attempt_state") == "measured" and row.get("integrity", {}).get("returned_token_ids_match_render_and_usage"):
            agent_history.append((case_id, list(row["emitted_sequence_token_ids"])))
            actual = str(row.get("response", {}).get("summary", {}).get("content") or "")
        messages = current + [{"role": "assistant", "content": actual}]
        if turn < 3:
            call_id = f"call_agent_{turn}"
            messages.extend(
                [
                    {"role": "user", "content": f"Inspect workspace step {turn}."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "inspect_workspace", "arguments": canonical({"step": turn})},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"EVIDENCE agent-step-{turn} {digest('r27-agent:' + str(turn))[:16]}",
                    },
                ]
            )


def run_boundary_suite(
    report: dict[str, Any], output: Path, args: argparse.Namespace, geometry: dict[str, Any], linkage: dict[str, str]
) -> None:
    port = loopback_port(args.base_url)
    for target in BOUNDARY_TARGETS:
        ids = [f"boundary-{target}-cold", f"boundary-{target}-exact-replay"]
        if target in BOUNDARY_NEGATIVE_TARGETS:
            ids.append(f"boundary-{target}-divergent-prefix")
        marker = f"R27_BOUNDARY_{target}"
        try:
            calibrated = calibrate_boundary_prompt(port, args.model, target, marker, args.request_timeout)
        except Exception as error:
            for case_id in ids:
                append_case(
                    report,
                    output,
                    unattempted(case_id, f"boundary calibration failed: {error}", {"traceback": traceback.format_exc()}),
                )
            continue
        report["boundary_prompts"][str(target)] = calibrated
        atomic_json(output, report)
        grid_identity = args.arm + ':' + args.spec + ':' + str(args.dcp) + ':' + args.kv + ':' + str(target)
        salt = f"r27-grid-{digest(grid_identity + ':' + linkage['config_sha256'])[:32]}"
        history: list[tuple[str, list[int]]] = []

        def run(messages: list[dict[str, Any]], expected: str, case_id: str, kind: str) -> dict[str, Any]:
            row = safe_case(
                report,
                output,
                case_id,
                lambda: execute_chat_case(
                    base_url=args.base_url,
                    port=port,
                    model=args.model,
                    case_id=case_id,
                    case_kind=kind,
                    messages=messages,
                    marker=expected,
                    cache_salt=salt,
                    tools=None,
                    history=history,
                    geometry=geometry,
                    instruction_tokens=None,
                    timeout=args.request_timeout,
                    linkage=linkage,
                ),
            )
            if row.get("attempt_state") == "measured" and row.get("integrity", {}).get("returned_token_ids_match_render_and_usage"):
                history.append((case_id, list(row["emitted_sequence_token_ids"])))
            return row

        run(calibrated["messages"], marker, f"boundary-{target}-cold", "cold")
        run(calibrated["messages"], marker, f"boundary-{target}-exact-replay", "exact-replay")
        if target in BOUNDARY_NEGATIVE_TARGETS:
            negative_marker = f"R27_BOUNDARY_{target}_NEG"
            divergent = boundary_messages(
                target, calibrated["filler_words"], negative_marker, divergent=True
            )
            run(divergent, negative_marker, f"boundary-{target}-divergent-prefix", "divergent-prefix")


def summarize_http(report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    measured = [row for row in cases if row.get("attempt_state") == "measured"]
    unattempted_rows = [row for row in cases if row.get("attempt_state") == "unattempted"]
    unsupported_cache = [
        row["case_id"]
        for row in measured
        if row.get("cache_oracle", {}).get("supported") is not True
    ]
    protocol_failures = [
        row["case_id"]
        for row in measured
        if row.get("integrity", {}).get("http_protocol_complete") is not True
    ]
    answer_failures = [
        row["case_id"]
        for row in measured
        if row.get("integrity", {}).get("visible_marker_correct") is not True
        and row.get("answer_evidence", {}).get("visible_final_complete") is True
    ]
    incomplete_finals = [
        row["case_id"]
        for row in measured
        if row.get("integrity", {}).get("http_protocol_complete") is True
        and row.get("answer_evidence", {}).get("visible_final_complete") is not True
    ]
    cache_failures = [
        row["case_id"]
        for row in measured
        if row.get("cache_oracle", {}).get("passed") is False
    ]
    negative_failures = [
        row["case_id"]
        for row in measured
        if row.get("cache_oracle", {}).get("negative_prefix_oracle") is True
        and row.get("cache_oracle", {}).get("passed") is not True
    ]
    negative_cases = [
        row for row in measured
        if row.get("cache_oracle", {}).get("negative_prefix_oracle") is True
    ]
    reconcile_failures = [
        row["case_id"]
        for row in measured
        if row.get("cache_observation", {}).get("reconciliation_supported") is True
        and row.get("cache_observation", {}).get("reconciled") is not True
    ]
    fixed_prompt_basis = [
        {
            "case_id": row["case_id"],
            "messages_sha256": row.get("messages_sha256"),
            "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
        }
        for row in measured
        if not row["case_id"].startswith("agent-turn-") and not row["case_id"].endswith("tool-append")
    ]
    config_identity_passed = report.get("config_provenance", {}).get("passed") is True
    budget_limited_finals = [
        {
            "case_id": row["case_id"],
            "outcome": row.get("answer_evidence", {}).get("outcome"),
            "visible_final_status": row.get("response", {})
            .get("summary", {})
            .get("visible_final_status"),
        }
        for row in measured
        if row.get("answer_evidence", {}).get("generation_budget_reached") is True
    ]
    cache_verification_nonempty = any(
        row.get("cache_oracle", {}).get("supported") is True for row in measured
    )
    return {
        "planned_cases": len(report["planned_case_ids"]),
        "recorded_cases": len(cases),
        "measured_cases": len(measured),
        "unattempted_cases": len(unattempted_rows),
        "unsupported_cache_count_cases": unsupported_cache,
        "all_requested_cases_recorded": len(cases) == len(report["planned_case_ids"]),
        "all_requested_cases_measured": len(measured) == len(report["planned_case_ids"]),
        "protocol_failures": protocol_failures,
        "visible_answer_failures": answer_failures,
        "incomplete_visible_final_cases": incomplete_finals,
        "cache_oracle_failures": cache_failures,
        "negative_prefix_oracle_failures": negative_failures,
        "metric_reconciliation_failures": reconcile_failures,
        "config_identity_passed": config_identity_passed,
        "budget_limited_final_cases": budget_limited_finals,
        "fixed_prompt_set_sha256": digest(canonical(fixed_prompt_basis)),
        "execution_integrity_passed": (
            config_identity_passed
            and len(measured) == len(report["planned_case_ids"])
            and not protocol_failures
            and all(
                row.get("integrity", {}).get("prompt_usage_matches_exact_tokenizer_render") is True
                and row.get("integrity", {}).get("returned_token_ids_match_render_and_usage") is True
                for row in measured
            )
        ),
        "release_cache_conformance_passed": (
            cache_verification_nonempty
            and not unsupported_cache
            and len(measured) == len(report["planned_case_ids"])
            and not cache_failures
            and not reconcile_failures
        ),
        "visible_marker_conformance_passed": bool(measured) and not (
            answer_failures or incomplete_finals or protocol_failures
        ),
        "negative_prefix_safety_passed": bool(negative_cases) and not (
            negative_failures or unsupported_cache or unattempted_rows
        ),
        "claim_limits": {
            "production_95_percent_hit_rate_claim": False,
            "speed_claim": False,
            "foreign_gpu_window_used_for_speed": False,
            "reasoning_only_marker_is_counted_correct": False,
            "length_limited_empty_final_is_counted_wrong_answer": False,
            "budget_limited_finals_are_classified_separately_and_never_retried": True,
            "unsupported_cache_stats_are_telemetry_only_not_failures": True,
        },
    }


def expected_auto_policy() -> dict[str, object]:
    return {
        "prefill_compute_share": "auto",
        "prefill_compute_half_life": "responsive",
        "max_parallel_prefills": "auto",
        "prefill_policy": "decode-aware",
        "decode_refill_target": "auto",
    }


def run_http(args: argparse.Namespace) -> int:
    output = Path(args.output)
    suites = tuple(part for part in args.suites.split(",") if part)
    if not suites or any(part not in {"shared", "boundary"} for part in suites):
        raise ValueError("--suites must be a comma-separated subset of shared,boundary")
    if len(set(suites)) != len(suites):
        raise ValueError("--suites contains a duplicate")
    source = source_provenance(Path(args.manifest), args.source_arm)
    config_path = Path(args.config)
    config = read_object(config_path)
    config_sha = file_sha256(config_path)
    probe_sha = file_sha256(Path(__file__).resolve())
    linkage = {
        "source_manifest_sha256": source["manifest_sha256"],
        "config_sha256": config_sha,
        "probe_sha256": probe_sha,
    }
    planned_geometry = geometry_plan(args.dcp, args.spec, args.geometry)
    initial_metrics_http = raw_get(args.base_url, "/metrics", min(args.request_timeout, 30))
    initial_metrics = parse_metrics(initial_metrics_http.get("raw_body") or "")
    actual_geometry = realized_geometry(initial_metrics, planned_geometry)
    # The realized cache geometry is authoritative for every reuse oracle:
    # the hash unit and recurrent page come from vllm:cache_config_info labels,
    # never from a per-mode hard-coded grid.
    oracle_geometry = {
        **planned_geometry,
        "public_hash_block_tokens": actual_geometry["effective_hash_unit_tokens"],
        "recurrent_page_tokens": actual_geometry["effective_recurrent_page_tokens"],
        "realized_hash_unit_source": actual_geometry["hash_unit_source"],
        "realized_recurrent_source": actual_geometry["recurrent_source"],
    }
    fairness = raw_get(args.base_url, "/prefill_fairness", min(args.request_timeout, 30))
    auto_expected = expected_auto_policy() if args.arm == "auto" else None
    auto_matches = (
        isinstance(fairness.get("parsed"), dict)
        and all(fairness["parsed"].get(key) == value for key, value in (auto_expected or {}).items())
        if auto_expected is not None
        else None
    )
    env = config.get("env") if isinstance(config.get("env"), dict) else {}
    fine = args.geometry == "fine-pages-256"
    env_geometry = {
        key: str(env[key]) for key in FINE_GEOMETRY_ENV if key in env
    }
    config_checks = {
        "image": config.get("image") == args.image,
        "tp": config.get("tp") == 4,
        "dcp": config.get("dcp") == args.dcp,
        "spec": config.get("spec") == args.spec,
        "cache": config.get("cache") == "vram",
        "kv": config.get("kv") == args.kv,
        "batch": str(env.get("MAX_NUM_BATCHED_TOKENS")) == str(args.batch),
        "fine_geometry_env": (
            env_geometry == FINE_GEOMETRY_ENV if fine else env_geometry == {}
        ),
        "fine_geometry_realized": (
            actual_geometry.get("realized_claim_available") is True
            and actual_geometry.get("attention_block_size_matches_plan") is True
            and actual_geometry.get("mamba_geometry_matches_plan") is True
        ) if fine else True,
        "geometry_cli_args_absent": list(config.get("extra_args") or []) == [],
        "source_image": source.get("image") == args.image,
        "source_hashes": source["all_manifest_files_verified"],
        "auto_policy_get": auto_matches if args.arm == "auto" else True,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "http",
        "cell_id": args.cell_id,
        "arm": args.arm,
        "image": args.image,
        "started_at": utc_now(),
        "source_provenance": source,
        "config_provenance": {
            "path": str(config_path),
            "sha256": config_sha,
            "launch": config,
            "checks": config_checks,
            "passed": all(config_checks.values()),
        },
        "probe_provenance": {"path": str(Path(__file__).resolve()), "sha256": probe_sha},
        "runtime": {
            "base_url": args.base_url,
            "model": args.model,
            "tp": 4,
            "dcp": args.dcp,
            "spec": args.spec,
            "kv": args.kv,
            "batch": args.batch,
            "cache": "vram",
            "suites": list(suites),
            "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        },
        "geometry": {
            "planned": planned_geometry,
            "realized": actual_geometry,
            "oracle": oracle_geometry,
        },
        "initial_metrics": initial_metrics_http,
        "as_shipped_auto_policy": {
            "required": args.arm == "auto",
            "expected": auto_expected,
            "get_receipt": fairness,
            "matches": auto_matches,
        },
        "planned_case_ids": planned_case_ids(suites),
        "cases": [],
        "instruction_preambles": {},
        "boundary_prompts": {},
        "progress": {},
        "summary": {},
        "finished_at": None,
    }
    atomic_json(output, report)
    if "shared" in suites:
        run_shared_suite(report, output, args, oracle_geometry, linkage)
    if "boundary" in suites:
        run_boundary_suite(report, output, args, oracle_geometry, linkage)
    report["summary"] = summarize_http(report)
    report["finished_at"] = utc_now()
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2))
    return 0 if report["summary"]["all_requested_cases_recorded"] else 1


def scalar_case(torch: Any, kernel: Any, block: int, slot: int, size: int) -> dict[str, Any]:
    case_id = f"scalar-size-{size}-block-{block}-slot-{slot}"
    started = utc_now()
    destination = torch.full((3, size), 165, dtype=torch.uint8, device="cuda")
    pool = (
        torch.arange(3 * (size + 11), dtype=torch.int64, device="cuda")
        .remainder(251)
        .to(torch.uint8)
        .reshape(3, size + 11)
    )
    pool_before = pool.clone()
    metadata = torch.tensor(
        [[destination.data_ptr(), destination.stride(0), size, 5]],
        dtype=torch.int64,
        device="cuda",
    )
    expected_affected = block == 1 or slot == 1
    try:
        kernel[(1,)](
            metadata,
            pool,
            pool.stride(0),
            block,
            slot,
            BLOCK=1024,
        )
        torch.cuda.synchronize()
        selected_exact = bool(torch.equal(destination[slot], pool[block, 5 : 5 + size]))
        neighbors_unchanged = all(
            bool(torch.all(destination[index] == 165).item()) for index in range(3) if index != slot
        )
        source_unchanged = bool(torch.equal(pool, pool_before))
        passed = selected_exact and neighbors_unchanged and source_unchanged
        error = None
        error_traceback = None
        specialization_signature = False
    except Exception as caught:
        passed = False
        error = f"{type(caught).__name__}: {caught}"
        error_traceback = traceback.format_exc()
        try:
            torch.cuda.synchronize()
        except Exception as synchronize_error:
            error += (
                f"; synchronize after original failure: "
                f"{type(synchronize_error).__name__}: {synchronize_error}"
            )
        lowered = error.lower()
        to_attribute = (
            "no attribute 'to'" in lowered
            or 'no attribute "to"' in lowered
            or ".to" in lowered
        )
        specialization_signature = (
            "has no attribute" in lowered and to_attribute
            and ("int" in lowered or "constexpr" in lowered)
        )
        selected_exact = False
        neighbors_unchanged = all(
            bool(torch.all(destination[index] == 165).item()) for index in range(3) if index != slot
        )
        source_unchanged = bool(torch.equal(pool, pool_before))
    return {
        "case_id": case_id,
        "attempt_state": "measured",
        "started_at": started,
        "finished_at": utc_now(),
        "fixture": {
            "pool_dtype": "torch.uint8",
            "pool_shape": [3, size + 11],
            "destination_shape": [3, size],
            "destination_fill": 165,
            "metadata": [["destination.data_ptr()", "destination.stride(0)", size, 5]],
            "pool_stride_0": int(pool.stride(0)),
            "block": block,
            "slot": slot,
            "size": size,
            "BLOCK": 1024,
        },
        "known_specialization_affected_coordinate": expected_affected,
        "kernel_returned": error is None,
        "selected_slot_exact": selected_exact,
        "unselected_neighbor_slots_unchanged": neighbors_unchanged,
        "source_pool_unchanged": source_unchanged,
        "passed": passed,
        "specialized_python_int_to_signature": specialization_signature,
        "error": error,
        "traceback": error_traceback,
    }


def run_scalar(args: argparse.Namespace) -> int:
    output = Path(args.output)
    config_path = Path(args.config)
    config = read_object(config_path)
    source_expected = args.expected_source_sha256
    source_path = Path(args.source_path)
    try:
        actual_source = file_sha256(source_path)
        source_error = None
    except OSError as caught:
        actual_source = None
        source_error = f"{type(caught).__name__}: {caught}"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "scalar",
        "arm": args.arm,
        "image": args.image,
        "started_at": utc_now(),
        "probe_provenance": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        "source_provenance": {
            "manifest_sha256": args.manifest_sha256,
            "container_path": str(source_path),
            "expected_sha256": source_expected,
            "actual_packaged_sha256": actual_source,
            "matches_manifest": actual_source == source_expected,
            "error": source_error,
            "source_lock_not_used_as_patch_evidence": True,
        },
        "config_provenance": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
            "container": config,
            "image_matches": config.get("image") == args.image,
            "arm_matches": config.get("arm") == args.arm,
        },
        "fixture_contract": {
            "matrix": {"blocks": [0, 1, 2], "slots": [0, 1, 2], "sizes": [17, 1057]},
            "case_count": 18,
            "kernel_import": "vllm.v1.worker.gpu.boundary_checkpoint._restore_auxiliary_state_kernel",
            "kernel_reimplemented": False,
            "copy_semantics_verified_against_packaged_kernel": (
                "state 0 copies pool[block, 5:5+size] into destination[slot] with "
                "masked partial chunks; other slots and the source pool must be untouched"
            ),
            "specialization_mechanism": (
                "Triton specializes the Python int literal 1 into a kernel constexpr; "
                "the stock kernel applies .to(tl.int64) to the block and slot "
                "parameters (AttributeError on the constexpr int) while the patched "
                "kernel uses tl.cast(block, tl.int64) / tl.cast(slot, tl.int64)"
            ),
            "affected_coordinates": "block == 1 or slot == 1 (10 of 18 cases)",
            "control_coordinates": "block and slot both in {0, 2} (8 of 18 cases)",
        },
        "expected_disposition": args.expected_disposition,
        "cases": [],
        "summary": {},
        "finished_at": None,
    }
    atomic_json(output, report)
    try:
        import torch
        from vllm.v1.worker.gpu.boundary_checkpoint import _restore_auxiliary_state_kernel
    except Exception as error:
        for size in (17, 1057):
            for block in range(3):
                for slot in range(3):
                    report["cases"].append(
                        unattempted(
                            f"scalar-size-{size}-block-{block}-slot-{slot}",
                            f"packaged kernel import failed: {type(error).__name__}: {error}",
                            {"traceback": traceback.format_exc()},
                        )
                    )
        report["summary"] = {
            "matrix_execution_complete": False,
            "known_stock_reproduction_succeeded": False,
            "release_kernel_conformance_passed": False,
            "qualification_expectation_passed": False,
        }
        report["finished_at"] = utc_now()
        atomic_json(output, report)
        return 1
    for size in (17, 1057):
        for block in range(3):
            for slot in range(3):
                try:
                    row = scalar_case(torch, _restore_auxiliary_state_kernel, block, slot, size)
                except Exception as error:
                    row = unattempted(
                        f"scalar-size-{size}-block-{block}-slot-{slot}",
                        f"fixture construction failed: {type(error).__name__}: {error}",
                        {"traceback": traceback.format_exc()},
                    )
                report["cases"].append(row)
                atomic_json(output, report)
    measured = [row for row in report["cases"] if row.get("attempt_state") == "measured"]
    failures = [row for row in measured if row.get("passed") is not True]
    affected = [row for row in measured if row.get("known_specialization_affected_coordinate") is True]
    controls = [row for row in measured if row.get("known_specialization_affected_coordinate") is False]
    exact_stock_pattern = (
        len(measured) == 18
        and len(affected) == 10
        and all(row.get("specialized_python_int_to_signature") is True for row in affected)
        and all(row.get("passed") is True for row in controls)
    )
    conformance = len(measured) == 18 and not failures
    source_verified = actual_source == source_expected and source_error is None
    identity_verified = config.get("image") == args.image and config.get("arm") == args.arm
    expectation = exact_stock_pattern if args.expected_disposition == "known-stock-specialization-bug" else conformance
    report["summary"] = {
        "planned_cases": 18,
        "recorded_cases": len(report["cases"]),
        "measured_cases": len(measured),
        "failed_case_ids": [row["case_id"] for row in failures],
        "known_affected_coordinates": len(affected),
        "passing_controls": sum(row.get("passed") is True for row in controls),
        "matrix_execution_complete": len(measured) == 18,
        "known_stock_reproduction_succeeded": exact_stock_pattern,
        "release_kernel_conformance_passed": conformance and source_verified and identity_verified,
        "stock_release_passed": conformance if args.arm == "stock" else None,
        "qualification_expectation_passed": expectation and source_verified and identity_verified,
        "source_verified": source_verified,
        "local_reproduction_and_release_pass_are_separate": True,
    }
    report["finished_at"] = utc_now()
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2))
    return 0 if len(measured) == 18 else 1


def write_unattempted_http_receipt(
    output: Path,
    *,
    cell: dict[str, Any],
    suites: tuple[str, ...],
    reason: str,
    detail: object,
    source: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    now = utc_now()
    ids = planned_case_ids(suites)
    report = {
        "schema": SCHEMA,
        "mode": "http",
        "cell_id": cell["cell_id"],
        "arm": cell["arm"],
        "image": cell["image"],
        "started_at": now,
        "source_provenance": source,
        "config_provenance": {"launch": config, "available": config is not None},
        "runtime": cell,
        "geometry": {"planned": geometry_plan(cell["dcp"], cell["spec"], cell["geometry"]), "realized": None},
        "planned_case_ids": ids,
        "cases": [unattempted(case_id, reason, detail) for case_id in ids],
        "instruction_preambles": {},
        "boundary_prompts": {},
        "summary": {
            "planned_cases": len(ids),
            "recorded_cases": len(ids),
            "measured_cases": 0,
            "unattempted_cases": len(ids),
            "all_requested_cases_recorded": True,
            "all_requested_cases_measured": False,
            "execution_integrity_passed": False,
            "release_cache_conformance_passed": False,
            "visible_marker_conformance_passed": False,
            "negative_prefix_safety_passed": False,
            "failure": {"reason": reason, "detail": detail},
        },
        "finished_at": now,
    }
    atomic_json(output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    scalar = subparsers.add_parser("scalar", help="Run the packaged Triton scalar matrix on one GPU")
    scalar.add_argument("--arm", required=True, choices=("stock", "patched", "auto"))
    scalar.add_argument("--image", required=True)
    scalar.add_argument("--config", required=True)
    scalar.add_argument("--output", required=True)
    scalar.add_argument("--manifest-sha256", required=True)
    scalar.add_argument("--source-path", default=BOUNDARY_SOURCE_PATH)
    scalar.add_argument("--expected-source-sha256", required=True)
    scalar.add_argument(
        "--expected-disposition",
        required=True,
        choices=("known-stock-specialization-bug", "fixed-kernel-conformance"),
    )

    http = subparsers.add_parser("http", help="Run one source/config-linked model HTTP cell")
    http.add_argument("--arm", required=True, choices=("stock", "patched", "auto"))
    http.add_argument("--source-arm", required=True)
    http.add_argument("--image", required=True)
    http.add_argument("--manifest", required=True)
    http.add_argument("--config", required=True)
    http.add_argument("--output", required=True)
    http.add_argument("--cell-id", required=True)
    http.add_argument("--base-url", required=True)
    http.add_argument("--model", required=True)
    http.add_argument("--suites", required=True)
    http.add_argument("--dcp", required=True, type=int, choices=(1, 4))
    http.add_argument("--spec", required=True, choices=("mtp0", "mtp3", "dflash2"))
    http.add_argument("--kv", required=True, choices=("fp8_ds_mla", "nvfp4_ds_mla"))
    http.add_argument("--batch", required=True, type=int, choices=(4096, 8192))
    http.add_argument("--geometry", required=True, choices=("default", "fine-pages-256"))
    http.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.mode == "http" and args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_scalar(args) if args.mode == "scalar" else run_http(args)


if __name__ == "__main__":
    raise SystemExit(main())
