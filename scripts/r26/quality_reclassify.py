#!/usr/bin/env python3
"""Reclassify immutable R26 quality receipts without touching their raw data.

The original probes intentionally retain their first-pass assessments.  This
post-processor distinguishes a sampled wrong final answer from a request that
used its generation budget without ever emitting a visible final answer.  It
also keeps repetition, API/runtime failures, parser failures, and unverified
long-form completions separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

RAW_SCHEMA = "r26-quality-probe/v1"
REPORT_SCHEMA = "r26-quality-reclassification/v1"
DEFAULT_ROOT = Path(
    os.environ.get("BATTERY_ROOT", "/home/josh/omp-workspace/drock-lmcache/r26-battery")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def raw_choice(call: object) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(call, dict):
        return None, "missing raw_api object"
    if call.get("ok") is not True:
        return None, str(call.get("error") or f"HTTP {call.get('status')}")
    response = call.get("response")
    if not isinstance(response, dict):
        return None, "raw API response is not an object"
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, "raw API response has no first choice"
    return choices[0], ""


def visible_and_reasoning(choice: dict[str, Any]) -> tuple[str, str]:
    message = choice.get("message")
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    return (
        content if isinstance(content, str) else "",
        reasoning if isinstance(reasoning, str) else "",
    )


def usage_evidence(call: dict[str, Any]) -> dict[str, int | None]:
    request = call.get("request") if isinstance(call.get("request"), dict) else {}
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )

    def integer(value: object) -> int | None:
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    return {
        "requested_max_tokens": integer(request.get("max_tokens")),
        "completion_tokens": integer(usage.get("completion_tokens")),
        "reasoning_tokens": integer(details.get("reasoning_tokens")),
    }


def verifier_result(row: dict[str, Any]) -> bool | None:
    assessment = row.get("assessment")
    if isinstance(assessment, dict) and isinstance(assessment.get("correct"), bool):
        return assessment["correct"]
    verifier = row.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("correct"), bool):
        return verifier["correct"]
    if isinstance(row.get("correct"), bool):
        return row["correct"]
    return None


def original_category(row: dict[str, Any]) -> str:
    assessment = row.get("assessment")
    if isinstance(assessment, dict) and isinstance(assessment.get("category"), str):
        return assessment["category"]
    return "unclassified"


def reclassify_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    call = row.get("raw_api")
    choice, schema_error = raw_choice(call)
    original = original_category(row)
    verified = verifier_result(row)
    finish_reason: str | None = None
    content = ""
    reasoning = ""
    if choice is not None:
        finish = choice.get("finish_reason")
        finish_reason = str(finish) if finish is not None else None
        content, reasoning = visible_and_reasoning(choice)
    elif isinstance(row.get("finish_reason"), str):
        finish_reason = row["finish_reason"]

    call_object = call if isinstance(call, dict) else {}
    usage = usage_evidence(call_object)
    requested = usage["requested_max_tokens"]
    completion = usage["completion_tokens"]
    reasoning_tokens = usage["reasoning_tokens"]
    token_budget_exhausted = bool(
        finish_reason == "length"
        and (
            requested is None
            or completion is None
            or completion >= requested
        )
    )
    reasoning_consumed_budget = bool(
        token_budget_exhausted
        and reasoning_tokens is not None
        and completion is not None
        and reasoning_tokens >= completion
    )
    has_visible_final = bool(content.strip())
    repetition = original == "repetition" or bool(
        isinstance(row.get("assessment"), dict)
        and row["assessment"].get("repetition")
    )

    if schema_error or original == "runtime_error":
        derived = "runtime_error"
        basis = schema_error or "original runtime classification"
    elif repetition:
        derived = "repetition"
        basis = "repetition detector fired"
    elif verified is True and has_visible_final:
        derived = "correct_final"
        basis = "visible final answer passed its real verifier"
    elif token_budget_exhausted and (verified is not None or not has_visible_final):
        derived = "budget_limited_incomplete"
        basis = (
            "generation ended at its token limit before a verified visible final answer; "
            "intermediate reasoning numbers are not final-answer evidence"
        )
    elif verified is False and has_visible_final:
        derived = "wrong_final"
        basis = "visible completed answer failed its real verifier"
    elif not has_visible_final:
        derived = "incomplete_no_visible_final"
        basis = "API completed without visible answer content"
    elif original == "wrong_answer":
        derived = "model_quality_flag"
        basis = "non-verifier model-quality detector fired on visible output"
    else:
        derived = "completed_unverified"
        basis = "visible completion has no objective answer verifier"

    assessment = row.get("assessment") if isinstance(row.get("assessment"), dict) else {}
    return {
        "index": index,
        "phase": row.get("phase"),
        "wave": row.get("wave"),
        "run_index": row.get("run_index"),
        "topic": row.get("topic"),
        "setting": row.get("setting"),
        "original_category": original,
        "original_reason": assessment.get("reason"),
        "derived_category": derived,
        "basis": basis,
        "verifier_result": verified,
        "finish_reason": finish_reason,
        "has_visible_final": has_visible_final,
        "visible_chars": len(content),
        "reasoning_chars": len(reasoning),
        "token_budget_exhausted": token_budget_exhausted,
        "reasoning_consumed_entire_completion_budget": reasoning_consumed_budget,
        **usage,
        "raw_api_ok": call_object.get("ok"),
        "http_status": call_object.get("status"),
        "schema_error": schema_error,
    }


def receipt_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("outcomes", "results"):
        rows = receipt.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def classify_receipt(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    rows = receipt_rows(receipt)
    derived_rows = [reclassify_row(row, index) for index, row in enumerate(rows)]
    derived_counts = Counter(row["derived_category"] for row in derived_rows)
    transitions = Counter(
        f"{row['original_category']} -> {row['derived_category']}" for row in derived_rows
    )
    suite = receipt.get("suite")
    receipt_level: dict[str, Any] | None = None
    if not rows and suite == "tool_result_reordering":
        summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
        if summary.get("runtime_errors", 0):
            category = "runtime_error"
        elif summary.get("parser_and_reordering_correct") is True:
            category = "parser_pass"
        else:
            category = "parser_failure"
        receipt_level = {
            "derived_category": category,
            "basis": "receipt-level parser/reordering checks; not a model answer score",
        }
        derived_counts[category] += 1
    elif not rows and receipt.get("status") == "harness_error":
        receipt_level = {
            "derived_category": "runtime_error",
            "basis": receipt.get("error") or "probe harness error",
        }
        derived_counts["runtime_error"] += 1

    cache_effects = receipt.get("cache_effects")
    return {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "label": receipt.get("label"),
        "suite": suite,
        "started_at": receipt.get("started_at"),
        "finished_at": receipt.get("finished_at"),
        "original_summary": receipt.get("summary"),
        "derived_counts": dict(sorted(derived_counts.items())),
        "original_to_derived": dict(sorted(transitions.items())),
        "receipt_level": receipt_level,
        "cache_effects": cache_effects if isinstance(cache_effects, dict) else None,
        "items": derived_rows,
    }


def candidate_paths(root: Path, explicit: Iterable[Path], output: Path) -> list[Path]:
    supplied = [path.resolve() for path in explicit]
    paths = supplied or [path.resolve() for path in root.glob("quality-*.json")]
    output_resolved = output.resolve()
    return sorted({path for path in paths if path != output_resolved})


def cache_probe_assessment(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    cache_receipts = [receipt for receipt in receipts if receipt.get("suite") == "cache_correctness"]
    budget_limited = [
        {
            "label": receipt.get("label"),
            "phase": item.get("phase"),
            "finish_reason": item.get("finish_reason"),
            "completion_tokens": item.get("completion_tokens"),
            "reasoning_tokens": item.get("reasoning_tokens"),
            "visible_chars": item.get("visible_chars"),
        }
        for receipt in cache_receipts
        for item in receipt.get("items", [])
        if item.get("derived_category") == "budget_limited_incomplete"
    ]
    return {
        "observed_budget_limited_items": budget_limited,
        "design_assessment": (
            "The divergent suffix currently uses the benchmark's hotel-lights reasoning puzzle, not the "
            "LAVD ledger profile. That adds reasoning-depth and completion-budget failure modes to a KV "
            "correctness check. A length-finished, reasoning-only response is incomplete and cannot be "
            "called an observed wrong final answer."
        ),
        "recommended_future_probe": (
            "Because the dedicated cache phase already tests boundary and restore mechanics, use a short "
            "known-answer retrieval question over the same shared archive for any future divergent-suffix "
            "quality check. Keep the suffix different, verify an unambiguous archived fact, and reserve "
            "reasoning puzzles for their dedicated quality profile."
        ),
        "original_receipts_modified": False,
    }


def build_report(root: Path, paths: list[Path], output: Path) -> tuple[dict[str, Any], int]:
    receipts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            receipt = read_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if receipt.get("schema") != RAW_SCHEMA:
            continue
        receipts.append(classify_receipt(path, receipt))

    totals = Counter()
    transitions = Counter()
    for receipt in receipts:
        totals.update(receipt["derived_counts"])
        transitions.update(receipt["original_to_derived"])
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "root": str(root),
        "output": str(output),
        "policy": {
            "runtime_error": "HTTP/transport/JSON/choice/message failure",
            "repetition": "the repetition detector fired, independently of answer correctness",
            "budget_limited_incomplete": (
                "finish_reason=length before a verified visible final; reasoning text and numbers are "
                "not treated as a final answer"
            ),
            "wrong_final": "a visible final answer exists and fails its real verifier",
            "correct_final": "a visible final answer exists and passes its real verifier",
            "completed_unverified": "usable visible output exists but the workload has no answer key",
            "parser_failure": "tool/parser semantic checks failed without an API runtime failure",
        },
        "raw_receipt_policy": {
            "read_only": True,
            "original_receipts_modified": False,
            "integrity": "each included input is recorded by path and SHA-256",
        },
        "input_count": len(receipts),
        "input_errors": errors,
        "derived_totals": dict(sorted(totals.items())),
        "original_to_derived": dict(sorted(transitions.items())),
        "cache_probe_assessment": cache_probe_assessment(receipts),
        "receipts": receipts,
    }
    return report, int(bool(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or (args.root / "quality-reclassification.json")
    paths = candidate_paths(args.root, args.input, output)
    if any(path == output.resolve() for path in paths):
        raise SystemExit("output path must not overwrite an input receipt")
    report, status = build_report(args.root, paths, output)
    write_atomic(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "input_count": report["input_count"],
                "input_errors": len(report["input_errors"]),
                "derived_totals": report["derived_totals"],
            },
            indent=2,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
