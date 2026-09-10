#!/usr/bin/env python3
"""R26 model-quality, corruption, parser, sampling, and cache-effect phase.

Every GPU configuration is booted serially through ``runtime.py``.  Probe
processes preserve raw API traffic; this coordinator records separate runtime,
model-quality, repetition, cache-effect, and configuration gates and continues
through independent cells after failures.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROBE = Path(__file__).with_name("quality_probes.py")
ACCEPTANCE_PROBE = Path(__file__).with_name("acceptance_probe.py")
MTP_HEAD_ENV = "VLLM_GLM53_MTP_DRAFT_HEAD"
PROFILE_RUNS = 24
PROBE_SCHEMA = "r26-quality-phase/v1"
PRIOR_ROOT = Path("/home/josh/omp-workspace/drock-lmcache/r25-battery")


@dataclass(frozen=True)
class ProbePlan:
    label: str
    suite: str
    timeout: int
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfilePlan:
    label: str
    profile: str
    runs: int = PROFILE_RUNS


@dataclass(frozen=True)
class Arm:
    label: str
    image: str
    dcp: int
    spec: str
    cache: str
    kv: str
    proposal_head: str
    role: str
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    acceptance_probe: bool = False
    probes: tuple[ProbePlan, ...] = ()
    profiles: tuple[ProfilePlan, ...] = ()


_LOCAL_GATES: list[dict[str, Any]] = []
_RECEIPTS: dict[str, dict[str, Any]] = {}


def load_runtime() -> Any:
    """Load Main's runtime lazily so importing this phase has no side effects."""
    try:
        return importlib.import_module(".runtime", package=__package__)
    except (ImportError, TypeError):
        return importlib.import_module("runtime")


def record_gate(rt: Any, name: str, passed: bool, gate_type: str, detail: object) -> None:
    enriched = {"gate_type": gate_type, "detail": detail}
    rt.record_gate(name, bool(passed), enriched)
    _LOCAL_GATES.append(
        {"name": name, "passed": bool(passed), "gate_type": gate_type, "detail": detail}
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "read_error", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(value, dict):
        return {"status": "read_error", "error": "JSON root is not an object"}
    return value


def save_failure_receipt(rt: Any, label: str, suite: str, reason: str, detail: object) -> dict[str, Any]:
    receipt = {
        "schema": PROBE_SCHEMA,
        "label": label,
        "suite": suite,
        "status": "not_run",
        "reason": reason,
        "detail": detail,
        "summary": {
            "requested": expected_requests(suite),
            "completed": 0,
            "runtime_errors": expected_requests(suite) or 1,
            "wrong_answers": 0,
            "repetitions": 0,
        },
    }
    rt.save_json(f"{label}.json", receipt)
    _RECEIPTS[label] = receipt
    return receipt


def expected_requests(suite: str) -> int:
    return {
        "cache": 3,
        "long": 24,
        "sampling": 24,
        "tool-order": 2,
        "estonia": PROFILE_RUNS,
        "lavd-test": PROFILE_RUNS,
    }.get(suite, 0)


def build_plan(rt: Any) -> tuple[Arm, ...]:
    cache = lambda label, *extra: ProbePlan(label, "cache", 2400, tuple(extra))
    long = lambda label: ProbePlan(
        label,
        "long",
        3600,
        ("--waves", "3", "--max-tokens", "8192", "--seed-base", "26090500"),
    )
    return (
        Arm(
            label="quality-r26-nospec-fp8-kv",
            image=rt.IMAGE,
            dcp=4,
            spec="mtp0",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="not active (no speculation)",
            role="official R26 no-speculation FP8-KV cache-correctness arm",
            probes=(cache("quality-r26-nospec-fp8-kv-cache"),),
        ),
        Arm(
            label="quality-r26-nospec-nvfp4-kv",
            image=rt.IMAGE,
            dcp=4,
            spec="mtp0",
            cache="vram",
            kv="nvfp4_ds_mla",
            proposal_head="not active (no speculation)",
            role="official R26 no-speculation packed-NVFP4-KV cache-correctness arm",
            probes=(cache("quality-r26-nospec-nvfp4-kv-cache"),),
        ),
        Arm(
            label="quality-r26-mtp3-default-nvfp4-head",
            image=rt.IMAGE,
            dcp=4,
            spec="mtp3",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="image default NVFP4 proposal head; BF16 target/verifier head",
            role="official R26 MTP3 default-head qualification; FP8 KV isolates proposal-head behavior",
            acceptance_probe=True,
            probes=(
                cache("quality-r26-mtp3-default-nvfp4-cache"),
                long("quality-r26-mtp3-default-nvfp4-long"),
            ),
        ),
        Arm(
            label="quality-r26-mtp3-bf16-head",
            image=rt.IMAGE,
            dcp=4,
            spec="mtp3",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="explicit BF16 proposal-head control",
            role="stock R26 fixed-BF16 MTP3/DCP4 correctness comparison arm",
            extra_env={MTP_HEAD_ENV: "bf16"},
            acceptance_probe=True,
            probes=(
                cache("quality-r26-mtp3-bf16-cache"),
                long("quality-r26-mtp3-bf16-long"),
            ),
        ),
        Arm(
            label="quality-r25-mtp3-bf16-head",
            image=rt.R25_IMAGE,
            dcp=4,
            spec="mtp3",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="R25 native BF16 proposal head; explicit bf16 request retained in settings",
            role="matched R25 MTP3/DCP4 correctness baseline",
            extra_env={MTP_HEAD_ENV: "bf16"},
            acceptance_probe=True,
            probes=(
                cache("quality-r25-mtp3-bf16-cache"),
                long("quality-r25-mtp3-bf16-long"),
            ),
        ),
        Arm(
            label="quality-overlay-mtp3-bf16-head",
            image=rt.OVERLAY_IMAGE,
            dcp=4,
            spec="mtp3",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="explicit BF16 proposal-head control",
            role=(
                "D-Rock Python overlay MTP3/DCP4 correctness candidate; same fixed head and prompts as "
                "stock R26 and R25, explicitly exercising the #561 integration"
            ),
            extra_env={MTP_HEAD_ENV: "bf16", "FAIRNESS_ENGINE": "none"},
            extra_args=("--prefill-compute-share", "0.4"),
            acceptance_probe=True,
            probes=(
                cache("quality-overlay-mtp3-bf16-cache"),
                long("quality-overlay-mtp3-bf16-long"),
            ),
        ),
        Arm(
            label="quality-r26-dflash-k7-dcp4",
            image=rt.IMAGE,
            dcp=4,
            spec="dflash2",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="not applicable (external DFlash2 MXFP8 draft, depth 7)",
            role="official R26 DFlash2 K7 quality, corruption, constrained-sampling, and parser arm",
            probes=(
                cache("quality-r26-dflash-k7-cache"),
                long("quality-r26-dflash-k7-long"),
                ProbePlan(
                    "quality-r26-dflash-k7-sampling",
                    "sampling",
                    1800,
                    ("--runs-per-setting", "6", "--seed-base", "26090600"),
                ),
                ProbePlan("quality-r26-dflash-k7-tool-order", "tool-order", 1200),
            ),
        ),
        Arm(
            label="quality-r26-dflash-k7-lmcache-history-control",
            image=rt.IMAGE,
            dcp=4,
            spec="dflash2",
            cache="lmcache",
            kv="nvfp4_ds_mla",
            proposal_head="not applicable (external DFlash2 MXFP8 draft, depth 7)",
            role=(
                "official R26 control matched to the prior R25 LMCache/NVFP4-KV Apollo runaway "
                "configuration"
            ),
            probes=(long("quality-r26-dflash-lmcache-history-long"),),
        ),
        Arm(
            label="quality-r25-dflash-k7-lmcache-history-control",
            image=rt.R25_IMAGE,
            dcp=4,
            spec="dflash2",
            cache="lmcache",
            kv="nvfp4_ds_mla",
            proposal_head="not applicable (external DFlash2 MXFP8 draft, depth 7)",
            role=(
                "matched R25 LMCache/NVFP4-KV control for the historical 23/24 result and "
                "Apollo repetition"
            ),
            probes=(long("quality-r25-dflash-lmcache-history-long"),),
        ),
        Arm(
            label="quality-r26-dflash-k7-dcp1-profiles",
            image=rt.IMAGE,
            dcp=1,
            spec="dflash2",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="not applicable (external DFlash2 MXFP8 draft, depth 7)",
            role="official R26 matched PR646-era DCP1/FP8/DFlash quality profiles",
            profiles=(
                ProfilePlan("quality-r26-dflash-estonia", "estonia"),
                ProfilePlan("quality-r26-dflash-lavd", "lavd-test"),
            ),
        ),
        Arm(
            label="quality-r25-dflash-k7-dcp1-profiles",
            image=rt.R25_IMAGE,
            dcp=1,
            spec="dflash2",
            cache="vram",
            kv="fp8_ds_mla",
            proposal_head="not applicable (external DFlash2 MXFP8 draft, depth 7)",
            role="matched R25 DCP1/FP8/DFlash quality-profile control",
            profiles=(
                ProfilePlan("quality-r25-dflash-estonia", "estonia"),
                ProfilePlan("quality-r25-dflash-lavd", "lavd-test"),
            ),
        ),
        Arm(
            label="quality-overlay-mtp3-bf16-lmcache",
            image=rt.OVERLAY_IMAGE,
            dcp=4,
            spec="mtp3",
            cache="lmcache",
            kv="nvfp4_ds_mla",
            proposal_head="explicit BF16 proposal-head control",
            role=(
                "separately labelled D-Rock overlay LMCache correctness candidate; this is not stock R26 "
                "and is not PR646"
            ),
            extra_env={MTP_HEAD_ENV: "bf16", "FAIRNESS_ENGINE": "none"},
            extra_args=("--prefill-compute-share", "0.4"),
            probes=(
                cache(
                    "quality-overlay-lmcache-correctness",
                    "--reset-local-between",
                    "--external-cache-expected",
                    "--store-wait-seconds",
                    "10",
                ),
            ),
        ),
    )


def arm_manifest(arm: Arm) -> dict[str, Any]:
    rt = load_runtime()
    return {
        "schema": PROBE_SCHEMA,
        "label": arm.label,
        "role": arm.role,
        "image": arm.image,
        "runtime": {
            "tp": 4,
            "dcp": arm.dcp,
            "spec": arm.spec,
            "cache": arm.cache,
            "kv": arm.kv,
            "extra_env": arm.extra_env,
            "extra_args": list(arm.extra_args),
            "proposal_head": arm.proposal_head,
            "scheduler_tokens": 4096,
            "max_sequences": 32,
            "prefill_schedule_interval": 1,
            "fairness": "compute_share=0.4",
        },
        "weights": {
            "target": str(rt.MODEL),
            "draft": str(rt.DRAFT) if arm.spec == "dflash2" else None,
            "checkpoint_cache_tag": rt.MODEL_CACHE_TAG,
            "checkpoint_immutable_during_arm": True,
        },
        "acceptance_probe": (
            {
                "script": str(ACCEPTANCE_PROBE),
                "label": arm.label,
                "contexts": [0, 32768],
                "concurrency": [1, 8],
                "duration_seconds": 30,
                "repeats": 2,
                "ordering": "before quality prompts",
                "metric_interpretation_owner": "Main",
            }
            if arm.acceptance_probe
            else None
        ),
        "probes": [
            {
                "label": probe.label,
                "suite": probe.suite,
                "arguments": list(probe.arguments),
                "timeout": probe.timeout,
            }
            for probe in arm.probes
        ],
        "profiles": [
            {
                "label": profile.label,
                "profile": profile.profile,
                "runs": profile.runs,
                "reasoning_effort": "low",
                "raw_request_and_response_saved": True,
                "verifier": "reused from llm_decode_bench.py",
            }
            for profile in arm.profiles
        ],
    }


def mark_probe_gates(rt: Any, probe: ProbePlan, receipt: dict[str, Any], rc: int) -> None:
    summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
    requested = expected_requests(probe.suite)
    runtime_ok = (
        rc == 0
        and summary.get("requested", summary.get("requested_turns")) == requested
        and summary.get("completed", summary.get("completed_turns")) == requested
        and summary.get("runtime_errors") == 0
    )
    if probe.suite in {"cache", "long"}:
        runtime_ok = runtime_ok and summary.get("metrics_errors", 0) == 0
    if probe.suite == "cache":
        runtime_ok = runtime_ok and summary.get("reset_errors", 0) == 0
    record_gate(
        rt,
        f"runtime:{probe.label}",
        runtime_ok,
        "runtime",
        {"return_code": rc, "summary": summary},
    )

    if probe.suite == "cache":
        quality_ok = (
            summary.get("correct") == requested
            and summary.get("wrong_answers") == 0
            and summary.get("repetitions") == 0
        )
        record_gate(
            rt,
            f"quality:{probe.label}:answers",
            quality_ok,
            "model_quality",
            summary,
        )
        effects = receipt.get("cache_effects") if isinstance(receipt.get("cache_effects"), dict) else {}
        external_expected = bool(
            (receipt.get("request_policy") or {}).get("external_cache_expected")
            if isinstance(receipt.get("request_policy"), dict)
            else False
        )
        hit_ok = effects.get(
            "external_hit_observed_on_both_replays"
            if external_expected
            else "local_hit_observed_on_both_replays"
        ) is True
        record_gate(
            rt,
            f"cache:{probe.label}:cold-to-hit",
            effects.get("cold_compute_observed") is True and hit_ok,
            "cache_effect",
            {"external_expected": external_expected, **effects},
        )
    elif probe.suite == "long":
        record_gate(
            rt,
            f"quality:{probe.label}:wrong-answers",
            summary.get("wrong_answers") == 0,
            "model_quality",
            summary,
        )
        record_gate(
            rt,
            f"quality:{probe.label}:repetition",
            summary.get("repetitions") == 0,
            "model_quality",
            summary,
        )
        record_gate(
            rt,
            f"quality:{probe.label}:twenty-four-long-generations",
            summary.get("requested") == 24 and summary.get("completed") == 24,
            "configuration",
            summary,
        )
    elif probe.suite == "sampling":
        record_gate(
            rt,
            f"quality:{probe.label}:observable-support",
            summary.get("valid_support") == requested,
            "model_quality",
            {
                "summary": summary,
                "claim_scope": receipt.get("claim_scope"),
                "settings": receipt.get("settings"),
            },
        )
        record_gate(
            rt,
            f"quality:{probe.label}:answer-verifier",
            summary.get("correct") == requested and summary.get("wrong_answers") == 0,
            "model_quality",
            summary,
        )
    elif probe.suite == "tool-order":
        record_gate(
            rt,
            f"quality:{probe.label}:parser-and-reordering",
            summary.get("parser_and_reordering_correct") is True,
            "model_quality",
            {
                "summary": summary,
                "semantic_checks": receipt.get("semantic_checks"),
                "first_call_order": receipt.get("first_call_order"),
            },
        )


def run_probe(rt: Any, probe: ProbePlan) -> dict[str, Any]:
    output = rt.ROOT / f"{probe.label}.json"
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(PROBE),
        "--base-url",
        rt.BASE_URL,
        "--model",
        rt.MODEL_NAME,
        "--label",
        probe.label,
        "--output",
        str(output),
        "--timeout",
        str(min(probe.timeout, 900)),
        probe.suite,
    ]
    if probe.suite == "cache":
        command.extend(["--bench-path", str(rt.BENCH)])
    command.extend(probe.arguments)
    try:
        rc = rt.run(
            command,
            label=probe.label,
            timeout=probe.timeout,
            env=rt.PROXY_ENV,
        )
    except Exception as exc:
        rc = 2
        receipt = save_failure_receipt(
            rt,
            probe.label,
            probe.suite,
            "runtime.run raised",
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
    else:
        if output.exists():
            receipt = load_json(output)
        else:
            receipt = save_failure_receipt(
                rt,
                probe.label,
                probe.suite,
                "probe produced no JSON receipt",
                {"return_code": rc, "command": command},
            )
    _RECEIPTS[probe.label] = receipt
    mark_probe_gates(rt, probe, receipt, rc)
    return receipt


def profile_settings(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    keys = (
        "model",
        "test_profile",
        "test_profile_description",
        "prompt_source",
        "prompt_chars",
        "max_tokens",
        "max_tokens_omitted",
        "fixed_concurrency",
        "requested_runs",
        "concurrency_levels_requested",
        "min_results",
        "probe_waves",
        "auto_stop",
        "correct_regex",
        "score_source",
        "profile_scorer",
        "expected_answer",
        "approx_tolerance",
        "dataset_rows",
        "dataset_sha256",
        "prompt_sha256",
        "prefill_scout",
        "temperature",
        "top_p",
        "reasoning_effort",
        "seed_base",
    )
    return {key: metadata.get(key) for key in keys}


def run_profile(rt: Any, profile: ProfilePlan) -> dict[str, Any]:
    output = rt.ROOT / f"{profile.label}.json"
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(PROBE),
        "--base-url",
        rt.BASE_URL,
        "--model",
        rt.MODEL_NAME,
        "--label",
        profile.label,
        "--output",
        str(output),
        "--timeout",
        "900",
        "profile",
        "--bench-path",
        str(rt.BENCH),
        "--profile-name",
        profile.profile,
        "--runs",
        str(profile.runs),
        "--seed-base",
        "26090700",
    ]
    try:
        return_code = rt.run(
            command,
            label=profile.label,
            timeout=3600,
            env=rt.PROXY_ENV,
        )
        ok = return_code == 0 and output.exists()
    except Exception as exc:
        return_code = 2
        ok = False
        data = save_failure_receipt(
            rt,
            profile.label,
            profile.profile,
            "explicit-low-reasoning raw-output profile run raised",
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
    else:
        if output.exists():
            data = load_json(output)
        else:
            data = save_failure_receipt(
                rt,
                profile.label,
                profile.profile,
                "explicit-low-reasoning raw-output profile produced no JSON receipt",
                {"return_code": return_code, "command": command},
            )
    _RECEIPTS[profile.label] = data
    summary = data.get("selected_summary") if isinstance(data.get("selected_summary"), dict) else {}
    runtime_ok = (
        ok
        and summary.get("attempted") == profile.runs
        and summary.get("completed") == profile.runs
        and summary.get("errors") == 0
    )
    record_gate(
        rt,
        f"runtime:{profile.label}",
        runtime_ok,
        "runtime",
        {
            "return_code": return_code,
            "raw_request_and_response_saved": True,
            "reasoning_effort": "low",
            "requested_runs": profile.runs,
            "summary": summary,
            "settings": profile_settings(data),
        },
    )
    scored = (
        summary.get("score_available") is True
        and summary.get("correct", 0) + summary.get("wrong", 0) == profile.runs
    )
    record_gate(
        rt,
        f"quality:{profile.label}:all-results-scored",
        scored,
        "model_quality",
        summary,
    )
    record_gate(
        rt,
        f"quality:{profile.label}:all-answers-correct",
        summary.get("correct") == profile.runs
        and summary.get("wrong") == 0
        and summary.get("wrong_answers") == 0
        and summary.get("repetitions") == 0,
        "model_quality",
        summary,
    )
    return data


def run_acceptance_probe(rt: Any, arm: Arm) -> dict[str, Any]:
    receipt_label = f"{arm.label}-acceptance"
    output = rt.ROOT / f"{receipt_label}.json"
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(ACCEPTANCE_PROBE),
        "--label",
        arm.label,
        "--contexts",
        "0,32768",
        "--concurrency",
        "1,8",
        "--duration",
        "30",
        "--repeats",
        "2",
    ]
    if not ACCEPTANCE_PROBE.exists():
        receipt = save_failure_receipt(
            rt,
            receipt_label,
            "acceptance",
            "Main-owned acceptance_probe.py is missing",
            {"script": str(ACCEPTANCE_PROBE), "command": command},
        )
        return_code = 127
    else:
        try:
            return_code = rt.run(
                command,
                label=receipt_label,
                timeout=1800,
                env=rt.PROXY_ENV,
            )
        except Exception as exc:
            return_code = 2
            receipt = save_failure_receipt(
                rt,
                receipt_label,
                "acceptance",
                "acceptance probe run raised",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "command": command,
                },
            )
        else:
            if output.exists():
                receipt = load_json(output)
            else:
                receipt = save_failure_receipt(
                    rt,
                    receipt_label,
                    "acceptance",
                    "acceptance probe produced no JSON receipt",
                    {"return_code": return_code, "command": command},
                )
    _RECEIPTS[receipt_label] = receipt
    readable = receipt.get("status") != "read_error"
    samples = receipt.get("samples")
    receipt_complete = (
        receipt.get("passed") is True
        and receipt.get("sample_count") == 8
        and isinstance(samples, list)
        and len(samples) == 8
    )
    record_gate(
        rt,
        f"runtime:{receipt_label}",
        return_code == 0 and output.exists() and readable and receipt_complete,
        "runtime",
        {
            "return_code": return_code,
            "result": str(output),
            "settings": {
                "contexts": [0, 32768],
                "concurrency": [1, 8],
                "duration_seconds": 30,
                "repeats": 2,
            },
            "receipt_passed": receipt.get("passed"),
            "sample_count": receipt.get("sample_count"),
            "metric_interpretation_owner": "Main",
        },
    )
    return receipt


def mark_arm_not_run(rt: Any, arm: Arm, reason: str, detail: object) -> None:
    if arm.acceptance_probe:
        receipt_label = f"{arm.label}-acceptance"
        receipt = save_failure_receipt(
            rt,
            receipt_label,
            "acceptance",
            reason,
            detail,
        )
        _RECEIPTS[receipt_label] = receipt
        record_gate(
            rt,
            f"runtime:{receipt_label}",
            False,
            "runtime",
            {"reason": reason, "detail": detail},
        )
    for probe in arm.probes:
        receipt = save_failure_receipt(rt, probe.label, probe.suite, reason, detail)
        mark_probe_gates(rt, probe, receipt, 2)
    for profile in arm.profiles:
        data = save_failure_receipt(rt, profile.label, profile.profile, reason, detail)
        _RECEIPTS[profile.label] = data
        record_gate(
            rt,
            f"runtime:{profile.label}",
            False,
            "runtime",
            {"reason": reason, "detail": detail},
        )
        record_gate(
            rt,
            f"quality:{profile.label}:all-results-scored",
            False,
            "model_quality",
            {"reason": reason},
        )
        record_gate(
            rt,
            f"quality:{profile.label}:all-answers-correct",
            False,
            "model_quality",
            {"reason": reason},
        )


def run_arm(rt: Any, arm: Arm) -> None:
    rt.save_json(f"{arm.label}-settings.json", arm_manifest(arm))
    booted = False
    try:
        try:
            booted = bool(
                rt.boot(
                    arm.label,
                    image=arm.image,
                    tp=4,
                    dcp=arm.dcp,
                    spec=arm.spec,
                    cache=arm.cache,
                    kv=arm.kv,
                    extra_env=arm.extra_env or None,
                    extra_args=list(arm.extra_args) or None,
                )
            )
        except Exception as exc:
            record_gate(
                rt,
                f"runtime:{arm.label}:boot",
                False,
                "runtime",
                {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
            )
            mark_arm_not_run(
                rt,
                arm,
                "boot raised",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        record_gate(
            rt,
            f"runtime:{arm.label}:boot",
            booted,
            "runtime",
            arm_manifest(arm)["runtime"],
        )
        if not booted:
            mark_arm_not_run(rt, arm, "boot returned false", arm_manifest(arm)["runtime"])
            return
        if arm.acceptance_probe:
            try:
                run_acceptance_probe(rt, arm)
            except Exception as exc:
                detail = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                receipt_label = f"{arm.label}-acceptance"
                save_failure_receipt(
                    rt,
                    receipt_label,
                    "acceptance",
                    "unexpected acceptance-probe cell failure",
                    detail,
                )
                record_gate(
                    rt,
                    f"runtime:{receipt_label}",
                    False,
                    "harness",
                    detail,
                )
        for probe in arm.probes:
            try:
                run_probe(rt, probe)
            except Exception as exc:
                detail = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                receipt = save_failure_receipt(
                    rt,
                    probe.label,
                    probe.suite,
                    "unexpected probe cell failure",
                    detail,
                )
                mark_probe_gates(rt, probe, receipt, 2)
                record_gate(
                    rt,
                    f"harness:{probe.label}:cell-boundary",
                    False,
                    "harness",
                    detail,
                )
        for profile in arm.profiles:
            try:
                run_profile(rt, profile)
            except Exception as exc:
                detail = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                save_failure_receipt(
                    rt,
                    profile.label,
                    profile.profile,
                    "unexpected profile cell failure",
                    detail,
                )
                record_gate(
                    rt,
                    f"runtime:{profile.label}",
                    False,
                    "runtime",
                    detail,
                )
                record_gate(
                    rt,
                    f"quality:{profile.label}:all-results-scored",
                    False,
                    "model_quality",
                    detail,
                )
                record_gate(
                    rt,
                    f"quality:{profile.label}:all-answers-correct",
                    False,
                    "model_quality",
                    detail,
                )
                record_gate(
                    rt,
                    f"harness:{profile.label}:cell-boundary",
                    False,
                    "harness",
                    detail,
                )
    except Exception as exc:
        record_gate(
            rt,
            f"runtime:{arm.label}:cell-boundary",
            False,
            "harness",
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        rt.save_json(
            f"{arm.label}-cell-error.json",
            {
                "schema": PROBE_SCHEMA,
                "arm": arm.label,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        try:
            rt.stop()
        except Exception as exc:
            record_gate(
                rt,
                f"runtime:{arm.label}:capture-and-stop",
                False,
                "runtime",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        else:
            record_gate(
                rt,
                f"runtime:{arm.label}:capture-and-stop",
                True,
                "runtime",
                "runtime.stop captured final logs/inspect/metrics and removed only the owned test container",
            )


def compact_probe(label: str) -> dict[str, Any]:
    receipt = _RECEIPTS.get(label, {})
    return {
        "label": label,
        "suite": receipt.get("suite"),
        "suite_fingerprint": receipt.get("suite_fingerprint"),
        "summary": receipt.get("summary"),
        "cache_effects": receipt.get("cache_effects"),
        "method": receipt.get("method") or receipt.get("request_policy"),
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def profile_observation(label: str) -> dict[str, Any]:
    data = _RECEIPTS.get(label, {})
    summary = data.get("selected_summary") if isinstance(data.get("selected_summary"), dict) else {}
    correct = int(summary.get("correct") or 0)
    attempted = int(summary.get("attempted") or 0)
    return {
        "label": label,
        "settings": profile_settings(data),
        "attempted": attempted,
        "completed": summary.get("completed"),
        "errors": summary.get("errors"),
        "correct": correct,
        "wrong": summary.get("wrong"),
        "correct_rate": correct / attempted if attempted else None,
        "wilson_95": wilson_interval(correct, attempted),
        "score_counts": summary.get("score_counts"),
    }


def same_nonempty_fingerprint(labels: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
    fingerprints = {label: _RECEIPTS.get(label, {}).get("suite_fingerprint") for label in labels}
    values = list(fingerprints.values())
    return bool(values and all(values) and len(set(values)) == 1), fingerprints


def historical_context() -> dict[str, Any]:
    pr646 = load_json(PRIOR_ROOT / "pr646-summary.json")
    lavd = load_json(PRIOR_ROOT / "lavd-aggregate.json")
    corruption = load_json(PRIOR_ROOT / "corruption-lmcache-dflash-aggregate.json")
    correctness = pr646.get("correctness") if isinstance(pr646.get("correctness"), dict) else {}
    flagged = corruption.get("flagged_results") if isinstance(corruption.get("flagged_results"), list) else []
    return {
        "sources": {
            "pr646": str(PRIOR_ROOT / "pr646-summary.json"),
            "lavd": str(PRIOR_ROOT / "lavd-aggregate.json"),
            "corruption": str(PRIOR_ROOT / "corruption-lmcache-dflash-aggregate.json"),
        },
        "source_errors": {
            name: data.get("error")
            for name, data in {
                "pr646": pr646,
                "lavd": lavd,
                "corruption": corruption,
            }.items()
            if data.get("status") == "read_error"
        },
        "comparison_rule": (
            "Historical counts are context only and are not pooled with R26. The new R25/R26 controls "
            "both send explicit reasoning_effort=low with identical fixed seeds; older receipts used "
            "their then-current benchmark/template defaults."
        ),
        "prior_pr646_estonia": {
            "patched": [correctness.get("estonia_correct"), correctness.get("estonia_total")],
            "stock": [correctness.get("stock_estonia_correct"), correctness.get("stock_estonia_total")],
            "interpretation": (
                "The prior 10/12 versus 12/12 observation is historical context, not proof that PR646 "
                "caused the misses. PR646 is not part of the D-Rock R26 overlay tested here."
            ),
        },
        "prior_r25_lavd": {
            "correct": lavd.get("correct"),
            "attempted": lavd.get("attempted"),
            "errors": lavd.get("errors"),
        },
        "prior_r25_dflash_corruption": {
            "clean": corruption.get("clean"),
            "total": corruption.get("total"),
            "flagged": corruption.get("flagged"),
            "flagged_results": flagged,
        },
    }


def build_comparisons(rt: Any) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}

    kv_labels = (
        "quality-r26-nospec-fp8-kv-cache",
        "quality-r26-nospec-nvfp4-kv-cache",
    )
    kv_same, kv_fingerprints = same_nonempty_fingerprint(kv_labels)
    comparisons["kv_dtype"] = {
        "question": "FP8 versus packed NVFP4 KV under the same no-spec prompt and sampling suite",
        "same_request_suite": kv_same,
        "fingerprints": kv_fingerprints,
        "arms": [compact_probe(label) for label in kv_labels],
        "interpretation_scope": (
            "Answer verifier and cold/local-hit transitions only; proposal-head and weights are unchanged."
        ),
    }
    rt.save_json("quality-kv-dtype-comparison.json", comparisons["kv_dtype"])
    record_gate(
        rt,
        "configuration:quality-kv-dtype:same-request-suite",
        kv_same,
        "configuration",
        kv_fingerprints,
    )

    head_labels = (
        "quality-r26-mtp3-default-nvfp4-long",
        "quality-r26-mtp3-bf16-long",
    )
    heads_same, head_fingerprints = same_nonempty_fingerprint(head_labels)
    comparisons["mtp_proposal_heads"] = {
        "question": "R26 default NVFP4 proposal head versus explicit BF16 proposal-head control",
        "same_request_suite": heads_same,
        "fingerprints": head_fingerprints,
        "constant_conditions": {"tp": 4, "dcp": 4, "spec": "mtp3", "kv": "fp8_ds_mla"},
        "arms": [compact_probe(label) for label in head_labels],
        "claim_scope": "Correctness and corruption outcomes; Main owns output/verifier-step performance.",
    }
    rt.save_json("quality-mtp3-head-comparison.json", comparisons["mtp_proposal_heads"])
    record_gate(
        rt,
        "configuration:quality-mtp3-heads:same-request-suite",
        heads_same,
        "configuration",
        head_fingerprints,
    )

    overlay_long_labels = (
        "quality-r25-mtp3-bf16-long",
        "quality-r26-mtp3-bf16-long",
        "quality-overlay-mtp3-bf16-long",
    )
    overlay_cache_labels = (
        "quality-r25-mtp3-bf16-cache",
        "quality-r26-mtp3-bf16-cache",
        "quality-overlay-mtp3-bf16-cache",
    )
    long_same, long_fingerprints = same_nonempty_fingerprint(overlay_long_labels)
    cache_same, cache_fingerprints = same_nonempty_fingerprint(overlay_cache_labels)
    comparisons["overlay_mtp3_dcp4"] = {
        "question": "Baseline R25 versus stock R26 versus D-Rock overlay MTP3/DCP4 correctness",
        "fixed_conditions": {
            "tp": 4,
            "dcp": 4,
            "spec": "mtp3",
            "proposal_head": "bf16",
            "kv": "fp8_ds_mla",
            "prompts_and_sampling": "fingerprinted identical suites",
        },
        "long_suite_same": long_same,
        "long_fingerprints": long_fingerprints,
        "cache_suite_same": cache_same,
        "cache_fingerprints": cache_fingerprints,
        "long_arms": [compact_probe(label) for label in overlay_long_labels],
        "cache_arms": [compact_probe(label) for label in overlay_cache_labels],
        "attribution_limit": (
            "The overlay is a combined candidate. Field differences cannot be attributed to #561 or any "
            "other included change in isolation."
        ),
    }
    rt.save_json("quality-overlay-mtp3-dcp4-comparison.json", comparisons["overlay_mtp3_dcp4"])
    record_gate(
        rt,
        "configuration:quality-overlay-mtp3-dcp4:same-prompts",
        long_same and cache_same,
        "configuration",
        {"long": long_fingerprints, "cache": cache_fingerprints},
    )

    dflash_history_labels = (
        "quality-r25-dflash-lmcache-history-long",
        "quality-r26-dflash-lmcache-history-long",
    )
    dflash_history_same, dflash_history_fingerprints = same_nonempty_fingerprint(
        dflash_history_labels
    )
    comparisons["dflash_lmcache_history_control"] = {
        "question": "Matched recheck of the prior R25 DFlash/LMCache Apollo runaway",
        "fixed_conditions": {
            "tp": 4,
            "dcp": 4,
            "spec": "DFlash2 K7",
            "cache": "lmcache",
            "kv": "nvfp4_ds_mla",
            "reasoning_effort": "low",
            "waves": 3,
            "long_generations_per_release": 24,
            "apollo_family_generations_per_release": 12,
        },
        "same_request_suite": dflash_history_same,
        "fingerprints": dflash_history_fingerprints,
        "arms": [compact_probe(label) for label in dflash_history_labels],
        "claim_scope": (
            "Observed R25/R26 behavior under matched settings. A different count does not by itself "
            "identify which release component caused the difference."
        ),
    }
    rt.save_json(
        "quality-dflash-lmcache-history-comparison.json",
        comparisons["dflash_lmcache_history_control"],
    )
    record_gate(
        rt,
        "configuration:quality-dflash-lmcache-history:same-prompts",
        dflash_history_same,
        "configuration",
        dflash_history_fingerprints,
    )

    profile_pairs = {
        "estonia": ("quality-r25-dflash-estonia", "quality-r26-dflash-estonia"),
        "lavd-test": ("quality-r25-dflash-lavd", "quality-r26-dflash-lavd"),
    }
    profile_comparison: dict[str, Any] = {
        "conditions": {"tp": 4, "dcp": 1, "spec": "DFlash2 K7", "kv": "fp8_ds_mla"},
        "runs_per_release_and_profile": PROFILE_RUNS,
        "causal_claim": (
            "None. These repeated matched observations can identify an observed release regression but do "
            "not assign causality to PR646, cache code, or a proposal component."
        ),
        "profiles": {},
    }
    for profile_name, labels in profile_pairs.items():
        observations = [profile_observation(label) for label in labels]
        settings_match = (
            observations[0]["attempted"] == PROFILE_RUNS
            and observations[1]["attempted"] == PROFILE_RUNS
            and observations[0]["settings"].get("test_profile") == profile_name
            and observations[0]["settings"] == observations[1]["settings"]
        )
        profile_comparison["profiles"][profile_name] = {
            "settings_match": settings_match,
            "r25": observations[0],
            "r26": observations[1],
            "observed_correct_rate_delta_r26_minus_r25": (
                observations[1]["correct_rate"] - observations[0]["correct_rate"]
                if observations[0]["correct_rate"] is not None
                and observations[1]["correct_rate"] is not None
                else None
            ),
        }
        record_gate(
            rt,
            f"configuration:quality-profile:{profile_name}:matched-settings",
            settings_match,
            "configuration",
            {label: observation["settings"] for label, observation in zip(labels, observations)},
        )
    comparisons["profiles"] = profile_comparison
    rt.save_json("quality-dflash-profile-comparison.json", profile_comparison)
    return comparisons


def write_static_manifests(rt: Any, plan: tuple[Arm, ...]) -> None:
    rt.save_json(
        "quality-plan.json",
        {
            "schema": PROBE_SCHEMA,
            "phase": "quality",
            "arms": [arm_manifest(arm) for arm in plan],
            "proposal_head_control": {
                "environment_variable": MTP_HEAD_ENV,
                "allowed_values": ["bf16", "nvfp4"],
                "official_R26_image_default": "nvfp4",
                "bf16_control": {MTP_HEAD_ENV: "bf16"},
                "verification": {
                    "image": rt.IMAGE,
                    "image_config_env": f"{MTP_HEAD_ENV}=nvfp4",
                    "locked_vllm_commit": "7f53b30481b4110f293521d09f7af18f9e8f9d9e",
                    "locked_source": (
                        "https://raw.githubusercontent.com/local-inference-lab/vllm/"
                        "7f53b30481b4110f293521d09f7af18f9e8f9d9e/vllm/envs.py"
                    ),
                    "locked_type": "Literal['bf16', 'nvfp4']",
                },
            },
            "weight_policy": {
                "target": str(rt.MODEL),
                "draft": str(rt.DRAFT),
                "checkpoint_cache_tag": rt.MODEL_CACHE_TAG,
                "upgrade_weights_in_place": False,
                "comparison_scope": "All phase arms use runtime.MODEL; cross-checkpoint comparisons are explicit separate matched phases.",
            },
            "storage_scope": {
                "test_L2": "runtime-owned isolated capped test root",
                "production_L2_touched": False,
                "filesystem": "local NVMe; iSource's SATA restoration case is not reproduced or claimed",
            },
        },
    )
    rt.save_json(
        "quality-overlay-scope.json",
        {
            "schema": PROBE_SCHEMA,
            "candidate": rt.OVERLAY_IMAGE,
            "base": rt.IMAGE,
            "source_revision": "7db6a2d2f5680513ae1a396ff61169c4cacf8a95",
            "label": "D-Rock R26 Python overlay; not stock R26",
            "included": [
                {"pr": 561, "scope": "fused speculative DSA draft-metadata refresh"},
                {"pr": 574, "scope": "remaining recurrent-source publication delta not already in R26"},
                {"pr": 643, "scope": "sparse hybrid replay-boundary retention"},
                {"pr": 645, "scope": "truthful sparse BlockStored event publication"},
                {"pr": 648, "scope": "automatic compute sharing and interleaved prefills"},
                {"pr": 599, "scope": "direct-DCP peer-access guard; physical-path validation owned by Main"},
            ],
            "cache_changes_actually_included": [
                "remaining #574 delta not already in stock R26",
                "#643 sparse hybrid replay-boundary retention",
                "#645 sparse cache-event publication",
            ],
            "explicitly_not_included_or_not_attributed": {
                "PR646": "not this overlay; prior 10/12 Estonia observation is historical context only",
                "individual_patch_causality": "combined-image field results cannot isolate one included PR",
                "native_rebuild": "none; Python-only overlay",
            },
            "quality_checks": {
                "mtp3_dcp4": "fixed BF16 proposal head, FP8 KV, same prompts versus stock R26 and R25",
                "lmcache": (
                    "separate MTP3/DCP4/NVFP4-KV cold then local-reset external-hit correctness receipt"
                ),
            },
        },
    )


def main() -> int:
    rt = load_runtime()
    rt.ROOT.mkdir(parents=True, exist_ok=True)
    _LOCAL_GATES.clear()
    _RECEIPTS.clear()
    plan = build_plan(rt)
    write_static_manifests(rt, plan)
    rt.note("R26 QUALITY PHASE START")
    for arm in plan:
        rt.note(f"QUALITY ARM START {arm.label}")
        run_arm(rt, arm)
        rt.note(f"QUALITY ARM END {arm.label}")

    comparisons = build_comparisons(rt)
    history = historical_context()
    rt.save_json("quality-historical-context.json", history)
    history_ok = not history.get("source_errors")
    record_gate(
        rt,
        "configuration:quality:historical-receipts-loaded",
        history_ok,
        "configuration",
        history,
    )

    failures = [gate for gate in _LOCAL_GATES if not gate["passed"]]
    blocking_failures = [
        gate
        for gate in failures
        if gate["gate_type"] in {"runtime", "harness", "configuration"}
    ]
    summary = {
        "schema": PROBE_SCHEMA,
        "phase": "quality",
        "completed": True,
        "receipts": sorted(_RECEIPTS),
        "gates": _LOCAL_GATES,
        "gate_counts": {
            "total": len(_LOCAL_GATES),
            "passed": len(_LOCAL_GATES) - len(failures),
            "failed": len(failures),
            "blocking_runtime_harness_configuration": len(blocking_failures),
            "failed_by_type": {
                gate_type: sum(
                    not gate["passed"] and gate["gate_type"] == gate_type
                    for gate in _LOCAL_GATES
                )
                for gate_type in sorted({gate["gate_type"] for gate in _LOCAL_GATES})
            },
        },
        "comparisons": comparisons,
        "historical_context": history,
        "interpretation": {
            "wrong_answers_are_not_runtime_errors": True,
            "repetition_is_separate_from_wrong_answers": True,
            "cache_effects_have_separate_gates": True,
            "no_patch_causality_from_small_samples": True,
            "QAD_weight_change_excluded": True,
            "performance_owned_by_main_matrix": True,
        },
    }
    rt.save_json("quality-summary.json", summary)
    rt.note(
        "R26 QUALITY PHASE DONE "
        f"({len(_LOCAL_GATES) - len(failures)}/{len(_LOCAL_GATES)} gates passed; "
        f"{len(blocking_failures)} runtime/harness/configuration failures)"
    )
    return int(bool(blocking_failures))


if __name__ == "__main__":
    raise SystemExit(main())
