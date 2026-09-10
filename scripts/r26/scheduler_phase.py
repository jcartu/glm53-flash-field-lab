#!/usr/bin/env python3
"""Serial R26 mixed-agent scheduler qualification phase.

This phase keeps the scheduler-only series in VRAM, runs one separately labelled
LMCache confirmation, and leaves quick decode/prefill batch and DMA measurements
to the core matrix. It records every unsupported capability as a failed gate;
it never relabels an old fairness API as the native interleaving API.
"""
from __future__ import annotations

import json
import math
import re
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from . import runtime
except ImportError:
    import runtime  # type: ignore[no-redef]


WORKLOAD = Path(__file__).with_name("agent_workload.py")
SOURCE_GUIDE = Path(
    "/home/josh/omp-workspace/drock-lmcache/"
    "release-review-20260905T154745Z/attachment-1545754297390862357-0.txt"
)
TRACE_SEED = "r26-mixed-agent-v1"
NATIVE_FIELDS = (
    "prefill_compute_share",
    "prefill_compute_half_life",
    "max_parallel_prefills",
    "prefill_interleave_policy",
    "decode_reservoir_low_watermark",
)
LEGACY_FIELDS = (
    "fairness_engine",
    "prefill_compute_share",
    "max_num_prefill_tokens_per_step",
    "max_num_partial_prefills",
    "decode_prefill_min_decode_steps",
    "decode_prefill_max_wait_ms",
)
PROFILE_SECONDS = {
    "baseline": 30.0,
    "periodic-128k": 60.0,
    "short-prefill-heavy": 60.0,
    "analytics-200k-burst": 90.0,
}
BASE_ENV = {
    "FAIRNESS_ENGINE": "none",
    "PREFILL_COMPUTE_SHARE": "0.4",
    "PREFILL_SCHEDULE_INTERVAL": "1",
    "MAX_NUM_SEQS": "32",
    "MAX_NUM_BATCHED_TOKENS": "4096",
}
HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    image: str
    expected_capability: str


@dataclass(frozen=True)
class Policy:
    key: str
    compute_share: float | str | None
    half_life: float | str | None
    lanes: int
    interleave: str
    watermark: int


@dataclass(frozen=True)
class WorkloadPlan:
    name: str
    policy: Policy
    profiles: tuple[str, ...]
    concurrencies: tuple[int, ...]
    repeats: int
    cache_namespace: str
    series: str
    headline: bool = False
    updates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Factor:
    name: str
    batch_tokens: int
    dma_value: str | None


OFF = Policy("off", None, None, 1, "fcfs", 0)
STATIC_04 = Policy("static0.4", 0.4, None, 1, "fcfs", 0)
STATIC_06 = Policy("static0.6", 0.6, None, 1, "fcfs", 0)
STATIC_07 = Policy("static0.7", 0.7, None, 1, "fcfs", 0)
AUTO_RESPONSIVE_FCFS = Policy("auto-responsive-fcfs", "auto", "responsive", 1, "fcfs", 0)
AUTO_RESPONSIVE_RR = Policy(
    "auto-responsive-round-robin", "auto", "responsive", 4, "round-robin", 0
)
AUTO_RESPONSIVE_SR = Policy(
    "auto-responsive-shortest-remaining",
    "auto",
    "responsive",
    4,
    "shortest-remaining",
    0,
)
AUTO_RESPONSIVE_DA = Policy(
    "auto-responsive-decode-aware", "auto", "responsive", 4, "decode-aware", 4
)
AUTO_SMOOTH_DA = Policy(
    "auto-smooth-decode-aware", "auto", "smooth", 4, "decode-aware", 4
)

OFFICIAL = Candidate(
    key="official-r26",
    label="Official R26",
    image=runtime.IMAGE,
    expected_capability="probe; official source lock does not claim PR647/PR648",
)
OVERLAY = Candidate(
    key="drock-r26-overlay",
    label="D-Rock R26 overlay (not stock R26)",
    image=runtime.OVERLAY_IMAGE,
    expected_capability="native five-field PR647/PR648 API required",
)

OFFICIAL_PLANS = (
    WorkloadPlan(
        "official-off-headline",
        OFF,
        ("baseline", "periodic-128k"),
        (8, 16),
        2,
        "headline-official-off",
        "vram-headline",
        True,
    ),
    WorkloadPlan(
        "official-static04-headline",
        STATIC_04,
        ("baseline", "periodic-128k"),
        (8, 16),
        2,
        "headline-official-static04",
        "vram-headline",
        True,
    ),
    WorkloadPlan(
        "official-static06-sweep",
        STATIC_06,
        ("periodic-128k",),
        (8, 16),
        1,
        "share-static06",
        "vram-share-sweep",
    ),
    WorkloadPlan(
        "official-static07-sweep",
        STATIC_07,
        ("periodic-128k",),
        (8, 16),
        1,
        "share-static07",
        "vram-share-sweep",
    ),
    WorkloadPlan(
        "official-off-profile-shapes",
        OFF,
        ("short-prefill-heavy", "analytics-200k-burst"),
        (8, 16),
        1,
        "profile-shapes-official-off",
        "vram-profile-shapes",
    ),
)

OVERLAY_PLANS = (
    WorkloadPlan(
        "overlay-recommended-headline",
        AUTO_RESPONSIVE_DA,
        ("baseline", "periodic-128k"),
        (8, 16),
        2,
        "headline-overlay-recommended",
        "vram-headline",
        True,
    ),
    WorkloadPlan(
        "overlay-off-sweep",
        OFF,
        ("periodic-128k",),
        (8, 16),
        1,
        "share-overlay-off",
        "vram-share-sweep",
    ),
    WorkloadPlan(
        "overlay-static04-sweep",
        STATIC_04,
        ("periodic-128k",),
        (8, 16),
        1,
        "share-static04",
        "vram-share-sweep",
    ),
    WorkloadPlan(
        "overlay-static06-sweep",
        STATIC_06,
        ("periodic-128k",),
        (8, 16),
        1,
        "share-static06",
        "vram-share-sweep",
    ),
    WorkloadPlan(
        "overlay-static07-sweep",
        STATIC_07,
        ("periodic-128k",),
        (8, 16),
        1,
        "share-static07",
        "vram-share-sweep",
    ),
    WorkloadPlan(
        "overlay-auto-smooth",
        AUTO_SMOOTH_DA,
        ("periodic-128k",),
        (8, 16),
        1,
        "auto-smooth",
        "vram-auto-sweep",
    ),
    WorkloadPlan(
        "overlay-policy-fcfs",
        AUTO_RESPONSIVE_FCFS,
        ("periodic-128k",),
        (8, 16),
        1,
        "policy-fcfs",
        "vram-interleave-sweep",
    ),
    WorkloadPlan(
        "overlay-policy-round-robin",
        AUTO_RESPONSIVE_RR,
        ("periodic-128k",),
        (8, 16),
        1,
        "policy-round-robin",
        "vram-interleave-sweep",
    ),
    WorkloadPlan(
        "overlay-policy-shortest-remaining",
        AUTO_RESPONSIVE_SR,
        ("periodic-128k",),
        (8, 16),
        1,
        "policy-shortest-remaining",
        "vram-interleave-sweep",
    ),
    WorkloadPlan(
        "overlay-recommended-profile-shapes",
        AUTO_RESPONSIVE_DA,
        ("short-prefill-heavy", "analytics-200k-burst"),
        (8, 16),
        1,
        "profile-shapes-overlay-recommended",
        "vram-profile-shapes",
    ),
    WorkloadPlan(
        "overlay-off-profile-shapes",
        OFF,
        ("short-prefill-heavy", "analytics-200k-burst"),
        (8, 16),
        1,
        "profile-shapes-overlay-off",
        "vram-profile-shapes",
    ),
    WorkloadPlan(
        "overlay-live-policy-updates",
        AUTO_RESPONSIVE_DA,
        ("periodic-128k",),
        (16,),
        1,
        "live-updates",
        "vram-live-update",
        updates=(
            {
                "offset_seconds": 10.0,
                "config": {
                    "prefill_compute_share": 0.6,
                    "prefill_compute_half_life": None,
                    "max_parallel_prefills": 4,
                    "prefill_interleave_policy": "round-robin",
                    "decode_reservoir_low_watermark": 0,
                },
            },
            {
                "offset_seconds": 25.0,
                "config": {
                    "prefill_compute_share": "auto",
                    "prefill_compute_half_life": "smooth",
                    "max_parallel_prefills": 4,
                    "prefill_interleave_policy": "shortest-remaining",
                    "decode_reservoir_low_watermark": 0,
                },
            },
            {
                "offset_seconds": 40.0,
                "config": {
                    "prefill_compute_share": "auto",
                    "prefill_compute_half_life": "responsive",
                    "max_parallel_prefills": 4,
                    "prefill_interleave_policy": "decode-aware",
                    "decode_reservoir_low_watermark": 4,
                },
            },
        ),
    ),
)

FACTORS = (
    Factor("overlay-mixed-bt4096-default-dma", 4096, None),
    Factor("overlay-mixed-bt8192-default-dma", 8192, None),
    Factor("overlay-mixed-bt12288-default-dma", 12288, None),
    Factor("overlay-mixed-bt16384-default-dma", 16384, None),
    Factor("overlay-mixed-bt4096-dma512k", 4096, "512KB"),
    Factor("overlay-mixed-bt12288-dma512k", 12288, "512KB"),
)


class PhaseState:
    def __init__(self) -> None:
        self.invocations: dict[str, dict[str, Any]] = {}
        self.runtime_facts: dict[str, dict[str, Any]] = {}
        self.api_discoveries: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, str]] = []
        self.lmcache_same_root: bool | None = None


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def policy_native(policy: Policy) -> dict[str, Any]:
    return {
        "prefill_compute_share": policy.compute_share,
        "prefill_compute_half_life": policy.half_life,
        "max_parallel_prefills": policy.lanes,
        "prefill_interleave_policy": policy.interleave,
        "decode_reservoir_low_watermark": policy.watermark,
    }


def detect_api_schema(config: object) -> str:
    if not isinstance(config, dict):
        return "unknown"
    if all(field in config for field in NATIVE_FIELDS):
        return "native"
    if "fairness_engine" in config:
        return "legacy"
    return "unknown"


def extract_config(exchange: dict[str, Any]) -> dict[str, Any] | None:
    body = exchange.get("body")
    if not isinstance(body, dict):
        return None
    if isinstance(body.get("config"), dict):
        return body["config"]
    return body


def http_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{runtime.BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with HTTP_OPENER.open(request, timeout=timeout) as response:
            raw = response.read().decode(errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        status = error.code
    except Exception as error:
        return {
            "method": method,
            "url": f"{runtime.BASE_URL}{path}",
            "request": payload,
            "status_code": None,
            "body": None,
            "body_text": "",
            "error": f"{type(error).__name__}: {error}",
        }
    try:
        body: object = json.loads(raw)
    except json.JSONDecodeError:
        body = None
    return {
        "method": method,
        "url": f"{runtime.BASE_URL}{path}",
        "request": payload,
        "status_code": status,
        "body": body,
        "body_text": raw,
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


def config_mismatches(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
) -> dict[str, dict[str, object]]:
    if actual is None:
        return {"config": {"expected": expected, "actual": None}}
    result: dict[str, dict[str, object]] = {}
    for field, wanted in expected.items():
        observed = actual.get(field)
        if not values_equal(observed, wanted):
            result[field] = {"expected": wanted, "actual": observed}
    return result


def compatible_payload(schema: str, policy: Policy) -> tuple[dict[str, Any] | None, str | None]:
    if schema == "native":
        return policy_native(policy), None
    if schema != "legacy":
        return None, f"unrecognized policy API schema {schema!r}"
    if policy.compute_share == "auto":
        return None, "legacy fairness selector has no automatic compute-share mode"
    if policy.interleave != "fcfs" or policy.lanes != 1 or policy.watermark != 0:
        return None, "legacy fairness selector has no prefill interleaving policy or lanes"
    return {
        "fairness_engine": (
            None if policy.compute_share is None else "compute_share"
        ),
        "prefill_compute_share": policy.compute_share,
        "max_num_prefill_tokens_per_step": 0,
        "max_num_partial_prefills": 0,
        "decode_prefill_min_decode_steps": 0,
        "decode_prefill_max_wait_ms": 0,
    }, None


def record_gate(name: str, passed: bool, detail: object) -> None:
    runtime.record_gate(name, bool(passed), detail)


def discover_api(candidate: Candidate, label: str) -> dict[str, Any]:
    exchange = http_json("GET", "/prefill_fairness")
    config = extract_config(exchange)
    schema = detect_api_schema(config) if exchange.get("status_code") == 200 else "unavailable"
    discovery = {
        "candidate": asdict(candidate),
        "schema": schema,
        "exchange": exchange,
        "native_fields": list(NATIVE_FIELDS),
        "legacy_fields": list(LEGACY_FIELDS),
    }
    runtime.save_json(f"scheduler/{safe_name(label)}-api-discovery.json", discovery)
    recognized = schema in {"native", "legacy"}
    record_gate(
        f"scheduler-{safe_name(label)}-api-recognized",
        recognized,
        discovery,
    )
    if candidate.key == OVERLAY.key:
        record_gate(
            f"scheduler-{safe_name(label)}-native-api-required",
            schema == "native",
            {
                "expected": "native",
                "observed": schema,
                "reason": "overlay PR647/PR648 cells require the five-field native API",
            },
        )
    return discovery


def check_native_atomic_rejection(label: str) -> dict[str, Any]:
    before = http_json("GET", "/prefill_fairness")
    before_config = extract_config(before)
    invalid = {
        "prefill_compute_share": None,
        "prefill_compute_half_life": None,
        "max_parallel_prefills": 4,
        "prefill_interleave_policy": "round-robin",
        "decode_reservoir_low_watermark": 4,
    }
    posted = http_json("POST", "/prefill_fairness", invalid)
    after = http_json("GET", "/prefill_fairness")
    after_config = extract_config(after)
    before_configured = (
        {field: before_config.get(field) for field in NATIVE_FIELDS}
        if before_config is not None
        else None
    )
    after_configured = (
        {field: after_config.get(field) for field in NATIVE_FIELDS}
        if after_config is not None
        else None
    )
    passed = (
        detect_api_schema(before_config) == "native"
        and posted.get("status_code") == 422
        and before_configured == after_configured
    )
    receipt = {
        "request_semantics": "complete five-field replacement, intentionally invalid",
        "before": before,
        "post": posted,
        "after": after,
        "before_configured": before_configured,
        "after_configured": after_configured,
        "passed": passed,
    }
    runtime.save_json(f"scheduler/{label}-atomic-rejection.json", receipt)
    record_gate(
        f"scheduler-{label}-native-post-atomic-rejection",
        passed,
        receipt,
    )
    return receipt


def configure_policy(
    candidate: Candidate,
    plan: WorkloadPlan,
) -> dict[str, Any]:
    before = http_json("GET", "/prefill_fairness")
    before_config = extract_config(before)
    schema = (
        detect_api_schema(before_config)
        if before.get("status_code") == 200
        else "unavailable"
    )
    payload, unsupported = compatible_payload(schema, plan.policy)
    receipt: dict[str, Any] = {
        "candidate": asdict(candidate),
        "plan": plan_to_json(plan),
        "schema": schema,
        "before": before,
        "payload": payload,
        "unsupported": unsupported,
    }
    if payload is None:
        receipt["passed"] = False
        runtime.save_json(f"scheduler/{plan.name}-policy.json", receipt)
        record_gate(
            f"scheduler-{plan.name}-policy-effective",
            False,
            receipt,
        )
        return receipt
    posted = http_json("POST", "/prefill_fairness", payload)
    after = http_json("GET", "/prefill_fairness")
    after_config = extract_config(after)
    after_schema = (
        detect_api_schema(after_config)
        if after.get("status_code") == 200
        else "unavailable"
    )
    mismatches = config_mismatches(after_config, payload)
    post_body = posted.get("body")
    applied = (
        posted.get("status_code") == 200
        and isinstance(post_body, dict)
        and post_body.get("applied") is True
    )
    passed = applied and after_schema == schema and not mismatches
    receipt.update(
        {
            "post": posted,
            "after": after,
            "after_schema": after_schema,
            "effective_readback": after_config,
            "mismatches": mismatches,
            "applied": applied,
            "passed": passed,
        }
    )
    runtime.save_json(f"scheduler/{plan.name}-policy.json", receipt)
    record_gate(
        f"scheduler-{plan.name}-policy-effective",
        passed,
        {
            "schema": schema,
            "request": payload,
            "readback": after_config,
            "mismatches": mismatches,
            "http_status": posted.get("status_code"),
            "applied": applied,
        },
    )
    return receipt


def plan_measurement_seconds(plan: WorkloadPlan) -> float:
    return plan.repeats * len(plan.concurrencies) * sum(
        PROFILE_SECONDS[profile] for profile in plan.profiles
    )


def plan_scenario_count(plan: WorkloadPlan) -> int:
    return plan.repeats * len(plan.concurrencies) * len(plan.profiles)


def plan_to_json(plan: WorkloadPlan) -> dict[str, Any]:
    return {
        "name": plan.name,
        "policy": asdict(plan.policy),
        "profiles": list(plan.profiles),
        "concurrencies": list(plan.concurrencies),
        "repeats": plan.repeats,
        "cache_namespace": plan.cache_namespace,
        "series": plan.series,
        "headline": plan.headline,
        "updates": list(plan.updates),
        "scenario_count": plan_scenario_count(plan),
        "measurement_seconds": plan_measurement_seconds(plan),
    }


def workload_command(
    candidate: Candidate,
    plan: WorkloadPlan,
    schema: str,
    policy_payload: dict[str, Any],
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(WORKLOAD),
        "--base-url",
        runtime.BASE_URL,
        "--model",
        runtime.MODEL_NAME,
        "--candidate-label",
        candidate.label,
        "--policy-label",
        plan.policy.key,
        "--expected-api-schema",
        schema,
        "--expected-policy-json",
        json.dumps(policy_payload, separators=(",", ":")),
        "--policy-updates-json",
        json.dumps(list(plan.updates), separators=(",", ":")),
        "--profiles",
        ",".join(plan.profiles),
        "--concurrencies",
        ",".join(str(value) for value in plan.concurrencies),
        "--repeats",
        str(plan.repeats),
        "--trace-seed",
        TRACE_SEED,
        "--cache-namespace",
        plan.cache_namespace,
        "--session-context-tokens",
        "8192",
        "--turn-decode-tokens",
        "128",
        "--incremental-filler-tokens",
        "220",
        "--hot-turn-period-seconds",
        "6",
        "--baseline-seconds",
        str(PROFILE_SECONDS["baseline"]),
        "--periodic-seconds",
        str(PROFILE_SECONDS["periodic-128k"]),
        "--short-heavy-seconds",
        str(PROFILE_SECONDS["short-prefill-heavy"]),
        "--analytics-seconds",
        str(PROFILE_SECONDS["analytics-200k-burst"]),
        "--periodic-cold-tokens",
        "131072",
        "--cold-period-seconds",
        "15",
        "--short-prefill-tokens",
        "2048,4096,8192",
        "--short-prefill-period-seconds",
        "1.5",
        "--analytics-cold-tokens",
        "204800",
        "--analytics-offsets-seconds",
        "0,1,2",
        "--drain-timeout-seconds",
        "240",
        "--request-timeout-seconds",
        "300",
        "--output",
        str(output),
    ]


def read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, f"expected JSON object, got {type(value).__name__}"
    return value, None


def find_cell_gate(cell: dict[str, Any], name: str) -> dict[str, Any] | None:
    for gate in cell.get("gates", []):
        if isinstance(gate, dict) and gate.get("name") == name:
            return gate
    return None


def run_workload_plan(
    state: PhaseState,
    candidate: Candidate,
    plan: WorkloadPlan,
    cache_mode: str,
) -> dict[str, Any]:
    runtime.note(
        f"SCHEDULER CELL START {plan.name} candidate={candidate.label} "
        f"cache={cache_mode} policy={plan.policy.key}"
    )
    configuration = configure_policy(candidate, plan)
    record: dict[str, Any] = {
        "candidate": asdict(candidate),
        "plan": plan_to_json(plan),
        "cache_mode": cache_mode,
        "configuration": configuration,
    }
    state.invocations[plan.name] = record
    if not configuration.get("passed"):
        record["status"] = "unsupported_or_configuration_failed"
        return record
    output = runtime.ROOT / "scheduler" / f"{plan.name}.json"
    schema = str(configuration["schema"])
    payload = dict(configuration["payload"])
    timeout = int(
        plan_measurement_seconds(plan)
        + plan_scenario_count(plan) * (240.0 + 300.0)
        + 900.0
    )
    return_code = runtime.run(
        workload_command(candidate, plan, schema, payload, output),
        label=plan.name,
        timeout=timeout,
        env=runtime.PROXY_ENV,
    )
    parsed, parse_error = read_json_file(output)
    cells = parsed.get("cells", []) if parsed else []
    expected_cells = plan_scenario_count(plan)
    complete_cells = [
        cell for cell in cells if isinstance(cell, dict) and cell.get("status") == "complete"
    ]
    no_lost_failures = []
    update_failures = []
    for cell in cells:
        if not isinstance(cell, dict):
            no_lost_failures.append({"cell": cell, "reason": "not an object"})
            continue
        gate = find_cell_gate(cell, "no_lost_or_corrupt_responses")
        if not gate or gate.get("passed") is not True:
            no_lost_failures.append(
                {
                    "profile": cell.get("profile"),
                    "concurrency": cell.get("concurrency"),
                    "repeat": cell.get("repeat"),
                    "gate": gate,
                }
            )
        if plan.updates:
            update_gate = find_cell_gate(
                cell, "live_policy_updates_applied_during_active_work"
            )
            if not update_gate or update_gate.get("passed") is not True:
                update_failures.append(
                    {
                        "profile": cell.get("profile"),
                        "concurrency": cell.get("concurrency"),
                        "repeat": cell.get("repeat"),
                        "gate": update_gate,
                    }
                )
    workload_passed = (
        return_code == 0
        and parse_error is None
        and len(cells) == expected_cells
        and len(complete_cells) == expected_cells
    )
    no_lost_passed = len(cells) == expected_cells and not no_lost_failures
    record.update(
        {
            "status": "complete" if workload_passed else "failed",
            "output": str(output),
            "log": str(runtime.ROOT / f"{plan.name}.log"),
            "return_code": return_code,
            "parse_error": parse_error,
            "expected_cells": expected_cells,
            "observed_cells": len(cells),
            "complete_cells": len(complete_cells),
            "no_lost_failures": no_lost_failures,
            "update_failures": update_failures,
            "result": parsed,
        }
    )
    record_gate(
        f"scheduler-{plan.name}-workload-complete",
        workload_passed,
        {
            "return_code": return_code,
            "parse_error": parse_error,
            "expected_cells": expected_cells,
            "observed_cells": len(cells),
            "complete_cells": len(complete_cells),
            "output": str(output),
        },
    )
    record_gate(
        f"scheduler-{plan.name}-no-lost-or-corrupt",
        no_lost_passed,
        {
            "expected_cells": expected_cells,
            "failures": no_lost_failures,
        },
    )
    if plan.updates:
        record_gate(
            f"scheduler-{plan.name}-live-updates",
            len(cells) == expected_cells and not update_failures,
            {
                "expected_updates_per_cell": len(plan.updates),
                "failures": update_failures,
            },
        )
    try:
        runtime.capture(plan.name)
    except Exception as error:
        record["capture_error"] = f"{type(error).__name__}: {error}"
        record_gate(
            f"scheduler-{plan.name}-capture",
            False,
            record["capture_error"],
        )
    runtime.note(f"SCHEDULER CELL DONE {plan.name} status={record['status']}")
    return record


def read_receipt_text(label: str) -> str:
    path = runtime.ROOT / f"{label}.log"
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def parse_json_line(text: str, expected_type: type) -> object | None:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, expected_type):
            return value
    return None


def capture_runtime_facts(
    state: PhaseState,
    label: str,
    requested_env: dict[str, str],
    cache_mode: str,
    require_dma_512k: bool = False,
    require_dma_off: bool = False,
) -> dict[str, Any]:
    inspect_label = f"{label}-env-inspect"
    mounts_label = f"{label}-mount-inspect"
    logs_label = f"{label}-server-log"
    inspect_rc = runtime.run(
        ["docker", "inspect", "--format", "{{json .Config.Env}}", runtime.NAME],
        label=inspect_label,
        timeout=60,
    )
    mounts_rc = runtime.run(
        ["docker", "inspect", "--format", "{{json .Mounts}}", runtime.NAME],
        label=mounts_label,
        timeout=60,
    )
    logs_rc = runtime.run(
        ["docker", "logs", runtime.NAME],
        label=logs_label,
        timeout=120,
    )
    inspect_text = read_receipt_text(inspect_label)
    mounts_text = read_receipt_text(mounts_label)
    server_text = read_receipt_text(logs_label)
    env_rows = parse_json_line(inspect_text, list)
    mounts = parse_json_line(mounts_text, list)
    actual_env: dict[str, str] = {}
    if isinstance(env_rows, list):
        for row in env_rows:
            if isinstance(row, str) and "=" in row:
                key, value = row.split("=", 1)
                actual_env[key] = value
    selected_keys = {
        "CACHE_MODE",
        "FAIRNESS_ENGINE",
        "PREFILL_COMPUTE_SHARE",
        "PREFILL_SCHEDULE_INTERVAL",
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "VLLM_PCIE_DMA_MIN_BYTES",
        "LMCACHE_L2_ROOT",
        "LMCACHE_L2_ROOT_HOST",
        "LMCACHE_L2_MAX_CAPACITY_GB",
        "LMCACHE_INSTANCE_ID",
        "LMCACHE_L2_ENABLED",
        "LMCACHE_MP_HOST",
        "LMCACHE_MP_PORT",
        "LMCACHE_HTTP_PORT",
        "LMCACHE_PROMETHEUS_PORT",
    }
    selected_env = {key: actual_env.get(key) for key in sorted(selected_keys)}
    batch_values = sorted(
        {
            int(value)
            for value in re.findall(
                r"max_num_batched_tokens['\"]?\s*[:=]\s*(\d+)",
                server_text,
            )
        }
    )
    dma_values = sorted(
        set(re.findall(r"DMA min=([^,)\s]+)", server_text, flags=re.IGNORECASE))
    )
    metric_text = ""
    metric_error: str | None = None
    try:
        metric_text = runtime.metric_snapshot()
    except Exception as error:
        metric_error = f"{type(error).__name__}: {error}"
    kv_values = {
        int(value)
        for value in re.findall(r'kv_cache_size_tokens="(\d+)"', metric_text)
    }
    kv_values.update(
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", server_text)
    )
    relevant_log_lines = [
        line
        for line in server_text.splitlines()
        if any(
            needle.lower() in line.lower()
            for needle in (
                "non-default args",
                "max_num_batched_tokens",
                "GPU KV cache size",
                "DMA min=",
                "VLLM_PCIE_DMA_MIN_BYTES",
                "unknown vLLM environment variable",
            )
        )
    ]
    expected_batch = int(requested_env["MAX_NUM_BATCHED_TOKENS"])
    batch_parser_passed = bool(batch_values) and set(batch_values) == {expected_batch}
    dma_512k_active = any(
        value.lower() in {"524288", "512kb", "512kib"} for value in dma_values
    )
    dma_off_active = any(value.lower() in {"off", "disabled", "none"} for value in dma_values)
    normalized_mounts = mounts if isinstance(mounts, list) else []
    facts = {
        "label": label,
        "cache_mode": cache_mode,
        "requested_env": requested_env,
        "inspect_return_code": inspect_rc,
        "mounts_return_code": mounts_rc,
        "logs_return_code": logs_rc,
        "actual_selected_env": selected_env,
        "container_mounts": normalized_mounts,
        "parser": {
            "max_num_batched_tokens_values": batch_values,
            "requested_max_num_batched_tokens": expected_batch,
            "matched": batch_parser_passed,
            "relevant_log_lines": relevant_log_lines,
        },
        "kv_capacity": {
            "gpu_kv_cache_size_tokens": sorted(kv_values),
            "source": "server GPU KV cache log and cache_config_info metric",
            "naive_block_multiplication_used": False,
        },
        "dma": {
            "requested": requested_env.get("VLLM_PCIE_DMA_MIN_BYTES"),
            "initialized_dma_min_values": dma_values,
            "threshold_512k_active": dma_512k_active,
            "default_dcp4_dma_off_active": dma_off_active,
            "activation_evidence": [
                line for line in relevant_log_lines if "DMA min=" in line
            ],
        },
        "metrics_error": metric_error,
        "raw_receipts": {
            "environment": str(runtime.ROOT / f"{inspect_label}.log"),
            "mounts": str(runtime.ROOT / f"{mounts_label}.log"),
            "server": str(runtime.ROOT / f"{logs_label}.log"),
        },
    }
    runtime.save_json(f"scheduler/{label}-runtime-facts.json", facts)
    state.runtime_facts[label] = facts
    record_gate(
        f"scheduler-{label}-batch-parser-effective",
        batch_parser_passed,
        facts["parser"],
    )
    record_gate(
        f"scheduler-{label}-kv-capacity-observed",
        bool(kv_values),
        facts["kv_capacity"],
    )
    if require_dma_512k:
        record_gate(
            f"scheduler-{label}-dma512k-threshold-active",
            dma_512k_active,
            facts["dma"],
        )
    if require_dma_off:
        record_gate(
            f"scheduler-{label}-dcp4-default-dma-off",
            dma_off_active,
            facts["dma"],
        )
    return facts


def boot_candidate(
    candidate: Candidate,
    label: str,
    cache_mode: str,
    env: dict[str, str],
) -> bool:
    passed = runtime.boot(
        label,
        image=candidate.image,
        tp=4,
        dcp=4,
        spec="mtp0",
        cache=cache_mode,
        kv="fp8_ds_mla",
        extra_env=env,
    )
    record_gate(
        f"scheduler-{label}-boot",
        passed,
        {
            "candidate": asdict(candidate),
            "cache_mode": cache_mode,
            "tp": 4,
            "dcp": 4,
            "spec": "mtp0",
            "kv": "fp8_ds_mla",
            "environment": env,
        },
    )
    return passed


def run_vram_candidate(
    state: PhaseState,
    candidate: Candidate,
    plans: tuple[WorkloadPlan, ...],
) -> None:
    boot_label = f"{candidate.key}-scheduler-vram"
    runtime.note(f"SCHEDULER CANDIDATE START {candidate.label} cache=vram")
    try:
        if not boot_candidate(candidate, boot_label, "vram", dict(BASE_ENV)):
            for plan in plans:
                state.invocations[plan.name] = {
                    "candidate": asdict(candidate),
                    "plan": plan_to_json(plan),
                    "cache_mode": "vram",
                    "status": "boot_failed",
                }
                record_gate(
                    f"scheduler-{plan.name}-workload-complete",
                    False,
                    f"candidate boot failed: {boot_label}",
                )
            return
        discovery = discover_api(candidate, boot_label)
        state.api_discoveries[boot_label] = discovery
        capture_runtime_facts(state, boot_label, dict(BASE_ENV), "vram")
        if candidate.key == OVERLAY.key and discovery.get("schema") == "native":
            discovery["atomic_rejection"] = check_native_atomic_rejection(boot_label)
        for plan in plans:
            try:
                run_workload_plan(state, candidate, plan, "vram")
            except Exception as error:
                detail = {
                    "candidate": candidate.key,
                    "plan": plan.name,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                state.errors.append(detail)
                state.invocations[plan.name] = {
                    "candidate": asdict(candidate),
                    "plan": plan_to_json(plan),
                    "cache_mode": "vram",
                    "status": "phase_exception",
                    "error": detail,
                }
                runtime.save_json(f"scheduler/{plan.name}-phase-error.json", detail)
                record_gate(
                    f"scheduler-{plan.name}-workload-complete",
                    False,
                    detail,
                )
    finally:
        try:
            runtime.capture(f"{boot_label}-final")
        except Exception as error:
            state.errors.append(
                {
                    "candidate": candidate.key,
                    "plan": "final-capture",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        runtime.stop()
        runtime.note(f"SCHEDULER CANDIDATE DONE {candidate.label} cache=vram")


def dma_parser_rejected_literal(label: str) -> dict[str, Any]:
    receipt_label = f"{label}-literal-failure-log"
    runtime.run(
        ["docker", "logs", runtime.NAME],
        label=receipt_label,
        timeout=60,
    )
    text = read_receipt_text(receipt_label)
    matching = [
        line
        for line in text.splitlines()
        if "512kb" in line.lower()
        or "vllm_pcie_dma_min_bytes" in line.lower()
        or (
            "byte" in line.lower()
            and any(
                word in line.lower()
                for word in ("parse", "invalid", "suffix", "integer", "literal")
            )
        )
    ]
    parser_rejection = any(
        re.search(
            r"(VLLM_PCIE_DMA_MIN_BYTES|512KB).*(invalid|parse|integer|literal|byte|suffix|size)",
            line,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(invalid|parse|integer|literal|byte|suffix|size).*(VLLM_PCIE_DMA_MIN_BYTES|512KB)",
            line,
            flags=re.IGNORECASE,
        )
        for line in matching
    )
    return {
        "parser_rejection": parser_rejection,
        "matching_lines": matching,
        "receipt": str(runtime.ROOT / f"{receipt_label}.log"),
    }


def factor_plan(factor: Factor) -> WorkloadPlan:
    return WorkloadPlan(
        factor.name,
        AUTO_RESPONSIVE_DA,
        ("periodic-128k",),
        (8,),
        1,
        "mixed-factor-identical-trace",
        "vram-mixed-batch-dma-factors",
    )


def run_factor(state: PhaseState, factor: Factor) -> None:
    candidate = OVERLAY
    plan = factor_plan(factor)
    env = dict(BASE_ENV)
    env["MAX_NUM_BATCHED_TOKENS"] = str(factor.batch_tokens)
    if factor.dma_value is not None:
        env["VLLM_PCIE_DMA_MIN_BYTES"] = factor.dma_value
    boot_label = f"{factor.name}-boot"
    used_representation = factor.dma_value
    literal_behavior: dict[str, Any] | None = None
    booted = False
    runtime.note(
        f"SCHEDULER MIXED FACTOR START {factor.name} batch={factor.batch_tokens} "
        f"dma={factor.dma_value or 'image-default'}"
    )
    try:
        booted = boot_candidate(candidate, boot_label, "vram", env)
        if not booted and factor.dma_value == "512KB":
            literal_behavior = dma_parser_rejected_literal(boot_label)
            runtime.save_json(
                f"scheduler/{factor.name}-dma-literal-behavior.json",
                literal_behavior,
            )
            if literal_behavior["parser_rejection"]:
                runtime.stop()
                env["VLLM_PCIE_DMA_MIN_BYTES"] = "524288"
                used_representation = "524288"
                retry_label = f"{factor.name}-numeric-bytes-boot"
                booted = boot_candidate(candidate, retry_label, "vram", env)
                boot_label = retry_label
        if factor.dma_value == "512KB":
            record_gate(
                f"scheduler-{factor.name}-dma-literal-parser-discovered",
                booted or bool(literal_behavior and literal_behavior["parser_rejection"]),
                {
                    "literal_boot_succeeded": booted and used_representation == "512KB",
                    "numeric_fallback_used": used_representation == "524288",
                    "literal_failure": literal_behavior,
                },
            )
        if not booted:
            state.invocations[plan.name] = {
                "candidate": asdict(candidate),
                "plan": plan_to_json(plan),
                "cache_mode": "vram",
                "factor": asdict(factor),
                "status": "boot_failed",
                "literal_behavior": literal_behavior,
            }
            record_gate(
                f"scheduler-{plan.name}-workload-complete",
                False,
                "factor boot failed",
            )
            return
        discovery = discover_api(candidate, boot_label)
        state.api_discoveries[boot_label] = discovery
        facts = capture_runtime_facts(
            state,
            boot_label,
            env,
            "vram",
            require_dma_512k=factor.dma_value is not None,
            require_dma_off=factor.dma_value is None,
        )
        result = run_workload_plan(state, candidate, plan, "vram")
        result["factor"] = {
            **asdict(factor),
            "used_dma_representation": used_representation,
            "literal_behavior": literal_behavior,
            "runtime_facts": facts,
        }
    except Exception as error:
        detail = {
            "candidate": candidate.key,
            "plan": plan.name,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        state.errors.append(detail)
        state.invocations[plan.name] = {
            "candidate": asdict(candidate),
            "plan": plan_to_json(plan),
            "cache_mode": "vram",
            "factor": asdict(factor),
            "status": "phase_exception",
            "error": detail,
        }
        runtime.save_json(f"scheduler/{plan.name}-phase-error.json", detail)
        record_gate(f"scheduler-{plan.name}-workload-complete", False, detail)
    finally:
        if booted:
            try:
                runtime.capture(f"{factor.name}-final")
            except Exception as error:
                state.errors.append(
                    {
                        "candidate": candidate.key,
                        "plan": f"{factor.name}-final-capture",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        runtime.stop()
        runtime.note(f"SCHEDULER MIXED FACTOR DONE {factor.name}")


def lmcache_plan(candidate: Candidate) -> WorkloadPlan:
    if candidate.key == OFFICIAL.key:
        return WorkloadPlan(
            "official-static04-lmcache-confirmation",
            STATIC_04,
            ("periodic-128k",),
            (8, 16),
            1,
            "lmcache-official-cold-namespace",
            "lmcache-confirmation",
        )
    return WorkloadPlan(
        "overlay-recommended-lmcache-confirmation",
        AUTO_RESPONSIVE_DA,
        ("periodic-128k",),
        (8, 16),
        1,
        "lmcache-overlay-cold-namespace",
        "lmcache-confirmation",
    )


def lmcache_storage_gate(label: str, facts: dict[str, Any]) -> dict[str, Any]:
    env = facts.get("actual_selected_env", {})
    raw_cap = env.get("LMCACHE_L2_MAX_CAPACITY_GB")
    try:
        capacity_gb = float(raw_cap)
    except (TypeError, ValueError):
        capacity_gb = None
    mounts = facts.get("container_mounts", [])
    l2_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict)
        and str(mount.get("Destination", "")).rstrip("/") == "/lmcache-l2"
    ]
    forbidden_sources = {
        "/mnt/2king/lmcache-l2",
        "/mnt/2king/lmcache-l2-r25-hot",
    }
    sources = [str(mount.get("Source")) for mount in l2_mounts]
    sidecar = {
        "host": env.get("LMCACHE_MP_HOST"),
        "rpc_port": env.get("LMCACHE_MP_PORT"),
        "http_port": env.get("LMCACHE_HTTP_PORT"),
        "prometheus_port": env.get("LMCACHE_PROMETHEUS_PORT"),
    }
    sidecar_matches_launch = sidecar == {
        "host": "127.0.0.1",
        "rpc_port": "15555",
        "http_port": "18085",
        "prometheus_port": "19095",
    }
    isolated_sources = bool(sources) and all(
        Path(source).resolve().is_relative_to(runtime.L2_HOST_ROOT.resolve())
        for source in sources
    )
    passed = (
        env.get("CACHE_MODE") == "lmcache"
        and capacity_gb is not None
        and 0.0 < capacity_gb <= 160.0
        and bool(l2_mounts)
        and not any(source in forbidden_sources for source in sources)
        and isolated_sources
        and sidecar_matches_launch
    )
    detail = {
        "cache_mode": env.get("CACHE_MODE"),
        "capacity_gb": capacity_gb,
        "l2_mounts": l2_mounts,
        "forbidden_sources": sorted(forbidden_sources),
        "sources_under_runtime_l2_root": isolated_sources,
        "sidecar_from_container_inspect": sidecar,
        "sidecar_matches_runtime_launch": sidecar_matches_launch,
        "runtime_managed_isolated_root_required": True,
        "production_l2_touched": False,
        "deletion_performed": False,
        "host_storage": "NVMe; not an iSource SATA reproduction",
    }
    record_gate(f"scheduler-{label}-isolated-capped-lmcache", passed, detail)
    return {"passed": passed, "detail": detail}


def run_lmcache_confirmation(state: PhaseState) -> None:
    labels_and_roots: list[tuple[str, list[str]]] = []
    for candidate in (OFFICIAL, OVERLAY):
        plan = lmcache_plan(candidate)
        boot_label = f"{candidate.key}-scheduler-lmcache"
        booted = False
        runtime.note(f"SCHEDULER LMCACHE START {candidate.label}")
        try:
            booted = boot_candidate(
                candidate,
                boot_label,
                "lmcache",
                dict(BASE_ENV),
            )
            if not booted:
                state.invocations[plan.name] = {
                    "candidate": asdict(candidate),
                    "plan": plan_to_json(plan),
                    "cache_mode": "lmcache",
                    "status": "boot_failed",
                }
                record_gate(
                    f"scheduler-{plan.name}-workload-complete",
                    False,
                    f"candidate LMCache boot failed: {boot_label}",
                )
                continue
            discovery = discover_api(candidate, boot_label)
            state.api_discoveries[boot_label] = discovery
            facts = capture_runtime_facts(
                state,
                boot_label,
                dict(BASE_ENV),
                "lmcache",
            )
            storage = lmcache_storage_gate(boot_label, facts)
            sources = [
                str(mount.get("Source"))
                for mount in facts.get("container_mounts", [])
                if isinstance(mount, dict)
                and str(mount.get("Destination", "")).rstrip("/") == "/lmcache-l2"
            ]
            labels_and_roots.append((boot_label, sources))
            result = run_workload_plan(state, candidate, plan, "lmcache")
            result["lmcache_storage"] = storage
        except Exception as error:
            detail = {
                "candidate": candidate.key,
                "plan": plan.name,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            state.errors.append(detail)
            state.invocations[plan.name] = {
                "candidate": asdict(candidate),
                "plan": plan_to_json(plan),
                "cache_mode": "lmcache",
                "status": "phase_exception",
                "error": detail,
            }
            runtime.save_json(f"scheduler/{plan.name}-phase-error.json", detail)
            record_gate(f"scheduler-{plan.name}-workload-complete", False, detail)
        finally:
            if booted:
                try:
                    runtime.capture(f"{boot_label}-final")
                except Exception as error:
                    state.errors.append(
                        {
                            "candidate": candidate.key,
                            "plan": "lmcache-final-capture",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            runtime.stop()
            runtime.note(f"SCHEDULER LMCACHE DONE {candidate.label}")
    same_root = (
        len(labels_and_roots) == 2
        and bool(labels_and_roots[0][1])
        and labels_and_roots[0][1] == labels_and_roots[1][1]
    )
    state.lmcache_same_root = same_root
    record_gate(
        "scheduler-lmcache-confirmation-same-isolated-root",
        same_root,
        {
            "observed": labels_and_roots,
            "series_separate_from_vram": True,
        },
    )


def cells_by_trace(record: dict[str, Any]) -> dict[tuple[str, int, int], str]:
    result: dict[tuple[str, int, int], str] = {}
    payload = record.get("result")
    if not isinstance(payload, dict):
        return result
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        trace = cell.get("trace")
        if not isinstance(trace, dict) or not isinstance(trace.get("trace_hash"), str):
            continue
        key = (
            str(cell.get("profile")),
            int(cell.get("concurrency")),
            int(cell.get("repeat")),
        )
        result[key] = trace["trace_hash"]
    return result


def compare_trace_group(
    state: PhaseState,
    name: str,
    members: tuple[str, ...],
    expected_keys: tuple[tuple[str, int, int], ...] | None = None,
) -> dict[str, Any]:
    traces = {
        member: cells_by_trace(state.invocations.get(member, {})) for member in members
    }
    if expected_keys is None:
        key_sets = [set(values) for values in traces.values()]
        keys = set.intersection(*key_sets) if key_sets else set()
        exact_key_sets = bool(key_sets) and all(values == key_sets[0] for values in key_sets)
    else:
        keys = set(expected_keys)
        exact_key_sets = all(keys <= set(values) for values in traces.values())
    mismatches: dict[str, dict[str, str | None]] = {}
    for key in sorted(keys):
        values = {member: traces[member].get(key) for member in members}
        if None in values.values() or len(set(values.values())) != 1:
            mismatches[str(key)] = values
    passed = exact_key_sets and bool(keys) and not mismatches
    receipt = {
        "name": name,
        "members": list(members),
        "expected_keys": [list(key) for key in expected_keys] if expected_keys else None,
        "observed_keys": {
            member: [list(key) for key in sorted(values)]
            for member, values in traces.items()
        },
        "trace_hashes": {
            member: {str(key): value for key, value in sorted(values.items())}
            for member, values in traces.items()
        },
        "mismatches": mismatches,
        "passed": passed,
    }
    record_gate(f"scheduler-identical-trace-{safe_name(name)}", passed, receipt)
    return receipt


def trace_checks(state: PhaseState) -> list[dict[str, Any]]:
    checks = [
        compare_trace_group(
            state,
            "vram-headline-three-arm",
            (
                "official-off-headline",
                "official-static04-headline",
                "overlay-recommended-headline",
            ),
        ),
        compare_trace_group(
            state,
            "vram-profile-shapes",
            (
                "official-off-profile-shapes",
                "overlay-off-profile-shapes",
                "overlay-recommended-profile-shapes",
            ),
        ),
        compare_trace_group(
            state,
            "vram-off-image-parity",
            ("official-off-headline", "overlay-off-sweep"),
            (("periodic-128k", 8, 1), ("periodic-128k", 16, 1)),
        ),
        compare_trace_group(
            state,
            "vram-static04-image-parity",
            ("official-static04-headline", "overlay-static04-sweep"),
            (("periodic-128k", 8, 1), ("periodic-128k", 16, 1)),
        ),
        compare_trace_group(
            state,
            "vram-static06-image-parity",
            ("official-static06-sweep", "overlay-static06-sweep"),
        ),
        compare_trace_group(
            state,
            "vram-static07-image-parity",
            ("official-static07-sweep", "overlay-static07-sweep"),
        ),
        compare_trace_group(
            state,
            "overlay-interleave-and-auto-ab",
            (
                "overlay-recommended-headline",
                "overlay-auto-smooth",
                "overlay-policy-fcfs",
                "overlay-policy-round-robin",
                "overlay-policy-shortest-remaining",
            ),
            (("periodic-128k", 8, 1), ("periodic-128k", 16, 1)),
        ),
        compare_trace_group(
            state,
            "overlay-live-update-ab",
            (
                "overlay-recommended-headline",
                "overlay-live-policy-updates",
            ),
            (("periodic-128k", 16, 1),),
        ),
        compare_trace_group(
            state,
            "lmcache-matched-confirmation",
            (
                "official-static04-lmcache-confirmation",
                "overlay-recommended-lmcache-confirmation",
            ),
        ),
        compare_trace_group(
            state,
            "overlay-mixed-batch-dma-factors",
            tuple(factor.name for factor in FACTORS),
        ),
    ]
    runtime.save_json("scheduler/trace-equality-checks.json", checks)
    return checks


def phase_plan() -> dict[str, Any]:
    all_plans = [*OFFICIAL_PLANS, *OVERLAY_PLANS]
    factor_plans = [factor_plan(factor) for factor in FACTORS]
    lmcache_plans = [lmcache_plan(candidate) for candidate in (OFFICIAL, OVERLAY)]
    measurement_seconds = sum(plan_measurement_seconds(plan) for plan in all_plans)
    factor_seconds = sum(plan_measurement_seconds(plan) for plan in factor_plans)
    lmcache_seconds = sum(plan_measurement_seconds(plan) for plan in lmcache_plans)
    return {
        "phase": "scheduler",
        "source_guide": str(SOURCE_GUIDE),
        "trace_seed": TRACE_SEED,
        "candidates": [asdict(OFFICIAL), asdict(OVERLAY)],
        "overlay_source_revision": "7db6a2d2f5680513ae1a396ff61169c4cacf8a95",
        "candidate_labelling": {
            "official": "stock Official R26 image",
            "overlay": "D-Rock R26 Python overlay; never labelled stock R26",
        },
        "topology": {
            "tp": 4,
            "dcp": 4,
            "speculation": "off (mtp0) to avoid scheduler/speculation confounding",
            "kv": "fp8_ds_mla",
            "max_num_seqs": 32,
            "cache": "VRAM except the explicitly separate matched LMCache confirmation",
        },
        "api_compatibility": {
            "native_detection": list(NATIVE_FIELDS),
            "native_post_semantics": "complete five-field replacement, never PATCH",
            "legacy_detection": "fairness_engine field in successful GET",
            "legacy_post_fields": list(LEGACY_FIELDS),
            "legacy_use": "only off/static FCFS cells after actual legacy readback",
            "unsupported": "auto/interleaving on a legacy endpoint is a failed gate and skipped",
            "readback": "GET before and after every cell; workload samples effective state",
            "source_discovery": {
                "official_r26": (
                    "source lock lists PRs 619,573,571,572,642,552,558,553 and "
                    "the base launcher maps FAIRNESS_ENGINE/prefill share to the old selector; "
                    "it does not claim PR647/PR648"
                ),
                "drock_overlay": (
                    "image label claims the PR648 overlay; the runtime GET field set remains "
                    "authoritative before any native request is sent"
                ),
            },
        },
        "arrival_trace": {
            "hot_sessions": [8, 16],
            "hot_shape": (
                "fixed offered-arrival replay of decode -> small incremental prefill -> "
                "decode cycles; one in-flight request per session and client wait included"
            ),
            "session_context_tokens": 8192,
            "turn_decode_tokens": 128,
            "turn_period_seconds": 6,
            "periodic_cold_tokens": 131072,
            "periodic_cold_seconds": 15,
            "short_prefill_tokens": [2048, 4096, 8192],
            "short_prefill_period_seconds": 1.5,
            "analytics_cold_tokens": 204800,
            "analytics_offsets_seconds": [0, 1, 2],
            "cache_salt_note": (
                "cache namespaces prevent cross-cell hits; namespace is excluded from the "
                "trace hash while prompt bodies and arrival offsets remain identical"
            ),
        },
        "duration_shape": {
            "profile_measurement_seconds": PROFILE_SECONDS,
            "vram_measurement_seconds": measurement_seconds,
            "mixed_factor_measurement_seconds": factor_seconds,
            "lmcache_measurement_seconds": lmcache_seconds,
            "total_measurement_seconds": measurement_seconds
            + factor_seconds
            + lmcache_seconds,
            "headline_repeats": 2,
            "coverage_repeats": 1,
            "drain_cutoff_seconds_per_scenario": 240,
            "excludes": "model boot, prompt construction, cooldown, capture, and drain wall time",
        },
        "vram_plans": [plan_to_json(plan) for plan in all_plans],
        "mixed_factors": [
            {**asdict(factor), "plan": plan_to_json(factor_plan(factor))}
            for factor in FACTORS
        ],
        "mixed_factor_contract": {
            "candidate": OVERLAY.label,
            "workload": "8-session periodic unique-128K mixed traffic only",
            "dcp4_control_dma": (
                "off, as selected by the pinned leaf launcher when "
                "VLLM_PCIE_DMA_MIN_BYTES is unset"
            ),
            "dma_treatment": "explicit 512KB (524288-byte fallback only on proven parser rejection)",
            "isolated_factors": ["batch token budget", "512KB DMA threshold"],
            "combined_factor": "12288 scheduler tokens plus 512KB DMA threshold",
        },
        "lmcache_confirmation": {
            "plans": [plan_to_json(plan) for plan in lmcache_plans],
            "series_separate_from_vram": True,
            "runtime_managed_root": True,
            "maximum_l2_capacity_gb": 160,
            "delete_or_clean_l2": False,
            "host_storage": "NVMe; not an iSource SATA reproduction",
            "host_root": str(runtime.L2_HOST_ROOT / "shared"),
            "sidecar_ports_from_runtime_launch": {
                "rpc": 15555,
                "http": 18085,
                "prometheus": 19095,
            },
        },
        "core_matrix_handoff": {
            "core_owns": "quick decode/prefill 4096/8192/12288/16384 and DMA A/B",
            "scheduler_owns": "mixed 8-session periodic-128K effects for those factors",
            "duplication": False,
        },
        "raw_observations": [
            "every offered hot and cold request, including errors/censoring",
            "scheduled/submitted/first-token/finish times and stream integrity",
            "p50/p95/p99 hot TTFT, end-to-end, and stream-chunk decode gaps",
            "absolute completions plus per-second/per-session normalization",
            "queue depth samples, growth and slope",
            "prefill/decode model-compute counter deltas",
            "configured and effective policy samples",
            "GPU KV capacity from server-reported cache size",
            "container environment and launcher parser evidence",
        ],
    }


def derive_qualification_checks(
    state: PhaseState,
    traces: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_boots = 2 + len(FACTORS) + 2
    api_details = {
        label: {
            "candidate": discovery.get("candidate", {}).get("key"),
            "schema": discovery.get("schema"),
        }
        for label, discovery in state.api_discoveries.items()
    }
    api_recognized = (
        len(state.api_discoveries) == expected_boots
        and all(
            discovery.get("schema") in {"native", "legacy"}
            for discovery in state.api_discoveries.values()
        )
    )
    overlay_native = all(
        discovery.get("schema") == "native"
        for discovery in state.api_discoveries.values()
        if discovery.get("candidate", {}).get("key") == OVERLAY.key
    ) and any(
        discovery.get("candidate", {}).get("key") == OVERLAY.key
        for discovery in state.api_discoveries.values()
    )
    base_overlay_discovery = state.api_discoveries.get(
        f"{OVERLAY.key}-scheduler-vram", {}
    )
    atomic_rejection = bool(
        base_overlay_discovery.get("atomic_rejection", {}).get("passed")
    )
    runtime_fact_details = {
        label: {
            "batch_parser": facts.get("parser", {}).get("matched"),
            "kv_cache_size_tokens": facts.get("kv_capacity", {}).get(
                "gpu_kv_cache_size_tokens"
            ),
        }
        for label, facts in state.runtime_facts.items()
    }
    runtime_facts_complete = (
        len(state.runtime_facts) == expected_boots
        and all(
            facts.get("parser", {}).get("matched") is True
            and bool(
                facts.get("kv_capacity", {}).get("gpu_kv_cache_size_tokens")
            )
            for facts in state.runtime_facts.values()
        )
    )
    factor_transport_details: dict[str, Any] = {}
    factor_transport_passed = True
    for factor in FACTORS:
        record = state.invocations.get(factor.name, {})
        facts = record.get("factor", {}).get("runtime_facts", {})
        dma = facts.get("dma", {})
        expected = "512KB" if factor.dma_value is not None else "off"
        observed = (
            dma.get("threshold_512k_active") is True
            if factor.dma_value is not None
            else dma.get("default_dcp4_dma_off_active") is True
        )
        factor_transport_details[factor.name] = {
            "expected": expected,
            "observed": dma,
            "passed": observed,
        }
        factor_transport_passed = factor_transport_passed and observed
    lmcache_details = {
        name: state.invocations.get(name, {}).get("lmcache_storage")
        for name in (
            "official-static04-lmcache-confirmation",
            "overlay-recommended-lmcache-confirmation",
        )
    }
    lmcache_storage_passed = (
        state.lmcache_same_root is True
        and all(
            isinstance(detail, dict) and detail.get("passed") is True
            for detail in lmcache_details.values()
        )
    )
    trace_passed = bool(traces) and all(
        receipt.get("passed") is True for receipt in traces
    )
    return {
        "api_schemas_recognized": {
            "passed": api_recognized,
            "detail": {
                "expected_boots": expected_boots,
                "discoveries": api_details,
            },
        },
        "overlay_native_api": {
            "passed": overlay_native,
            "detail": api_details,
        },
        "native_post_atomic_rejection": {
            "passed": atomic_rejection,
            "detail": base_overlay_discovery.get("atomic_rejection"),
        },
        "runtime_batch_parser_and_kv_capacity": {
            "passed": runtime_facts_complete,
            "detail": {
                "expected_boots": expected_boots,
                "facts": runtime_fact_details,
            },
        },
        "mixed_factor_dma_transport": {
            "passed": factor_transport_passed,
            "detail": factor_transport_details,
        },
        "lmcache_isolated_capped_same_root": {
            "passed": lmcache_storage_passed,
            "detail": {
                "same_root": state.lmcache_same_root,
                "arms": lmcache_details,
            },
        },
        "identical_ab_traces": {
            "passed": trace_passed,
            "detail": traces,
        },
    }


def main() -> int:
    state = PhaseState()
    checks: list[dict[str, Any]] = []
    plan = phase_plan()
    runtime.save_json("scheduler/phase-plan.json", plan)
    runtime.note(
        "R26 SCHEDULER PHASE START: official and separately labelled D-Rock overlay"
    )
    try:
        try:
            run_vram_candidate(state, OFFICIAL, OFFICIAL_PLANS)
        except Exception as error:
            state.errors.append(
                {
                    "candidate": OFFICIAL.key,
                    "plan": "vram-candidate",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        try:
            run_vram_candidate(state, OVERLAY, OVERLAY_PLANS)
        except Exception as error:
            state.errors.append(
                {
                    "candidate": OVERLAY.key,
                    "plan": "vram-candidate",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        for factor in FACTORS:
            try:
                run_factor(state, factor)
            except Exception as error:
                state.errors.append(
                    {
                        "candidate": OVERLAY.key,
                        "plan": factor.name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                runtime.stop()
        try:
            run_lmcache_confirmation(state)
        except Exception as error:
            state.errors.append(
                {
                    "candidate": "official-and-overlay",
                    "plan": "lmcache-confirmation",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            runtime.stop()
        try:
            checks = trace_checks(state)
        except Exception as error:
            state.errors.append(
                {
                    "candidate": "all",
                    "plan": "trace-equality-checks",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            record_gate(
                "scheduler-identical-trace-checks-completed",
                False,
                state.errors[-1],
            )
    finally:
        runtime.stop()
    expected_invocations = (
        len(OFFICIAL_PLANS) + len(OVERLAY_PLANS) + len(FACTORS) + 2
    )
    completed_invocations = sum(
        record.get("status") == "complete" for record in state.invocations.values()
    )
    all_recorded = len(state.invocations) == expected_invocations
    all_complete = all_recorded and completed_invocations == expected_invocations
    qualification_checks = derive_qualification_checks(state, checks)
    derived_checks_passed = all(
        check["passed"] is True for check in qualification_checks.values()
    )
    all_qualified = (
        all_complete and derived_checks_passed and not state.errors
    )
    for check_name, check in qualification_checks.items():
        record_gate(
            f"scheduler-phase-{safe_name(check_name)}",
            bool(check["passed"]),
            check["detail"],
        )
    summary = {
        "phase": "scheduler",
        "plan": plan,
        "expected_invocations": expected_invocations,
        "recorded_invocations": len(state.invocations),
        "completed_invocations": completed_invocations,
        "invocations": {
            name: {key: value for key, value in record.items() if key != "result"}
            for name, record in state.invocations.items()
        },
        "api_discoveries": state.api_discoveries,
        "runtime_facts": state.runtime_facts,
        "trace_checks": checks,
        "qualification_checks": qualification_checks,
        "errors": state.errors,
        "status": "complete" if all_qualified else "failed",
    }
    runtime.save_json("scheduler/phase-summary.json", summary)
    record_gate(
        "scheduler-phase-all-cells-recorded",
        all_recorded,
        {
            "expected": expected_invocations,
            "recorded": len(state.invocations),
        },
    )
    record_gate(
        "scheduler-phase-all-cells-qualified",
        all_qualified,
        {
            "expected": expected_invocations,
            "completed": completed_invocations,
            "derived_checks": qualification_checks,
            "errors": state.errors,
        },
    )
    runtime.note(
        f"R26 SCHEDULER PHASE DONE status={summary['status']} "
        f"completed={completed_invocations}/{expected_invocations}"
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
