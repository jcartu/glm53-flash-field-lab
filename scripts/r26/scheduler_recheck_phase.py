#!/usr/bin/env python3
"""Recheck the pinned R26 scheduler and overlay with the shipped API contract.

The phase is serial. Structural overlay settings are selected only at process
boot, while the development endpoint is used only for the two compute-share
fields that the shipped scheduler accepts at runtime. All receipts live below
ROOT/scheduler-recheck so failed predecessor evidence remains untouched.

With --resume, validated prior receipts under ROOT/scheduler-recheck are reused
verbatim (including measured qualification failures), partial plans run only
their missing scenarios, groups without remaining work are not booted, and every
record carries explicit pre-reboot/post-reboot measurement provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

try:
    from . import runtime
    from . import agent_workload_recheck as workload
except ImportError:
    import runtime  # type: ignore[no-redef]
    import agent_workload_recheck as workload  # type: ignore[no-redef]


OUTPUT_DIR = "scheduler-recheck"
WORKLOAD = Path(__file__).with_name("agent_workload_recheck.py")
TRACE_SEED = "r26-mixed-agent-recheck-v2"
DEFAULT_EVIDENCE_ROOT = Path(
    os.environ.get(
        "BATTERY_EVIDENCE_ROOT",
        "/home/josh/omp-workspace/drock-lmcache/r26-battery",
    )
)
RESUME_PROVENANCE_REUSED = "reused_pre_reboot_core+250"
RESUME_PROVENANCE_RESUMED = "reused_post_reboot_core0"
RESUME_PROVENANCE_NEW = "new_post_reboot_core0"
PROFILE_SECONDS = {
    "baseline": 30.0,
    "periodic-128k": 60.0,
    "short-prefill-heavy": 60.0,
    "analytics-200k-burst": 90.0,
}
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
OVERLAY_STRUCTURAL_FIELDS = (
    "max_parallel_prefills",
    "prefill_policy",
    "decode_refill_target",
)
LEGACY_FIELDS = (
    "fairness_engine",
    "prefill_compute_share",
    "max_num_prefill_tokens_per_step",
    "max_num_partial_prefills",
    "decode_prefill_min_decode_steps",
    "decode_prefill_max_wait_ms",
)
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
    api_schema: str
    source_status: str


@dataclass(frozen=True)
class Policy:
    key: str
    compute_share: float | str | None
    half_life: float | str | None
    max_parallel_prefills: int | str
    prefill_policy: str
    decode_refill_target: int | str


@dataclass(frozen=True)
class WorkloadPlan:
    name: str
    candidate_key: str
    policy: Policy
    profiles: tuple[str, ...]
    concurrencies: tuple[int, ...]
    repeats: int
    cache_namespace: str
    series: str
    headline: bool = False
    updates: tuple[dict[str, Any], ...] = ()
    cache_mode: str = "vram"
    memory_role: str = "none"


@dataclass(frozen=True)
class BootGroup:
    name: str
    candidate: Candidate
    policy_structure: Policy
    plans: tuple[WorkloadPlan, ...]
    cache_mode: str = "vram"
    batch_tokens: int = 4096
    dma_value: str | None = None
    factor_role: str | None = None


OFFICIAL = Candidate(
    key="official-r26",
    label="Official R26",
    image=runtime.IMAGE,
    api_schema="legacy-r26",
    source_status="stock pinned R26 image; captured six-field legacy API",
)
OVERLAY = Candidate(
    key="drock-r26-overlay",
    label="D-Rock R26 scheduler overlay (not stock R26)",
    image=runtime.OVERLAY_IMAGE,
    api_schema="overlay-r26",
    source_status="pinned overlay image and captured shipped scheduler source",
)


def policy(
    key: str,
    share: float | str | None,
    half_life: float | str | None = None,
    lanes: int | str = 1,
    prefill_policy: str = "round-robin",
    refill: int | str = "auto",
) -> Policy:
    result = Policy(key, share, half_life, lanes, prefill_policy, refill)
    validate_policy(result)
    return result


def validate_policy(value: Policy) -> None:
    share = value.compute_share
    valid_share = share in (None, "auto") or (
        isinstance(share, (int, float))
        and not isinstance(share, bool)
        and 0.0 < float(share) < 1.0
    )
    if not valid_share:
        raise ValueError(f"invalid compute share in {value}")
    half = value.half_life
    valid_half = half in (None, "smooth", "responsive") or (
        isinstance(half, (int, float))
        and not isinstance(half, bool)
        and math.isfinite(float(half))
        and float(half) > 0.0
    )
    if not valid_half or (half is not None and share != "auto"):
        raise ValueError(f"invalid compute half-life in {value}")
    lanes = value.max_parallel_prefills
    if not (
        lanes == "auto"
        or (isinstance(lanes, int) and not isinstance(lanes, bool) and lanes >= 1)
    ):
        raise ValueError(f"invalid max_parallel_prefills in {value}")
    if value.prefill_policy not in {"round-robin", "decode-aware"}:
        raise ValueError(f"invalid prefill_policy in {value}")
    refill = value.decode_refill_target
    if not (
        refill == "auto"
        or (isinstance(refill, int) and not isinstance(refill, bool) and refill >= 1)
    ):
        raise ValueError(f"invalid decode_refill_target in {value}")
    if lanes == 1 and value.prefill_policy != "round-robin":
        raise ValueError("single-lane mode must use round-robin")
    if value.prefill_policy != "decode-aware" and refill != "auto":
        raise ValueError("explicit decode refill requires decode-aware policy")


OFF = policy("off-single-lane", None)
STATIC_04 = policy("static0.4-single-lane", 0.4)
STATIC_06 = policy("static0.6-single-lane", 0.6)
STATIC_07 = policy("static0.7-single-lane", 0.7)


def automatic_policy(
    half_life: str,
    prefill_policy: str,
    lanes: int | str,
) -> Policy:
    refill: int | str = (
        4 if lanes == 4 and prefill_policy == "decode-aware" else "auto"
    )
    lane_name = "auto" if lanes == "auto" else str(lanes)
    policy_name = "rr" if prefill_policy == "round-robin" else "decode-aware"
    return policy(
        f"auto-{half_life}-{policy_name}-lanes{lane_name}-refill{refill}",
        "auto",
        half_life,
        lanes,
        prefill_policy,
        refill,
    )


def make_plan(
    name: str,
    candidate: Candidate,
    selected_policy: Policy,
    profiles: tuple[str, ...],
    concurrencies: tuple[int, ...] = (8, 16),
    repeats: int = 1,
    *,
    headline: bool = False,
    series: str,
    cache_namespace: str | None = None,
    updates: tuple[dict[str, Any], ...] = (),
    cache_mode: str = "vram",
    memory_role: str = "none",
) -> WorkloadPlan:
    return WorkloadPlan(
        name=name,
        candidate_key=candidate.key,
        policy=selected_policy,
        profiles=profiles,
        concurrencies=concurrencies,
        repeats=repeats,
        cache_namespace=cache_namespace or name,
        series=series,
        headline=headline,
        updates=updates,
        cache_mode=cache_mode,
        memory_role=memory_role,
    )


def control_plans(candidate: Candidate, prefix: str) -> tuple[WorkloadPlan, ...]:
    return (
        make_plan(
            f"{prefix}-off-headline",
            candidate,
            OFF,
            ("baseline", "periodic-128k"),
            repeats=2,
            headline=True,
            series="single-lane-headline",
        ),
        make_plan(
            f"{prefix}-static04-headline",
            candidate,
            STATIC_04,
            ("baseline", "periodic-128k"),
            repeats=2,
            headline=True,
            series="single-lane-headline",
        ),
        make_plan(
            f"{prefix}-static06-coverage",
            candidate,
            STATIC_06,
            ("periodic-128k",),
            series="single-lane-share-coverage",
        ),
        make_plan(
            f"{prefix}-static07-coverage",
            candidate,
            STATIC_07,
            ("periodic-128k",),
            series="single-lane-share-coverage",
        ),
        make_plan(
            f"{prefix}-off-profile-coverage",
            candidate,
            OFF,
            ("short-prefill-heavy", "analytics-200k-burst"),
            series="profile-shape-coverage",
        ),
    )


OFFICIAL_CONTROL_PLANS = control_plans(OFFICIAL, "official")
OVERLAY_CONTROL_PLANS = control_plans(OVERLAY, "overlay-single")
RECOMMENDED = automatic_policy("responsive", "decode-aware", "auto")
LIVE_UPDATES = (
    {
        "offset_seconds": 10.0,
        "config": {
            "prefill_compute_share": 0.6,
            "prefill_compute_half_life": None,
        },
    },
    {
        "offset_seconds": 25.0,
        "config": {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "smooth",
        },
    },
    {
        "offset_seconds": 40.0,
        "config": {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "responsive",
        },
    },
)


def automatic_plans() -> tuple[WorkloadPlan, ...]:
    result: list[WorkloadPlan] = []
    for lanes, prefill_mode, half_life in product(
        ("auto", 4), ("round-robin", "decode-aware"), ("responsive", "smooth")
    ):
        selected = automatic_policy(half_life, prefill_mode, lanes)
        lane_name = "auto" if lanes == "auto" else str(lanes)
        mode_name = "rr" if prefill_mode == "round-robin" else "decode-aware"
        name = f"overlay-auto-{half_life}-{mode_name}-lanes{lane_name}"
        if selected == RECOMMENDED:
            result.append(
                make_plan(
                    f"{name}-headline",
                    OVERLAY,
                    selected,
                    ("baseline", "periodic-128k"),
                    repeats=2,
                    headline=True,
                    series="overlay-auto-headline",
                )
            )
        else:
            result.append(
                make_plan(
                    f"{name}-coverage",
                    OVERLAY,
                    selected,
                    ("periodic-128k",),
                    series="overlay-auto-factorial",
                )
            )
    result.extend(
        (
            make_plan(
                "overlay-auto-responsive-decode-aware-lanesauto-profile-coverage",
                OVERLAY,
                RECOMMENDED,
                ("short-prefill-heavy", "analytics-200k-burst"),
                series="profile-shape-coverage",
            ),
            make_plan(
                "overlay-auto-responsive-decode-aware-lanesauto-live-update",
                OVERLAY,
                RECOMMENDED,
                ("periodic-128k",),
                concurrencies=(16,),
                series="overlay-live-update",
                updates=LIVE_UPDATES,
            ),
        )
    )
    return tuple(result)


OVERLAY_AUTOMATIC_PLANS = automatic_plans()
FACTOR_SPECS = (
    ("overlay-factor-bt4096-default-dma", 4096, None, "batch-control"),
    ("overlay-factor-bt8192-default-dma", 8192, None, "batch-isolated"),
    ("overlay-factor-bt12288-default-dma", 12288, None, "batch-isolated"),
    ("overlay-factor-bt16384-default-dma", 16384, None, "batch-isolated"),
    ("overlay-factor-bt4096-dma512k", 4096, "512KB", "dma-isolated"),
    ("overlay-factor-bt12288-dma512k", 12288, "512KB", "batch-dma-combined"),
)
FACTOR_POLICY = automatic_policy("responsive", "decode-aware", 4)


def structural_key(value: Policy) -> tuple[int | str, str, int | str]:
    return (
        value.max_parallel_prefills,
        value.prefill_policy,
        value.decode_refill_target,
    )




def build_boot_groups() -> tuple[BootGroup, ...]:
    groups: list[BootGroup] = [
        BootGroup(
            "official-controls-vram",
            OFFICIAL,
            OFF,
            OFFICIAL_CONTROL_PLANS,
        ),
        BootGroup(
            "overlay-single-lane-controls-vram",
            OVERLAY,
            OFF,
            OVERLAY_CONTROL_PLANS,
        ),
    ]
    for lanes, prefill_mode in product(("auto", 4), ("round-robin", "decode-aware")):
        matching = tuple(
            plan
            for plan in OVERLAY_AUTOMATIC_PLANS
            if structural_key(plan.policy)
            == structural_key(automatic_policy("responsive", prefill_mode, lanes))
        )
        lane_name = "auto" if lanes == "auto" else str(lanes)
        mode_name = "rr" if prefill_mode == "round-robin" else "decode-aware"
        groups.append(
            BootGroup(
                f"overlay-{mode_name}-lanes{lane_name}-vram",
                OVERLAY,
                automatic_policy("responsive", prefill_mode, lanes),
                matching,
            )
        )
    for name, batch_tokens, dma_value, role in FACTOR_SPECS:
        factor_plan = make_plan(
            name,
            OVERLAY,
            FACTOR_POLICY,
            ("periodic-128k",),
            concurrencies=(8,),
            series="mixed-batch-dma-factors",
            cache_namespace="mixed-factor-identical-trace",
        )
        groups.append(
            BootGroup(
                name,
                OVERLAY,
                FACTOR_POLICY,
                (factor_plan,),
                batch_tokens=batch_tokens,
                dma_value=dma_value,
                factor_role=role,
            )
        )
    lmcache_namespace = "lmcache-matched-recheck"
    official_lmcache = make_plan(
        "official-static04-lmcache-matched-writer",
        OFFICIAL,
        STATIC_04,
        ("periodic-128k",),
        series="lmcache-matched-confirmation",
        cache_namespace=lmcache_namespace,
        cache_mode="lmcache",
        memory_role="writer",
    )
    overlay_lmcache = make_plan(
        "overlay-static04-lmcache-matched-reader",
        OVERLAY,
        STATIC_04,
        ("periodic-128k",),
        series="lmcache-matched-confirmation",
        cache_namespace=lmcache_namespace,
        cache_mode="lmcache",
        memory_role="reader",
    )
    groups.extend(
        (
            BootGroup(
                "official-lmcache-matched-writer",
                OFFICIAL,
                STATIC_04,
                (official_lmcache,),
                cache_mode="lmcache",
            ),
            BootGroup(
                "overlay-lmcache-matched-reader",
                OVERLAY,
                STATIC_04,
                (overlay_lmcache,),
                cache_mode="lmcache",
            ),
        )
    )
    return tuple(groups)


BOOT_GROUPS = build_boot_groups()
ALL_PLANS = tuple(plan for group in BOOT_GROUPS for plan in group.plans)
UNSUPPORTED_REQUESTED_MODES = (
    {
        "requested_feature": "prefill_interleave_policy=fcfs",
        "status": "unsupported_not_mapped",
        "reason": (
            "the shipped field is prefill_policy and allows only round-robin or "
            "decode-aware; lanes=1 plus round-robin is labelled legacy single-prefill "
            "behavior, not a standalone FCFS feature"
        ),
        "attempted": False,
    },
    {
        "requested_feature": "prefill_interleave_policy=shortest-remaining",
        "status": "unsupported_not_mapped",
        "reason": (
            "shortest-remaining is not an allowed shipped prefill_policy value; "
            "decode-aware is not relabelled as shortest-remaining"
        ),
        "attempted": False,
    },
    {
        "requested_feature": "decode_reservoir_low_watermark=0",
        "status": "obsolete_field_not_mapped",
        "reason": (
            "the shipped field is decode_refill_target; integer zero is invalid and "
            "round-robin cells use 'auto'"
        ),
        "attempted": False,
    },
    {
        "requested_feature": "structural policy/lanes/refill live hot-swap",
        "status": "unsupported_by_post",
        "reason": (
            "POST accepts only prefill_compute_share and prefill_compute_half_life; "
            "structural fields are selected at boot and individually rejection-probed"
        ),
        "attempted": False,
    },
)


class PhaseState:
    def __init__(self) -> None:
        self.invocations: dict[str, dict[str, Any]] = {}
        self.api_discoveries: dict[str, dict[str, Any]] = {}
        self.runtime_facts: dict[str, dict[str, Any]] = {}
        self.structural_rejections: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []
        self.boots_attempted = 0
        self.boots_ready = 0
        self.lmcache_roots: list[dict[str, Any]] = []
        self.resume_enabled = False
        self.resume_started_at: str | None = None
        self.resume_plans: dict[str, dict[str, Any]] = {}
        self.reused_groups: set[str] = set()
        self.booted_groups: set[str] = set()


def out_path(name: str) -> str:
    return f"{OUTPUT_DIR}/{name}"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def record_gate(name: str, passed: bool, detail: object) -> None:
    gate_name = f"scheduler-recheck-{safe_name(name)}"
    runtime.record_gate(gate_name, bool(passed), detail)
    path = runtime.ROOT / OUTPUT_DIR / "gates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "name": gate_name,
                    "passed": bool(passed),
                    "detail": detail,
                    "timestamp": time.time(),
                },
                default=str,
            )
            + "\n"
        )


def source_path(name: str) -> Path:
    under_result_root = runtime.ROOT / name
    if under_result_root.exists():
        return under_result_root
    return DEFAULT_EVIDENCE_ROOT / name


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def source_contract() -> dict[str, Any]:
    config_path = source_path("overlay-source-config-scheduler.py")
    scheduler_path = source_path("overlay-source-v1-core-sched-scheduler.py")
    arg_utils_path = source_path("peer-source-arg_utils.py")
    official_path = source_path(
        "scheduler/official-r26-scheduler-vram-api-discovery.json"
    )
    overlay_receipt_path = source_path(
        "scheduler/drock-r26-overlay-scheduler-vram-api-discovery.json"
    )
    official_post_path = source_path(
        "scheduler/official-static04-headline-policy.json"
    )
    required_config = (
        'PrefillPolicy = Literal["round-robin", "decode-aware"]',
        'MaxParallelPrefills = Annotated[int, Field(ge=1)] | Literal["auto"]',
        'DecodeRefillTarget = Annotated[int, Field(ge=1)] | Literal["auto"]',
        "prefill_compute_share: PrefillComputeShare | None = None",
        "prefill_compute_half_life: PrefillComputeHalfLife | None = None",
        "max_parallel_prefills: MaxParallelPrefills = 1",
        'prefill_policy: PrefillPolicy = "round-robin"',
        'decode_refill_target: DecodeRefillTarget = "auto"',
    )
    required_scheduler = (
        '"effective_max_parallel_prefills": self.max_parallel_prefills',
        '"prefill_policy": self.scheduler_config.prefill_policy',
        '"effective_decode_refill_target": self.decode_refill_target',
        'unknown_fields = set(config) - {',
        '"prefill_compute_share",',
        '"prefill_compute_half_life",',
        "return self.get_prefill_fairness()",
    )
    required_boot_args = (
        '"--max-parallel-prefills"',
        '"--prefill-policy"',
        '"--decode-refill-target"',
        "positive_int_or_auto_type",
    )
    evidence_paths = (
        config_path,
        scheduler_path,
        arg_utils_path,
        official_path,
        overlay_receipt_path,
        official_post_path,
    )
    try:
        config_text = config_path.read_text()
        scheduler_text = scheduler_path.read_text()
        arg_utils_text = arg_utils_path.read_text()
        official = json.loads(official_path.read_text())
        overlay_receipt = json.loads(overlay_receipt_path.read_text())
        official_post_receipt = json.loads(official_post_path.read_text())
    except Exception as error:
        return {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "paths": [str(path) for path in evidence_paths],
        }
    official_body = official.get("exchange", {}).get("body", {})
    overlay_body = overlay_receipt.get("exchange", {}).get("body", {})
    official_post_body = official_post_receipt.get("post", {}).get("body")
    missing_config = [value for value in required_config if value not in config_text]
    missing_scheduler = [value for value in required_scheduler if value not in scheduler_text]
    missing_boot_args = [
        value for value in required_boot_args if value not in arg_utils_text
    ]
    official_legacy = isinstance(official_body, dict) and all(
        field in official_body for field in LEGACY_FIELDS
    )
    overlay_direct_get = isinstance(overlay_body, dict) and all(
        field in overlay_body for field in OVERLAY_GET_FIELDS
    )
    official_wrapped_post = (
        isinstance(official_post_body, dict)
        and isinstance(official_post_body.get("config"), dict)
        and official_post_body.get("applied") is True
    )
    passed = (
        not missing_config
        and not missing_scheduler
        and not missing_boot_args
        and official_legacy
        and overlay_direct_get
        and official_wrapped_post
    )
    return {
        "passed": passed,
        "overlay_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "reviewed_ranges": ["22-30", "145-199", "277-318"],
            "missing_required_fragments": missing_config,
        },
        "overlay_scheduler": {
            "path": str(scheduler_path),
            "sha256": sha256_file(scheduler_path),
            "reviewed_range": "1920-2033",
            "missing_required_fragments": missing_scheduler,
            "setter_return": "direct get_prefill_fairness dictionary",
        },
        "overlay_boot_argument_parser": {
            "path": str(arg_utils_path),
            "sha256": sha256_file(arg_utils_path),
            "reviewed_range": "1669-1684",
            "missing_required_fragments": missing_boot_args,
        },
        "official_api_receipt": {
            "path": str(official_path),
            "sha256": sha256_file(official_path),
            "schema": "legacy-r26" if official_legacy else "unrecognized",
            "captured_get": official_body,
            "get_response_shape": "direct config dictionary",
        },
        "overlay_api_receipt": {
            "path": str(overlay_receipt_path),
            "sha256": sha256_file(overlay_receipt_path),
            "schema": "overlay-r26" if overlay_direct_get else "unrecognized",
            "captured_get": overlay_body,
            "get_response_shape": "direct config and effective-state dictionary",
        },
        "http_post_wrapper_receipt": {
            "path": str(official_post_path),
            "sha256": sha256_file(official_post_path),
            "observed_shape": (
                "body.applied plus body.config"
                if official_wrapped_post
                else "unrecognized"
            ),
            "captured_body": official_post_body,
            "runtime_rule": (
                "record the exact body shape; qualify success from 2xx plus an "
                "independent GET, without inventing or requiring applied"
            ),
        },
    }


def overlay_expected(selected: Policy) -> dict[str, Any]:
    return {
        "prefill_compute_share": selected.compute_share,
        "prefill_compute_half_life": selected.half_life,
        "max_parallel_prefills": selected.max_parallel_prefills,
        "prefill_policy": selected.prefill_policy,
        "decode_refill_target": selected.decode_refill_target,
    }


def legacy_expected(selected: Policy) -> dict[str, Any]:
    if (
        selected.compute_share == "auto"
        or selected.half_life is not None
        or selected.max_parallel_prefills != 1
        or selected.prefill_policy != "round-robin"
        or selected.decode_refill_target != "auto"
    ):
        raise ValueError(f"policy {selected.key} cannot be represented by legacy R26")
    return {
        "fairness_engine": (
            None if selected.compute_share is None else "compute_share"
        ),
        "prefill_compute_share": selected.compute_share,
        "max_num_prefill_tokens_per_step": 0,
        "max_num_partial_prefills": 0,
        "decode_prefill_min_decode_steps": 0,
        "decode_prefill_max_wait_ms": 0,
    }


def expected_config(candidate: Candidate, selected: Policy) -> dict[str, Any]:
    if candidate.api_schema == "overlay-r26":
        return overlay_expected(selected)
    if candidate.api_schema == "legacy-r26":
        return legacy_expected(selected)
    raise ValueError(f"unknown candidate schema {candidate.api_schema}")


def mutable_payload(candidate: Candidate, selected: Policy) -> dict[str, Any]:
    expected = expected_config(candidate, selected)
    if candidate.api_schema == "overlay-r26":
        return {field: expected[field] for field in OVERLAY_MUTABLE_FIELDS}
    return expected


def boot_args(group: BootGroup) -> list[str]:
    if group.candidate.api_schema != "overlay-r26":
        return []
    selected = group.policy_structure
    return [
        "--max-parallel-prefills",
        str(selected.max_parallel_prefills),
        "--prefill-policy",
        selected.prefill_policy,
        "--decode-refill-target",
        str(selected.decode_refill_target),
    ]


def boot_environment(group: BootGroup, dma_value: str | None = None) -> dict[str, str]:
    env = dict(BASE_ENV)
    env["MAX_NUM_BATCHED_TOKENS"] = str(group.batch_tokens)
    requested_dma = group.dma_value if dma_value is None else dma_value
    if requested_dma is not None:
        env["VLLM_PCIE_DMA_MIN_BYTES"] = requested_dma
    return env


def plan_measurement_seconds(plan: WorkloadPlan) -> float:
    return plan.repeats * len(plan.concurrencies) * sum(
        PROFILE_SECONDS[profile] for profile in plan.profiles
    )


def plan_scenario_count(plan: WorkloadPlan) -> int:
    return plan.repeats * len(plan.concurrencies) * len(plan.profiles)


def plan_to_json(plan: WorkloadPlan) -> dict[str, Any]:
    return {
        **asdict(plan),
        "profiles": list(plan.profiles),
        "concurrencies": list(plan.concurrencies),
        "updates": list(plan.updates),
        "scenario_count": plan_scenario_count(plan),
        "measurement_seconds": plan_measurement_seconds(plan),
        "expected_configured_readback": expected_config(
            OFFICIAL if plan.candidate_key == OFFICIAL.key else OVERLAY,
            plan.policy,
        ),
    }


def group_to_json(group: BootGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "candidate": asdict(group.candidate),
        "policy_structure": asdict(group.policy_structure),
        "cache_mode": group.cache_mode,
        "batch_tokens": group.batch_tokens,
        "dma_value": group.dma_value,
        "factor_role": group.factor_role,
        "boot_environment": boot_environment(group),
        "boot_extra_args": boot_args(group),
        "plans": [plan_to_json(plan) for plan in group.plans],
        "service_isolation": "one complete boot/service lifetime for this structural configuration",
    }


def validate_phase_plan() -> dict[str, Any]:
    names = [plan.name for plan in ALL_PLANS]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    all_concurrencies = sorted(
        {concurrency for plan in ALL_PLANS for concurrency in plan.concurrencies}
    )
    all_profiles = sorted({profile for plan in ALL_PLANS for profile in plan.profiles})
    headline_repeat_errors = [
        plan.name for plan in ALL_PLANS if plan.headline and plan.repeats != 2
    ]
    coverage_repeat_errors = [
        plan.name for plan in ALL_PLANS if not plan.headline and plan.repeats != 1
    ]
    structural_mismatches = [
        plan.name
        for group in BOOT_GROUPS
        for plan in group.plans
        if plan.candidate_key != group.candidate.key
        or (
            group.candidate.api_schema == "overlay-r26"
            and structural_key(plan.policy) != structural_key(group.policy_structure)
        )
    ]
    update_field_errors = [
        {"plan": plan.name, "update": update}
        for plan in ALL_PLANS
        for update in plan.updates
        if not isinstance(update.get("config"), dict)
        or set(update["config"]) != set(OVERLAY_MUTABLE_FIELDS)
    ]
    automatic_matrix = {
        (
            plan.policy.half_life,
            plan.policy.prefill_policy,
            plan.policy.max_parallel_prefills,
        )
        for plan in OVERLAY_AUTOMATIC_PLANS
        if plan.series in {"overlay-auto-headline", "overlay-auto-factorial"}
    }
    expected_matrix = set(
        product(
            ("responsive", "smooth"),
            ("round-robin", "decode-aware"),
            ("auto", 4),
        )
    )
    # itertools.product order above differs from construction order but tuple sets agree.
    factor_observed = {
        (group.batch_tokens, group.dma_value, group.factor_role)
        for group in BOOT_GROUPS
        if group.factor_role is not None
    }
    factor_expected = {
        (batch, dma, role) for _, batch, dma, role in FACTOR_SPECS
    }
    source = source_contract()
    pinned_images = (
        runtime.IMAGE
        == "voipmonitor/vllm@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68"
        and runtime.OVERLAY_IMAGE
        == "ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:27fe7a2f1df6d01e824cd24d6b83119edea471f4ec4997aa122d82c670133236"
    )
    lmcache_plans = [plan for plan in ALL_PLANS if plan.cache_mode == "lmcache"]
    lmcache_matched = (
        len(lmcache_plans) == 2
        and len({plan.cache_namespace for plan in lmcache_plans}) == 1
        and {plan.memory_role for plan in lmcache_plans} == {"writer", "reader"}
    )
    checks = {
        "unique_plan_names": not duplicate_names,
        "only_required_session_counts": all_concurrencies == [8, 16],
        "all_required_profiles": set(all_profiles) == set(PROFILE_SECONDS),
        "headline_two_repeats": not headline_repeat_errors,
        "coverage_one_repeat": not coverage_repeat_errors,
        "boot_structure_matches_plans": not structural_mismatches,
        "live_updates_only_two_mutable_fields": not update_field_errors,
        "automatic_factorial_complete": automatic_matrix == expected_matrix,
        "batch_and_dma_factors_complete": factor_observed == factor_expected,
        "unsupported_modes_explicit": len(UNSUPPORTED_REQUESTED_MODES) == 4
        and all(item["attempted"] is False for item in UNSUPPORTED_REQUESTED_MODES),
        "source_contract_matched": source.get("passed") is True,
        "loopback_only": runtime.BASE_URL.startswith("http://127.0.0.1:"),
        "pinned_candidate_images": pinned_images,
        "workload_tool_present": WORKLOAD.is_file(),
        "r25_reference_not_scheduled": all(
            group.candidate.image != runtime.R25_IMAGE for group in BOOT_GROUPS
        ),
        "lmcache_writer_reader_matched": lmcache_matched,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "plan_count": len(ALL_PLANS),
            "scenario_count": sum(plan_scenario_count(plan) for plan in ALL_PLANS),
            "measurement_seconds": sum(
                plan_measurement_seconds(plan) for plan in ALL_PLANS
            ),
            "duplicate_names": duplicate_names,
            "concurrencies": all_concurrencies,
            "profiles": all_profiles,
            "headline_repeat_errors": headline_repeat_errors,
            "coverage_repeat_errors": coverage_repeat_errors,
            "structural_mismatches": structural_mismatches,
            "update_field_errors": update_field_errors,
            "automatic_matrix_observed": [list(value) for value in sorted(automatic_matrix, key=str)],
            "automatic_matrix_expected": [list(value) for value in sorted(expected_matrix, key=str)],
            "factor_observed": [list(value) for value in sorted(factor_observed, key=str)],
            "factor_expected": [list(value) for value in sorted(factor_expected, key=str)],
            "source_contract": source,
            "pinned_images": pinned_images,
            "workload_tool": str(WORKLOAD),
            "lmcache_matched": lmcache_matched,
        },
    }


def phase_plan() -> dict[str, Any]:
    validation = validate_phase_plan()
    all_vram = [plan for plan in ALL_PLANS if plan.cache_mode == "vram"]
    lmcache = [plan for plan in ALL_PLANS if plan.cache_mode == "lmcache"]
    factors = [
        plan
        for group in BOOT_GROUPS
        if group.factor_role is not None
        for plan in group.plans
    ]
    factor_names = {plan.name for plan in factors}
    scheduler_vram = [plan for plan in all_vram if plan.name not in factor_names]
    return {
        "phase": "scheduler-recheck",
        "output_directory": str(runtime.ROOT / OUTPUT_DIR),
        "trace_seed": TRACE_SEED,
        "validation": validation,
        "source_contract": source_contract(),
        "candidate_and_tool_settings": {
            "candidates": [asdict(OFFICIAL), asdict(OVERLAY)],
            "official_image": runtime.IMAGE,
            "overlay_image": runtime.OVERLAY_IMAGE,
            "r25_reference_image_not_run": runtime.R25_IMAGE,
            "model_directory": str(runtime.MODEL),
            "served_model_name": runtime.MODEL_NAME,
            "model_override_environment": "BATTERY_MODEL_DIR",
            "model_cache_tag": runtime.MODEL_CACHE_TAG,
            "workload_tool": str(WORKLOAD),
            "python": sys.executable,
            "base_url": runtime.BASE_URL,
            "container": runtime.NAME,
            "tp": 4,
            "dcp": 4,
            "speculation": "mtp0",
            "kv": "fp8_ds_mla",
            "max_num_seqs": 32,
            "development_api_loopback_only": True,
        },
        "api_contract": {
            "overlay_get_required_fields": list(OVERLAY_GET_FIELDS),
            "overlay_post_only_fields": list(OVERLAY_MUTABLE_FIELDS),
            "overlay_structural_boot_fields": list(OVERLAY_STRUCTURAL_FIELDS),
            "overlay_allowed_prefill_policy": ["round-robin", "decode-aware"],
            "overlay_max_parallel_prefills": "integer >=1 or 'auto'",
            "overlay_decode_refill_target": "positive integer or 'auto'",
            "single_lane_semantics": (
                "max_parallel_prefills=1 plus round-robin is legacy single-prefill behavior"
            ),
            "round_robin_refill": "decode_refill_target='auto'; no explicit zero",
            "post_success": (
                "HTTP 2xx plus independent GET readback; an applied field is recorded "
                "when present but never invented or required"
            ),
            "structural_rejection": (
                "each structural field is POSTed individually, must receive 4xx, and "
                "configured state must remain unchanged"
            ),
            "official_legacy_get_fields": list(LEGACY_FIELDS),
        },
        "unsupported_requested_modes": list(UNSUPPORTED_REQUESTED_MODES),
        "workload_contract": {
            "profiles": {
                "baseline": "cyclic 8K-context agent sessions without injected cold requests",
                "periodic-128k": "unique 128K cold prefill every 15 seconds",
                "short-prefill-heavy": "2K/4K/8K cold prefills every 1.5 seconds",
                "analytics-200k-burst": "unique 200K cold prefills at offsets 0/1/2 seconds",
            },
            "sessions": [8, 16],
            "session_context_tokens": 8192,
            "turn_decode_tokens": 128,
            "turn_period_seconds": 6,
            "headline_repeats": 2,
            "coverage_repeats": 1,
            "raw_request_events": "each result cell's requests array",
            "trace_equality": (
                "cache namespace excluded from hash; prompt bodies, token counts, payload "
                "shapes, offsets, profiles, concurrency, and repeat remain included"
            ),
            "answer_classification": (
                "protocol completion and length-budget completion are separate; model "
                "semantic correctness is not assessed"
            ),
        },
        "boot_groups": [group_to_json(group) for group in BOOT_GROUPS],
        "plans": [plan_to_json(plan) for plan in ALL_PLANS],
        "plan_size": {
            "boot_groups": len(BOOT_GROUPS),
            "plans": len(ALL_PLANS),
            "scenarios": sum(plan_scenario_count(plan) for plan in ALL_PLANS),
            "scheduler_vram_scenarios": sum(
                plan_scenario_count(plan) for plan in scheduler_vram
            ),
            "all_vram_scenarios_including_factors": sum(
                plan_scenario_count(plan) for plan in all_vram
            ),
            "factor_scenarios": sum(plan_scenario_count(plan) for plan in factors),
            "lmcache_scenarios": sum(plan_scenario_count(plan) for plan in lmcache),
            "measurement_seconds": sum(
                plan_measurement_seconds(plan) for plan in ALL_PLANS
            ),
        },
        "factor_contract": {
            "batch_tokens": [4096, 8192, 12288, 16384],
            "dma_512kb": {
                "isolated": "4096 batch tokens versus the 4096 default-DMA control",
                "combined": "12288 batch tokens plus 512KB threshold",
                "literal": (
                    "try 512KB; use 524288 only after captured parser rejection"
                ),
            },
            "identical_workload": "8-session periodic-128K, one coverage repeat",
            "capture": (
                "configured/effective lane budget, queue/backlog non-progress, scheduler "
                "compute counters, and raw request events"
            ),
        },
        "lmcache_confirmation": {
            "separate_from_vram": True,
            "writer": "Official R26 static0.4 single-lane",
            "reader": "Overlay static0.4 single-lane",
            "same_trace_and_cache_namespace": True,
            "shared_runtime_root": str(runtime.L2_SHARED),
            "transfer_evidence": (
                "external-prefix query/hit and prompt_tokens source counters; response text "
                "is never treated as KV-byte identity evidence"
            ),
            "failed_retrieval_interpretation": "absence of transfer evidence, not corruption",
        },
    }


def detect_api_schema(config: object) -> str:
    if not isinstance(config, dict):
        return "unknown"
    if all(field in config for field in OVERLAY_GET_FIELDS):
        return "overlay-r26"
    if all(field in config for field in LEGACY_FIELDS):
        return "legacy-r26"
    return "unknown"


def extract_config(exchange: dict[str, Any]) -> dict[str, Any] | None:
    body = exchange.get("body")
    if not isinstance(body, dict):
        return None
    nested = body.get("config")
    return nested if isinstance(nested, dict) else body


def response_shape(body: object) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {
            "body_type": type(body).__name__,
            "top_level_keys": [],
            "config_location": None,
            "config_keys": [],
            "applied_marker_present": False,
            "applied_marker": None,
        }
    nested = body.get("config")
    known_fields = set(OVERLAY_GET_FIELDS) | set(LEGACY_FIELDS)
    direct_config = bool(known_fields.intersection(body))
    config = nested if isinstance(nested, dict) else body if direct_config else None
    return {
        "body_type": "dict",
        "top_level_keys": sorted(body),
        "config_location": (
            "body.config"
            if isinstance(nested, dict)
            else "body"
            if direct_config
            else None
        ),
        "config_keys": sorted(config) if config is not None else [],
        "applied_marker_present": "applied" in body,
        "applied_marker": body.get("applied"),
    }


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
            "response_shape": response_shape(None),
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
        "response_shape": response_shape(body),
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
    actual: dict[str, Any] | None, expected: dict[str, Any]
) -> dict[str, dict[str, object]]:
    if actual is None:
        return {"config": {"expected": expected, "actual": None}}
    return {
        field: {"expected": wanted, "actual": actual.get(field)}
        for field, wanted in expected.items()
        if not values_equal(actual.get(field), wanted)
    }


def discover_api(
    state: PhaseState, group: BootGroup, label: str
) -> dict[str, Any]:
    exchange = http_json("GET", "/prefill_fairness")
    config = extract_config(exchange)
    schema = (
        detect_api_schema(config)
        if exchange.get("status_code") == 200
        else "unavailable"
    )
    structural_expected = (
        {
            field: overlay_expected(group.policy_structure)[field]
            for field in OVERLAY_STRUCTURAL_FIELDS
        }
        if group.candidate.api_schema == "overlay-r26"
        else {}
    )
    structural_mismatches = config_mismatches(config, structural_expected)
    effective_lane_budget = (
        config.get("effective_max_parallel_prefills")
        if isinstance(config, dict)
        else None
    )
    effective_refill_target = (
        config.get("effective_decode_refill_target")
        if isinstance(config, dict)
        else None
    )
    effective_state_coherent = True
    if group.candidate.api_schema == "overlay-r26":
        configured_lanes = group.policy_structure.max_parallel_prefills
        configured_refill = group.policy_structure.decode_refill_target
        # Both explicit and auto lane counts are upper bounds. The shipped
        # resolver clips them to floor(scheduled_tokens / cache_block_size).
        effective_state_coherent = (
            isinstance(effective_lane_budget, int)
            and not isinstance(effective_lane_budget, bool)
            and effective_lane_budget >= 1
            and (
                (
                    configured_lanes == "auto"
                    and effective_lane_budget <= 4
                )
                or (
                    isinstance(configured_lanes, int)
                    and effective_lane_budget <= configured_lanes
                )
            )
            and isinstance(effective_refill_target, int)
            and not isinstance(effective_refill_target, bool)
            and effective_refill_target >= 1
            and (
                (
                    configured_refill == "auto"
                    and effective_refill_target == effective_lane_budget
                )
                or effective_refill_target == configured_refill
            )
        )
    discovery = {
        "candidate": asdict(group.candidate),
        "boot_group": group_to_json(group),
        "schema": schema,
        "exchange": exchange,
        "expected_schema": group.candidate.api_schema,
        "overlay_get_fields": list(OVERLAY_GET_FIELDS),
        "legacy_fields": list(LEGACY_FIELDS),
        "expected_boot_structure": structural_expected,
        "boot_structure_mismatches": structural_mismatches,
        "effective_lane_budget": effective_lane_budget,
        "effective_decode_refill_target": effective_refill_target,
        "effective_state_coherent": effective_state_coherent,
    }
    passed = (
        schema == group.candidate.api_schema
        and not structural_mismatches
        and effective_state_coherent
    )
    discovery["passed"] = passed
    runtime.save_json(out_path(f"{safe_name(group.name)}-api-discovery.json"), discovery)
    state.api_discoveries[group.name] = discovery
    record_gate(f"{group.name}-api-and-boot-structure", passed, discovery)
    return discovery


def configured_projection(config: dict[str, Any] | None, schema: str) -> object:
    if config is None:
        return None
    fields = OVERLAY_CONFIGURED_FIELDS if schema == "overlay-r26" else LEGACY_FIELDS
    return {field: config.get(field) for field in fields}


def check_structural_post_rejections(
    state: PhaseState, group: BootGroup
) -> dict[str, Any]:
    before = http_json("GET", "/prefill_fairness")
    before_config = extract_config(before)
    mutable = {
        field: before_config.get(field) if before_config is not None else None
        for field in OVERLAY_MUTABLE_FIELDS
    }
    current_lanes = before_config.get("max_parallel_prefills") if before_config else None
    current_policy = before_config.get("prefill_policy") if before_config else None
    current_refill = before_config.get("decode_refill_target") if before_config else None
    probes = (
        (
            "max_parallel_prefills",
            1 if current_lanes != 1 else 4,
        ),
        (
            "prefill_policy",
            "decode-aware" if current_policy != "decode-aware" else "round-robin",
        ),
        (
            "decode_refill_target",
            2 if current_refill != 2 else 3,
        ),
    )
    events: list[dict[str, Any]] = []
    original = configured_projection(before_config, "overlay-r26")
    for field, value in probes:
        request = {**mutable, field: value}
        posted = http_json("POST", "/prefill_fairness", request)
        after = http_json("GET", "/prefill_fairness")
        after_config = extract_config(after)
        after_projection = configured_projection(after_config, "overlay-r26")
        status = posted.get("status_code")
        rejected = isinstance(status, int) and 400 <= status < 500
        unchanged = after_projection == original
        events.append(
            {
                "field": field,
                "attempted_value": value,
                "before": before if not events else None,
                "post": posted,
                "after": after,
                "http_4xx_rejection": rejected,
                "configured_state_unchanged": unchanged,
                "passed": rejected and unchanged,
            }
        )
    receipt = {
        "boot_group": group.name,
        "only_supported_post_fields": list(OVERLAY_MUTABLE_FIELDS),
        "structural_fields": list(OVERLAY_STRUCTURAL_FIELDS),
        "original_configured_state": original,
        "events": events,
        "passed": len(events) == len(OVERLAY_STRUCTURAL_FIELDS)
        and all(event["passed"] for event in events),
        "structural_hot_swap_claimed": False,
    }
    runtime.save_json(
        out_path(f"{safe_name(group.name)}-structural-post-rejections.json"),
        receipt,
    )
    state.structural_rejections[group.name] = receipt
    record_gate(f"{group.name}-structural-post-rejections", receipt["passed"], receipt)
    return receipt


def configure_policy(candidate: Candidate, plan: WorkloadPlan) -> dict[str, Any]:
    before = http_json("GET", "/prefill_fairness")
    before_config = extract_config(before)
    before_schema = (
        detect_api_schema(before_config)
        if before.get("status_code") == 200
        else "unavailable"
    )
    expected = expected_config(candidate, plan.policy)
    payload = mutable_payload(candidate, plan.policy)
    receipt: dict[str, Any] = {
        "candidate": asdict(candidate),
        "plan": plan_to_json(plan),
        "schema": before_schema,
        "before": before,
        "expected_configured_readback": expected,
        "posted_fields": sorted(payload),
        "payload": payload,
        "unsupported": None,
    }
    if before_schema != candidate.api_schema:
        receipt.update(
            {
                "passed": False,
                "failure_class": "harness_or_api_schema",
                "reason": (
                    f"expected {candidate.api_schema}, observed {before_schema}"
                ),
            }
        )
    else:
        posted = http_json("POST", "/prefill_fairness", payload)
        after = http_json("GET", "/prefill_fairness")
        after_config = extract_config(after)
        after_schema = (
            detect_api_schema(after_config)
            if after.get("status_code") == 200
            else "unavailable"
        )
        mismatches = config_mismatches(after_config, expected)
        status = posted.get("status_code")
        body = posted.get("body")
        marker = body.get("applied") if isinstance(body, dict) else None
        passed = (
            isinstance(status, int)
            and 200 <= status < 300
            and marker is not False
            and after_schema == candidate.api_schema
            and not mismatches
        )
        receipt.update(
            {
                "post": posted,
                "after": after,
                "after_schema": after_schema,
                "effective_readback": after_config,
                "mismatches": mismatches,
                "wrapper_applied_marker": marker,
                "applied_marker_required": False,
                "success_basis": "2xx POST and independent GET readback",
                "passed": passed,
                "failure_class": None if passed else "harness_or_scheduler_api_contract",
            }
        )
    runtime.save_json(out_path(f"{plan.name}-policy.json"), receipt)
    record_gate(
        f"{plan.name}-policy-effective",
        bool(receipt.get("passed")),
        receipt,
    )
    return receipt


def workload_command(
    candidate: Candidate,
    plan: WorkloadPlan,
    output: Path,
) -> list[str]:
    expected = expected_config(candidate, plan.policy)
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
        candidate.api_schema,
        "--expected-policy-json",
        json.dumps(expected, separators=(",", ":")),
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
        "--cache-mode",
        plan.cache_mode,
        "--memory-role",
        plan.memory_role,
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

def cell_identity(cell: dict[str, Any]) -> tuple[str, int, int] | None:
    try:
        return (
            str(cell.get("profile")),
            int(cell.get("concurrency")),
            int(cell.get("repeat")),
        )
    except (TypeError, ValueError):
        return None


def identity_label(key: tuple[str, int, int]) -> str:
    return f"{key[0]}/c{key[1]}/r{key[2]}"


def identity_record(key: tuple[str, int, int]) -> dict[str, Any]:
    return {"profile": key[0], "concurrency": key[1], "repeat": key[2]}


def cell_terminal(cell: dict[str, Any]) -> bool:
    """True for a cell that finished its measurement window and evaluated gates.

    Boundary/prime failures and harness exceptions never measured anything;
    resume re-executes those and reuses terminal cells verbatim, including
    cells whose observed qualification gates failed. Keep in sync with
    agent_workload_recheck.py.
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


def plan_grid(plan: WorkloadPlan) -> list[tuple[str, int, int]]:
    return [
        (profile, concurrency, repeat)
        for repeat in range(1, plan.repeats + 1)
        for concurrency in plan.concurrencies
        for profile in plan.profiles
    ]


def cell_measurement_era(
    metadata: dict[str, Any], key: tuple[str, int, int]
) -> str:
    """Era of a reused cell, walked through any prior resume provenance chain.

    Cells listed as re-executed by a prior post-reboot resume keep that era;
    everything else was measured in the original pre-reboot run.
    """
    block = metadata.get("resume")
    while isinstance(block, dict):
        for entry in block.get("reexecuted_cells") or []:
            if isinstance(entry, dict) and cell_identity(entry) == key:
                return RESUME_PROVENANCE_RESUMED
        block = block.get("prior_resume")
    return RESUME_PROVENANCE_REUSED


def resume_metadata_mismatches(
    group: BootGroup, plan: WorkloadPlan, metadata: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output = runtime.ROOT / OUTPUT_DIR / f"{plan.name}.json"
    command = workload_command(group.candidate, plan, output)
    expected = workload.build_metadata(workload.parse_args(command[2:]))
    return workload.resume_metadata_mismatches(metadata, expected)


def resume_boot_identity(group: BootGroup) -> dict[str, Any]:
    """Validate the copied pre-reboot launch artifact for a reused group.

    The workload metadata only carries the generic served model name, so reuse
    is attributed to the pinned boot by checking the immutable image digest and
    checkpoint directory recorded at boot time against the current runtime.
    """
    path = runtime.ROOT / OUTPUT_DIR / f"{group.name}.launch.json"
    receipt: dict[str, Any] = {
        "boot_group": group.name,
        "launch_receipt": str(path),
        "launch_receipt_sha256": sha256_file(path),
        "matched": False,
        "mismatches": {},
        "problems": [],
    }
    if not path.is_file():
        receipt["problems"].append("launch receipt is missing")
        return receipt
    parsed, parse_error = read_json_file(path)
    if parsed is None:
        receipt["problems"].append(f"unreadable launch receipt: {parse_error}")
        return receipt
    expected = {
        "label": out_path(group.name),
        "image": group.candidate.image,
        "tp": 4,
        "dcp": 4,
        "spec": "mtp0",
        "cache": group.cache_mode,
        "kv": "fp8_ds_mla",
        "model_dir": str(runtime.MODEL),
    }
    mismatches = {
        field: {"saved": parsed.get(field), "expected": wanted}
        for field, wanted in expected.items()
        if parsed.get(field) != wanted
    }
    env = parsed.get("env")
    if isinstance(env, dict):
        mismatches.update(
            {
                f"env.{field}": {"saved": env.get(field), "expected": wanted}
                for field, wanted in (
                    ("SERVED_MODEL_NAME", runtime.MODEL_NAME),
                    ("MAX_NUM_BATCHED_TOKENS", str(group.batch_tokens)),
                )
                if env.get(field) != wanted
            }
        )
    else:
        mismatches["env"] = {"saved": env, "expected": "object"}
    receipt["mismatches"] = mismatches
    receipt["matched"] = not mismatches
    return receipt


def triage_plan_receipt(group: BootGroup, plan: WorkloadPlan) -> dict[str, Any]:
    """Classify a plan's prior receipt for --resume.

    absent: no receipt; reusable_complete: every canonical scenario present
    exactly once and terminal; partial: some terminal cells plus missing or
    non-terminal ones; invalid: corrupt, duplicate, identity-mismatched, or
    configuration-mismatched receipts, which fail closed before any boot.
    """
    path = runtime.ROOT / OUTPUT_DIR / f"{plan.name}.json"
    grid = plan_grid(plan)
    triage: dict[str, Any] = {
        "plan": plan.name,
        "boot_group": group.name,
        "receipt": str(path),
        "receipt_sha256": sha256_file(path),
        "state": "absent",
        "problems": [],
        "metadata_mismatches": {},
        "grid": [list(key) for key in grid],
        "reused_cells": [],
        "missing_cells": [identity_record(key) for key in grid],
        "dropped_nonterminal_cells": [],
    }
    if not path.is_file():
        return triage
    parsed, parse_error = read_json_file(path)
    if parsed is None:
        triage["state"] = "invalid"
        triage["problems"].append(f"unreadable receipt: {parse_error}")
        return triage
    triage["parsed"] = parsed
    metadata = parsed.get("metadata")
    cells = parsed.get("cells")
    if not isinstance(metadata, dict) or not isinstance(cells, list):
        triage["state"] = "invalid"
        triage["problems"].append(
            "receipt must contain a metadata object and a cells list"
        )
        return triage
    triage["metadata"] = metadata
    triage["metadata_mismatches"] = resume_metadata_mismatches(group, plan, metadata)
    if triage["metadata_mismatches"]:
        triage["state"] = "invalid"
        return triage
    problems: list[str] = []
    grid_set = set(grid)
    reused: dict[tuple[str, int, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int], int] = {}
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            problems.append(f"cells[{index}] is not an object")
            continue
        key = cell_identity(cell)
        if key is None or key not in grid_set:
            problems.append(
                f"cells[{index}] identity {key} is outside the canonical plan grid"
            )
            continue
        if key in seen:
            problems.append(
                f"cells[{index}] duplicates cells[{seen[key]}] for {identity_label(key)}"
            )
            continue
        seen[key] = index
        if cell.get("candidate") != group.candidate.label:
            problems.append(
                f"cells[{index}] ({identity_label(key)}) candidate label mismatch"
            )
            continue
        if cell.get("policy") != plan.policy.key:
            problems.append(
                f"cells[{index}] ({identity_label(key)}) policy label mismatch"
            )
            continue
        if not cell_terminal(cell):
            if cell.get("status") == "complete":
                problems.append(
                    f"cells[{index}] ({identity_label(key)}) claims status complete "
                    "without a full measurement receipt"
                )
            else:
                dropped.append(
                    {
                        "index": index,
                        **identity_record(key),
                        "status": cell.get("status"),
                        "failure_class": cell.get("failure_class"),
                    }
                )
            continue
        trace = cell["trace"]
        if trace.get("trace_seed") != TRACE_SEED:
            problems.append(
                f"cells[{index}] ({identity_label(key)}) trace seed mismatch"
            )
            continue
        if trace.get("cache_namespace") != plan.cache_namespace:
            problems.append(
                f"cells[{index}] ({identity_label(key)}) cache namespace mismatch"
            )
            continue
        reused[key] = cell
    if problems:
        triage["state"] = "invalid"
        triage["problems"] = problems
        return triage
    missing = [key for key in grid if key not in reused]
    triage["reused_cells"] = [
        {
            **identity_record(key),
            "status": reused[key].get("status"),
            "failure_class": reused[key].get("failure_class"),
            "trace_hash": reused[key]["trace"]["trace_hash"],
            "measurement_provenance": cell_measurement_era(metadata, key),
        }
        for key in grid
        if key in reused
    ]
    triage["missing_cells"] = [identity_record(key) for key in missing]
    triage["dropped_nonterminal_cells"] = dropped
    triage["state"] = "reusable_complete" if not missing and not dropped else "partial"
    return triage


def load_resume_receipts() -> dict[str, Any]:
    boot_identities = {
        group.name: resume_boot_identity(group) for group in BOOT_GROUPS
    }
    plans: dict[str, dict[str, Any]] = {}
    for group in BOOT_GROUPS:
        for plan in group.plans:
            triage = triage_plan_receipt(group, plan)
            # A receipt may only be reused under the boot that produced it.
            if triage["state"] in {"reusable_complete", "partial"}:
                identity = boot_identities[group.name]
                triage["boot_identity"] = identity
                if not identity["matched"]:
                    triage["state"] = "invalid"
                    triage["problems"].extend(identity["problems"])
                    triage["problems"].extend(
                        f"launch {field} mismatch: "
                        + json.dumps(detail, sort_keys=True, default=str)
                        for field, detail in sorted(identity["mismatches"].items())
                    )
            plans[plan.name] = triage
    invalid = sorted(
        name for name, triage in plans.items() if triage["state"] == "invalid"
    )
    states = [triage["state"] for triage in plans.values()]
    return {
        "passed": not invalid,
        "invalid_plans": invalid,
        "counts": {
            "reusable_complete": states.count("reusable_complete"),
            "partial": states.count("partial"),
            "absent": states.count("absent"),
            "invalid": len(invalid),
        },
        "boot_identities": boot_identities,
        "plans": plans,
    }


def archive_resume_evidence(resume_state: dict[str, Any]) -> None:
    """Keep prior boot, policy, invocation and cell evidence before reuse."""
    source = runtime.ROOT / OUTPUT_DIR
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise RuntimeError("Refusing symlinks in scheduler resume evidence")
    archive = runtime.ROOT / f"{OUTPUT_DIR}-pre-resume-{time.time_ns()}"
    shutil.copytree(source, archive, copy_function=shutil.copy2)
    for identity in resume_state["boot_identities"].values():
        prior = Path(identity["launch_receipt"])
        if not prior.is_file():
            continue
        preserved = archive / prior.relative_to(source)
        if sha256_file(preserved) != identity["launch_receipt_sha256"]:
            raise RuntimeError(f"Archived boot evidence changed: {prior}")
        identity["launch_receipt"] = str(preserved)
    for triage in resume_state["plans"].values():
        if triage["state"] not in {"reusable_complete", "partial"}:
            continue
        prior = Path(triage["receipt"])
        preserved = archive / prior.relative_to(source)
        if sha256_file(preserved) != triage["receipt_sha256"]:
            raise RuntimeError(f"Archived scenario evidence changed: {prior}")
        triage["working_receipt"] = str(prior)
        triage["receipt"] = str(preserved)
    resume_state["prior_evidence_root"] = str(archive)


def reused_configuration_reference(
    plan: WorkloadPlan, triage: dict[str, Any]
) -> dict[str, Any]:
    path = Path(triage["receipt"]).with_name(f"{plan.name}-policy.json")
    parsed, parse_error = read_json_file(path)
    reference: dict[str, Any] = {
        "provenance": "reused_pre_reboot_receipt",
        "receipt": str(path),
        "receipt_sha256": sha256_file(path),
        "passed": parsed.get("passed") if isinstance(parsed, dict) else None,
    }
    if parse_error is not None:
        reference["parse_error"] = parse_error
    return reference


def classify_plan_cells(
    plan: WorkloadPlan, cells: list[Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    protocol_failures: list[dict[str, Any]] = []
    progress_failures: list[dict[str, Any]] = []
    update_failures: list[dict[str, Any]] = []
    transfer_cells: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            protocol_failures.append({"cell": cell, "reason": "not an object"})
            continue
        identity = {
            "profile": cell.get("profile"),
            "concurrency": cell.get("concurrency"),
            "repeat": cell.get("repeat"),
        }
        protocol_gate = find_cell_gate(
            cell, "all_stream_protocol_responses_complete"
        )
        if not protocol_gate or protocol_gate.get("passed") is not True:
            protocol_failures.append({**identity, "gate": protocol_gate})
        progress_gate = find_cell_gate(
            cell, "no_zero_prefill_quantum_signature"
        )
        if not progress_gate or progress_gate.get("passed") is not True:
            progress_failures.append({**identity, "gate": progress_gate})
        if plan.updates:
            update_gate = find_cell_gate(
                cell, "mutable_compute_fields_updated_during_active_work"
            )
            if not update_gate or update_gate.get("passed") is not True:
                update_failures.append({**identity, "gate": update_gate})
        transfer = cell.get("summary", {}).get("memory_transfer_evidence")
        if isinstance(transfer, dict):
            transfer_cells.append({**identity, "evidence": transfer})
    return protocol_failures, progress_failures, update_failures, transfer_cells


def reused_invocation_record(
    state: PhaseState, group: BootGroup, plan: WorkloadPlan, triage: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild a plan record from a fully reusable pre-reboot receipt."""
    parsed = triage["parsed"]
    cells = [cell for cell in parsed.get("cells", []) if isinstance(cell, dict)]
    expected_cells = plan_scenario_count(plan)
    complete_cells = [cell for cell in cells if cell.get("status") == "complete"]
    protocol_failures, progress_failures, update_failures, transfer_cells = (
        classify_plan_cells(plan, cells)
    )
    all_attempted = len(cells) == expected_cells
    workload_passed = all_attempted and len(complete_cells) == expected_cells
    eras = {
        identity_label(key): cell_measurement_era(triage["metadata"], key)
        for key in (cell_identity(cell) for cell in cells)
        if key is not None
    }
    era_values = sorted(set(eras.values()))
    plan_provenance = era_values[0] if len(era_values) == 1 else "mixed_resume"
    record: dict[str, Any] = {
        "candidate": asdict(group.candidate),
        "boot_group": group.name,
        "plan": plan_to_json(plan),
        "cache_mode": plan.cache_mode,
        "configuration": reused_configuration_reference(plan, triage),
        "status": "complete" if workload_passed else "qualification_failed",
        "scenario_attempt_classification": (
            "all_applicable_scenarios_attempted"
            if all_attempted
            else "scenario_set_incomplete_harness_failure"
        ),
        "failure_class": None if workload_passed else "workload_or_qualification",
        "output": triage["receipt"],
        "return_code": None,
        "return_code_provenance": (
            "the workload process exit code is not part of a reused receipt; "
            "qualification is judged from the recorded cells"
        ),
        "parse_error": None,
        "expected_cells": expected_cells,
        "observed_cells": len(cells),
        "complete_cells": len(complete_cells),
        "protocol_failures": protocol_failures,
        "prefill_progress_failures": progress_failures,
        "live_update_failures": update_failures,
        "memory_transfer_evidence": transfer_cells,
        "result": parsed,
        "measurement_provenance": plan_provenance,
        "cell_measurement_provenance": eras,
        "resume": {
            "receipt_sha256": triage["receipt_sha256"],
            "reused_cells": triage["reused_cells"],
            "boot_identity": triage.get("boot_identity"),
        },
    }
    record_gate(
        f"{plan.name}-reused-receipt-validated",
        True,
        {
            "receipt": triage["receipt"],
            "receipt_sha256": triage["receipt_sha256"],
            "measurement_provenance": plan_provenance,
        },
    )
    record_gate(
        f"{plan.name}-all-scenarios-attempted",
        all_attempted,
        {
            "expected": expected_cells,
            "observed": len(cells),
            "provenance": plan_provenance,
            "output": triage["receipt"],
        },
    )
    record_gate(
        f"{plan.name}-workload-qualified",
        workload_passed,
        {
            "return_code": "not_applicable_reused_receipt",
            "expected": expected_cells,
            "complete": len(complete_cells),
            "protocol_failures": protocol_failures,
            "prefill_progress_failures": progress_failures,
            "live_update_failures": update_failures,
            "measurement_provenance": plan_provenance,
        },
    )
    return record



def run_workload_plan(
    state: PhaseState,
    group: BootGroup,
    plan: WorkloadPlan,
    triage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime.note(
        f"SCHEDULER RECHECK CELL START {plan.name} candidate={group.candidate.label} "
        f"cache={plan.cache_mode} policy={plan.policy.key}"
    )
    resuming = triage is not None and triage.get("state") == "partial"
    configuration = configure_policy(group.candidate, plan)
    record: dict[str, Any] = {
        "candidate": asdict(group.candidate),
        "boot_group": group.name,
        "plan": plan_to_json(plan),
        "cache_mode": plan.cache_mode,
        "configuration": configuration,
        "scenario_attempt_classification": "not_started",
    }
    if state.resume_enabled:
        record["measurement_provenance"] = (
            "mixed_resume" if resuming else RESUME_PROVENANCE_NEW
        )
    state.invocations[plan.name] = record
    if not configuration.get("passed"):
        record.update(
            {
                "status": "harness_configuration_failed",
                "scenario_attempt_classification": "not_attempted_configuration_failed",
                "failure_class": configuration.get("failure_class"),
            }
        )
        return record
    output = runtime.ROOT / OUTPUT_DIR / f"{plan.name}.json"
    command = workload_command(group.candidate, plan, output)
    prior_cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    if resuming:
        command.append("--resume")
        prior_cells = {
            cell_identity(cell): cell
            for cell in triage["parsed"].get("cells", [])
            if isinstance(cell, dict)
            and cell_identity(cell) is not None
            and cell_terminal(cell)
        }
    timeout = int(
        plan_measurement_seconds(plan)
        + plan_scenario_count(plan) * (240.0 + 300.0)
        + 900.0
    )
    return_code = runtime.run(
        command,
        label=out_path(plan.name),
        timeout=timeout,
        env=runtime.PROXY_ENV,
    )
    parsed, parse_error = read_json_file(output)
    cells = parsed.get("cells", []) if parsed else []
    expected_cells = plan_scenario_count(plan)
    complete_cells = [
        cell for cell in cells if isinstance(cell, dict) and cell.get("status") == "complete"
    ]
    protocol_failures, progress_failures, update_failures, transfer_cells = (
        classify_plan_cells(plan, cells)
    )
    all_attempted = parse_error is None and len(cells) == expected_cells
    reused_preserved = True
    preservation_failures: list[dict[str, Any]] = []
    if resuming:
        merged: dict[tuple[str, int, int], dict[str, Any]] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            key = cell_identity(cell)
            if key is not None:
                merged[key] = cell
        for key, prior_cell in prior_cells.items():
            kept = merged.get(key)
            if kept is None:
                reused_preserved = False
                preservation_failures.append(
                    {"cell": identity_label(key), "problem": "missing from merged receipt"}
                )
            elif kept != prior_cell:
                reused_preserved = False
                preservation_failures.append(
                    {"cell": identity_label(key), "problem": "reused cell was modified"}
                )
    workload_passed = (
        return_code == 0
        and all_attempted
        and len(complete_cells) == expected_cells
        and reused_preserved
    )
    record.update(
        {
            "status": "complete" if workload_passed else "qualification_failed",
            "scenario_attempt_classification": (
                "all_applicable_scenarios_attempted"
                if all_attempted
                else "scenario_set_incomplete_harness_failure"
            ),
            "failure_class": None if workload_passed else "workload_or_qualification",
            "output": str(output),
            "log": str(runtime.ROOT / OUTPUT_DIR / f"{plan.name}.log"),
            "command": command,
            "return_code": return_code,
            "parse_error": parse_error,
            "expected_cells": expected_cells,
            "observed_cells": len(cells),
            "complete_cells": len(complete_cells),
            "protocol_failures": protocol_failures,
            "prefill_progress_failures": progress_failures,
            "live_update_failures": update_failures,
            "memory_transfer_evidence": transfer_cells,
            "result": parsed,
        }
    )
    if resuming:
        eras: dict[str, str] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            key = cell_identity(cell)
            if key is None:
                continue
            eras[identity_label(key)] = (
                cell_measurement_era(triage["metadata"], key)
                if key in prior_cells
                else RESUME_PROVENANCE_NEW
            )
        record["cell_measurement_provenance"] = eras
        record["resume"] = {
            "receipt": str(output),
            "receipt_sha256_before_resume": triage["receipt_sha256"],
            "prior_receipt": triage["receipt"],
            "prior_configuration": reused_configuration_reference(plan, triage),
            "reused_cells": triage["reused_cells"],
            "cells_pending_at_resume": triage["missing_cells"],
            "reused_cells_preserved": reused_preserved,
            "preservation_failures": preservation_failures,
            "boot_identity": triage.get("boot_identity"),
        }
    record_gate(
        f"{plan.name}-all-scenarios-attempted",
        all_attempted,
        {
            "expected": expected_cells,
            "observed": len(cells),
            "parse_error": parse_error,
            "output": str(output),
        },
    )
    if resuming:
        record_gate(
            f"{plan.name}-reused-cells-preserved",
            reused_preserved,
            {
                "reused": len(prior_cells),
                "failures": preservation_failures,
                "receipt_sha256_before_resume": triage["receipt_sha256"],
            },
        )
    record_gate(
        f"{plan.name}-workload-qualified",
        workload_passed,
        {
            "return_code": return_code,
            "expected": expected_cells,
            "complete": len(complete_cells),
            "protocol_failures": protocol_failures,
            "prefill_progress_failures": progress_failures,
            "live_update_failures": update_failures,
            "reused_cells_preserved": reused_preserved if resuming else None,
        },
    )
    try:
        runtime.capture(out_path(f"{plan.name}-capture"))
    except Exception as error:
        capture_error = f"{type(error).__name__}: {error}"
        record["capture_error"] = capture_error
        record["status"] = "qualification_failed"
        record["failure_class"] = "harness_capture"
        state.errors.append(
            {
                "candidate": group.candidate.key,
                "boot_group": group.name,
                "plan": plan.name,
                "error": f"capture failed: {capture_error}",
            }
        )
        record_gate(f"{plan.name}-capture", False, capture_error)
    runtime.note(f"SCHEDULER RECHECK CELL DONE {plan.name} status={record['status']}")
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
    group: BootGroup,
    label: str,
    requested_env: dict[str, str],
    used_dma_representation: str | None,
) -> dict[str, Any]:
    inspect_label = out_path(f"{group.name}-env-inspect")
    mounts_label = out_path(f"{group.name}-mount-inspect")
    logs_label = out_path(f"{group.name}-server-log")
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
        ["docker", "logs", runtime.NAME], label=logs_label, timeout=120
    )
    env_rows = parse_json_line(read_receipt_text(inspect_label), list)
    mounts = parse_json_line(read_receipt_text(mounts_label), list)
    server_text = read_receipt_text(logs_label)
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
                r"max_num_batched_tokens['\"]?\s*[:=]\s*(\d+)", server_text
            )
        }
    )
    dma_values = sorted(
        set(re.findall(r"DMA min=([^,)\s]+)", server_text, flags=re.IGNORECASE))
    )
    metric_text = runtime.metric_snapshot()
    kv_values = {
        int(value)
        for value in re.findall(r'kv_cache_size_tokens="(\d+)"', metric_text)
    }
    kv_values.update(
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", server_text)
    )
    expected_batch = group.batch_tokens
    batch_parser_passed = bool(batch_values) and set(batch_values) == {expected_batch}
    dma_512k_active = any(
        value.lower() in {"524288", "512kb", "512kib"} for value in dma_values
    )
    dma_off_active = any(
        value.lower() in {"off", "disabled", "none"} for value in dma_values
    )
    normalized_mounts = mounts if isinstance(mounts, list) else []
    l2_sources = [
        str(mount.get("Source"))
        for mount in normalized_mounts
        if isinstance(mount, dict)
        and str(mount.get("Destination", "")).rstrip("/") == "/lmcache-l2"
    ]
    if group.cache_mode == "lmcache":
        state.lmcache_roots.append(
            {"group": group.name, "sources": l2_sources, "plans": [p.name for p in group.plans]}
        )
    facts = {
        "boot_group": group_to_json(group),
        "label": label,
        "requested_env": requested_env,
        "boot_extra_args": boot_args(group),
        "inspect_return_code": inspect_rc,
        "mounts_return_code": mounts_rc,
        "logs_return_code": logs_rc,
        "actual_selected_env": selected_env,
        "container_mounts": normalized_mounts,
        "batch_parser": {
            "values": batch_values,
            "requested": expected_batch,
            "matched": batch_parser_passed,
        },
        "kv_capacity": {
            "gpu_kv_cache_size_tokens": sorted(kv_values),
            "source": "server GPU KV cache log and cache_config_info metric",
            "naive_block_multiplication_used": False,
        },
        "dma": {
            "requested_literal": group.dma_value,
            "used_representation": used_dma_representation,
            "initialized_values": dma_values,
            "threshold_512k_active": dma_512k_active,
            "default_dcp4_dma_off_active": dma_off_active,
        },
        "lmcache_l2_sources": l2_sources,
        "raw_receipts": {
            "environment": str(runtime.ROOT / f"{inspect_label}.log"),
            "mounts": str(runtime.ROOT / f"{mounts_label}.log"),
            "server": str(runtime.ROOT / f"{logs_label}.log"),
            "launch": str(runtime.ROOT / f"{label}.launch.json"),
        },
    }
    if group.cache_mode == "lmcache":
        isolated = bool(l2_sources) and all(
            Path(source).resolve().is_relative_to(runtime.L2_HOST_ROOT.resolve())
            for source in l2_sources
        )
        capacity_raw = selected_env.get("LMCACHE_L2_MAX_CAPACITY_GB")
        try:
            capacity = float(capacity_raw) if capacity_raw is not None else None
        except ValueError:
            capacity = None
        storage_detail = {
            "sources": l2_sources,
            "under_runtime_root": isolated,
            "capacity_gb": capacity,
            "deletion_performed": False,
        }
        storage_ok = isolated and capacity is not None and 0.0 < capacity <= 160.0
        facts["lmcache_storage"] = {
            "applicable": True,
            "passed": storage_ok,
            "detail": storage_detail,
        }
    else:
        facts["lmcache_storage"] = {
            "applicable": False,
            "passed": True,
            "detail": None,
        }
    runtime.save_json(out_path(f"{group.name}-runtime-facts.json"), facts)
    state.runtime_facts[group.name] = facts
    record_gate(f"{group.name}-batch-parser", batch_parser_passed, facts["batch_parser"])
    record_gate(f"{group.name}-kv-capacity", bool(kv_values), facts["kv_capacity"])
    if group.dma_value is not None:
        record_gate(f"{group.name}-dma512k-active", dma_512k_active, facts["dma"])
    elif group.factor_role is not None:
        record_gate(f"{group.name}-default-dma-off", dma_off_active, facts["dma"])
    if group.cache_mode == "lmcache":
        record_gate(
            f"{group.name}-isolated-capped-lmcache",
            bool(facts["lmcache_storage"]["passed"]),
            facts["lmcache_storage"]["detail"],
        )
    return facts


def dma_literal_rejection(group: BootGroup) -> dict[str, Any]:
    label = out_path(f"{group.name}-dma-literal-failure-log")
    runtime.run(["docker", "logs", runtime.NAME], label=label, timeout=60)
    text = read_receipt_text(label)
    matching = [
        line
        for line in text.splitlines()
        if "512kb" in line.lower() or "vllm_pcie_dma_min_bytes" in line.lower()
    ]
    rejected = any(
        any(word in line.lower() for word in ("invalid", "parse", "integer", "literal", "byte"))
        for line in matching
    )
    receipt = {
        "parser_rejection": rejected,
        "matching_lines": matching,
        "receipt": str(runtime.ROOT / f"{label}.log"),
    }
    runtime.save_json(out_path(f"{group.name}-dma-literal-behavior.json"), receipt)
    return receipt


def mark_group_unattempted(
    state: PhaseState,
    group: BootGroup,
    detail: object,
    *,
    status: str,
    attempt_classification: str,
    failure_class: str,
    plans: list[WorkloadPlan] | None = None,
) -> None:
    # On resume only plans with remaining work are marked; reused plans keep
    # their pre-reboot receipt records.
    for plan in (plans if plans is not None else group.plans):
        state.invocations[plan.name] = {
            "candidate": asdict(group.candidate),
            "boot_group": group.name,
            "plan": plan_to_json(plan),
            "status": status,
            "scenario_attempt_classification": attempt_classification,
            "failure_class": failure_class,
            "detail": detail,
        }
        record_gate(f"{plan.name}-all-scenarios-attempted", False, detail)


def boot_group(state: PhaseState, group: BootGroup) -> None:
    triages: dict[str, dict[str, Any]] | None = None
    pending: list[WorkloadPlan] = list(group.plans)
    if state.resume_enabled:
        triages = {
            plan.name: state.resume_plans.get(
                plan.name, {"state": "absent", "reused_cells": [], "missing_cells": []}
            )
            for plan in group.plans
        }
        reusable = [
            plan for plan in group.plans if triages[plan.name]["state"] == "reusable_complete"
        ]
        pending = [
            plan for plan in group.plans if triages[plan.name]["state"] != "reusable_complete"
        ]
        for plan in reusable:
            state.invocations[plan.name] = reused_invocation_record(
                state, group, plan, triages[plan.name]
            )
        if not pending:
            state.reused_groups.add(group.name)
            runtime.note(
                f"SCHEDULER RECHECK RESUME REUSE {group.name} "
                f"plans={[plan.name for plan in reusable]}"
            )
            return
        if reusable:
            runtime.note(
                f"SCHEDULER RECHECK RESUME PARTIAL GROUP {group.name} "
                f"reused={[plan.name for plan in reusable]} "
                f"pending={[plan.name for plan in pending]}"
            )
    label = out_path(group.name)
    state.boots_attempted += 1
    used_dma = group.dma_value
    requested_env = boot_environment(group)
    booted = False
    literal_receipt: dict[str, Any] | None = None
    runtime.note(
        f"SCHEDULER RECHECK BOOT START {group.name} candidate={group.candidate.label} "
        f"cache={group.cache_mode} structure={structural_key(group.policy_structure)}"
    )
    try:
        booted = runtime.boot(
            label,
            image=group.candidate.image,
            tp=4,
            dcp=4,
            spec="mtp0",
            cache=group.cache_mode,
            kv="fp8_ds_mla",
            extra_env=requested_env,
            extra_args=boot_args(group),
            model=runtime.MODEL,
        )
        if not booted and group.dma_value == "512KB":
            literal_receipt = dma_literal_rejection(group)
            if literal_receipt["parser_rejection"]:
                runtime.stop()
                used_dma = "524288"
                requested_env = boot_environment(group, dma_value=used_dma)
                label = out_path(f"{group.name}-numeric-bytes")
                state.boots_attempted += 1
                booted = runtime.boot(
                    label,
                    image=group.candidate.image,
                    tp=4,
                    dcp=4,
                    spec="mtp0",
                    cache=group.cache_mode,
                    kv="fp8_ds_mla",
                    extra_env=requested_env,
                    extra_args=boot_args(group),
                    model=runtime.MODEL,
                )
        record_gate(
            f"{group.name}-boot",
            booted,
            {
                "candidate": asdict(group.candidate),
                "cache_mode": group.cache_mode,
                "requested_env": requested_env,
                "extra_args": boot_args(group),
                "used_dma_representation": used_dma,
                "literal_receipt": literal_receipt,
            },
        )
        if not booted:
            mark_group_unattempted(
                state,
                group,
                "service failed to become healthy",
                status="boot_failed",
                attempt_classification="not_attempted_boot_failed",
                failure_class="harness_boot",
                plans=pending,
            )
            return
        state.boots_ready += 1
        state.booted_groups.add(group.name)
        discovery = discover_api(state, group, label)
        facts = capture_runtime_facts(state, group, label, requested_env, used_dma)
        if group.candidate.api_schema == "overlay-r26":
            check_structural_post_rejections(state, group)
        if not discovery.get("passed"):
            mark_group_unattempted(
                state,
                group,
                {
                    "classification": "api_or_boot_structure_mismatch",
                    "discovery": discovery,
                    "runtime_facts": facts,
                },
                status="api_or_boot_structure_failed",
                attempt_classification="not_attempted_api_contract_failed",
                failure_class="harness_or_scheduler_api_contract",
                plans=pending,
            )
            return
        for plan in pending:
            try:
                run_workload_plan(
                    state, group, plan, triages[plan.name] if triages else None
                )
            except Exception as error:
                detail = {
                    "candidate": group.candidate.key,
                    "boot_group": group.name,
                    "plan": plan.name,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                state.errors.append(detail)
                state.invocations[plan.name] = {
                    "candidate": asdict(group.candidate),
                    "boot_group": group.name,
                    "plan": plan_to_json(plan),
                    "status": "harness_exception",
                    "scenario_attempt_classification": "incomplete_harness_exception",
                    "failure_class": "harness_exception",
                    "detail": detail,
                }
                runtime.save_json(out_path(f"{plan.name}-phase-error.json"), detail)
                record_gate(f"{plan.name}-all-scenarios-attempted", False, detail)
    except Exception as error:
        detail = {
            "candidate": group.candidate.key,
            "boot_group": group.name,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        state.errors.append(detail)
        mark_group_unattempted(
            state,
            group,
            detail,
            status="harness_exception",
            attempt_classification="not_attempted_harness_exception",
            failure_class="harness_exception",
            plans=pending,
        )
    finally:
        if booted:
            try:
                runtime.capture(out_path(f"{group.name}-final"))
            except Exception as error:
                state.errors.append(
                    {
                        "candidate": group.candidate.key,
                        "boot_group": group.name,
                        "error": f"final capture: {type(error).__name__}: {error}",
                    }
                )
        runtime.stop()
        runtime.note(f"SCHEDULER RECHECK BOOT DONE {group.name}")


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
        try:
            key = (
                str(cell.get("profile")),
                int(cell.get("concurrency")),
                int(cell.get("repeat")),
            )
        except (TypeError, ValueError):
            continue
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
        exact_sets = bool(key_sets) and all(values == key_sets[0] for values in key_sets)
    else:
        keys = set(expected_keys)
        exact_sets = all(keys <= set(values) for values in traces.values())
    mismatches: dict[str, dict[str, str | None]] = {}
    for key in sorted(keys):
        values = {member: traces[member].get(key) for member in members}
        if None in values.values() or len(set(values.values())) != 1:
            mismatches[str(key)] = values
    passed = exact_sets and bool(keys) and not mismatches
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
    record_gate(f"identical-trace-{name}", passed, receipt)
    return receipt


def automatic_primary_plan_names() -> tuple[str, ...]:
    return tuple(
        plan.name
        for plan in OVERLAY_AUTOMATIC_PLANS
        if plan.series in {"overlay-auto-headline", "overlay-auto-factorial"}
    )


def trace_checks(state: PhaseState) -> list[dict[str, Any]]:
    periodic_keys = (("periodic-128k", 8, 1), ("periodic-128k", 16, 1))
    factor_names = tuple(name for name, _, _, _ in FACTOR_SPECS)
    checks = [
        compare_trace_group(
            state,
            "headline-five-arm",
            (
                "official-off-headline",
                "official-static04-headline",
                "overlay-single-off-headline",
                "overlay-single-static04-headline",
                "overlay-auto-responsive-decode-aware-lanesauto-headline",
            ),
        ),
        compare_trace_group(
            state,
            "single-lane-off-image-parity",
            ("official-off-headline", "overlay-single-off-headline"),
        ),
        compare_trace_group(
            state,
            "single-lane-static04-image-parity",
            ("official-static04-headline", "overlay-single-static04-headline"),
        ),
        compare_trace_group(
            state,
            "single-lane-static06-image-parity",
            ("official-static06-coverage", "overlay-single-static06-coverage"),
        ),
        compare_trace_group(
            state,
            "single-lane-static07-image-parity",
            ("official-static07-coverage", "overlay-single-static07-coverage"),
        ),
        compare_trace_group(
            state,
            "profile-shape-three-arm",
            (
                "official-off-profile-coverage",
                "overlay-single-off-profile-coverage",
                "overlay-auto-responsive-decode-aware-lanesauto-profile-coverage",
            ),
        ),
        compare_trace_group(
            state,
            "overlay-auto-factorial",
            automatic_primary_plan_names(),
            periodic_keys,
        ),
        compare_trace_group(
            state,
            "overlay-live-update",
            (
                "overlay-auto-responsive-decode-aware-lanesauto-headline",
                "overlay-auto-responsive-decode-aware-lanesauto-live-update",
            ),
            (("periodic-128k", 16, 1),),
        ),
        compare_trace_group(
            state,
            "batch-dma-factors",
            factor_names,
        ),
        compare_trace_group(
            state,
            "lmcache-matched-confirmation",
            (
                "official-static04-lmcache-matched-writer",
                "overlay-static04-lmcache-matched-reader",
            ),
        ),
    ]
    runtime.save_json(out_path("trace-equality-checks.json"), checks)
    return checks


def summarize_phase(
    state: PhaseState,
    plan: dict[str, Any],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = len(ALL_PLANS)
    attempted = sum(
        record.get("scenario_attempt_classification")
        == "all_applicable_scenarios_attempted"
        for record in state.invocations.values()
    )
    complete = sum(
        record.get("status") == "complete" for record in state.invocations.values()
    )
    all_recorded = len(state.invocations) == expected
    all_attempted = all_recorded and attempted == expected
    traces_equal = bool(traces) and all(receipt.get("passed") is True for receipt in traces)
    overlay_groups = [
        group for group in BOOT_GROUPS if group.candidate.api_schema == "overlay-r26"
    ]

    def boot_required(group: BootGroup) -> bool:
        return not (state.resume_enabled and group.name in state.reused_groups)

    overlay_api_complete = all(
        state.api_discoveries.get(group.name, {}).get("passed") is True
        for group in overlay_groups
        if boot_required(group)
    )
    structural_rejections_complete = all(
        state.structural_rejections.get(group.name, {}).get("passed") is True
        for group in overlay_groups
        if boot_required(group)
    )
    lmcache_booted_groups = [
        group
        for group in BOOT_GROUPS
        if group.cache_mode == "lmcache" and boot_required(group)
    ]
    lmcache_same_root = (
        len(lmcache_booted_groups) == 2
        and len(state.lmcache_roots) == 2
        and bool(state.lmcache_roots[0]["sources"])
        and state.lmcache_roots[0]["sources"] == state.lmcache_roots[1]["sources"]
    )
    reader_record = state.invocations.get(
        "overlay-static04-lmcache-matched-reader", {}
    )
    reader_transfer = reader_record.get("memory_transfer_evidence", [])
    expected_reader_cells = plan_scenario_count(
        next(
            candidate_plan
            for candidate_plan in ALL_PLANS
            if candidate_plan.name == "overlay-static04-lmcache-matched-reader"
        )
    )
    external_transfer_observed = (
        len(reader_transfer) == expected_reader_cells
        and all(
            cell.get("evidence", {}).get("external_memory_transfer_observed") is True
            for cell in reader_transfer
        )
    )
    all_api_complete = all(
        state.api_discoveries.get(group.name, {}).get("passed") is True
        for group in BOOT_GROUPS
        if boot_required(group)
    )
    runtime_facts_complete = all(
        state.runtime_facts.get(group.name, {})
        .get("batch_parser", {})
        .get("matched")
        is True
        and bool(
            state.runtime_facts.get(group.name, {})
            .get("kv_capacity", {})
            .get("gpu_kv_cache_size_tokens")
        )
        for group in BOOT_GROUPS
        if boot_required(group)
    )
    factor_transport_complete = all(
        (
            state.runtime_facts.get(group.name, {})
            .get("dma", {})
            .get(
                "threshold_512k_active"
                if group.dma_value is not None
                else "default_dcp4_dma_off_active"
            )
            is True
        )
        for group in BOOT_GROUPS
        if group.factor_role is not None and boot_required(group)
    )
    lmcache_storage_complete = all(
        state.runtime_facts.get(group.name, {})
        .get("lmcache_storage", {})
        .get("passed")
        is True
        for group in BOOT_GROUPS
        if group.cache_mode == "lmcache" and boot_required(group)
    )
    if state.resume_enabled:
        services_booted = (
            (state.booted_groups | state.reused_groups)
            == {group.name for group in BOOT_GROUPS}
            and state.boots_ready == len(state.booted_groups)
        )
    else:
        services_booted = state.boots_ready == len(BOOT_GROUPS)
    checks = {
        "all_services_booted": services_booted,
        "source_and_plan_contract": plan.get("validation", {}).get("passed") is True,
        "all_plans_recorded": all_recorded,
        "all_applicable_scenarios_attempted": all_attempted,
        "all_scenarios_qualified": complete == expected,
        "overlay_api_and_boot_structure": overlay_api_complete,
        "all_api_schemas_and_effective_boot_state": all_api_complete,
        "runtime_batch_parser_and_kv_capacity": runtime_facts_complete,
        "factor_dma_transport": factor_transport_complete,
        "lmcache_storage_isolated_and_capped": lmcache_storage_complete,
        "structural_post_rejections_leave_state_unchanged": structural_rejections_complete,
        "deterministic_trace_equality": traces_equal,
        "lmcache_same_isolated_root": lmcache_same_root,
        "lmcache_reader_external_transfer_observed": external_transfer_observed,
        "no_harness_exceptions": not state.errors,
    }
    for name, passed in checks.items():
        record_gate(
            f"phase-{name}",
            passed,
            {
                "expected_plans": expected,
                "recorded": len(state.invocations),
                "attempted": attempted,
                "complete": complete,
                "boot_groups": len(BOOT_GROUPS),
                "boots_attempted_including_dma_literal_retries": state.boots_attempted,
                "boots_ready": state.boots_ready,
                "overlay_api_complete": overlay_api_complete,
                "all_api_complete": all_api_complete,
                "runtime_facts_complete": runtime_facts_complete,
                "factor_transport_complete": factor_transport_complete,
                "lmcache_storage_complete": lmcache_storage_complete,
                "structural_rejections_complete": structural_rejections_complete,
                "lmcache_roots": state.lmcache_roots,
                "reader_transfer": reader_transfer,
                "errors": state.errors,
            },
        )
    passed = all(checks.values())
    interpretation_contract = {
        "harness_failures_separate": True,
        "model_wrong_answers_assessed": False,
        "budget_limited_answers_separate_from_protocol_failure": True,
        "text_equality_is_not_kv_byte_equality": True,
        "failed_retrieval_is_not_corrupted_memory": True,
    }
    recovery_provenance: dict[str, Any] | None = None
    if state.resume_enabled:
        interpretation_contract["matched_speed_claims_across_clock_boundary"] = False
        triage_by_state = {
            triage_state: [
                triage
                for triage in state.resume_plans.values()
                if triage["state"] == triage_state
            ]
            for triage_state in ("reusable_complete", "partial", "absent")
        }
        recovery_provenance = {
            "resumed": True,
            "resume_started_at": state.resume_started_at,
            "reused_plans": [
                {
                    "plan": triage["plan"],
                    "boot_group": triage["boot_group"],
                    "reused_cells": len(triage["reused_cells"]),
                    "measurement_provenance": state.invocations.get(
                        triage["plan"], {}
                    ).get("measurement_provenance"),
                }
                for triage in triage_by_state["reusable_complete"]
            ],
            "resumed_plans": [
                {
                    "plan": triage["plan"],
                    "boot_group": triage["boot_group"],
                    "reused_cells": len(triage["reused_cells"]),
                    "cells_pending_at_resume": len(triage["missing_cells"]),
                    "measurement_provenance": state.invocations.get(
                        triage["plan"], {}
                    ).get("measurement_provenance"),
                }
                for triage in triage_by_state["partial"]
            ],
            "new_plans": [
                triage["plan"] for triage in triage_by_state["absent"]
            ],
            "reused_boot_groups": sorted(state.reused_groups),
            "booted_groups": sorted(state.booted_groups),
            "clock_profile_discontinuity": {
                "explanation": (
                    "Cells reused from the interrupted pre-reboot run were measured "
                    "while the GPUs ran with core VF offset +250 and memory VF offset "
                    "+6000. This resumed run executed after the reboot with core VF "
                    "offset 0, unchanged memory VF offset +6000, and a 600 W power "
                    "limit. Per-cell measurement_provenance records the era of every "
                    "cell; latency and throughput must not be compared across the "
                    "boundary and no matched-speed claim is valid across it."
                ),
                "pre_reboot_provenance_label": RESUME_PROVENANCE_REUSED,
                "post_reboot_provenance_label": RESUME_PROVENANCE_NEW,
                "cross_boundary_speed_comparison_valid": False,
                "cross_boundary_trace_equality_valid": True,
            },
        }
    summary = {
        "phase": "scheduler-recheck",
        "status": "complete" if passed else "failed",
        "plan_size": plan["plan_size"],
        "checks": checks,
        "expected_invocations": expected,
        "recorded_invocations": len(state.invocations),
        "all_scenarios_attempted_invocations": attempted,
        "qualified_invocations": complete,
        "invocations": {
            name: {key: value for key, value in record.items() if key != "result"}
            for name, record in state.invocations.items()
        },
        "api_discoveries": state.api_discoveries,
        "runtime_facts": state.runtime_facts,
        "structural_rejections": state.structural_rejections,
        "trace_checks": traces,
        "lmcache_roots": state.lmcache_roots,
        "errors": state.errors,
        "interpretation_contract": interpretation_contract,
    }
    if recovery_provenance is not None:
        summary["recovery_provenance"] = recovery_provenance
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write and validate the CPU-only plan; with --resume, inspect without modifying files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse validated prior receipts under ROOT/scheduler-recheck: "
            "complete plans are reused verbatim, partial plans run only their "
            "missing scenarios, and corrupt or configuration-mismatched "
            "receipts fail closed before any boot"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = phase_plan()
    resume_state: dict[str, Any] | None = (
        load_resume_receipts() if args.resume else None
    )
    if (
        resume_state is not None
        and resume_state["passed"]
        and plan["validation"]["passed"]
        and not args.plan_only
    ):
        archive_resume_evidence(resume_state)
    if not (args.plan_only and args.resume):
        runtime.save_json(out_path("phase-plan.json"), plan)
        runtime.save_json(
            out_path("unsupported-requested-modes.json"),
            {
                "features": list(UNSUPPORTED_REQUESTED_MODES),
                "mapped_to_other_modes": False,
            },
        )
        runtime.save_json(out_path("plan-validation.json"), plan["validation"])
        record_gate(
            "plan-source-and-coverage-contract",
            bool(plan["validation"]["passed"]),
            plan["validation"],
        )
        if resume_state is not None:
            runtime.save_json(
                out_path("resume-receipts.json"),
                {
                    **resume_state,
                    "plans": {
                        name: {key: value for key, value in triage.items() if key != "parsed"}
                        for name, triage in resume_state["plans"].items()
                    },
                },
            )
    if args.plan_only:
        result = {
            "status": "plan-valid" if plan["validation"]["passed"] else "plan-invalid",
            "output": str(runtime.ROOT / OUTPUT_DIR / "phase-plan.json"),
            "plan_size": plan["plan_size"],
            "services_touched": False,
            "gpu_jobs_started": False,
        }
        if resume_state is not None:
            result["resume"] = {
                "passed": resume_state.get("passed") is True,
                "counts": resume_state["counts"],
                "invalid_plans": resume_state["invalid_plans"],
            }
            if resume_state.get("passed") is not True:
                result["status"] = "resume-receipts-invalid"
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if plan["validation"]["passed"] else 1
    if not plan["validation"]["passed"]:
        preflight_failure = {
            "phase": "scheduler-recheck",
            "status": "failed",
            "failure_class": "harness_preflight",
            "plan_size": plan["plan_size"],
            "validation": plan["validation"],
            "services_touched": False,
            "gpu_jobs_started": False,
        }
        runtime.save_json(out_path("phase-summary.json"), preflight_failure)
        runtime.save_json(out_path("phase-results.json"), preflight_failure)
        print(json.dumps(preflight_failure, sort_keys=True), flush=True)
        return 1

    if resume_state is not None and resume_state.get("passed") is not True:
        resume_failure = {
            "phase": "scheduler-recheck",
            "status": "failed",
            "failure_class": "resume_receipt_validation_failed",
            "plan_size": plan["plan_size"],
            "resume": resume_state,
            "services_touched": False,
            "gpu_jobs_started": False,
        }
        runtime.save_json(out_path("phase-summary.json"), resume_failure)
        runtime.save_json(out_path("phase-results.json"), resume_failure)
        print(json.dumps(resume_failure, sort_keys=True), flush=True)
        return 1

    state = PhaseState()
    state.resume_enabled = bool(args.resume)
    if state.resume_enabled:
        state.resume_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state.resume_plans = resume_state["plans"]
    traces: list[dict[str, Any]] = []
    runtime.note("R26 SCHEDULER RECHECK START")
    try:
        for group in BOOT_GROUPS:
            boot_group(state, group)
        traces = trace_checks(state)
    except Exception as error:
        state.errors.append(
            {
                "scope": "phase",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        runtime.stop()
    summary = summarize_phase(state, plan, traces)
    runtime.save_json(out_path("phase-summary.json"), summary)
    runtime.save_json(
        out_path("phase-results.json"),
        {
            "status": summary["status"],
            "plan_size": summary["plan_size"],
            "checks": summary["checks"],
            "result_files": {
                plan.name: str(runtime.ROOT / OUTPUT_DIR / f"{plan.name}.json")
                for plan in ALL_PLANS
            },
            "raw_request_events": "each result file: cells[].requests",
            "summary": str(runtime.ROOT / OUTPUT_DIR / "phase-summary.json"),
            "gates": str(runtime.ROOT / OUTPUT_DIR / "gates.jsonl"),
        },
    )
    runtime.note(
        f"R26 SCHEDULER RECHECK DONE status={summary['status']} "
        f"qualified={summary['qualified_invocations']}/{summary['expected_invocations']}"
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
