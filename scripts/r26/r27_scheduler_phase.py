#!/usr/bin/env python3
"""Run the bounded, source-pinned R27 scheduler attribution phase."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import agent_workload_recheck as workload_consumer
import r27_config as r27
import runtime as rt
import steady_metrics

PHASE_SCHEMA = "glm-r27-scheduler-phase/v1"
PLAN_SCHEMA = "glm-r27-scheduler-plan/v1"
REANALYSIS_SCHEMA = "glm-r27-scheduler-reanalysis/v1"
OUTPUT_DIR = "r27-scheduler"
POLICY_PATH = "/prefill_fairness"
EXISTING_CONSUMER_SCHEMA = "overlay-r26"
BOUNDARY_CHECKPOINT_PATH = (
    "/opt/glm53-flash/vllm/vllm/v1/worker/gpu/boundary_checkpoint.py"
)
TRACE_SEED = "r27-scheduler-qualified-v1"
BUDGETS = (4096, 8192, 12288, 16384)
DECODE_CONCURRENCIES = (1, 8)
MIXED_CONCURRENCIES = (8, 16)
MIXED_PROFILES = ("short-prefill-heavy", "periodic-128k")
PROFILE_SECONDS = {"short-prefill-heavy": 60.0, "periodic-128k": 60.0}
MIXED_IDENTITY_FIELDS = ("profile", "concurrency", "repeat")
CANDIDATE_CRASH_MARKERS = {
    "engine_fatal": "EngineCore encountered a fatal error.",
    "worker_exception": "WorkerProc hit an exception.",
    "boundary_checkpoint_restore": "boundary_checkpoint.py",
    "restore_kernel": "_restore_auxiliary_state_kernel",
    "triton_compile": "triton.compiler.errors.CompilationError",
    "root_cause": "AttributeError(\"'int' object has no attribute 'to'\")",
}
CONFIGURED_FIELDS = (
    "prefill_compute_share",
    "prefill_compute_half_life",
    "max_parallel_prefills",
    "prefill_policy",
    "decode_refill_target",
)
MUTABLE_FIELDS = ("prefill_compute_share", "prefill_compute_half_life")
STRUCTURAL_FIELDS = (
    "max_parallel_prefills",
    "prefill_policy",
    "decode_refill_target",
)
LIVE_UPDATES: tuple[dict[str, Any], ...] = (
    {
        "offset_seconds": 5.0,
        "config": {
            "prefill_compute_share": 0.2,
            "prefill_compute_half_life": None,
        },
    },
    {
        "offset_seconds": 20.0,
        "config": {
            "prefill_compute_share": 0.8,
            "prefill_compute_half_life": None,
        },
    },
    {
        "offset_seconds": 35.0,
        "config": {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "smooth",
        },
    },
    {
        "offset_seconds": 50.0,
        "config": {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "responsive",
        },
    },
)
HTTP = requests.Session()
HTTP.trust_env = False


@dataclass(frozen=True)
class Policy:
    key: str
    prefill_compute_share: float | str | None
    prefill_compute_half_life: float | str | None
    max_parallel_prefills: int | str
    prefill_policy: str
    decode_refill_target: int | str

    def configured(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in CONFIGURED_FIELDS
        }


@dataclass(frozen=True)
class BootCase:
    key: str
    arm: str
    candidate_label: str
    batch_tokens: int
    policy: Policy
    role: str
    run_behavior: bool = False
    run_live_tuning: bool = False
    as_shipped_auto: bool = False


FIXED = Policy(
    key="fixed0.4-single-lane",
    prefill_compute_share=0.4,
    prefill_compute_half_life=None,
    max_parallel_prefills=1,
    prefill_policy="round-robin",
    decode_refill_target="auto",
)
AUTO = Policy(
    key="auto-responsive-decode-aware-auto-lanes-refill",
    prefill_compute_share="auto",
    prefill_compute_half_life="responsive",
    max_parallel_prefills="auto",
    prefill_policy="decode-aware",
    decode_refill_target="auto",
)


def build_cases() -> tuple[BootCase, ...]:
    cases: list[BootCase] = [
        BootCase(
            key="stock-fixed-control",
            arm="stock",
            candidate_label="Stock R27",
            batch_tokens=4096,
            policy=FIXED,
            role="fixed-policy code control",
            run_behavior=True,
        ),
        BootCase(
            key="patched-fixed-control",
            arm="patched",
            candidate_label="Patched R27",
            batch_tokens=4096,
            policy=FIXED,
            role="fixed-policy source-effect candidate",
            run_behavior=True,
        ),
    ]
    for budget in BUDGETS:
        cases.extend(
            (
                BootCase(
                    key=f"stock-auto-lanes-bt{budget}",
                    arm="stock",
                    candidate_label="Stock R27",
                    batch_tokens=budget,
                    policy=AUTO,
                    role="stock effective-lane observational control",
                ),
                BootCase(
                    key=f"patched-auto-lanes-bt{budget}",
                    arm="patched",
                    candidate_label="Patched R27",
                    batch_tokens=budget,
                    policy=AUTO,
                    role=(
                        "configured-auto policy-effect and lane candidate"
                        if budget == 4096
                        else "patched geometry-independent lane candidate"
                    ),
                    run_behavior=budget == 4096,
                    run_live_tuning=budget == 4096,
                ),
            )
        )
    cases.append(
        BootCase(
            key="auto-image-as-shipped",
            arm="auto",
            candidate_label="Patched-auto R27 image (as shipped)",
            batch_tokens=4096,
            policy=AUTO,
            role="default-delivery attribution",
            run_behavior=True,
            as_shipped_auto=True,
        )
    )
    return tuple(cases)


CASES = build_cases()
BEHAVIOR_KEYS = tuple(case.key for case in CASES if case.run_behavior)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, f"expected JSON object, got {type(value).__name__}"
    return value, None


def values_equal(left: object, right: object) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def mismatches(actual: dict[str, Any] | None, expected: dict[str, Any]) -> dict[str, Any]:
    if actual is None:
        return {"config": {"expected": expected, "actual": None}}
    return {
        field: {"expected": wanted, "actual": actual.get(field)}
        for field, wanted in expected.items()
        if not values_equal(actual.get(field), wanted)
    }


def projection(config: dict[str, Any] | None, fields: tuple[str, ...] | list[str]) -> object:
    if config is None:
        return None
    return {field: config.get(field) for field in fields}


def ast_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def named_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"class {name!r} not found")


def named_function(body: list[ast.stmt], name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"function {name!r} not found")


def annotated_fields(cls: ast.ClassDef) -> list[str]:
    return [
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


def config_extra_forbid(cls: ast.ClassDef) -> bool:
    for node in cls.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "model_config" for target in node.targets):
            continue
        if isinstance(node.value, ast.Call):
            for keyword in node.value.keywords:
                if keyword.arg == "extra" and isinstance(keyword.value, ast.Constant):
                    return keyword.value.value == "forbid"
    return False


def return_dict_fields(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    candidates: list[list[str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        fields = [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        candidates.append(fields)
    if not candidates:
        raise ValueError(f"no literal dict return found in {function.name}")
    return max(candidates, key=len)


def set_difference_fields(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "unknown_fields" for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub) and isinstance(value.right, ast.Set):
            return [
                item.value
                for item in value.right.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
    raise ValueError(f"allowed-field set not found in {function.name}")


def module_constant(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise ValueError(f"constant {name!r} not found")


def class_constants(cls: ast.ClassDef, names: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id in names and node.value is not None:
            result[node.target.id] = ast.literal_eval(node.value)
    return result


def function_signature(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[str]]:
    return {
        "positional": [
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
            )
        ],
        "keyword_only": [argument.arg for argument in function.args.kwonlyargs],
    }


def snapshot_path(arm: str, container_path: str) -> Path:
    return r27.SOURCE_ROOT / r27.SOURCE_DIRS[arm] / container_path.lstrip("/")


def image_env(inspect: dict[str, Any]) -> dict[str, str]:
    config = inspect.get("Config") if isinstance(inspect, dict) else None
    values = config.get("Env", []) if isinstance(config, dict) else []
    result: dict[str, str] = {}
    for value in values:
        if isinstance(value, str) and "=" in value:
            key, item = value.split("=", 1)
            result[key] = item
    return result


def source_contract() -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = r27.SOURCE_ROOT / "manifest.json"
    manifest, manifest_error = read_json_object(manifest_path)
    if manifest_error is not None:
        return {
            "status": "invalid",
            "manifest": str(manifest_path),
            "manifest_error": manifest_error,
            "errors": [manifest_error],
        }
    assert manifest is not None
    manifest_images = {
        row.get("arm"): row
        for row in manifest.get("images", [])
        if isinstance(row, dict) and isinstance(row.get("arm"), str)
    }
    verified: dict[str, Any] = {}
    file_hashes: dict[str, dict[str, str]] = {}
    inspect_records: dict[str, Any] = {}
    for arm in ("stock", "patched", "auto"):
        manifest_arm = r27.SOURCE_DIRS[arm]
        entry = manifest_images.get(manifest_arm)
        if not isinstance(entry, dict):
            errors.append(f"manifest arm missing: {manifest_arm}")
            continue
        if entry.get("image") != r27.IMAGES[arm]:
            errors.append(f"image mismatch for {arm}")
        files: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for row in entry.get("files", []):
            if not isinstance(row, dict) or not isinstance(row.get("container_path"), str):
                errors.append(f"malformed file row for {arm}")
                continue
            container_path = row["container_path"]
            actual_path = snapshot_path(arm, container_path)
            actual_hash = sha256_file(actual_path)
            expected_path = str(actual_path)
            row_path = str(row.get("path", ""))
            passed = (
                actual_hash is not None
                and actual_hash == row.get("sha256")
                and Path(row_path) == actual_path
            )
            if not passed:
                errors.append(f"snapshot verification failed: {arm}:{container_path}")
            files[container_path] = {
                "path": expected_path,
                "manifest_path": row_path,
                "manifest_sha256": row.get("sha256"),
                "actual_sha256": actual_hash,
                "verified": passed,
            }
            if actual_hash is not None:
                hashes[container_path] = actual_hash
        verified[arm] = {
            "manifest_arm": manifest_arm,
            "image": entry.get("image"),
            "expected_image": r27.IMAGES[arm],
            "files": files,
            "all_files_verified": bool(files) and all(row["verified"] for row in files.values()),
        }
        file_hashes[arm] = hashes
        inspect_path = r27.SOURCE_ROOT / r27.SOURCE_DIRS[arm] / "image-inspect.json"
        inspect, inspect_error = read_json_object(inspect_path)
        expected_id = "sha256:" + r27.IMAGES[arm].rsplit("@sha256:", 1)[-1]
        inspect_ok = inspect_error is None and inspect is not None and inspect.get("Id") == expected_id
        if not inspect_ok:
            errors.append(f"image inspect identity failed: {arm}: {inspect_error or 'id mismatch'}")
        inspect_records[arm] = {
            "path": str(inspect_path),
            "sha256": sha256_file(inspect_path),
            "expected_id": expected_id,
            "observed_id": inspect.get("Id") if inspect else None,
            "entrypoint": (
                inspect.get("Config", {}).get("Entrypoint")
                if inspect and isinstance(inspect.get("Config"), dict)
                else None
            ),
            "env": image_env(inspect or {}),
            "verified": inspect_ok,
        }

    api_path = "/opt/glm53-flash/vllm/vllm/entrypoints/serve/dev/fairness/api_router.py"
    config_path = "/opt/glm53-flash/vllm/vllm/config/scheduler.py"
    scheduler_path = "/opt/glm53-flash/vllm/vllm/v1/core/sched/scheduler.py"
    lane_path = "/opt/glm53-flash/vllm/vllm/v1/core/sched/prefill_interleave.py"
    fairness_path = "/opt/glm53-flash/vllm/vllm/v1/core/sched/compute_fairness.py"
    normal_launcher_path = "/usr/local/bin/serve-glm53-flash.sh"
    downstream_launcher_path = (
        "/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh"
    )
    auto_launcher_path = "/usr/local/bin/serve-glm53-flash-auto.sh"
    api_semantics: dict[str, Any] = {}
    lane_semantics: dict[str, Any] = {}
    compute_semantics: dict[str, Any] = {}
    launcher_semantics: dict[str, Any] = {}
    decode_contract: dict[str, Any] = {}
    try:
        api_tree = ast_tree(snapshot_path("patched", api_path))
        request_class = named_class(api_tree, "PrefillFairnessRequest")
        request_fields = annotated_fields(request_class)
        router_extra_forbid = config_extra_forbid(request_class)
        scheduler_tree = ast_tree(snapshot_path("patched", scheduler_path))
        scheduler_class = named_class(scheduler_tree, "Scheduler")
        get_method = named_function(scheduler_class.body, "get_prefill_fairness")
        set_method = named_function(scheduler_class.body, "set_prefill_fairness")
        get_fields = return_dict_fields(get_method)
        engine_allowed_fields = set_difference_fields(set_method)
        structural_fields = [field for field in CONFIGURED_FIELDS if field not in request_fields]
        api_hashes = {arm: file_hashes.get(arm, {}).get(api_path) for arm in file_hashes}
        api_semantics = {
            "request_model_fields": request_fields,
            "engine_live_mutable_fields": engine_allowed_fields,
            "structural_fields": structural_fields,
            "get_fields": get_fields,
            "router_extra_fields_forbidden": router_extra_forbid,
            "source_sha256_by_arm": api_hashes,
            "source_identical_all_arms": len(set(api_hashes.values())) == 1,
            "consumer_schema_selector": EXISTING_CONSUMER_SCHEMA,
            "consumer_selector_is_release_label": False,
            "consumer_required_get_fields": list(workload_consumer.OVERLAY_GET_FIELDS),
            "consumer_post_fields": list(workload_consumer.OVERLAY_MUTABLE_FIELDS),
            "consumer_compatible": (
                request_fields == list(MUTABLE_FIELDS)
                and set(engine_allowed_fields) == set(MUTABLE_FIELDS)
                and set(workload_consumer.OVERLAY_GET_FIELDS).issubset(get_fields)
                and tuple(workload_consumer.OVERLAY_MUTABLE_FIELDS) == MUTABLE_FIELDS
                and structural_fields == list(STRUCTURAL_FIELDS)
                and router_extra_forbid
            ),
        }
        if not api_semantics["consumer_compatible"]:
            errors.append("existing overlay-r26 consumer fields do not match shipped R27 API")
    except Exception as error:
        errors.append(f"API source semantics: {type(error).__name__}: {error}")

    try:
        stock_lane_tree = ast_tree(snapshot_path("stock", lane_path))
        patched_lane_tree = ast_tree(snapshot_path("patched", lane_path))
        stock_resolver = named_function(stock_lane_tree.body, "resolve_max_parallel_prefills")
        patched_resolver = named_function(patched_lane_tree.body, "resolve_max_parallel_prefills")
        auto_cap = int(module_constant(patched_lane_tree, "AUTO_MAX_PARALLEL_PREFILLS"))
        patched_returns = [
            ast.unparse(node.value)
            for node in ast.walk(patched_resolver)
            if isinstance(node, ast.Return) and node.value is not None
        ]
        patched_signature = function_signature(patched_resolver)
        stock_signature = function_signature(stock_resolver)
        patched_geometry_independent = (
            patched_signature["keyword_only"] == ["max_num_seqs"]
            and any(value == "min(requested, max_num_seqs)" for value in patched_returns)
            and auto_cap == 4
        )
        lane_semantics = {
            "auto_lane_cap": auto_cap,
            "stock_resolver_sha256": file_hashes.get("stock", {}).get(lane_path),
            "stock_resolver_inputs": stock_signature,
            "stock_effective_lane_oracle": "observe live readback only; no legacy geometry formula is asserted",
            "patched_resolver_sha256": file_hashes.get("patched", {}).get(lane_path),
            "patched_resolver_inputs": patched_signature,
            "patched_return_expressions": patched_returns,
            "patched_geometry_independent_of_token_and_cache_geometry": patched_geometry_independent,
            "patched_source_expected_auto_lanes_at_max_num_seqs_32": min(auto_cap, 32),
            "configured_and_effective_lanes_are_distinct_fields": True,
        }
        if not patched_geometry_independent:
            errors.append("patched lane resolver does not have pinned geometry-independent semantics")
    except Exception as error:
        errors.append(f"lane source semantics: {type(error).__name__}: {error}")

    try:
        fairness_tree = ast_tree(snapshot_path("patched", fairness_path))
        fairness_class = named_class(fairness_tree, "PrefillComputeShareController")
        constants = class_constants(
            fairness_class,
            (
                "AUTO_INITIAL_SHARE",
                "AUTO_MIN_SHARE",
                "AUTO_MAX_SHARE",
                "SMOOTH_HALF_LIFE_SECONDS",
                "RESPONSIVE_HALF_LIFE_SECONDS",
            ),
        )
        compute_semantics = {
            "source_sha256": file_hashes.get("patched", {}).get(fairness_path),
            "constants": constants,
            "live_boundary_plan": list(LIVE_UPDATES),
        }
        expected_constants = {
            "AUTO_INITIAL_SHARE": 0.4,
            "AUTO_MIN_SHARE": 0.2,
            "AUTO_MAX_SHARE": 0.8,
            "SMOOTH_HALF_LIFE_SECONDS": 2.0,
            "RESPONSIVE_HALF_LIFE_SECONDS": 0.5,
        }
        if constants != expected_constants:
            errors.append(f"unexpected compute fairness constants: {constants}")
    except Exception as error:
        errors.append(f"compute source semantics: {type(error).__name__}: {error}")

    try:
        normal_hashes = {
            arm: file_hashes.get(arm, {}).get(normal_launcher_path)
            for arm in ("stock", "patched", "auto")
        }
        downstream_hashes = {
            arm: file_hashes.get(arm, {}).get(downstream_launcher_path)
            for arm in ("stock", "patched", "auto")
        }
        normal_text = snapshot_path("patched", normal_launcher_path).read_text()
        auto_text = snapshot_path("auto", auto_launcher_path).read_text()
        downstream_text = snapshot_path("patched", downstream_launcher_path).read_text()
        auto_env = {
            key: inspect_records.get("auto", {}).get("env", {}).get(key)
            for key in r27.AUTO_ENV
        }
        auto_entrypoint = inspect_records.get("auto", {}).get("entrypoint")
        normal_entrypoints = {
            arm: inspect_records.get(arm, {}).get("entrypoint")
            for arm in ("stock", "patched")
        }
        launcher_semantics = {
            "normal_launcher_sha256_by_arm": normal_hashes,
            "normal_launcher_identical_all_arms": len(set(normal_hashes.values())) == 1,
            "normal_entrypoints": normal_entrypoints,
            "auto_entrypoint": auto_entrypoint,
            "auto_launcher_sha256": file_hashes.get("auto", {}).get(auto_launcher_path),
            "auto_image_env_projection": auto_env,
            "expected_auto_env": dict(r27.AUTO_ENV),
            "auto_image_env_matches_contract": auto_env == r27.AUTO_ENV,
            "auto_wrapper_native_cli_authoritative": all(
                marker in auto_text
                for marker in (
                    "runtime_args+=(--prefill-compute-share",
                    "runtime_args+=(--max-parallel-prefills",
                    "runtime_args+=(--prefill-policy",
                    "runtime_args+=(--decode-refill-target",
                    "export FAIRNESS_ENGINE=none",
                    "unset PREFILL_COMPUTE_SHARE",
                )
            ),
            "vram_target_page_default_2048_source_backed": (
                "target_block_size=2048" in normal_text
                and "elif [[ ${cache_mode} == lmcache ]]" in normal_text
            ),
            "downstream_launcher": {
                "container_path": downstream_launcher_path,
                "sha256_by_arm": downstream_hashes,
                "manifest_verified_all_arms": all(
                    verified.get(arm, {})
                    .get("files", {})
                    .get(downstream_launcher_path, {})
                    .get("verified")
                    for arm in ("stock", "patched", "auto")
                ),
                "identical_all_arms": len(set(downstream_hashes.values())) == 1,
                "vram_path_preserves_base_serving_command": (
                    'if [[ ${LMCACHE_ENABLED:-0} == 0 ]]' in downstream_text
                    and 'exec "${base_launcher}" "$@"' in downstream_text
                ),
            },
        }
        if not launcher_semantics["normal_launcher_identical_all_arms"]:
            errors.append("normal launcher differs across R27 arms")
        if not launcher_semantics["downstream_launcher"]["manifest_verified_all_arms"]:
            errors.append("downstream launcher is not manifest-verified across all R27 arms")
        if not launcher_semantics["downstream_launcher"]["identical_all_arms"]:
            errors.append("downstream launcher differs across R27 arms")
        if not launcher_semantics["downstream_launcher"][
            "vram_path_preserves_base_serving_command"
        ]:
            errors.append("downstream launcher vram delegation semantics were not found")
        if not launcher_semantics["auto_image_env_matches_contract"]:
            errors.append("AUTO_ENV differs from the pinned auto-image environment")
        if not launcher_semantics["auto_wrapper_native_cli_authoritative"]:
            errors.append("auto wrapper native-CLI semantics were not found")
        if auto_entrypoint != ["/usr/local/bin/serve-glm53-flash-auto.sh"]:
            errors.append(f"unexpected auto entrypoint: {auto_entrypoint}")
    except Exception as error:
        errors.append(f"launcher source semantics: {type(error).__name__}: {error}")

    try:
        bench_tree = ast_tree(rt.BENCH)
        generation_prompt = str(module_constant(bench_tree, "GENERATION_PROMPT"))
        build_messages = named_function(bench_tree.body, "build_messages")
        build_source = ast.get_source_segment(rt.BENCH.read_text(), build_messages) or ""
        decode_contract = {
            "benchmark_path": str(rt.BENCH),
            "benchmark_sha256": sha256_file(rt.BENCH),
            "generation_prompt_sha256": sha256_text(generation_prompt),
            "generation_prompt_characters": len(generation_prompt),
            "build_messages_sha256": sha256_text(build_source),
            "context_zero_uses_generation_prompt": (
                "if context_tokens > 0" in build_source
                and 'messages.append({"role": "user", "content": GENERATION_PROMPT})' in build_source
            ),
            "arguments": {
                "concurrency": list(DECODE_CONCURRENCIES),
                "contexts": [0],
                "duration_seconds_per_cell": 30,
                "skip_prefill": True,
                "max_tokens": 8192,
            },
        }
        if not decode_contract["context_zero_uses_generation_prompt"]:
            errors.append("decode benchmark context-zero prompt construction changed")
    except Exception as error:
        errors.append(f"decode prompt source semantics: {type(error).__name__}: {error}")

    patched_hashes = file_hashes.get("patched", {})
    auto_hashes = file_hashes.get("auto", {})
    patched_auto_shared = {
        path: patched_hashes[path] == auto_hashes.get(path)
        for path in patched_hashes
    }
    harness_paths = (
        Path(__file__),
        Path(r27.__file__),
        Path(rt.__file__),
        Path(workload_consumer.__file__),
        Path(workload_consumer.base.__file__),
        Path(steady_metrics.__file__),
        rt.BENCH,
    )
    contract: dict[str, Any] = {
        "status": "valid" if not errors else "invalid",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "schema": manifest.get("schema"),
        },
        "hash_truth": (
            "actual extracted file SHA-256 values and pinned image digests; inherited "
            "source.lock contents and image labels are not used as patch proof"
        ),
        "images": verified,
        "image_inspects": inspect_records,
        "patched_auto_shared_file_hashes": patched_auto_shared,
        "patched_auto_code_identical_for_all_patched_manifest_members": (
            bool(patched_auto_shared) and all(patched_auto_shared.values())
        ),
        "api": api_semantics,
        "lane_resolver": lane_semantics,
        "compute_fairness": compute_semantics,
        "launchers": launcher_semantics,
        "decode_input": decode_contract,
        "harness_source_sha256": {
            str(path): sha256_file(path) for path in harness_paths
        },
        "errors": errors,
    }
    contract["contract_sha256"] = sha256_text(canonical_json(contract))
    return contract


def native_policy_args(policy: Policy) -> list[str]:
    if policy.prefill_compute_share is None:
        raise ValueError("this R27 phase launches only fixed or automatic compute sharing")
    args = ["--prefill-compute-share", str(policy.prefill_compute_share)]
    if policy.prefill_compute_half_life is not None:
        args.extend(
            ["--prefill-compute-half-life", str(policy.prefill_compute_half_life)]
        )
    args.extend(
        [
            "--max-parallel-prefills",
            str(policy.max_parallel_prefills),
            "--prefill-policy",
            policy.prefill_policy,
            "--decode-refill-target",
            str(policy.decode_refill_target),
        ]
    )
    return args


def launch_contract(case: BootCase) -> dict[str, Any]:
    if case.as_shipped_auto:
        extra_env = dict(r27.AUTO_ENV)
        extra_args: list[str] = []
        delivery = "pinned auto-image entrypoint plus its exact image-default environment"
    else:
        extra_env = {
            "FAIRNESS_ENGINE": "none",
            "PREFILL_SCHEDULE_INTERVAL": "1",
            "VLLM_SERVER_DEV_MODE": "1",
        }
        extra_args = native_policy_args(case.policy)
        delivery = "native scheduler CLI with compatibility fairness wrapper disabled"
    stable = {
        "arm": case.arm,
        "image": r27.IMAGES[case.arm],
        "tp": 4,
        "dcp": 1,
        "spec": "mtp3",
        "cache": "vram",
        "kv": "fp8_ds_mla",
        "batch_tokens": case.batch_tokens,
        "max_model_len": 1048576,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.93,
        "nccl_channels": 16,
        "nccl_buffer_bytes": 2097152,
        "page_tokens": 2048,
        "base_url": "http://127.0.0.1:5002",
        "policy": case.policy.configured(),
        "policy_delivery": delivery,
        "extra_env": extra_env,
        "extra_args": extra_args,
    }
    return {**stable, "config_sha256": sha256_text(canonical_json(stable))}


def expected_effective_lanes(case: BootCase, sources: dict[str, Any]) -> int | None:
    configured = case.policy.max_parallel_prefills
    if isinstance(configured, int) and not isinstance(configured, bool):
        if configured == 1:
            return 1
        if case.arm in {"patched", "auto"}:
            return min(configured, 32)
        return None
    if configured == "auto" and case.arm in {"patched", "auto"}:
        value = sources.get("lane_resolver", {}).get(
            "patched_source_expected_auto_lanes_at_max_num_seqs_32"
        )
        return int(value) if isinstance(value, int) else None
    return None


def case_plan(case: BootCase, sources: dict[str, Any]) -> dict[str, Any]:
    return {
        **asdict(case),
        "policy": case.policy.configured(),
        "policy_sha256": sha256_text(canonical_json(case.policy.configured())),
        "launch": launch_contract(case),
        "source_expected_effective_lanes": expected_effective_lanes(case, sources),
        "stock_auto_effective_lane_expectation": (
            "observed readback only" if case.arm == "stock" and case.policy.max_parallel_prefills == "auto" else None
        ),
        "planned_measurements": {
            "api_readback": 1,
            "structural_rejection_posts": len(STRUCTURAL_FIELDS) if case.run_behavior else 0,
            "decode_cells": len(DECODE_CONCURRENCIES) if case.run_behavior else 0,
            "mixed_cells": (
                len(MIXED_PROFILES) * len(MIXED_CONCURRENCIES)
                if case.run_behavior
                else 0
            ),
            "live_tuning_cells": 1 if case.run_live_tuning else 0,
        },
    }


def phase_plan() -> dict[str, Any]:
    sources = source_contract()
    cases = [case_plan(case, sources) for case in CASES]
    behavior_count = sum(case.run_behavior for case in CASES)
    decode_cells = behavior_count * len(DECODE_CONCURRENCIES)
    mixed_cells = behavior_count * len(MIXED_PROFILES) * len(MIXED_CONCURRENCIES)
    live_cells = sum(case.run_live_tuning for case in CASES)
    errors: list[str] = []
    keys = [case.key for case in CASES]
    if len(keys) != len(set(keys)):
        errors.append("duplicate boot case keys")
    if BEHAVIOR_KEYS != (
        "stock-fixed-control",
        "patched-fixed-control",
        "patched-auto-lanes-bt4096",
        "auto-image-as-shipped",
    ):
        errors.append(f"unexpected behavior arms: {BEHAVIOR_KEYS}")
    if FIXED.configured() != CASES[0].policy.configured() or FIXED.configured() != CASES[1].policy.configured():
        errors.append("stock and patched fixed controls are not identical")
    if sources.get("status") != "valid":
        errors.append("source contract is invalid")
    stable: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "objective": (
            "fixed stock-vs-patched source attribution, patched fixed-vs-configured-auto "
            "policy attribution, and configured-auto-vs-auto-image default delivery attribution"
        ),
        "serial_execution": True,
        "cases": cases,
        "case_counts": {
            "boot_cases": len(CASES),
            "api_readbacks": len(CASES),
            "structural_rejection_posts": behavior_count * len(STRUCTURAL_FIELDS),
            "decode_bench_invocations": behavior_count,
            "decode_cells": decode_cells,
            "mixed_workload_invocations": behavior_count,
            "mixed_cells": mixed_cells,
            "live_tuning_invocations": live_cells,
            "live_tuning_cells": live_cells,
            "measurement_case_count": decode_cells + mixed_cells + live_cells,
        },
        "comparison_contract": {
            "source_effect": ["stock-fixed-control", "patched-fixed-control"],
            "configured_policy_effect": [
                "patched-fixed-control",
                "patched-auto-lanes-bt4096",
            ],
            "default_delivery_effect": [
                "patched-auto-lanes-bt4096",
                "auto-image-as-shipped",
            ],
            "lane_sweep_budgets": list(BUDGETS),
            "stock_lane_values": "observed controls, never predicted with a legacy geometry formula",
            "patched_lane_source_expectation": "constant auto cap clipped only by max_num_seqs",
            "configured_and_effective_lanes_compared_separately": True,
        },
        "workload_contract": {
            "consumer": str(Path(workload_consumer.__file__).resolve()),
            "consumer_schema_selector": EXISTING_CONSUMER_SCHEMA,
            "candidate_labels_are_actual_R27_labels": True,
            "profiles": list(MIXED_PROFILES),
            "concurrencies": list(MIXED_CONCURRENCIES),
            "seconds_per_profile": PROFILE_SECONDS,
            "trace_seed": TRACE_SEED,
            "trace_hash_excludes_only_cache_namespace": True,
            "live_updates": list(LIVE_UPDATES),
        },
        "decode_contract": sources.get("decode_input"),
        "interpretation_contract": {
            "speed_requires_clean_gpu_isolation_coverage": True,
            "raw_receipts_are_never rewritten_by_summary analysis": True,
            "request_completion_valid_stream_and_idle_are_separate_from_policy_qos": True,
            "reasoning_only_or_empty_visible_content_is_not_scored": True,
            "length_limited_and_non_length_empty_visible_results_are_distinct": True,
            "no_spec_verifier_steps_would_be unavailable_not_failed": True,
            "off_mode_missing_compute_split_would_not_fail_functional_workload": True,
            "no_retry_changes_inputs_or_relabels_a_result": True,
        },
        "source_contract": sources,
        "validation_errors": errors,
    }
    stable["plan_sha256"] = sha256_text(canonical_json(stable))
    stable["valid"] = not errors
    return stable


def ensure_runtime_scope() -> None:
    r27.ensure_scope()
    if rt.PORT != 5002 or rt.BASE_URL != "http://127.0.0.1:5002":
        raise RuntimeError("R27 scheduler phase requires loopback http://127.0.0.1:5002")


def ensure_fresh_outputs() -> None:
    collisions: list[Path] = []
    phase_root = rt.ROOT / OUTPUT_DIR
    if phase_root.exists():
        collisions.append(phase_root)
    collisions.extend(rt.ROOT.glob("r27-scheduler-*.launch.json"))
    collisions.extend(rt.ROOT.glob("r27-scheduler-*.boot.command.json"))
    if collisions:
        raise RuntimeError(
            "R27 scheduler outputs already exist; use a fresh BATTERY_ROOT to preserve raw receipts: "
            + ", ".join(str(path) for path in collisions[:8])
        )


def raw_json(name: str, value: object) -> str:
    return str(rt.save_json(f"{OUTPUT_DIR}/raw/{name}.json", value))


def http_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.time()
    try:
        response = HTTP.request(
            method,
            rt.BASE_URL + path,
            json=payload,
            timeout=15.0,
        )
        text = response.text
        try:
            body: object = response.json()
        except (ValueError, json.JSONDecodeError):
            body = None
        return {
            "observed_at_utc": utc_now(),
            "method": method,
            "url": rt.BASE_URL + path,
            "request": payload,
            "status_code": response.status_code,
            "body": body,
            "body_text": text,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as error:
        return {
            "observed_at_utc": utc_now(),
            "method": method,
            "url": rt.BASE_URL + path,
            "request": payload,
            "status_code": None,
            "body": None,
            "body_text": "",
            "elapsed_seconds": time.time() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def exchange_config(exchange: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    return workload_consumer.response_config(exchange.get("body"))


def expected_launch_env(case: BootCase) -> dict[str, str]:
    expected = {
        "MODEL": "/model",
        "SERVED_MODEL_NAME": rt.MODEL_NAME,
        "HOST": "127.0.0.1",
        "PORT": "5002",
        "TP": "4",
        "DCP": "1",
        "CACHE_MODE": "vram",
        "KV_CACHE_QUANT": "fp8_ds_mla",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "MAX_MODEL_LEN": "1048576",
        "MAX_NUM_SEQS": "32",
        "MAX_NUM_BATCHED_TOKENS": str(case.batch_tokens),
        "PREFILL_SCHEDULE_INTERVAL": "1",
        "FAIRNESS_ENGINE": "compute_share",
        "PREFILL_COMPUTE_SHARE": "0.4",
        "GPU_MEMORY_UTILIZATION": "0.93",
        "NCCL_MIN_NCHANNELS": "16",
        "NCCL_MAX_NCHANNELS": "16",
        "NCCL_BUFFSIZE": "2097152",
        "VLLM_SERVER_DEV_MODE": "1",
        "SPECULATOR": "mtp",
        "MTP_DEPTH": "3",
    }
    expected.update(launch_contract(case)["extra_env"])
    return expected


def runtime_contract(case: BootCase, boot_label: str) -> dict[str, Any]:
    launch_path = rt.ROOT / f"{boot_label}.launch.json"
    launch, launch_error = read_json_object(launch_path)
    expected_env = expected_launch_env(case)
    env = launch.get("env") if launch and isinstance(launch.get("env"), dict) else None
    env_mismatches = mismatches(env, expected_env)
    expected_args = launch_contract(case)["extra_args"]
    top_level_expected = {
        "image": r27.IMAGES[case.arm],
        "tp": 4,
        "dcp": 1,
        "spec": "mtp3",
        "cache": "vram",
        "kv": "fp8_ds_mla",
        "extra_args": expected_args,
    }
    top_level_mismatches = mismatches(launch, top_level_expected)
    metrics_path = rt.ROOT / f"{boot_label}.metrics.txt"
    try:
        metrics_text = metrics_path.read_text()
        metrics_error = None
    except Exception as error:
        metrics_text = ""
        metrics_error = f"{type(error).__name__}: {error}"
    parsed_metrics = workload_consumer.parse_metrics(metrics_text)
    cache_configs = parsed_metrics.get("cache_configs", [])
    cache_checks: list[dict[str, Any]] = []
    for item in cache_configs:
        labels = item.get("labels", {}) if isinstance(item, dict) else {}
        expected_cache = {
            "block_size": "2048",
            "cache_dtype": "fp8",
            "gpu_memory_utilization": "0.93",
        }
        cache_checks.append(
            {
                "labels": labels,
                "mismatches": mismatches(labels, expected_cache),
                "matched": not mismatches(labels, expected_cache),
            }
        )
    passed = (
        launch_error is None
        and not env_mismatches
        and not top_level_mismatches
        and metrics_error is None
        and bool(cache_checks)
        and all(item["matched"] for item in cache_checks)
    )
    receipt = {
        "case": case.key,
        "launch_path": str(launch_path),
        "launch_sha256": sha256_file(launch_path),
        "launch_read_error": launch_error,
        "expected_launch_config_sha256": launch_contract(case)["config_sha256"],
        "expected_env": expected_env,
        "env_mismatches": env_mismatches,
        "top_level_mismatches": top_level_mismatches,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics_error": metrics_error,
        "cache_config_info": cache_checks,
        "matched_page_kv_and_runtime_config": passed,
        "loopback_dev_api": rt.BASE_URL,
        "passed": passed,
    }
    receipt["receipt_path"] = raw_json(f"{case.key}-runtime-contract", receipt)
    rt.record_gate(f"r27-scheduler:{case.key}:runtime-contract", passed, receipt)
    return receipt


def api_readback(case: BootCase, sources: dict[str, Any]) -> dict[str, Any]:
    exchange = http_json("GET", POLICY_PATH)
    config, location = exchange_config(exchange)
    observed_schema = workload_consumer.detect_policy_schema(config)
    expected_config = case.policy.configured()
    configured_mismatches = mismatches(config, expected_config)
    get_fields = sources.get("api", {}).get("get_fields", [])
    source_get_fields_present = (
        isinstance(config, dict) and set(get_fields).issubset(config)
    )
    effective_lane = config.get("effective_max_parallel_prefills") if config else None
    source_expected_lane = expected_effective_lanes(case, sources)
    lane_is_integer = isinstance(effective_lane, int) and not isinstance(effective_lane, bool)
    if source_expected_lane is None:
        effective_lane_coherent = lane_is_integer and 1 <= effective_lane <= int(
            sources.get("lane_resolver", {}).get("auto_lane_cap", 4)
        )
    else:
        effective_lane_coherent = lane_is_integer and effective_lane == source_expected_lane
    effective_refill = config.get("effective_decode_refill_target") if config else None
    configured_refill = case.policy.decode_refill_target
    refill_coherent = (
        effective_refill == effective_lane
        if configured_refill == "auto"
        else effective_refill == configured_refill
    )
    controller_constants = sources.get("compute_fairness", {}).get("constants", {})
    effective_share = config.get("effective_prefill_compute_share") if config else None
    effective_half_life = (
        config.get("effective_prefill_compute_half_life_seconds") if config else None
    )
    if case.policy.prefill_compute_share == "auto":
        share_coherent = (
            isinstance(effective_share, (int, float))
            and not isinstance(effective_share, bool)
            and float(controller_constants.get("AUTO_MIN_SHARE", 0.2))
            <= float(effective_share)
            <= float(controller_constants.get("AUTO_MAX_SHARE", 0.8))
        )
        half_life_coherent = values_equal(
            effective_half_life,
            controller_constants.get("RESPONSIVE_HALF_LIFE_SECONDS"),
        )
    else:
        share_coherent = values_equal(effective_share, case.policy.prefill_compute_share)
        half_life_coherent = effective_half_life is None
    passed = (
        exchange.get("status_code") == 200
        and location == "body"
        and observed_schema == EXISTING_CONSUMER_SCHEMA
        and source_get_fields_present
        and not configured_mismatches
        and effective_lane_coherent
        and refill_coherent
        and share_coherent
        and half_life_coherent
    )
    receipt = {
        "case": case.key,
        "candidate_label": case.candidate_label,
        "exchange": exchange,
        "config_location": location,
        "observed_config": config,
        "observed_consumer_schema": observed_schema,
        "consumer_schema_selector": EXISTING_CONSUMER_SCHEMA,
        "consumer_selector_is_not_candidate_release_alias": True,
        "expected_configured": expected_config,
        "configured_mismatches": configured_mismatches,
        "source_get_fields": get_fields,
        "source_get_fields_present": source_get_fields_present,
        "source_expected_effective_lanes": source_expected_lane,
        "stock_auto_lane_value_is_observational": (
            case.arm == "stock" and case.policy.max_parallel_prefills == "auto"
        ),
        "effective_lane_coherent": effective_lane_coherent,
        "effective_refill_coherent": refill_coherent,
        "effective_compute_share_coherent": share_coherent,
        "effective_half_life_coherent": half_life_coherent,
        "passed": passed,
        "failure_class": None if passed else "source_live_scheduler_contract_mismatch",
    }
    receipt["receipt_path"] = raw_json(f"{case.key}-api-readback", receipt)
    rt.record_gate(f"r27-scheduler:{case.key}:api-readback", passed, receipt)
    return receipt


def structural_rejections(case: BootCase, sources: dict[str, Any]) -> dict[str, Any]:
    before = http_json("GET", POLICY_PATH)
    original, _ = exchange_config(before)
    source_structural = tuple(sources.get("api", {}).get("structural_fields", []))
    alternatives: dict[str, Any] = {
        "max_parallel_prefills": 1 if case.policy.max_parallel_prefills != 1 else "auto",
        "prefill_policy": (
            "round-robin" if case.policy.prefill_policy == "decode-aware" else "decode-aware"
        ),
        "decode_refill_target": 2 if case.policy.decode_refill_target != 2 else "auto",
    }
    events: list[dict[str, Any]] = []
    state_fields = tuple(sources.get("api", {}).get("get_fields", []))
    original_state = projection(original, state_fields)
    for field in source_structural:
        payload = {field: alternatives[field]}
        posted = http_json("POST", POLICY_PATH, payload)
        after = http_json("GET", POLICY_PATH)
        after_config, _ = exchange_config(after)
        after_state = projection(after_config, state_fields)
        rejected = posted.get("status_code") == 422
        unchanged = after_state == original_state
        events.append(
            {
                "field": field,
                "attempted_value": alternatives[field],
                "request": payload,
                "post": posted,
                "after": after,
                "http_422_rejection": rejected,
                "full_source_get_state_unchanged": unchanged,
                "passed": rejected and unchanged,
            }
        )
    passed = (
        source_structural == STRUCTURAL_FIELDS
        and len(events) == len(STRUCTURAL_FIELDS)
        and all(event["passed"] for event in events)
    )
    receipt = {
        "case": case.key,
        "source_request_model_fields": sources.get("api", {}).get("request_model_fields"),
        "source_router_extra_fields_forbidden": sources.get("api", {}).get(
            "router_extra_fields_forbidden"
        ),
        "source_structural_fields": list(source_structural),
        "before": before,
        "original_state": original_state,
        "events": events,
        "state_unchanged_basis": "all fields returned by the shipped Scheduler.get_prefill_fairness source",
        "passed": passed,
        "failure_class": None if passed else "live_api_structural_rejection_defect",
    }
    receipt["receipt_path"] = raw_json(f"{case.key}-structural-rejections", receipt)
    rt.record_gate(f"r27-scheduler:{case.key}:structural-rejections", passed, receipt)
    return receipt


def isolation_for_command(command_path: Path) -> dict[str, Any]:
    command, command_error = read_json_object(command_path)
    isolation_path = rt.ROOT / "gpu-isolation-events.jsonl"
    if command_error is not None or command is None:
        return {
            "status": "coverage_unavailable",
            "measurement_supported": False,
            "command_path": str(command_path),
            "error": command_error,
        }
    try:
        lines = isolation_path.read_text().splitlines()
    except Exception as error:
        return {
            "status": "coverage_unavailable",
            "measurement_supported": False,
            "command_path": str(command_path),
            "isolation_path": str(isolation_path),
            "error": f"{type(error).__name__}: {error}",
        }
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(row, dict) and isinstance(row.get("timestamp"), (int, float)):
            rows.append(row)
    start = float(command.get("started_at", 0.0))
    end = float(command.get("finished_at", 0.0))
    selected = [row for row in rows if start <= float(row["timestamp"]) <= end]
    covered = (
        end > start
        and len(selected) >= 2
        and float(selected[0]["timestamp"]) - start <= 5.0
        and end - float(selected[-1]["timestamp"]) <= 5.0
        and all(
            float(right["timestamp"]) - float(left["timestamp"]) <= 5.0
            for left, right in zip(selected, selected[1:])
        )
    )
    foreign = [row for row in selected if row.get("foreign")]
    unhealthy = [row for row in selected if row.get("speed_eligible", True) is not True]
    if not covered:
        status = "coverage_unavailable"
    elif foreign:
        status = "foreign_gpu_overlap"
    elif unhealthy:
        status = "gpu_health_fault"
    else:
        status = "clean"
    return {
        "status": status,
        "measurement_supported": status == "clean",
        "command_path": str(command_path),
        "command_window": {"started_at": start, "finished_at": end},
        "isolation_path": str(isolation_path),
        "isolation_sha256": sha256_file(isolation_path),
        "selected_samples": len(selected),
        "malformed_rows": malformed,
        "first_sample_at": selected[0]["timestamp"] if selected else None,
        "last_sample_at": selected[-1]["timestamp"] if selected else None,
        "foreign_samples": foreign,
        "producer_ineligible_samples": len(unhealthy),
    }


def decode_metadata_matches(result: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    metadata = result.get("metadata", {}) if result else {}
    observed = {
        "concurrency_levels": metadata.get("concurrency_levels"),
        "context_lengths": metadata.get("context_lengths"),
        "duration_per_test": metadata.get("duration_per_test"),
        "skip_prefill": metadata.get("skip_prefill"),
        "max_tokens": metadata.get("max_tokens"),
    }
    expected = {
        "concurrency_levels": list(DECODE_CONCURRENCIES),
        "context_lengths": [0],
        "duration_per_test": 30.0,
        "skip_prefill": True,
        "max_tokens": 8192,
    }
    delta = mismatches(observed, expected)
    return not delta, {"expected": expected, "observed": observed, "mismatches": delta}


def run_decode(case: BootCase, sources: dict[str, Any]) -> dict[str, Any]:
    label = f"{OUTPUT_DIR}/raw/{case.key}-decode"
    sample_path = rt.ROOT / f"{label}.steady.metrics.jsonl"
    started = utc_now()
    benchmark_ok = False
    execution_error: str | None = None
    try:
        with steady_metrics.Recorder(rt.BASE_URL, sample_path):
            benchmark_ok = rt.bench(
                label,
                conc="1,8",
                contexts="0",
                duration=30,
                prefill=False,
            )
    except Exception as error:
        execution_error = f"{type(error).__name__}: {error}"
    result_path = rt.ROOT / f"{label}.json"
    result, result_error = read_json_object(result_path)
    metadata_ok, metadata_receipt = decode_metadata_matches(result)
    steady_summary: dict[str, Any] | None = None
    steady_error: str | None = None
    if benchmark_ok:
        try:
            steady_summary = steady_metrics.summarize(rt.ROOT, label)
        except Exception as error:
            steady_error = f"{type(error).__name__}: {error}"
    command_path = rt.ROOT / f"{label}.bench.command.json"
    isolation = isolation_for_command(command_path)
    cells: list[dict[str, Any]] = []
    summary_cells = steady_summary.get("cells", []) if steady_summary else []
    by_key = {
        (row.get("concurrency"), row.get("context_tokens")): row
        for row in summary_cells
        if isinstance(row, dict)
    }
    for concurrency in DECODE_CONCURRENCIES:
        row = by_key.get((concurrency, 0))
        verifier_available = bool(
            row
            and row.get("valid")
            and row.get("aggregate_verifier_steps_per_second") is not None
            and row.get("accepted_draft_tokens_per_step") is not None
        )
        supported = bool(
            benchmark_ok
            and result_error is None
            and metadata_ok
            and steady_summary
            and steady_summary.get("all_windows_valid")
            and row
            and row.get("valid")
            and verifier_available
            and isolation.get("measurement_supported")
        )
        cells.append(
            {
                "concurrency": concurrency,
                "context_tokens": 0,
                "measurement_classification": "measured" if supported else "unsupported",
                "steady_values": row,
                "mtp3_verifier_counters_available": verifier_available,
                "no_spec_counter_interpretation": (
                    "not applicable to this MTP3 phase; a no-spec arm would not require verifier counters"
                ),
            }
        )
    passed = all(cell["measurement_classification"] == "measured" for cell in cells)
    receipt = {
        "case": case.key,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "benchmark_executed": execution_error is None,
        "benchmark_ok": benchmark_ok,
        "execution_error": execution_error,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "result_read_error": result_error,
        "command_path": str(command_path),
        "command_sha256": sha256_file(command_path),
        "steady_samples_path": str(sample_path),
        "steady_samples_sha256": sha256_file(sample_path),
        "steady_summary_path": str(rt.ROOT / f"{label}.steady-summary.json"),
        "steady_error": steady_error,
        "metadata_contract": metadata_receipt,
        "same_prompt_contract": sources.get("decode_input"),
        "isolation": isolation,
        "cells": cells,
        "measurement_classification": "measured" if passed else "unsupported",
        "failure_class": None if passed else "decode_measurement_unsupported_or_failed",
        "no_retry_performed": True,
    }
    receipt["receipt_path"] = raw_json(f"{case.key}-decode-receipt", receipt)
    rt.record_gate(f"r27-scheduler:{case.key}:decode-measured", passed, receipt)
    return receipt


def workload_command(
    case: BootCase,
    output: Path,
    *,
    live_tuning: bool,
) -> list[str]:
    profiles = ("periodic-128k",) if live_tuning else MIXED_PROFILES
    concurrencies = (16,) if live_tuning else MIXED_CONCURRENCIES
    updates = LIVE_UPDATES if live_tuning else ()
    namespace_suffix = "live" if live_tuning else "mixed"
    return [
        sys.executable,
        str(Path(workload_consumer.__file__).resolve()),
        "--base-url",
        rt.BASE_URL,
        "--model",
        rt.MODEL_NAME,
        "--candidate-label",
        case.candidate_label,
        "--policy-label",
        case.policy.key + ("-live-boundaries" if live_tuning else ""),
        "--expected-api-schema",
        EXISTING_CONSUMER_SCHEMA,
        "--expected-policy-json",
        canonical_json(case.policy.configured()),
        "--policy-updates-json",
        canonical_json(list(updates)),
        "--profiles",
        ",".join(profiles),
        "--concurrencies",
        ",".join(str(value) for value in concurrencies),
        "--repeats",
        "1",
        "--trace-seed",
        TRACE_SEED,
        "--cache-namespace",
        f"r27-scheduler-{case.key}-{namespace_suffix}",
        "--cache-mode",
        "vram",
        "--memory-role",
        "none",
        "--session-context-tokens",
        "8192",
        "--prime-decode-tokens",
        "32",
        "--turn-decode-tokens",
        "128",
        "--cold-decode-tokens",
        "1",
        "--incremental-filler-tokens",
        "220",
        "--synthetic-assistant-tokens",
        "96",
        "--hot-turn-period-seconds",
        "6",
        "--baseline-seconds",
        "30",
        "--periodic-seconds",
        "60",
        "--short-heavy-seconds",
        "60",
        "--periodic-cold-tokens",
        "131072",
        "--cold-period-seconds",
        "15",
        "--short-prefill-tokens",
        "2048,4096,8192",
        "--short-prefill-period-seconds",
        "1.5",
        "--drain-timeout-seconds",
        "240",
        "--request-timeout-seconds",
        "300",
        "--tokenize-timeout-seconds",
        "180",
        "--idle-timeout-seconds",
        "180",
        "--cooldown-seconds",
        "2",
        "--between-scenarios-seconds",
        "2",
        "--start-delay-seconds",
        "1",
        "--metric-sample-seconds",
        "0.25",
        "--policy-sample-seconds",
        "1",
        "--connect-timeout-seconds",
        "30",
        "--max-connections",
        "256",
        "--max-keepalive-connections",
        "128",
        "--tokenize-concurrency",
        "8",
        "--output",
        str(output),
    ]


def find_gate(cell: dict[str, Any], name: str) -> dict[str, Any] | None:
    for gate in cell.get("gates", []):
        if isinstance(gate, dict) and gate.get("name") == name:
            return gate
    return None


def answer_classification(cells: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for cell in cells:
        for row in cell.get("requests", []):
            if not isinstance(row, dict):
                continue
            reasons = {str(value) for value in row.get("finish_reasons", [])}
            visible = bool(str(row.get("content") or "").strip())
            reasoning = bool(str(row.get("reasoning_content") or "").strip())
            completed = row.get("status") == "completed"
            if not completed:
                label = "incomplete_not_scored"
            elif visible:
                label = "visible_answer_present_correctness_not_scored"
            elif reasoning and "length" in reasons:
                label = "reasoning_only_length_limited_not_scored"
            elif reasoning:
                label = "reasoning_only_non_length_empty_visible_requires_review_not_scored"
            elif "length" in reasons:
                label = "empty_visible_length_limited_not_scored"
            else:
                label = "empty_visible_non_length_requires_review_not_scored"
            counts[label] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "correctness_scored": False,
        "reasoning_only_empty_final_scored_as_wrong": False,
        "length_limit_distinguished_from_non_length_empty_visible": True,
    }


def trace_identities(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cell in cells:
        trace = cell.get("trace") if isinstance(cell, dict) else None
        if not isinstance(trace, dict):
            continue
        result.append(
            {
                "profile": cell.get("profile"),
                "concurrency": cell.get("concurrency"),
                "repeat": cell.get("repeat"),
                "trace_hash": trace.get("trace_hash"),
                "session_prompt_sha256": [
                    item.get("sha256")
                    for item in trace.get("session_prompts", [])
                    if isinstance(item, dict)
                ],
                "cold_prompt_sha256": [
                    item.get("sha256")
                    for item in trace.get("cold_prompts", [])
                    if isinstance(item, dict)
                ],
                "cache_namespace": trace.get("cache_namespace"),
                "cache_namespace_excluded_from_trace_hash": trace.get(
                    "cache_namespace_excluded_from_trace_hash"
                ),
            }
        )
    return result


def nested(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def workload_functional(cells: list[dict[str, Any]], expected_cells: int) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    aggregate_completion: Counter[str] = Counter()
    for cell in cells:
        protocol = find_gate(cell, "all_stream_protocol_responses_complete")
        idle = find_gate(cell, "server_idle_after_drain")
        completion = nested(cell, "summary", "completion_classification", "counts")
        if isinstance(completion, dict):
            for name, count in completion.items():
                if isinstance(count, int):
                    aggregate_completion[str(name)] += count
        details.append(
            {
                "profile": cell.get("profile"),
                "concurrency": cell.get("concurrency"),
                "repeat": cell.get("repeat"),
                "cell_status": cell.get("status"),
                "valid_stream_passed": bool(protocol and protocol.get("passed") is True),
                "idle_after_drain_passed": bool(idle and idle.get("passed") is True),
                "completion_classification": completion,
            }
        )
    return {
        "expected_cells": expected_cells,
        "observed_cells": len(cells),
        "all_cells_present": len(cells) == expected_cells,
        "all_valid_stream_checks_passed": (
            len(cells) == expected_cells and all(row["valid_stream_passed"] for row in details)
        ),
        "all_idle_after_drain_checks_passed": (
            len(cells) == expected_cells and all(row["idle_after_drain_passed"] for row in details)
        ),
        "request_completion_classifications": dict(sorted(aggregate_completion.items())),
        "cells": details,
        "independent_from_policy_qos": True,
    }


def qos_cells(cells: list[dict[str, Any]], isolation: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cell in cells:
        summary = cell.get("summary") if isinstance(cell.get("summary"), dict) else {}
        observed = {
            "hot_ttft_p95_seconds": nested(summary, "hot_ttft_seconds", "p95"),
            "hot_output_gap_p95_seconds": nested(
                summary, "hot_decode_chunk_gap_seconds", "p95"
            ),
            "cold_ttft_p95_seconds": nested(summary, "cold_ttft_seconds", "p95"),
            "completion_tokens_per_second": nested(
                summary,
                "normalization",
                "completion_tokens_in_completed_work_per_second",
            ),
            "prefill_compute_fraction": nested(
                summary, "model_work_split_measurement_window", "prefill_fraction"
            ),
            "decode_compute_fraction": nested(
                summary, "model_work_split_measurement_window", "decode_fraction"
            ),
            "effective_compute_share": nested(
                cell, "api", "summary", "effective_prefill_compute_share"
            ),
            "effective_max_parallel_prefills": nested(
                cell, "api", "summary", "effective_max_parallel_prefills"
            ),
            "effective_decode_refill_target": nested(
                cell, "api", "summary", "effective_decode_refill_target"
            ),
        }
        required = (
            observed["hot_ttft_p95_seconds"],
            observed["hot_output_gap_p95_seconds"],
            observed["completion_tokens_per_second"],
        )
        supported = (
            cell.get("status") == "complete"
            and isolation.get("measurement_supported") is True
            and all(value is not None for value in required)
        )
        result.append(
            {
                "profile": cell.get("profile"),
                "concurrency": cell.get("concurrency"),
                "repeat": cell.get("repeat"),
                "measurement_classification": "measured" if supported else "unsupported",
                "observed_values": observed,
                "unsupported_values_are_not_speed_claims": not supported,
            }
        )
    return result


def analyze_live_updates(
    cells: list[dict[str, Any]], sources: dict[str, Any]
) -> dict[str, Any]:
    constants = sources.get("compute_fairness", {}).get("constants", {})
    expected_half = {
        "smooth": constants.get("SMOOTH_HALF_LIFE_SECONDS"),
        "responsive": constants.get("RESPONSIVE_HALF_LIFE_SECONDS"),
    }
    checks: list[dict[str, Any]] = []
    for cell in cells:
        observations = nested(cell, "api", "live_updates")
        if not isinstance(observations, list):
            observations = []
        for index, expected in enumerate(LIVE_UPDATES):
            observed = observations[index] if index < len(observations) else None
            after_exchange = observed.get("after", {}) if isinstance(observed, dict) else {}
            after_config, _ = exchange_config(after_exchange)
            expected_config = expected["config"]
            configured_ok = not mismatches(after_config, expected_config)
            share = expected_config["prefill_compute_share"]
            half = expected_config["prefill_compute_half_life"]
            effective_share = (
                after_config.get("effective_prefill_compute_share") if after_config else None
            )
            effective_half = (
                after_config.get("effective_prefill_compute_half_life_seconds")
                if after_config
                else None
            )
            if isinstance(share, (int, float)) and not isinstance(share, bool):
                effective_ok = values_equal(effective_share, share) and effective_half is None
            else:
                effective_ok = (
                    isinstance(effective_share, (int, float))
                    and not isinstance(effective_share, bool)
                    and float(constants.get("AUTO_MIN_SHARE", 0.2))
                    <= float(effective_share)
                    <= float(constants.get("AUTO_MAX_SHARE", 0.8))
                    and values_equal(effective_half, expected_half.get(str(half)))
                )
            passed = bool(
                observed
                and observed.get("active_work_before")
                and observed.get("accepted_and_read_back")
                and observed.get("configured_state_changed")
                and configured_ok
                and effective_ok
            )
            checks.append(
                {
                    "cell": {
                        "profile": cell.get("profile"),
                        "concurrency": cell.get("concurrency"),
                        "repeat": cell.get("repeat"),
                    },
                    "index": index,
                    "expected": expected,
                    "observed": observed,
                    "configured_readback_matched": configured_ok,
                    "effective_state_source_coherent": effective_ok,
                    "passed": passed,
                }
            )
    passed = len(checks) == len(LIVE_UPDATES) and all(check["passed"] for check in checks)
    return {
        "source_allowed_mutable_fields": list(MUTABLE_FIELDS),
        "tested_fixed_share_endpoints": [0.2, 0.8],
        "tested_half_life_modes": ["smooth", "responsive"],
        "checks": checks,
        "passed": passed,
        "failure_class": None if passed else "live_compute_tuning_defect_or_inactive_probe",
    }


def run_workload(
    case: BootCase,
    sources: dict[str, Any],
    *,
    live_tuning: bool,
) -> dict[str, Any]:
    suffix = "live-tuning" if live_tuning else "mixed"
    output = rt.ROOT / OUTPUT_DIR / "raw" / f"{case.key}-{suffix}.json"
    command = workload_command(case, output, live_tuning=live_tuning)
    profiles = ("periodic-128k",) if live_tuning else MIXED_PROFILES
    concurrencies = (16,) if live_tuning else MIXED_CONCURRENCIES
    expected_identities = {
        (profile, concurrency, 1)
        for concurrency in concurrencies
        for profile in profiles
    }
    expected_cells = len(expected_identities)
    measurement_seconds = sum(PROFILE_SECONDS[profile] for profile in profiles) * len(
        concurrencies
    )
    timeout = int(measurement_seconds + expected_cells * (240 + 300 + 180) + 900)
    started = utc_now()
    return_code = rt.run(
        command,
        label=f"{OUTPUT_DIR}/raw/{case.key}-{suffix}",
        timeout=timeout,
        env=rt.PROXY_ENV,
    )
    parsed, parse_error = read_json_object(output)
    cells = (
        [cell for cell in parsed.get("cells", []) if isinstance(cell, dict)]
        if parsed
        else []
    )
    observed_identities = {
        (cell.get("profile"), cell.get("concurrency"), cell.get("repeat"))
        for cell in cells
    }
    all_recorded = (
        parse_error is None
        and len(cells) == expected_cells
        and observed_identities == expected_identities
    )
    directly_attempted_identities = {
        (cell.get("profile"), cell.get("concurrency"), cell.get("repeat"))
        for cell in cells
        if isinstance(cell.get("trace"), dict)
        and isinstance(cell["trace"].get("trace_hash"), str)
        and bool(cell["trace"].get("trace_hash"))
    }
    all_attempted = (
        all_recorded and directly_attempted_identities == expected_identities
    )
    command_path = rt.ROOT / OUTPUT_DIR / "raw" / f"{case.key}-{suffix}.command.json"
    isolation = isolation_for_command(command_path)
    functional = workload_functional(cells, expected_cells)
    qos = qos_cells(cells, isolation)
    measurement_classification = (
        "measured"
        if all_attempted
        and len(qos) == expected_cells
        and all(row["measurement_classification"] == "measured" for row in qos)
        else "unsupported"
    )
    qualification_passed = (
        all_attempted
        and len(cells) == expected_cells
        and all(cell.get("status") == "complete" for cell in cells)
    )
    receipt = {
        "case": case.key,
        "kind": suffix,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "candidate_label": case.candidate_label,
        "candidate_label_is_actual_R27_release": True,
        "consumer_schema_selector": EXISTING_CONSUMER_SCHEMA,
        "consumer_selector_is_not_candidate_alias": True,
        "command": command,
        "command_config_sha256": sha256_text(canonical_json(command[:-2])),
        "command_path": str(command_path),
        "command_sha256": sha256_file(command_path),
        "return_code": return_code,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "parse_error": parse_error,
        "expected_identities": [list(value) for value in sorted(expected_identities)],
        "observed_identities": [list(value) for value in sorted(observed_identities)],
        "directly_attempted_identities": [
            list(value) for value in sorted(directly_attempted_identities)
        ],
        "all_requested_cells_recorded": all_recorded,
        "all_requested_cells_attempted": all_attempted,
        "execution_status": "complete" if all_attempted else "incomplete",
        "qualification_status": "passed" if qualification_passed else "findings",
        "functional": functional,
        "policy_qos": {
            "measurement_classification": measurement_classification,
            "isolation": isolation,
            "cells": qos,
        },
        "answer_classification": answer_classification(cells),
        "trace_identities": trace_identities(cells),
        "live_tuning": analyze_live_updates(cells, sources) if live_tuning else None,
        "raw_result_is_immutable": True,
        "no_retry_performed": True,
    }
    receipt["receipt_path"] = raw_json(f"{case.key}-{suffix}-receipt", receipt)
    rt.record_gate(
        f"r27-scheduler:{case.key}:{suffix}:all-attempted",
        all_attempted,
        {
            "expected": receipt["expected_identities"],
            "recorded": receipt["observed_identities"],
            "directly_attempted": receipt["directly_attempted_identities"],
            "parse_error": parse_error,
            "output": str(output),
        },
    )
    rt.record_gate(
        f"r27-scheduler:{case.key}:{suffix}:qualification",
        qualification_passed,
        {
            "return_code": return_code,
            "cell_statuses": [cell.get("status") for cell in cells],
            "functional": functional,
        },
    )
    return receipt


def unattempted_measurement(kind: str, reason: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "measurement_classification": "unattempted",
        "reason": reason,
    }


def run_boot_case(case: BootCase, sources: dict[str, Any]) -> dict[str, Any]:
    boot_label = f"r27-scheduler-{case.key}"
    contract = launch_contract(case)
    record: dict[str, Any] = {
        "case": case_plan(case, sources),
        "boot_label": boot_label,
        "boot_attempted": False,
        "boot_status": "not_started",
        "started_at_utc": utc_now(),
        "errors": [],
    }
    booted = False
    rt.note(
        f"R27 SCHEDULER BOOT START case={case.key} arm={case.arm} "
        f"batch={case.batch_tokens} policy={case.policy.key}"
    )
    try:
        record["boot_attempted"] = True
        try:
            booted = r27.boot(
                boot_label,
                arm=case.arm,
                spec="mtp3",
                dcp=1,
                cache="vram",
                kv="fp8_ds_mla",
                batch=case.batch_tokens,
                extra_env=contract["extra_env"],
                extra_args=contract["extra_args"],
            )
        except Exception as error:
            record["boot_status"] = "exception"
            record["errors"].append(
                {
                    "operation": "boot",
                    "classification": "boot_or_harness_exception",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            record["boot_status"] = "ready" if booted else "failed_to_become_ready"
        rt.record_gate(
            f"r27-scheduler:{case.key}:boot",
            booted,
            {
                "boot_status": record["boot_status"],
                "image": r27.IMAGES[case.arm],
                "launch_contract": contract,
            },
        )
        if not booted:
            reason = f"dependent operation unavailable because boot status is {record['boot_status']}"
            record["api"] = unattempted_measurement("api_readback", reason)
            if case.run_behavior:
                record["structural_rejections"] = unattempted_measurement(
                    "structural_rejections", reason
                )
                record["decode"] = unattempted_measurement("decode", reason)
                record["mixed"] = unattempted_measurement("mixed", reason)
            if case.run_live_tuning:
                record["live_tuning"] = unattempted_measurement("live_tuning", reason)
        else:
            try:
                record["runtime_contract"] = runtime_contract(case, boot_label)
            except Exception as error:
                record["runtime_contract"] = {
                    "passed": False,
                    "failure_class": "runtime_contract_harness_exception",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                record["errors"].append(record["runtime_contract"])
            try:
                record["api"] = api_readback(case, sources)
            except Exception as error:
                record["api"] = {
                    "passed": False,
                    "failure_class": "api_probe_harness_exception",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                record["errors"].append(record["api"])
            if case.run_behavior and record["api"].get("observed_config") is not None:
                try:
                    record["structural_rejections"] = structural_rejections(case, sources)
                except Exception as error:
                    record["structural_rejections"] = {
                        "passed": False,
                        "failure_class": "structural_probe_harness_exception",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    record["errors"].append(record["structural_rejections"])
            elif case.run_behavior:
                record["structural_rejections"] = unattempted_measurement(
                    "structural_rejections", "live API schema/config was unavailable"
                )
            attribution_ready = bool(
                record.get("runtime_contract", {}).get("passed")
                and record.get("api", {}).get("passed")
            )
            if case.run_behavior and not attribution_ready:
                reason = "measurement attribution precondition failed: runtime or live policy mismatch"
                record["decode"] = unattempted_measurement("decode", reason)
                record["mixed"] = unattempted_measurement("mixed", reason)
                if case.run_live_tuning:
                    record["live_tuning"] = unattempted_measurement("live_tuning", reason)
            elif case.run_behavior:
                try:
                    record["decode"] = run_decode(case, sources)
                except Exception as error:
                    record["decode"] = {
                        "measurement_classification": "unsupported",
                        "failure_class": "decode_harness_exception",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    record["errors"].append(record["decode"])
                try:
                    record["mixed"] = run_workload(case, sources, live_tuning=False)
                except Exception as error:
                    record["mixed"] = {
                        "measurement_classification": "unsupported",
                        "execution_status": "incomplete",
                        "failure_class": "mixed_workload_harness_exception",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    record["errors"].append(record["mixed"])
                if case.run_live_tuning:
                    try:
                        record["live_tuning"] = run_workload(
                            case, sources, live_tuning=True
                        )
                    except Exception as error:
                        record["live_tuning"] = {
                            "measurement_classification": "unsupported",
                            "execution_status": "incomplete",
                            "failure_class": "live_tuning_harness_exception",
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                        }
                        record["errors"].append(record["live_tuning"])
    finally:
        try:
            rt.stop()
        except Exception as error:
            stop_error = {
                "operation": "stop_owned_runtime",
                "classification": "cleanup_harness_exception",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            record["errors"].append(stop_error)
            record["stop_error"] = stop_error
        record["finished_at_utc"] = utc_now()
        record["receipt_path"] = raw_json(f"{case.key}-case-receipt", record)
        rt.note(f"R27 SCHEDULER BOOT DONE case={case.key} status={record['boot_status']}")
    return record


def behavior_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("case", {}).get("key")): record
        for record in records
        if record.get("case", {}).get("key") in BEHAVIOR_KEYS
    }


def expected_mixed_identities() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (profile, concurrency, 1)
        for concurrency in MIXED_CONCURRENCIES
        for profile in MIXED_PROFILES
    )


def identity_of(value: dict[str, Any]) -> tuple[object, object, object]:
    return tuple(value.get(field) for field in MIXED_IDENTITY_FIELDS)


def evidence_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = rt.ROOT / path
    try:
        resolved = path.resolve()
        root = rt.ROOT.resolve()
    except OSError:
        return None
    return resolved if resolved.is_relative_to(root) else None


def candidate_runtime_failure_evidence(record: dict[str, Any]) -> dict[str, Any]:
    boot_label = str(record.get("boot_label") or "")
    log_candidates = (
        rt.ROOT / f"{boot_label}-final.docker.log",
        rt.ROOT / f"{boot_label}.docker.log",
    )
    log_path = next((path for path in log_candidates if path.is_file()), None)
    log_text = ""
    log_error: str | None = None
    if log_path is not None:
        try:
            log_text = log_path.read_text(errors="replace")
        except OSError as error:
            log_error = f"{type(error).__name__}: {error}"
    else:
        log_error = "no retained docker log"
    markers = {
        name: marker in log_text for name, marker in CANDIDATE_CRASH_MARKERS.items()
    }
    inspect_path = rt.ROOT / f"{boot_label}-final.inspect.json"
    inspect, inspect_error = read_json_object(inspect_path)
    state = inspect.get("State") if inspect and isinstance(inspect.get("State"), dict) else {}
    boot_inspect_path = rt.ROOT / f"{boot_label}.inspect.json"
    boot_inspect, boot_inspect_error = read_json_object(boot_inspect_path)
    expected_image = nested(record, "case", "launch", "image")
    expected_image_id = (
        "sha256:" + expected_image.rsplit("@sha256:", 1)[-1]
        if isinstance(expected_image, str) and "@sha256:" in expected_image
        else None
    )
    same_container = bool(
        boot_inspect
        and inspect
        and boot_inspect.get("Id")
        and boot_inspect.get("Id") == inspect.get("Id")
    )
    image_identity_matched = bool(
        inspect and expected_image_id and inspect.get("Image") == expected_image_id
    )
    proven = bool(
        markers
        and all(markers.values())
        and same_container
        and image_identity_matched
    )
    arm = nested(record, "case", "arm")
    source_path = (
        snapshot_path(str(arm), BOUNDARY_CHECKPOINT_PATH)
        if arm in r27.SOURCE_DIRS
        else None
    )
    patched_source_path = snapshot_path("patched", BOUNDARY_CHECKPOINT_PATH)
    source_text = (
        source_path.read_text(errors="replace")
        if source_path is not None and source_path.is_file()
        else ""
    )
    patched_source_text = (
        patched_source_path.read_text(errors="replace")
        if patched_source_path.is_file()
        else ""
    )
    source_failure_semantics_matched = bool(
        arm == "stock"
        and "destination = (base + slot.to(tl.int64) * stride).to" in source_text
        and (
            "destination = (base + tl.cast(slot, tl.int64) * stride).to"
            in patched_source_text
        )
    )
    proven = proven and source_failure_semantics_matched
    return {
        "classification": (
            "candidate_boundary_checkpoint_restore_triton_compile_failure"
            if proven
            else None
        ),
        "proven": proven,
        "marker_matches": markers,
        "root_cause": (
            "Boundary checkpoint auxiliary-state restore compiled slot as a Python int; "
            "Triton rejected slot.to(tl.int64), every worker failed, and EngineCore terminated."
            if proven
            else None
        ),
        "docker_log": str(log_path) if log_path else None,
        "docker_log_sha256": sha256_file(log_path) if log_path else None,
        "docker_log_error": log_error,
        "final_inspect": str(inspect_path) if inspect_path.is_file() else None,
        "final_inspect_sha256": sha256_file(inspect_path),
        "final_inspect_error": inspect_error,
        "boot_inspect": (
            str(boot_inspect_path) if boot_inspect_path.is_file() else None
        ),
        "boot_inspect_sha256": sha256_file(boot_inspect_path),
        "boot_inspect_error": boot_inspect_error,
        "same_container_from_boot_through_failure": same_container,
        "expected_image": expected_image,
        "expected_image_id": expected_image_id,
        "observed_final_image_id": inspect.get("Image") if inspect else None,
        "pinned_image_identity_matched": image_identity_matched,
        "candidate_source": str(source_path) if source_path else None,
        "candidate_source_sha256": sha256_file(source_path) if source_path else None,
        "patched_source": str(patched_source_path),
        "patched_source_sha256": sha256_file(patched_source_path),
        "source_failure_semantics_matched": source_failure_semantics_matched,
        "source_fix_basis": (
            "stock restore uses slot.to(tl.int64); pinned patched restore uses "
            "tl.cast(slot, tl.int64)"
            if proven
            else None
        ),
        "container_state": {
            "status": state.get("Status"),
            "running": state.get("Running"),
            "oom_killed": state.get("OOMKilled"),
            "exit_code": state.get("ExitCode"),
            "finished_at": state.get("FinishedAt"),
        },
        "candidate_failure_not_harness_exception": proven,
    }


def mixed_raw_evidence(mixed: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    output_path = evidence_path(mixed.get("output"))
    output: dict[str, Any] | None = None
    if output_path is None:
        errors.append("mixed output path is absent or outside BATTERY_ROOT")
    else:
        output, output_error = read_json_object(output_path)
        if output_error:
            errors.append(output_error)
        expected_hash = mixed.get("output_sha256")
        actual_hash = sha256_file(output_path)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            errors.append("mixed output SHA-256 does not match its preserved receipt")

    command_path = evidence_path(mixed.get("command_path"))
    command: dict[str, Any] | None = None
    if command_path is None:
        errors.append("mixed command path is absent or outside BATTERY_ROOT")
    else:
        command, command_error = read_json_object(command_path)
        if command_error:
            errors.append(command_error)
        expected_hash = mixed.get("command_sha256")
        actual_hash = sha256_file(command_path)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            errors.append("mixed command SHA-256 does not match its preserved receipt")

    workload_log_path = evidence_path(command.get("log")) if command else None
    workload_log = ""
    if workload_log_path is not None:
        try:
            workload_log = workload_log_path.read_text(errors="replace")
        except OSError as error:
            errors.append(f"mixed workload log unreadable: {type(error).__name__}: {error}")
    else:
        errors.append("mixed workload log is absent or outside BATTERY_ROOT")

    cells = (
        [cell for cell in output.get("cells", []) if isinstance(cell, dict)]
        if output
        else []
    )
    return {
        "integrity_passed": not errors,
        "integrity_errors": errors,
        "output": str(output_path) if output_path else None,
        "output_sha256": sha256_file(output_path) if output_path else None,
        "command": str(command_path) if command_path else None,
        "command_sha256": sha256_file(command_path) if command_path else None,
        "command_window": {
            "started_at": command.get("started_at") if command else None,
            "finished_at": command.get("finished_at") if command else None,
            "return_code": command.get("returncode") if command else None,
        },
        "workload_log": str(workload_log_path) if workload_log_path else None,
        "workload_log_sha256": (
            sha256_file(workload_log_path) if workload_log_path else None
        ),
        "workload_log_text": workload_log,
        "cells": cells,
    }


def parsed_utc_epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def mixed_cell_execution_accounting(record: dict[str, Any]) -> dict[str, Any]:
    mixed = record.get("mixed") if isinstance(record.get("mixed"), dict) else {}
    raw = mixed_raw_evidence(mixed)
    cells = raw.pop("cells")
    workload_log = str(raw.pop("workload_log_text"))
    by_identity: dict[tuple[object, object, object], dict[str, Any]] = {}
    duplicate_identities: list[list[object]] = []
    for cell in cells:
        identity = identity_of(cell)
        if identity in by_identity:
            duplicate_identities.append(list(identity))
        else:
            by_identity[identity] = cell
    if duplicate_identities:
        raw["integrity_passed"] = False
        raw["integrity_errors"].append(
            f"duplicate mixed identities: {duplicate_identities}"
        )

    runtime_failure = candidate_runtime_failure_evidence(record)
    ordered = expected_mixed_identities()

    def consumer_started(identity: tuple[str, int, int]) -> bool:
        trace_id = (
            f"{TRACE_SEED}:{identity[0]}:c{identity[1]}:r{identity[2]}"
        )
        return f"START {trace_id} " in workload_log

    runtime_finished_at = nested(
        runtime_failure, "container_state", "finished_at"
    )
    runtime_finished_epoch = parsed_utc_epoch(runtime_finished_at)
    crash_index: int | None = None
    if raw["integrity_passed"] and runtime_failure["proven"]:
        crash_index = next(
            (
                index
                for index, identity in enumerate(ordered)
                if (
                    not isinstance(
                        by_identity.get(identity, {}).get("trace"), dict
                    )
                    and consumer_started(identity)
                    and str(
                        by_identity.get(identity, {}).get("error") or ""
                    ).startswith("ConnectError: All connection attempts failed")
                )
            ),
            None,
        )

    qos_rows = nested(mixed, "policy_qos", "cells")
    qos_by_identity = {
        identity_of(row): row
        for row in qos_rows
        if isinstance(row, dict)
    } if isinstance(qos_rows, list) else {}
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(ordered):
        cell = by_identity.get(identity)
        status = cell.get("status") if cell else None
        trace = cell.get("trace") if cell else None
        trace_hash = trace.get("trace_hash") if isinstance(trace, dict) else None
        qos = qos_by_identity.get(identity, {})
        qos_classification = qos.get("measurement_classification")
        consumer_cell_entered = consumer_started(identity)
        boundary_observed_at = nested(
            cell, "api", "cell_boundary_apply", "before", "observed_at_utc"
        )
        boundary_observed_epoch = parsed_utc_epoch(boundary_observed_at)
        boundary_after_runtime_exit = bool(
            boundary_observed_epoch is not None
            and runtime_finished_epoch is not None
            and boundary_observed_epoch > runtime_finished_epoch
        )
        boundary_connection_errors = [
            nested(
                cell,
                "api",
                "cell_boundary_apply",
                exchange,
                "error",
            )
            for exchange in ("before", "post", "after")
        ]
        boundary_connection_failed = bool(boundary_connection_errors) and all(
            isinstance(error, str)
            and error.startswith("ConnectError: All connection attempts failed")
            for error in boundary_connection_errors
        )
        if not raw["integrity_passed"]:
            disposition = "preserved_receipt_integrity_failure"
            coverage_terminal = False
            direct_attempt = False
            reason = "; ".join(raw["integrity_errors"])
        elif isinstance(trace, dict):
            direct_attempt = True
            coverage_terminal = True
            if status == "complete" and qos_classification == "measured":
                disposition = "measured"
                reason = None
            else:
                disposition = "direct_workload_candidate_finding"
                reason = (
                    f"cell returned status={status!r}, "
                    f"failure_class={cell.get('failure_class')!r}"
                )
        elif crash_index is not None and index == crash_index:
            disposition = "terminal_candidate_runtime_crash"
            coverage_terminal = True
            direct_attempt = True
            reason = runtime_failure["root_cause"]
        elif (
            crash_index is not None
            and index > crash_index
            and boundary_after_runtime_exit
            and boundary_connection_failed
        ):
            disposition = "blocked_by_terminal_candidate_runtime_crash"
            coverage_terminal = True
            direct_attempt = False
            reason = (
                "The preserved consumer entered this cell only after the candidate "
                "EngineCore had terminated in the preceding cell; boundary API calls "
                "could not connect, so no workload or trace was measured."
            )
        else:
            disposition = "unresolved_incomplete_execution"
            coverage_terminal = False
            direct_attempt = False
            reason = (
                f"cell has no trace and no proven candidate-fatal cascade "
                f"(status={status!r}, failure_class="
                f"{cell.get('failure_class') if cell else None!r})"
            )
        rows.append(
            {
                "profile": identity[0],
                "concurrency": identity[1],
                "repeat": identity[2],
                "raw_cell_present": cell is not None,
                "raw_cell_status": status,
                "raw_failure_class": cell.get("failure_class") if cell else None,
                "raw_error": cell.get("error") if cell else None,
                "consumer_cell_entered": consumer_cell_entered,
                "cell_boundary_observed_at_utc": boundary_observed_at,
                "candidate_runtime_finished_at_utc": runtime_finished_at,
                "boundary_observed_after_candidate_exit": boundary_after_runtime_exit,
                "boundary_connection_errors": boundary_connection_errors,
                "all_boundary_calls_failed_to_connect": boundary_connection_failed,
                "direct_measurement_attempted": direct_attempt,
                "measurement_disposition_classification": (
                    "measured"
                    if disposition == "measured"
                    else (
                        "unattempted"
                        if disposition
                        == "blocked_by_terminal_candidate_runtime_crash"
                        else "unsupported"
                    )
                ),
                "trace_available": isinstance(trace_hash, str) and bool(trace_hash),
                "trace_hash": trace_hash,
                "policy_qos_classification": qos_classification,
                "comparison_eligible": (
                    isinstance(trace_hash, str)
                    and bool(trace_hash)
                    and qos_classification == "measured"
                ),
                "disposition": disposition,
                "coverage_terminal": coverage_terminal,
                "candidate_failure_is_not_measurement_success": disposition
                in {
                    "terminal_candidate_runtime_crash",
                    "blocked_by_terminal_candidate_runtime_crash",
                    "direct_workload_candidate_finding",
                },
                "reason": reason,
                "caused_by_identity": (
                    {
                        "profile": ordered[crash_index][0],
                        "concurrency": ordered[crash_index][1],
                        "repeat": ordered[crash_index][2],
                    }
                    if disposition == "blocked_by_terminal_candidate_runtime_crash"
                    and crash_index is not None
                    else None
                ),
            }
        )
    return {
        "case": nested(record, "case", "key"),
        "planned_cells": len(ordered),
        "raw_receipt_integrity": raw,
        "candidate_runtime_failure": runtime_failure,
        "failure_cascade_order_basis": (
            "consumer command order is concurrency outer/profile inner; the first "
            "trace-less cell is the direct fatal cell and later trace-less boundary "
            "failures are blocked only when the retained EngineCore fatal markers match"
        ),
        "cells": rows,
        "all_planned_cells_terminally_accounted": (
            len(rows) == len(ordered) and all(row["coverage_terminal"] for row in rows)
        ),
        "all_planned_cells_directly_measured": all(
            row["disposition"] == "measured" for row in rows
        ),
    }


def annotate_mixed_execution(records: list[dict[str, Any]]) -> None:
    for record in records:
        if not record.get("case", {}).get("run_behavior"):
            continue
        mixed = record.get("mixed")
        if not isinstance(mixed, dict):
            continue
        accounting = mixed_cell_execution_accounting(record)
        mixed["receipt_all_requested_cells_attempted"] = mixed.get(
            "all_requested_cells_attempted"
        )
        mixed["all_requested_cells_attempted"] = all(
            row["direct_measurement_attempted"] for row in accounting["cells"]
        )
        mixed["all_planned_cells_terminally_accounted"] = accounting[
            "all_planned_cells_terminally_accounted"
        ]
        mixed["cell_execution_accounting"] = accounting
        if accounting["all_planned_cells_terminally_accounted"]:
            if accounting["candidate_runtime_failure"]["proven"]:
                mixed["execution_status"] = "terminal_candidate_failure"
            else:
                mixed["execution_status"] = "complete"
        else:
            mixed["execution_status"] = "incomplete"


def mixed_coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {"case": record.get("case", {}).get("key"), **cell}
        for record in records
        for cell in nested(record, "mixed", "cell_execution_accounting", "cells") or []
        if isinstance(cell, dict)
    ]
    counts = Counter(str(row.get("disposition")) for row in rows)
    expected = len(BEHAVIOR_KEYS) * len(expected_mixed_identities())
    return {
        "expected_cells": expected,
        "accounted_cells": len(rows),
        "disposition_counts": dict(sorted(counts.items())),
        "all_planned_cells_terminally_accounted": (
            len(rows) == expected and all(row.get("coverage_terminal") is True for row in rows)
        ),
        "all_planned_cells_directly_measured": (
            len(rows) == expected and all(row.get("disposition") == "measured" for row in rows)
        ),
        "comparison_eligible_cells": sum(
            row.get("comparison_eligible") is True for row in rows
        ),
        "cells": rows,
    }


def source_execution_projection(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"contract_sha256", "harness_source_sha256"}
    }


def preserved_file_error(
    receipt: dict[str, Any],
    path_field: str,
    hash_field: str,
    label: str,
) -> str | None:
    path = evidence_path(receipt.get(path_field))
    expected = receipt.get(hash_field)
    if path is None or not isinstance(expected, str):
        return f"{label} path/hash provenance is absent or outside BATTERY_ROOT"
    actual = sha256_file(path)
    if actual != expected:
        return f"{label} SHA-256 changed: expected {expected}, observed {actual}"
    return None


def load_preserved_phase(
    current_plan: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str]:
    errors: list[str] = []
    plan_path = rt.ROOT / OUTPUT_DIR / "phase-plan.json"
    prior_plan, plan_error = read_json_object(plan_path)
    if plan_error or prior_plan is None:
        raise RuntimeError(f"preserved scheduler plan unavailable: {plan_error}")
    if prior_plan.get("schema") != PLAN_SCHEMA or prior_plan.get("valid") is not True:
        errors.append("preserved scheduler plan schema/validity does not match")
    plan_hash = prior_plan.get("plan_sha256")
    plan_without_hash = dict(prior_plan)
    plan_without_hash.pop("plan_sha256", None)
    plan_without_hash.pop("valid", None)
    if plan_hash != sha256_text(canonical_json(plan_without_hash)):
        errors.append("preserved scheduler plan self-hash does not match")

    prior_sources = prior_plan.get("source_contract", {})
    current_sources = current_plan.get("source_contract", {})
    if canonical_json(source_execution_projection(prior_sources)) != canonical_json(
        source_execution_projection(current_sources)
    ):
        errors.append("pinned image/source contract changed since the preserved run")

    prior_harness = prior_sources.get("harness_source_sha256", {})
    current_harness = current_sources.get("harness_source_sha256", {})
    current_script = Path(__file__).resolve()
    helper_mismatches: dict[str, dict[str, Any]] = {}
    for path, prior_hash in prior_harness.items():
        if Path(path).resolve() == current_script:
            continue
        current_hash = current_harness.get(path)
        if current_hash != prior_hash:
            helper_mismatches[path] = {
                "preserved": prior_hash,
                "current": current_hash,
            }
    if helper_mismatches:
        errors.append(f"preserved harness helper hashes changed: {helper_mismatches}")

    prior_cases = {
        row.get("key"): row
        for row in prior_plan.get("cases", [])
        if isinstance(row, dict)
    }
    current_cases = {
        row.get("key"): row
        for row in current_plan.get("cases", [])
        if isinstance(row, dict)
    }
    case_contract_fields = (
        "key",
        "arm",
        "candidate_label",
        "batch_tokens",
        "policy",
        "role",
        "run_behavior",
        "run_live_tuning",
        "as_shipped_auto",
        "policy_sha256",
        "launch",
        "source_expected_effective_lanes",
        "stock_auto_effective_lane_expectation",
        "planned_measurements",
    )
    for case in CASES:
        expected = {
            field: current_cases.get(case.key, {}).get(field)
            for field in case_contract_fields
        }
        actual = prior_cases.get(case.key)
        if actual is None or mismatches(actual, expected):
            errors.append(f"preserved plan case contract changed: {case.key}")

    records: list[dict[str, Any]] = []
    receipt_hashes: dict[str, str | None] = {}
    for case in CASES:
        case_path = rt.ROOT / OUTPUT_DIR / "raw" / f"{case.key}-case-receipt.json"
        record, record_error = read_json_object(case_path)
        if record_error or record is None:
            errors.append(f"{case.key} case receipt unavailable: {record_error}")
            continue
        record["receipt_path"] = str(case_path)
        receipt_hashes[case.key] = sha256_file(case_path)
        expected_case = {
            field: current_cases.get(case.key, {}).get(field)
            for field in case_contract_fields
        }
        if mismatches(record.get("case"), expected_case):
            errors.append(f"{case.key} receipt case/source/image contract changed")
        runtime_receipt = record.get("runtime_contract", {})
        for error in (
            preserved_file_error(
                runtime_receipt,
                "launch_path",
                "launch_sha256",
                f"{case.key} launch",
            ),
            preserved_file_error(
                runtime_receipt,
                "metrics_path",
                "metrics_sha256",
                f"{case.key} boot metrics",
            ),
        ):
            if error:
                errors.append(error)
        if case.run_behavior:
            decode = record.get("decode", {})
            mixed = record.get("mixed", {})
            checks = (
                (decode, "result_path", "result_sha256", "decode result"),
                (decode, "command_path", "command_sha256", "decode command"),
                (
                    decode,
                    "steady_samples_path",
                    "steady_samples_sha256",
                    "decode steady samples",
                ),
                (mixed, "output", "output_sha256", "mixed output"),
                (mixed, "command_path", "command_sha256", "mixed command"),
            )
            for receipt, path_field, hash_field, label in checks:
                error = preserved_file_error(
                    receipt,
                    path_field,
                    hash_field,
                    f"{case.key} {label}",
                )
                if error:
                    errors.append(error)
        records.append(record)

    summary_path = rt.ROOT / OUTPUT_DIR / "phase-summary.json"
    prior_summary, summary_error = read_json_object(summary_path)
    if summary_error or prior_summary is None:
        errors.append(f"preserved scheduler summary unavailable: {summary_error}")
        started_at = utc_now()
    else:
        started_at = str(prior_summary.get("started_at_utc") or utc_now())
    lineage = {
        "schema": REANALYSIS_SCHEMA,
        "preserved_plan": str(plan_path),
        "preserved_plan_file_sha256": sha256_file(plan_path),
        "preserved_plan_sha256": plan_hash,
        "preserved_summary": str(summary_path),
        "preserved_summary_sha256_before_reanalysis": sha256_file(summary_path),
        "preserved_case_receipt_sha256": receipt_hashes,
        "pinned_source_and_image_contract_unchanged": not any(
            "image/source contract changed" in error for error in errors
        ),
        "non_scheduler_harness_helpers_unchanged": not helper_mismatches,
        "scheduler_harness_sha256": {
            "measurement_version": prior_harness.get(str(current_script)),
            "reanalysis_version": sha256_file(current_script),
        },
        "raw_receipts_rewritten": False,
        "preserved_case_receipts_reused": True,
        "runtime_or_container_state_touched": False,
        "gpu_jobs_started": False,
    }
    if errors:
        raise RuntimeError(
            "preserved scheduler evidence failed reanalysis preflight: "
            + "; ".join(errors)
        )
    return prior_plan, records, lineage, started_at


def delta_value(candidate: object, baseline: object) -> dict[str, Any] | None:
    if not (
        isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
        and isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
    ):
        return None
    absolute = float(candidate) - float(baseline)
    relative = absolute / float(baseline) * 100.0 if float(baseline) != 0.0 else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_delta": absolute,
        "relative_delta_percent": relative,
    }


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    identity_fields: tuple[str, ...],
    value_field: str,
    metrics: tuple[str, ...],
    *,
    identity_comparable: dict[tuple[object, ...], bool] | None = None,
) -> list[dict[str, Any]]:
    baseline = {
        tuple(row.get(field) for field in identity_fields): row for row in baseline_rows
    }
    candidate = {
        tuple(row.get(field) for field in identity_fields): row for row in candidate_rows
    }
    identities = sorted(set(baseline) | set(candidate))
    result: list[dict[str, Any]] = []
    for identity in identities:
        left = baseline.get(identity)
        right = candidate.get(identity)
        trace_comparable = (
            True
            if identity_comparable is None
            else identity_comparable.get(identity) is True
        )
        supported = bool(
            left
            and right
            and trace_comparable
            and left.get("measurement_classification") == "measured"
            and right.get("measurement_classification") == "measured"
        )
        deltas = {
            metric: delta_value(
                nested(right, value_field, metric),
                nested(left, value_field, metric),
            )
            for metric in metrics
        }
        result.append(
            {
                "identity": dict(zip(identity_fields, identity)),
                "measurement_classification": "measured" if supported else "unsupported",
                "input_identity_comparable": (
                    trace_comparable if identity_comparable is not None else None
                ),
                "baseline": left,
                "candidate": right,
                "deltas": deltas if supported else None,
                "unsupported_reason": (
                    "trace identity unavailable or mismatched"
                    if identity_comparable is not None and not trace_comparable
                    else (
                        "one or both measurements are unsupported"
                        if not supported
                        else None
                    )
                ),
                "unsupported_observations_are_not_claims": not supported,
            }
        )
    return result


def trace_comparison_projection(trace: object) -> dict[str, Any] | None:
    if not isinstance(trace, dict):
        return None
    trace_hash = trace.get("trace_hash")
    sessions = trace.get("session_prompt_sha256")
    cold = trace.get("cold_prompt_sha256")
    namespace_excluded = trace.get("cache_namespace_excluded_from_trace_hash")
    if (
        not isinstance(trace_hash, str)
        or not isinstance(sessions, list)
        or not isinstance(cold, list)
        or namespace_excluded is not True
    ):
        return None
    return {
        "trace_hash": trace_hash,
        "session_prompt_sha256": sessions,
        "cold_prompt_sha256": cold,
        "cache_namespace_excluded_from_trace_hash": True,
    }


def mixed_pair_trace_eligibility(
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any],
) -> dict[tuple[object, ...], bool]:
    def indexed(record: dict[str, Any]) -> dict[tuple[object, ...], dict[str, Any]]:
        traces = nested(record, "mixed", "trace_identities")
        return {
            identity_of(trace): trace
            for trace in traces
            if isinstance(trace, dict)
        } if isinstance(traces, list) else {}

    baseline = indexed(baseline_record)
    candidate = indexed(candidate_record)
    return {
        identity: bool(
            trace_comparison_projection(baseline.get(identity))
            and trace_comparison_projection(baseline.get(identity))
            == trace_comparison_projection(candidate.get(identity))
        )
        for identity in expected_mixed_identities()
    }


def comparison_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = behavior_records(records)
    pairs = {
        "fixed_source_effect": ("stock-fixed-control", "patched-fixed-control"),
        "configured_auto_policy_effect": (
            "patched-fixed-control",
            "patched-auto-lanes-bt4096",
        ),
        "auto_image_default_delivery_effect": (
            "patched-auto-lanes-bt4096",
            "auto-image-as-shipped",
        ),
    }
    result: dict[str, Any] = {}
    for name, (baseline_key, candidate_key) in pairs.items():
        baseline_record = indexed.get(baseline_key, {})
        candidate_record = indexed.get(candidate_key, {})
        baseline_decode = baseline_record.get("decode", {}).get("cells", [])
        candidate_decode = candidate_record.get("decode", {}).get("cells", [])
        baseline_mixed = baseline_record.get("mixed", {}).get("policy_qos", {}).get(
            "cells", []
        )
        candidate_mixed = candidate_record.get("mixed", {}).get("policy_qos", {}).get(
            "cells", []
        )
        mixed_identity_eligibility = mixed_pair_trace_eligibility(
            baseline_record, candidate_record
        )
        result[name] = {
            "baseline_case": baseline_key,
            "candidate_case": candidate_key,
            "baseline_policy": baseline_record.get("case", {}).get("policy"),
            "candidate_policy": candidate_record.get("case", {}).get("policy"),
            "decode": compare_rows(
                baseline_decode,
                candidate_decode,
                ("concurrency", "context_tokens"),
                "steady_values",
                (
                    "output_tokens_per_second",
                    "aggregate_verifier_steps_per_second",
                    "acceptance_fraction",
                    "accepted_draft_tokens_per_step",
                    "emitted_tokens_per_verifier_step",
                ),
            ),
            "mixed_policy_qos": compare_rows(
                baseline_mixed,
                candidate_mixed,
                ("profile", "concurrency", "repeat"),
                "observed_values",
                (
                    "hot_ttft_p95_seconds",
                    "hot_output_gap_p95_seconds",
                    "cold_ttft_p95_seconds",
                    "completion_tokens_per_second",
                    "prefill_compute_fraction",
                    "decode_compute_fraction",
                ),
                identity_comparable=mixed_identity_eligibility,
            ),
            "completion_and_valid_stream_are_reported_outside_these_qos_deltas": True,
        }
    return result


def trace_identity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = behavior_records(records)
    rows: list[dict[str, Any]] = []
    for identity in sorted(expected_mixed_identities()):
        by_case: dict[str, Any] = {}
        dispositions: dict[str, Any] = {}
        for key in BEHAVIOR_KEYS:
            record = indexed.get(key, {})
            traces = nested(record, "mixed", "trace_identities")
            traces = traces if isinstance(traces, list) else []
            match = next(
                (
                    trace
                    for trace in traces
                    if isinstance(trace, dict) and identity_of(trace) == identity
                ),
                None,
            )
            by_case[key] = match
            accounting = nested(
                record, "mixed", "cell_execution_accounting", "cells"
            )
            accounting = accounting if isinstance(accounting, list) else []
            disposition = next(
                (
                    row
                    for row in accounting
                    if isinstance(row, dict) and identity_of(row) == identity
                ),
                None,
            )
            dispositions[key] = disposition
        projections_by_case = {
            key: trace_comparison_projection(value)
            for key, value in by_case.items()
        }
        available_projections = {
            key: value
            for key, value in projections_by_case.items()
            if value is not None
        }
        serialized_projections = {
            canonical_json(value) for value in available_projections.values()
        }
        all_available = len(available_projections) == len(BEHAVIOR_KEYS)
        observed_identities_consistent = bool(
            available_projections
        ) and len(serialized_projections) == 1
        mismatch = (
            len(available_projections) >= 2
            and len(serialized_projections) > 1
        )
        matched = all_available and observed_identities_consistent
        rows.append(
            {
                "profile": identity[0],
                "concurrency": identity[1],
                "repeat": identity[2],
                "by_case": by_case,
                "comparison_projection_by_case": projections_by_case,
                "unavailable_case_dispositions": {
                    key: dispositions[key]
                    for key in BEHAVIOR_KEYS
                    if projections_by_case[key] is None
                },
                "all_behavior_arm_traces_available": all_available,
                "observed_trace_hashes_consistent": observed_identities_consistent,
                "observed_prompt_and_trace_identities_consistent": (
                    observed_identities_consistent
                ),
                "trace_identity_mismatch": mismatch,
                "comparison_eligible": matched,
                "comparison_status": (
                    "matched"
                    if matched
                    else ("mismatched" if mismatch else "unavailable")
                ),
                "all_behavior_arms_same_trace": matched,
            }
        )
    mismatches_found = [
        {
            "profile": row["profile"],
            "concurrency": row["concurrency"],
            "repeat": row["repeat"],
            "by_case": row["by_case"],
        }
        for row in rows
        if row["trace_identity_mismatch"]
    ]
    unavailable = [
        {
            "profile": row["profile"],
            "concurrency": row["concurrency"],
            "repeat": row["repeat"],
            "case_dispositions": row["unavailable_case_dispositions"],
        }
        for row in rows
        if not row["all_behavior_arm_traces_available"]
    ]
    return {
        "trace_seed": TRACE_SEED,
        "rows": rows,
        "all_matched": bool(rows)
        and all(row["all_behavior_arms_same_trace"] for row in rows),
        "all_planned_traces_available": bool(rows)
        and all(row["all_behavior_arm_traces_available"] for row in rows),
        "all_observed_trace_hashes_consistent": not mismatches_found,
        "all_observed_prompt_and_trace_identities_consistent": not mismatches_found,
        "trace_identity_mismatches": mismatches_found,
        "unavailable_trace_identities": unavailable,
        "missing_traces_are_never_comparable": True,
        "actual_prompt_hashes_recorded": True,
    }


def lane_sweep_summary(records: list[dict[str, Any]], sources: dict[str, Any]) -> dict[str, Any]:
    indexed = {record.get("case", {}).get("key"): record for record in records}
    rows: list[dict[str, Any]] = []
    patched_values: list[int] = []
    for budget in BUDGETS:
        stock_key = f"stock-auto-lanes-bt{budget}"
        patched_key = f"patched-auto-lanes-bt{budget}"
        stock_api = indexed.get(stock_key, {}).get("api", {})
        patched_api = indexed.get(patched_key, {}).get("api", {})
        stock_config = stock_api.get("observed_config")
        patched_config = patched_api.get("observed_config")
        stock_effective = (
            stock_config.get("effective_max_parallel_prefills")
            if isinstance(stock_config, dict)
            else None
        )
        patched_effective = (
            patched_config.get("effective_max_parallel_prefills")
            if isinstance(patched_config, dict)
            else None
        )
        if isinstance(patched_effective, int) and not isinstance(patched_effective, bool):
            patched_values.append(patched_effective)
        rows.append(
            {
                "batch_tokens": budget,
                "stock": {
                    "configured_max_parallel_prefills": (
                        stock_config.get("max_parallel_prefills")
                        if isinstance(stock_config, dict)
                        else None
                    ),
                    "effective_max_parallel_prefills": stock_effective,
                    "effective_decode_refill_target": (
                        stock_config.get("effective_decode_refill_target")
                        if isinstance(stock_config, dict)
                        else None
                    ),
                    "readback_passed": stock_api.get("passed"),
                    "expectation": "observed control only",
                },
                "patched": {
                    "configured_max_parallel_prefills": (
                        patched_config.get("max_parallel_prefills")
                        if isinstance(patched_config, dict)
                        else None
                    ),
                    "effective_max_parallel_prefills": patched_effective,
                    "effective_decode_refill_target": (
                        patched_config.get("effective_decode_refill_target")
                        if isinstance(patched_config, dict)
                        else None
                    ),
                    "readback_passed": patched_api.get("passed"),
                    "source_expected_effective_lanes": sources.get("lane_resolver", {}).get(
                        "patched_source_expected_auto_lanes_at_max_num_seqs_32"
                    ),
                },
                "effective_lane_delta_patched_minus_stock": delta_value(
                    patched_effective, stock_effective
                ),
            }
        )
    expected = sources.get("lane_resolver", {}).get(
        "patched_source_expected_auto_lanes_at_max_num_seqs_32"
    )
    patched_complete = len(patched_values) == len(BUDGETS)
    patched_geometry_independent = (
        patched_complete
        and len(set(patched_values)) == 1
        and patched_values[0] == expected
    )
    return {
        "rows": rows,
        "patched_geometry_independent_auto_lanes_observed": patched_geometry_independent,
        "patched_observed_values": patched_values,
        "patched_source_expected_value": expected,
        "stock_values_are_not_evaluated_with_a_legacy_batch_or_page_formula": True,
        "configured_explicit_and_effective_lane_values_are_not_assumed_equal": True,
    }


def result_classifications(records: list[dict[str, Any]]) -> dict[str, int]:
    values: list[str] = []
    for record in records:
        if record.get("case", {}).get("run_behavior"):
            values.append(
                str(record.get("decode", {}).get("measurement_classification", "unattempted"))
            )
            mixed = record.get("mixed", {})
            values.append(
                str(
                    mixed.get("policy_qos", {}).get(
                        "measurement_classification",
                        mixed.get("measurement_classification", "unattempted"),
                    )
                )
            )
        if record.get("case", {}).get("run_live_tuning"):
            live = record.get("live_tuning", {})
            values.append(
                str(
                    live.get("policy_qos", {}).get(
                        "measurement_classification",
                        live.get("measurement_classification", "unattempted"),
                    )
                )
            )
    counts = Counter(values)
    for expected in ("measured", "unsupported", "unattempted"):
        counts.setdefault(expected, 0)
    return dict(sorted(counts.items()))


def classify_findings(
    records: list[dict[str, Any]],
    lane_summary: dict[str, Any],
    traces: dict[str, Any],
    mixed_coverage: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    harness: list[dict[str, Any]] = []
    product: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for record in records:
        key = record.get("case", {}).get("key")
        if record.get("boot_status") == "exception":
            harness.append(
                {
                    "case": key,
                    "finding": "boot exception",
                    "detail": record.get("errors"),
                }
            )
        elif record.get("boot_status") != "ready":
            product.append({"case": key, "finding": "candidate did not become ready"})
        if record.get("runtime_contract") and not record["runtime_contract"].get("passed"):
            product.append({"case": key, "finding": "runtime/page/KV launch mismatch"})
        api = record.get("api", {})
        if (
            api.get("measurement_classification") != "unattempted"
            and not api.get("passed")
        ):
            product.append(
                {"case": key, "finding": "source/live API or policy mismatch"}
            )
        structural = record.get("structural_rejections", {})
        if (
            structural
            and structural.get("measurement_classification") != "unattempted"
            and not structural.get("passed")
        ):
            product.append(
                {
                    "case": key,
                    "finding": "structural API mutation rejection defect",
                }
            )
        for kind in ("decode", "mixed", "live_tuning"):
            value = record.get(kind)
            if not isinstance(value, dict):
                continue
            accounting = (
                nested(value, "cell_execution_accounting", "cells")
                if kind == "mixed"
                else None
            )
            if isinstance(accounting, list):
                crash_rows = [
                    row
                    for row in accounting
                    if isinstance(row, dict)
                    and row.get("disposition")
                    == "terminal_candidate_runtime_crash"
                ]
                blocked_rows = [
                    row
                    for row in accounting
                    if isinstance(row, dict)
                    and row.get("disposition")
                    == "blocked_by_terminal_candidate_runtime_crash"
                ]
                for row in accounting:
                    if not isinstance(row, dict):
                        continue
                    identity = {
                        field: row.get(field) for field in MIXED_IDENTITY_FIELDS
                    }
                    disposition = row.get("disposition")
                    if disposition == "terminal_candidate_runtime_crash":
                        product.append(
                            {
                                "case": key,
                                "operation": kind,
                                "identity": identity,
                                "finding": (
                                    "candidate EngineCore terminated during the "
                                    "requested workload cell"
                                ),
                                "failure_class": nested(
                                    value,
                                    "cell_execution_accounting",
                                    "candidate_runtime_failure",
                                    "classification",
                                ),
                                "evidence": nested(
                                    value,
                                    "cell_execution_accounting",
                                    "candidate_runtime_failure",
                                ),
                                "blocked_later_identities": [
                                    {
                                        field: blocked.get(field)
                                        for field in MIXED_IDENTITY_FIELDS
                                    }
                                    for blocked in blocked_rows
                                ],
                            }
                        )
                        terminal.append(
                            {
                                "case": key,
                                "operation": kind,
                                "identity": identity,
                                "classification": "unsupported",
                                "disposition": disposition,
                                "reason": row.get("reason"),
                                "candidate_failure_is_not_measurement_success": True,
                                "comparison_eligible": False,
                            }
                        )
                    elif disposition == "blocked_by_terminal_candidate_runtime_crash":
                        terminal.append(
                            {
                                "case": key,
                                "operation": kind,
                                "identity": identity,
                                "classification": "unattempted",
                                "disposition": disposition,
                                "reason": row.get("reason"),
                                "caused_by_identity": row.get("caused_by_identity"),
                                "candidate_failure_is_not_measurement_success": True,
                                "comparison_eligible": False,
                            }
                        )
                    elif disposition == "direct_workload_candidate_finding":
                        product.append(
                            {
                                "case": key,
                                "operation": kind,
                                "identity": identity,
                                "finding": "direct workload qualification finding",
                                "reason": row.get("reason"),
                            }
                        )
                        terminal.append(
                            {
                                "case": key,
                                "operation": kind,
                                "identity": identity,
                                "classification": (
                                    row.get("policy_qos_classification")
                                    or "unsupported"
                                ),
                                "disposition": disposition,
                                "reason": row.get("reason"),
                                "comparison_eligible": row.get(
                                    "comparison_eligible"
                                ),
                            }
                        )
                    elif disposition not in {"measured"}:
                        evidence.append(
                            {
                                "case": key,
                                "operation": kind,
                                "identity": identity,
                                "classification": "unsupported",
                                "reason": row.get("reason"),
                                "disposition": disposition,
                            }
                        )
                if crash_rows:
                    value["terminal_candidate_failure"] = True
            else:
                classification = value.get("measurement_classification")
                if classification is None:
                    classification = nested(
                        value, "policy_qos", "measurement_classification"
                    )
                if classification in {"unsupported", "unattempted"}:
                    evidence.append(
                        {
                            "case": key,
                            "operation": kind,
                            "classification": classification,
                            "reason": value.get("reason") or value.get("failure_class"),
                        }
                    )
            if (
                value.get("execution_status") == "incomplete"
                or value.get("parse_error")
            ):
                if not (
                    isinstance(accounting, list)
                    and value.get("all_planned_cells_terminally_accounted") is True
                ):
                    harness.append(
                        {
                            "case": key,
                            "operation": kind,
                            "finding": "consumer output incomplete",
                            "parse_error": value.get("parse_error"),
                        }
                    )
            if value.get("qualification_status") == "findings":
                product.append(
                    {
                        "case": key,
                        "operation": kind,
                        "finding": "workload qualification finding",
                    }
                )
            if (
                kind == "live_tuning"
                and value.get("live_tuning")
                and not value["live_tuning"].get("passed")
            ):
                product.append(
                    {"case": key, "finding": "live compute tuning did not qualify"}
                )
        if record.get("stop_error"):
            harness.append(
                {
                    "case": key,
                    "finding": "owned runtime stop failed",
                    "detail": record["stop_error"],
                }
            )
    if not mixed_coverage.get("all_planned_cells_terminally_accounted"):
        harness.append(
            {
                "finding": "planned mixed workload coverage has unresolved cells",
                "disposition_counts": mixed_coverage.get("disposition_counts"),
                "accounted_cells": mixed_coverage.get("accounted_cells"),
                "expected_cells": mixed_coverage.get("expected_cells"),
            }
        )
    if not lane_summary.get("patched_geometry_independent_auto_lanes_observed"):
        product.append(
            {
                "finding": (
                    "patched geometry-independent lane sweep did not match source"
                )
            }
        )
    if traces.get("trace_identity_mismatches"):
        harness.append(
            {
                "finding": "mixed workload trace identities genuinely mismatched",
                "mismatches": traces["trace_identity_mismatches"],
            }
        )
    return {
        "harness_failures": harness,
        "candidate_findings": product,
        "evidence_gaps": evidence,
        "terminal_dispositions": terminal,
    }


def final_summary(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    annotate_mixed_execution(records)
    sources = plan["source_contract"]
    lane_summary = lane_sweep_summary(records, sources)
    mixed_coverage = mixed_coverage_summary(records)
    traces = trace_identity_summary(records)
    comparisons = comparison_summary(records)
    findings = classify_findings(
        records, lane_summary, traces, mixed_coverage
    )
    classifications = result_classifications(records)
    all_boots_attempted = len(records) == len(CASES) and all(
        record.get("boot_attempted") for record in records
    )
    execution_complete = bool(
        all_boots_attempted
        and mixed_coverage["all_planned_cells_terminally_accounted"]
        and not findings["harness_failures"]
    )
    evidence_complete = bool(
        execution_complete and not findings["evidence_gaps"]
    )
    supported_measurements_complete = (
        classifications.get("unsupported", 0) == 0
        and classifications.get("unattempted", 0) == 0
        and mixed_coverage["all_planned_cells_directly_measured"]
    )
    if not execution_complete:
        status = "incomplete"
    elif not evidence_complete:
        status = "complete_with_unsupported_or_unattempted_measurements"
    elif findings["candidate_findings"]:
        status = "complete_with_candidate_findings"
    else:
        status = "complete"
    exit_code = 0 if execution_complete and evidence_complete else 1
    summary = {
        "schema": PHASE_SCHEMA,
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "plan_path": str(rt.ROOT / OUTPUT_DIR / "phase-plan.json"),
        "plan_sha256": plan.get("plan_sha256"),
        "source_contract_sha256": sources.get("contract_sha256"),
        "source_contract_status": sources.get("status"),
        "expected_case_counts": plan.get("case_counts"),
        "observed": {
            "boot_case_records": len(records),
            "all_boot_cases_attempted": all_boots_attempted,
            "measurement_classifications": classifications,
            "mixed_workload_coverage": {
                key: value
                for key, value in mixed_coverage.items()
                if key != "cells"
            },
        },
        "actual_policy_and_control_configs": {
            record.get("case", {}).get("key"): {
                "role": record.get("case", {}).get("role"),
                "arm": record.get("case", {}).get("arm"),
                "image": nested(record, "case", "launch", "image"),
                "requested_policy": record.get("case", {}).get("policy"),
                "policy_sha256": record.get("case", {}).get("policy_sha256"),
                "launch_config_sha256": nested(record, "case", "launch", "config_sha256"),
                "observed_policy": nested(record, "api", "observed_config"),
                "api_receipt": nested(record, "api", "receipt_path"),
            }
            for record in records
        },
        "lane_sweep": lane_summary,
        "mixed_workload_execution_accounting": mixed_coverage,
        "trace_and_prompt_identity": traces,
        "comparisons_and_deltas": comparisons,
        "functional_results": {
            record.get("case", {}).get("key"): {
                "mixed": nested(record, "mixed", "functional"),
                "live_tuning": nested(record, "live_tuning", "functional"),
                "answers": nested(record, "mixed", "answer_classification"),
                "cell_execution_accounting": nested(
                    record, "mixed", "cell_execution_accounting"
                ),
            }
            for record in records
            if record.get("case", {}).get("run_behavior")
        },
        "live_tuning": nested(
            next(
                (
                    record
                    for record in records
                    if record.get("case", {}).get("run_live_tuning")
                ),
                {},
            ),
            "live_tuning",
            "live_tuning",
        ),
        "findings": findings,
        "interpretation": {
            **(plan.get("interpretation_contract") or {}),
            "terminal_candidate_failure_is_not_measurement_success": True,
            "cells_blocked_after_a_proven_terminal_candidate_failure_remain_unattempted": True,
            "missing_or_mismatched_traces_are_never_comparison_eligible": True,
            "execution_finality_means_every_planned_cell_has_a_terminal_disposition": True,
        },
        "unavailable_information": [
            (
                "The decode benchmark emits no rendered chat-template byte digest; the "
                "same-prompt proof is the pinned GENERATION_PROMPT hash, build_messages hash, "
                "benchmark hash, and identical context-zero command contract."
            ),
        ],
        "case_receipts": [record.get("receipt_path") for record in records],
        "execution_complete": execution_complete,
        "evidence_complete": evidence_complete,
        "candidate_qualification_passed": not findings["candidate_findings"],
        "supported_measurements_complete": supported_measurements_complete,
        "performance_comparisons_complete": bool(
            supported_measurements_complete and traces["all_matched"]
        ),
        "exit_code": exit_code,
        "exit_basis": (
            "nonzero only for unresolved harness execution or unresolved evidence gaps; "
            "a proven terminal candidate crash finalizes the arm as failed, leaves its "
            "blocked cells unattempted and comparison-ineligible, and is never a success claim"
        ),
    }
    summary["summary_sha256_without_self"] = sha256_text(canonical_json(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="print the source-backed bounded plan without runtime mutation",
    )
    mode.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help=(
            "rebuild only the derived phase summary from hash-validated preserved "
            "receipts; never boot a runtime or start GPU work"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = phase_plan()
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(
            f"expected_measurement_cases={plan['case_counts']['measurement_case_count']} "
            f"expected_boot_cases={plan['case_counts']['boot_cases']}"
        )
        return 0 if plan.get("valid") else 2

    ensure_runtime_scope()
    if args.reanalyze_existing:
        if not plan.get("valid"):
            result = {
                "schema": REANALYSIS_SCHEMA,
                "status": "current_source_contract_invalid",
                "errors": plan.get("validation_errors"),
                "runtime_or_container_state_touched": False,
                "gpu_jobs_started": False,
                "exit_code": 2,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        try:
            preserved_plan, records, lineage, started_at = load_preserved_phase(plan)
        except RuntimeError as error:
            result = {
                "schema": REANALYSIS_SCHEMA,
                "status": "preserved_evidence_preflight_failed",
                "error": str(error),
                "runtime_or_container_state_touched": False,
                "gpu_jobs_started": False,
                "phase_summary_rewritten": False,
                "exit_code": 2,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        summary = final_summary(preserved_plan, records, started_at)
        summary.pop("summary_sha256_without_self", None)
        summary["reanalysis"] = {
            **lineage,
            "finished_at_utc": utc_now(),
            "phase_summary_rewritten": True,
            "only_derived_summary_rewritten": True,
        }
        summary["summary_sha256_without_self"] = sha256_text(
            canonical_json(summary)
        )
        rt.save_json(f"{OUTPUT_DIR}/phase-summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return int(summary["exit_code"])

    ensure_fresh_outputs()
    started_at = utc_now()
    rt.save_json(f"{OUTPUT_DIR}/phase-plan.json", plan)
    if not plan.get("valid"):
        summary = {
            "schema": PHASE_SCHEMA,
            "status": "source_contract_invalid_no_gpu_work_attempted",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "plan_sha256": plan.get("plan_sha256"),
            "source_contract": plan.get("source_contract"),
            "exit_code": 2,
        }
        rt.save_json(f"{OUTPUT_DIR}/phase-summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    records: list[dict[str, Any]] = []
    for case in CASES:
        record = run_boot_case(case, plan["source_contract"])
        records.append(record)
        rt.save_json(
            f"{OUTPUT_DIR}/phase-progress.json",
            {
                "schema": PHASE_SCHEMA,
                "started_at_utc": started_at,
                "updated_at_utc": utc_now(),
                "planned_boot_cases": len(CASES),
                "completed_boot_case_records": len(records),
                "case_receipts": [item.get("receipt_path") for item in records],
            },
        )

    summary = final_summary(plan, records, started_at)
    rt.save_json(f"{OUTPUT_DIR}/phase-summary.json", summary)
    rt.record_gate(
        "r27-scheduler:phase-evidence-complete",
        bool(summary["execution_complete"] and summary["evidence_complete"]),
        {
            "status": summary["status"],
            "findings": summary["findings"],
            "summary": str(rt.ROOT / OUTPUT_DIR / "phase-summary.json"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
