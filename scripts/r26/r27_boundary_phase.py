#!/usr/bin/env python3
"""Run the focused R27 scalar, instruction-prefix, transcript, and DCP boundary phase."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROBE = Path(__file__).with_name("r27_boundary_probe.py")
PROBE_TIMEOUT_SECONDS = 14_400
SCALAR_TIMEOUT_SECONDS = 900
OWNERSHIP_LABEL = "field-lab.battery=r26"
FINE_GEOMETRY_ENV = {"GLM53_TARGET_BLOCK_SIZE": "256", "GLM53_MAMBA_BLOCK_SIZE": "256"}
FINE_GEOMETRY_SOURCE = {
    "mechanism": (
        "serve-glm53-flash.sh resolves GLM53_TARGET_BLOCK_SIZE and "
        "GLM53_MAMBA_BLOCK_SIZE environment variables and exports "
        "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE / VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE "
        "for the engine; no --mamba-block-size CLI flag exists"
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
        "labels; the probe reads them before every suite and drives the reuse oracle"
    ),
    "status": "source-supported separate diagnostic via the launcher environment contract; not a launcher default",
}


@dataclass(frozen=True)
class Cell:
    arm: str
    dcp: int
    spec: str
    kv: str
    batch: int
    geometry: str
    suites: tuple[str, ...]

    @property
    def cell_id(self) -> str:
        geom = "default" if self.geometry == "default" else "fine256"
        return f"{self.arm}-dcp{self.dcp}-{self.spec}-{self.kv}-b{self.batch}-{geom}"

    @property
    def launch_label(self) -> str:
        return "r27-boundary-" + self.cell_id.replace("fp8_ds_mla", "fp8").replace(
            "nvfp4_ds_mla", "nvfp4"
        )

    @property
    def extra_env(self) -> dict[str, str]:
        return dict(FINE_GEOMETRY_ENV) if self.geometry == "fine-pages-256" else {}

    def record(self, image: str) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "image": image,
            "tp": 4,
            "dcp": self.dcp,
            "spec": self.spec,
            "cache": "vram",
            "kv": self.kv,
            "batch": self.batch,
            "geometry": self.geometry,
            "suites": list(self.suites),
            "extra_env": dict(self.extra_env),
            "extra_args": [],
        }


def load_module(name: str) -> Any:
    try:
        return importlib.import_module("." + name, package=__package__)
    except (ImportError, TypeError):
        return importlib.import_module(name)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def arm_names(args: argparse.Namespace, r27: Any) -> tuple[str, ...]:
    return tuple(r27.IMAGES) if args.arm == "all" else (args.arm,)


def full_cells(arms: tuple[str, ...]) -> list[Cell]:
    merged: dict[tuple[object, ...], Cell] = {}

    def add(cell: Cell) -> None:
        key = (cell.arm, cell.dcp, cell.spec, cell.kv, cell.batch, cell.geometry)
        prior = merged.get(key)
        if prior is None:
            merged[key] = cell
            return
        suites = tuple(name for name in ("shared", "boundary") if name in set(prior.suites + cell.suites))
        merged[key] = replace(prior, suites=suites)

    for arm in arms:
        for spec in ("mtp0", "mtp3"):
            for batch in (4096, 8192):
                add(Cell(arm, 1, spec, "fp8_ds_mla", batch, "default", ("shared",)))
        for dcp in (1, 4):
            for spec in ("mtp3", "dflash2"):
                for kv in ("fp8_ds_mla", "nvfp4_ds_mla"):
                    add(Cell(arm, dcp, spec, kv, 4096, "default", ("boundary",)))
        add(Cell(arm, 4, "dflash2", "nvfp4_ds_mla", 4096, "fine-pages-256", ("boundary",)))
    return list(merged.values())


def selected_cells(args: argparse.Namespace, r27: Any) -> list[Cell]:
    cells = full_cells(arm_names(args, r27))
    if args.suite in {"all", "scalar"}:
        return cells if args.suite == "all" else []
    selected: list[Cell] = []
    for cell in cells:
        suites = tuple(name for name in cell.suites if name == args.suite)
        if suites:
            selected.append(replace(cell, suites=suites))
    return selected


def source_records(r27: Any, probe: Any, arms: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    manifest = r27.SOURCE_ROOT / "manifest.json"
    return {
        arm: probe.source_provenance(manifest, r27.SOURCE_DIRS[arm])
        for arm in arms
    }


def auxiliary_source_hashes(r27: Any, arm: str) -> dict[str, str]:
    """Hash the launcher and source.lock files shipped in the snapshot.

    These files are snapshot evidence for the geometry and checkpoint-policy
    contract even when they are not part of the manifest file list.
    """
    arm_root = r27.SOURCE_ROOT / r27.SOURCE_DIRS[arm]
    candidates = {
        "base_launcher": "usr/local/bin/serve-glm53-flash.sh",
        "auto_launcher": "usr/local/bin/serve-glm53-flash-auto.sh",
        "cache_launcher": "usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh",
        "source_lock": "opt/glm53-flash/source.lock",
        "image_inspect": "image-inspect.json",
    }
    result: dict[str, str] = {}
    for name, relative in candidates.items():
        path = arm_root / relative
        try:
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[name] = ""
    return result


def scalar_output(rt: Any, arm: str) -> Path:
    return rt.ROOT / f"r27-boundary-scalar-{arm}" / "receipt.json"


def cell_output(rt: Any, cell: Cell) -> Path:
    return rt.ROOT / f"r27-boundary-{cell.cell_id}.json"


def phase_plan(args: argparse.Namespace, rt: Any, r27: Any, probe: Any) -> dict[str, Any]:
    arms = arm_names(args, r27)
    sources = source_records(r27, probe, arms)
    cells = selected_cells(args, r27)
    scalar_selected = args.suite in {"all", "scalar"}
    cell_rows: list[dict[str, Any]] = []
    for cell in cells:
        row = cell.record(r27.IMAGES[cell.arm])
        case_ids = probe.planned_case_ids(cell.suites)
        row.update(
            {
                "launch_label": cell.launch_label,
                "output": str(cell_output(rt, cell)),
                "planned_case_count": len(case_ids),
                "planned_case_ids": case_ids,
                "geometry_contract": probe.geometry_plan(cell.dcp, cell.spec, cell.geometry),
                "auto_environment": dict(r27.AUTO_ENV) if cell.arm == "auto" else {},
                "policy_control": (
                    "as-shipped patched-auto policy restored deliberately"
                    if cell.arm == "auto"
                    else "matched static compute-share 0.4 runtime baseline"
                ),
            }
        )
        cell_rows.append(row)
    source_summary = {
        arm: {
            "source_arm": record["source_arm"],
            "image": record["image"],
            "manifest_sha256": record["manifest_sha256"],
            "all_manifest_files_verified": record["all_manifest_files_verified"],
            "boundary_checkpoint_sha256": record["relevant_files"].get(
                probe.BOUNDARY_SOURCE_PATH, {}
            ).get("actual_sha256"),
            "single_type_manager_sha256": record["relevant_files"].get(
                probe.MANAGER_SOURCE_PATH, {}
            ).get("actual_sha256"),
            "base_launcher_sha256": record["relevant_files"].get(
                probe.LAUNCHER_SOURCE_PATH, {}
            ).get("actual_sha256"),
            "cache_launcher_sha256": record["relevant_files"].get(
                probe.LMCACHE_LAUNCHER_SOURCE_PATH, {}
            ).get("actual_sha256"),
            "snapshot_auxiliary_hashes": auxiliary_source_hashes(r27, arm),
        }
        for arm, record in sources.items()
    }
    planned_http = sum(row["planned_case_count"] for row in cell_rows)
    return {
        "schema": "r27-boundary-phase-plan/v1",
        "plan_only_is_read_only": True,
        "phase_script": str(Path(__file__).resolve()),
        "phase_script_sha256": digest_bytes(Path(__file__).read_bytes()),
        "probe_script": str(PROBE.resolve()),
        "probe_script_sha256": digest_bytes(PROBE.read_bytes()),
        "battery_root": str(rt.ROOT),
        "required_container_name": "r27-test",
        "selected_arm_names": list(arms),
        "source_manifest": str(r27.SOURCE_ROOT / "manifest.json"),
        "source_arms": source_summary,
        "source_comparison_contract": {
            "stock_vs_patched": "same static scheduler policy; differing actual boundary/manager file hashes",
            "patched_vs_auto": "same patched source hashes; deliberately differing as-shipped auto policy",
            "source_lock_labels_are_not_patch_evidence": True,
        },
        "scalar": {
            "selected": scalar_selected,
            "arms": [
                {
                    "arm": arm,
                    "image": r27.IMAGES[arm],
                    "output": str(scalar_output(rt, arm)),
                    "planned_cases": 18,
                    "blocks": [0, 1, 2],
                    "slots": [0, 1, 2],
                    "sizes": [17, 1057],
                    "expected_disposition": (
                        "known-stock-specialization-bug" if arm == "stock" else "fixed-kernel-conformance"
                    ),
                }
                for arm in arms
            ],
            "kernel_fixture_contract": {
                "kernel": "vllm.v1.worker.gpu.boundary_checkpoint._restore_auxiliary_state_kernel",
                "packaged_not_reimplemented": True,
                "copy_semantics": (
                    "state 0 copies pool[block, 5:5+size] into destination[slot] "
                    "with masked partial chunks (size 17 single-chunk, 1057 "
                    "two-chunk); unselected slots and the source pool must be unchanged"
                ),
                "specialization_mechanism": (
                    "Triton specializes the Python int literal 1 into a kernel "
                    "constexpr; the stock kernel applies .to(tl.int64) to block and "
                    "slot, the patched kernel uses tl.cast"
                ),
                "affected_coordinates": "block == 1 or slot == 1 (10 of 18 cases)",
                "control_coordinates": "block and slot both in {0, 2} (8 of 18 cases)",
                "stock_defect_is_a_recorded_finding_not_overall_success": True,
            },
            "standalone_container_contract": {
                "name": "unique r27-scalar-* per attempt; never r27-test",
                "ownership_label": OWNERSHIP_LABEL,
                "gpu_devices": "device=0",
                "network": "none",
                "image": "the arm's explicit pinned digest",
                "cleanup": "remove only the actual ID returned by docker create",
            },
        },
        "http_cells": cell_rows,
        "geometry_modes": {
            "default": {
                "launcher_target_page_tokens": 2048,
                "recurrent_page": "auto resolves to target page",
                "override": None,
                "source_support": FINE_GEOMETRY_SOURCE,
            },
            "fine-pages-256": {
                "launcher_target_page_tokens": 256,
                "recurrent_page_tokens": 256,
                "environment": dict(FINE_GEOMETRY_ENV),
                "no_cli_flags": True,
                "source_support": FINE_GEOMETRY_SOURCE,
                "separate_diagnostic": True,
                "public_20_1_percent_prefill_penalty_is_not_a_local_measurement": True,
            },
        },
        "oracle_contract": {
            "realized_geometry_is_authoritative": (
                "the probe reads vllm:cache_config_info before every suite; the "
                "reuse oracle floors to the realized prefix_match_unit (or block_size "
                "when unset) and realized mamba_block_size, never a hard-coded 256 grid"
            ),
            "correctness_ceiling": (
                "cached tokens must never exceed the reuse the exact rendered "
                "common prefix justifies: the full prefix for request-boundary "
                "checkpoints, the hash-aligned floor for aligned retention"
            ),
            "reduced_hits_are_quality_observations": (
                "one hash-unit EAGLE rewind below the recurrent floor is tolerated "
                "for aligned cells and for the attention tail of exact endpoint "
                "checkpoints; divergent-prefix negatives accept any reuse up to the "
                "exact rendered common prefix and reject anything beyond it"
            ),
            "budget_96_truncation": (
                "budget-limited finals are classified separately per case and "
                "counted as wrong answers; the probe never retries with a larger budget"
            ),
        },
        "case_counts": {
            "scalar_cases": 18 * len(arms) if scalar_selected else 0,
            "model_boots": len(cell_rows),
            "http_cases": planned_http,
            "shared_cases_per_cell": probe.planned_request_count(("shared",)),
            "boundary_cases_per_cell": probe.planned_request_count(("boundary",)),
            "combined_shared_boundary_cases": probe.planned_request_count(("shared", "boundary")),
            "total_cases": planned_http + (18 * len(arms) if scalar_selected else 0),
        },
        "batch_question_method": {
            "budgets": [4096, 8192],
            "matched_instruction_prefixes": list(probe.SHARED_PREFIX_TARGETS),
            "comparison": "same tokenizer-rendered messages and token IDs; cache salts remain isolated",
            "outcomes_after_execution": ["no_observed_dependency", "observed_difference", "unsupported"],
            "universal_95_percent_hit_rate_claim": False,
        },
        "claim_policy": {
            "plan_is_not_execution_evidence": True,
            "plan_only_touches_no_gpu_container_service_or_output_file": True,
            "stock_known_failures_are_findings_not_harness_failures": True,
            "empty_reasoning-only_final_is_not_a_correct_visible answer": True,
            "budget_limited_finals_are_incomplete_not_wrong_and_never_retried": True,
            "foreign_gpu_windows_are_not_speed_evidence": True,
            "raw_receipts_are_not_overwritten": True,
            "unsupported_cache_stats_are_telemetry_only": True,
        },
    }


def save_command(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def captured_command(args: list[str], timeout: int) -> dict[str, Any]:
    started = time.time()
    try:
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "args": args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "started_at": started,
            "finished_at": time.time(),
            "error": None,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "args": args,
            "returncode": 124,
            "stdout": error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or ""),
            "stderr": error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or ""),
            "started_at": started,
            "finished_at": time.time(),
            "error": f"TimeoutExpired after {timeout}s",
        }
    except OSError as error:
        return {
            "args": args,
            "returncode": 127,
            "stdout": "",
            "stderr": "",
            "started_at": started,
            "finished_at": time.time(),
            "error": f"{type(error).__name__}: {error}",
        }


def reusable_scalar_result(output: Path, arm: str, image: str) -> dict[str, Any] | None:
    if not output.exists():
        return None
    try:
        receipt = read_object(output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "arm": arm,
            "output": str(output),
            "reused_existing_immutable_receipt": True,
            "execution_complete": False,
            "expectation_passed": False,
            "release_passed": False,
            "issues": [f"existing receipt unreadable: {type(error).__name__}: {error}"],
        }
    summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
    issues = []
    if receipt.get("arm") != arm:
        issues.append("receipt arm mismatch")
    if receipt.get("image") != image:
        issues.append("receipt image mismatch")
    if receipt.get("probe_provenance", {}).get("sha256") != digest_bytes(PROBE.read_bytes()):
        issues.append("receipt probe version mismatch; preserve it and use a fresh result root")
    if summary.get("source_verified") is not True:
        issues.append("receipt packaged-source identity was not verified")
    return {
        "arm": arm,
        "output": str(output),
        "reused_existing_immutable_receipt": True,
        "execution_complete": summary.get("matrix_execution_complete") is True and not issues,
        "expectation_passed": summary.get("qualification_expectation_passed") is True and not issues,
        "release_passed": summary.get("release_kernel_conformance_passed") is True and not issues,
        "known_stock_reproduction_succeeded": summary.get("known_stock_reproduction_succeeded"),
        "cleanup_by_created_id": None,
        "issues": issues,
        "summary": summary,
    }


def run_scalar_arm(rt: Any, r27: Any, probe: Any, arm: str, source: dict[str, Any]) -> dict[str, Any]:
    image = r27.IMAGES[arm]
    output = scalar_output(rt, arm)
    existing = reusable_scalar_result(output, arm, image)
    if existing is not None:
        rt.record_gate(
            f"r27-boundary-scalar:{arm}",
            existing["execution_complete"] and existing["expectation_passed"] and not existing["issues"],
            existing,
        )
        return existing

    stage = output.parent
    stage.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.time_ns()}"
    container_name = f"r27-scalar-{arm}-{nonce}"
    config_name = f"container-config-{nonce}.json"
    config_path = stage / config_name
    relevant = source["relevant_files"].get(probe.BOUNDARY_SOURCE_PATH)
    if not isinstance(relevant, dict) or not relevant.get("actual_sha256"):
        result = {
            "arm": arm,
            "output": str(output),
            "execution_complete": False,
            "expectation_passed": False,
            "release_passed": False,
            "issues": ["boundary checkpoint source hash is unavailable"],
        }
        rt.record_gate(f"r27-boundary-scalar:{arm}", False, result)
        return result
    disposition = "known-stock-specialization-bug" if arm == "stock" else "fixed-kernel-conformance"
    create_args = [
        "docker",
        "create",
        "--name",
        container_name,
        "--label",
        OWNERSHIP_LABEL,
        "--init",
        "--gpus",
        "device=0",
        "--network",
        "none",
        "--shm-size",
        "1g",
        "--entrypoint",
        "python",
        "-v",
        f"{PROBE.resolve()}:/r27-probe.py:ro",
        "-v",
        f"{stage.resolve()}:/r27-output",
        image,
        "/r27-probe.py",
        "scalar",
        "--arm",
        arm,
        "--image",
        image,
        "--config",
        f"/r27-output/{config_name}",
        "--output",
        "/r27-output/receipt.json",
        "--manifest-sha256",
        source["manifest_sha256"],
        "--source-path",
        probe.BOUNDARY_SOURCE_PATH,
        "--expected-source-sha256",
        relevant["actual_sha256"],
        "--expected-disposition",
        disposition,
    ]
    rt.note(f"R27 SCALAR ARM START {arm}")
    created_id: str | None = None
    issues: list[str] = []
    cleanup_by_id = False
    start_receipt: dict[str, Any] | None = None
    # docker create blocks until any missing pinned image pull completes;
    # keep a generous bound so a first-use pull is not misread as a harness error.
    create_receipt = captured_command(create_args, 600)
    save_command(stage / f"docker-create-{nonce}.json", create_receipt)
    if create_receipt["returncode"] == 0:
        candidate = str(create_receipt["stdout"]).strip().splitlines()[-1]
        if re.fullmatch(r"[0-9a-f]{12,64}", candidate):
            created_id = candidate
        else:
            issues.append("docker create did not return a valid container ID")
    else:
        issues.append(f"docker create failed with exit {create_receipt['returncode']}")
    try:
        if created_id is not None:
            container_config = {
                "schema": "r27-scalar-container/v1",
                "arm": arm,
                "image": image,
                "container_name": container_name,
                "container_id": created_id,
                "ownership_label": OWNERSHIP_LABEL,
                "gpu_devices": "device=0",
                "network": "none",
                "create_args": create_args,
                "source_manifest_sha256": source["manifest_sha256"],
                "boundary_checkpoint_expected_sha256": relevant["actual_sha256"],
                "created_at": time.time(),
            }
            save_command(config_path, container_config)
            inspect_receipt = captured_command(["docker", "inspect", created_id], 30)
            save_command(stage / f"docker-inspect-{nonce}.json", inspect_receipt)
            start_receipt = captured_command(["docker", "start", "-a", created_id], SCALAR_TIMEOUT_SECONDS)
            save_command(stage / f"docker-start-{nonce}.json", start_receipt)
            if start_receipt["returncode"] != 0:
                issues.append(f"scalar container exited {start_receipt['returncode']}")
    except Exception as error:
        issues.append(f"{type(error).__name__}: {error}")
        issues.append(traceback.format_exc())
    finally:
        if created_id is not None:
            cleanup_receipt = captured_command(["docker", "rm", "-f", created_id], 60)
            save_command(stage / f"docker-remove-{nonce}.json", cleanup_receipt)
            cleanup_by_id = cleanup_receipt["returncode"] == 0
            if not cleanup_by_id:
                issues.append(f"failed to remove created container ID {created_id}")

    receipt: dict[str, Any] | None = None
    if output.exists():
        try:
            receipt = read_object(output)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"scalar receipt unreadable: {type(error).__name__}: {error}")
    else:
        issues.append("scalar probe produced no receipt")
    summary = receipt.get("summary") if isinstance(receipt, dict) and isinstance(receipt.get("summary"), dict) else {}
    execution_complete = summary.get("matrix_execution_complete") is True
    expectation_passed = summary.get("qualification_expectation_passed") is True
    release_passed = summary.get("release_kernel_conformance_passed") is True
    result = {
        "arm": arm,
        "image": image,
        "output": str(output),
        "created_container_id": created_id,
        "container_name": container_name,
        "reused_existing_immutable_receipt": False,
        "execution_complete": execution_complete,
        "expectation_passed": expectation_passed,
        "release_passed": release_passed,
        "known_stock_reproduction_succeeded": summary.get("known_stock_reproduction_succeeded"),
        "cleanup_by_created_id": cleanup_by_id,
        "start_returncode": start_receipt.get("returncode") if start_receipt else None,
        "issues": issues,
        "summary": summary,
    }
    rt.record_gate(
        f"r27-boundary-scalar:{arm}",
        execution_complete and cleanup_by_id and not issues,
        result,
    )
    rt.note(f"R27 SCALAR ARM COMPLETE {arm}")
    return result


def reusable_http_result(output: Path, cell: Cell, image: str) -> dict[str, Any] | None:
    if not output.exists():
        return None
    try:
        receipt = read_object(output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "cell_id": cell.cell_id,
            "output": str(output),
            "reused_existing_immutable_receipt": True,
            "execution_complete": False,
            "release_conformant": False,
            "visible_answers_conformant": False,
            "negative_prefix_safe": False,
            "issues": [f"existing receipt unreadable: {type(error).__name__}: {error}"],
        }
    summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
    issues: list[str] = []
    if receipt.get("cell_id") != cell.cell_id:
        issues.append("receipt cell mismatch")
    if receipt.get("image") != image:
        issues.append("receipt image mismatch")
    if receipt.get("probe_provenance", {}).get("sha256") != digest_bytes(PROBE.read_bytes()):
        issues.append("receipt probe version mismatch; preserve it and use a fresh result root")
    config_provenance = receipt.get("config_provenance") or {}
    config_path = Path(str(config_provenance.get("path", "")))
    if (config_path.resolve().parent != output.resolve().parent
            or not config_path.is_file()
            or digest_bytes(config_path.read_bytes()) != config_provenance.get("sha256")):
        issues.append("receipt launch identity is missing or changed")
    return {
        "cell_id": cell.cell_id,
        "arm": cell.arm,
        "output": str(output),
        "reused_existing_immutable_receipt": True,
        "execution_complete": summary.get("execution_integrity_passed") is True and not issues,
        "release_conformant": summary.get("release_cache_conformance_passed") is True and not issues,
        "visible_answers_conformant": summary.get("visible_marker_conformance_passed") is True and not issues,
        "negative_prefix_safe": summary.get("negative_prefix_safety_passed") is True and not issues,
        "measured_cases": summary.get("measured_cases"),
        "unattempted_cases": summary.get("unattempted_cases"),
        "unsupported_cache_count_cases": summary.get("unsupported_cache_count_cases", []),
        "issues": issues,
        "summary": summary,
    }


def probe_command(
    rt: Any, r27: Any, cell: Cell, output: Path, config_path: Path, request_timeout: int
) -> list[str]:
    return [
        sys.executable,
        str(PROBE),
        "http",
        "--arm",
        cell.arm,
        "--source-arm",
        r27.SOURCE_DIRS[cell.arm],
        "--image",
        r27.IMAGES[cell.arm],
        "--manifest",
        str(r27.SOURCE_ROOT / "manifest.json"),
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--cell-id",
        cell.cell_id,
        "--base-url",
        rt.BASE_URL,
        "--model",
        rt.MODEL_NAME,
        "--suites",
        ",".join(cell.suites),
        "--dcp",
        str(cell.dcp),
        "--spec",
        cell.spec,
        "--kv",
        cell.kv,
        "--batch",
        str(cell.batch),
        "--geometry",
        cell.geometry,
        "--request-timeout",
        str(request_timeout),
    ]


def run_http_cell(
    rt: Any,
    r27: Any,
    probe: Any,
    cell: Cell,
    source: dict[str, Any],
    request_timeout: int,
) -> dict[str, Any]:
    image = r27.IMAGES[cell.arm]
    output = cell_output(rt, cell)
    existing = reusable_http_result(output, cell, image)
    if existing is not None:
        rt.record_gate(
            f"r27-boundary-http:{cell.cell_id}",
            existing["execution_complete"]
            and existing["visible_answers_conformant"]
            and existing["negative_prefix_safe"]
            and not existing["issues"],
            existing,
        )
        return existing

    rt.note(f"R27 BOUNDARY CELL START {cell.cell_id}")
    booted = False
    stopped = True
    returncode: int | None = None
    issues: list[str] = []
    receipt: dict[str, Any] | None = None
    config_path = rt.ROOT / f"{cell.launch_label}.launch.json"
    launch: dict[str, Any] | None = None
    try:
        boot_env: dict[str, str] = {}
        if cell.arm == "auto":
            boot_env.update(r27.AUTO_ENV)
        boot_env.update(cell.extra_env)
        booted = bool(
            r27.boot(
                cell.launch_label,
                arm=cell.arm,
                spec=cell.spec,
                dcp=cell.dcp,
                cache="vram",
                kv=cell.kv,
                batch=cell.batch,
                extra_env=boot_env or None,
                extra_args=None,
            )
        )
        if config_path.exists():
            launch = read_object(config_path)
        if not booted:
            issues.append("r27_config.boot returned false")
        else:
            returncode = rt.run(
                probe_command(rt, r27, cell, output, config_path, request_timeout),
                label=cell.launch_label + ".probe",
                timeout=PROBE_TIMEOUT_SECONDS,
                env=rt.PROXY_ENV,
            )
            if output.exists():
                receipt = read_object(output)
            else:
                issues.append("HTTP probe produced no receipt")
    except Exception as error:
        issues.append(f"{type(error).__name__}: {error}")
        issues.append(traceback.format_exc())
    finally:
        try:
            rt.stop()
        except Exception as error:
            stopped = False
            issues.append(f"runtime.stop failed: {type(error).__name__}: {error}")

    if receipt is None and output.exists():
        try:
            receipt = read_object(output)
            issues.append("probe did not return normally; preserved its partial receipt")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"partial receipt unreadable: {type(error).__name__}: {error}")
    if receipt is None:
        receipt = probe.write_unattempted_http_receipt(
            output,
            cell=cell.record(image),
            suites=cell.suites,
            reason="model HTTP cell could not be measured",
            detail={
                "booted": booted,
                "stopped": stopped,
                "probe_returncode": returncode,
                "issues": issues,
            },
            source=source,
            config=launch,
        )
    summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
    execution_complete = summary.get("execution_integrity_passed") is True
    release_conformant = summary.get("release_cache_conformance_passed") is True
    visible_conformant = summary.get("visible_marker_conformance_passed") is True
    negative_safe = summary.get("negative_prefix_safety_passed") is True
    result = {
        "cell_id": cell.cell_id,
        "arm": cell.arm,
        "image": image,
        "suites": list(cell.suites),
        "output": str(output),
        "booted": booted,
        "stopped": stopped,
        "probe_returncode": returncode,
        "reused_existing_immutable_receipt": False,
        "execution_complete": execution_complete,
        "release_conformant": release_conformant,
        "visible_answers_conformant": visible_conformant,
        "negative_prefix_safe": negative_safe,
        "measured_cases": summary.get("measured_cases"),
        "unattempted_cases": summary.get("unattempted_cases"),
        "unsupported_cache_count_cases": summary.get("unsupported_cache_count_cases", []),
        "issues": issues,
        "summary": summary,
    }
    rt.record_gate(
        f"r27-boundary-http:{cell.cell_id}",
        booted and stopped and execution_complete and not issues,
        result,
    )
    rt.note(f"R27 BOUNDARY CELL COMPLETE {cell.cell_id}")
    return result


def case_map(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in receipt.get("cases") or []
        if isinstance(row, dict) and row.get("case_id")
    }


def batch_comparisons(rt: Any, cells: list[Cell]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], dict[int, Cell]] = {}
    for cell in cells:
        if "shared" in cell.suites:
            groups.setdefault((cell.arm, cell.spec), {})[cell.batch] = cell
    for (arm, spec), by_batch in groups.items():
        row: dict[str, Any] = {
            "arm": arm,
            "spec": spec,
            "batches": [4096, 8192],
            "status": "unsupported",
            "matched_cases": [],
            "prompt_identity_preserved": False,
        }
        if set(by_batch) != {4096, 8192}:
            row["reason"] = "one matched batch cell is absent"
            results.append(row)
            continue
        try:
            left = case_map(read_object(cell_output(rt, by_batch[4096])))
            right = case_map(read_object(cell_output(rt, by_batch[8192])))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            row["reason"] = f"receipt unavailable: {type(error).__name__}: {error}"
            results.append(row)
            continue
        comparable = sorted(
            case_id
            for case_id in set(left) & set(right)
            if case_id.startswith("shared-") and not case_id.endswith("tool-append")
        )
        complete = True
        identity = True
        differences = False
        for case_id in comparable:
            a, b = left[case_id], right[case_id]
            same_prompt = a.get("prompt_token_ids_sha256") == b.get("prompt_token_ids_sha256")
            cached_a = a.get("cache_observation", {}).get("response_cached_tokens")
            cached_b = b.get("cache_observation", {}).get("response_cached_tokens")
            supported = isinstance(cached_a, int) and isinstance(cached_b, int)
            row["matched_cases"].append(
                {
                    "case_id": case_id,
                    "same_exact_token_ids": same_prompt,
                    "batch4096_cached_tokens": cached_a,
                    "batch8192_cached_tokens": cached_b,
                    "difference_tokens": cached_b - cached_a if supported else None,
                }
            )
            identity = identity and same_prompt
            complete = complete and supported
            differences = differences or (supported and cached_a != cached_b)
        row["prompt_identity_preserved"] = bool(comparable) and identity
        if comparable and identity and complete:
            row["status"] = "observed_difference" if differences else "no_observed_dependency"
            row["reason"] = "matched exact tokenizer-rendered prompts compared request by request"
        else:
            row["reason"] = "matched token IDs or per-request cached-token evidence incomplete"
        results.append(row)
    return results


def source_and_policy_comparisons(rt: Any, r27: Any, cells: list[Cell]) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], dict[str, Cell]] = {}
    for cell in cells:
        key = (cell.dcp, cell.spec, cell.kv, cell.batch, cell.geometry, cell.suites)
        grouped.setdefault(key, {})[cell.arm] = cell
    rows: list[dict[str, Any]] = []
    for key, arms in grouped.items():
        if not {"stock", "patched"} <= set(arms):
            continue
        row: dict[str, Any] = {
            "configuration": {
                "dcp": key[0],
                "spec": key[1],
                "kv": key[2],
                "batch": key[3],
                "geometry": key[4],
                "suites": list(key[5]),
            },
            "stock_vs_patched": {
                "comparison_type": "source effect under the same static scheduler policy",
                "receipts": {
                    arm: str(cell_output(rt, arms[arm])) for arm in ("stock", "patched")
                },
            },
        }
        if "auto" in arms:
            row["patched_vs_auto"] = {
                "comparison_type": "as-shipped auto-policy effect on identical patched source",
                "receipts": {
                    arm: str(cell_output(rt, arms[arm])) for arm in ("patched", "auto")
                },
                "auto_environment": dict(r27.AUTO_ENV),
            }
        rows.append(row)
    return rows


def final_summary(
    rt: Any,
    r27: Any,
    plan: dict[str, Any],
    cells: list[Cell],
    scalar_results: list[dict[str, Any]],
    http_results: list[dict[str, Any]],
) -> dict[str, Any]:
    source_verified = all(
        item.get("all_manifest_files_verified") is True for item in plan["source_arms"].values()
    )
    scalar_complete = all(row.get("execution_complete") is True for row in scalar_results)
    http_complete = all(row.get("execution_complete") is True for row in http_results)
    cleanup_complete = all(
        row.get("reused_existing_immutable_receipt") is True or row.get("cleanup_by_created_id") is True
        for row in scalar_results
    )
    fixed_scalar = all(
        row.get("release_passed") is True for row in scalar_results if row.get("arm") in {"patched", "auto"}
    )
    fixed_http = all(
        row.get("release_conformant") is True
        and row.get("visible_answers_conformant") is True
        and row.get("negative_prefix_safe") is True
        for row in http_results
        if row.get("arm") in {"patched", "auto"}
    )
    stock = {
        "scalar": [row for row in scalar_results if row.get("arm") == "stock"],
        "http": [row for row in http_results if row.get("arm") == "stock"],
        "known_failures_are_findings_not_harness_failures": True,
    }
    measured = sum(int(row.get("measured_cases") or 0) for row in http_results)
    unattempted_count = sum(int(row.get("unattempted_cases") or 0) for row in http_results)
    unsupported = sum(len(row.get("unsupported_cache_count_cases") or []) for row in http_results)
    execution_complete = source_verified and scalar_complete and http_complete and cleanup_complete
    passed = execution_complete and fixed_scalar and fixed_http
    return {
        "schema": "r27-boundary-phase-summary/v1",
        "phase": "r27_boundary",
        "plan": str(rt.ROOT / "r27-boundary-plan.json"),
        "source_manifest": str(r27.SOURCE_ROOT / "manifest.json"),
        "source_verified": source_verified,
        "scalar_results": scalar_results,
        "http_results": http_results,
        "case_accounting": {
            "planned_scalar": plan["case_counts"]["scalar_cases"],
            "planned_http": plan["case_counts"]["http_cases"],
            "measured_http": measured,
            "unattempted_http": unattempted_count,
            "unsupported_cached_count_http": unsupported,
        },
        "batch_budget_answer": batch_comparisons(rt, cells),
        "source_and_policy_comparisons": source_and_policy_comparisons(rt, r27, cells),
        "stock_release_observations": stock,
        "gates": {
            "execution_complete": execution_complete,
            "scalar_containers_removed_by_created_id": cleanup_complete,
            "fixed_images_scalar_conformant": fixed_scalar,
            "fixed_images_http_conformant": fixed_http,
            "stock_product_failures_gate_harness": False,
        },
        "passed": passed,
        "claim_limits": {
            "no_speed_claim": True,
            "no_production_95_percent_hit_rate_claim": True,
            "fine_geometry_public_prefill_cost_not_remeasured_here": True,
            "reasoning_only_empty_final_not_scored_correct": True,
            "unattempted_and_unsupported_are_not_measured": True,
        },
        "finished_at": time.time(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--arm", choices=("all", "stock", "patched", "auto"), default="all")
    parser.add_argument(
        "--suite", choices=("all", "scalar", "shared", "boundary"), default="all"
    )
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rt = load_module("runtime")
    r27 = load_module("r27_config")
    probe = load_module("r27_boundary_probe")
    plan = phase_plan(args, rt, r27, probe)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    r27.ensure_scope()
    arms = arm_names(args, r27)
    sources = source_records(r27, probe, arms)
    source_issues = [
        f"{arm}: manifest files or image identity failed verification"
        for arm, source in sources.items()
        if source.get("all_manifest_files_verified") is not True
        or source.get("image") != r27.IMAGES[arm]
    ]
    if source_issues:
        raise RuntimeError("; ".join(source_issues))
    rt.ROOT.mkdir(parents=True, exist_ok=True)
    rt.save_json("r27-boundary-plan.json", plan)
    scalar_results: list[dict[str, Any]] = []
    if args.suite in {"all", "scalar"}:
        for arm in arms:
            scalar_results.append(run_scalar_arm(rt, r27, probe, arm, sources[arm]))
    cells = selected_cells(args, r27)
    http_results = [
        run_http_cell(rt, r27, probe, cell, sources[cell.arm], args.request_timeout)
        for cell in cells
    ]
    summary = final_summary(rt, r27, plan, cells, scalar_results, http_results)
    rt.save_json("r27-boundary-phase-summary.json", summary)
    rt.record_gate("r27-boundary-phase", summary["passed"], summary["gates"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
