#!/usr/bin/env python3
"""Preserve the valid pre-fault QAD slice and run the remaining original phases."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

from run_qad_battery import checkpoint_complete

HERE = Path(__file__).resolve().parent
ORIGINAL_DRIVER = HERE / "run_qad_battery.py"
COORDINATOR = HERE / "run_qualification.py"

PINNED_REVISION = "3959f8a063b77cfdb22ab2e085a1f76fd38b195b"
PINNED_IMAGE = (
    "voipmonitor/vllm@sha256:"
    "d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68"
)
PINNED_MODEL = Path("/mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD-step2500")
PUBLISHED_MODEL = Path("/mnt/2king/models/GLM-5.3-Flash-NVFP4")
MODEL_TAG = "qad-step2500-3959f8a063b7"
L2_SHARED = Path(f"/mnt/2king/lmcache-r26-battery/shared-{MODEL_TAG}")
DEFAULT_FAULT_TIMESTAMP = "2026-09-07T09:25:37+02:00"
FIRST_PHASE = "qad-matched-published-vs-candidate"

# This is the immutable QAD runbook contract.  The same literal assignment is
# also recovered from run_qad_battery.py and compared with the source receipt.
PINNED_PHASES = [
    (FIRST_PHASE, "qad_matched_phase.py", [], 14400),
    ("quality-and-mtp-dcp-correctness", "quality_phase.py", [], 21600),
    ("festr-matched-acceptance", "matrix_phase.py", ["--section", "priority"], 18000),
    ("realistic-mtp3-acceptance", "realistic_acceptance_phase.py", [], 10800),
    ("cache-lifecycle-and-boundaries", "cache_phase.py", [], 21600),
    ("complete-topology-matrix", "matrix_phase.py", ["--section", "matrix"], 21600),
    ("agentic-prefix-cache-reuse", "agent_cache_probe.py", ["--arm", "all"], 7200),
    ("batch-dma-and-overlay-controls", "matrix_phase.py", ["--section", "tuning"], 10800),
    ("tp2-capacity", "matrix_phase.py", ["--section", "tp2"], 7200),
    ("mixed-agent-scheduling", "scheduler_recheck_phase.py", [], 21600),
    ("direct-dcp-peer-guard", "peer_phase.py", [], 7200),
]

MATCHED_LABELS = [
    ("qad-matched-dcp1-dflash-published", "dcp1-dflash", "published", PUBLISHED_MODEL, 1, "dflash2"),
    ("qad-matched-dcp1-dflash-candidate", "dcp1-dflash", "candidate", PINNED_MODEL, 1, "dflash2"),
    ("qad-matched-dcp4-mtp3-candidate", "dcp4-mtp3", "candidate", PINNED_MODEL, 4, "mtp3"),
    ("qad-matched-dcp4-mtp3-published", "dcp4-mtp3", "published", PUBLISHED_MODEL, 4, "mtp3"),
]
EXPLICIT_PRESERVED_FILES = (
    "checkpoint-verification.json",
    "checkpoint-template-comparison.json",
    f"phase-{FIRST_PHASE}.command.json",
    f"phase-{FIRST_PHASE}.log",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    path = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"Required regular file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON evidence: {path}") from error


def phase_rows(phases: list[tuple]) -> list[dict]:
    return [
        {"name": name, "script": script, "args": args, "timeout_seconds": timeout}
        for name, script, args, timeout in phases
    ]


def source_driver_phases() -> list[tuple]:
    tree = ast.parse(ORIGINAL_DRIVER.read_text(), filename=str(ORIGINAL_DRIVER))
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "coordinator"
            and target.attr == "PHASES"
            for target in node.targets
        ):
            assignments.append(node)
    require(len(assignments) == 1, "Original QAD driver must contain one coordinator.PHASES assignment")
    try:
        value = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Original QAD phase plan is not a literal") from error
    require(isinstance(value, list), "Original QAD phase plan is not a list")
    return value


def fault_epoch(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"Invalid fault timestamp: {value}") from error
    require(parsed.tzinfo is not None, "Fault timestamp must include a UTC offset")
    return parsed.timestamp()


def validate_result(path: Path, label: str, kind: str) -> None:
    result = read_json(path)
    require(isinstance(result, dict), f"Matched result is not an object: {path}")
    if kind in {"estonia", "lavd-test"}:
        summary = result.get("summary")
        require(result.get("label") == f"{label}-{kind}", f"Matched result label mismatch: {path}")
        require(isinstance(summary, dict), f"Matched quality summary is absent: {path}")
        require(
            summary.get("requested") == 24
            and summary.get("completed") == 24
            and summary.get("runtime_errors") == 0
            and summary.get("metrics_errors") == 0,
            f"Matched quality result is technically incomplete: {path}",
        )
    elif kind == "lavd-template-default":
        metadata = result.get("metadata")
        summary = result.get("selected_summary")
        require(isinstance(metadata, dict) and isinstance(summary, dict), f"Template result is incomplete: {path}")
        require(
            metadata.get("interrupted") is False
            and metadata.get("requested_runs") == 8
            and summary.get("attempted") == 8
            and summary.get("completed") == 8
            and summary.get("errors") == 0,
            f"Template result did not complete eight runs: {path}",
        )
    elif kind == "long":
        summary = result.get("summary")
        require(result.get("label") == f"{label}-long", f"Matched long result label mismatch: {path}")
        require(isinstance(summary, dict), f"Matched long summary is absent: {path}")
        require(
            summary.get("requested") == 24
            and summary.get("completed") == 24
            and summary.get("runtime_errors") == 0
            and summary.get("metrics_errors") == 0,
            f"Matched long result is technically incomplete: {path}",
        )
    elif kind == "acceptance":
        samples = result.get("samples")
        require(result.get("label") == label, f"Matched acceptance label mismatch: {path}")
        require(
            result.get("sample_count") == 8
            and result.get("passed") is True
            and isinstance(samples, list)
            and len(samples) == 8
            and all(isinstance(sample, dict) and sample.get("passed") is True for sample in samples),
            f"Matched acceptance result is incomplete: {path}",
        )
    else:
        raise AssertionError(kind)


def validate_source(source_root: Path, output_root: Path, fault_timestamp: str) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    require(source_root.is_dir(), f"Source evidence root is missing: {source_root}")
    require(not output_root.exists(), f"Refusing to overwrite output root: {output_root}")
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("Output root must not be inside the immutable source root")

    expected_plan = phase_rows(PINNED_PHASES)
    require(source_driver_phases() == PINNED_PHASES, "run_qad_battery.py no longer has the pinned 11-phase plan")
    source_plan_path = source_root / "phase-plan.json"
    source_plan = read_json(source_plan_path)
    require(source_plan == expected_plan, "Source phase-plan.json is not the pinned original 11-phase plan")

    checkpoint_path = source_root / "checkpoint-verification.json"
    checkpoint = read_json(checkpoint_path)
    require(isinstance(checkpoint, dict), "Checkpoint verification receipt is not an object")
    require(
        checkpoint.get("model_dir") == str(PINNED_MODEL)
        and checkpoint.get("revision") == PINNED_REVISION
        and checkpoint.get("complete") is True,
        "Source checkpoint receipt does not verify the pinned QAD step2500 checkpoint",
    )
    current_checkpoint = checkpoint_complete(PINNED_MODEL, PINNED_REVISION)
    require(current_checkpoint.get("complete") is True, f"Pinned checkpoint is no longer complete: {current_checkpoint}")

    template_identity = {
        "schema": "checkpoint-template-identity/v1",
        "verified_at": time.time(),
        "files": {},
        "scope": "Fresh verification of all served tokenizer, generation and standalone template artifacts; historical comparison receipts are preserved unchanged.",
    }
    for filename in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        published = file_identity(PUBLISHED_MODEL / filename)
        candidate = file_identity(PINNED_MODEL / filename)
        require(published["sha256"] == candidate["sha256"],
                f"Checkpoint tokenizer/template artifact differs: {filename}")
        template_identity["files"][filename] = {
            "published": published, "candidate": candidate, "identical": True,
        }

    completed = read_json(source_root / "qad-matched-completed.json")
    ledger = read_json(source_root / "qad-matched-ledger.json")
    require(
        isinstance(completed, dict)
        and completed.get("complete") is True
        and completed.get("cells") == 4,
        "Matched checkpoint completion receipt is not complete",
    )
    require(
        isinstance(ledger, dict)
        and ledger.get("published") == str(PUBLISHED_MODEL)
        and ledger.get("candidate") == str(PINNED_MODEL)
        and ledger.get("image") == PINNED_IMAGE
        and isinstance(ledger.get("cells"), list),
        "Matched checkpoint ledger does not use the pinned models and R26 image",
    )
    cells = ledger["cells"]
    require([cell.get("label") for cell in cells if isinstance(cell, dict)] == [row[0] for row in MATCHED_LABELS],
            "Matched checkpoint ledger is not the complete original four-arm sequence")

    for cell, (label, config, weights, model, dcp, spec) in zip(cells, MATCHED_LABELS, strict=True):
        require(
            isinstance(cell, dict)
            and cell.get("config") == config
            and cell.get("weights") == weights
            and cell.get("model_dir") == str(model)
            and cell.get("booted") is True
            and cell.get("acceptance_returncode") == 0,
            f"Matched checkpoint arm is incomplete: {label}",
        )
        expected_receipts = [
            f"{label}-estonia.json",
            f"{label}-lavd-test.json",
            f"{label}-lavd-template-default.json",
            f"{label}-long.json",
            f"{label}-acceptance.json",
        ]
        require(cell.get("receipts") == expected_receipts, f"Matched result ledger changed: {label}")
        for kind, filename in zip(
            ("estonia", "lavd-test", "lavd-template-default", "long", "acceptance"),
            expected_receipts,
            strict=True,
        ):
            validate_result(source_root / filename, label, kind)

        launch = read_json(source_root / f"{label}.launch.json")
        boot = read_json(source_root / f"{label}.boot.command.json")
        require(
            isinstance(launch, dict)
            and launch.get("label") == label
            and launch.get("image") == PINNED_IMAGE
            and launch.get("model_dir") == str(model)
            and launch.get("gpus") == "0,1,2,3"
            and launch.get("dcp") == dcp
            and launch.get("spec") == spec,
            f"Matched launch is not the pinned four-GPU arm: {label}",
        )
        require(
            isinstance(boot, dict)
            and boot.get("returncode") == 0
            and isinstance(boot.get("args"), list)
            and PINNED_IMAGE in boot["args"]
            and '"device=0,1,2,3"' in boot["args"],
            f"Matched boot receipt is not successful and four-GPU pinned: {label}",
        )

    invocation_path = source_root / f"phase-{FIRST_PHASE}.command.json"
    invocation = read_json(invocation_path)
    cutoff = fault_epoch(fault_timestamp)
    require(isinstance(invocation, dict), "Matched phase invocation receipt is not an object")
    invocation_args = invocation.get("args")
    require(
        invocation.get("returncode") == 0
        and isinstance(invocation_args, list)
        and len(invocation_args) == 2
        and Path(invocation_args[1]).resolve() == HERE / "qad_matched_phase.py"
        and isinstance(invocation.get("started_at"), (int, float))
        and not isinstance(invocation.get("started_at"), bool)
        and isinstance(invocation.get("finished_at"), (int, float))
        and not isinstance(invocation.get("finished_at"), bool)
        and invocation["started_at"] <= invocation["finished_at"] < cutoff
        and invocation.get("log") == str(source_root / f"phase-{FIRST_PHASE}.log"),
        "Matched phase was not successfully completed before the hardware fault cutoff",
    )

    isolation_path = source_root / "gpu-isolation-events.jsonl"
    isolation_bytes = isolation_path.read_bytes()
    selected_lines = []
    for line in isolation_bytes.splitlines(keepends=True):
        if not line.strip():
            continue
        sample = json.loads(line)
        if invocation["started_at"] - 5 <= sample["timestamp"] <= invocation["finished_at"] + 5:
            require(sample["timestamp"] < cutoff, "Preserved isolation sample reaches the hardware fault")
            require(not sample.get("foreign") and sample.get("speed_eligible", True) is True,
                    "Preserved matched phase has an ineligible GPU isolation sample")
            selected_lines.append(line)
    require(len(selected_lines) >= 2, "Preserved matched phase lacks GPU isolation evidence")
    isolation_bootstrap = b"".join(selected_lines)

    matched_paths = sorted(source_root.glob("qad-matched-*"), key=lambda path: path.name)
    require(matched_paths, "Source root has no qad-matched-* artifacts")
    preserved_paths = matched_paths + [source_root / name for name in EXPLICIT_PRESERVED_FILES]
    require(len({path.name for path in preserved_paths}) == len(preserved_paths), "Preserved artifact names collide")
    preserved = {path.name: file_identity(path) for path in preserved_paths}

    script_names = sorted({script for _name, script, _args, _timeout in PINNED_PHASES[1:]})
    drivers = {
        "continuation": file_identity(Path(__file__)),
        "original_qad_driver": file_identity(ORIGINAL_DRIVER),
        "coordinator": file_identity(COORDINATOR),
        "remaining_phase_scripts": {name: file_identity(HERE / name) for name in script_names},
    }
    return {
        "source_root": source_root,
        "output_root": output_root,
        "fault_timestamp": fault_timestamp,
        "fault_epoch": cutoff,
        "source_plan_bytes": source_plan_path.read_bytes(),
        "source_plan_identity": file_identity(source_plan_path),
        "checkpoint": checkpoint,
        "current_checkpoint": current_checkpoint,
        "template_identity": template_identity,
        "invocation": invocation,
        "preserved": preserved,
        "isolation_bootstrap_bytes": isolation_bootstrap,
        "isolation_bootstrap": {
            "source": file_identity(isolation_path),
            "sample_count": len(selected_lines),
            "sha256": hashlib.sha256(isolation_bootstrap).hexdigest(),
            "window": [invocation["started_at"] - 5, invocation["finished_at"] + 5],
            "policy": "Original sampler lines only; new coordinator samples append to this stream. Consumers still qualify each measurement window.",
        },
        "drivers": drivers,
        "excluded_attempts": {
            "qualification_interrupted": file_identity(source_root / "qualification-interrupted.json"),
            "phase_progress": file_identity(source_root / "phase-progress.json"),
        },
    }


def battery_environment(output_root: Path) -> dict[str, str]:
    return {
        "BATTERY_ROOT": str(output_root),
        "BATTERY_CONTAINER": "r26-test",
        "BATTERY_PORT": "5002",
        "BATTERY_IMAGE": PINNED_IMAGE,
        "BATTERY_MODEL_DIR": str(PINNED_MODEL),
        "BATTERY_MODEL_CACHE_TAG": MODEL_TAG,
        "BATTERY_L2_SHARED": str(L2_SHARED),
        "GPU_ISOLATION_MODE": "strict",
    }


def compact_hashes(identities: dict[str, dict]) -> dict[str, dict]:
    return {
        name: {"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}
        for name, identity in identities.items()
    }


def plan_receipt(state: dict, copied: dict[str, dict] | None = None) -> dict:
    artifact_hashes = compact_hashes(copied if copied is not None else state["preserved"])
    return {
        "schema": "r26-qad-recovery-plan/v1",
        "created_at": time.time(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(state["source_root"]),
        "output_root": str(state["output_root"]),
        "fault_timestamp": state["fault_timestamp"],
        "fault_epoch": state["fault_epoch"],
        "pinned": {
            "checkpoint": str(PINNED_MODEL),
            "checkpoint_revision": PINNED_REVISION,
            "image": PINNED_IMAGE,
            "model_cache_tag": MODEL_TAG,
            "l2_namespace": str(L2_SHARED),
        },
        "phase_plan": {
            "phase_count": len(PINNED_PHASES),
            "source_receipt": state["source_plan_identity"],
            "phases": phase_rows(PINNED_PHASES),
        },
        "stages": [
            {
                "stage": 1,
                "disposition": "preserved-successful-pre-fault-invocation",
                "phase": FIRST_PHASE,
                "returncode": 0,
                "finished_at": state["invocation"]["finished_at"],
                "artifact_count": len(artifact_hashes),
                "artifact_hashes": artifact_hashes,
                "copy_verified": copied is not None,
                "raw_embedded_paths_rewritten": False,
            },
            {
                "stage": 2,
                "disposition": "fresh-post-recovery-execution-through-run_qualification",
                "phases": [name for name, _script, _args, _timeout in PINNED_PHASES[1:]],
            },
            {
                "stage": 3,
                "disposition": "merge-recorded-returncodes-and-restore-original-phase-plan",
                "all_phases_attempted_policy": "written only after all ten remaining invocations are recorded",
            },
        ],
        "preserved_isolation": state["isolation_bootstrap"],
        "preserved_phase_count": 1,
        "remaining_phase_count": 10,
        "drivers": state["drivers"],
        "environment": battery_environment(state["output_root"]),
        "excluded_fault-era_source_attempts": {
            **state["excluded_attempts"],
            "policy": (
                "Only the completed qad-matched-* slice and its verification/invocation receipts are copied. "
                "All later source-root attempts remain immutable there and are not treated as recovered findings."
            ),
        },
        "checkpoint_revalidation": state["current_checkpoint"],
        "template_identity": state["template_identity"],
        "gpu_jobs_started_by_planning": False,
    }


def write_json_new(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    require(not path.exists() and not temporary.exists(), f"Refusing to overwrite receipt: {path}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"Stale temporary file blocks recovery: {temporary}")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    data = (json.dumps(value, indent=2) + "\n").encode()
    atomic_write(path, data)


def copy_preserved(state: dict) -> dict[str, dict]:
    copied = {}
    for name, source_identity in state["preserved"].items():
        source = state["source_root"] / name
        destination = state["output_root"] / name
        require(not destination.exists(), f"Refusing to overwrite preserved artifact: {destination}")
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        shutil.copystat(source, destination)
        destination_identity = file_identity(destination)
        require(
            destination_identity["sha256"] == source_identity["sha256"]
            and destination_identity["size_bytes"] == source_identity["size_bytes"],
            f"Preserved artifact changed during relocation: {name}",
        )
        copied[name] = destination_identity
    return copied


def reused_phase_row(state: dict) -> dict:
    name = f"phase-{FIRST_PHASE}.command.json"
    identity = state["preserved"][name]
    return {
        "phase": FIRST_PHASE,
        "returncode": 0,
        "execution": "reused-successful-pre-fault-invocation",
        "source_invocation_receipt": identity["path"],
        "copied_invocation_receipt": str(state["output_root"] / name),
        "invocation_receipt_sha256": identity["sha256"],
    }


def remaining_progress(output_root: Path) -> list[dict]:
    path = output_root / "phase-progress.json"
    if not path.exists():
        return []
    rows = read_json(path)
    require(isinstance(rows, list), "Recovery phase-progress.json is not a list")
    expected = [phase[0] for phase in PINNED_PHASES[1:]]
    observed = []
    for row in rows:
        require(isinstance(row, dict), "Recovery phase progress has a non-object row")
        phase = row.get("phase")
        code = row.get("returncode")
        require(
            isinstance(phase, str) and isinstance(code, int) and not isinstance(code, bool),
            "Recovery phase progress has an invalid invocation row",
        )
        observed.append(phase)
    if observed and observed[0] == FIRST_PHASE:
        require(rows[0]["returncode"] == 0, "Preserved matched phase progress is not successful")
        rows = rows[1:]
        observed = observed[1:]
    require(observed == expected[: len(observed)], "Recovery phase progress is not an original-plan prefix")
    return rows


def restore_plan_and_progress(state: dict) -> None:
    atomic_write(state["output_root"] / "phase-plan.json", state["source_plan_bytes"])
    rows = remaining_progress(state["output_root"])
    atomic_write_json(state["output_root"] / "phase-progress.json", [reused_phase_row(state), *rows])


def validate_remaining_execution(state: dict) -> tuple[dict, list[dict]]:
    execution_path = state["output_root"] / "qualification-executed.json"
    execution = read_json(execution_path)
    require(isinstance(execution, dict), "Coordinator execution receipt is not an object")
    rows = execution.get("phases")
    require(
        execution.get("all_phases_attempted") is True and isinstance(rows, list),
        "Coordinator did not record all ten remaining phase attempts",
    )
    expected_names = [phase[0] for phase in PINNED_PHASES[1:]]
    require(
        [row.get("phase") for row in rows if isinstance(row, dict)] == expected_names,
        "Coordinator execution ledger is not the original ten-phase suffix",
    )
    for row, (phase, script, arguments, _timeout) in zip(rows, PINNED_PHASES[1:], strict=True):
        code = row.get("returncode") if isinstance(row, dict) else None
        require(isinstance(code, int) and not isinstance(code, bool), f"Invalid return code for {phase}")
        command_path = state["output_root"] / f"phase-{phase}.command.json"
        command = read_json(command_path)
        argv = command.get("args") if isinstance(command, dict) else None
        require(
            isinstance(command, dict)
            and command.get("returncode") == code
            and isinstance(argv, list)
            and len(argv) >= 2
            and Path(argv[1]).resolve() == HERE / script
            and argv[2:] == arguments
            and isinstance(command.get("started_at"), (int, float))
            and not isinstance(command.get("started_at"), bool)
            and isinstance(command.get("finished_at"), (int, float))
            and not isinstance(command.get("finished_at"), bool),
            f"Actual invocation receipt does not match the recorded phase: {phase}",
        )
    return execution, rows


def merge_execution(state: dict, coordinator_returned: bool) -> None:
    execution_path = state["output_root"] / "qualification-executed.json"
    if not execution_path.exists():
        require(not coordinator_returned, "Coordinator returned without an execution receipt")
        return
    execution, rows = validate_remaining_execution(state)
    raw_execution = execution_path.read_bytes()
    remaining_path = state["output_root"] / "recovery-remaining-qualification-executed.json"
    if not remaining_path.exists():
        with remaining_path.open("xb") as handle:
            handle.write(raw_execution)
    remaining_identity = file_identity(remaining_path)
    recovery_plan_identity = file_identity(state["output_root"] / "recovery-plan.json")
    merged = {
        **execution,
        "all_phases_attempted": True,
        "phases": [reused_phase_row(state), *rows],
        "recovery": {
            "schema": "r26-qad-recovery-execution/v1",
            "coordinator_returned": coordinator_returned,
            "preserved_phase_count": 1,
            "fresh_phase_count": 10,
            "remaining_coordinator_execution": remaining_identity,
            "recovery_plan": recovery_plan_identity,
            "source_root": str(state["source_root"]),
            "raw_embedded_paths_rewritten": False,
        },
    }
    atomic_write_json(execution_path, merged)


def execute(state: dict) -> None:
    output_root = state["output_root"]
    output_root.mkdir(parents=True, exist_ok=False)
    copied = copy_preserved(state)
    write_json_new(output_root / "recovery-plan.json", plan_receipt(state, copied))
    write_json_new(output_root / "checkpoint-template-identity.json", state["template_identity"])
    atomic_write(output_root / "gpu-isolation-events.jsonl", state["isolation_bootstrap_bytes"])
    atomic_write(output_root / "phase-plan.json", state["source_plan_bytes"])
    write_json_new(output_root / "phase-progress.json", [reused_phase_row(state)])

    require("runtime" not in sys.modules and "run_qualification" not in sys.modules,
            "Runtime/coordinator was imported before the pinned recovery environment was installed")
    os.environ.update(battery_environment(output_root))
    coordinator = importlib.import_module("run_qualification")
    remaining = PINNED_PHASES[1:]
    coordinator.PHASES = remaining

    try:
        coordinator.main()
    except BaseException as error:
        try:
            merge_execution(state, coordinator_returned=False)
        except Exception as merge_error:
            error.add_note(f"Could not merge a post-attempt recovery ledger: {merge_error}")
        raise
    else:
        merge_execution(state, coordinator_returned=True)
    finally:
        restore_plan_and_progress(state)

    print("QAD RECOVERY COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fault-timestamp", default=DEFAULT_FAULT_TIMESTAMP)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    state = validate_source(args.source_root, args.output_root, args.fault_timestamp)
    if args.plan_only:
        receipt = plan_receipt(state)
        receipt.update({
            "plan_only": True,
            "filesystem_mutation_performed": False,
            "gpu_jobs_started": False,
        })
        print(json.dumps(receipt, indent=2))
        return
    execute(state)


if __name__ == "__main__":
    main()
