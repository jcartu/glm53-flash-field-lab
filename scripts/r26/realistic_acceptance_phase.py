#!/usr/bin/env python3
"""Run the matched realistic-chat MTP acceptance slice across four pinned arms."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROBE = Path(__file__).with_name("realistic_acceptance_probe.py")
MTP_HEAD_ENV = "VLLM_GLM53_MTP_DRAFT_HEAD"
PROBE_TIMEOUT_SECONDS = 14_400


@dataclass(frozen=True)
class Arm:
    name: str
    image: str
    draft_head: str
    description: str
    fairness_env: str = "compute_share"
    extra_args: tuple[str, ...] = ()


def load_runtime() -> Any:
    """Import Main's live runtime contract without executing a phase on import."""
    try:
        return importlib.import_module(".runtime", package=__package__)
    except (ImportError, TypeError):
        return importlib.import_module("runtime")


def load_probe() -> Any:
    try:
        return importlib.import_module(
            ".realistic_acceptance_probe", package=__package__
        )
    except (ImportError, TypeError):
        return importlib.import_module("realistic_acceptance_probe")


def arm_plan(rt: Any) -> tuple[Arm, ...]:
    return (
        Arm(
            name="r25-mtp3-bf16",
            image=rt.R25_IMAGE,
            draft_head="bf16",
            description="Official R25 baseline with its BF16 MTP proposal head explicitly pinned",
        ),
        Arm(
            name="r26-mtp3-bf16",
            image=rt.IMAGE,
            draft_head="bf16",
            description="Official R26 with the actual BF16 proposal-head toggle",
        ),
        Arm(
            name="r26-mtp3-default-nvfp4",
            image=rt.IMAGE,
            draft_head="nvfp4",
            description=(
                "Official R26 image-default NVFP4 proposal-head path, explicitly pinned to the actual "
                "nvfp4 toggle for auditable matching"
            ),
        ),
        Arm(
            name="drock-r26-overlay-mtp3-bf16",
            image=rt.OVERLAY_IMAGE,
            draft_head="bf16",
            description="D-Rock R26 overlay with the BF16 MTP proposal head explicitly pinned",
            fairness_env="none",
            extra_args=("--prefill-compute-share", "0.4"),
        ),
    )


def arm_extra_env(arm: Arm) -> dict[str, str]:
    return {
        MTP_HEAD_ENV: arm.draft_head,
        "FAIRNESS_ENGINE": arm.fairness_env,
        "PREFILL_COMPUTE_SHARE": "0.4",
    }


def output_path(rt: Any, arm: Arm) -> Path:
    return rt.ROOT / f"realistic-acceptance-{arm.name}.json"


def launch_label(arm: Arm) -> str:
    return f"realistic-acceptance-{arm.name}"


def launch_path(rt: Any, arm: Arm) -> Path:
    return rt.ROOT / f"{launch_label(arm)}.launch.json"


def phase_plan(rt: Any, probe: Any) -> dict[str, Any]:
    inputs = probe.input_manifest(rt.MODEL_NAME, probe.DEFAULT_SEEDS)
    return {
        "schema": "r26-realistic-acceptance-plan/v1",
        "phase_script": str(Path(__file__).resolve()),
        "probe_script": str(PROBE.resolve()),
        "root": str(rt.ROOT),
        "fixed_input_set": {
            "name": inputs["name"],
            "input_set_sha256": inputs["input_set_sha256"],
            "request_payload_fingerprint_sha256": inputs[
                "request_payload_fingerprint_sha256"
            ],
            "prompt_count": inputs["prompt_count"],
            "domain_counts": inputs["domain_counts"],
            "request_settings": inputs["request_settings"],
            "execution": inputs["execution"],
        },
        "matched_runtime": {
            "tp": 4,
            "dcp": 4,
            "spec": "mtp3",
            "cache": "vram",
            "kv": "fp8_ds_mla",
            "gpu_memory_utilization": 0.93,
            "target_weights": str(rt.MODEL),
            "weights_changed": False,
            "proposal_head_environment": MTP_HEAD_ENV,
            "fairness_semantic_target": "static prefill compute share 0.4",
            "fairness_interface_note": (
                "Official R25/R26 use legacy FAIRNESS_ENGINE=compute_share; the D-Rock overlay disables "
                "that rejected wrapper flag and supplies the semantically equivalent native "
                "--prefill-compute-share 0.4 CLI."
            ),
        },
        "arms": [
            {
                "arm": arm.name,
                "description": arm.description,
                "image": arm.image,
                "draft_head": arm.draft_head,
                "extra_env": arm_extra_env(arm),
                "extra_args": list(arm.extra_args),
                "output": str(output_path(rt, arm)),
                "launch_receipt": str(launch_path(rt, arm)),
                "inspect_receipt": str(
                    rt.ROOT / f"{launch_label(arm)}.inspect.json"
                ),
                "probe_command_receipt": str(
                    rt.ROOT / f"{launch_label(arm)}.probe.command.json"
                ),
            }
            for arm in arm_plan(rt)
        ],
        "gate_policy": (
            "Only launch matching, runtime completion, request accounting, idle counter-window integrity, "
            "and response/counter consistency are gated. No semantic-quality or statistical-equivalence "
            "gate is created."
        ),
    }


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def failure_receipt(
    rt: Any,
    probe: Any,
    arm: Arm,
    reason: str,
    detail: object,
    launch: dict[str, Any] | None,
) -> dict[str, Any]:
    now = time.time()
    receipt = {
        "schema": probe.SCHEMA,
        "suite": "realistic_acceptance",
        "arm": arm.name,
        "image": arm.image,
        "config": launch or {
            "image": arm.image,
            "tp": 4,
            "dcp": 4,
            "spec": "mtp3",
            "cache": "vram",
            "kv": "fp8_ds_mla",
            "env": {
                "GPU_MEMORY_UTILIZATION": "0.93",
                **arm_extra_env(arm),
            },
            "extra_args": list(arm.extra_args),
        },
        "config_provenance": {
            "path": str(launch_path(rt, arm)),
            "available": launch is not None,
        },
        "configuration_integrity": {
            "passed": False,
            "mismatches": {},
            "error": reason,
        },
        "input_set": probe.input_manifest(rt.MODEL_NAME, probe.DEFAULT_SEEDS),
        "started_at": now,
        "started_at_iso": probe.epoch_iso(now),
        "endpoint": f"{rt.BASE_URL.rstrip('/')}/v1/chat/completions",
        "groups": [],
        "summary": {
            "runtime_count_integrity": {
                "passed": False,
                "gate_scope": "runtime and count integrity only",
                "issues": [reason],
                "invalid_groups": [],
                "expected_groups": len(probe.DEFAULT_SEEDS) * (len(probe.PROMPTS) + 1),
                "observed_groups": 0,
                "expected_requests": len(probe.DEFAULT_SEEDS) * len(probe.PROMPTS) * 2,
                "observed_requests": 0,
            },
            "response_classifications": {},
            "claim_limits": {
                "semantic_quality_proof": False,
                "statistical_equivalence_test": False,
                "performance_kernel_benchmark": False,
            },
        },
        "measurement_scope": (
            "Natural-completion chat acceptance follow-up; not the sustained repeated-text ceiling probe"
        ),
        "semantic_quality_claim": False,
        "statistical_equivalence_claim": False,
        "failure": {"reason": reason, "detail": detail},
        "finished_at": now,
        "finished_at_iso": probe.epoch_iso(now),
    }
    rt.save_json(output_path(rt, arm).name, receipt)
    return receipt


def validate_receipt(
    rt: Any, probe: Any, arm: Arm, receipt: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    if receipt.get("schema") != probe.SCHEMA:
        issues.append(f"schema is {receipt.get('schema')!r}, expected {probe.SCHEMA!r}")
    if receipt.get("arm") != arm.name:
        issues.append(f"arm is {receipt.get('arm')!r}, expected {arm.name!r}")
    if receipt.get("image") != arm.image:
        issues.append("receipt image does not match the planned pinned image")
    config = receipt.get("config")
    if not isinstance(config, dict):
        issues.append("config is not an object")
    else:
        env = config.get("env") if isinstance(config.get("env"), dict) else {}
        expected_config = {
            "tp": 4,
            "dcp": 4,
            "spec": "mtp3",
            "cache": "vram",
            "kv": "fp8_ds_mla",
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                issues.append(
                    f"config.{key} is {config.get(key)!r}, expected {expected!r}"
                )
        if env.get("GPU_MEMORY_UTILIZATION") != "0.93":
            issues.append("config.env.GPU_MEMORY_UTILIZATION is not 0.93")
        if env.get(MTP_HEAD_ENV) != arm.draft_head:
            issues.append(
                f"config.env.{MTP_HEAD_ENV} is not the planned {arm.draft_head!r} toggle"
            )
        if env.get("FAIRNESS_ENGINE") != arm.fairness_env:
            issues.append(
                f"config.env.FAIRNESS_ENGINE is not the planned {arm.fairness_env!r} interface"
            )
        if env.get("PREFILL_COMPUTE_SHARE") != "0.4":
            issues.append("config.env.PREFILL_COMPUTE_SHARE is not 0.4")
        if config.get("extra_args") != list(arm.extra_args):
            issues.append(
                f"config.extra_args is {config.get('extra_args')!r}, expected {list(arm.extra_args)!r}"
            )
    expected_inputs = probe.input_manifest(rt.MODEL_NAME, probe.DEFAULT_SEEDS)
    observed_inputs = receipt.get("input_set")
    if not isinstance(observed_inputs, dict):
        issues.append("input_set is not an object")
    elif observed_inputs.get("input_set_sha256") != expected_inputs["input_set_sha256"]:
        issues.append("input_set_sha256 does not match the fixed phase input set")

    groups = receipt.get("groups")
    expected_groups = len(probe.DEFAULT_SEEDS) * (len(probe.PROMPTS) + 1)
    expected_requests = len(probe.DEFAULT_SEEDS) * len(probe.PROMPTS) * 2
    if not isinstance(groups, list):
        issues.append("groups is not a list")
    else:
        if len(groups) != expected_groups:
            issues.append(f"observed {len(groups)} groups, expected {expected_groups}")
        observed_requests = sum(
            len(group.get("requests", []))
            for group in groups
            if isinstance(group, dict) and isinstance(group.get("requests"), list)
        )
        if observed_requests != expected_requests:
            issues.append(
                f"observed {observed_requests} request records, expected {expected_requests}"
            )
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                issues.append(f"group {index} is not an object")
                continue
            required = (
                "concurrency",
                "prompt_ids",
                "started_at",
                "finished_at",
                "counter_before",
                "counter_after",
                "counter_delta",
                "acceptance_fraction",
                "accepted_draft_tokens_per_step",
                "emitted_tokens_per_verifier_step",
                "requests",
            )
            missing = [key for key in required if key not in group]
            if missing:
                issues.append(f"group {index} missing fields {missing}")
            if group.get("measurement_eligible") is not True:
                issues.append(f"group {index} failed runtime/count integrity")
    summary = receipt.get("summary")
    integrity = summary.get("runtime_count_integrity") if isinstance(summary, dict) else None
    if not isinstance(integrity, dict) or integrity.get("passed") is not True:
        issues.append("receipt runtime_count_integrity did not pass")
    return issues


def probe_command(
    rt: Any, probe: Any, arm: Arm, request_timeout: float
) -> list[str]:
    return [
        sys.executable,
        str(PROBE),
        "--arm",
        arm.name,
        "--image",
        arm.image,
        "--config",
        str(launch_path(rt, arm)),
        "--expected-draft-head",
        arm.draft_head,
        "--target-weights",
        str(rt.MODEL),
        "--output",
        str(output_path(rt, arm)),
        "--base-url",
        rt.BASE_URL,
        "--model",
        rt.MODEL_NAME,
        "--seeds",
        ",".join(str(seed) for seed in probe.DEFAULT_SEEDS),
        "--request-timeout",
        str(request_timeout),
    ]


def record_result_gate(
    rt: Any,
    arm: Arm,
    passed: bool,
    issues: list[str],
    return_code: int | None,
    reused: bool,
) -> None:
    rt.record_gate(
        f"runtime-count:realistic-acceptance:{arm.name}",
        passed,
        {
            "scope": (
                "launch/runtime/count integrity only; not semantic quality, statistical equivalence, pure "
                "decode timing, or GPU kernel-step evidence"
            ),
            "returncode": return_code,
            "receipt": str(output_path(rt, arm)),
            "command_receipt": str(
                rt.ROOT / f"{launch_label(arm)}.probe.command.json"
            ),
            "immutable_existing_receipt_reused": reused,
            "issues": issues,
        },
    )


def run_arm(
    rt: Any, probe: Any, arm: Arm, request_timeout: float
) -> dict[str, Any]:
    output = output_path(rt, arm)
    if output.exists():
        try:
            existing = read_object(output)
            issues = validate_receipt(rt, probe, arm, existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues = [f"existing immutable receipt is unreadable: {type(error).__name__}: {error}"]
        passed = not issues
        record_result_gate(rt, arm, passed, issues, None, True)
        return {
            "arm": arm.name,
            "output": str(output),
            "passed": passed,
            "reused_existing_immutable_receipt": True,
            "issues": issues,
        }

    rt.note(f"REALISTIC ACCEPTANCE ARM START {arm.name}")
    booted = False
    stopped = True
    return_code: int | None = None
    launch: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    phase_issues: list[str] = []
    try:
        booted = bool(
            rt.boot(
                launch_label(arm),
                image=arm.image,
                tp=4,
                dcp=4,
                spec="mtp3",
                cache="vram",
                kv="fp8_ds_mla",
                extra_env=arm_extra_env(arm),
                extra_args=list(arm.extra_args) or None,
            )
        )
        if launch_path(rt, arm).exists():
            launch = read_object(launch_path(rt, arm))
        if not booted:
            phase_issues.append("runtime.boot returned false")
        else:
            command = probe_command(rt, probe, arm, request_timeout)
            return_code = rt.run(
                command,
                label=f"{launch_label(arm)}.probe",
                timeout=PROBE_TIMEOUT_SECONDS,
                env=rt.PROXY_ENV,
            )
            if output.exists():
                receipt = read_object(output)
            else:
                phase_issues.append("probe produced no JSON receipt")
    except Exception as error:
        phase_issues.append(f"{type(error).__name__}: {error}")
        phase_issues.append(traceback.format_exc())
    finally:
        try:
            rt.stop()
        except Exception as error:
            stopped = False
            phase_issues.append(f"runtime.stop failed: {type(error).__name__}: {error}")

    if receipt is None and output.exists():
        try:
            receipt = read_object(output)
            phase_issues.append("probe did not return normally; preserved its partial raw receipt")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            phase_issues.append(
                f"partial probe receipt is unreadable: {type(error).__name__}: {error}"
            )

    if receipt is None:
        receipt = failure_receipt(
            rt,
            probe,
            arm,
            "realistic acceptance arm did not produce a complete probe receipt",
            {
                "booted": booted,
                "returncode": return_code,
                "stopped": stopped,
                "issues": phase_issues,
            },
            launch,
        )
    receipt_issues = validate_receipt(rt, probe, arm, receipt)
    issues = phase_issues + receipt_issues
    passed = booted and stopped and return_code == 0 and not issues
    record_result_gate(rt, arm, passed, issues, return_code, False)
    rt.note(f"REALISTIC ACCEPTANCE ARM COMPLETE {arm.name}")
    return {
        "arm": arm.name,
        "output": str(output),
        "passed": passed,
        "booted": booted,
        "stopped": stopped,
        "returncode": return_code,
        "reused_existing_immutable_receipt": False,
        "issues": issues,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--arm", default="all")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rt = load_runtime()
    probe = load_probe()
    arms = arm_plan(rt)
    arm_names = {arm.name for arm in arms}
    if args.arm != "all" and args.arm not in arm_names:
        raise SystemExit(
            f"unknown --arm {args.arm!r}; choose all or one of {', '.join(sorted(arm_names))}"
        )
    plan = phase_plan(rt, probe)
    if args.print_plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    selected = [arm for arm in arms if args.arm in ("all", arm.name)]
    rt.ROOT.mkdir(parents=True, exist_ok=True)
    results = [run_arm(rt, probe, arm, args.request_timeout) for arm in selected]
    summary = {
        "phase": "realistic_acceptance",
        "schema": probe.SCHEMA,
        "input_set_sha256": plan["fixed_input_set"]["input_set_sha256"],
        "results": results,
        "passed": all(result["passed"] for result in results),
        "gate_scope": "runtime and count integrity only",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
