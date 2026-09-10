#!/usr/bin/env python3
"""Normalize R26 battery receipts into an auditable, rerunnable qualification report.

Read-only over the battery root.  Writes only results/r26/r26-report-*.

Every planned cell of every phase is listed; a cell without a receipt is
``missing``, never a pass.  Official R26, matched R25 controls, the D-Rock
overlay, the #599 hardware phase and unsupported-capability outcomes stay in
separate buckets.  Speed figures are qualified only when the command window is
fully covered by GPU-isolation samples that show no foreign GPU process; every
other speed sample is listed as contaminated or uncovered and is never charted.
"""
from __future__ import annotations

import ast
import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_reclassify  # noqa: E402
import runtime as rt  # noqa: E402

SCHEMA = "r26-qualification-report/v2"
REALISTIC_SCHEMA = "r26-realistic-acceptance/v1"
CACHE_PROBE_SCHEMA = "r26-cache-probe/v2"
CACHE_PHASE_SCHEMA = "r26-cache-phase/v2"
CACHE_RECHECK_SCHEMA = "r26-cache-recheck/v1"
AGENT_CACHE_SCHEMA = "r26-agent-cache/v1"
STEADY_SCHEMA = "r26-steady-counters/v1"
FINAL_DIAGNOSTICS_SCHEMA = "r26-final-runtime-diagnostics/v1"
LANE_RECHECK_SCHEMA = "r26-scheduler-lane-cap-recheck/v1"
FOLLOWUP_PLAN_SCHEMA = "r26-followup-plan/v1"
DEFAULT_OUT = HERE.parent.parent / "results" / "r26"
EXCLUDED_DIR = "excluded-initial-gpu-contention"
HISTORY_KV_EVIDENCE = DEFAULT_OUT / "history-kv-evidence.json"
FOLLOWUPS_DIR = "followups"
CLEAN_RERUNS_DIR = "clean-reruns"
SUBSTAGES = (FOLLOWUPS_DIR, CLEAN_RERUNS_DIR)
RECHECK_MAP = f"{FOLLOWUPS_DIR}/overlay-quality-recheck-map.json"
FOLLOWUP_EXECUTION_PLAN = "followup-execution-plan.json"
CACHE_RECHECK_SUMMARY = f"{FOLLOWUPS_DIR}/cache-recheck-summary.json"
CACHE_RECHECK_COMPLETE = f"{FOLLOWUPS_DIR}/cache-recheck-complete.json"
CLEAN_RERUN_PLAN = f"{CLEAN_RERUNS_DIR}/clean-rerun-plan.json"
FINAL_DIAGNOSTICS = f"{CLEAN_RERUNS_DIR}/final-runtime-diagnostics.json"
LANE_RECHECK = f"{CLEAN_RERUNS_DIR}/scheduler-lane-cap-recheck.json"
REASONING_PROTOCOL_PROOF = "reasoning-parameter-protocol-proof.json"
# Same line release_history.py reads; bench metadata.max_total_tokens is not capacity evidence.
SERVER_KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
ARMS = {
    rt.IMAGE: "official-r26",
    rt.R25_IMAGE: "r25-control",
    rt.OVERLAY_IMAGE: "drock-overlay",
}
ARM_ORDER = ("official-r26", "r25-control", "drock-overlay", "hardware-599", "unattributed")
# qad_matched_phase.PUBLISHED: the released NVFP4 weights; any other mounted checkpoint defines its own arm.
PUBLISHED_WEIGHTS = Path("/mnt/2king/models/GLM-5.3-Flash-NVFP4")
# mirrors agent_cache_probe.ARMS and its series order
AGENT_CACHE_ARMS = (("stock-r26", rt.IMAGE), ("drock-overlay", rt.OVERLAY_IMAGE))
AGENT_CACHE_SERIES = ("vram", "lmcache")
SAMPLE_CADENCE_SECONDS = 2.0
MAX_SAMPLE_GAP_SECONDS = 10.0
LAUNCH_IDENTITY_KEYS = (
    "image", "tp", "dcp", "spec", "cache", "kv", "env", "extra_args",
    "model_dir", "draft_dir", "l2_host",
)
CACHE_CELLS = (
    "storage-evidence",
    "source-audit",
    "official-fp8-lifecycles",
    "official-packed-nvfp4-lifecycles",
    "official-packed-nvfp4-needles",
    "eviction",
    "native-canaries",
    "prefill-overhead",
    "alternating-r25-r26",
    "focused-stock-vs-overlay",
)
CACHE_RECHECK_GROUPS = {
    "official-fp8-lifecycles": ("lifecycle-fp8-80k", "lifecycle-fp8-exact-1m"),
    "official-packed-nvfp4-lifecycles": (
        "lifecycle-packed-nvfp4-80k",
        "lifecycle-packed-nvfp4-exact-1m",
    ),
    "official-packed-nvfp4-needles": ("cold-needles-packed-nvfp4",),
    "eviction": ("40-document-eviction",),
    "focused-stock-vs-overlay": ("focused-stock-vs-overlay",),
    "native-canaries": ("native-canaries",),
}
CACHE_GATE_CELLS = (
    ("cache:storage:", "storage-evidence"),
    ("cache:byte-level-external-transfer:", "source-audit"),
    ("cache:official-r26:fp8:", "official-fp8-lifecycles"),
    ("cache:prompt:cache-official-r26-fp8-", "official-fp8-lifecycles"),
    ("cache:restart:cache-official-r26-fp8-", "official-fp8-lifecycles"),
    ("cache:l2-cleanup:cleanup-recheck-fp8-lifecycles", "official-fp8-lifecycles"),
    ("cache:official-r26:packed-nvfp4:unique-1m-needles", "official-packed-nvfp4-needles"),
    ("cache:prompt:cache-official-r26-packed-nvfp4-unique-needles", "official-packed-nvfp4-needles"),
    ("cache:l2-cleanup:cleanup-recheck-packed-nvfp4-needles", "official-packed-nvfp4-needles"),
    ("cache:official-r26:packed-nvfp4:", "official-packed-nvfp4-lifecycles"),
    ("cache:prompt:cache-official-r26-packed-nvfp4-", "official-packed-nvfp4-lifecycles"),
    ("cache:restart:cache-official-r26-packed-nvfp4-", "official-packed-nvfp4-lifecycles"),
    ("cache:l2-cleanup:cleanup-recheck-packed-nvfp4-lifecycles", "official-packed-nvfp4-lifecycles"),
    ("cache:official-r26:40-document-eviction", "eviction"),
    ("cache:official-r26:eviction-exception", "eviction"),
    ("cache:l2-cleanup:cleanup-official-eviction", "eviction"),
    ("cache:official-r26:native-offload", "native-canaries"),
    ("cache:native:", "native-canaries"),
    ("cache:prompt:cache-native-", "native-canaries"),
    ("cache:prefill-overhead:", "prefill-overhead"),
    ("cache:alternating:", "alternating-r25-r26"),
    ("cache:r25-r26:", "alternating-r25-r26"),
    ("cache:focused:", "focused-stock-vs-overlay"),
    ("cache:drock-overlay:", "focused-stock-vs-overlay"),
    ("cache:stock-r26:", "focused-stock-vs-overlay"),
    ("cache:stock-vs-drock-overlay:", "focused-stock-vs-overlay"),
    ("cache:prompt:cache-focused-", "focused-stock-vs-overlay"),
    ("cache:restart:cache-focused-", "focused-stock-vs-overlay"),
    ("cache:l2-cleanup:cleanup-focused-", "focused-stock-vs-overlay"),
)
CACHE_STOCK_CONTROL_GATES = ("cache:stock-r26:#574:control-completed", "cache:stock-r26:#643-#645:control-completed")
# mirrors cache_phase.prefill_overhead arms and alternating_release_control sequence (cache_phase.py:955-1146, 1193-1201)
CACHE_PREFILL_ARMS = ("l2-off", "fresh-l2", "existing-l2")
CACHE_ALTERNATING_SEQUENCE = (("r25", "seed-cold"), ("r26", "warm-1"), ("r25", "warm-1"), ("r26", "warm-2"), ("r25", "warm-2"), ("r26", "warm-3"), ("r25", "warm-3"))
STATUSES = ("pass", "flagged", "fail", "unsupported", "skipped", "not-run", "superseded", "running", "missing")
PEER_MODES = (
    ("policy", "peer-policy", "isolated CPU-only #599 policy fault injection"),
    ("physical", "peer-physical", "real CUDA IPC peer matrix on this host"),
    ("collective", "peer599-model-collective", "model path with direct DCP A2A forced off"),
    ("auto", "peer599-model-auto", "model path with the #599 guard deciding"),
    ("forced-direct", "peer599-model-forced-direct", "operator override; only after a passing physical matrix"),
)
# overlay_quality_recheck.semantic_controls source contract.
SEMANTIC_CONTROLS = (("mtp0", "explicit-low", 24), ("dflash2", "template-default", 8))
SEMANTIC_CONTROL_NOTES = {
    "explicit-low": "matched to the battery's explicit low-reasoning DFlash LAVD runs (3/24 R26 vs 11/24 R25); no speculation, so it isolates target execution",
    "template-default": "matched to the historical R25 default-template LAVD observation (~22K generated tokens); never compared with low-reasoning runs",
}
LIFECYCLE_SIZES = ("80k", "exact-1m")
# Observation gaps flag a row. They do not turn successful serving into a runtime failure.
CACHE_OBSERVATION_CLASSES = {
    "harness-observation", "environment-evidence", "text-byte-inequality",
}


def cache_gate_cell(name: str) -> str | None:
    for prefix, cell in CACHE_GATE_CELLS:
        if name.startswith(prefix):
            return cell
    return None


def cache_failure_class(gate: dict[str, Any]) -> str:
    """Class a cache gate without turning observation gaps into product failures."""
    name = gate["name"]
    detail = gate.get("detail") if isinstance(gate.get("detail"), dict) else {}
    producer_class = detail.get("failure_class")
    if (
        any(token in name for token in ("exception", "boot", ":stop", ":restart", "l2-cleanup", "cell:"))
        or detail.get("reason") == "boot failed"
        or (isinstance(producer_class, str) and ("oom" in producer_class or "harness" in producer_class))
    ):
        return "runtime-protocol"
    if name == "cache:drock-overlay:#645:truthful-cache-events-and-replay":
        # final_runtime_diagnostics.py explicitly replays this after correcting the
        # probe's REQ/ROUTER client mismatch. The retained failed gate is an
        # observation-client result, not failed model serving or cache corruption.
        return "harness-observation"
    if name.startswith(("cache:storage:", "cache:byte-level-external-transfer:")):
        return "environment-evidence"
    if "prefill-overhead" in name or "alternating" in name:
        return "performance-window"
    if name.endswith(":generated-config"):
        checks = detail.get("checks") if isinstance(detail.get("checks"), dict) else {}
        failing = {key for key, value in checks.items() if value is False}
        if failing and failing <= {"runtime_metrics_and_status", "effective_mode", "capture_complete"}:
            return "harness-observation"
    if name.endswith((":warm-l1-path", ":l2-metrics")):
        metric_maps = [value for key, value in detail.items() if isinstance(value, dict) and key != "checks"]
        if metric_maps and not any(metric_maps):
            return "harness-observation"
    if name.endswith(":lifecycle"):
        checks = detail.get("checks") if isinstance(detail.get("checks"), dict) else {}
        failing = {key for key, value in checks.items() if value is False}
        if failing == {"generated_output_byte_equal"}:
            return "text-byte-inequality"
    return "cache-metadata"


# ---------------------------------------------------------------- helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def to_epoch(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def digest_of(image: str | None) -> str | None:
    if not image or "@sha256:" not in image:
        return image
    return image.split("@sha256:", 1)[1][:12]


def weights_tag(model_dir: str | Path | None) -> str | None:
    """Short arm tag for a non-published checkpoint (e.g. GLM-5.3-Flash-NVFP4-QAD-step2500 -> qad-step2500)."""
    if not model_dir:
        return None
    path = Path(model_dir)
    if path.resolve() == PUBLISHED_WEIGHTS.resolve():
        return None
    name = path.name
    if name.startswith(PUBLISHED_WEIGHTS.name + "-"):
        name = name[len(PUBLISHED_WEIGHTS.name) + 1:]
    return name.lower()


def arm_of(image: str | None, model_dir: str | Path | None = None) -> str:
    """Image arm, qualified by the mounted checkpoint; a candidate checkpoint is its own arm, never pooled."""
    base = ARMS.get(image or "", "unattributed")
    tag = weights_tag(model_dir)
    if tag is None:
        return base
    return tag if base == "official-r26" else f"{base}@{tag}"


class Root:
    """Read-only view of the battery root; the excluded initial pass is never opened."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.read_errors: list[dict[str, str]] = []
        self.sources: dict[str, dict[str, Any]] = {}

    def file(self, name: str) -> Path:
        path = self.path / name
        if EXCLUDED_DIR in path.parts:
            raise ValueError(f"refusing excluded path {path}")
        return path

    def json(self, name: str) -> dict[str, Any] | list | None:
        path = self.file(name)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            self.read_errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
            return None
        self.sources.setdefault(str(path), {"bytes": path.stat().st_size})
        return value

    def text(self, name: str) -> str | None:
        path = self.file(name)
        if not path.is_file():
            return None
        try:
            content = path.read_text(errors="replace")
        except OSError as error:
            self.read_errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
            return None
        self.sources.setdefault(str(path), {"bytes": path.stat().st_size})
        return content

    def mtime(self, name: str) -> float | None:
        path = self.file(name)
        return path.stat().st_mtime if path.is_file() else None

    def glob(self, pattern: str) -> list[Path]:
        return sorted(path for path in self.path.glob(pattern) if path.is_file() and not path.name.endswith(".tmp"))

    def window(self, stem: str) -> dict[str, Any] | None:
        """Command window written by runtime.run for ``stem``."""
        for name in (f"{stem}.command.json", f"{stem}.bench.command.json"):
            data = self.json(name)
            if isinstance(data, dict) and data.get("started_at") is not None:
                return {
                    "started_at": to_epoch(data.get("started_at")),
                    "finished_at": to_epoch(data.get("finished_at")),
                    "returncode": data.get("returncode"),
                    "source": str(self.file(name)),
                }
        return None

    def server_kv_tokens(self, boot_label: str) -> dict[str, Any] | None:
        for name in (f"{boot_label}.docker.log", f"{boot_label}-final.docker.log"):
            text = self.text(name)
            if text is None:
                continue
            match = SERVER_KV_RE.search(text)
            if match:
                return {"tokens": int(match.group(1).replace(",", "")), "source": str(self.file(name)), "line": match.group(0)[:160]}
        return None

    def finalize_sources(self) -> list[dict[str, Any]]:
        rows = []
        for path, meta in sorted(self.sources.items()):
            file = Path(path)
            if file.is_file():
                rows.append({"path": path, "bytes": meta["bytes"], "sha256": sha256_file(file)})
        return rows


# ---------------------------------------------------------------- GPU isolation timeline
class Isolation:
    """Merged GPU-isolation timeline: coordinator samples plus any follow-up stage samples."""

    def __init__(self, root: Root) -> None:
        self.files = [root.file("gpu-isolation-events.jsonl"), *(path for stage in SUBSTAGES for path in root.glob(f"{stage}/**/gpu-isolation-events.jsonl"))]
        samples: list[tuple[float, tuple[tuple[int, str], ...], bool]] = []
        self.bad_lines = 0
        for path in self.files:
            if not path.is_file():
                continue
            root.sources.setdefault(str(path), {"bytes": path.stat().st_size})
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                    ts = float(event["timestamp"])
                    foreign = tuple(sorted((int(row["pid"]), str(row["name"])) for row in event.get("foreign", [])))
                except (ValueError, KeyError, TypeError):
                    self.bad_lines += 1
                    continue
                samples.append((ts, foreign, event.get("speed_eligible", True) is True))
        samples.sort()
        self.timestamps = [ts for ts, _, _ in samples]
        self.foreign = [foreign for _, foreign, _ in samples]
        self.producer_eligible = [eligible for _, _, eligible in samples]
        self.interruption = root.json("gpu-isolation-interruption.json")
        self.monitor_error = root.json("gpu-isolation-monitor-error.json")

    def summary(self) -> dict[str, Any]:
        names = Counter()
        foreign_samples = 0
        vllm_named = []
        for ts, foreign in zip(self.timestamps, self.foreign):
            if foreign:
                foreign_samples += 1
                names.update(name for _, name in foreign)
                if any(name.startswith("VLLM::") for _, name in foreign):
                    vllm_named.append({"timestamp": iso(ts), "pids": [pid for pid, name in foreign if name.startswith("VLLM::")]})
        return {
            "events_files": [str(path) for path in self.files if path.is_file()],
            "present": bool(self.timestamps),
            "samples": len(self.timestamps),
            "bad_lines": self.bad_lines,
            "first_sample": iso(self.timestamps[0]) if self.timestamps else None,
            "last_sample": iso(self.timestamps[-1]) if self.timestamps else None,
            "samples_with_foreign_gpu_process": foreign_samples,
            "producer_ineligible_samples": sum(not eligible for eligible in self.producer_eligible),
            "foreign_process_names": dict(names),
            # vLLM worker pids appear 'foreign' for a sample or two while the owned container restarts (docker restart
            # drops it from `docker ps` before its workers exit); they still contaminate any window they touch
            "vllm_named_foreign_samples": vllm_named[:50],
            "vllm_named_foreign_sample_count": len(vllm_named),
            "interruption": self.interruption,
            "monitor_error": self.monitor_error,
        }

    def verdict(self, window: dict[str, Any] | None) -> dict[str, Any]:
        if not window or window.get("started_at") is None or window.get("finished_at") is None:
            return {"status": "no-window", "coverage_complete": False, "foreign_overlap": None, "foreign": []}
        start, end = window["started_at"], window["finished_at"]
        if not self.timestamps:
            return {"status": "coverage-gap", "coverage_complete": False, "foreign_overlap": None, "foreign": [], "reason": "no isolation samples recorded"}
        pad = SAMPLE_CADENCE_SECONDS * 2
        lo = bisect.bisect_left(self.timestamps, start - pad)
        hi = bisect.bisect_right(self.timestamps, end + pad)
        inside = self.timestamps[lo:hi]
        if not inside:
            return {"status": "coverage-gap", "coverage_complete": False, "foreign_overlap": None, "foreign": [], "reason": "no isolation samples inside the window"}
        gaps = [inside[0] - start, end - inside[-1]] + [b - a for a, b in zip(inside, inside[1:])]
        max_gap = max(gaps)
        coverage_complete = max_gap <= MAX_SAMPLE_GAP_SECONDS
        foreign: Counter = Counter()
        foreign_samples = 0
        ineligible_samples = sum(not eligible for eligible in self.producer_eligible[lo:hi])
        for index in range(lo, hi):
            if self.foreign[index]:
                foreign_samples += 1
                foreign.update(f"{name}[{pid}]" for pid, name in self.foreign[index])
        if foreign_samples:
            status = "foreign-gpu-overlap"
        elif ineligible_samples:
            status = "gpu-health-fault"
        elif not coverage_complete:
            status = "coverage-gap"
        else:
            status = "clean"
        return {
            "status": status,
            "coverage_complete": coverage_complete,
            "foreign_overlap": bool(foreign_samples),
            "samples": len(inside),
            "foreign_samples": foreign_samples,
            "producer_ineligible_samples": ineligible_samples,
            "foreign": sorted(foreign),
            "max_sample_gap_seconds": round(max_gap, 1),
        }


# ---------------------------------------------------------------- gates
GATE_FAILURE_CLASS = (
    ("boot:", "runtime-protocol"),
    ("runtime:", "runtime-protocol"),
    ("harness:", "harness"),
    ("phase-execution:", "runtime-protocol"),
    ("benchmark-execution:", "runtime-protocol"),
    ("profile-execution:", "runtime-protocol"),
    ("matrix-cell:", "runtime-protocol"),
    ("acceptance-window:", "performance-window"),
    ("configuration:", "configuration"),
    ("cache:", "cache-metadata"),
    ("quality:", "model-quality"),
    ("scheduler-", "scheduler"),
    ("peer599:", "hardware-599"),
    ("runtime-count:", "runtime-protocol"),
    ("overlay-quality-recheck:", "harness"),
    ("release-history", "history"),
)


def load_gates(root: Root) -> list[dict[str, Any]]:
    """Coordinator gates first, then follow-up stage gates; every gate is tagged with its stage."""
    gates = []
    files = [("root", root.file("gates.jsonl"))] + [(stage, path) for stage in SUBSTAGES for path in root.glob(f"{stage}/**/gates.jsonl")]
    for stage, path in files:
        text = root.text(str(path.relative_to(root.path))) or ""
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                gate = json.loads(line)
            except ValueError:
                root.read_errors.append({"path": str(path), "error": f"line {number} is not JSON"})
                continue
            gate["line"] = number
            gate["stage"] = stage
            gate["file"] = str(path)
            gates.append(gate)
    return gates


def gate_category(name: str) -> str:
    for prefix, category in GATE_FAILURE_CLASS:
        if name.startswith(prefix):
            return category
    return "other"


class Gates:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.by_stage: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:  # within a stage the last record for a name wins; earlier ones stay in rows
            self.by_stage.setdefault(row["stage"], {})[row["name"]] = row

    def passed(self, name: str, stage: str = "root") -> bool | None:
        row = self.by_stage.get(stage, {}).get(name)
        return None if row is None else bool(row.get("passed"))

    def detail(self, name: str, stage: str = "root") -> Any:
        row = self.by_stage.get(stage, {}).get(name)
        return None if row is None else row.get("detail")

    def names(self, stage: str = "root") -> list[str]:
        return list(self.by_stage.get(stage, {}))

    def with_prefix(self, prefix: str, stage: str = "root") -> list[dict[str, Any]]:
        return [row for row in self.rows if row["stage"] == stage and row["name"].startswith(prefix)]


def recheck_map(root: Root) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact shape written by overlay_quality_recheck.py (ROOT/followups/overlay-quality-recheck-map.json):

    ``{original_attempts_preserved: true, rechecks: [{original_arm, recheck_arm, image, reason, native_args,
    receipts: {'<old>.json': '<new>.json'}, booted?, error?}]}`` — old names resolve against ROOT, new names
    against ROOT/followups.  Returns ``{"arms": {original_arm: entry}, "receipts": {old: followups/new}}``.
    """
    path = str(root.file(RECHECK_MAP))
    data = root.json(RECHECK_MAP)
    if data is None:
        return {"arms": {}, "receipts": {}}, {"present": False, "path": path}
    if not isinstance(data, dict) or not isinstance(data.get("rechecks"), list):
        root.read_errors.append({"path": path, "error": "recheck map is not {rechecks: [...]}"})
        return {"arms": {}, "receipts": {}}, {"present": True, "path": path, "valid": False}
    arms: dict[str, dict[str, Any]] = {}
    receipts: dict[str, str] = {}
    for entry in data["rechecks"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("original_arm"), str):
            root.read_errors.append({"path": path, "error": f"recheck entry without original_arm: {str(entry)[:120]}"})
            continue
        arms[entry["original_arm"]] = entry
        for old, new in (entry.get("receipts") or {}).items():
            if isinstance(old, str) and isinstance(new, str):
                receipts[old] = f"{FOLLOWUPS_DIR}/{new}"
    return {"arms": arms, "receipts": receipts}, {
        "present": True, "path": path, "valid": True, "original_attempts_preserved": data.get("original_attempts_preserved"),
        "arms": {name: {"recheck_arm": entry.get("recheck_arm"), "booted": entry.get("booted"), "error": entry.get("error"), "receipts": len(entry.get("receipts") or {})} for name, entry in arms.items()},
        "unreadable_new_receipts": [new for new in receipts.values() if root.json(new) is None],
    }


def label_of(value: str) -> str:
    return Path(value).name.removesuffix(".json")


def stage_prefix(stage: str) -> str:
    return "" if stage == "root" else f"{stage}/"


# ---------------------------------------------------------------- plans
def root_weights(root: Root) -> dict[str, Any]:
    """Checkpoint mounted by default in this root: run_qad_battery.py writes checkpoint-verification.json before booting."""
    verification = root.json("checkpoint-verification.json")
    if isinstance(verification, dict) and verification.get("model_dir"):
        return {"model_dir": verification["model_dir"], "tag": weights_tag(verification["model_dir"]), "verification": verification, "kind": "candidate-checkpoint"}
    return {"model_dir": str(PUBLISHED_WEIGHTS), "tag": None, "verification": None, "kind": "published"}


def normalized_phases(value: object, source: Path) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source}: phase plan is not a list")
    phases: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            required = ("name", "script", "args", "timeout_seconds")
            if not all(key in item for key in required):
                raise ValueError(f"{source}: phase {index} lacks {required}")
            phase = {key: item[key] for key in required}
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, script, args, timeout = item
            phase = {"name": name, "script": script, "args": args, "timeout_seconds": timeout}
        else:
            raise ValueError(f"{source}: phase {index} has an unsupported shape")
        if (
            not isinstance(phase["name"], str)
            or not isinstance(phase["script"], str)
            or not isinstance(phase["args"], list)
            or not isinstance(phase["timeout_seconds"], int)
        ):
            raise ValueError(f"{source}: phase {index} has invalid field types")
        phases.append(phase)
    return phases


def source_assigned_phases(script: str) -> tuple[list[dict[str, Any]], str]:
    """Read the literal coordinator.PHASES assignment from a driver source."""
    source = HERE / script
    tree = ast.parse(source.read_text(), filename=str(source))
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "coordinator"
            and target.attr == "PHASES"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError(f"{source}: expected exactly one coordinator.PHASES assignment")
    phases = normalized_phases(ast.literal_eval(assignments[0].value), source)
    return phases, f"{source} coordinator.PHASES (source SHA-256 {sha256_file(source)})"


def phase_plan(root: Root) -> tuple[list[dict[str, Any]], str]:
    plan = root.json("phase-plan.json")
    if isinstance(plan, list) and plan:
        return normalized_phases(plan, root.file("phase-plan.json")), str(root.file("phase-plan.json"))
    if root_weights(root)["kind"] == "candidate-checkpoint":
        phases, source = source_assigned_phases("run_qad_battery.py")
        return phases, f"{source}; phase-plan.json absent"
    import run_qualification

    return (
        normalized_phases(run_qualification.PHASES, HERE / "run_qualification.py"),
        "scripts/r26/run_qualification.py PHASES (phase-plan.json absent)",
    )


RECHECK_SUFFIX = "-native-recheck"


def followup_phase_plan(root: Root) -> tuple[list[dict[str, Any]], str]:
    executed_name = f"{FOLLOWUPS_DIR}/executed-followup-plan.json"
    executed = root.json(executed_name)
    if isinstance(executed, dict):
        if executed.get("schema") != FOLLOWUP_PLAN_SCHEMA:
            root.read_errors.append({"path": str(root.file(executed_name)), "error": f"expected schema {FOLLOWUP_PLAN_SCHEMA}"})
        elif isinstance(executed.get("phases"), list):
            return normalized_phases(executed["phases"], root.file(executed_name)), str(root.file(executed_name))
    stage_name = f"{FOLLOWUPS_DIR}/phase-plan.json"
    stage_plan = root.json(stage_name)
    if isinstance(stage_plan, list) and stage_plan:
        return normalized_phases(stage_plan, root.file(stage_name)), str(root.file(stage_name))
    queued = root.json(FOLLOWUP_EXECUTION_PLAN)
    if isinstance(queued, dict):
        if queued.get("schema") != FOLLOWUP_PLAN_SCHEMA:
            root.read_errors.append({"path": str(root.file(FOLLOWUP_EXECUTION_PLAN)), "error": f"expected schema {FOLLOWUP_PLAN_SCHEMA}"})
        elif isinstance(queued.get("phases"), list):
            return normalized_phases(queued["phases"], root.file(FOLLOWUP_EXECUTION_PLAN)), str(root.file(FOLLOWUP_EXECUTION_PLAN))
    return [], "followup execution plan unavailable; no phase names inferred"


def clean_phase_plan(root: Root) -> tuple[list[dict[str, Any]], str]:
    name = f"{CLEAN_RERUNS_DIR}/phase-plan.json"
    plan = root.json(name)
    if isinstance(plan, list) and plan:
        return normalized_phases(plan, root.file(name)), str(root.file(name))
    phases, source = source_assigned_phases("run_clean_reruns.py")
    return phases, f"{source}; {name} absent"


def realistic_arms() -> list[dict[str, Any]]:
    import realistic_acceptance_phase

    return [{"name": arm.name, "image": arm.image, "draft_head": arm.draft_head, "description": arm.description,
             "extra_env": realistic_acceptance_phase.arm_extra_env(arm), "extra_args": list(arm.extra_args)}
            for arm in realistic_acceptance_phase.arm_plan(rt)]


def quality_arms(root: Root) -> tuple[list[dict[str, Any]], str]:
    plan = root.json("quality-plan.json")
    if isinstance(plan, dict) and isinstance(plan.get("arms"), list):
        arms = []
        for arm in plan["arms"]:
            acceptance = arm.get("acceptance_probe") if isinstance(arm.get("acceptance_probe"), dict) else None
            arms.append({
                "label": arm["label"], "image": arm.get("image"), "role": arm.get("role"), "runtime": arm.get("runtime") or {},
                "acceptance_probe": {key: acceptance.get(key) for key in ("contexts", "concurrency", "repeats")} if acceptance else None,
                "probes": [{"label": probe["label"], "suite": probe["suite"]} for probe in arm.get("probes", [])],
                "profiles": [{"label": profile["label"], "profile": profile.get("profile"), "runs": profile.get("runs")} for profile in arm.get("profiles", [])],
            })
        return arms, str(root.file("quality-plan.json"))
    import quality_phase

    arms = []
    for arm in quality_phase.build_plan(rt):
        manifest = quality_phase.arm_manifest(arm)
        acceptance = manifest.get("acceptance_probe")
        arms.append({
            "label": arm.label, "image": arm.image, "role": arm.role, "runtime": manifest["runtime"],
            "acceptance_probe": {key: acceptance.get(key) for key in ("contexts", "concurrency", "repeats")} if acceptance else None,
            "probes": [{"label": probe.label, "suite": probe.suite} for probe in arm.probes],
            "profiles": [{"label": profile.label, "profile": profile.profile, "runs": profile.runs} for profile in arm.profiles],
        })
    return arms, "scripts/r26/quality_phase.py build_plan (quality-plan.json absent)"


LAUNCH_DEFAULTS = {"MAX_NUM_BATCHED_TOKENS": "4096", "FAIRNESS_ENGINE": "compute_share", "DCP_CKV_GATHER": "auto"}
LAUNCH_INTERESTING = ("VLLM_GLM53_MTP_DRAFT_HEAD", "MAX_NUM_BATCHED_TOKENS", "VLLM_PCIE_DMA_MIN_BYTES", "FAIRNESS_ENGINE",
                      "GLM53_KDA_PREFILL_BACKEND", "DCP_CKV_GATHER", "VLLM_USE_DIRECT_DCP_A2A", "NATIVE_KV_OFFLOADING_SIZE_GB", "LMCACHE_L2_MAX_CAPACITY_GB")


def launch_overrides(env: dict[str, Any]) -> dict[str, Any]:
    """Launcher settings that deviate from runtime.boot defaults or select an experimental knob."""
    return {key: value for key, value in env.items() if key in LAUNCH_INTERESTING and LAUNCH_DEFAULTS.get(key) != str(value)}


def bench_options(args: list[str]) -> dict[str, Any]:
    """Bench cell options recorded in a runtime.bench command receipt (mirrors clean_rerun_phase.bench_options)."""
    options: dict[str, Any] = {}
    for flag, key in (("--concurrency", "conc"), ("--contexts", "contexts"), ("--duration", "duration")):
        if flag in args:
            options[key] = args[args.index(flag) + 1]
    options["duration"] = int(options.get("duration", 30))
    options["prefill"] = "--skip-prefill" not in args
    return options


class _Recorder:
    """Stand-in runtime that records the cells matrix_phase would boot and bench."""

    IMAGE, R25_IMAGE, OVERLAY_IMAGE = rt.IMAGE, rt.R25_IMAGE, rt.OVERLAY_IMAGE

    def __init__(self) -> None:
        self.boots: list[dict[str, Any]] = []
        self.benches: list[dict[str, Any]] = []
        self._current: str | None = None

    def boot(self, label: str, **kwargs: Any) -> bool:
        self.boots.append({"label": label, **kwargs})
        self._current = label
        return True

    def bench(self, label: str, **kwargs: Any) -> bool:
        self.benches.append({"label": label, "boot_label": self._current, **kwargs})
        return True

    def stop(self) -> None:
        self._current = None

    def record_gate(self, *args: Any, **kwargs: Any) -> None:
        pass

    def save_json(self, *args: Any, **kwargs: Any) -> None:
        pass

    def note(self, *args: Any, **kwargs: Any) -> None:
        pass


def matrix_section_plan(section: str) -> dict[str, list[dict[str, Any]]]:
    import matrix_phase

    recorder = _Recorder()
    original = matrix_phase.rt
    matrix_phase.rt = recorder
    try:
        getattr(matrix_phase, section)()
    finally:
        matrix_phase.rt = original
    return {"boots": recorder.boots, "benches": recorder.benches}


def scheduler_plan(root: Root) -> tuple[list[dict[str, Any]], str]:
    plan = root.json("scheduler/phase-plan.json")
    source = str(root.file("scheduler/phase-plan.json"))
    if not isinstance(plan, dict):
        import scheduler_phase

        plan = scheduler_phase.phase_plan()
        source = "scripts/r26/scheduler_phase.py phase_plan (scheduler/phase-plan.json absent)"
    overlay_label = (plan.get("mixed_factor_contract") or {}).get("candidate")
    cells = []
    for entry in plan.get("vram_plans", []):
        cells.append({"name": entry["name"], "series": entry.get("series"), "image": rt.IMAGE if entry["name"].startswith("official-") else rt.OVERLAY_IMAGE, "scenario_count": entry.get("scenario_count")})
    for factor in plan.get("mixed_factors", []):
        entry = factor.get("plan") or {}
        cells.append({"name": entry.get("name") or factor.get("name"), "series": "mixed-factor", "image": rt.OVERLAY_IMAGE if overlay_label else None, "scenario_count": entry.get("scenario_count"), "factor": {"batch_tokens": factor.get("batch_tokens"), "dma_value": factor.get("dma_value")}})
    for entry in (plan.get("lmcache_confirmation") or {}).get("plans", []):
        cells.append({"name": entry["name"], "series": "lmcache-confirmation", "image": rt.IMAGE if entry["name"].startswith("official-") else rt.OVERLAY_IMAGE, "scenario_count": entry.get("scenario_count")})
    return cells, source


# ---------------------------------------------------------------- row construction
class Report:
    def __init__(self, root: Root) -> None:
        self.root = root
        self.gates = Gates(load_gates(root))
        self.isolation = Isolation(root)
        self.rechecks, self.recheck_map_summary = recheck_map(root)
        self.rows: list[dict[str, Any]] = []
        self.phase_windows: dict[str, dict[str, Any]] = {}
        self.phase_stages: dict[str, str] = {}
        self.plan_sources: dict[str, str] = {}
        self.coverage_gaps: list[str] = []
        self.weights = root_weights(root)
        proof = root.json(REASONING_PROTOCOL_PROOF)
        self.reasoning_proof = proof if isinstance(proof, dict) else None
        self.superseded_gates: dict[tuple[str, str], str] = {}
        self.cache_recheck_boot_labels: set[str] = set()
        self.cache_recheck_groups_available: set[str] = set()

    def arm(self, image: str | None, model_dir: str | None = None) -> str:
        """Arm for an image booted with ``model_dir`` (a launch receipt's value) or this root's default checkpoint."""
        return arm_of(image, model_dir or self.weights["model_dir"])

    # -- generic pieces
    def row(self, **fields: Any) -> dict[str, Any]:
        base = {
            "phase": None, "cell": None, "kind": None, "arm": "unattributed", "image": None, "config": None,
            "status": "missing", "failure_classes": [], "outcomes": None, "speed": None, "speed_qualification": None,
            "window": None, "isolation": None, "kv_tokens_server": None, "kv_tokens_bench": None,
            "stage": "root", "source_state": None, "recheck_of": None, "superseded_by": None,
            "superseded_checks": [], "sources": [], "gates": [], "notes": [],
        }
        base.update(fields)
        if base["window"]:
            base["window"] = {**base["window"], "started_at_iso": iso(base["window"].get("started_at")), "finished_at_iso": iso(base["window"].get("finished_at"))}
        if base["speed"] is not None:
            verdict = self.isolation.verdict(base["window"])
            base["isolation"] = verdict
            base["speed_qualification"] = verdict["status"]
        elif base["window"]:
            base["isolation"] = self.isolation.verdict(base["window"])
        self.rows.append(base)
        return base

    def phase_state(self, name: str) -> str:
        window = self.phase_windows.get(name)
        if window and window.get("finished_at") is not None:
            return "finished" if window.get("returncode") == 0 else f"finished-rc{window.get('returncode')}"
        stage = self.phase_stages.get(name, "root")
        current = self.root.json(f"{stage_prefix(stage)}current-phase.json")
        if isinstance(current, dict) and current.get("phase") == name:
            return "running"
        if stage != "root" and not self.root.file(f"{stage_prefix(stage)}phase-plan.json").is_file():
            return "queued"
        return "not-started"

    def boot_row(self, phase: str, label: str, *, expected_image: str | None = None, expected: dict[str, Any] | None = None, stage: str = "root",
                 expected_model_dir: str | None = None) -> dict[str, Any]:
        prefix = stage_prefix(stage)
        launch = self.root.json(f"{prefix}{label}.launch.json")
        passed = self.gates.passed(f"boot:{label}", stage)
        kv = self.root.server_kv_tokens(f"{prefix}{label}")
        sources = [str(self.root.file(f"{prefix}{label}.launch.json"))] if launch else []
        if kv:
            sources.append(kv["source"])
        notes: list[str] = []
        outcomes: dict[str, Any] = {"boot_gate": self.gates.detail(f"boot:{label}", stage)}
        classes: list[str] = []
        model_dir = expected_model_dir
        if isinstance(launch, dict):
            image = launch.get("image")
            model_dir = launch.get("model_dir") or model_dir
            config = {key: launch.get(key) for key in ("tp", "dcp", "spec", "cache", "kv")}
            config["extra_env"] = launch_overrides(launch.get("env") or {})
            if launch.get("extra_args"):
                config["extra_args"] = launch["extra_args"]
            if launch.get("model_dir"):
                config["model_dir"] = launch["model_dir"]
            status = "pass" if passed else ("fail" if passed is False else "running")
            if expected_image and image != expected_image:
                notes.append(f"launch image differs from planned image {digest_of(expected_image)}")
            if expected_model_dir and launch.get("model_dir") and Path(launch["model_dir"]).resolve() != Path(expected_model_dir).resolve():
                notes.append(f"launch model_dir {launch['model_dir']} differs from planned {expected_model_dir}")
        else:
            image, config, status = expected_image, expected, ("fail" if passed is False else "missing")
        if status == "fail":
            errors = self.server_log_errors(f"{prefix}{label}")
            outcomes["server_log_errors"] = errors
            if any("unrecognized arguments" in line for line in errors):
                classes.append("launch-config-incompatible")
                notes.append("the image rejected a launcher argument; this is a launch-configuration incompatibility, not a model result")
            else:
                classes.append("runtime-protocol")
        return self.row(phase=phase, cell=label, kind="boot", arm=self.arm(image, model_dir), image=digest_of(image), config=config, status=status, stage=stage,
                        failure_classes=classes, outcomes=outcomes, window=self.root.window(f"{prefix}{label}.boot"),
                        kv_tokens_server=kv["tokens"] if kv else None, sources=sources, gates=[f"boot:{label}"], notes=notes)

    def server_log_errors(self, label: str) -> list[str]:
        for name in (f"{label}-final.docker.log", f"{label}.docker.log"):
            text = self.root.text(name)
            if text is None:
                continue
            hits = [line[:240] for line in text.splitlines() if re.search(r"\berror\b|Traceback|CUDA out of memory|Killed|exited", line)]
            if hits:
                return hits[-4:]
        return []

    # -- quality phase (coordinator stage) and its follow-up native rechecks
    def quality_rows(self, phase: str) -> None:
        arms, source = quality_arms(self.root)
        self.plan_sources[phase] = source
        for arm in arms:
            self.quality_arm_rows(phase, arm, self.phase_stages[phase])

    def quality_arm_rows(self, phase: str, arm: dict[str, Any], stage: str, suffix: str = "", model_dir: str | None = None) -> None:
        label, image = arm["label"] + suffix, arm["image"]
        boot = self.boot_row(phase, label, expected_image=image, expected=arm["runtime"], stage=stage, expected_model_dir=model_dir)
        boot_failed = boot["status"] == "fail" or (stage != "root" and boot["status"] == "missing" and self.phase_state(phase).startswith("finished"))
        common = {"stage": stage, "boot_failed": boot_failed, "arm": boot["arm"]}
        if arm["acceptance_probe"]:
            self.synthetic_acceptance_rows(phase, label, image, arm["acceptance_probe"], **common)
        for probe in arm["probes"]:
            self.quality_probe_row(phase, probe["label"] + suffix, probe["suite"], image, **common)
        for profile in arm["profiles"]:
            self.quality_profile_row(phase, {**profile, "label": profile["label"] + suffix}, image, **common)

    def recheck_rows(self, phase: str) -> None:
        """overlay_quality_recheck.py main(): native overlay arms, then semantic controls."""
        arms, _ = quality_arms(self.root)
        self.plan_sources[phase] = (
            "scripts/r26/overlay_quality_recheck.py: quality_phase.build_plan overlay arms "
            "+ '-native-recheck', then semantic_controls(); cache work is a separate "
            "cache_recheck_phase.py phase"
        )
        stage = self.phase_stages[phase]
        for arm in arms:
            if arm["image"] == rt.OVERLAY_IMAGE:
                self.quality_arm_rows(phase, arm, stage, RECHECK_SUFFIX)
        self.semantic_control_rows(phase, stage)

    def resolve_rechecks(self) -> None:
        """Redirect launch-dependent originals to their native rechecks, only where the new receipt is readable."""
        for old, new in self.rechecks["receipts"].items():
            if self.root.json(new) is None:
                self.coverage_gaps.append(f"recheck map names {new} for {old} but it is absent or unreadable; the original stays unresolved")
                continue
            self.supersede(self.rows_for_receipt(label_of(old), "root"), self.rows_for_receipt(label_of(new), FOLLOWUPS_DIR), old, new)
        for original_arm, entry in self.rechecks["arms"].items():
            recheck_arm = entry.get("recheck_arm")
            launch = f"{FOLLOWUPS_DIR}/{recheck_arm}.launch.json"
            if not isinstance(recheck_arm, str) or self.root.json(launch) is None:
                continue
            originals = [row for row in self.rows if row["stage"] == "root" and row["kind"] == "boot" and row["cell"] == original_arm]
            rechecks = [row for row in self.rows if row["stage"] == FOLLOWUPS_DIR and row["kind"] == "boot" and row["cell"] == recheck_arm]
            self.supersede(originals, rechecks, f"{original_arm}.launch.json", launch)
        self.resolve_cache_rechecks()

    def resolve_cache_rechecks(self) -> None:
        """Make the fresh-root corrected methodology authoritative without deleting old attempts."""
        for cell, source_keys in CACHE_RECHECK_GROUPS.items():
            originals = [
                row for row in self.rows
                if row["stage"] == "root" and row["cell"] == f"cache:{cell}"
            ]
            rechecks = [
                row for row in self.rows
                if row["stage"] == FOLLOWUPS_DIR and row["cell"] == f"cache:{cell}"
            ]
            source = f"{CACHE_RECHECK_SUMMARY}#cells/{','.join(source_keys)}"
            original_source_cell = "native" if cell == "native-canaries" else cell
            if cell in self.cache_recheck_groups_available:
                self.supersede(originals, rechecks, f"cache-phase-summary.json#cells/{original_source_cell}", source)
            else:
                for row in originals:
                    row["outcomes"] = {
                        **(row["outcomes"] or {}),
                        "corrected_recheck_state": "source unavailable",
                        "expected_source": str(self.root.file(CACHE_RECHECK_SUMMARY)),
                        "expected_source_cells": list(source_keys),
                    }
                    row["notes"] = [
                        *row["notes"],
                        "corrected recheck source is unavailable; original attempt remains authoritative and is not relabelled",
                    ]
        for label in sorted(self.cache_recheck_boot_labels):
            originals = [
                row for row in self.rows
                if row["stage"] == "root" and row["kind"] == "boot" and row["cell"] == label
            ]
            rechecks = [
                row for row in self.rows
                if row["stage"] == FOLLOWUPS_DIR and row["kind"] == "boot" and row["cell"] == label
            ]
            if originals and rechecks:
                self.supersede(
                    originals,
                    rechecks,
                    f"{label}.launch.json",
                    f"{FOLLOWUPS_DIR}/{label}.launch.json",
                )

    def rows_for_receipt(self, label: str, stage: str) -> list[dict[str, Any]]:
        if label.endswith("-acceptance"):  # one receipt fans out to per-sample rows
            return [row for row in self.rows if row["stage"] == stage and row["kind"] == "acceptance-window-synthetic" and row["cell"].startswith(label + "-c")]
        return [row for row in self.rows if row["stage"] == stage and row["cell"] == label and row["kind"] != "boot"]

    def supersede(self, originals: list[dict[str, Any]], rechecks: list[dict[str, Any]], old: str, new: str) -> None:
        if not originals or not rechecks:
            self.coverage_gaps.append(f"recheck mapping pairs {old} -> {new} but rows are missing on one side (originals {len(originals)}, rechecks {len(rechecks)})")
            return
        for row in rechecks:
            row["recheck_of"] = str(self.root.file(old))
            row["notes"] = [
                *row["notes"],
                "source-backed recheck of a retained original attempt; the replacement row carries the current verdict",
            ]
        for row in originals:
            row["outcomes"] = {
                **(row["outcomes"] or {}),
                "original_status": row["status"],
                "original_failure_classes": row["failure_classes"],
                "original_speed_observation": row["speed"],
                "original_speed_qualification": row["speed_qualification"],
                "original_isolation": row["isolation"],
            }
            row["status"] = "superseded"
            row["failure_classes"] = []
            row["speed"] = None
            row["speed_qualification"] = None
            row["superseded_by"] = str(self.root.file(new))
            row["notes"] = [
                *row["notes"],
                "superseded by a source-backed recheck; original attempt and observations remain in outcomes and source receipts",
            ]

    def synthetic_acceptance_rows(self, phase: str, label: str, image: str, settings: dict[str, Any], *, stage: str = "root", boot_failed: bool = False, arm: str | None = None) -> None:
        arm = arm or self.arm(image)
        prefix = stage_prefix(stage)
        receipt_name = f"{prefix}{label}-acceptance.json"
        receipt = self.root.json(receipt_name)
        # quality_phase labels the probe run '<label>-acceptance'; qad_matched_phase labels it '<label>-acceptance-run'
        stem_window = self.root.window(f"{prefix}{label}-acceptance") or self.root.window(f"{prefix}{label}-acceptance-run")
        scope = "synthetic stress/ceiling only: repeated fixed text, temperature 0, ignore_eos; not representative acceptance"
        gates = [f"acceptance-window:{label}", f"runtime:{label}-acceptance"]
        if not isinstance(receipt, dict) or not isinstance(receipt.get("samples"), list):
            planned = [(context, concurrency, repeat)
                       for repeat in range(1, int(settings.get("repeats") or 2) + 1)
                       for context in settings.get("contexts") or [0, 32768]
                       for concurrency in settings.get("concurrency") or [1, 8]]
            not_run = (isinstance(receipt, dict) and receipt.get("status") == "not_run") or (receipt is None and boot_failed)
            for context, concurrency, repeat in planned:
                self.row(phase=phase, cell=f"{label}-acceptance-c{concurrency}-ctx{context}-repeat{repeat}", kind="acceptance-window-synthetic", stage=stage,
                         arm=arm, image=digest_of(image), config={"context_tokens": context, "concurrency": concurrency, "repeat": repeat},
                         status="not-run" if not_run else ("missing" if receipt is None else "fail"),
                         failure_classes=[] if (receipt is None or not_run) else ["runtime-protocol"],
                         outcomes={"reason": receipt.get("reason")} if isinstance(receipt, dict) else ({"reason": "arm boot failed; probe never ran"} if not_run else None),
                         window=stem_window, gates=gates, notes=[scope])
            return
        for sample in receipt["samples"]:
            context, concurrency, repeat = sample.get("context_tokens_requested"), sample.get("concurrency"), sample.get("repeat")
            stem = f"{label}-acceptance-c{concurrency}-ctx{context}-repeat{repeat}"
            first = [stream.get("first_token_at") for stream in sample.get("streams", []) if isinstance(stream.get("first_token_at"), (int, float))]
            end = self.root.mtime(f"{prefix}{stem}.after.metrics.txt")
            if first and end:
                window = {"started_at": min(first), "finished_at": end, "source": f"streams.first_token_at .. mtime {prefix}{stem}.after.metrics.txt"}
            else:
                window = stem_window
            speed = None
            if sample.get("passed"):
                speed = {key: sample.get(key) for key in ("output_tokens_per_second", "acceptance_fraction", "accepted_draft_tokens_per_step", "emitted_tokens_per_verifier_step", "aggregate_verifier_steps_per_second", "elapsed_seconds")}
            self.row(phase=phase, cell=stem, kind="acceptance-window-synthetic", arm=arm, image=digest_of(image), stage=stage,
                     config={"context_tokens": context, "concurrency": concurrency, "repeat": repeat, "sampling": sample.get("sampling")},
                     status="pass" if sample.get("passed") else "fail", failure_classes=[] if sample.get("passed") else ["performance-window"],
                     outcomes={"error": sample.get("error"), "counter_delta": sample.get("counter_delta")}, speed=speed, window=window,
                     sources=[str(self.root.file(receipt_name))], gates=gates, notes=[scope])

    def quality_probe_row(self, phase: str, label: str, suite: str, image: str, *, stage: str = "root", boot_failed: bool = False, arm: str | None = None) -> None:
        arm = arm or self.arm(image)
        prefix = stage_prefix(stage)
        receipt_name = f"{prefix}{label}.json"
        receipt = self.root.json(receipt_name)
        window = self.root.window(f"{prefix}{label}")
        gate_names = [name for name in self.gates.names(stage) if name.split(":")[1:2] == [label]]
        kind = f"quality-probe:{suite}"
        if receipt is None:
            self.row(phase=phase, cell=label, kind=kind, arm=arm, image=digest_of(image), stage=stage, status="not-run" if boot_failed else "missing",
                     outcomes={"reason": "arm boot failed; probe never ran"} if boot_failed else None, window=window, gates=gate_names)
            return
        if receipt.get("status") == "not_run":
            self.row(phase=phase, cell=label, kind=kind, arm=arm, image=digest_of(image), stage=stage, status="not-run",
                     outcomes={"reason": receipt.get("reason"), "detail": receipt.get("detail")}, window=window,
                     sources=[str(self.root.file(receipt_name))], gates=gate_names,
                     notes=["probe never ran (boot or harness failure upstream); not a model-quality result"])
            return
        derived = quality_reclassify.classify_receipt(self.root.file(receipt_name), receipt)
        counts = dict(derived["derived_counts"])
        classes: list[str] = []
        if self.gates.passed(f"runtime:{label}", stage) is False or counts.get("runtime_error"):
            classes.append("runtime-protocol")
        if counts.get("wrong_final"):
            classes.append("wrong-final")
        if counts.get("repetition"):
            classes.append("repetition")
        if counts.get("parser_failure"):
            classes.append("parser-failure")
        if counts.get("budget_limited_incomplete") or counts.get("incomplete_no_visible_final"):
            classes.append("budget-limited")
        if counts.get("model_quality_flag"):
            classes.append("model-quality-flag")
        cache_gate = self.gates.passed(f"cache:{label}:cold-to-hit", stage)
        if cache_gate is False:
            classes.append("cache-metadata")
        hard = {"runtime-protocol", "wrong-final", "repetition", "parser-failure", "cache-metadata"}
        status = "fail" if hard & set(classes) else ("flagged" if classes else "pass")
        if self.gates.passed(f"runtime:{label}", stage) is None and window and window.get("finished_at") is None:
            status = "running"
        summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
        effects = receipt.get("cache_effects") if isinstance(receipt.get("cache_effects"), dict) else None
        outcomes = {
            "requested": summary.get("requested", summary.get("requested_turns")), "completed": summary.get("completed", summary.get("completed_turns")),
            "original_summary_categories": summary.get("categories"), "derived_categories": counts,
            "original_to_derived": derived["original_to_derived"], "suite_fingerprint": receipt.get("suite_fingerprint"),
            "cache_effect_gate": cache_gate,
            "cache_effects": {key: effects.get(key) for key in ("cold_compute_observed", "local_hit_observed_on_both_replays", "external_hit_observed_on_both_replays") if key in effects} if effects else None,
            "receipt_level": derived.get("receipt_level"),
        }
        notes = ["derived categories from quality_reclassify.py; raw receipt untouched"]
        if counts and set(counts) <= {"completed_unverified", "parser_pass"}:
            notes.append("suite has no objective answer key; pass means no repetition/wrong-answer/parser detector fired, not verified correctness")
        self.row(phase=phase, cell=label, kind=kind, arm=arm, image=digest_of(image), stage=stage, status=status, failure_classes=classes,
                 outcomes=outcomes, window=window, sources=[str(self.root.file(receipt_name))], gates=gate_names, notes=notes)

    def quality_profile_row(self, phase: str, profile: dict[str, Any], image: str, *, stage: str = "root", boot_failed: bool = False, arm: str | None = None) -> None:
        arm = arm or self.arm(image)
        prefix = stage_prefix(stage)
        label = profile["label"]
        receipt_name = f"{prefix}{label}.json"
        receipt = self.root.json(receipt_name)
        window = self.root.window(f"{prefix}{label}")
        # quality_phase.run_profile records runtime:/quality: gates; runtime.profile (template-default controls) records profile-execution:
        gate_names = [name for name in self.gates.names(stage) if name in (f"runtime:{label}", f"profile-execution:{label}") or name.startswith(f"quality:{label}:")]
        kind = f"quality-profile:{profile.get('profile')}"
        if receipt is None:
            self.row(phase=phase, cell=label, kind=kind, arm=arm, image=digest_of(image), stage=stage, status="not-run" if boot_failed else "missing",
                     outcomes={"reason": "arm boot failed; profile never ran"} if boot_failed else None, window=window, gates=gate_names)
            return
        if receipt.get("status") == "not_run":
            self.row(phase=phase, cell=label, kind=kind, arm=arm, image=digest_of(image), stage=stage, status="not-run",
                     outcomes={"reason": receipt.get("reason")}, window=window, sources=[str(self.root.file(receipt_name))], gates=gate_names,
                     notes=["profile never ran (boot or harness failure upstream); not a model-quality result"])
            return
        summary = receipt.get("selected_summary") if isinstance(receipt.get("selected_summary"), dict) else {}
        metadata = receipt.get("metadata") if isinstance(receipt.get("metadata"), dict) else {}
        classes = []
        if self.gates.passed(f"runtime:{label}", stage) is False or self.gates.passed(f"profile-execution:{label}", stage) is False or summary.get("errors"):
            classes.append("runtime-protocol")
        if summary.get("wrong") or summary.get("wrong_answers"):
            classes.append("wrong-final")
        if summary.get("repetitions"):
            classes.append("repetition")
        if summary.get("score_available") is False:
            classes.append("unscored")
        derived = None
        if receipt.get("schema") == quality_reclassify.RAW_SCHEMA:
            derived = dict(quality_reclassify.classify_receipt(self.root.file(receipt_name), receipt)["derived_counts"])
            if derived.get("budget_limited_incomplete"):
                classes.append("budget-limited")
        status = "fail" if {"runtime-protocol", "wrong-final", "repetition"} & set(classes) else ("flagged" if classes else "pass")
        self.row(phase=phase, cell=label, kind=kind, arm=arm, image=digest_of(image), stage=stage, status=status, failure_classes=classes,
                 outcomes={"attempted": summary.get("attempted"), "completed": summary.get("completed"), "correct": summary.get("correct"), "wrong": summary.get("wrong"),
                           "errors": summary.get("errors"), "score_available": summary.get("score_available"), "runs_planned": profile.get("runs"), "derived_categories": derived,
                           "template_settings": {key: metadata.get(key) for key in ("test_profile", "requested_runs", "reasoning_effort", "max_tokens", "max_tokens_omitted", "temperature", "seed_base")},
                           "completion_tokens": summary.get("completion_tokens")},
                 window=window, sources=[str(self.root.file(receipt_name))], gates=gate_names)

    # -- realistic acceptance (realistic_acceptance_phase.py; follow-up stage in the R26 root, coordinator stage in the QAD root)
    def realistic_rows(self, phase: str) -> None:
        stage = self.phase_stages[phase]
        prefix = stage_prefix(stage)
        self.plan_sources[phase] = f"scripts/r26/realistic_acceptance_phase.py arm_plan; receipts {prefix}realistic-acceptance-<arm>.json"
        scope = "natural-completion chat acceptance on the fixed realistic input set, no ignore_eos; runtime/count integrity gate only, not a semantic-quality proof"
        for arm in realistic_arms():
            label = f"realistic-acceptance-{arm['name']}"
            boot = self.boot_row(phase, label, expected_image=arm["image"], stage=stage,
                                 expected={"tp": 4, "dcp": 4, "spec": "mtp3", "cache": "vram", "kv": "fp8_ds_mla", "extra_env": launch_overrides(arm["extra_env"]), "extra_args": arm["extra_args"] or None})
            receipt_name = f"{prefix}{label}.json"
            receipt = self.root.json(receipt_name)
            window = self.root.window(f"{prefix}{label}.probe")
            gate = f"runtime-count:realistic-acceptance:{arm['name']}"
            base = {"phase": phase, "kind": "acceptance-realistic", "arm": boot["arm"], "image": digest_of(arm["image"]), "stage": stage}
            if not isinstance(receipt, dict):
                boot_failed = boot["status"] == "fail" or (boot["status"] == "missing" and self.phase_state(phase).startswith("finished"))
                self.row(**base, cell=label, status="not-run" if boot_failed else "missing", window=window, gates=[gate],
                         outcomes={"reason": "arm boot failed; probe never ran"} if boot_failed else None, notes=[scope])
                continue
            summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
            integrity = summary.get("runtime_count_integrity") if isinstance(summary.get("runtime_count_integrity"), dict) else {}
            passed = self.gates.passed(gate, stage)
            groups = [group for group in (receipt.get("groups") or []) if isinstance(group, dict)]
            eligible = [group for group in groups if group.get("measurement_eligible") is True]
            classes: list[str] = []
            if receipt.get("schema") != REALISTIC_SCHEMA:
                classes.append("runtime-protocol")
            if passed is False or integrity.get("passed") is False or receipt.get("failure"):
                classes.append("runtime-protocol")
            status = "fail" if classes else ("pass" if passed else "running")
            pooled = summary.get("all_measurements") if isinstance(summary.get("all_measurements"), dict) else {}
            by_conc = summary.get("by_concurrency") if isinstance(summary.get("by_concurrency"), dict) else {}
            speed = None
            if eligible:
                speed = {"token_weighted_pooled_acceptance_fraction": pooled.get("token_weighted_pooled_acceptance_fraction"),
                         "accepted_draft_tokens_per_step": pooled.get("accepted_draft_tokens_per_step"), "emitted_tokens_per_verifier_step": pooled.get("emitted_tokens_per_verifier_step"),
                         "by_concurrency": {conc: {key: value.get(key) for key in ("token_weighted_pooled_acceptance_fraction", "emitted_tokens_per_verifier_step", "eligible_group_count")} for conc, value in by_conc.items() if isinstance(value, dict)}}
            arm_window = {"started_at": to_epoch(receipt.get("started_at")), "finished_at": to_epoch(receipt.get("finished_at")), "source": f"{receipt_name} started_at..finished_at"} if receipt.get("started_at") else window
            self.row(**base, cell=label, status=status, failure_classes=classes, config={"draft_head": arm["draft_head"], "extra_env": launch_overrides(arm["extra_env"]), "extra_args": arm["extra_args"], "input_set_sha256": (receipt.get("input_set") or {}).get("input_set_sha256") if isinstance(receipt.get("input_set"), dict) else None},
                     outcomes={"groups": len(groups), "eligible_groups": len(eligible), "integrity_issues": (integrity.get("issues") or [])[:8], "response_classifications": summary.get("response_classifications"), "failure": receipt.get("failure")},
                     speed=speed, window=arm_window, sources=[str(self.root.file(receipt_name))], gates=[gate], notes=[scope, "arm-level figures are token-weighted pools over measurement-eligible groups only"])
            for group in groups:
                requests = group.get("requests") if isinstance(group.get("requests"), list) else []
                errors = [request.get("error") for request in requests if isinstance(request, dict) and request.get("error")]
                eligible_group = group.get("measurement_eligible") is True
                group_window = {"started_at": to_epoch(group.get("started_at")), "finished_at": to_epoch(group.get("finished_at")), "source": f"{receipt_name} groups[{group.get('group_id')}]"}
                self.row(**base, cell=f"{label}:{group.get('group_id')}", status="pass" if eligible_group else "fail", failure_classes=[] if eligible_group else ["runtime-protocol"],
                         config={"concurrency": group.get("concurrency"), "domain": group.get("domain"), "seed": group.get("seed"), "prompt_ids": group.get("prompt_ids")},
                         outcomes={"requests": len(requests), "errors": errors[:5], "finish_reasons": dict(Counter(str(request.get("finish_reason")) for request in requests if isinstance(request, dict))),
                                   "classifications": dict(Counter(str(request.get("classification")) for request in requests if isinstance(request, dict))), "counter_delta": group.get("counter_delta")},
                         speed={key: group.get(key) for key in ("acceptance_fraction", "accepted_draft_tokens_per_step", "emitted_tokens_per_verifier_step")} if eligible_group else None,
                         window=group_window, sources=[str(self.root.file(receipt_name))], notes=[scope])

    # -- cache phase
    def cache_rows(self, phase: str) -> None:
        self.plan_sources[phase] = "scripts/r26/cache_phase.py CachePhase.main/finish cell names; retained pre-schema roots explicitly map legacy cells.native to v2 cells.native-canaries"
        summary = self.root.json("cache-phase-summary.json")
        summary_cells = summary.get("cells") if isinstance(summary, dict) and isinstance(summary.get("cells"), dict) else {}
        if isinstance(summary, dict) and summary.get("schema") == CACHE_PHASE_SCHEMA:
            attempted = set(summary_cells)
        elif isinstance(summary, dict) and "schema" not in summary:
            attempted = set(summary_cells)
            if "native" in attempted:
                attempted.remove("native")
                attempted.add("native-canaries")
        else:
            attempted = set()
            if summary is not None:
                self.root.read_errors.append({
                    "path": str(self.root.file("cache-phase-summary.json")),
                    "error": f"expected {CACHE_PHASE_SCHEMA} or retained pre-schema summary",
                })
        # cache_phase boot labels depend on run-time shapes/sizes; every cache-*.launch.json is a cache-phase boot
        for path in self.root.glob("cache-*.launch.json"):
            self.boot_row(phase, path.name[: -len(".launch.json")])
        self.cache_cell_rows(phase, "root", CACHE_CELLS, attempted, [str(self.root.file("cache-phase-summary.json"))] if summary else [])
        # speed-bearing cache sub-cells (planned in cache_phase.prefill_overhead / alternating_release_control) get
        # their own command windows so contamination is judged per request, and absence is reported as missing
        for arm in CACHE_PREFILL_ARMS:
            stem = f"cache-prefill-overhead-{arm}"
            receipt = self.root.json(f"{stem}.json")
            summary_block = receipt.get("summary") if isinstance(receipt, dict) and isinstance(receipt.get("summary"), dict) else {}
            self.row(phase=phase, cell=stem, kind="cache-speed", arm=self.arm(rt.IMAGE),
                     status="missing" if receipt is None else ("pass" if receipt.get("passed") else "fail"),
                     failure_classes=[] if (receipt is None or receipt.get("passed")) else ["cache-metadata"],
                     speed={key: summary_block.get(key) for key in ("steady_median_tokens_per_second", "median_tokens_per_second", "aggregate_tokens_per_second", "first_request_seconds")} if receipt else None,
                     window=self.root.window(stem), sources=[str(self.root.file(f"{stem}.json"))] if receipt else [])
        for index, (release, step) in enumerate(CACHE_ALTERNATING_SEQUENCE):
            stem = f"cache-alternating-{index:02d}-{release}-{step}"
            receipt = self.root.json(f"{stem}.json")
            self.row(phase=phase, cell=stem, kind="cache-speed", arm=self.arm(rt.R25_IMAGE if release == "r25" else rt.IMAGE),
                     status="missing" if receipt is None else ("pass" if receipt.get("passed") else "fail"),
                     failure_classes=[] if (receipt is None or receipt.get("passed")) else ["cache-metadata"],
                     speed={"api_wall_seconds": receipt.get("wall_seconds")} if isinstance(receipt, dict) else None,
                     outcomes={"lmcache_hit_tokens": (receipt.get("summary") or {}).get("lmcache_hit_tokens")} if isinstance(receipt, dict) and isinstance(receipt.get("summary"), dict) else None,
                     window=self.root.window(stem), sources=[str(self.root.file(f"{stem}.json"))] if receipt else [],
                     notes=["exact-1M shared-L2 alternating step; api wall seconds are speed-class, output hash equality is functional"])

    def cache_recheck_rows(self, phase: str) -> None:
        """Normalize the fresh-root cache_recheck_phase.py contract and its exact source cells."""
        stage = self.phase_stages[phase]
        summary = self.root.json(CACHE_RECHECK_SUMMARY)
        complete = self.root.json(CACHE_RECHECK_COMPLETE)
        self.plan_sources[phase] = (
            f"scripts/r26/cache_recheck_phase.py EXPECTED_RECHECK_CELLS; "
            f"{CACHE_RECHECK_SUMMARY} ({CACHE_RECHECK_SCHEMA})"
        )
        summary_schema_ok = isinstance(summary, dict) and summary.get("schema") == CACHE_RECHECK_SCHEMA
        completion_schema_ok = isinstance(complete, dict) and complete.get("schema") == CACHE_RECHECK_SCHEMA
        if isinstance(summary, dict) and not summary_schema_ok:
            self.root.read_errors.append({
                "path": str(self.root.file(CACHE_RECHECK_SUMMARY)),
                "error": f"expected schema {CACHE_RECHECK_SCHEMA}, observed {summary.get('schema')!r}",
            })
        if isinstance(complete, dict) and not completion_schema_ok:
            self.root.read_errors.append({
                "path": str(self.root.file(CACHE_RECHECK_COMPLETE)),
                "error": f"expected schema {CACHE_RECHECK_SCHEMA}, observed {complete.get('schema')!r}",
            })
        completion_lineage_ok = bool(
            summary_schema_ok
            and completion_schema_ok
            and complete.get("run_id") == summary.get("run_id")
            and complete.get("summary") == str(self.root.file(CACHE_RECHECK_SUMMARY))
        )
        if completion_schema_ok and summary_schema_ok and not completion_lineage_ok:
            self.root.read_errors.append({
                "path": str(self.root.file(CACHE_RECHECK_COMPLETE)),
                "error": "completion run_id or summary path does not match cache-recheck-summary.json",
            })
        cells = summary.get("cells") if summary_schema_ok and isinstance(summary.get("cells"), dict) else {}
        coverage = summary.get("coverage") if summary_schema_ok and isinstance(summary.get("coverage"), dict) else {}
        attempted_source_keys = {
            value for value in coverage.get("attempted", []) if isinstance(value, str)
        }
        attempted_groups = {
            group for group, source_keys in CACHE_RECHECK_GROUPS.items()
            if set(source_keys).issubset(attempted_source_keys)
            and all(
                isinstance(cells.get(key), dict)
                and cells[key].get("schema") == CACHE_PHASE_SCHEMA
                for key in source_keys
            )
        }
        authoritative = bool(
            summary_schema_ok
            and completion_schema_ok
            and completion_lineage_ok
            and complete.get("all_requested_cells_attempted") is True
            and summary.get("original_receipts_preserved") is True
        )
        if authoritative:
            self.cache_recheck_groups_available.update(attempted_groups)
        prefix = stage_prefix(stage)
        for path in self.root.glob(f"{prefix}cache-*.launch.json"):
            label = path.name[: -len(".launch.json")]
            if authoritative:
                self.cache_recheck_boot_labels.add(label)
            self.boot_row(phase, label, stage=stage)
        sources = [
            str(self.root.file(name))
            for name, present in (
                (CACHE_RECHECK_SUMMARY, summary is not None),
                (CACHE_RECHECK_COMPLETE, complete is not None),
            )
            if present
        ]
        before = len(self.rows)
        self.cache_cell_rows(
            phase,
            stage,
            tuple(CACHE_RECHECK_GROUPS),
            attempted_groups,
            sources,
        )
        rows = {
            row["cell"].removeprefix("cache:"): row
            for row in self.rows[before:]
            if row["kind"] == "cache-cell"
        }
        for group, source_keys in CACHE_RECHECK_GROUPS.items():
            row = rows[group]
            source_cells = {
                key: cells[key] for key in source_keys if isinstance(cells.get(key), dict)
            }
            invalid_source_schemas = {
                key: cells[key].get("schema")
                for key in source_keys
                if isinstance(cells.get(key), dict)
                and cells[key].get("schema") != CACHE_PHASE_SCHEMA
            }
            row["source_state"] = (
                "available"
                if group in attempted_groups and summary_schema_ok
                else "unavailable"
            )
            row["outcomes"] = {
                **(row["outcomes"] or {}),
                "source_schema": summary.get("schema") if isinstance(summary, dict) else None,
                "expected_source_schema": CACHE_RECHECK_SCHEMA,
                "source_cells": source_cells,
                "coverage": {
                    "source_keys": list(source_keys),
                    "attempted": sorted(set(source_keys).intersection(attempted_source_keys)),
                },
                "terminal_completion": complete if completion_schema_ok else None,
                "original_receipts_preserved": summary.get("original_receipts_preserved") if summary_schema_ok else None,
                "completion_lineage_matches_summary": completion_lineage_ok,
                "authoritative_for_supersession": authoritative and group in attempted_groups,
                "expected_cell_schema": CACHE_PHASE_SCHEMA,
                "invalid_source_cell_schemas": invalid_source_schemas,
            }
            if summary is None:
                row["notes"] = [
                    *row["notes"],
                    f"corrected cache source unavailable: {self.root.file(CACHE_RECHECK_SUMMARY)} has not been written",
                ]
            elif not summary_schema_ok:
                row["status"] = "fail"
                row["failure_classes"] = sorted({*row["failure_classes"], "source-schema"})
                row["source_state"] = "invalid-schema"
            elif invalid_source_schemas:
                row["status"] = "fail"
                row["failure_classes"] = sorted({*row["failure_classes"], "source-schema"})
                row["source_state"] = "invalid-schema"
            elif group not in attempted_groups:
                row["status"] = "missing"
                row["failure_classes"] = []
                row["notes"] = [
                    *row["notes"],
                    f"corrected source cells not all attempted: {', '.join(source_keys)}",
                ]
            elif complete is None:
                row["status"] = "running" if not row["failure_classes"] else row["status"]
                row["notes"] = [*row["notes"], "rolling summary present; terminal completion receipt unavailable"]
            elif not completion_schema_ok:
                row["status"] = "fail"
                row["failure_classes"] = sorted({*row["failure_classes"], "source-schema"})
                row["source_state"] = "invalid-schema"
            elif not completion_lineage_ok:
                row["status"] = "fail"
                row["failure_classes"] = sorted({*row["failure_classes"], "source-lineage"})
                row["source_state"] = "invalid-lineage"

    def cache_cell_rows(self, phase: str, stage: str, cells: tuple[str, ...], attempted: set[str], sources: list[str]) -> None:
        cache_gates = [gate for gate in self.gates.with_prefix("cache:", stage) if not (isinstance(gate.get("detail"), dict) and gate["detail"].get("gate_type") == "cache_effect")]
        by_cell: dict[str, list[dict[str, Any]]] = {name: [] for name in cells}
        for gate in cache_gates:
            cell = cache_gate_cell(gate["name"])
            if cell in by_cell:
                by_cell[cell].append(gate)
        official = self.arm(rt.IMAGE)
        arms = {
            "alternating-r25-r26": f"{official}+{self.arm(rt.R25_IMAGE)}",
            "focused-stock-vs-overlay": f"{official}+{self.arm(rt.OVERLAY_IMAGE)}",
            "native-canaries": f"{official}+{self.arm(rt.R25_IMAGE)}",
        }
        for cell in cells:
            gates = by_cell[cell]
            failed = [gate for gate in gates if not gate.get("passed") and gate["name"] not in CACHE_STOCK_CONTROL_GATES]
            control = [gate["name"] for gate in gates if gate["name"] in CACHE_STOCK_CONTROL_GATES]
            classes = sorted({cache_failure_class(gate) for gate in failed})
            if not gates and cell not in attempted:
                status = "missing"
            elif set(classes) - CACHE_OBSERVATION_CLASSES:
                status = "fail"
            elif classes:
                status = "flagged"
            else:
                status = "pass"
            outcomes: dict[str, Any] = {"gates_total": len(gates), "gates_failed": {gate["name"]: cache_failure_class(gate) for gate in failed},
                                        "control_observation_gates": control, "attempted_in_summary": cell in attempted}
            if cell == "prefill-overhead" and stage == "root":
                overhead = self.root.json("cache-prefill-overhead-summary.json")
                if isinstance(overhead, dict):
                    # comparison figures are speed-class; they are only quotable when every per-arm sub-row is clean
                    outcomes["overhead_comparison"] = {key: overhead.get(key) for key in ("steady_median_tokens_per_second", "throughput_overhead_vs_effective_l2_off_percent")}
            notes = []
            if control:
                notes.append("stock R26 control gates only require completion; semantic misses there are the expected control baseline (#574/#643/#645 live in the overlay)")
            if "harness-observation" in classes:
                if any(gate["name"] == "cache:drock-overlay:#645:truthful-cache-events-and-replay" for gate in failed):
                    notes.append("the #645 replay observation used the pre-fix REQ/ROUTER consumer; serving, live events and cached-prefix publication completed, and final_runtime_diagnostics.py owns the corrected replay")
                else:
                    notes.append("harness-observation: telemetry or status was not observable through the probe; this is not cache-corruption evidence")
            if "environment-evidence" in classes:
                notes.append("environment-evidence failure: a host filesystem/source-scope check did not hold; scope limitation, not a cache behaviour")
            if "text-byte-inequality" in classes:
                notes.append("legacy full-channel text hash inequality: transfer metadata and visible reference retention are reported separately; reasoning-channel variation is not KV-corruption evidence")
            if cell.endswith("-lifecycles"):
                shape = "fp8" if cell.startswith("official-fp8") else "packed-nvfp4"
                outcomes["lifecycle_subchecks"] = self.lifecycle_subchecks(stage, shape, {gate["name"]: gate for gate in gates})
            self.row(phase=phase, cell=f"cache:{cell}", kind="cache-cell", arm=arms.get(cell, official), stage=stage, status=status, failure_classes=classes,
                     source_state="available" if sources else "unavailable", outcomes=outcomes, window=self.cache_cell_window(cell, stage),
                     sources=sources, gates=[gate["name"] for gate in gates], notes=notes)

    def lifecycle_subchecks(self, stage: str, shape: str, gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Normalize legacy and v2 lifecycle receipts without conflating visible and reasoning text."""
        prefix = stage_prefix(stage)
        result: dict[str, Any] = {}
        for size in LIFECYCLE_SIZES:
            gate = gates.get(f"cache:official-r26:{shape}:{size}:lifecycle")
            if gate is None:
                continue
            detail = gate.get("detail") if isinstance(gate.get("detail"), dict) else {}
            checks = detail.get("checks") if isinstance(detail.get("checks"), dict) else {}
            current_verdict = detail.get("schema") == CACHE_PHASE_SCHEMA
            stages: dict[str, dict[str, Any] | None] = {}
            for step in ("cold", "warm", "restart"):
                receipt_name = f"{prefix}cache-official-r26-{shape}-{size}-{step}.json"
                receipt = self.root.json(receipt_name)
                if not isinstance(receipt, dict):
                    stages[step] = None
                    continue
                receipt_schema = receipt.get("schema")
                if receipt_schema not in (None, CACHE_PROBE_SCHEMA):
                    stages[step] = {
                        "receipt": str(self.root.file(receipt_name)),
                        "receipt_schema": receipt_schema,
                        "source_error": f"expected {CACHE_PROBE_SCHEMA} or the retained pre-schema legacy shape",
                    }
                    continue
                summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
                request = receipt.get("request") if isinstance(receipt.get("request"), dict) else {}
                content = summary.get("content") if isinstance(summary.get("content"), str) else ""
                reasoning = summary.get("reasoning_content") if isinstance(summary.get("reasoning_content"), str) else ""
                completion, max_tokens = summary.get("completion_tokens"), request.get("max_tokens")
                if receipt_schema == CACHE_PROBE_SCHEMA:
                    generation_budget_reached = summary.get("generation_budget_reached") is True
                    visible_hash = summary.get("visible_output_sha256")
                    full_hash = summary.get("full_output_sha256")
                    visible_status = summary.get("visible_final_status")
                    visible_complete = summary.get("visible_final_complete")
                    response_complete = summary.get("response_protocol_complete")
                    budget_limited_empty = summary.get("budget_limited_empty_final")
                    hash_contract = CACHE_PROBE_SCHEMA
                else:
                    generation_budget_reached = bool(
                        summary.get("finish_reason") == "length"
                        or (
                            isinstance(completion, int)
                            and isinstance(max_tokens, int)
                            and completion >= max_tokens
                            and summary.get("finish_reason") != "stop"
                        )
                    )
                    visible_hash = hashlib.sha256(content.encode()).hexdigest()
                    full_hash = summary.get("output_sha256")
                    visible_complete = bool(content.strip()) and summary.get("finish_reason") == "stop"
                    budget_limited_empty = generation_budget_reached and not bool(content.strip())
                    visible_status = (
                        "budget_limited_empty_final"
                        if budget_limited_empty
                        else "budget_limited_partial_visible_final"
                        if generation_budget_reached
                        else "complete_visible_final"
                        if visible_complete
                        else "incomplete_visible_final"
                        if content.strip()
                        else "empty_visible_final"
                    )
                    response_complete = receipt.get("ok") is True
                    hash_contract = "retained pre-schema receipt: visible hash recomputed; output_sha256 retained only as full-channel diagnostic"
                answer_evidence = receipt.get("answer_evidence") if isinstance(receipt.get("answer_evidence"), dict) else None
                stages[step] = {
                    "receipt": str(self.root.file(receipt_name)),
                    "receipt_schema": receipt.get("schema"),
                    "hash_contract": hash_contract,
                    "finish_reason": summary.get("finish_reason"),
                    "completion_tokens": completion,
                    "max_tokens": max_tokens,
                    "response_protocol_complete": response_complete,
                    "generation_budget_reached": generation_budget_reached,
                    "budget_limited_empty_final": budget_limited_empty,
                    "visible_final_complete": visible_complete,
                    "visible_final_status": visible_status,
                    "visible_output_sha256": visible_hash,
                    "full_output_sha256_diagnostic": full_hash,
                    "visible_output_utf8_bytes": (
                        summary.get("visible_output_utf8_bytes")
                        if receipt.get("schema") == CACHE_PROBE_SCHEMA
                        else len(content.encode())
                    ),
                    "reasoning_output_utf8_bytes": (
                        summary.get("reasoning_output_utf8_bytes")
                        if receipt.get("schema") == CACHE_PROBE_SCHEMA
                        else len(reasoning.encode())
                    ),
                    "lmcache_cached_tokens": (
                        summary.get("cache_stats", {}).get("num_lmcache_cached_tokens")
                        if isinstance(summary.get("cache_stats"), dict)
                        else None
                    ),
                    "visible_answer_outcome": answer_evidence.get("outcome") if answer_evidence else None,
                    "reference_present": (
                        receipt.get("checks", {}).get("reference_present")
                        if isinstance(receipt.get("checks"), dict)
                        else None
                    ),
                    "template_mode": self.reasoning_template_mode(request, "r26"),
                }
            present = [row for row in stages.values() if row and not row.get("source_error")]
            complete = len(present) == 3 and all(stages.values())
            visible_identical = (
                len({row["visible_output_sha256"] for row in present}) == 1
                if complete and all(row["visible_output_sha256"] is not None for row in present)
                else None
            )
            full_identical = (
                detail.get("full_output_byte_equal_diagnostic")
                if current_verdict
                else checks.get("generated_output_byte_equal")
            )
            if current_verdict:
                visible_identical = detail.get("visible_output_byte_equal")
                reference_retained = checks.get("reference_retained_in_visible_final")
            else:
                reference_retained = checks.get("reference_retained")
            cut = [step for step, row in stages.items() if row and row["generation_budget_reached"]]
            empty_cutoffs = [step for step, row in stages.items() if row and row["budget_limited_empty_final"]]
            single_token = complete and all(row["max_tokens"] == 1 for row in present)
            transfer = {
                key: checks.get(key)
                for key in ("cold_external_miss", "warm_external_hit", "restart_l2_hit")
            }
            findings = []
            if all(value is True for value in transfer.values()):
                findings.append("transfer metadata shows cold miss, warm hit and restart L2 hit")
            if reference_retained is True and detail.get("kind") == "reference":
                findings.append("all naturally completed visible finals returned the reference")
            if visible_identical is False:
                findings.append("visible-final bytes differed")
            if full_identical is False and visible_identical is True:
                findings.append("full content+reasoning diagnostic differed while visible-final bytes matched")
            if single_token:
                findings.append("one-token forced period: latency/cache-transfer only; empty visible finals are budget-limited, not wrong answers")
            elif cut:
                findings.append(f"generation budget reached at {', '.join(cut)}; answer completeness is reported separately")
            result[f"{shape}-{size}"] = {
                "source_contract": CACHE_PHASE_SCHEMA if current_verdict else "legacy pre-v2 lifecycle gate",
                "gate_passed": gate.get("passed"),
                "kind": detail.get("kind"),
                "measurement_scope": detail.get("measurement_scope"),
                "answer_quality_evaluated": detail.get("answer_quality_evaluated") if current_verdict else detail.get("kind") == "reference",
                "target_tokens": detail.get("target_tokens"),
                "transfer_hit_metadata": {**transfer, "lmcache_hit_tokens": detail.get("lmcache_hit_tokens")},
                "reference_retention": {
                    "reference_retained_in_visible_final": reference_retained,
                    "per_stage_reference_present": {
                        step: (row or {}).get("reference_present") for step, row in stages.items()
                    },
                },
                "output_identity": {
                    "visible_output_byte_equal": visible_identical,
                    "visible_output_sha256": {
                        step: (row or {}).get("visible_output_sha256") for step, row in stages.items()
                    },
                    "full_output_byte_equal_diagnostic": full_identical,
                    "full_output_sha256_diagnostic": {
                        step: (row or {}).get("full_output_sha256_diagnostic") for step, row in stages.items()
                    },
                    "full_channel_is_answer_oracle": False,
                    "visible_final_is_answer_oracle": detail.get("kind") == "reference",
                },
                "budget_cutoff": {
                    "single_token_by_design": single_token,
                    "stages_cut_off": cut,
                    "budget_limited_empty_final_stages": empty_cutoffs,
                    "max_tokens": {step: (row or {}).get("max_tokens") for step, row in stages.items()},
                    "budget_limited_empty_final_is_wrong_answer": False,
                },
                "template_mode": sorted({row["template_mode"] for row in present}),
                "template_proof": (
                    str(self.root.file(REASONING_PROTOCOL_PROOF))
                    if self.reasoning_proof is not None
                    else None
                ),
                "stages": stages,
                "findings": findings,
                "not_claimed": "full GLM KV byte identity or corruption; visible answers, full-channel diagnostics and transfer metadata are separate evidence",
            }
        return result

    def reasoning_template_mode(self, request: dict[str, Any], release: str) -> str:
        template_kwargs = request.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict) and template_kwargs.get("reasoning_effort") == "low":
            return "explicit-low via chat_template_kwargs"
        if request.get("reasoning_effort") == "low":
            proof_rows = self.reasoning_proof.get(release) if isinstance(self.reasoning_proof, dict) else None
            mapped = any(
                isinstance(row, dict)
                and row.get("input") == {"reasoning_effort": "low"}
                and isinstance(row.get("template_kwargs"), dict)
                and row["template_kwargs"].get("reasoning_effort") == "low"
                for row in (proof_rows if isinstance(proof_rows, list) else [])
            )
            return (
                "explicit-low via top-level reasoning_effort (verified template mapping)"
                if mapped
                else "explicit-low via top-level reasoning_effort (protocol proof unavailable)"
            )
        return "template-default (reasoning omitted)"

    def cache_cell_window(self, cell: str, stage: str = "root") -> dict[str, Any] | None:
        stems = {
            "storage-evidence": ["cache-storage-filesystem-evidence"], "source-audit": ["cache-byte-transfer-source-audit"],
            "official-packed-nvfp4-needles": ["cache-official-r26-packed-nvfp4-unique-needles"], "eviction": ["cache-official-r26-40-document-eviction"],
        }.get(cell, [])
        windows = [window for window in (self.root.window(f"{stage_prefix(stage)}{stem}") for stem in stems) if window]
        if not windows:
            return None
        return {"started_at": min(window["started_at"] for window in windows), "finished_at": max(window["finished_at"] or 0 for window in windows) or None, "source": ", ".join(window["source"] for window in windows)}


    def semantic_control_rows(self, phase: str, stage: str) -> None:
        """overlay_quality_recheck.semantic_controls: matched LAVD controls separating speculation from target execution."""
        controls = self.root.json(f"{stage_prefix(stage)}semantic-controls.json")
        recorded = {row.get("label"): row for row in (controls or {}).get("controls", []) if isinstance(row, dict)} if isinstance(controls, dict) else {}
        for release, image in (("r25", rt.R25_IMAGE), ("r26", rt.IMAGE)):
            for spec, reasoning, runs in SEMANTIC_CONTROLS:
                label = f"semantic-{release}-{spec}-{reasoning}-lavd"
                boot = self.boot_row(phase, label, expected_image=image, expected={"tp": 4, "dcp": 1, "spec": spec, "cache": "vram", "kv": "fp8_ds_mla"}, stage=stage)
                record = recorded.get(label, {})
                boot_failed = boot["status"] == "fail" or record.get("booted") is False
                self.quality_profile_row(phase, {"label": label, "profile": "lavd-test", "runs": runs}, image, stage=stage, boot_failed=boot_failed, arm=boot["arm"])
                row = self.rows[-1]
                row["kind"] = f"semantic-control:lavd-test:{reasoning}"
                row["config"] = {"spec": spec, "dcp": 1, "kv": "fp8_ds_mla", "reasoning": reasoning, "runs": runs}
                row["outcomes"] = {**(row["outcomes"] or {}), "control_record": record or None}
                row["notes"] = [*row["notes"], SEMANTIC_CONTROL_NOTES[reasoning]]
                if record.get("error"):
                    row["failure_classes"] = sorted({*row["failure_classes"], "runtime-protocol"})
                    row["status"] = "fail"

    # -- matrix sections (smoke, priority, matrix, tuning, tp2)
    def matrix_rows(self, phase: str, section: str) -> None:
        plan = matrix_section_plan(section)
        self.plan_sources[phase] = f"scripts/r26/matrix_phase.py {section}() dry-run enumeration"
        boots = {boot["label"]: boot for boot in plan["boots"]}
        boot_arms: dict[str, str] = {}
        for boot in plan["boots"]:
            expected = {key: boot.get(key, default) for key, default in (("tp", 4), ("dcp", 4), ("spec", "mtp0"), ("cache", "vram"), ("kv", "fp8_ds_mla"), ("extra_env", None))}
            boot_arms[boot["label"]] = self.boot_row(phase, boot["label"], expected_image=boot.get("image", rt.IMAGE), expected=expected)["arm"]
            if self.gates.passed(f"matrix-cell:{boot['label']}") is False:
                self.row(phase=phase, cell=f"matrix-cell:{boot['label']}", kind="harness", arm=boot_arms[boot["label"]], status="fail", failure_classes=["harness"],
                         outcomes=self.gates.detail(f"matrix-cell:{boot['label']}"), gates=[f"matrix-cell:{boot['label']}"])
        for bench in plan["benches"]:
            boot = boots.get(bench["boot_label"]) or {}
            self.bench_row(phase, bench, boot, arm=boot_arms.get(bench["boot_label"]))

    def bench_row(self, phase: str, bench: dict[str, Any], boot: dict[str, Any], *, stage: str = "root", arm: str | None = None, extra_gates: tuple[str, ...] = ()) -> dict[str, Any]:
        prefix = stage_prefix(stage)
        label = bench["label"]
        image = boot.get("image", rt.IMAGE)
        arm = arm or self.arm(image, boot.get("model_dir"))
        receipt = self.root.json(f"{prefix}{label}.json")
        window = self.root.window(f"{prefix}{label}.bench")
        passed = self.gates.passed(f"benchmark-execution:{label}", stage)
        kv = self.root.server_kv_tokens(f"{prefix}{bench['boot_label']}")
        gates = [f"benchmark-execution:{label}", *extra_gates]
        config = {"conc": bench.get("conc"), "contexts": bench.get("contexts"), "duration": bench.get("duration"), "spec": boot.get("spec", "mtp0"), "dcp": boot.get("dcp", 4), "tp": boot.get("tp", 4), "kv": boot.get("kv", "fp8_ds_mla"), "cache": boot.get("cache", "vram"), "extra_env": boot.get("extra_env")}
        if not isinstance(receipt, dict):
            boot_failed = self.gates.passed(f"boot:{bench['boot_label']}", stage) is False
            return self.row(phase=phase, cell=label, kind="bench", arm=arm, image=digest_of(image), config=config, stage=stage,
                            status="fail" if (passed is False or boot_failed) else "missing", failure_classes=["runtime-protocol"] if (passed is False or boot_failed) else [],
                            window=window, gates=gates, notes=["boot failed; bench never ran"] if boot_failed else [])
        metadata = receipt.get("metadata") if isinstance(receipt.get("metadata"), dict) else {}
        cells = []
        for cell in receipt.get("results") or []:
            if not isinstance(cell, dict):
                continue
            cells.append({
                "conc": cell.get("concurrency"), "ctx": cell.get("context_tokens"), "aggregate_tps": cell.get("aggregate_tps"),
                "per_user_tps_p50": cell.get("output_tps_per_user_p50"), "ttft_p50": cell.get("ttft_p50"),
                "spec_accept_rate": cell.get("server_spec_accept_rate"), "spec_accept_length": cell.get("server_spec_accept_length"),
                "errors": cell.get("num_errors"), "underfilled": cell.get("underfilled"), "capacity_limited": cell.get("capacity_limited"),
                "effective_concurrency": cell.get("effective_concurrency"),
            })
        prefill = receipt.get("prefill") if isinstance(receipt.get("prefill"), dict) else {}
        speed = {"cells": cells, "prefill": {ctx: {"tok_per_sec": row.get("tok_per_sec"), "ttft_seconds": row.get("ttft_seconds")} for ctx, row in prefill.items() if isinstance(row, dict)}}
        errors = sum(int(cell.get("errors") or 0) for cell in cells)
        status = "pass" if passed and cells and not errors else "fail"
        classes = [] if status == "pass" else ["runtime-protocol"]
        notes = ["bench metadata.max_total_tokens is not capacity evidence; kv_tokens_server comes from the server log"]
        if kv and metadata.get("max_total_tokens") and metadata["max_total_tokens"] != kv["tokens"]:
            notes.append(f"bench KV metadata {metadata['max_total_tokens']:,} != server GPU KV cache size {kv['tokens']:,}")
        return self.row(phase=phase, cell=label, kind="bench", arm=arm, image=digest_of(image), config=config, status=status, failure_classes=classes, stage=stage,
                        outcomes={"cells": len(cells), "errors": errors, "bench_version": metadata.get("version"), "bench_timestamp": metadata.get("timestamp")},
                        speed=speed, window=window, kv_tokens_server=kv["tokens"] if kv else None, kv_tokens_bench=metadata.get("max_total_tokens"),
                        sources=[str(self.root.file(f"{prefix}{label}.json"))] + ([kv["source"]] if kv else []), gates=gates, notes=notes)

    # -- clean speed reruns (clean_rerun_phase.py under ROOT/clean-reruns)
    def clean_rerun_rows(self, phase: str) -> None:
        """Ingest one canonical row per result_label and retain every ledger attempt."""
        stage = self.phase_stages[phase]
        self.plan_sources[phase] = (
            "scripts/r26/clean_rerun_phase.py contaminated_cells(), resume_state() and "
            f"steady_metrics.py; ledger {CLEAN_RERUN_PLAN}"
        )
        ledger = self.root.json(CLEAN_RERUN_PLAN)
        histories: dict[str, list[dict[str, Any]]] = {}
        boot_histories: dict[str, list[dict[str, Any]]] = {}
        if isinstance(ledger, dict):
            for group_index, entry in enumerate(ledger.get("reruns") or []):
                if not isinstance(entry, dict) or not isinstance(entry.get("boot_label"), str):
                    continue
                boot_label = entry["boot_label"]
                boot_histories.setdefault(boot_label, []).append({
                    "ledger_group_index": group_index,
                    "image": entry.get("image"),
                    "quiet_wait": entry.get("quiet_wait"),
                    "booted": entry.get("booted"),
                    "blocked": entry.get("blocked"),
                    "error": entry.get("error"),
                })
                for cell_index, cell in enumerate(entry.get("cells") or []):
                    if not isinstance(cell, dict) or not isinstance(cell.get("result_label"), str):
                        continue
                    histories.setdefault(cell["result_label"], []).append({
                        "ledger_group_index": group_index,
                        "ledger_cell_index": cell_index,
                        "boot_label": boot_label,
                        "image": entry.get("image"),
                        "entry_quiet_wait": entry.get("quiet_wait"),
                        "entry_booted": entry.get("booted"),
                        "entry_blocked": entry.get("blocked"),
                        "entry_error": entry.get("error"),
                        "clean": cell.get("clean"),
                        "resume_validation_failed": cell.get("resume_validation_failed"),
                        "attempts": cell.get("attempts") if isinstance(cell.get("attempts"), list) else [],
                    })
        planned: dict[str, dict[str, dict[str, Any]]] = {}
        for parent in self.rows:
            needs_verifier_window = str(parent["cell"]).startswith("acceptance-")
            dirty = parent["speed_qualification"] != "clean"
            if (
                parent["stage"] != "root"
                or parent["kind"] != "bench"
                or parent["cell"].startswith("smoke")
                or (parent["window"] or {}).get("returncode") != 0
                or (not dirty and not needs_verifier_window)
            ):
                continue
            boot_label = parent["cell"].split("-repeat")[0]
            launch = self.root.json(f"{boot_label}.launch.json")
            if isinstance(launch, dict):
                planned.setdefault(boot_label, {})[parent["cell"]] = parent
        for label, attempts in histories.items():
            boot_labels = {attempt["boot_label"] for attempt in attempts}
            if len(boot_labels) != 1:
                self.coverage_gaps.append(
                    f"clean-rerun ledger result_label {label} has conflicting boot labels {sorted(boot_labels)}"
                )
                continue
            boot_label = next(iter(boot_labels))
            if label not in planned.get(boot_label, {}):
                parent = next((
                    row for row in self.rows
                    if row["stage"] == "root" and row["kind"] == "bench" and row["cell"] == label
                ), None)
                if parent is None:
                    self.coverage_gaps.append(
                        f"clean-rerun ledger names {label} but the parent battery has no such bench row"
                    )
                else:
                    planned.setdefault(boot_label, {})[label] = parent
        for boot_label, parent_map in planned.items():
            parent_launch = self.root.json(f"{boot_label}.launch.json")
            launch = parent_launch if isinstance(parent_launch, dict) else {}
            boot = self.boot_row(
                phase,
                boot_label,
                expected_image=launch.get("image"),
                stage=stage,
                expected_model_dir=launch.get("model_dir"),
                expected={key: launch.get(key) for key in ("tp", "dcp", "spec", "cache", "kv")},
            )
            boot["outcomes"] = {
                **(boot["outcomes"] or {}),
                "ledger_history": boot_histories.get(boot_label, []),
            }
            for label, parent in parent_map.items():
                parent_command = self.root.json(f"{label}.bench.command.json")
                options = bench_options(parent_command.get("args") or []) if isinstance(parent_command, dict) else {}
                bench = {"label": label, "boot_label": boot_label, **options}
                boot_plan = {
                    **{key: launch.get(key) for key in ("image", "tp", "dcp", "spec", "cache", "kv", "model_dir")},
                    "extra_env": (parent["config"] or {}).get("extra_env"),
                }
                row = self.bench_row(
                    phase,
                    bench,
                    boot_plan,
                    stage=stage,
                    arm=boot["arm"],
                    extra_gates=(f"clean-rerun:{label}",),
                )
                validation = self.clean_rerun_validation(
                    label,
                    boot_label,
                    launch,
                    parent_command if isinstance(parent_command, dict) else None,
                    histories.get(label, []),
                )
                producer_speed = row["speed"]
                row["source_state"] = "available" if validation["canonical_receipts_present"] else "unavailable"
                row["outcomes"] = {
                    **(row["outcomes"] or {}),
                    "parent_verdict": parent["speed_qualification"],
                    "ledger_history": histories.get(label, []),
                    "ledger_occurrences": len(histories.get(label, [])),
                    "canonical_validation": validation,
                }
                row["sources"] = list(dict.fromkeys([
                    *row["sources"],
                    *[
                        str(self.root.file(name))
                        for name in (
                            f"{CLEAN_RERUNS_DIR}/{boot_label}.launch.json",
                            f"{CLEAN_RERUNS_DIR}/{label}.bench.command.json",
                            f"{CLEAN_RERUNS_DIR}/{label}.steady-summary.json",
                            f"{CLEAN_RERUNS_DIR}/{label}.steady.metrics.jsonl",
                        )
                        if self.root.file(name).is_file()
                    ],
                ]))
                row["notes"] = [
                    *row["notes"],
                    "one canonical same-label receipt is published; every failed and successful ledger attempt remains in ledger_history and its archive",
                    "speed is qualified only when command and every steady counter window have complete isolation coverage with no foreign GPU work, and launch/checkpoint/config match the parent",
                ]
                summary = self.root.json(f"{CLEAN_RERUNS_DIR}/{label}.steady-summary.json")
                if validation["qualified"] and isinstance(summary, dict) and producer_speed is not None:
                    producer_speed["steady_counter_cells"] = summary["cells"]
                    producer_speed["steady_counter_schema"] = summary["schema"]
                    row["status"] = "pass"
                    row["failure_classes"] = []
                elif producer_speed is not None:
                    row["outcomes"]["unqualified_speed_observation"] = producer_speed
                    row["speed"] = None
                    row["speed_qualification"] = None
                    row["status"] = "fail"
                    row["failure_classes"] = sorted({*row["failure_classes"], "clean-rerun-validation"})
                elif row["status"] == "missing" and (
                    boot["status"] == "fail"
                    or any(entry.get("booted") is False for entry in boot_histories.get(boot_label, []))
                    or any(entry.get("blocked") for entry in boot_histories.get(boot_label, []))
                ):
                    row["status"] = "not-run"
                    row["outcomes"]["reason"] = "rerun boot failed or no quiet GPU window was available; benchmark did not run"
        self.final_diagnostic_rows(phase, stage)

    def clean_rerun_validation(
        self,
        label: str,
        boot_label: str,
        parent_launch: dict[str, Any],
        parent_command: dict[str, Any] | None,
        ledger_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prefix = f"{CLEAN_RERUNS_DIR}/"
        launch = self.root.json(f"{prefix}{boot_label}.launch.json")
        command = self.root.json(f"{prefix}{label}.bench.command.json")
        benchmark = self.root.json(f"{prefix}{label}.json")
        summary = self.root.json(f"{prefix}{label}.steady-summary.json")
        canonical_present = all(isinstance(value, dict) for value in (launch, command, benchmark, summary))
        launch_mismatches = {
            key: {"parent": parent_launch.get(key), "canonical": launch.get(key)}
            for key in LAUNCH_IDENTITY_KEYS
            if isinstance(launch, dict) and launch.get(key) != parent_launch.get(key)
        }
        command_options_match = (
            isinstance(command, dict)
            and isinstance(parent_command, dict)
            and bench_options(command.get("args") or []) == bench_options(parent_command.get("args") or [])
        )
        sample_path = self.root.file(f"{prefix}{label}.steady.metrics.jsonl")
        if sample_path.is_file():
            self.root.sources.setdefault(str(sample_path), {"bytes": sample_path.stat().st_size})
        summary_schema_ok = isinstance(summary, dict) and summary.get("schema") == STEADY_SCHEMA
        summary_label_ok = summary_schema_ok and summary.get("label") == label
        expected_benchmark = str(self.root.file(f"{prefix}{label}.json"))
        expected_samples = str(self.root.file(f"{prefix}{label}.steady.metrics.jsonl"))
        source_paths_match = bool(
            summary_label_ok
            and summary.get("source_benchmark") == expected_benchmark
            and summary.get("source_samples") == expected_samples
        )
        steady_cells = summary.get("cells") if summary_schema_ok and isinstance(summary.get("cells"), list) else []
        benchmark_cells = benchmark.get("results") if isinstance(benchmark, dict) and isinstance(benchmark.get("results"), list) else []
        steady_identity = {
            (cell.get("concurrency"), cell.get("context_tokens"))
            for cell in steady_cells if isinstance(cell, dict)
        }
        benchmark_identity = {
            (cell.get("concurrency"), cell.get("context_tokens"))
            for cell in benchmark_cells if isinstance(cell, dict)
        }
        identity_match = (
            bool(steady_cells)
            and len(steady_cells) == len(benchmark_cells)
            and steady_identity == benchmark_identity
        )
        speculation_counters_valid = bool(
            isinstance(launch, dict)
            and (
                launch.get("spec") == "mtp0"
                or (
                    steady_cells
                    and all(
                        isinstance(cell, dict)
                        and isinstance(cell.get("aggregate_verifier_steps_per_second"), (int, float))
                        and cell["aggregate_verifier_steps_per_second"] > 0
                        for cell in steady_cells
                    )
                )
            )
        )
        steady_windows = []
        for cell in steady_cells:
            if not isinstance(cell, dict):
                continue
            window = {
                "started_at": to_epoch(cell.get("started_at")),
                "finished_at": to_epoch(cell.get("finished_at")),
                "source": f"{prefix}{label}.steady-summary.json cells[{cell.get('concurrency')},{cell.get('context_tokens')}]",
            }
            steady_windows.append({
                "concurrency": cell.get("concurrency"),
                "context_tokens": cell.get("context_tokens"),
                "valid": cell.get("valid"),
                "window": window,
                "isolation": self.isolation.verdict(window),
            })
        command_window = (
            {
                "started_at": to_epoch(command.get("started_at")),
                "finished_at": to_epoch(command.get("finished_at")),
                "returncode": command.get("returncode"),
                "source": str(self.root.file(f"{prefix}{label}.bench.command.json")),
            }
            if isinstance(command, dict)
            else None
        )
        command_isolation = self.isolation.verdict(command_window)
        ledger_clean = any(entry.get("clean") is True for entry in ledger_history)
        checks = {
            "canonical_receipts_present": canonical_present,
            "launch_image_checkpoint_config_match_parent": isinstance(launch, dict) and not launch_mismatches,
            "bench_options_match_parent": command_options_match,
            "command_returncode_zero": isinstance(command, dict) and command.get("returncode") == 0,
            "steady_schema": summary_schema_ok,
            "steady_result_label": summary_label_ok,
            "steady_source_paths": source_paths_match,
            "steady_cells_match_benchmark": identity_match,
            "speculation_counter_windows_valid": speculation_counters_valid,
            "steady_windows_valid": bool(summary_schema_ok and summary.get("all_windows_valid") is True and steady_windows and all(cell["valid"] is True for cell in steady_windows)),
            "command_isolation_fully_clean": command_isolation.get("status") == "clean",
            "every_steady_window_isolation_fully_clean": bool(steady_windows) and all(cell["isolation"].get("status") == "clean" for cell in steady_windows),
            "ledger_has_clean_attempt": ledger_clean,
        }
        return {
            "qualified": all(checks.values()),
            "canonical_receipts_present": canonical_present,
            "checks": checks,
            "launch_mismatches": launch_mismatches,
            "command_isolation": command_isolation,
            "steady_windows": steady_windows,
            "expected_schema": STEADY_SCHEMA,
        }

    def final_diagnostic_rows(self, phase: str, stage: str) -> None:
        """Ingest final-runtime diagnostics and the lane-cap child recheck when they exist."""
        diagnostic = self.root.json(FINAL_DIAGNOSTICS)
        schema_ok = isinstance(diagnostic, dict) and diagnostic.get("schema") == FINAL_DIAGNOSTICS_SCHEMA
        if diagnostic is not None and not schema_ok:
            observed = diagnostic.get("schema") if isinstance(diagnostic, dict) else type(diagnostic).__name__
            self.root.read_errors.append({
                "path": str(self.root.file(FINAL_DIAGNOSTICS)),
                "error": f"expected schema {FINAL_DIAGNOSTICS_SCHEMA}, observed {observed!r}",
            })
        source = [str(self.root.file(FINAL_DIAGNOSTICS))] if diagnostic is not None else []
        replay = diagnostic.get("replay") if schema_ok and isinstance(diagnostic.get("replay"), dict) else None
        replay_result = replay.get("result") if isinstance(replay, dict) and isinstance(replay.get("result"), dict) else None
        replay_result_schema_ok = bool(
            replay is None
            or replay.get("error")
            or replay_result is not None and replay_result.get("schema") == CACHE_PHASE_SCHEMA
        )
        if diagnostic is not None and not schema_ok:
            replay_status, replay_classes = "fail", ["source-schema"]
        elif replay is None:
            replay_status, replay_classes = "missing", []
        elif not replay_result_schema_ok:
            replay_status, replay_classes = "fail", ["source-schema"]
        elif replay.get("passed") is True:
            replay_status, replay_classes = "pass", []
        elif replay.get("error"):
            replay_status, replay_classes = "fail", ["runtime-protocol"]
        else:
            replay_status, replay_classes = "fail", ["cache-metadata"]
        replay_row = self.row(
            phase=phase,
            cell="final-diagnostic:focused-overlay-replay",
            kind="cache-final-diagnostic:replay",
            arm=self.arm(rt.OVERLAY_IMAGE),
            image=digest_of(rt.OVERLAY_IMAGE),
            stage=stage,
            source_state=("invalid-schema" if diagnostic is not None and (not schema_ok or replay is not None and not replay_result_schema_ok) else "available" if replay is not None else "unavailable"),
            status=replay_status,
            failure_classes=replay_classes,
            outcomes={
                "expected_schema": FINAL_DIAGNOSTICS_SCHEMA,
                "expected_result_schema": CACHE_PHASE_SCHEMA,
                "result_schema_valid": replay_result_schema_ok,
                "replay": replay,
                "replay_client_source_sha256": diagnostic.get("replay_client_source_sha256") if schema_ok else None,
                "all_diagnostics_attempted": diagnostic.get("all_diagnostics_attempted") if schema_ok else None,
            },
            recheck_of=str(self.root.file(f"{FOLLOWUPS_DIR}/cache-focused-drock-overlay.json")) if replay is not None else None,
            gates=[
                gate["name"]
                for gate in (replay.get("gates") if isinstance(replay, dict) else []) or []
                if isinstance(gate, dict) and isinstance(gate.get("name"), str)
            ],
            sources=source,
            notes=[
                "corrected model-backed #645 replay after the documented REQ/ROUTER client mismatch",
                "the retained prior failed replay observation is superseded only when this structured result exists",
            ],
        )
        if replay_result is not None and schema_ok and replay_result_schema_ok:
            gate_name = "cache:drock-overlay:#645:truthful-cache-events-and-replay"
            target = next((
                row for row in self.rows
                if row["stage"] == FOLLOWUPS_DIR
                and row["cell"] == "cache:focused-stock-vs-overlay"
            ), None)
            if target is not None:
                target["superseded_checks"] = [
                    *target["superseded_checks"],
                    {
                        "gate": gate_name,
                        "superseded_by": str(self.root.file(FINAL_DIAGNOSTICS)) + "#replay",
                        "replacement_status": replay_row["status"],
                    },
                ]
                target["outcomes"] = {
                    **(target["outcomes"] or {}),
                    "superseded_checks": target["superseded_checks"],
                }
                failed = dict((target["outcomes"] or {}).get("gates_failed") or {})
                failed.pop(gate_name, None)
                target["outcomes"]["gates_failed_after_supersession"] = failed
                target["failure_classes"] = sorted(set(failed.values()))
                target["status"] = (
                    "fail"
                    if set(target["failure_classes"]) - CACHE_OBSERVATION_CLASSES
                    else "flagged"
                    if target["failure_classes"]
                    else "pass"
                )
                target["notes"] = [
                    *target["notes"],
                    "the pre-fix #645 consumer result remains in gates_failed; gates_failed_after_supersession excludes it and the replacement row carries the current verdict",
                ]
                self.superseded_gates[(FOLLOWUPS_DIR, gate_name)] = str(self.root.file(FINAL_DIAGNOSTICS)) + "#replay"
        native_rows = diagnostic.get("native_loader") if schema_ok and isinstance(diagnostic.get("native_loader"), list) else []
        by_kv: dict[str, dict[str, Any]] = {}
        for native in native_rows:
            if not isinstance(native, dict):
                continue
            configuration = native.get("configuration") if isinstance(native.get("configuration"), dict) else {}
            kv = configuration.get("kv")
            if not isinstance(kv, str) and native.get("error") and isinstance(native.get("kv"), str):
                kv = native["kv"]
            if isinstance(kv, str):
                by_kv[kv] = native
        for kv in ("fp8_ds_mla", "nvfp4_ds_mla"):
            native = by_kv.get(kv)
            native_schema_ok = bool(
                native is None
                or native.get("error")
                or native.get("schema") == CACHE_PHASE_SCHEMA
            )
            if diagnostic is not None and not schema_ok:
                status, native_classes, native_source_state = "fail", ["source-schema"], "invalid-schema"
            elif native is None:
                status, native_classes, native_source_state = "missing", [], "unavailable"
            elif not native_schema_ok:
                status, native_classes, native_source_state = "fail", ["source-schema"], "invalid-schema"
            elif native.get("passed") is True:
                status, native_classes, native_source_state = "pass", [], "available"
            else:
                status, native_classes, native_source_state = "fail", ["runtime-protocol"], "available"
            self.row(
                phase=phase,
                cell=f"final-diagnostic:native-safetensors-loader:{kv}",
                kind="cache-final-diagnostic:native-loader",
                arm=self.arm(rt.IMAGE),
                image=digest_of(rt.IMAGE),
                config={"cache": "native", "kv": kv, "load_format": "safetensors", "diagnostic_only": True},
                stage=stage,
                status=status,
                failure_classes=native_classes,
                source_state=native_source_state,
                outcomes={
                    "expected_schema": FINAL_DIAGNOSTICS_SCHEMA,
                    "expected_result_schema": CACHE_PHASE_SCHEMA,
                    "result_schema_valid": native_schema_ok,
                    "result": native,
                    "supersedes_default_loader_result": False,
                    "all_diagnostics_attempted": diagnostic.get("all_diagnostics_attempted") if schema_ok else None,
                },
                recheck_of=str(self.root.file(f"{FOLLOWUPS_DIR}/cache-native-canary-summary.json")) if native is not None else None,
                sources=source,
                notes=[
                    "safetensors is an explicitly different loader diagnostic",
                    "success does not erase or relabel the default InstantTensor/cumem OOM at GMU 0.93 or 0.90",
                ],
            )
        self.scheduler_lane_recheck_rows(phase, stage)

    def scheduler_lane_recheck_rows(self, phase: str, stage: str) -> None:
        report = self.root.json(LANE_RECHECK)
        report_ok = isinstance(report, dict) and report.get("schema") == LANE_RECHECK_SCHEMA
        if report is not None and not report_ok:
            observed = report.get("schema") if isinstance(report, dict) else type(report).__name__
            self.root.read_errors.append({
                "path": str(self.root.file(LANE_RECHECK)),
                "error": f"expected schema {LANE_RECHECK_SCHEMA}, observed {observed!r}",
            })
        groups = (
            report.get("groups")
            if report_ok and isinstance(report.get("groups"), list)
            else [] if report is not None
            else None
        )
        if groups is None:
            try:
                import scheduler_lane_cap_recheck

                selected = scheduler_lane_cap_recheck.selected_groups(self.root.path)
                groups = [
                    scheduler_lane_cap_recheck.scheduler.group_to_json(group)
                    for group in selected
                ]
                plan_source = (
                    f"{HERE / 'scheduler_lane_cap_recheck.py'} selected_groups(); "
                    f"source SHA-256 {sha256_file(HERE / 'scheduler_lane_cap_recheck.py')}"
                )
            except Exception as error:
                groups = []
                plan_source = f"scheduler lane-cap source plan unavailable: {type(error).__name__}: {error}"
                self.coverage_gaps.append(plan_source)
        else:
            plan_source = str(self.root.file(LANE_RECHECK)) + "#groups"
        results = report.get("results") if report_ok and isinstance(report.get("results"), dict) else {}
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("name"), str):
                continue
            candidate = group.get("candidate") if isinstance(group.get("candidate"), dict) else {}
            self.boot_row(
                phase,
                f"scheduler-recheck/{group['name']}",
                expected_image=candidate.get("image"),
                expected={
                    "tp": 4,
                    "dcp": 4,
                    "spec": "mtp0",
                    "cache": group.get("cache_mode"),
                    "kv": "fp8_ds_mla",
                    "extra_env": group.get("boot_environment"),
                    "extra_args": group.get("boot_extra_args"),
                },
                stage=stage,
            )
            for plan in group.get("plans") or []:
                if not isinstance(plan, dict) or not isinstance(plan.get("name"), str):
                    continue
                name = plan["name"]
                receipt = self.root.json(f"{CLEAN_RERUNS_DIR}/scheduler-recheck/{name}.json")
                record = results.get(name)
                new_row = self.scheduler_recheck_result_row(
                    phase=phase,
                    stage=stage,
                    cell_prefix="scheduler-lane-recheck",
                    output_dir="scheduler-recheck",
                    plan=plan,
                    candidate=candidate,
                    record=record if isinstance(record, dict) else None,
                    receipt=receipt,
                    plan_source=LANE_RECHECK,
                    summary_source=LANE_RECHECK,
                )
                new_row["outcomes"] = {
                    **(new_row["outcomes"] or {}),
                    "lane_recheck_schema": report.get("schema") if isinstance(report, dict) else None,
                    "expected_lane_recheck_schema": LANE_RECHECK_SCHEMA,
                    "lane_plan_source": plan_source,
                    "prior_exact_lane_assertion_corrected": True,
                }
                if isinstance(record, dict) and isinstance(receipt, dict):
                    originals = [
                        row for row in self.rows
                        if row["stage"] == FOLLOWUPS_DIR
                        and row["cell"] == f"scheduler-recheck:{name}"
                    ]
                    if originals:
                        self.supersede(
                            originals,
                            [new_row],
                            f"{FOLLOWUPS_DIR}/scheduler-recheck/phase-summary.json#invocations/{name}",
                            f"{LANE_RECHECK}#results/{name}",
                        )
        if not groups:
            self.row(
                phase=phase,
                cell="scheduler-lane-recheck:plan",
                kind="scheduler-lane-recheck:source",
                stage=stage,
                source_state="invalid-schema" if report is not None else "unavailable",
                status="fail" if report is not None else "missing",
                failure_classes=["source-schema"] if report is not None else [],
                outcomes={"expected_schema": LANE_RECHECK_SCHEMA, "plan_source": plan_source},
                sources=[str(self.root.file(LANE_RECHECK))] if report is not None else [],
                notes=["no lane-cap GPU result is inferred without a source-backed group plan"],
            )

    def resolve_clean_reruns(self) -> None:
        """Publish one clean canonical speed result while retaining the parent observation."""
        for rerun in [
            row for row in self.rows
            if row["stage"] == CLEAN_RERUNS_DIR
            and row["kind"] == "bench"
            and row["speed"] is not None
        ]:
            parent = next((
                row for row in self.rows
                if row["stage"] == "root"
                and row["kind"] == "bench"
                and row["cell"] == rerun["cell"]
            ), None)
            if parent is None:
                continue
            rerun["recheck_of"] = str(self.root.file(f"{parent['cell']}.json"))
            if rerun["speed_qualification"] != "clean":
                self.coverage_gaps.append(
                    f"clean rerun of {rerun['cell']} was itself {rerun['speed_qualification']}; "
                    "the parent speed observation remains on record"
                )
                continue
            original_speed = parent["speed"]
            original_qualification = parent["speed_qualification"]
            parent["outcomes"] = {
                **(parent["outcomes"] or {}),
                "original_speed_observation": original_speed,
                "original_speed_qualification": original_qualification,
            }
            parent["speed"] = None
            parent["speed_qualification"] = None
            parent["superseded_by"] = str(
                self.root.file(f"{CLEAN_RERUNS_DIR}/{rerun['cell']}.json")
            )
            parent["notes"] = [
                *parent["notes"],
                f"speed superseded by the canonical clean same-label rerun under {CLEAN_RERUNS_DIR}/; original speed and qualification remain in outcomes",
            ]
            rerun["notes"] = [
                *rerun["notes"],
                "canonical clean same-label speed result; parent receipt retained as explicit superseded speed evidence",
            ]

    # -- QAD matched checkpoint phase (qad_matched_phase.py)
    def qad_matched_rows(self, phase: str) -> None:
        import qad_matched_phase

        stage = self.phase_stages[phase]
        ledger = self.root.json("qad-matched-ledger.json")
        completed = self.root.json("qad-matched-completed.json")
        cells = {
            cell.get("label"): cell
            for cell in (ledger or {}).get("cells", [])
            if isinstance(cell, dict) and isinstance(cell.get("label"), str)
        } if isinstance(ledger, dict) else {}
        published = ledger.get("published") if isinstance(ledger, dict) else None
        candidate = ledger.get("candidate") if isinstance(ledger, dict) else self.weights["model_dir"]
        self.plan_sources[phase] = (
            "scripts/r26/qad_matched_phase.py CONFIGS x (published, candidate), "
            "including explicit-low Estonia/LAVD24 and separate template-default LAVD8"
        )
        weights = {"published": published or str(PUBLISHED_WEIGHTS), "candidate": candidate}
        for config_name, config, head_env in qad_matched_phase.CONFIGS:
            for weights_name in ("published", "candidate"):
                label = f"qad-matched-{config_name}-{weights_name}"
                model_dir = (cells.get(label) or {}).get("model_dir") or weights[weights_name]
                arm_spec = {
                    "label": label,
                    "image": rt.IMAGE,
                    "runtime": {"tp": 4, **config, "cache": "vram", "extra_env": head_env or None},
                    "acceptance_probe": {
                        "contexts": [0, 32768], "concurrency": [1, 8], "repeats": 2,
                    } if config["spec"] != "mtp0" else None,
                    "probes": [{"label": f"{label}-long", "suite": "long"}],
                    "profiles": [
                        {"label": f"{label}-estonia", "profile": "estonia", "runs": 24},
                        {"label": f"{label}-lavd-test", "profile": "lavd-test", "runs": 24},
                        {"label": f"{label}-lavd-template-default", "profile": "lavd-test", "runs": 8},
                    ],
                }
                before = len(self.rows)
                self.quality_arm_rows(phase, arm_spec, stage, model_dir=model_dir)
                cell = cells.get(label)
                for row in self.rows[before:]:
                    template_scope = (
                        "template-default"
                        if row["cell"].endswith("-lavd-template-default")
                        else "explicit-low"
                    )
                    row["config"] = {
                        **(row["config"] or {}),
                        "weights": weights_name,
                        "model_dir": model_dir,
                        "matched_config": config_name,
                        "template_scope": template_scope,
                    }
                    row["outcomes"] = {
                        **(row["outcomes"] or {}),
                        "qad_phase_completion_receipt": completed,
                    }
                    row["notes"] = [
                        *row["notes"],
                        "same R26 image, rig and prompts; only the mounted checkpoint differs; small-sample observations are not fidelity proofs",
                        "template-default and explicit-low rows are distinct identities and are never pooled",
                    ]
                    if row["kind"] == "boot" and cell:
                        row["outcomes"]["qad_ledger_booted"] = cell.get("booted")
                        row["sources"].append(str(self.root.file("qad-matched-ledger.json")))
                        ledger_matches = (
                            ledger.get("image") == rt.IMAGE
                            and cell.get("config") == config_name
                            and cell.get("weights") == weights_name
                            and cell.get("model_dir") == weights[weights_name]
                        )
                        if (
                            row["status"] == "running"
                            and not row["failure_classes"]
                            and ledger_matches
                            and isinstance(cell.get("booted"), bool)
                        ):
                            row["status"] = "pass" if cell["booted"] else "fail"
                            row["failure_classes"] = [] if cell["booted"] else ["runtime-protocol"]
                            row["source_state"] = "validated-qad-boot-ledger"
                            row["notes"].append(
                                "No separate boot gate was retained; readiness is recorded by the matched per-arm ledger."
                            )
                    if cell and cell.get("error") and row["kind"] == "boot":
                        row["failure_classes"] = sorted({*row["failure_classes"], "runtime-protocol"})
                        row["outcomes"]["ledger_error"] = cell["error"]

    # -- agentic prefix-cache reuse (agent_cache_probe.py)
    def agent_cache_rows(self, phase: str) -> None:
        stage = self.phase_stages[phase]
        prefix = stage_prefix(stage)
        self.plan_sources[phase] = (
            "scripts/r26/agent_cache_probe.py schema r26-agent-cache/v1; "
            "ARMS x series (vram, lmcache), 8 sessions x 12 turns"
        )
        optional_checks = {"lmcache_response_stats_exposed"}
        for arm_name, image in AGENT_CACHE_ARMS:
            receipt_name = f"{prefix}agent-cache-{arm_name}.json"
            receipt = self.root.json(receipt_name)
            schema_ok = isinstance(receipt, dict) and receipt.get("schema") == AGENT_CACHE_SCHEMA
            if isinstance(receipt, dict) and not schema_ok:
                self.root.read_errors.append({
                    "path": str(self.root.file(receipt_name)),
                    "error": f"expected schema {AGENT_CACHE_SCHEMA}, observed {receipt.get('schema')!r}",
                })
            summary = receipt.get("summary") if schema_ok and isinstance(receipt.get("summary"), dict) else {}
            config = receipt.get("config") if schema_ok and isinstance(receipt.get("config"), dict) else {}
            producer_gates = [
                gate for gate in (receipt.get("gates") if schema_ok else []) or []
                if isinstance(gate, dict)
            ]
            series_records = {
                row.get("name"): row
                for row in config.get("series", [])
                if isinstance(row, dict) and isinstance(row.get("name"), str)
            }
            configured_requests = (
                config.get("sessions") * config.get("turns_per_session")
                if isinstance(config.get("sessions"), int)
                and isinstance(config.get("turns_per_session"), int)
                else None
            )
            for cache in AGENT_CACHE_SERIES:
                label = f"agent-cache-{arm_name}-{cache}"
                boot = self.boot_row(
                    phase,
                    label,
                    expected_image=image,
                    stage=stage,
                    expected={"tp": 4, "dcp": 4, "spec": "mtp0", "cache": cache, "kv": "fp8_ds_mla"},
                )
                gate_name = f"runtime:agent-cache:{arm_name}:{cache}"
                observation_name = f"observation:agent-cache-hit-rate-at-least-0.95:{arm_name}:{cache}"
                runtime_gate = next((gate for gate in producer_gates if gate.get("name") == gate_name), None)
                observation_gate = next((gate for gate in producer_gates if gate.get("name") == observation_name), None)
                series = summary.get(cache) if isinstance(summary.get(cache), dict) else None
                series_record = series_records.get(cache)
                if not schema_ok or series is None:
                    boot_failed = boot["status"] == "fail" or (
                        boot["status"] == "missing" and self.phase_state(phase).startswith("finished")
                    )
                    self.row(
                        phase=phase,
                        cell=label,
                        kind="agent-cache",
                        arm=boot["arm"],
                        image=digest_of(image),
                        stage=stage,
                        source_state="unavailable" if receipt is None else "invalid-schema",
                        status="not-run" if receipt is None and boot_failed else ("missing" if receipt is None else "fail"),
                        failure_classes=[] if receipt is None else ["source-schema"],
                        gates=[gate_name],
                        sources=[str(self.root.file(receipt_name))] if receipt else [],
                        outcomes={"expected_schema": AGENT_CACHE_SCHEMA},
                    )
                    continue
                checks = runtime_gate.get("checks") if isinstance(runtime_gate, dict) and isinstance(runtime_gate.get("checks"), dict) else {}
                hard_failed_checks = sorted(
                    key for key, value in checks.items()
                    if key not in optional_checks and value is not True
                )
                optional_gaps = sorted(
                    key for key in optional_checks if checks.get(key) is not True
                )
                series_errors = (
                    series_record.get("errors")
                    if isinstance(series_record, dict) and isinstance(series_record.get("errors"), list)
                    else []
                )
                requests_expected = series.get("requests_expected")
                requests_completed = series.get("requests_completed")
                runtime_complete = (
                    configured_requests == 96
                    and requests_expected == configured_requests
                    and requests_completed == requests_expected
                    and not hard_failed_checks
                    and not series_errors
                )
                if not runtime_complete:
                    status, classes = "fail", ["runtime-protocol"]
                elif optional_gaps:
                    status, classes = "flagged", ["optional-telemetry-gap"]
                else:
                    status, classes = "pass", []
                hit_rate = series.get("hit_rate")
                metric_deltas = series.get("server_metric_deltas") if isinstance(series.get("server_metric_deltas"), dict) else {}
                cached_prompt_tokens = metric_deltas.get("prompt_tokens_cached")
                reusable_prefix_tokens = series.get("expected_reusable_tokens")
                recomputed_ratio = (
                    cached_prompt_tokens / reusable_prefix_tokens
                    if isinstance(cached_prompt_tokens, (int, float))
                    and isinstance(reusable_prefix_tokens, (int, float))
                    and reusable_prefix_tokens > 0
                    else None
                )
                self.row(
                    phase=phase,
                    cell=label,
                    kind="agent-cache",
                    arm=boot["arm"],
                    image=digest_of(image),
                    stage=stage,
                    source_state="available",
                    status=status,
                    failure_classes=classes,
                    outcomes={
                        "runtime_completion": {
                            "passed": runtime_complete,
                            "configured_requests": configured_requests,
                            "requests_expected": requests_expected,
                            "requests_completed": requests_completed,
                            "hard_failed_checks": hard_failed_checks,
                            "errors": series_errors,
                        },
                        "telemetry_observation": {
                            "optional_gaps": optional_gaps,
                            "per_request_cached_token_stats_required_for_runtime": False,
                            "aggregate_cached_over_reusable_prompt_token_ratio": hit_rate,
                            "recomputed_ratio": recomputed_ratio,
                            "producer_ratio_matches_recomputed": (
                                isinstance(hit_rate, (int, float))
                                and isinstance(recomputed_ratio, (int, float))
                                and abs(hit_rate - recomputed_ratio) <= 1e-12
                            ),
                            "aggregate_cached_prompt_tokens": cached_prompt_tokens,
                            "aggregate_tokenizer_reusable_prefix_tokens": reusable_prefix_tokens,
                            "ratio_definition": "sum(round metrics_delta.prompt_tokens_cached) / sum(turn expected_reusable_tokens where expected_reusable_tokens > 0)",
                            "reusable_prefix_definition": "tokenizer longest-common-prefix with the previous turn",
                            "prefix_cache_query_ratio": False,
                            "producer_threshold_observation": observation_gate,
                            "production_95_percent_claim": False,
                        },
                        "series_summary": series,
                        "producer_runtime_gate": runtime_gate,
                        "producer_overall_runtime_integrity_passed": summary.get("runtime_integrity_passed"),
                    },
                    gates=[gate_name],
                    sources=[str(self.root.file(receipt_name))],
                    notes=[
                        "all 96 streamed turns are runtime completion; optional per-response cached-token stats are a telemetry observation",
                        "hit ratio is aggregate cached prompt tokens divided by tokenizer-derived reusable prior-turn prefix tokens on this short trace; it is not a prefix-cache query ratio or production95% claim",
                        "no attribution to private fixes",
                    ],
                )

    # -- scheduler phase
    def scheduler_rows(self, phase: str) -> None:
        cells, source = scheduler_plan(self.root)
        self.plan_sources[phase] = source
        summary = self.root.json("scheduler/phase-summary.json")
        invocations = (summary or {}).get("invocations", {}) if isinstance(summary, dict) else {}
        # scheduler_phase.py boots: <candidate.key>-scheduler-vram, <factor.name>-boot (overlay), <candidate.key>-scheduler-lmcache
        candidates = {"official-r26": rt.IMAGE, "drock-r26-overlay": rt.OVERLAY_IMAGE}
        for key, image in candidates.items():
            self.boot_row(phase, f"{key}-scheduler-vram", expected_image=image, expected={"tp": 4, "dcp": 4, "spec": "mtp0", "cache": "vram", "kv": "fp8_ds_mla"})
        for cell in cells:
            if cell.get("series") == "mixed-factor":
                self.boot_row(phase, f"{cell['name']}-boot", expected_image=rt.OVERLAY_IMAGE, expected={"tp": 4, "dcp": 4, "spec": "mtp0", "cache": "vram", "kv": "fp8_ds_mla", "factor": cell.get("factor")})
        for key, image in candidates.items():
            self.boot_row(phase, f"{key}-scheduler-lmcache", expected_image=image, expected={"tp": 4, "dcp": 4, "spec": "mtp0", "cache": "lmcache", "kv": "fp8_ds_mla"})
        for cell in cells:
            name = cell["name"]
            record = invocations.get(name) if isinstance(invocations, dict) else None
            policy = self.root.json(f"scheduler/{name}-policy.json")
            receipt = self.root.json(f"scheduler/{name}.json")
            window = self.root.window(name)
            gate_names = [f"scheduler-{name}-policy-effective", f"scheduler-{name}-workload-complete", f"scheduler-{name}-no-lost-or-corrupt", f"scheduler-{name}-live-updates"]
            status_word = (record or {}).get("status")
            unsupported = (policy or {}).get("unsupported") if isinstance(policy, dict) else None
            if status_word is None and isinstance(policy, dict) and policy.get("passed") is False and unsupported:
                status_word = "unsupported_or_configuration_failed"
            elif status_word is None and isinstance(receipt, dict):
                status_word = "receipt-present"
            classes: list[str] = []
            if status_word is None:
                status = "missing"
            elif status_word == "unsupported_or_configuration_failed":
                if cell["image"] == rt.IMAGE and unsupported:
                    status = "unsupported"
                    classes = ["unsupported-feature"]
                else:
                    status = "fail"
                    classes = ["unsupported-feature" if unsupported else "configuration"]
            elif status_word in ("boot_failed", "phase_exception"):
                status, classes = "fail", ["runtime-protocol"]
            else:
                lost = self.gates.passed(f"scheduler-{name}-no-lost-or-corrupt")
                complete = self.gates.passed(f"scheduler-{name}-workload-complete")
                updates = self.gates.passed(f"scheduler-{name}-live-updates")
                if lost is False:
                    classes.append("lost-or-corrupt-response")
                if complete is False:
                    classes.append("runtime-protocol")
                if updates is False:
                    classes.append("live-update")
                status = "fail" if classes else ("pass" if complete else "running")
            speed = None
            if isinstance(receipt, dict) and isinstance(receipt.get("cells"), list):
                speed = {"cells": [self.scheduler_cell_speed(cell_data) for cell_data in receipt["cells"] if isinstance(cell_data, dict)]}
            self.row(phase=phase, cell=f"scheduler:{name}", kind=f"scheduler:{cell.get('series')}", arm=self.arm(cell["image"]), status=status, failure_classes=classes,
                     config={"scenario_count": cell.get("scenario_count"), "factor": cell.get("factor"), "policy_schema": (policy or {}).get("schema") if isinstance(policy, dict) else None},
                     outcomes={"invocation_status": status_word, "unsupported": unsupported, "observed_cells": (record or {}).get("observed_cells"), "complete_cells": (record or {}).get("complete_cells")},
                     speed=speed, window=window, sources=[str(self.root.file(f"scheduler/{name}.json"))] if receipt else [], gates=[gate for gate in gate_names if gate in self.gates.by_stage.get("root", {})],
                     notes=["fairness/latency figures are speed-class and need a clean GPU window; lost/corrupt and policy readback are functional"])

    def scheduler_recheck_rows(self, phase: str) -> None:
        """Normalize the shipped scheduler/API recheck from its recorded plan."""
        stage = self.phase_stages[phase]
        prefix = stage_prefix(stage)
        output_dir = "scheduler-recheck"
        plan_name = f"{prefix}{output_dir}/phase-plan.json"
        summary_name = f"{prefix}{output_dir}/phase-summary.json"
        plan = self.root.json(plan_name)
        summary = self.root.json(summary_name)
        plan_ok = isinstance(plan, dict) and plan.get("phase") == "scheduler-recheck"
        summary_ok = isinstance(summary, dict) and summary.get("phase") == "scheduler-recheck"
        self.plan_sources[phase] = (
            f"{plan_name} written by scripts/r26/scheduler_recheck_phase.py phase_plan(); "
            f"recorded size {(plan.get('plan_size') if plan_ok else None)}"
        )
        if isinstance(plan, dict) and not plan_ok:
            self.root.read_errors.append({
                "path": str(self.root.file(plan_name)),
                "error": "expected phase discriminator 'scheduler-recheck'",
            })
        if isinstance(summary, dict) and not summary_ok:
            self.root.read_errors.append({
                "path": str(self.root.file(summary_name)),
                "error": "expected phase discriminator 'scheduler-recheck'",
            })
        candidates = {
            row.get("key"): row
            for row in (
                plan.get("candidate_and_tool_settings", {}).get("candidates", [])
                if plan_ok and isinstance(plan.get("candidate_and_tool_settings"), dict)
                else []
            )
            if isinstance(row, dict) and isinstance(row.get("key"), str)
        }
        groups = plan.get("boot_groups") if plan_ok and isinstance(plan.get("boot_groups"), list) else []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("name"), str):
                continue
            candidate = group.get("candidate") if isinstance(group.get("candidate"), dict) else {}
            self.boot_row(
                phase,
                f"{output_dir}/{group['name']}",
                expected_image=candidate.get("image"),
                expected={
                    "tp": 4,
                    "dcp": 4,
                    "spec": "mtp0",
                    "cache": group.get("cache_mode"),
                    "kv": "fp8_ds_mla",
                    "extra_env": group.get("boot_environment"),
                    "extra_args": group.get("boot_extra_args"),
                },
                stage=stage,
            )
        invocations = summary.get("invocations") if summary_ok and isinstance(summary.get("invocations"), dict) else {}
        plans = plan.get("plans") if plan_ok and isinstance(plan.get("plans"), list) else []
        for plan_row in plans:
            if not isinstance(plan_row, dict) or not isinstance(plan_row.get("name"), str):
                continue
            candidate = candidates.get(plan_row.get("candidate_key"), {})
            receipt_name = f"{prefix}{output_dir}/{plan_row['name']}.json"
            receipt = self.root.json(receipt_name)
            self.scheduler_recheck_result_row(
                phase=phase,
                stage=stage,
                cell_prefix="scheduler-recheck",
                output_dir=output_dir,
                plan=plan_row,
                candidate=candidate,
                record=invocations.get(plan_row["name"]),
                receipt=receipt,
                plan_source=plan_name,
                summary_source=summary_name,
            )
        if not plan_ok:
            self.row(
                phase=phase,
                cell="scheduler-recheck:plan",
                kind="scheduler-recheck:source",
                stage=stage,
                status="missing" if plan is None else "fail",
                failure_classes=[] if plan is None else ["source-schema"],
                source_state="unavailable" if plan is None else "invalid-schema",
                outcomes={"expected_phase": "scheduler-recheck"},
                sources=[str(self.root.file(plan_name))] if plan is not None else [],
                notes=["scheduler rows are not inferred when the recorded producer plan is unavailable"],
            )

    @staticmethod
    def scheduler_lane_assertion_only(record: dict[str, Any] | None) -> bool:
        if not isinstance(record, dict) or record.get("status") != "api_or_boot_structure_failed":
            return False
        detail = record.get("detail") if isinstance(record.get("detail"), dict) else {}
        discovery = detail.get("discovery") if isinstance(detail.get("discovery"), dict) else {}
        return bool(
            discovery.get("schema") == "overlay-r26"
            and discovery.get("passed") is False
            and not discovery.get("boot_structure_mismatches")
            and isinstance(discovery.get("effective_lane_budget"), int)
            and discovery["effective_lane_budget"] >= 1
        )

    def scheduler_recheck_result_row(
        self,
        *,
        phase: str,
        stage: str,
        cell_prefix: str,
        output_dir: str,
        plan: dict[str, Any],
        candidate: dict[str, Any],
        record: dict[str, Any] | None,
        receipt: dict[str, Any] | list | None,
        plan_source: str,
        summary_source: str,
    ) -> dict[str, Any]:
        name = plan["name"]
        prefix = stage_prefix(stage)
        cells = receipt.get("cells") if isinstance(receipt, dict) and isinstance(receipt.get("cells"), list) else []
        expected_cells = plan.get("scenario_count")
        core_gate_names = {
            "entire_deterministic_trace_offered",
            "all_stream_protocol_responses_complete",
            "server_idle_after_drain",
        }
        core_failures: list[dict[str, Any]] = []
        observation_failures: list[dict[str, Any]] = []
        completion_counts: Counter = Counter()
        for cell in cells:
            if not isinstance(cell, dict):
                core_failures.append({"cell": cell, "reason": "cell is not an object"})
                continue
            gate_rows = [
                gate for gate in cell.get("gates", [])
                if isinstance(gate, dict) and isinstance(gate.get("name"), str)
            ]
            by_name = {gate["name"]: gate for gate in gate_rows}
            identity = {
                "profile": cell.get("profile"),
                "concurrency": cell.get("concurrency"),
                "repeat": cell.get("repeat"),
            }
            for gate_name in core_gate_names:
                if by_name.get(gate_name, {}).get("passed") is not True:
                    core_failures.append({**identity, "gate": by_name.get(gate_name), "required": gate_name})
            observation_failures.extend(
                {**identity, "gate": gate}
                for gate in gate_rows
                if gate.get("passed") is False and gate["name"] not in core_gate_names
            )
            completion = cell.get("summary", {}).get("completion_classification")
            if isinstance(completion, dict) and isinstance(completion.get("counts"), dict):
                completion_counts.update({
                    str(key): int(value)
                    for key, value in completion["counts"].items()
                    if isinstance(value, int)
                })
        runtime_complete = bool(
            isinstance(receipt, dict)
            and isinstance(expected_cells, int)
            and len(cells) == expected_cells
            and not core_failures
        )
        lane_assertion = self.scheduler_lane_assertion_only(record)
        producer_status = record.get("status") if isinstance(record, dict) else None
        if runtime_complete:
            timing_only = bool(observation_failures) and all(
                failure.get("gate", {}).get("name") == "model_prefill_decode_work_split_observed"
                for failure in observation_failures
            )
            if not observation_failures:
                status, classes = "pass", []
            elif timing_only:
                status, classes = "flagged", ["timing-telemetry-gap"]
            else:
                status, classes = "flagged", ["scheduler-qos-observation"]
        elif lane_assertion and receipt is None:
            status, classes = "not-run", []
        elif receipt is None and producer_status is None:
            status, classes = "missing", []
        elif receipt is None and producer_status in (
            "boot_failed",
            "api_or_boot_structure_failed",
            "harness_configuration_failed",
        ):
            status, classes = "not-run", ["runtime-protocol"] if producer_status == "boot_failed" else ["configuration"]
        else:
            status, classes = "fail", ["runtime-protocol"]
        speed = (
            {"cells": [self.scheduler_cell_speed(cell) for cell in cells if isinstance(cell, dict)]}
            if runtime_complete
            else None
        )
        scheduler_gate_names = [
            gate["name"]
            for gate in self.gates.with_prefix(f"scheduler-recheck-{name}-", stage)
        ]
        sources = list(dict.fromkeys(
            str(self.root.file(source))
            for source, available in (
                (plan_source, self.root.file(plan_source).is_file()),
                (summary_source, self.root.file(summary_source).is_file()),
                (f"{output_dir}/{name}.json" if stage == "root" else f"{stage}/{output_dir}/{name}.json", isinstance(receipt, dict)),
            )
            if available
        ))
        off_policy = isinstance(plan.get("policy"), dict) and plan["policy"].get("compute_share") is None
        row = self.row(
            phase=phase,
            cell=f"{cell_prefix}:{name}",
            kind=f"{cell_prefix}:{plan.get('series')}",
            arm=self.arm(candidate.get("image")),
            image=digest_of(candidate.get("image")),
            config={
                "candidate": plan.get("candidate_key"),
                "policy": plan.get("policy"),
                "cache": plan.get("cache_mode"),
                "scenario_count": expected_cells,
                "headline": plan.get("headline"),
            },
            stage=stage,
            source_state="available" if isinstance(receipt, dict) else "unavailable",
            status=status,
            failure_classes=classes,
            outcomes={
                "runtime_completion": {
                    "passed": runtime_complete,
                    "expected_scenarios": expected_cells,
                    "observed_scenarios": len(cells),
                    "core_gate_names": sorted(core_gate_names),
                    "core_failures": core_failures,
                    "completion_classifications": dict(sorted(completion_counts.items())),
                    "model_answer_correctness_assessed": False,
                },
                "observation_and_qos": {
                    "failed_gates": observation_failures,
                    "off_policy": off_policy,
                    "timing_split_required_for_serving_completion": False,
                },
                "producer_status": producer_status,
                "producer_failure_class": record.get("failure_class") if isinstance(record, dict) else None,
                "scenario_attempt_classification": record.get("scenario_attempt_classification") if isinstance(record, dict) else None,
                "positive_lane_budget_assertion_rejected_old_probe": lane_assertion,
            },
            speed=speed,
            window=self.root.window(f"{prefix}{output_dir}/{name}"),
            sources=sources,
            gates=scheduler_gate_names,
            notes=[
                "runtime completion, scheduler observations, QoS and unassessed answer quality are separate",
                "a missing model prefill/decode timing split on off-policy controls is a telemetry gap, not failed serving",
                "budget-limited streamed responses are protocol-complete; no semantic-quality claim is made",
            ],
        )
        return row

    @staticmethod
    def scheduler_cell_speed(cell: dict[str, Any]) -> dict[str, Any]:
        summary = cell.get("summary") if isinstance(cell.get("summary"), dict) else {}
        ttft = summary.get("hot_ttft_seconds") if isinstance(summary.get("hot_ttft_seconds"), dict) else {}
        normalization = summary.get("normalization") if isinstance(summary.get("normalization"), dict) else {}
        return {
            "profile": cell.get("profile"), "concurrency": cell.get("concurrency"), "repeat": cell.get("repeat"), "status": cell.get("status"),
            "hot_ttft_p50": ttft.get("p50"), "hot_ttft_p95": ttft.get("p95"),
            "hot_turns_completed_per_measurement_second": normalization.get("hot_turns_completed_per_measurement_second"),
        }

    # -- peer / #599 phase
    def peer_rows(self, phase: str) -> None:
        self.plan_sources[phase] = "scripts/r26/peer_phase.py main() mode sequence"
        for mode, stem, meaning in PEER_MODES:
            receipt = self.root.json(f"{stem}.json")
            passed = self.gates.passed(f"peer599:{mode}")
            prerequisite = self.gates.detail("peer599:forced-direct-safe-prerequisite")
            skipped = mode == "forced-direct" and receipt is None and isinstance(prerequisite, dict) and prerequisite.get("not_run")
            if stem.startswith("peer599-model-"):  # model_path boots the overlay through runtime.boot(label)
                boot = self.boot_row(phase, stem, expected_image=rt.OVERLAY_IMAGE, expected={"tp": 4, "dcp": 4, "spec": "mtp0", "cache": "vram", "kv": "fp8_ds_mla", "extra_args": ["--dcp-comm-backend", "a2a"]})
                boot["arm"] = "hardware-599"
                if skipped and boot["status"] == "missing":
                    boot["status"] = "skipped"
                    boot["outcomes"] = {"reason": prerequisite.get("reason")}
            if skipped:
                self.row(phase=phase, cell=stem, kind="peer-599", arm="hardware-599", image=digest_of(rt.OVERLAY_IMAGE), status="skipped",
                         outcomes={"reason": prerequisite.get("reason")}, gates=["peer599:forced-direct-safe-prerequisite"], notes=[meaning])
                continue
            if receipt is None:
                self.row(phase=phase, cell=stem, kind="peer-599", arm="hardware-599", image=digest_of(rt.OVERLAY_IMAGE), status="fail" if passed is False else "missing",
                         failure_classes=["hardware-599"] if passed is False else [], window=self.root.window(f"peer-{mode}") or self.root.window(stem), gates=[f"peer599:{mode}"], notes=[meaning])
                continue
            outcomes = {key: receipt.get(key) for key in ("passed", "scope", "error", "direct_path_selected", "peer_fallback_logged") if key in receipt}
            if isinstance(receipt.get("requests"), list):
                outcomes["requests_correct"] = sum(bool(row.get("correct")) for row in receipt["requests"] if isinstance(row, dict))
                outcomes["requests"] = len(receipt["requests"])
            self.row(phase=phase, cell=stem, kind="peer-599", arm="hardware-599", image=digest_of(rt.OVERLAY_IMAGE), status="pass" if passed else "fail",
                     failure_classes=[] if passed else ["hardware-599"], outcomes=outcomes, window=self.root.window(f"peer-{mode}") or self.root.window(stem),
                     sources=[str(self.root.file(f"{stem}.json"))], gates=[f"peer599:{mode}"],
                     notes=[meaning, "physical checks apply only to this host's topology"])

    # -- driver
    BUILDERS = {
        "quality_phase.py": "quality_rows",
        "cache_phase.py": "cache_rows",
        "cache_recheck_phase.py": "cache_recheck_rows",
        "scheduler_phase.py": "scheduler_rows",
        "scheduler_recheck_phase.py": "scheduler_recheck_rows",
        "peer_phase.py": "peer_rows",
        "overlay_quality_recheck.py": "recheck_rows",
        "realistic_acceptance_phase.py": "realistic_rows",
        "clean_rerun_phase.py": "clean_rerun_rows",
        "qad_matched_phase.py": "qad_matched_rows",
        "agent_cache_probe.py": "agent_cache_rows",
    }

    def build(self) -> dict[str, Any]:
        phases, phase_source = phase_plan(self.root)
        plan_sources = {"root": phase_source}
        planned = [(phase, "root") for phase in phases]
        # Post-coordinator stages belong only to the published R26 root.
        if self.weights["kind"] == "published":
            followups, plan_sources[FOLLOWUPS_DIR] = followup_phase_plan(self.root)
            clean, plan_sources[CLEAN_RERUNS_DIR] = clean_phase_plan(self.root)
            planned += [(phase, FOLLOWUPS_DIR) for phase in followups]
            planned += [(phase, CLEAN_RERUNS_DIR) for phase in clean]
        for phase, stage in planned:
            name = phase["name"]
            self.phase_stages[name] = stage
            self.phase_windows[name] = self.root.window(f"{stage_prefix(stage)}phase-{name}") or {}
            script, args = phase.get("script"), phase.get("args") or []
            if script == "matrix_phase.py":
                section = args[args.index("--section") + 1] if "--section" in args else "all"
                for part in (("smoke", "priority", "matrix", "tuning", "tp2") if section == "all" else (section,)):
                    self.matrix_rows(name, part)
            elif script in self.BUILDERS:
                getattr(self, self.BUILDERS[script])(name)
            else:
                self.coverage_gaps.append(f"phase {name} uses unknown script {script}; no normalizer")
        self.resolve_rechecks()
        self.resolve_clean_reruns()
        # rows that exist on disk but no plan claims them (never silently dropped)
        for stage in ("root", *SUBSTAGES):
            planned_boots = {row["cell"] for row in self.rows if row["kind"] == "boot" and row["stage"] == stage}
            for path in self.root.glob(f"{stage_prefix(stage)}*.launch.json"):
                label = path.name[: -len(".launch.json")]
                if label not in planned_boots:
                    self.boot_row("unplanned", label, stage=stage)
                    self.coverage_gaps.append(f"boot {stage_prefix(stage)}{label} has a launch receipt but no plan entry")
        summaries = [self.phase_summary(phase, stage) for phase, stage in planned]
        return self.assemble(phases, plan_sources, summaries)

    def phase_summary(self, phase: dict[str, Any], stage: str) -> dict[str, Any]:
        name = phase["name"]
        rows = [row for row in self.rows if row["phase"] == name and row["stage"] == stage]
        counts = Counter(row["status"] for row in rows)
        state = self.phase_state(name)
        window = self.phase_windows.get(name) or {}
        if counts.get("missing"):
            self.coverage_gaps.append(f"{name}: {counts['missing']} planned cell(s) without receipt ({state})")
        return {
            "name": name, "stage": stage, "script": phase.get("script"), "args": phase.get("args") or [], "state": state, "plan_source": self.plan_sources.get(name),
            "window": {**window, "started_at_iso": iso(window.get("started_at")), "finished_at_iso": iso(window.get("finished_at"))} if window else None,
            "phase_execution_gate": self.gates.passed(f"phase-execution:{name}", stage),
            "planned_rows": len(rows), "status_counts": dict(sorted(counts.items())),
            "missing_cells": [f"{row['kind']} {row['cell']}" for row in rows if row["status"] == "missing"],
        }

    def assemble(self, phases: list[dict[str, Any]], plan_sources: dict[str, str], phase_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        speed_rows = [row for row in self.rows if row["speed"] is not None]
        speed_windows = {key: [] for key in ("clean", "foreign-gpu-overlap", "gpu-health-fault", "coverage-gap", "no-window")}
        for row in speed_rows:
            speed_windows[row["speed_qualification"]].append({"cell": row["cell"], "kind": row["kind"], "arm": row["arm"], "phase": row["phase"], "stage": row["stage"], "window": row["window"], "isolation": row["isolation"]})
        failed_gates = []
        for gate in self.gates.rows:
            if gate.get("passed"):
                continue
            detail = gate.get("detail") if isinstance(gate.get("detail"), dict) else {}
            failed_gates.append({"name": gate["name"], "stage": gate["stage"], "line": gate["line"], "timestamp": iso(gate.get("timestamp")), "category": gate_category(gate["name"]),
                                 "gate_type": detail.get("gate_type"), "failure_class": self.failure_class_for_gate(gate), "phase": self.phase_for_timestamp(gate.get("timestamp"), gate["stage"])})
        resolved = [
            {
                "original": row["cell"],
                "kind": row["kind"],
                "original_status": (row["outcomes"] or {}).get("original_status"),
                "superseded_by": row["superseded_by"],
            }
            for row in self.rows if row["status"] == "superseded"
        ]
        partially_resolved = [
            {"cell": row["cell"], "stage": row["stage"], "checks": row["superseded_checks"]}
            for row in self.rows if row["superseded_checks"]
        ]
        speed_reruns = [
            {
                "parent": row["cell"],
                "parent_verdict": (row["outcomes"] or {}).get("original_speed_qualification"),
                "clean_rerun": row["superseded_by"],
            }
            for row in self.rows
            if row["stage"] == "root" and row["kind"] == "bench" and row["superseded_by"]
        ]
        rerun_ledger = self.root.json(CLEAN_RERUN_PLAN)
        by_arm: dict[str, Counter] = {arm: Counter() for arm in ARM_ORDER}
        for row in self.rows:
            if row["kind"].startswith("quality-probe") and row["status"] != "superseded" and isinstance(row["outcomes"], dict) and row["outcomes"].get("derived_categories"):
                by_arm.setdefault(row["arm"], Counter()).update(row["outcomes"]["derived_categories"])
        status_totals = Counter(row["status"] for row in self.rows)
        planned = len(self.rows)
        with_receipt = sum(1 for row in self.rows if row["status"] not in ("missing",))
        isolation_summary = self.isolation.summary()
        battery_log = self.root.text("battery.log") or ""
        mode_match = re.search(r"isolation=(\w+)", battery_log)
        provenance = self.root.json("provenance.json")
        history_kv = json.loads(HISTORY_KV_EVIDENCE.read_text()) if HISTORY_KV_EVIDENCE.is_file() else None
        try:
            qad_phases, qad_source = source_assigned_phases("run_qad_battery.py")
        except (OSError, SyntaxError, ValueError) as error:
            qad_phases, qad_source = [], f"unavailable: {type(error).__name__}: {error}"
        qad_plan = self.root.json("qad-battery-plan.json")
        phase_execution_complete = all(
            summary["state"].startswith("finished") for summary in phase_summaries
        )
        pending_rows = sum(status_totals.get(status, 0) for status in ("missing", "running"))
        execution_complete = pending_rows == 0 and phase_execution_complete
        disqualifying_rows = sum(status_totals.get(status, 0) for status in ("fail", "flagged", "not-run"))
        qualification_passed = execution_complete and disqualifying_rows == 0
        qualification_status = (
            "incomplete" if not execution_complete
            else "passed" if qualification_passed
            else "failed"
        )
        report = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "generator": {"script": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "root": str(self.root.path),
            "weights": {**{key: value for key, value in self.weights.items() if key != "verification"}, "published": str(PUBLISHED_WEIGHTS),
                        "checkpoint_verification": self.weights.get("verification"),
                        "rule": "arms are image x mounted checkpoint; a candidate checkpoint is reported as its own arm and is never pooled with published-weight rows",
                        "arms_present": sorted({row["arm"] for row in self.rows})},
            "completeness": {
                "planned_rows": planned,
                "rows_with_receipt": with_receipt,
                "missing_rows": status_totals.get("missing", 0),
                "pending_rows": pending_rows,
                "phase_execution_complete": phase_execution_complete,
                "status_totals": dict(sorted(status_totals.items())),
                "claim": (
                    "complete execution record"
                    if execution_complete
                    else "INCOMPLETE: planned receipts or phase completions remain; nothing here is a final verdict"
                ),
            },
            "qualification": {
                "status": qualification_status,
                "passed": qualification_passed,
                "rule": "passed is possible only after every phase finishes, no receipt is missing/running, and no non-superseded row is failed, flagged or not-run",
                "disqualifying_rows": disqualifying_rows,
            },
            "provenance": {
                "images": {arm: image for image, arm in ARMS.items()},
                "runtime_constants_source": str(HERE / "runtime.py"),
                "phase_plan_sources": plan_sources,
                "plan_sources": self.plan_sources,
                "followups": {
                    "directory": str(self.root.path / FOLLOWUPS_DIR),
                    "execution_plan": str(self.root.file(FOLLOWUP_EXECUTION_PLAN)),
                    "recheck_map": self.recheck_map_summary,
                    "resolved_originals": resolved,
                    "partially_resolved_checks": partially_resolved,
                    "cache_recheck_summary": str(self.root.file(CACHE_RECHECK_SUMMARY)),
                    "cache_recheck_complete": str(self.root.file(CACHE_RECHECK_COMPLETE)),
                    "rule": "a source-backed changed-methodology recheck may supersede a named original; every original attempt and its observations remain recorded",
                },
                "clean_reruns": {
                    "directory": str(self.root.path / CLEAN_RERUNS_DIR),
                    "ledger_present": isinstance(rerun_ledger, dict),
                    "ledger_planned": (rerun_ledger or {}).get("planned") if isinstance(rerun_ledger, dict) else None,
                    "ledger_result_occurrences": (
                        sum(len(entry.get("cells") or []) for entry in rerun_ledger.get("reruns", []) if isinstance(entry, dict))
                        if isinstance(rerun_ledger, dict)
                        else 0
                    ),
                    "ledger_unique_result_labels": (
                        len({
                            cell.get("result_label")
                            for entry in rerun_ledger.get("reruns", [])
                            if isinstance(entry, dict)
                            for cell in entry.get("cells") or []
                            if isinstance(cell, dict) and isinstance(cell.get("result_label"), str)
                        })
                        if isinstance(rerun_ledger, dict)
                        else 0
                    ),
                    "ledger_unique_clean_result_labels": (
                        len({
                            cell.get("result_label")
                            for entry in rerun_ledger.get("reruns", [])
                            if isinstance(entry, dict)
                            for cell in entry.get("cells") or []
                            if isinstance(cell, dict)
                            and cell.get("clean") is True
                            and isinstance(cell.get("result_label"), str)
                        })
                        if isinstance(rerun_ledger, dict)
                        else 0
                    ),
                    "speed_superseded": speed_reruns,
                    "final_diagnostics": str(self.root.file(FINAL_DIAGNOSTICS)),
                    "lane_cap_recheck": str(self.root.file(LANE_RECHECK)),
                    "rule": "one schema-checked canonical same-label rerun supplies speed only after matching launch/checkpoint/config and clean command plus steady windows; every ledger attempt remains in row outcomes",
                },
                "qad_runbook": {
                    "plan_receipt": qad_plan,
                    "source": qad_source,
                    "planned_phase_count": len(qad_phases),
                    "phases": qad_phases,
                    "performance_measurements_ingested": (
                        self.weights["kind"] == "candidate-checkpoint"
                        and any(row["speed"] is not None for row in self.rows)
                    ),
                    "completion_or_pass_claimed_from_plan": False,
                    "status": (
                        "candidate checkpoint report root"
                        if self.weights["kind"] == "candidate-checkpoint"
                        else "planned external runbook; no QAD measurements ingested"
                    ),
                },
                "reasoning_parameter_protocol_proof": {
                    "path": str(self.root.file(REASONING_PROTOCOL_PROOF)),
                    "present": self.reasoning_proof is not None,
                    "top_level_low_maps_to_template_kwargs_for": (
                        sorted(self.reasoning_proof)
                        if isinstance(self.reasoning_proof, dict)
                        else []
                    ),
                },
                "qad_matched_ledger": self.root.json("qad-matched-ledger.json"),
                "tail_recovery": self.root.json("tail-recovery-plan.json"),
                "clock_comparison_rule": "Retained pre-reboot measurements and newly executed tail measurements may have different core VF offsets; do not present them as matched-clock performance comparisons.",
                "provenance_json": provenance,
                "excluded_initial_pass": {"directory": str(self.root.path / EXCLUDED_DIR), "ingested": False, "marker": self.root.json("initial-pass-exclusion.json"),
                                          "exists": (self.root.path / EXCLUDED_DIR).is_dir()},
                "gpu_isolation": {"mode": mode_match.group(1) if mode_match else None, "rule": "speed is qualified only when the command window is fully sampled (gap <= 10 s) and no sample shows a foreign GPU process", **isolation_summary},
                "kv_capacity_policy": {
                    "source": "server log line 'GPU KV cache size: N tokens' from <label>.docker.log / <label>-final.docker.log", "regex": SERVER_KV_RE.pattern,
                    "bench_metadata_used_for_capacity": False,
                    "historical_correction": {"file": str(HISTORY_KV_EVIDENCE), "summary_claims_to_correct": (history_kv or {}).get("summary_claims_to_correct")} if history_kv else {"file": str(HISTORY_KV_EVIDENCE), "present": False},
                },
                "acceptance_scopes": {
                    "acceptance-window-synthetic": "acceptance_probe.py: repeated fixed text, temperature 0, ignore_eos, 32768 max tokens; stress/ceiling only, never representative acceptance",
                    "acceptance-realistic": f"schema {REALISTIC_SCHEMA} from realistic_acceptance_phase.py; realistic prompts without ignore_eos",
                },
                "read_errors": self.root.read_errors,
            },
            "battery_state": {
                "started": battery_log.splitlines()[0] if battery_log else None,
                "current_phase": self.root.json("current-phase.json"),
                "phase_progress": self.root.json("phase-progress.json"),
                "executed": self.root.json("qualification-executed.json"),
                "interrupted": self.root.json("qualification-interrupted.json"),
                "production_restored": self.root.json("production-restored.json"),
                "baseline_recovery": self.root.json("baseline-recovery-after-interruption.json"),
                "followups": {
                    "current_phase": self.root.json(f"{FOLLOWUPS_DIR}/current-phase.json"),
                    "phase_progress": self.root.json(f"{FOLLOWUPS_DIR}/phase-progress.json"),
                    "executed": self.root.json(f"{FOLLOWUPS_DIR}/qualification-executed.json"),
                    "interrupted": self.root.json(f"{FOLLOWUPS_DIR}/qualification-interrupted.json"),
                    "rechecks_completed": self.root.json(f"{FOLLOWUPS_DIR}/overlay-quality-rechecks-completed.json"),
                },
                "clean_reruns": {
                    "current_phase": self.root.json(f"{CLEAN_RERUNS_DIR}/current-phase.json"),
                    "phase_progress": self.root.json(f"{CLEAN_RERUNS_DIR}/phase-progress.json"),
                    "executed": self.root.json(f"{CLEAN_RERUNS_DIR}/qualification-executed.json"),
                    "interrupted": self.root.json(f"{CLEAN_RERUNS_DIR}/qualification-interrupted.json"),
                    "completed": self.root.json(f"{CLEAN_RERUNS_DIR}/clean-rerun-completed.json"),
                    "production_restored": self.root.json(f"{CLEAN_RERUNS_DIR}/production-restored.json"),
                    "final_runtime_diagnostics": self.root.json(FINAL_DIAGNOSTICS),
                    "scheduler_lane_cap_recheck": self.root.json(LANE_RECHECK),
                },
            },
            "phases": phase_summaries,
            "rows": self.rows,
            "speed_windows": {"counts": {key: len(value) for key, value in speed_windows.items()}, **speed_windows},
            "gates": {"total": len(self.gates.rows), "passed": sum(1 for gate in self.gates.rows if gate.get("passed")), "failed": len(failed_gates),
                      "by_category": dict(sorted(Counter(gate_category(gate["name"]) for gate in self.gates.rows).items())),
                      "failed_by_class": dict(sorted(Counter(gate["failure_class"] for gate in failed_gates).items())), "failed_gates": failed_gates},
            "quality_derived_totals_by_arm": {arm: dict(sorted(counter.items())) for arm, counter in by_arm.items() if counter},
            "coverage_gaps": self.coverage_gaps,
            "sources": self.root.finalize_sources(),
        }
        return report

    def failure_class_for_gate(self, gate: dict[str, Any]) -> str:
        """Class of a failed gate, taken from the same-stage row that owns the gate's label when one exists."""
        name, stage = gate["name"], gate["stage"]
        detail = gate.get("detail") if isinstance(gate.get("detail"), dict) else {}
        if (stage, name) in self.superseded_gates:
            return "superseded-by-recheck"
        if name.startswith("runtime:agent-cache:"):
            checks = detail.get("checks") if isinstance(detail.get("checks"), dict) else {}
            failed = {key for key, value in checks.items() if value is not True}
            if failed and failed <= {"lmcache_response_stats_exposed"}:
                return "optional-telemetry-gap"
        if name == "phase-execution:agentic-prefix-cache-reuse":
            agent_rows = [
                row for row in self.rows
                if row["stage"] == stage and row["kind"] == "agent-cache"
            ]
            if agent_rows and all(
                ((row["outcomes"] or {}).get("runtime_completion") or {}).get("passed") is True
                for row in agent_rows
            ):
                return "optional-telemetry-gap"
        if name == "phase-execution:shipped-api-scheduler-qualification":
            return "scheduler-observation"
        if name.startswith("phase-execution:clean-speed-reruns"):
            return "resource-unavailable"
        if name == "clean-rerun-resource-availability":
            return "resource-unavailable"
        if name.startswith("clean-rerun:"):
            result_label = name.split(":", 1)[1]
            canonical = next((
                candidate for candidate in self.rows
                if candidate["stage"] == CLEAN_RERUNS_DIR
                and candidate["kind"] == "bench"
                and candidate["cell"] == result_label
            ), None)
            if canonical and canonical["speed_qualification"] == "clean":
                return "superseded-clean-attempt"
            return "clean-rerun-validation"
        parts = name.split(":")
        label = parts[1] if len(parts) > 1 else None
        row = None
        if label:
            row = next((row for row in self.rows if row["stage"] == stage and (row["cell"] == label or row["cell"].startswith(label + "-c") and label.endswith("-acceptance"))), None)
        if detail.get("not_run"):
            return "skipped-by-design"
        if row and row["status"] == "superseded":
            return "superseded-by-recheck"
        if row and row["status"] == "not-run":
            return "not-run-upstream"
        if row and row["status"] == "missing":
            return "missing-receipt"
        if row and row["failure_classes"]:
            priority = (
                "launch-config-incompatible", "wrong-final", "repetition", "parser-failure",
                "budget-limited", "model-quality-flag", "clean-rerun-validation",
                "timing-telemetry-gap", "optional-telemetry-gap", "cache-metadata",
                "configuration", "runtime-protocol",
            )
            return next((cls for cls in priority if cls in row["failure_classes"]), row["failure_classes"][0])
        if name.startswith("quality:"):
            return "model-quality"
        if name.startswith("cache:"):
            if stage == CLEAN_RERUNS_DIR and name == "cache:drock-overlay:#645:truthful-cache-events-and-replay":
                replay = next((
                    candidate for candidate in self.rows
                    if candidate["stage"] == CLEAN_RERUNS_DIR
                    and candidate["cell"] == "final-diagnostic:focused-overlay-replay"
                ), None)
                if replay and replay["failure_classes"]:
                    return replay["failure_classes"][0]
            if detail.get("gate_type") == "cache_effect":
                return "cache-metadata"
            cell = cache_gate_cell(name)
            owner = next((row for row in self.rows if row["stage"] == stage and row["cell"] == f"cache:{cell}"), None) if cell else None
            if owner and owner["status"] == "superseded":
                return "superseded-by-recheck"
            return cache_failure_class(gate)
        if name.startswith("scheduler-recheck-"):
            owners = [
                row for row in self.rows
                if row["stage"] == stage
                and row["cell"].startswith(("scheduler-recheck:", "scheduler-lane-recheck:"))
            ]
            owner = next((
                row for row in owners
                if name.startswith("scheduler-recheck-" + row["cell"].split(":", 1)[1] + "-")
            ), None)
            if owner and owner["status"] == "superseded":
                return "superseded-by-recheck"
            if owner and owner["failure_classes"]:
                return owner["failure_classes"][0]
            return "scheduler-observation"
        if name.startswith("scheduler-"):
            stem = name[len("scheduler-"):]
            for suffix in ("-policy-effective", "-workload-complete", "-no-lost-or-corrupt", "-live-updates", "-capture"):
                stem = stem.removesuffix(suffix)
            owner = next((row for row in self.rows if row["cell"] == f"scheduler:{stem}"), None)
            if owner and owner["failure_classes"]:
                return owner["failure_classes"][0]
            return "scheduler"
        if name.startswith("acceptance-window:"):
            return "performance-window"
        if name.startswith("configuration:"):
            # comparison gates key their detail by the receipts they compare; inherit the worst upstream state of those rows
            labels = nested_keys(detail)
            statuses = {row["status"] for row in self.rows if row["stage"] == stage and row["cell"] in labels}
            if "superseded" in statuses:
                return "superseded-by-recheck"
            if "not-run" in statuses:
                return "not-run-upstream"
        return gate_category(name)

    def phase_for_timestamp(self, ts: float | None, stage: str = "root") -> str | None:
        if ts is None:
            return None
        current = self.root.json(f"{stage_prefix(stage)}current-phase.json")
        if isinstance(current, dict):
            started = to_epoch(current.get("started_at"))
            running = not (self.phase_windows.get(current.get("phase")) or {}).get("finished_at")
            if started is not None and running and ts >= started:
                return current.get("phase")
        for name, window in self.phase_windows.items():
            if self.phase_stages.get(name, "root") != stage:
                continue
            start, end = window.get("started_at"), window.get("finished_at")
            if start is not None and ts >= start and (end is None or ts <= end + 1):
                return name
        return None


# ---------------------------------------------------------------- text tables
def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[fmt(value) for value in row] for row in rows]
    widths = [max(len(row[index]) for row in cells) for index in range(len(headers))]
    lines = ["  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip() for row in cells]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def render_tables(report: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"R26 FIELD QUALIFICATION — NORMALIZED REPORT ({report['generated_at']})")
    out.append(f"root: {report['root']}")
    out.append(f"completeness: {report['completeness']['claim']}")
    out.append(
        f"qualification: {report['qualification']['status']} "
        f"(passed={fmt(report['qualification']['passed'])}; disqualifying rows={report['qualification']['disqualifying_rows']})"
    )
    out.append(f"rows planned {report['completeness']['planned_rows']}, with receipt {report['completeness']['rows_with_receipt']}, missing {report['completeness']['missing_rows']}; status totals {report['completeness']['status_totals']}")
    isolation = report["provenance"]["gpu_isolation"]
    out.append(f"gpu isolation: mode={isolation.get('mode')} samples={isolation['samples']} with-foreign={isolation['samples_with_foreign_gpu_process']} foreign={isolation['foreign_process_names']} interruption={'yes' if isolation.get('interruption') else 'no'}")
    if isolation.get("vllm_named_foreign_sample_count"):
        stamps = ", ".join(sample["timestamp"] for sample in isolation["vllm_named_foreign_samples"][:6])
        out.append(f"  note: {isolation['vllm_named_foreign_sample_count']} sample(s) list VLLM:: worker pids as foreign ({stamps}); consistent with the owned container's own docker restart transition, still counted as contamination for any window they touch")
    out.append(f"excluded initial pass: {report['provenance']['excluded_initial_pass']['directory']} (ingested: no)")
    out.append("images: " + "; ".join(f"{arm}={digest_of(image)}" for arm, image in report["provenance"]["images"].items()))
    weights = report["weights"]
    out.append(f"weights: {weights['kind']} {weights['model_dir']}" + (f" (arm tag {weights['tag']}; never pooled with published-weight rows)" if weights["tag"] else "") + f"; arms present: {', '.join(weights['arms_present'])}")
    out.append("")
    out.append("PHASES")
    out.append(table(["phase", "stage", "state", "rc", "planned rows", *STATUSES], [
        [p["name"], p["stage"], p["state"], (p["window"] or {}).get("returncode"), p["planned_rows"], *(p["status_counts"].get(key, 0) for key in STATUSES)]
        for p in report["phases"]]))
    out.append("")
    out.append("QUALITY / CORRECTNESS RECEIPTS (derived categories from quality_reclassify.py; functional results stand under recorded background GPU work)")
    out.append("'superseded' originals keep their attempt on record; the published result is the follow-up native recheck row they point at.")
    quality_rows = [row for row in report["rows"] if row["kind"].startswith(("quality-probe", "quality-profile")) and row["status"] != "missing"]
    out.append(table(["arm", "stage", "cell", "kind", "status", "failure classes", "requested", "completed", "derived categories", "cache gate", "resolution"], [
        [row["arm"], row["stage"], row["cell"], row["kind"], row["status"], ",".join(row["failure_classes"]) or "-",
         (row["outcomes"] or {}).get("requested", (row["outcomes"] or {}).get("attempted")), (row["outcomes"] or {}).get("completed"),
         json.dumps((row["outcomes"] or {}).get("derived_categories"), separators=(",", ":")) if (row["outcomes"] or {}).get("derived_categories") else "-",
         (row["outcomes"] or {}).get("cache_effect_gate", "-"), resolution_of(row)]
        for row in quality_rows]) if quality_rows else "(no quality receipts yet)")
    if report["quality_derived_totals_by_arm"]:
        out.append("derived totals by arm: " + json.dumps(report["quality_derived_totals_by_arm"], separators=(",", ":")))
    out.append("")
    controls = [row for row in report["rows"] if row["kind"].startswith(("quality-profile", "semantic-control")) and row["status"] not in ("missing", "superseded")]
    if controls:
        out.append("LAVD / ESTONIA PROFILE RUNS (reasoning template is part of the identity; low-reasoning and default-template rows are never compared with each other)")
        out.append(table(["arm", "stage", "cell", "kind", "status", "reasoning", "runs", "correct", "wrong", "errors", "completion tokens p50"], [
            [row["arm"], row["stage"], row["cell"], row["kind"], row["status"], ((row["outcomes"] or {}).get("template_settings") or {}).get("reasoning_effort") or "template-default",
             (row["outcomes"] or {}).get("attempted"), (row["outcomes"] or {}).get("correct"), (row["outcomes"] or {}).get("wrong"), (row["outcomes"] or {}).get("errors"),
             (((row["outcomes"] or {}).get("completion_tokens") or {}).get("p50") if isinstance((row["outcomes"] or {}).get("completion_tokens"), dict) else None)]
            for row in controls]))
        out.append("")
    cache_rows = [row for row in report["rows"] if row["kind"] == "cache-cell"]
    if cache_rows:
        out.append("CACHE QUALIFICATION CELLS (old attempts remain visible; corrected v2/recheck rows supersede only their named source cells)")
        out.append(table(
            ["stage", "cell", "status", "source", "failure classes", "failed gates", "resolution"],
            [
                [
                    row["stage"], row["cell"], row["status"], row["source_state"],
                    ",".join(row["failure_classes"]) or "-",
                    len(((row["outcomes"] or {}).get("gates_failed") or {})),
                    resolution_of(row),
                ]
                for row in cache_rows
            ],
        ))
        out.append("")
    lifecycle_rows = [(row, key, sub) for row in report["rows"] if row["kind"] == "cache-cell" and isinstance((row["outcomes"] or {}).get("lifecycle_subchecks"), dict)
                      for key, sub in row["outcomes"]["lifecycle_subchecks"].items()]
    if lifecycle_rows:
        out.append("CACHE LIFECYCLE SUB-CHECKS (visible-final bytes are the answer oracle; full content+reasoning hashes are diagnostic only)")
        out.append(table(["stage", "cell", "kind", "gate", "cold miss", "warm hit", "restart hit", "hit tokens", "visible reference", "visible bytes equal", "full bytes equal (diagnostic)", "budget cutoff", "template mode"], [
            [row["stage"], key, sub.get("kind"), "pass" if sub.get("gate_passed") else "FAIL",
             sub["transfer_hit_metadata"].get("cold_external_miss"), sub["transfer_hit_metadata"].get("warm_external_hit"), sub["transfer_hit_metadata"].get("restart_l2_hit"),
             "/".join(fmt(value) for value in (sub["transfer_hit_metadata"].get("lmcache_hit_tokens") or [])) or "-",
             sub["reference_retention"].get("reference_retained_in_visible_final"),
             sub["output_identity"].get("visible_output_byte_equal"),
             sub["output_identity"].get("full_output_byte_equal_diagnostic"),
             "single-token transfer-only" if sub["budget_cutoff"].get("single_token_by_design") else (",".join(sub["budget_cutoff"].get("stages_cut_off") or []) or "none"),
             "; ".join(sub.get("template_mode") or []) or "-"]
            for row, key, sub in lifecycle_rows]))
        for row, key, sub in lifecycle_rows:
            if sub.get("findings"):
                out.append(f"  {row['stage']} {key}: " + "; ".join(sub["findings"]))
        out.append("")
    agent_rows = [row for row in report["rows"] if row["kind"] == "agent-cache"]
    if agent_rows:
        out.append("AGENT CACHE TRACE (runtime completion is separate from optional streaming telemetry and the non-gating hit-rate observation)")
        out.append(table(
            ["arm", "cache", "status", "completed", "expected", "optional gaps", "cached/reusable prompt-token ratio", "production95 claim"],
            [
                [
                    row["arm"], row["cell"].rsplit("-", 1)[-1], row["status"],
                    ((row["outcomes"] or {}).get("runtime_completion") or {}).get("requests_completed"),
                    ((row["outcomes"] or {}).get("runtime_completion") or {}).get("requests_expected"),
                    ",".join(((row["outcomes"] or {}).get("telemetry_observation") or {}).get("optional_gaps") or []) or "-",
                    ((row["outcomes"] or {}).get("telemetry_observation") or {}).get("aggregate_cached_over_reusable_prompt_token_ratio"),
                    ((row["outcomes"] or {}).get("telemetry_observation") or {}).get("production_95_percent_claim"),
                ]
                for row in agent_rows
            ],
        ))
        out.append("")
    scheduler_rows = [
        row for row in report["rows"]
        if row["kind"].startswith(("scheduler-recheck:", "scheduler-lane-recheck:"))
    ]
    if scheduler_rows:
        out.append("SHIPPED SCHEDULER RECHECK (serving completion, timing observation and QoS are separate)")
        out.append(table(
            ["stage", "arm", "cell", "status", "runtime complete", "observed/expected scenarios", "observation failures", "resolution"],
            [
                [
                    row["stage"], row["arm"], row["cell"], row["status"],
                    ((row["outcomes"] or {}).get("runtime_completion") or {}).get("passed"),
                    f"{((row['outcomes'] or {}).get('runtime_completion') or {}).get('observed_scenarios')}/{((row['outcomes'] or {}).get('runtime_completion') or {}).get('expected_scenarios')}",
                    len((((row["outcomes"] or {}).get("observation_and_qos") or {}).get("failed_gates") or [])),
                    resolution_of(row),
                ]
                for row in scheduler_rows
            ],
        ))
        out.append("")
    diagnostics = [row for row in report["rows"] if row["kind"].startswith("cache-final-diagnostic:")]
    if diagnostics:
        out.append("FINAL RUNTIME DIAGNOSTICS (missing means not run; safetensors-loader diagnostics never erase default-loader OOM)")
        out.append(table(
            ["cell", "status", "source", "failure classes", "resolution"],
            [[row["cell"], row["status"], row["source_state"], ",".join(row["failure_classes"]) or "-", resolution_of(row)] for row in diagnostics],
        ))
        out.append("")
    out.append("BOOTS AND SERVER KV CAPACITY (server-logged 'GPU KV cache size'; bench metadata never used)")
    boots = [row for row in report["rows"] if row["kind"] == "boot" and row["status"] != "missing"]
    out.append(table(["arm", "stage", "label", "status", "dcp", "spec", "kv", "cache", "extra env", "extra args", "server KV tokens", "resolution"], [
        [row["arm"], row["stage"], row["cell"], row["status"], (row["config"] or {}).get("dcp"), (row["config"] or {}).get("spec"), (row["config"] or {}).get("kv"), (row["config"] or {}).get("cache"),
         json.dumps((row["config"] or {}).get("extra_env") or {}, separators=(",", ":")), " ".join((row["config"] or {}).get("extra_args") or []) or "-", row["kv_tokens_server"], resolution_of(row)] for row in boots]) if boots else "(no boots yet)")
    out.append("")
    counts = report["speed_windows"]["counts"]
    out.append(f"SPEED WINDOWS — clean {counts['clean']}, foreign GPU overlap {counts['foreign-gpu-overlap']}, GPU health fault {counts['gpu-health-fault']}, coverage gap {counts['coverage-gap']}, no window {counts['no-window']}")
    out.append("Only 'clean' rows may ever be quoted or charted as speed. Everything else is listed for the record.")
    speed_rows = [row for row in report["rows"] if row["speed"] is not None]
    out.append(table(["verdict", "arm", "stage", "cell", "kind", "window start (UTC)", "window end (UTC)", "foreign", "headline", "resolution"], [
        [row["speed_qualification"], row["arm"], row["stage"], row["cell"], row["kind"], (row["window"] or {}).get("started_at_iso"), (row["window"] or {}).get("finished_at_iso"),
         ",".join(sorted({name.split("[")[0] for name in (row["isolation"] or {}).get("foreign", [])})) or "-", speed_headline(row), resolution_of(row)]
        for row in speed_rows]) if speed_rows else "(no speed samples yet)")
    out.append("")
    qad = [row for row in report["rows"] if row["cell"].startswith("qad-matched-") and row["kind"].startswith("quality-") and row["status"] != "missing"]
    if qad:
        out.append("QAD MATCHED PAIRS (same R26 image, rig and prompts; only the mounted checkpoint differs; small-sample observations, not fidelity proofs; acceptance windows are in the SPEED table)")
        out.append(table(["config", "weights", "arm", "cell", "kind", "status", "failure classes", "correct", "wrong", "derived categories", "headline"], [
            [(row["config"] or {}).get("matched_config"), (row["config"] or {}).get("weights"), row["arm"], row["cell"], row["kind"], row["status"], ",".join(row["failure_classes"]) or "-",
             (row["outcomes"] or {}).get("correct"), (row["outcomes"] or {}).get("wrong"),
             json.dumps((row["outcomes"] or {}).get("derived_categories"), separators=(",", ":")) if (row["outcomes"] or {}).get("derived_categories") else "-",
             speed_headline(row) if row["speed"] else "-"]
            for row in qad]))
    else:
        runbook = report["provenance"]["qad_runbook"]
        out.append(f"QAD RUNBOOK — {runbook['status']}; {runbook['planned_phase_count']} phases from run_qad_battery.py; no completed or passed claim")
    out.append("")
    unsupported = [row for row in report["rows"] if row["status"] in ("unsupported", "skipped")]
    out.append("UNSUPPORTED CAPABILITY / DELIBERATE SKIPS")
    out.append(table(["arm", "cell", "status", "reason"], [[row["arm"], row["cell"], row["status"], json.dumps((row["outcomes"] or {}).get("unsupported") or (row["outcomes"] or {}).get("reason"))[:120]] for row in unsupported]) if unsupported else "(none recorded)")
    out.append("")
    out.append(f"GATES total {report['gates']['total']} passed {report['gates']['passed']} failed {report['gates']['failed']}; failed by class {report['gates']['failed_by_class']}")
    out.append(table(["line", "time (UTC)", "phase", "class", "gate"], [[g["line"], g["timestamp"], g["phase"], g["failure_class"], g["name"]] for g in report["gates"]["failed_gates"]]) if report["gates"]["failed_gates"] else "(no failed gates)")
    out.append("")
    out.append("MISSING PLANNED CELLS (reported as missing, never as pass)")
    for phase in report["phases"]:
        if phase["missing_cells"]:
            shown = phase["missing_cells"][:12]
            more = len(phase["missing_cells"]) - len(shown)
            out.append(f"- {phase['name']} [{phase['state']}]: {len(phase['missing_cells'])} missing: {', '.join(shown)}{f', ... +{more} more' if more else ''}")
    out.append("")
    out.append("COVERAGE GAPS / NOTES")
    out.extend(f"- {gap}" for gap in report["coverage_gaps"]) if report["coverage_gaps"] else out.append("- none")
    if report["provenance"]["read_errors"]:
        out.append("READ ERRORS")
        out.extend(f"- {error['path']}: {error['error']}" for error in report["provenance"]["read_errors"])
    return "\n".join(out) + "\n"


def resolution_of(row: dict[str, Any]) -> str:
    if row["superseded_by"]:
        return "→ " + label_of(row["superseded_by"]).removesuffix(".launch")
    if row.get("superseded_checks"):
        return f"{len(row['superseded_checks'])} check(s) superseded"
    if row["recheck_of"]:
        return "recheck of " + label_of(row["recheck_of"]).removesuffix(".launch")
    return "-"


def nested_keys(value: Any, depth: int = 4) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict) and depth:
        for key, inner in value.items():
            if isinstance(key, str):
                keys.add(key)
            keys |= nested_keys(inner, depth - 1)
    return keys


def speed_headline(row: dict[str, Any]) -> str:
    speed = row["speed"] or {}
    if row["kind"].startswith("acceptance"):
        fraction = speed.get("acceptance_fraction", speed.get("token_weighted_pooled_acceptance_fraction"))
        parts = [f"accept {fmt(fraction, 3)}", f"emitted/step {fmt(speed.get('emitted_tokens_per_verifier_step'))}"]
        if speed.get("output_tokens_per_second") is not None:
            parts.append(f"tok/s {fmt(speed['output_tokens_per_second'], 1)}")
        return " ".join(parts)
    if row["kind"] == "bench":
        cells = speed.get("cells") or []
        return "; ".join(f"c{cell['conc']}/ctx{cell['ctx']} {fmt(cell.get('aggregate_tps'), 1)} tok/s" for cell in cells[:4]) or "-"
    if row["kind"].startswith("scheduler"):
        return f"{len(speed.get('cells') or [])} scenario cells"
    return " ".join(f"{key}={fmt(value)}" for key, value in speed.items() if value is not None and not isinstance(value, (dict, list)))[:90] or "-"


# ---------------------------------------------------------------- chart
PALETTE = {"background": "#0D1117", "card": "#161B22", "text": "#E6EDF3", "muted": "#AAB7C7", "grid": "#21262D",
           "green": "#3FB950", "blue": "#58A6FF", "amber": "#E3B341", "red": "#F85149", "purple": "#BC8CFF", "grey": "#8B949E", "pink": "#F778BA", "slate": "#484F58"}
CATEGORY_COLORS = {"correct_final": "green", "completed_unverified": "blue", "parser_pass": "blue", "budget_limited_incomplete": "amber", "incomplete_no_visible_final": "amber",
                   "model_quality_flag": "amber", "wrong_final": "red", "repetition": "purple", "runtime_error": "grey", "parser_failure": "red"}
STATUS_COLORS = {"pass": "green", "flagged": "amber", "fail": "red", "unsupported": "purple", "skipped": "grey", "not-run": "pink", "superseded": "slate", "running": "blue", "missing": "grid"}
ARM_SHORT = {"official-r26": "R26", "r25-control": "R25 ctl", "drock-overlay": "D-Rock", "hardware-599": "#599", "unattributed": "?"}


def render_chart(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    quality = [row for row in report["rows"] if row["kind"].startswith("quality-probe") and row["status"] != "superseded" and isinstance(row["outcomes"], dict) and row["outcomes"].get("derived_categories")]
    quality.sort(key=lambda row: (ARM_ORDER.index(row["arm"]) if row["arm"] in ARM_ORDER else 9, row["cell"]))
    clean_speed = [row for row in report["rows"] if row["speed_qualification"] == "clean"]
    counts = report["speed_windows"]["counts"]
    phases = report["phases"]
    images = report["provenance"]["images"]
    text, muted = PALETTE["text"], PALETTE["muted"]
    fig = plt.figure(figsize=(10, 7.6), dpi=140, facecolor=PALETTE["background"])
    weights = report["weights"]
    title_arm = f" — candidate checkpoint {weights['tag']}" if weights["tag"] else ""
    record_state = "COMPLETE RECORD" if report["completeness"]["phase_execution_complete"] and not report["completeness"]["missing_rows"] and not report["completeness"]["pending_rows"] else "PROVISIONAL"
    fig.text(0.03, 0.97, f"GLM-5.3-Flash R26 field qualification{title_arm} — {record_state}", color=text, fontsize=13, weight="bold", va="top")
    fig.text(0.03, 0.935, f"CPU render of receipts at {report['generated_at']} · {report['completeness']['claim'].split(':')[0]} · "
             f"{report['completeness']['rows_with_receipt']}/{report['completeness']['planned_rows']} planned rows have receipts",
             color=muted, fontsize=7.5, va="top")
    caveat = (f"Own arm '{weights['tag']}', never pooled with published-weight rows; 'official-r26' rows here are the matched published-weight controls from the same run."
              if title_arm else "Correctness receipts are functional and stand under recorded background GPU work; speed figures do not.")
    fig.text(0.03, 0.912, caveat, color=PALETTE["amber"] if title_arm else muted, fontsize=7.5, va="top")

    fig.text(0.03, 0.87, "\n".join([
        "Arms (never pooled)", f" official R26   {digest_of(images['official-r26'])}", f" matched R25    {digest_of(images['r25-control'])}", f" D-Rock overlay {digest_of(images['drock-overlay'])}",
        f" weights: {Path(weights['model_dir']).name}", "",
        "Rules", " speed qualified only in fully", " sampled windows with zero", " foreign GPU processes", "", " synthetic acceptance probe =", " stress ceiling, not typical", "",
        " KV pool = server-logged", " 'GPU KV cache size'", "", " initial contaminated pass", " excluded, never ingested",
    ]), color=muted, fontsize=6.6, va="top", family="DejaVu Sans", linespacing=1.35)

    # panel 1: correctness outcomes per quality receipt
    ax1 = fig.add_axes([0.40, 0.50, 0.36, 0.37], facecolor=PALETTE["card"])
    ax1.set_title("Correctness receipts so far (derived categories)", color=text, fontsize=8, loc="left", pad=5)
    if quality:
        for index, row in enumerate(quality):
            left = 0
            for category, count in sorted(row["outcomes"]["derived_categories"].items()):
                ax1.barh(index, count, left=left, color=PALETTE[CATEGORY_COLORS.get(category, "grey")], edgecolor=PALETTE["background"], height=0.7)
                left += count
        ax1.set_yticks(range(len(quality)))
        ax1.set_yticklabels([chart_label(row) for row in quality], color=text, fontsize=6.3)
        ax1.invert_yaxis()
        ax1.set_xlabel("requests", color=muted, fontsize=7)
    else:
        ax1.text(0.5, 0.5, "no quality receipts yet", color=muted, ha="center", va="center", transform=ax1.transAxes)
    style_axis(ax1)
    legend = [Patch(color=PALETTE[color], label=category) for category, color in (("correct_final", "green"), ("completed_unverified / parser_pass", "blue"), ("budget_limited_incomplete", "amber"), ("wrong_final", "red"), ("repetition", "purple"), ("runtime_error", "grey"))]
    ax1.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.3, frameon=False, labelcolor=text, handlelength=1.0, columnspacing=1.0)

    # panel 2: speed qualification counts (numbers only; no speed values unless clean)
    ax2 = fig.add_axes([0.80, 0.50, 0.18, 0.37], facecolor=PALETTE["card"])
    ax2.set_title("Speed windows", color=text, fontsize=8, loc="left", pad=5)
    labels = ["clean", "foreign\nGPU", "health\nfault", "sample\ngap", "no\nwindow"]
    values = [counts["clean"], counts["foreign-gpu-overlap"], counts["gpu-health-fault"], counts["coverage-gap"], counts["no-window"]]
    ax2.bar(range(5), values, color=[PALETTE["green"], PALETTE["red"], PALETTE["purple"], PALETTE["amber"], PALETTE["grey"]], width=0.7)
    for index, value in enumerate(values):
        ax2.text(index, value, str(value), color=text, ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, color=text, fontsize=6)
    ax2.set_ylim(0, max(values + [1]) * 1.25)
    style_axis(ax2)
    if clean_speed:
        message = f"{len(clean_speed)} clean window(s); values in tables"
    else:
        foreign = ", ".join(sorted(report["provenance"]["gpu_isolation"]["foreign_process_names"])) or "none"
        message = f"0 clean windows → no speed charted\nforeign GPU processes:\n{foreign[:44]}{'…' if len(foreign) > 44 else ''}"
    ax2.text(0.5, -0.16, message, color=PALETTE["amber"], fontsize=6.3, ha="center", va="top", transform=ax2.transAxes)

    # panel 3: phase coverage
    ax3 = fig.add_axes([0.40, 0.10, 0.58, 0.24], facecolor=PALETTE["card"])
    ax3.set_title("Planned cells per phase (missing counted as missing, never as pass)", color=text, fontsize=8, loc="left", pad=5)
    for index, phase in enumerate(phases):
        left = 0
        for status in STATUSES:
            value = phase["status_counts"].get(status, 0)
            if value:
                ax3.barh(index, value, left=left, color=PALETTE[STATUS_COLORS[status]], edgecolor=PALETTE["background"], height=0.7, hatch="//" if status == "missing" else None)
                left += value
        ax3.text(left + 0.5, index, phase["state"], color=muted, fontsize=6, va="center")
    ax3.set_yticks(range(len(phases)))
    ax3.set_yticklabels([phase["name"] for phase in phases], color=text, fontsize=6.3)
    ax3.invert_yaxis()
    ax3.set_xlim(0, max([phase["planned_rows"] for phase in phases] + [1]) * 1.18)
    ax3.set_xlabel("planned rows", color=muted, fontsize=7)
    style_axis(ax3)
    ax3.legend(handles=[Patch(facecolor=PALETTE[color], label=status, hatch="//" if status == "missing" else None) for status, color in STATUS_COLORS.items()],
               loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=9, fontsize=6.0, frameon=False, labelcolor=text, handlelength=1.0, columnspacing=0.8)

    fig.text(0.03, 0.30, f"generated by\nscripts/r26/qualification_report.py\nroot {report['root']}".replace("/drock-lmcache/", "/drock-lmcache/\n  "),
             color=PALETTE["grey"], fontsize=5.8, va="top", linespacing=1.4)
    fig.savefig(path, facecolor=PALETTE["background"])
    plt.close(fig)


def chart_label(row: dict[str, Any]) -> str:
    """Short y label: arm alias plus the cell with its arm-identifying prefixes stripped, capped for the panel width."""
    cell = row["cell"]
    for prefix in ("quality-", "qad-matched-"):
        cell = cell.removeprefix(prefix)
    label = f"{ARM_SHORT.get(row['arm'], row['arm'])} · {cell}"
    return label if len(label) <= 44 else label[:43] + "…"


def style_axis(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=7)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- post draft
def render_post_draft(report: dict[str, Any]) -> str:
    counts = report["speed_windows"]["counts"]
    quality = [row for row in report["rows"] if row["kind"].startswith("quality-probe") and row["status"] != "superseded" and isinstance(row["outcomes"], dict) and row["outcomes"].get("derived_categories")]
    lines = ["DRAFT — NOT FOR PUBLICATION. Inputs for Main's short casual post; regenerate after the full battery.", f"(generated {report['generated_at']}; completeness: {report['completeness']['claim']})", ""]
    lines.append("Facts available right now (only completed receipts):")
    lines.append(
        f"- Overall qualification status: {report['qualification']['status']}; "
        f"passed={report['qualification']['passed']}. Completion and pass are separate."
    )
    for arm in ARM_ORDER:
        rows = [row for row in quality if row["arm"] == arm]
        if not rows:
            continue
        totals = Counter()
        for row in rows:
            totals.update(row["outcomes"]["derived_categories"])
        lines.append(f"- {arm}: {len(rows)} correctness receipt(s) → {dict(sorted(totals.items()))}")
    if counts["clean"]:
        lines.append(f"- {counts['clean']} speed window(s) are clean and quotable; {counts['foreign-gpu-overlap']} overlapped foreign GPU work and {counts['gpu-health-fault']} hit a GPU health fault; both are withheld.")
    else:
        lines.append(f"- Speed: nothing quotable yet. {counts['foreign-gpu-overlap']} window(s) overlapped foreign GPU processes, {counts['gpu-health-fault']} hit a GPU health fault and {counts['coverage-gap']} lacked sampling coverage; all withheld.")
    profiles = [row for row in report["rows"] if row["kind"].startswith(("quality-profile", "semantic-control")) and row["status"] not in ("missing", "superseded", "not-run")]
    for row in profiles:
        outcomes = row["outcomes"] or {}
        reasoning = (outcomes.get("template_settings") or {}).get("reasoning_effort") or "template-default"
        lines.append(f"- {row['arm']} {row['cell']} [{reasoning}]: {outcomes.get('correct')}/{outcomes.get('attempted')} correct, {outcomes.get('wrong')} wrong, {outcomes.get('errors')} errors — compare only with rows of the same reasoning setting")
    superseded = [row for row in report["rows"] if row["status"] == "superseded"]
    if superseded:
        lines.append(f"- {len(superseded)} original receipt row(s) are explicitly superseded by source-backed rechecks; every original attempt and observation remains in the manifest.")
    cache_flags = Counter(cls for row in report["rows"] if row["kind"] == "cache-cell" and row["status"] != "superseded" for cls in row["failure_classes"])
    if cache_flags:
        lines.append(f"- Cache cells: failure classes {dict(sorted(cache_flags.items()))}; visible-final bytes are the answer oracle, full content+reasoning hashes are diagnostic, and single-token empty finals are budget-limited rather than wrong answers.")
    agent_rows = [row for row in report["rows"] if row["kind"] == "agent-cache" and row["status"] != "missing"]
    for row in agent_rows:
        runtime_completion = ((row["outcomes"] or {}).get("runtime_completion") or {})
        telemetry = ((row["outcomes"] or {}).get("telemetry_observation") or {})
        lines.append(
            f"- {row['cell']}: runtime {runtime_completion.get('requests_completed')}/{runtime_completion.get('requests_expected')} turns; "
            f"status={row['status']}, optional telemetry gaps={telemetry.get('optional_gaps') or []}, "
            f"aggregate short-trace cached/reusable prompt-token ratio={fmt(telemetry.get('aggregate_cached_over_reusable_prompt_token_ratio'), 3)} (not a query ratio or production95 claim)."
        )
    timing_rows = [
        row for row in report["rows"]
        if "timing-telemetry-gap" in row["failure_classes"]
    ]
    if timing_rows:
        lines.append(f"- Scheduler: {len(timing_rows)} row(s) completed serving but lack the off-policy prefill/decode timing split; they are flagged observations, not failed serving.")
    qad = report["provenance"]["qad_runbook"]
    lines.append(f"- QAD: {qad['status']}; {qad['planned_phase_count']} source-read runbook phases, with no completion/pass or performance claim from the plan.")
    missing = report["completeness"]["missing_rows"]
    lines.append(f"- {missing} planned cell(s) still have no receipt; phases: " + ", ".join(f"{phase['name']}={phase['state']}" for phase in report["phases"]))
    lines.append("")
    lines.append("Casual copy skeleton (fill only from the facts above; keep arms separate, no pooled score):")
    lines.append(f"  R26 field qualification update — completed receipts first; overall status is {report['qualification']['status']}, never inferred from a phase plan.")
    lines.append("  Stock R26, the matched R25 control, and D-Rock's overlay are reported side by side, never pooled.")
    lines.append("  The synthetic acceptance probe is a ceiling number, not what real prompts see; realistic acceptance is its own receipt.")
    lines.append("  LAVD numbers only compare like with like: explicit low-reasoning runs against low, default-template runs against the old default-template result.")
    lines.append("  KV pool figures come from the server's own 'GPU KV cache size' line, not the bench's block math.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=rt.ROOT, help="battery root (default BATTERY_ROOT / runtime.ROOT)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for r26-report-* files")
    parser.add_argument("--tag", help="output name tag (default: the root's candidate checkpoint tag, empty for published weights)")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    root = Root(args.root.resolve())
    if not root.path.is_dir():
        raise SystemExit(f"battery root does not exist: {root.path}")
    report = Report(root).build()
    tag = args.tag if args.tag is not None else (report["weights"]["tag"] or "")
    prefix = args.out / (f"r26-report-{tag}-" if tag else "r26-report-")
    write_atomic(Path(f"{prefix}manifest.json"), json.dumps(report, indent=1, default=str) + "\n")
    write_atomic(Path(f"{prefix}tables.txt"), render_tables(report))
    write_atomic(Path(f"{prefix}post-draft.txt"), render_post_draft(report))
    outputs = [f"{prefix}manifest.json", f"{prefix}tables.txt", f"{prefix}post-draft.txt"]
    if not args.no_chart:
        render_chart(report, Path(f"{prefix}chart.png"))
        outputs.append(f"{prefix}chart.png")
    print(json.dumps({
        "outputs": outputs, "weights": {key: report["weights"][key] for key in ("model_dir", "tag", "kind")}, "elapsed_seconds": round(time.time() - started, 2),
        "completeness": report["completeness"], "qualification": report["qualification"],
        "speed_windows": report["speed_windows"]["counts"], "gates": {key: report["gates"][key] for key in ("total", "passed", "failed")},
        "coverage_gaps": len(report["coverage_gaps"]), "read_errors": len(report["provenance"]["read_errors"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
