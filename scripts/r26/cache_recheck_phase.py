#!/usr/bin/env python3
"""Run only the corrected cache qualification cells in a fresh receipt root."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

try:
    from . import cache_phase as cache
    from . import runtime as rt
except ImportError:
    import cache_phase as cache
    import runtime as rt

RECHECK_SCHEMA = "r26-cache-recheck/v1"
ORIGINAL_RECEIPT_ROOT = Path(
    "/home/josh/omp-workspace/drock-lmcache/r26-battery"
)
SOURCE_FILES = {
    "cache_probe": Path(__file__).with_name("cache_probe.py"),
    "cache_phase": Path(__file__).with_name("cache_phase.py"),
    "cache_recheck_phase": Path(__file__),
    "runtime": Path(__file__).with_name("runtime.py"),
    "cache_config_probe": Path(__file__).with_name("cache_config_probe.py"),
}
PRIOR_RECEIPTS = {
    "original_cache_phase_summary": ORIGINAL_RECEIPT_ROOT
    / "cache-phase-summary.json",
    "original_eviction_receipt": ORIGINAL_RECEIPT_ROOT
    / "cache-official-r26-40-document-eviction.json",
    "original_focused_stock_prepare_failure": ORIGINAL_RECEIPT_ROOT
    / "cache-focused-stock-r26-retention-prepare.json",
    "original_focused_overlay_prepare_failure": ORIGINAL_RECEIPT_ROOT
    / "cache-focused-drock-overlay-retention-prepare.json",
    "original_native_r26_gmu093_oom_log": ORIGINAL_RECEIPT_ROOT
    / "cache-official-r26-native-offload-warm.docker.log",
    "original_native_r26_gmu093_launch": ORIGINAL_RECEIPT_ROOT
    / "cache-official-r26-native-offload-warm.launch.json",
}
EXPECTED_RECHECK_CELLS = frozenset(
    {
        "lifecycle-fp8-80k",
        "lifecycle-fp8-exact-1m",
        "lifecycle-packed-nvfp4-80k",
        "lifecycle-packed-nvfp4-exact-1m",
        "cold-needles-packed-nvfp4",
        "40-document-eviction",
        "focused-stock-vs-overlay",
        "native-canaries",
    }
)


def sha256_file(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as error:
        return {
            "path": str(path),
            "available": False,
            "sha256": None,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "path": str(path),
        "available": True,
        "bytes": size,
        "sha256": digest,
    }

def contract() -> dict[str, Any]:
    return {
        "schema": RECHECK_SCHEMA,
        "entrypoint": "cache_recheck_phase.py",
        "runtime_configuration": "environment variables consumed by runtime.py",
        "required_fresh_root": "BATTERY_ROOT must not equal the original R26 receipt root",
        "schemas": {
            "probe_receipts": "r26-cache-probe/v2",
            "phase_cells": cache.PHASE_SCHEMA,
            "recheck_summary": RECHECK_SCHEMA,
        },
        "receipts": {
            "rolling_summary": "cache-recheck-summary.json",
            "terminal_completion": "cache-recheck-complete.json",
            "source_hashes_in_summary": True,
            "raw_per-request_receipts_retained": True,
        },
        "cells": {
            "lifecycles": {
                "kv_formats": ["fp8_ds_mla", "nvfp4_ds_mla"],
                "requests": [
                    {
                        "tokens": cache.TARGET_80K,
                        "kind": "period",
                        "scope": "one-token latency/cache-transfer only",
                    },
                    {
                        "tokens": cache.TARGET_1M,
                        "kind": "reference",
                        "scope": "complete visible-answer recall",
                    },
                ],
            },
            "cold_needles": {
                "kv": "nvfp4_ds_mla",
                "tokens": cache.NEEDLE_TOKENS,
                "depth_percent": [10, 50, 90],
            },
            "eviction": {
                "kv": "nvfp4_ds_mla",
                "documents": 40,
                "random_words_per_document": 12_000,
                "l1_gb": cache.EVICTION_L1_GB,
                "l2_gb": cache.EVICTION_L2_GB,
            },
            "focused": {
                "arms": ["stock-r26", "drock-overlay"],
                "issues": ["#574", "#643", "#645"],
                "tokens": cache.FOCUSED_TOKENS,
            },
            "native": {
                "matched_gmu_093": {
                    "releases": ["r25", "r26"],
                    "kv_formats": ["fp8_ds_mla", "nvfp4_ds_mla"],
                },
                "r26_gmu_090_diagnostic": {
                    "releases": ["r26"],
                    "kv_formats": ["fp8_ds_mla", "nvfp4_ds_mla"],
                },
                "diagnostic_cannot_override_gmu_093": True,
            },
        },
        "method": {
            "unique_preamble_rows": cache.UNIQUE_PREAMBLE_ROWS,
            "minimum_exact_tokenizer_coverage": 4096,
            "focused_target_tokens": cache.FOCUSED_TOKENS,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "semantic_max_tokens": cache.SEMANTIC_MAX_TOKENS,
            "needle_max_tokens": cache.NEEDLE_MAX_TOKENS,
            "automatic_budget_retry": False,
            "l2_normal_capacity_gb": cache.NORMAL_L2_CAP_GB,
            "l2_cleanup": "runtime.cleanup_l2_child on cache-phase-owned children only",
        },
    }


class CacheRecheckPhase:
    def __init__(self) -> None:
        if rt.ROOT.resolve() == ORIGINAL_RECEIPT_ROOT.resolve():
            raise RuntimeError(
                "cache recheck requires a fresh BATTERY_ROOT; refusing to overwrite "
                f"original receipts at {ORIGINAL_RECEIPT_ROOT}"
            )
        self.phase = cache.CachePhase()
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.cells: dict[str, Any] = {}
        runtime_sources = {
            **SOURCE_FILES,
            "model_tokenizer_json": rt.MODEL / "tokenizer.json",
            "model_tokenizer_config": rt.MODEL / "tokenizer_config.json",
            "model_chat_template": rt.MODEL / "chat_template.jinja",
            "overlay_declaration": Path(
                "/home/josh/omp-workspace/drock-lmcache/"
                "release-review-20260905T154745Z/drock-temp.txt"
            ),
        }
        self.source_hashes = {
            name: sha256_file(path) for name, path in runtime_sources.items()
        }
        self.prior_receipt_hashes = {
            name: sha256_file(path) for name, path in PRIOR_RECEIPTS.items()
        }
        self.prior_receipt_hashes_after: dict[str, dict[str, Any]] | None = None
        self.methodology = {
            "schema": RECHECK_SCHEMA,
            "separate_recheck_not_old_phase_retry": True,
            "original_receipts": {
                "root": str(ORIGINAL_RECEIPT_ROOT),
                "preserved": True,
                "recheck_write_root": str(rt.ROOT),
            },
            "changes": [
                {
                    "old": "256 hash rows repeated the identity and consumed about 18.5K tokens",
                    "new": (
                        "128 identity-derived hash rows omit repeated identities; the exact "
                        "serving tokenizer must still prove >=4096 preamble tokens and exact "
                        "whole-prompt calibration"
                    ),
                },
                {
                    "old": "top-level reasoning_effort and 64/128-token semantic generations",
                    "new": (
                        "chat_template_kwargs.reasoning_effort=low with fixed 512-token "
                        "reference/eviction and 1024-token needle budgets"
                    ),
                },
                {
                    "old": "content+reasoning hash and reasoning-channel code matches acted as answer evidence",
                    "new": (
                        "exact visible-final hashes and naturally completed visible answers are "
                        "the only answer identity/oracle; full-channel hashes remain diagnostic"
                    ),
                },
                {
                    "old": "ordinal counting inside 12K random words",
                    "new": "direct retrieval of a unique labelled marker value at a random insertion offset",
                },
                {
                    "old": "R26 native GMU 0.93 failure followed by an unlabeled lower-GMU retry",
                    "new": (
                        "matched R25/R26 GMU 0.93 canaries plus separately labelled R26 "
                        "GMU 0.90 diagnostics for FP8 and NVFP4"
                    ),
                },
            ],
            "retry_policy": (
                "No request is automatically retried with a larger generation budget. This "
                "fresh-root recheck is an explicit changed methodology and retains old receipts."
            ),
            "config_probe_history": {
                "original_corrupted_metric_receipts": "harness failures, not product failures",
                "corrected_method": (
                    "actual loopback LMCache OpenAPI/status/metrics on port 18085 "
                    "and nested store_controller adapter count"
                ),
                "historical_observation": (
                    "preserved snapshots after 21:00 local reported every config check PASS"
                ),
                "probe_source_hash_recorded": True,
            },
            "classification": {
                "harness_failure": (
                    "client exception, tokenizer/calibration construction failure, "
                    "missing receipt, or coordinator failure"
                ),
                "api_protocol_failure": "HTTP failure or malformed/missing API response",
                "model_wrong_answer": "naturally completed visible final does not match the oracle",
                "incomplete_answer": "nonempty visible final without a natural stop",
                "budget_limited": "generation reaches its length budget; an empty visible final is incomplete",
                "protocol_capability": "effective config, status, metrics, event stream/replay, and actual pinned-runtime log names",
                "memory_transfer": "cache hit-token metadata and store/load/eviction metrics or cache events",
                "not_memory_transfer": "visible/full generated-text equality alone",
                "not_corruption_proof": "a failed marker or needle retrieval by itself",
            },
        }
        self.phase.gate(
            "cache:recheck:source-hashes-complete",
            all(row.get("available") is True for row in self.source_hashes.values()),
            self.source_hashes,
        )
        self.phase.gate(
            "cache:recheck:original-receipt-lineage-hashed",
            all(
                row.get("available") is True
                for row in self.prior_receipt_hashes.values()
            ),
            self.prior_receipt_hashes,
        )

    def save_summary(self) -> None:
        failed = [row for row in self.phase.checks if not row["passed"]]
        expected = EXPECTED_RECHECK_CELLS
        summary = {
            "schema": RECHECK_SCHEMA,
            "run_id": self.phase.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "write_root": str(rt.ROOT),
            "original_receipts_preserved": rt.ROOT.resolve()
            != ORIGINAL_RECEIPT_ROOT.resolve(),
            "contract": contract(),
            "methodology": self.methodology,
            "source_hashes": self.source_hashes,
            "prior_receipt_hashes": self.prior_receipt_hashes,
            "prior_receipt_hashes_after": self.prior_receipt_hashes_after,
            "artifacts": {
                "images": {
                    "official_r26": rt.IMAGE,
                    "matched_r25": rt.R25_IMAGE,
                    "drock_overlay": rt.OVERLAY_IMAGE,
                    "drock_overlay_declared_revision": "7db6a2d2f5680513ae1a396ff61169c4cacf8a95",
                },
                "model_path": str(rt.MODEL),
                "model_cache_tag": rt.MODEL_CACHE_TAG,
            },
            "cells": self.cells,
            "coverage": {
                "expected": sorted(expected),
                "attempted": sorted(expected.intersection(self.cells)),
                "missing": sorted(expected.difference(self.cells)),
                "all_requested_cells_attempted": expected.issubset(self.cells),
                "not_rechecked": [
                    "FP8 needle depths (the requested three-depth cold needle cell is the packed-NVFP4 original)",
                    "prefill-overhead and alternating-release latency cells",
                    "full byte/bit identity of real externally stored GLM KV tensors (no shipped end-to-end oracle)",
                    "SATA-backed cache latency (the available cache path is NVMe and is not throttled)",
                ],
            },
            "checks": self.phase.checks,
            "failed_gates": failed,
            "qualification_passed": self.finished_at is not None
            and expected.issubset(self.cells)
            and not failed,
        }
        rt.save_json("cache-recheck-summary.json", summary)

    def invoke(
        self, name: str, operation: Callable[[], Any]
    ) -> Any:
        try:
            result = operation()
            if not isinstance(result, dict):
                result = {
                    "schema": RECHECK_SCHEMA,
                    "attempted": True,
                    "failure_class": "missing_result_receipt",
                    "error": "cell returned no structured result",
                }
                self.phase.gate(
                    f"cache:recheck:{name}:missing-result", False, result
                )
        except Exception as error:
            result = {
                "schema": RECHECK_SCHEMA,
                "attempted": True,
                "failure_class": "harness_exception",
                "error": f"{type(error).__name__}: {error}",
            }
            self.phase.gate(f"cache:recheck:{name}:exception", False, result)
            try:
                rt.capture(f"cache-recheck-{name}-exception")
            except Exception as capture_error:
                result["capture_error"] = (
                    f"{type(capture_error).__name__}: {capture_error}"
                )
        finally:
            self.phase.stop(f"cache-recheck-{name}-final")
        self.cells[name] = result
        self.save_summary()
        return result

    def cleanup_l2(self, path: Path, label: str) -> None:
        try:
            self.phase.cleanup_owned_l2(path, label)
        except Exception as error:
            self.phase.gate(
                f"cache:recheck:{label}:cleanup-exception",
                False,
                {
                    "path": str(path),
                    "failure_class": "cleanup_harness_exception",
                    "error": f"{type(error).__name__}: {error}",
                },
            )

    def run_lifecycles(self) -> None:
        for kv in ("fp8_ds_mla", "nvfp4_ds_mla"):
            shape = "fp8" if kv == "fp8_ds_mla" else "packed-nvfp4"
            host_dir = self.phase.owned_l2(f"recheck-{shape}-lifecycles")
            namespace = f"r26-cache-{self.phase.run_id}-recheck-{shape}"
            try:
                for target, kind, size in (
                    (cache.TARGET_80K, "period", "80k"),
                    (cache.TARGET_1M, "reference", "exact-1m"),
                ):
                    self.invoke(
                        f"lifecycle-{shape}-{size}",
                        lambda kv=kv, target=target, kind=kind, host_dir=host_dir, namespace=namespace: self.phase.lifecycle_cell(
                            kv=kv,
                            target=target,
                            kind=kind,
                            host_dir=host_dir,
                            namespace=namespace,
                        ),
                    )
            finally:
                self.cleanup_l2(
                    host_dir, f"cleanup-recheck-{shape}-lifecycles"
                )
                self.save_summary()

    def run_needles(self) -> None:
        host_dir = self.phase.owned_l2("recheck-packed-nvfp4-needles")
        namespace = f"r26-cache-{self.phase.run_id}-recheck-packed-nvfp4-needles"
        try:
            self.invoke(
                "cold-needles-packed-nvfp4",
                lambda: self.phase.needle_cell(
                    host_dir, namespace, kv="nvfp4_ds_mla"
                ),
            )
        finally:
            self.cleanup_l2(
                host_dir, "cleanup-recheck-packed-nvfp4-needles"
            )
            self.save_summary()

    def run_eviction(self) -> dict[str, Any] | None:
        self.phase.eviction_cell()
        result = self.phase.cells.get("eviction")
        return result if isinstance(result, dict) else None

    def run_focused(self) -> dict[str, Any] | None:
        self.phase.focused_overlay_comparison()
        result = self.phase.cells.get("focused-stock-vs-overlay")
        return result if isinstance(result, dict) else None

    def verify_prior_receipts(self) -> None:
        self.prior_receipt_hashes_after = {
            name: sha256_file(path) for name, path in PRIOR_RECEIPTS.items()
        }
        lineage_unchanged = all(
            self.prior_receipt_hashes[name].get("sha256") is not None
            and self.prior_receipt_hashes[name].get("sha256")
            == self.prior_receipt_hashes_after[name].get("sha256")
            for name in PRIOR_RECEIPTS
        )
        self.phase.gate(
            "cache:recheck:original-receipts-unchanged",
            lineage_unchanged,
            {
                "before": self.prior_receipt_hashes,
                "after": self.prior_receipt_hashes_after,
            },
        )

    def cleanup_remaining_l2(self) -> None:
        for path in sorted(self.phase._owned_dirs):
            if path.exists():
                self.cleanup_l2(
                    path, f"cleanup-recheck-final-{path.name}"
                )

    def finish(self) -> None:
        expected = EXPECTED_RECHECK_CELLS
        attempted = expected.issubset(self.cells)
        self.phase.gate(
            "cache:recheck:all-requested-cells-attempted",
            attempted,
            {
                "expected": sorted(expected),
                "observed": sorted(expected.intersection(self.cells)),
                "missing": sorted(expected.difference(self.cells)),
            },
        )
        lifecycle_targets = {
            "lifecycle-fp8-80k": cache.TARGET_80K,
            "lifecycle-fp8-exact-1m": cache.TARGET_1M,
            "lifecycle-packed-nvfp4-80k": cache.TARGET_80K,
            "lifecycle-packed-nvfp4-exact-1m": cache.TARGET_1M,
        }
        lifecycle_preambles = {
            name: {
                "first_block_token_ids_sha256": self.cells.get(name, {})
                .get("prompt", {})
                .get("first_block_token_ids_sha256"),
                "preamble_text_sha256": self.cells.get(name, {})
                .get("prompt", {})
                .get("preamble_sha256"),
                "tokens": self.cells.get(name, {}).get("prompt", {}).get(
                    "preamble_tokens"
                ),
                "rows": self.cells.get(name, {}).get("prompt", {}).get(
                    "preamble_rows"
                ),
                "unique_first_block_token_span": self.cells.get(name, {})
                .get("prompt", {})
                .get("unique_first_block_token_span"),
                "observed_prompt_tokens": self.cells.get(name, {})
                .get("prompt", {})
                .get("observed_tokens"),
            }
            for name in lifecycle_targets
        }
        preamble_hashes = [
            row["first_block_token_ids_sha256"]
            for row in lifecycle_preambles.values()
        ]
        lifecycle_preambles_valid = (
            all(preamble_hashes)
            and len(set(preamble_hashes)) == len(lifecycle_targets)
            and all(
                isinstance(row["tokens"], int)
                and 4096 <= row["tokens"] < lifecycle_targets[name]
                and row["rows"] == cache.UNIQUE_PREAMBLE_ROWS
                and row["unique_first_block_token_span"] == 4096
                and row["observed_prompt_tokens"] == lifecycle_targets[name]
                for name, row in lifecycle_preambles.items()
            )
        )
        self.phase.gate(
            "cache:recheck:lifecycle-preambles-exact-and-unique",
            lifecycle_preambles_valid,
            lifecycle_preambles,
        )
        self.finished_at = time.time()
        self.save_summary()
        failed = [row for row in self.phase.checks if not row["passed"]]
        rt.save_json(
            "cache-recheck-complete.json",
            {
                "schema": RECHECK_SCHEMA,
                "run_id": self.phase.run_id,
                "completed_at": self.finished_at,
                "summary": str(rt.ROOT / "cache-recheck-summary.json"),
                "all_requested_cells_attempted": attempted,
                "failed_gate_count": len(failed),
                "qualification_passed": attempted and not failed,
            },
        )

    def run(self) -> None:
        rt.note(f"R26 CACHE RECHECK START run_id={self.phase.run_id}")
        self.save_summary()
        try:
            for stage, operation in (
                ("lifecycles", self.run_lifecycles),
                ("needles", self.run_needles),
                (
                    "40-document-eviction",
                    lambda: self.invoke(
                        "40-document-eviction", self.run_eviction
                    ),
                ),
                (
                    "focused-stock-vs-overlay",
                    lambda: self.invoke(
                        "focused-stock-vs-overlay", self.run_focused
                    ),
                ),
                (
                    "native-canaries",
                    lambda: self.invoke(
                        "native-canaries", self.phase.native_matrix
                    ),
                ),
            ):
                try:
                    operation()
                except Exception as error:
                    self.phase.gate(
                        f"cache:recheck:{stage}:coordinator-exception",
                        False,
                        {
                            "failure_class": "coordinator_harness_exception",
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    self.phase.stop(f"cache-recheck-{stage}-coordinator")
                    self.save_summary()
        finally:
            self.phase.stop("cache-recheck-final")
            self.cleanup_remaining_l2()
            self.verify_prior_receipts()
            self.finish()
            rt.note("R26 CACHE RECHECK COMPLETE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run corrected R26 cache qualification cells in a fresh BATTERY_ROOT."
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="print the fixed recheck plan/schema without starting a runtime",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.print_contract:
        print(json.dumps(contract(), indent=2, sort_keys=True))
        return
    CacheRecheckPhase().run()


if __name__ == "__main__":
    main()
