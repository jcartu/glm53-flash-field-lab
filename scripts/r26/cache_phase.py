#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from . import runtime
except ImportError:
    import runtime

PROBE = Path(__file__).with_name("cache_probe.py")
CONFIG_PROBE = Path(__file__).with_name("cache_config_probe.py")
SOURCE_ROOT = Path("/home/josh/omp-workspace/drock-lmcache/LMCache")
NORMAL_L2_CAP_GB = 160
EVICTION_L1_GB = 4
EVICTION_L2_GB = 2
STORE_DRAIN_SECONDS = 15
TARGET_80K = 80_000
TARGET_1M = 1_000_000
NEEDLE_TOKENS = 990_000
PREFILL_TOKENS = 33_000
FOCUSED_TOKENS = 16_384
PHASE_SCHEMA = "r26-cache-phase/v2"
SEMANTIC_MAX_TOKENS = 512
NEEDLE_MAX_TOKENS = 1024
UNIQUE_PREAMBLE_ROWS = 128
PRODUCTION_L2 = Path("/mnt/2king/lmcache-l2")
CONNECTOR_TIMER_RE = re.compile(
    r"Retrieved\s+(?P<tokens>[0-9,]+)\s+tokens\s+in\s+(?P<seconds>[0-9.]+)\s+seconds",
    re.IGNORECASE,
)


class CachePhase:
    def __init__(self) -> None:
        requested = os.environ.get("CACHE_PHASE_RUN_ID")
        raw_id = requested or f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}"
        self.run_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", raw_id)
        self.checks: list[dict[str, Any]] = []
        self.cells: dict[str, Any] = {}
        self.limitations = [
            "Storage topology is recorded in cache-storage-filesystem-evidence.json; no stronger NVMe or same-filesystem claim is inferred when its checks fail. No disk throttling is performed, so this phase does not reproduce iSource's SATA observation.",
            "API wall time and connector-emitted retrieve timer segments are recorded separately. A connector segment is not treated as complete restore latency or scheduler wait time.",
            "Generated-text byte equality is not byte-level KV equality. Shipped source coverage is audited separately and no stronger claim is synthesized.",
            "One-token period completions are latency/cache-transfer probes only. Empty visible finals at a length stop are budget-limited and provide no recall or answer-quality evidence.",
            "Visible-answer hashes cover exact user-visible UTF-8 content. Full content-plus-reasoning hashes are retained only as diagnostics and are never used as KV-byte evidence.",
            "The D-Rock candidate is a separately labelled Python overlay. Its image digest and declared patch revision are recorded separately from the inherited base source.lock.",
        ]
        self._owned_dirs: set[Path] = set()

    def gate(self, name: str, passed: bool, detail: object) -> bool:
        row = {"name": name, "passed": bool(passed), "detail": detail}
        self.checks.append(row)
        runtime.record_gate(name, bool(passed), detail)
        return bool(passed)

    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def run_probe(
        self,
        command: str,
        label: str,
        arguments: list[str],
        *,
        timeout: int,
    ) -> tuple[int, dict[str, Any], Path]:
        output = runtime.ROOT / f"{label}.json"
        code = runtime.run(
            [
                "python3",
                str(PROBE),
                command,
                *arguments,
                "--out",
                str(output),
            ],
            label=label,
            timeout=timeout,
            env=runtime.PROXY_ENV,
        )
        return code, self.load_json(output), output

    def config_snapshot(
        self,
        label: str,
        *,
        image: str,
        mode: str,
        l2_host: Path | None = None,
        max_l2_gb: int = NORMAL_L2_CAP_GB,
        expected_l1_gb: int | None = None,
    ) -> tuple[int, dict[str, Any], Path]:
        output = runtime.ROOT / f"{label}.json"
        arguments = [
            "python3",
            str(CONFIG_PROBE),
            "--container",
            runtime.NAME,
            "--out",
            str(output),
            "--vllm-port",
            str(runtime.PORT),
            "--expected-image",
            image,
            "--expected-mode",
            mode,
            "--max-l2-gb",
            str(max_l2_gb),
        ]
        if l2_host is not None:
            arguments += ["--expected-l2-host", str(l2_host)]
        if expected_l1_gb is not None:
            arguments += ["--expected-l1-gb", str(expected_l1_gb)]
        code = runtime.run(arguments, label=label, timeout=180, env=runtime.PROXY_ENV)
        return code, self.load_json(output), output

    def owned_l2(self, suffix: str) -> Path:
        root = runtime.L2_HOST_ROOT.resolve()
        path = root / f"cache-phase-{self.run_id}-{suffix}"
        if path.parent.resolve() != root:
            raise RuntimeError(f"refusing non-child cache path {path}")
        self._owned_dirs.add(path)
        return path

    @staticmethod
    def inventory(path: Path) -> dict[str, Any]:
        files = 0
        byte_count = 0
        if path.exists():
            for current, _, names in os.walk(path):
                for name in names:
                    try:
                        byte_count += Path(current, name).stat().st_size
                        files += 1
                    except OSError:
                        continue
        return {"path": str(path), "exists": path.exists(), "files": files, "bytes": byte_count}

    def cleanup_owned_l2(self, path: Path, label: str) -> bool:
        root = runtime.L2_HOST_ROOT.resolve()
        resolved = path.resolve(strict=False)
        before = self.inventory(path)
        safe = (
            resolved.parent == root
            and resolved.name.startswith(f"cache-phase-{self.run_id}-")
            and resolved != PRODUCTION_L2.resolve(strict=False)
            and PRODUCTION_L2.resolve(strict=False) not in resolved.parents
        )
        error = None
        removed = False
        if safe:
            try:
                if path.exists():
                    runtime.cleanup_l2_child(path)
                removed = not path.exists()
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
        receipt = {
            "safe_path_check": safe,
            "before": before,
            "removed": removed,
            "error": error,
            "l2_root": str(root),
            "production_l2_blocklist": str(PRODUCTION_L2),
            "cleanup_helper": "runtime.cleanup_l2_child",
        }
        runtime.save_json(f"{label}.json", receipt)
        return self.gate(f"cache:l2-cleanup:{label}", safe and removed, receipt)

    @staticmethod
    def request_summary(data: dict[str, Any]) -> dict[str, Any]:
        summary = data.get("summary")
        return summary if isinstance(summary, dict) else {}

    @staticmethod
    def lmcache_hits(data: dict[str, Any]) -> int | None:
        stats = CachePhase.request_summary(data).get("cache_stats")
        if not isinstance(stats, dict):
            return None
        value = stats.get("num_lmcache_cached_tokens")
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def metric_totals(config: dict[str, Any], source: str = "lmcache_metrics") -> dict[str, float]:
        section = config.get(source)
        if not isinstance(section, dict):
            return {}
        totals = section.get("totals")
        if not isinstance(totals, dict):
            return {}
        result: dict[str, float] = {}
        for key, value in totals.items():
            if isinstance(value, (int, float)):
                result[str(key)] = float(value)
        return result

    @staticmethod
    def metrics_matching(config: dict[str, Any], pattern: str, source: str = "lmcache_metrics") -> dict[str, float]:
        regex = re.compile(pattern, re.IGNORECASE)
        return {
            name: value
            for name, value in CachePhase.metric_totals(config, source).items()
            if regex.search(name)
        }

    def boot(
        self,
        label: str,
        *,
        image: str,
        cache: str,
        kv: str,
        spec: str,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> bool:
        # This candidate removed the legacy selector but retains the R26
        # wrapper. Use its documented native CLI with the same fixed share.
        if image == runtime.OVERLAY_IMAGE:
            extra_env = {**(extra_env or {}), "FAIRNESS_ENGINE": "none"}
            extra_args = [*(extra_args or []), "--prefill-compute-share", "0.4"]
        try:
            return runtime.boot(
                label,
                image=image,
                tp=4,
                dcp=4,
                spec=spec,
                cache=cache,
                kv=kv,
                extra_env=extra_env,
                extra_args=extra_args,
            )
        except Exception as error:
            self.gate(
                f"cache:boot-exception:{label}",
                False,
                {"error": f"{type(error).__name__}: {error}"},
            )
            return False

    def stop(self, label: str) -> None:
        try:
            runtime.stop()
        except Exception as error:
            self.gate(
                f"cache:stop:{label}",
                False,
                {"error": f"{type(error).__name__}: {error}"},
            )

    def restart(self, label: str) -> bool:
        code = runtime.run(
            ["docker", "restart", runtime.NAME],
            label=f"{label}-docker-restart",
            timeout=900,
        )
        healthy = code == 0 and runtime.wait_health(timeout=900)
        runtime.capture(f"{label}-after-restart")
        self.gate(
            f"cache:restart:{label}",
            healthy,
            {"returncode": code, "healthy": healthy},
        )
        return healthy

    def prepare_prompt(
        self,
        label: str,
        *,
        target_tokens: int,
        identity: str,
        kind: str,
    ) -> tuple[bool, Path, dict[str, Any]]:
        prompt_path = runtime.ROOT / f"{label}.prompt.txt"
        code, data, output = self.run_probe(
            "prepare",
            f"{label}-prepare",
            [
                "--port",
                str(runtime.PORT),
                "--model",
                runtime.MODEL_NAME,
                "--target-tokens",
                str(target_tokens),
                "--identity",
                identity,
                "--kind",
                kind,
                "--prompt-out",
                str(prompt_path),
            ],
            timeout=900,
        )
        preamble_tokens = data.get("preamble_tokens")
        passed = (
            code == 0
            and prompt_path.exists()
            and data.get("observed_tokens") == target_tokens
            and isinstance(preamble_tokens, int)
            and 4096 <= preamble_tokens < target_tokens
            and data.get("preamble_rows") == UNIQUE_PREAMBLE_ROWS
            and data.get("unique_first_block_token_span") == 4096
            and isinstance(data.get("first_block_token_ids_sha256"), str)
        )
        self.gate(
            f"cache:prompt:{label}",
            passed,
            {
                "returncode": code,
                "metadata": str(output),
                "prompt": str(prompt_path),
                "observed_tokens": data.get("observed_tokens"),
                "preamble_tokens": data.get("preamble_tokens"),
                "preamble_rows": data.get("preamble_rows"),
                "unique_first_block_token_span": data.get(
                    "unique_first_block_token_span"
                ),
                "first_block_token_ids_sha256": data.get(
                    "first_block_token_ids_sha256"
                ),
                "preamble_fits_target": (
                    isinstance(preamble_tokens, int)
                    and 4096 <= preamble_tokens < target_tokens
                ),
                "tokenizer_method": data.get("tokenizer_method"),
            },
        )
        return passed, prompt_path, data

    def request(
        self,
        label: str,
        *,
        prompt: Path,
        cache_salt: str,
        max_tokens: int,
        deadline: int,
        expected_tokens: int,
        expected_reference: str | None = None,
        ignore_eos: bool = False,
    ) -> tuple[int, dict[str, Any], Path]:
        arguments = [
            "--port",
            str(runtime.PORT),
            "--model",
            runtime.MODEL_NAME,
            "--prompt",
            str(prompt),
            "--cache-salt",
            cache_salt,
            "--label",
            label,
            "--max-tokens",
            str(max_tokens),
            "--deadline",
            str(deadline),
            "--expected-tokens",
            str(expected_tokens),
        ]
        if expected_reference is not None:
            arguments += ["--expected-reference", expected_reference]
        if ignore_eos:
            arguments.append("--ignore-eos")
        return self.run_probe("request", label, arguments, timeout=deadline + 120)

    def reset_local(self, label: str) -> tuple[int, dict[str, Any], Path]:
        return self.run_probe(
            "reset-local",
            label,
            ["--port", str(runtime.PORT), "--label", label],
            timeout=180,
        )

    @staticmethod
    def lifecycle_verdict(
        target: int,
        kind: str,
        cold: dict[str, Any],
        warm: dict[str, Any],
        restart: dict[str, Any],
    ) -> dict[str, Any]:
        rows = [cold, warm, restart]
        summaries = [CachePhase.request_summary(row) for row in rows]
        visible_outputs = [
            str(summary.get("content") or "") for summary in summaries
        ]
        visible_hashes = [
            summary.get("visible_output_sha256") for summary in summaries
        ]
        full_hashes = [summary.get("full_output_sha256") for summary in summaries]
        visible_output_byte_equal = (
            visible_hashes[0] is not None
            and all(
                output.encode() == visible_outputs[0].encode()
                for output in visible_outputs[1:]
            )
        )
        hits = [CachePhase.lmcache_hits(row) for row in rows]
        answer_evidence = [
            row.get("answer_evidence")
            if isinstance(row.get("answer_evidence"), dict)
            else None
            for row in rows
        ]
        checks = {
            "all_requests_ok": all(row.get("passed") is True for row in rows),
            "exact_prompt_tokens": all(
                summary.get("prompt_tokens") == target for summary in summaries
            ),
            "cold_external_miss": hits[0] == 0,
            "warm_external_hit": isinstance(hits[1], int) and hits[1] > 0,
            "restart_l2_hit": isinstance(hits[2], int) and hits[2] > 0,
        }
        if kind == "reference":
            checks.update(
                {
                    "all_visible_answers_complete": all(
                        evidence is not None
                        and evidence.get("visible_final_complete") is True
                        for evidence in answer_evidence
                    ),
                    "reference_retained_in_visible_final": all(
                        evidence is not None
                        and evidence.get("accepted_visible_recall") is True
                        for evidence in answer_evidence
                    ),
                    "visible_answer_byte_equal": visible_output_byte_equal,
                    "no_semantic_answer_budget_limit": all(
                        evidence is not None
                        and evidence.get("generation_budget_reached") is False
                        for evidence in answer_evidence
                    ),
                }
            )
            measurement_scope = (
                "cache transfer plus complete user-visible reference recall"
            )
        else:
            measurement_scope = (
                "one-token forced period latency/cache-transfer only; "
                "no recall or answer-quality conclusion"
            )
        return {
            "schema": PHASE_SCHEMA,
            "target_tokens": target,
            "kind": kind,
            "measurement_scope": measurement_scope,
            "answer_quality_evaluated": kind == "reference",
            "response_protocol_complete": [
                summary.get("response_protocol_complete") for summary in summaries
            ],
            "request_ok": [bool(row.get("passed")) for row in rows],
            "prompt_tokens": [summary.get("prompt_tokens") for summary in summaries],
            "finish_reason": [summary.get("finish_reason") for summary in summaries],
            "visible_final_status": [
                summary.get("visible_final_status") for summary in summaries
            ],
            "visible_output_sha256": visible_hashes,
            "visible_output_byte_equal": visible_output_byte_equal,
            "full_output_sha256_diagnostic": full_hashes,
            "full_output_byte_equal_diagnostic": full_hashes[0] is not None
            and len(set(full_hashes)) == 1,
            "visible_answer_evidence": answer_evidence,
            "lmcache_hit_tokens": hits,
            "checks": checks,
        }

    def lifecycle_cell(
        self,
        *,
        kv: str,
        target: int,
        kind: str,
        host_dir: Path,
        namespace: str,
    ) -> dict[str, Any]:
        shape = "fp8" if kv == "fp8_ds_mla" else "packed-nvfp4"
        size = "80k" if target == TARGET_80K else "exact-1m"
        label = f"cache-official-r26-{shape}-{size}"
        env = {
            "LMCACHE_L2_HOST_DIR": str(host_dir),
            "LMCACHE_L2_MAX_CAPACITY_GB": str(NORMAL_L2_CAP_GB),
            "LMCACHE_INSTANCE_ID": namespace,
            "LMCACHE_SHM_NAME": namespace,
            "LMCACHE_L2_ENABLED": "1",
        }
        result: dict[str, Any] = {
            "schema": PHASE_SCHEMA,
            "label": label,
            "attempted": True,
            "method": {
                "target_tokens": target,
                "kind": kind,
                "semantic_max_tokens": (
                    SEMANTIC_MAX_TOKENS if kind == "reference" else None
                ),
                "period_max_tokens": 1 if kind == "period" else None,
                "period_scope": (
                    "latency/cache-transfer only" if kind == "period" else None
                ),
                "visible_final_is_semantic_oracle": kind == "reference",
            },
        }
        if not self.boot(
            label,
            image=runtime.IMAGE,
            cache="lmcache",
            kv=kv,
            spec="mtp0",
            extra_env=env,
        ):
            result["booted"] = False
            self.gate(
                f"cache:official-r26:{shape}:{size}:lifecycle",
                False,
                {"reason": "boot failed"},
            )
            self.stop(label)
            return result
        result["booted"] = True
        try:
            _, before, before_path = self.config_snapshot(
                f"{label}-config-before",
                image=runtime.IMAGE,
                mode="l2-on",
                l2_host=host_dir,
            )
            config_ok = bool(before.get("passed"))
            self.gate(
                f"cache:official-r26:{shape}:{size}:generated-config",
                config_ok,
                {"receipt": str(before_path), "checks": before.get("checks")},
            )
            if size == "80k":
                self.gate(
                    f"cache:official-r26:{shape}:cpu-only-sidecar",
                    before.get("sidecar", {}).get("cpu_only") is True,
                    before.get("sidecar"),
                )
                self.gate(
                    f"cache:official-r26:{shape}:private-128g-shm",
                    before.get("container_config", {}).get("private_shm_128g_or_larger")
                    is True,
                    before.get("container_config"),
                )
            identity = f"{self.run_id}-{shape}-{size}-unique"
            prompt_ok, prompt, prompt_meta = self.prepare_prompt(
                label,
                target_tokens=target,
                identity=identity,
                kind=kind,
            )
            result["prompt"] = prompt_meta
            if not prompt_ok:
                self.gate(
                    f"cache:official-r26:{shape}:{size}:requests",
                    False,
                    {"reason": "prompt preparation failed"},
                )
                return result
            salt = f"{self.run_id}-{shape}-{size}-salt"
            reference = prompt_meta.get("reference_code") if kind == "reference" else None
            max_tokens = SEMANTIC_MAX_TOKENS if kind == "reference" else 1
            deadline = 1800 if target >= TARGET_1M else 900
            _, cold, cold_path = self.request(
                f"{label}-cold",
                prompt=prompt,
                cache_salt=salt,
                max_tokens=max_tokens,
                deadline=deadline,
                expected_tokens=target,
                expected_reference=reference,
                ignore_eos=kind == "period",
            )
            time.sleep(STORE_DRAIN_SECONDS)
            _, post_cold, post_cold_path = self.config_snapshot(
                f"{label}-config-post-cold",
                image=runtime.IMAGE,
                mode="l2-on",
                l2_host=host_dir,
            )
            reset_code, reset_data, reset_path = self.reset_local(f"{label}-reset-local")
            self.gate(
                f"cache:official-r26:{shape}:{size}:local-reset",
                reset_code == 0 and reset_data.get("passed") is True,
                {"receipt": str(reset_path), "returncode": reset_code},
            )
            _, warm, warm_path = self.request(
                f"{label}-warm",
                prompt=prompt,
                cache_salt=salt,
                max_tokens=max_tokens,
                deadline=deadline,
                expected_tokens=target,
                expected_reference=reference,
                ignore_eos=kind == "period",
            )
            time.sleep(STORE_DRAIN_SECONDS)
            _, pre_restart, pre_restart_path = self.config_snapshot(
                f"{label}-config-pre-restart",
                image=runtime.IMAGE,
                mode="l2-on",
                l2_host=host_dir,
            )
            restarted = self.restart(label)
            restart_data: dict[str, Any] = {}
            restart_path = runtime.ROOT / f"{label}-restart.json"
            post_restart: dict[str, Any] = {}
            post_restart_path = runtime.ROOT / f"{label}-config-post-restart.json"
            if restarted:
                _, restart_data, restart_path = self.request(
                    f"{label}-restart",
                    prompt=prompt,
                    cache_salt=salt,
                    max_tokens=max_tokens,
                    deadline=deadline,
                    expected_tokens=target,
                    expected_reference=reference,
                    ignore_eos=kind == "period",
                )
                _, post_restart, post_restart_path = self.config_snapshot(
                    f"{label}-config-post-restart",
                    image=runtime.IMAGE,
                    mode="l2-on",
                    l2_host=host_dir,
                )
            verdict = self.lifecycle_verdict(target, kind, cold, warm, restart_data)
            result.update(
                {
                    "cold": str(cold_path),
                    "warm": str(warm_path),
                    "restart": str(restart_path),
                    "config_before": str(before_path),
                    "config_post_cold": str(post_cold_path),
                    "config_pre_restart": str(pre_restart_path),
                    "config_post_restart": str(post_restart_path),
                    "verdict": verdict,
                }
            )
            lifecycle_ok = all(verdict["checks"].values())
            self.gate(
                f"cache:official-r26:{shape}:{size}:lifecycle",
                lifecycle_ok,
                verdict,
            )
            l1_before_warm = self.metrics_matching(
                post_cold, r"(l0_l1|l1).*(load|retrieve|read)"
            )
            l1_after_warm = self.metrics_matching(
                pre_restart, r"(l0_l1|l1).*(load|retrieve|read)"
            )
            l1_metric_names = set(l1_before_warm) | set(l1_after_warm)
            l1_warm_metrics = {
                name: l1_after_warm.get(name, 0.0) - l1_before_warm.get(name, 0.0)
                for name in sorted(l1_metric_names)
            }
            l2_load_before_warm = self.metrics_matching(
                post_cold, r"l2.*(prefetch|load).*(hit|completed|chunks)"
            )
            l2_load_after_warm = self.metrics_matching(
                pre_restart, r"l2.*(prefetch|load).*(hit|completed|chunks)"
            )
            l2_metric_names = set(l2_load_before_warm) | set(l2_load_after_warm)
            l2_load_during_warm = {
                name: l2_load_after_warm.get(name, 0.0)
                - l2_load_before_warm.get(name, 0.0)
                for name in sorted(l2_metric_names)
            }
            warm_l1_path = (
                isinstance(self.lmcache_hits(warm), int)
                and self.lmcache_hits(warm) > 0
                and any(value > 0 for value in l1_warm_metrics.values())
                and not any(value > 0 for value in l2_load_during_warm.values())
            )
            self.gate(
                f"cache:official-r26:{shape}:{size}:warm-l1-path",
                warm_l1_path,
                {
                    "l1_metrics": l1_warm_metrics,
                    "l2_load_delta_during_warm": l2_load_during_warm,
                    "warm_external_hit_tokens": self.lmcache_hits(warm),
                },
            )
            store_metrics = self.metrics_matching(
                pre_restart, r"l2.*store.*(completed|succeeded|objects|chunks)"
            )
            load_metrics = self.metrics_matching(
                post_restart, r"l2.*(prefetch|load).*(hit|completed|chunks)"
            )
            activity_ok = any(value > 0 for value in store_metrics.values()) and any(
                value > 0 for value in load_metrics.values()
            )
            self.gate(
                f"cache:official-r26:{shape}:{size}:l2-metrics",
                activity_ok,
                {"store": store_metrics, "restart_load": load_metrics},
            )
            return result
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            self.gate(
                f"cache:official-r26:{shape}:{size}:exception",
                False,
                result["error"],
            )
            return result
        finally:
            self.stop(label)

    def official_lifecycles(self) -> None:
        for kv in ("fp8_ds_mla", "nvfp4_ds_mla"):
            shape = "fp8" if kv == "fp8_ds_mla" else "packed-nvfp4"
            host_dir = self.owned_l2(f"official-{shape}")
            namespace = f"r26-cache-{self.run_id}-{shape}"
            rows = []
            for target, kind in ((TARGET_80K, "period"), (TARGET_1M, "reference")):
                rows.append(
                    self.lifecycle_cell(
                        kv=kv,
                        target=target,
                        kind=kind,
                        host_dir=host_dir,
                        namespace=namespace,
                    )
                )
            self.cells[f"official-{shape}-lifecycles"] = rows
            if kv == "nvfp4_ds_mla":
                self.cells["official-packed-nvfp4-needles"] = self.needle_cell(
                    host_dir, namespace
                )
            self.cleanup_owned_l2(host_dir, f"cleanup-official-{shape}")

    def needle_cell(
        self,
        host_dir: Path,
        namespace: str,
        kv: str = "nvfp4_ds_mla",
    ) -> dict[str, Any]:
        shape = "fp8" if kv == "fp8_ds_mla" else "packed-nvfp4"
        label = f"cache-official-r26-{shape}-unique-needles"
        gate_prefix = f"cache:official-r26:{shape}:unique-1m-needles"
        env = {
            "LMCACHE_L2_HOST_DIR": str(host_dir),
            "LMCACHE_L2_MAX_CAPACITY_GB": str(NORMAL_L2_CAP_GB),
            "LMCACHE_INSTANCE_ID": namespace,
            "LMCACHE_SHM_NAME": namespace,
            "LMCACHE_L2_ENABLED": "1",
        }
        result = {
            "schema": PHASE_SCHEMA,
            "label": label,
            "shape": shape,
            "attempted": True,
            "method": {
                "depths_percent": [10, 50, 90],
                "target_tokens": NEEDLE_TOKENS,
                "cold_only": True,
                "semantic_max_tokens": NEEDLE_MAX_TOKENS,
                "visible_final_only_oracle": True,
            },
        }
        if not self.boot(
            label,
            image=runtime.IMAGE,
            cache="lmcache",
            kv=kv,
            spec="mtp0",
            extra_env=env,
        ):
            result["booted"] = False
            self.gate(gate_prefix, False, "boot failed")
            self.stop(label)
            return result
        result["booted"] = True
        try:
            _, config, config_path = self.config_snapshot(
                f"{label}-config",
                image=runtime.IMAGE,
                mode="l2-on",
                l2_host=host_dir,
            )
            self.gate(
                f"{gate_prefix}:generated-config",
                config.get("passed") is True,
                {"receipt": str(config_path), "checks": config.get("checks")},
            )
            code, data, output = self.run_probe(
                "needles",
                label,
                [
                    "--port",
                    str(runtime.PORT),
                    "--model",
                    runtime.MODEL_NAME,
                    "--suite-identity",
                    f"{self.run_id}-official-{shape}-unique-needles",
                    "--target-tokens",
                    str(NEEDLE_TOKENS),
                    "--depths",
                    "10",
                    "50",
                    "90",
                    "--deadline",
                    "1800",
                ],
                timeout=6000,
            )
            result.update(
                {
                    "returncode": code,
                    "receipt": str(output),
                    "config": str(config_path),
                    "checks": data.get("checks"),
                    "answer_outcomes": {
                        str(row.get("depth_percent_requested")): row.get(
                            "answer_evidence", {}
                        ).get("outcome")
                        for row in data.get("results", [])
                    },
                    "budget_limited_depths": [
                        row.get("depth_percent_requested")
                        for row in data.get("results", [])
                        if row.get("answer_evidence", {}).get(
                            "generation_budget_reached"
                        )
                    ],
                    "reasoning_only_matches_not_counted": [
                        row.get("depth_percent_requested")
                        for row in data.get("results", [])
                        if row.get("reasoning_only_match_diagnostic") is True
                    ],
                }
            )
            self.gate(
                gate_prefix,
                code == 0 and data.get("passed") is True and config.get("passed") is True,
                result,
            )
            return result
        finally:
            runtime.capture(label)
            self.stop(label)

    def eviction_cell(self) -> None:
        label = "cache-official-r26-40-document-eviction"
        host_dir = self.owned_l2("official-eviction")
        namespace = f"r26-cache-{self.run_id}-eviction"
        env = {
            "LMCACHE_L2_HOST_DIR": str(host_dir),
            "LMCACHE_L2_MAX_CAPACITY_GB": str(EVICTION_L2_GB),
            "LMCACHE_L1_SIZE_GB": str(EVICTION_L1_GB),
            "LMCACHE_L1_INIT_SIZE_GB": str(EVICTION_L1_GB),
            "LMCACHE_INSTANCE_ID": namespace,
            "LMCACHE_SHM_NAME": namespace,
            "LMCACHE_L2_ENABLED": "1",
        }
        result: dict[str, Any] = {
            "schema": PHASE_SCHEMA,
            "label": label,
            "attempted": True,
            "method": {
                "documents": 40,
                "random_words_per_document": 12_000,
                "oracle": "direct unique labelled-marker value retrieval",
                "semantic_max_tokens": SEMANTIC_MAX_TOKENS,
                "ground_truth_channel": "user-visible final only",
                "natural_completion_required": True,
                "exact_visible_byte_identity_checked_separately": True,
                "full_output_hash_scope": "diagnostic only",
                "l1_capacity_gb": EVICTION_L1_GB,
                "l2_capacity_gb": EVICTION_L2_GB,
            },
        }
        if not self.boot(
            label,
            image=runtime.IMAGE,
            cache="lmcache",
            kv="nvfp4_ds_mla",
            spec="mtp0",
            extra_env=env,
        ):
            result["booted"] = False
            self.gate("cache:official-r26:40-document-eviction-integrity", False, "boot failed")
            self.stop(label)
            self.cleanup_owned_l2(host_dir, "cleanup-official-eviction")
            self.cells["eviction"] = result
            return
        result["booted"] = True
        try:
            _, before, before_path = self.config_snapshot(
                f"{label}-config-before",
                image=runtime.IMAGE,
                mode="l2-on",
                l2_host=host_dir,
                max_l2_gb=EVICTION_L2_GB,
                expected_l1_gb=EVICTION_L1_GB,
            )
            self.gate(
                "cache:official-r26:40-document-eviction-generated-config",
                before.get("passed") is True,
                {"receipt": str(before_path), "checks": before.get("checks")},
            )
            code, data, output = self.run_probe(
                "eviction",
                label,
                [
                    "--port",
                    str(runtime.PORT),
                    "--model",
                    runtime.MODEL_NAME,
                    "--suite-identity",
                    f"{self.run_id}-40doc",
                    "--docs",
                    "40",
                    "--words",
                    "12000",
                    "--deadline",
                    "900",
                    "--store-drain-seconds",
                    "10",
                ],
                timeout=12_000,
            )
            _, after, after_path = self.config_snapshot(
                f"{label}-config-after",
                image=runtime.IMAGE,
                mode="l2-on",
                l2_host=host_dir,
                max_l2_gb=EVICTION_L2_GB,
                expected_l1_gb=EVICTION_L1_GB,
            )
            runtime.capture(label)
            integrity = (
                code == 0
                and data.get("passed") is True
                and data.get("docs") == 40
                and len(data.get("pass1", [])) == 40
                and len(data.get("churn", [])) == 40
                and len(data.get("pass2", [])) == 40
            )
            self.gate(
                "cache:official-r26:40-document-eviction-integrity",
                integrity,
                {"receipt": str(output), "returncode": code, "checks": data.get("checks")},
            )
            eviction_metrics = {
                **self.metrics_matching(
                    after,
                    r"(?:l1|l2)_evicted.*(?:objects|chunks)|eviction.*triggered",
                ),
                **self.metrics_matching(after, r"l2.*deleted"),
            }
            log_path = runtime.ROOT / f"{label}.docker.log"
            log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            log_evidence = bool(
                re.search(
                    r"above watermark|triggering eviction|evicted [1-9]|"
                    r"eviction_loop_triggered",
                    log_text,
                    re.IGNORECASE,
                )
            )
            effective = any(value > 0 for value in eviction_metrics.values()) or log_evidence
            self.gate(
                "cache:official-r26:40-document-eviction-observed",
                effective,
                {
                    "metrics": eviction_metrics,
                    "log_evidence": log_evidence,
                    "before": str(before_path),
                    "after": str(after_path),
                },
            )
            result.update(
                {
                    "receipt": str(output),
                    "config_before": str(before_path),
                    "config_after": str(after_path),
                    "integrity": integrity,
                    "probe_checks": data.get("checks"),
                    "probe_outcome_counts": data.get("outcome_counts"),
                    "document_counts": {
                        "pass1": len(data.get("pass1", [])),
                        "churn": len(data.get("churn", [])),
                        "pass2": len(data.get("pass2", [])),
                    },
                    "eviction_metrics": eviction_metrics,
                    "eviction_log_evidence": log_evidence,
                    "eviction_observed": effective,
                }
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            self.gate("cache:official-r26:eviction-exception", False, result["error"])
        finally:
            self.stop(label)
            self.cleanup_owned_l2(host_dir, "cleanup-official-eviction")
            self.cells["eviction"] = result

    def native_cell(
        self,
        *,
        release: str = "r26",
        image: str | None = None,
        gpu_memory_utilization: str = "0.93",
        kv: str = "nvfp4_ds_mla",
        role: str | None = None,
        load_format: str | None = None,
    ) -> dict[str, Any]:
        if release not in {"r25", "r26"}:
            raise ValueError(f"unsupported native canary release {release}")
        if gpu_memory_utilization not in {"0.90", "0.93"}:
            raise ValueError(
                f"unsupported native canary GPU memory utilization {gpu_memory_utilization}"
            )
        if kv not in {"fp8_ds_mla", "nvfp4_ds_mla"}:
            raise ValueError(f"unsupported native canary KV format {kv}")
        if load_format not in {None, "safetensors"}:
            raise ValueError(f"unsupported diagnostic weight loader {load_format}")
        image = image or (runtime.R25_IMAGE if release == "r25" else runtime.IMAGE)
        shape = "fp8" if kv == "fp8_ds_mla" else "packed-nvfp4"
        role = role or (
            "loader-diagnostic" if load_format else
            "matched-baseline" if gpu_memory_utilization == "0.93" else "diagnostic"
        )
        gmu_label = gpu_memory_utilization.replace(".", "")
        label = f"cache-native-{role}-{release}-gmu{gmu_label}-{shape}"
        cell_key = f"native-{release}-gmu{gmu_label}-{shape}"
        gate_name = (
            f"cache:native:{role}:{release}:gmu{gpu_memory_utilization}:{shape}"
        )
        if load_format:
            label += f"-{load_format}"
            cell_key += f"-{load_format}"
            gate_name += f":loader-{load_format}"
        result: dict[str, Any] = {
            "schema": PHASE_SCHEMA,
            "label": label,
            "cell_key": cell_key,
            "attempted": True,
            "role": role,
            "configuration": {
                "release": release,
                "image": image,
                "gpu_memory_utilization": gpu_memory_utilization,
                "kv": kv,
                "load_format_override": load_format,
                "cache": "native",
                "tp": 4,
                "dcp": 4,
                "spec": "mtp0",
            },
            "method": {
                "prompt_tokens": TARGET_80K,
                "completion_tokens": 1,
                "completion_scope": (
                    "forced one-token latency/cache-transfer canary only; "
                    "no recall or answer-quality conclusion"
                ),
                "memory_transfer_oracle": (
                    "native offload store and load metric activity plus effective "
                    "native configuration"
                ),
                "generated_output_hashes": "diagnostic only",
                "no_fallback": True,
            },
        }
        native_env = {"GPU_MEMORY_UTILIZATION": gpu_memory_utilization}
        if load_format:
            native_env["LOAD_FORMAT"] = load_format
        booted = self.boot(
            label,
            image=image,
            cache="native",
            kv=kv,
            spec="mtp0",
            extra_env=native_env,
        )
        result["booted"] = booted
        launch_path = runtime.ROOT / f"{label}.launch.json"
        launch = self.load_json(launch_path)
        launch_env = launch.get("env") if isinstance(launch.get("env"), dict) else {}
        launch_matches_requested = bool(
            launch.get("image") == image
            and launch.get("cache") == "native"
            and launch.get("kv") == kv
            and launch.get("tp") == 4
            and launch.get("dcp") == 4
            and launch.get("spec") == "mtp0"
            and launch.get("model_dir") == str(runtime.MODEL)
            and launch_env.get("GPU_MEMORY_UTILIZATION")
            == gpu_memory_utilization
            and (load_format is None or launch_env.get("LOAD_FORMAT") == load_format)
        )
        result["launch_evidence"] = {
            "receipt": str(launch_path),
            "requested_matches_launch": launch_matches_requested,
            "observed_image": launch.get("image"),
            "observed_cache": launch.get("cache"),
            "observed_kv": launch.get("kv"),
            "observed_tp": launch.get("tp"),
            "observed_dcp": launch.get("dcp"),
            "observed_spec": launch.get("spec"),
            "observed_model_dir": launch.get("model_dir"),
            "observed_gpu_memory_utilization": launch_env.get(
                "GPU_MEMORY_UTILIZATION"
            ),
            "observed_load_format_override": launch_env.get("LOAD_FORMAT"),
        }
        if not launch_matches_requested:
            result.update(
                {
                    "passed": False,
                    "failure_class": (
                        "launch_methodology_mismatch"
                        if launch
                        else "launch_receipt_missing"
                    ),
                }
            )
            self.gate(gate_name, False, result)
            self.stop(label)
            self.cells[cell_key] = result
            return result
        if not booted:
            log_path = runtime.ROOT / f"{label}.docker.log"
            log_text = (
                log_path.read_text(errors="replace") if log_path.exists() else ""
            )
            oom = bool(
                re.search(
                    r"CUDA (?:Error: )?out of memory|torch\.OutOfMemoryError|"
                    r"memory allocation failed with OOM",
                    log_text,
                    re.IGNORECASE,
                )
            )
            result.update(
                {
                    "passed": False,
                    "failure_class": (
                        "native_boot_cuda_oom"
                        if oom
                        else "native_boot_failure_non_oom"
                    ),
                    "docker_log": str(log_path),
                    "docker_log_sha256": (
                        hashlib.sha256(log_text.encode()).hexdigest()
                        if log_text
                        else None
                    ),
                }
            )
            self.gate(gate_name, False, result)
            self.stop(label)
            self.cells[cell_key] = result
            return result
        try:
            _, before, before_path = self.config_snapshot(
                f"{label}-config-before", image=image, mode="native"
            )
            identity = (
                f"{self.run_id}-native-{release}-gmu{gmu_label}-{shape}-80k"
            )
            prompt_ok, prompt, metadata = self.prepare_prompt(
                label,
                target_tokens=TARGET_80K,
                identity=identity,
                kind="period",
            )
            if not prompt_ok:
                result.update(
                    {"passed": False, "failure_class": "prompt_harness_failure"}
                )
                self.gate(gate_name, False, result)
                return result
            salt = f"{identity}-salt"
            _, cold, cold_path = self.request(
                f"{label}-cold",
                prompt=prompt,
                cache_salt=salt,
                max_tokens=1,
                deadline=900,
                expected_tokens=TARGET_80K,
                ignore_eos=True,
            )
            time.sleep(STORE_DRAIN_SECONDS)
            reset_code, reset_data, reset_path = self.reset_local(
                f"{label}-reset-local"
            )
            reset_ok = reset_code == 0 and reset_data.get("passed") is True
            self.gate(
                f"{gate_name}:local-reset",
                reset_ok,
                {"receipt": str(reset_path), "returncode": reset_code},
            )
            _, warm, warm_path = self.request(
                f"{label}-warm",
                prompt=prompt,
                cache_salt=salt,
                max_tokens=1,
                deadline=900,
                expected_tokens=TARGET_80K,
                ignore_eos=True,
            )
            _, after, after_path = self.config_snapshot(
                f"{label}-config-after", image=image, mode="native"
            )
            summaries = [self.request_summary(row) for row in (cold, warm)]
            visible_hashes = [
                summary.get("visible_output_sha256") for summary in summaries
            ]
            full_hashes = [
                summary.get("full_output_sha256") for summary in summaries
            ]
            walls = [cold.get("wall_seconds"), warm.get("wall_seconds")]
            offload_store_metrics = self.metrics_matching(
                after, r"kv_offload.*store.*(bytes|time|size)", "vllm_metrics"
            )
            offload_load_metrics = self.metrics_matching(
                after, r"kv_offload.*load.*(bytes|time|size)", "vllm_metrics"
            )
            activity = any(
                value > 0 for value in offload_store_metrics.values()
            ) and any(value > 0 for value in offload_load_metrics.values())
            passed = (
                cold.get("passed") is True
                and warm.get("passed") is True
                and reset_ok
                and all(
                    summary.get("prompt_tokens") == TARGET_80K
                    for summary in summaries
                )
                and all(isinstance(wall, (int, float)) for wall in walls)
                and before.get("passed") is True
                and after.get("passed") is True
                and activity
            )
            result.update(
                {
                    "passed": passed,
                    "failure_class": None if passed else "native_measurement_failure",
                    "prompt": metadata,
                    "cold": str(cold_path),
                    "warm": str(warm_path),
                    "local_reset": str(reset_path),
                    "config_before": str(before_path),
                    "config_after": str(after_path),
                    "api_wall_seconds": {"cold": walls[0], "warm": walls[1]},
                    "warm_latency_lower_diagnostic": (
                        isinstance(walls[0], (int, float))
                        and isinstance(walls[1], (int, float))
                        and float(walls[1]) < float(walls[0])
                    ),
                    "visible_output_sha256_diagnostic": visible_hashes,
                    "full_output_sha256_diagnostic": full_hashes,
                    "visible_final_status": [
                        summary.get("visible_final_status")
                        for summary in summaries
                    ],
                    "semantic_quality_evaluated": False,
                    "offload_store_metrics": offload_store_metrics,
                    "offload_load_metrics": offload_load_metrics,
                    "memory_transfer_activity_observed": activity,
                }
            )
            self.gate(gate_name, passed, result)
            return result
        except Exception as error:
            result.update(
                {
                    "passed": False,
                    "failure_class": "native_harness_exception",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            self.gate(f"{gate_name}:exception", False, result)
            return result
        finally:
            runtime.capture(label)
            self.stop(label)
            self.cells[cell_key] = result

    def native_matrix(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for release, image in (
            ("r25", runtime.R25_IMAGE),
            ("r26", runtime.IMAGE),
        ):
            for kv in ("fp8_ds_mla", "nvfp4_ds_mla"):
                rows.append(
                    self.native_cell(
                        release=release,
                        image=image,
                        gpu_memory_utilization="0.93",
                        kv=kv,
                        role="matched-baseline",
                    )
                )
        for kv in ("fp8_ds_mla", "nvfp4_ds_mla"):
            rows.append(
                self.native_cell(
                    release="r26",
                    image=runtime.IMAGE,
                    gpu_memory_utilization="0.90",
                    kv=kv,
                    role="diagnostic",
                )
            )
        baseline_rows = [
            row
            for row in rows
            if row["configuration"]["gpu_memory_utilization"] == "0.93"
        ]
        diagnostic_rows = [
            row
            for row in rows
            if row["configuration"]["gpu_memory_utilization"] == "0.90"
        ]
        summary = {
            "schema": PHASE_SCHEMA,
            "matrix": rows,
            "contract": {
                "matched_baseline": "R25 and R26 at GMU 0.93, both FP8 and NVFP4",
                "lower_gmu_diagnostic": "R26 at GMU 0.90, both FP8 and NVFP4",
                "no_fallback": True,
                "diagnostic_cannot_override_baseline": True,
            },
            "checks": {
                "all_six_cells_attempted": len(rows) == 6
                and all(row.get("attempted") is True for row in rows),
                "all_matched_gmu093_baselines_passed": all(
                    row.get("passed") is True for row in baseline_rows
                ),
                "all_r25_gmu093_controls_passed": all(
                    row.get("passed") is True
                    for row in baseline_rows
                    if row["configuration"]["release"] == "r25"
                ),
                "all_r26_gmu093_baselines_passed": all(
                    row.get("passed") is True
                    for row in baseline_rows
                    if row["configuration"]["release"] == "r26"
                ),
                "all_r26_gmu090_diagnostics_passed": all(
                    row.get("passed") is True for row in diagnostic_rows
                ),
            },
        }
        summary["baseline_qualification_passed"] = summary["checks"][
            "all_matched_gmu093_baselines_passed"
        ]
        summary["diagnostic_passed"] = summary["checks"][
            "all_r26_gmu090_diagnostics_passed"
        ]
        summary["passed"] = summary["baseline_qualification_passed"]
        runtime.save_json("cache-native-canary-summary.json", summary)
        self.gate(
            "cache:native:matched-gmu093-baseline",
            summary["baseline_qualification_passed"],
            {
                "receipt": str(
                    runtime.ROOT / "cache-native-canary-summary.json"
                ),
                "checks": summary["checks"],
            },
        )
        self.gate(
            "cache:native:r26-gmu090-diagnostic",
            summary["diagnostic_passed"],
            {
                "receipt": str(
                    runtime.ROOT / "cache-native-canary-summary.json"
                ),
                "cannot_override_baseline": True,
            },
        )
        self.cells["native-canaries"] = summary
        return summary

    def prefill_overhead(self) -> None:
        shared_host = self.owned_l2("prefill-overhead")
        off_host = self.owned_l2("prefill-overhead-off")
        namespace = f"r26-cache-{self.run_id}-prefill-overhead"
        arms = [
            ("l2-off", off_host, "0", "l2-off"),
            ("fresh-l2", shared_host, "1", "l2-on"),
            ("existing-l2", shared_host, "1", "l2-on"),
        ]
        results: dict[str, Any] = {}
        for arm, host_dir, enabled, expected_mode in arms:
            pre_boot_inventory = self.inventory(host_dir)
            label = f"cache-prefill-overhead-{arm}"
            env = {
                "LMCACHE_L2_HOST_DIR": str(host_dir),
                "LMCACHE_L2_MAX_CAPACITY_GB": str(NORMAL_L2_CAP_GB),
                "LMCACHE_INSTANCE_ID": namespace,
                "LMCACHE_SHM_NAME": namespace,
                "LMCACHE_L2_ENABLED": enabled,
            }
            arm_result: dict[str, Any] = {"attempted": True, "label": label}
            if not self.boot(
                label,
                image=runtime.IMAGE,
                cache="lmcache",
                kv="nvfp4_ds_mla",
                spec="mtp0",
                extra_env=env,
            ):
                self.gate(f"cache:prefill-overhead:{arm}", False, "boot failed")
                self.stop(label)
                results[arm] = arm_result
                continue
            try:
                _, before, before_path = self.config_snapshot(
                    f"{label}-config-before",
                    image=runtime.IMAGE,
                    mode=expected_mode,
                    l2_host=host_dir,
                )
                code, data, output = self.run_probe(
                    "prefill",
                    label,
                    [
                        "--port",
                        str(runtime.PORT),
                        "--model",
                        runtime.MODEL_NAME,
                        "--suite-identity",
                        f"{self.run_id}-prefill-{arm}",
                        "--label",
                        arm,
                        "--target-tokens",
                        str(PREFILL_TOKENS),
                        "--runs",
                        "8",
                        "--deadline",
                        "900",
                    ],
                    timeout=4200,
                )
                time.sleep(STORE_DRAIN_SECONDS if enabled == "1" else 1)
                _, after, after_path = self.config_snapshot(
                    f"{label}-config-after",
                    image=runtime.IMAGE,
                    mode=expected_mode,
                    l2_host=host_dir,
                )
                l2_metrics_before = self.metrics_matching(
                    before, r"l2.*(store|load|prefetch).*(completed|hit|chunks|objects)"
                )
                l2_metrics_after = self.metrics_matching(
                    after, r"l2.*(store|load|prefetch).*(completed|hit|chunks|objects)"
                )
                l2_metric_names = set(l2_metrics_before) | set(l2_metrics_after)
                l2_metric_delta = {
                    name: l2_metrics_after.get(name, 0.0)
                    - l2_metrics_before.get(name, 0.0)
                    for name in sorted(l2_metric_names)
                }
                if enabled == "1":
                    effective_metric_state = any(
                        value > 0
                        for name, value in l2_metric_delta.items()
                        if "store" in name.lower()
                        and "completed" in name.lower()
                    )
                else:
                    effective_metric_state = not any(
                        value > 0 for value in l2_metric_delta.values()
                    )
                pre_inventory = before.get("l2", {}).get("host_inventory", {})
                if arm == "fresh-l2":
                    namespace_state_ok = (
                        pre_boot_inventory["files"] == 0
                        and pre_boot_inventory["bytes"] == 0
                    )
                elif arm == "existing-l2":
                    namespace_state_ok = (
                        pre_boot_inventory["files"] > 0
                        and pre_boot_inventory["bytes"] > 0
                    )
                else:
                    namespace_state_ok = before.get("l2", {}).get("actual_enabled") is False
                passed = (
                    code == 0
                    and data.get("passed") is True
                    and before.get("passed") is True
                    and after.get("passed") is True
                    and effective_metric_state
                    and namespace_state_ok
                )
                arm_result.update(
                    {
                        "receipt": str(output),
                        "config_before": str(before_path),
                        "config_after": str(after_path),
                        "summary": data.get("summary"),
                        "probe_checks": data.get("checks"),
                        "namespace_state_ok": namespace_state_ok,
                        "l2_metric_delta": l2_metric_delta,
                        "effective_metric_state": effective_metric_state,
                        "pre_boot_inventory": pre_boot_inventory,
                        "pre_inventory": pre_inventory,
                        "post_inventory": after.get("l2", {}).get("host_inventory"),
                    }
                )
                self.gate(f"cache:prefill-overhead:{arm}", passed, arm_result)
            except Exception as error:
                arm_result["error"] = f"{type(error).__name__}: {error}"
                self.gate(f"cache:prefill-overhead:{arm}:exception", False, arm_result["error"])
            finally:
                runtime.capture(label)
                self.stop(label)
                results[arm] = arm_result
        summaries = {
            arm: row.get("summary") for arm, row in results.items() if isinstance(row, dict)
        }
        comparison_complete = all(
            isinstance(summaries.get(arm), dict)
            and summaries[arm].get("steady_median_tokens_per_second")
            for arm in ("l2-off", "fresh-l2", "existing-l2")
        )
        steady = {
            arm: (
                float(summary["steady_median_tokens_per_second"])
                if isinstance(summary, dict)
                and isinstance(
                    summary.get("steady_median_tokens_per_second"), (int, float)
                )
                else None
            )
            for arm, summary in summaries.items()
        }
        off_speed = steady.get("l2-off")
        overhead_percent = {
            arm: (
                (1.0 - float(speed) / float(off_speed)) * 100.0
                if isinstance(off_speed, (int, float))
                and off_speed > 0
                and isinstance(speed, (int, float))
                else None
            )
            for arm, speed in steady.items()
            if arm != "l2-off"
        }
        comparison = {
            "arms": summaries,
            "measurement": "unique cold 33,000-token prompts; eight serial runs per arm",
            "steady_median_tokens_per_second": steady,
            "throughput_overhead_vs_effective_l2_off_percent": overhead_percent,
            "same_request_settings": True,
            "prompt_identity": (
                "Each arm uses the same exact token target and sampling settings, "
                "but deliberately different first-block identities and salts so all "
                "measured prefills are cold."
            ),
            "first_request_reported_separately": True,
            "interpretation_limit": (
                "Observed on the field host's NVMe filesystem without artificial throttling. "
                "This separates effective generated L2 configuration, not a SATA emulation."
            ),
        }
        runtime.save_json("cache-prefill-overhead-summary.json", comparison)
        self.gate(
            "cache:prefill-overhead:l2-off-vs-fresh-vs-existing",
            comparison_complete,
            comparison,
        )
        self.cells["prefill-overhead"] = results
        self.cleanup_owned_l2(shared_host, "cleanup-prefill-overhead")
        self.cleanup_owned_l2(off_host, "cleanup-prefill-overhead-off")

    def parse_connector_timers(self, label: str) -> dict[str, Any]:
        code = runtime.run(
            ["docker", "logs", runtime.NAME],
            label=f"{label}-connector-log",
            timeout=120,
        )
        path = runtime.ROOT / f"{label}-connector-log.log"
        text = path.read_text(errors="replace") if path.exists() else ""
        matches = []
        for line_number, line in enumerate(text.splitlines(), 1):
            match = CONNECTOR_TIMER_RE.search(line)
            if match:
                matches.append(
                    {
                        "line": line_number,
                        "retrieved_tokens": int(match.group("tokens").replace(",", "")),
                        "connector_reported_seconds": float(match.group("seconds")),
                        "raw": line,
                    }
                )
        report = {
            "returncode": code,
            "raw_log": str(path),
            "matches": matches,
            "scope_note": (
                "These are connector-emitted 'Retrieved N tokens in S seconds' segments. "
                "They are kept separate from complete API wall time."
            ),
        }
        runtime.save_json(f"{label}-connector-timers.json", report)
        return report

    def alternating_release_control(self) -> None:
        host_dir = self.owned_l2("r25-r26-alternating")
        namespace = f"r26-cache-{self.run_id}-alternating"
        env = {
            "LMCACHE_L2_HOST_DIR": str(host_dir),
            "LMCACHE_L2_MAX_CAPACITY_GB": str(NORMAL_L2_CAP_GB),
            "LMCACHE_INSTANCE_ID": namespace,
            "LMCACHE_SHM_NAME": namespace,
            "LMCACHE_L2_ENABLED": "1",
        }
        prompt = runtime.ROOT / "cache-r25-r26-alternating.prompt.txt"
        prompt_meta: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        sequence = [
            ("R25", runtime.R25_IMAGE, "seed-cold"),
            ("R26", runtime.IMAGE, "warm-1"),
            ("R25", runtime.R25_IMAGE, "warm-1"),
            ("R26", runtime.IMAGE, "warm-2"),
            ("R25", runtime.R25_IMAGE, "warm-2"),
            ("R26", runtime.IMAGE, "warm-3"),
            ("R25", runtime.R25_IMAGE, "warm-3"),
        ]
        identity = f"{self.run_id}-alternating-exact-1m"
        salt = f"{self.run_id}-alternating-exact-1m-salt"
        for index, (release, image, phase) in enumerate(sequence):
            label = f"cache-alternating-{index:02d}-{release.lower()}-{phase}"
            row: dict[str, Any] = {
                "index": index,
                "release": release,
                "phase": phase,
                "image": image,
                "attempted": True,
            }
            if not self.boot(
                label,
                image=image,
                cache="lmcache",
                kv="nvfp4_ds_mla",
                spec="mtp0",
                extra_env=env,
            ):
                row["booted"] = False
                rows.append(row)
                self.gate(
                    f"cache:alternating:{index:02d}:{release}:{phase}",
                    False,
                    "boot failed",
                )
                self.stop(label)
                continue
            row["booted"] = True
            try:
                _, config, config_path = self.config_snapshot(
                    f"{label}-config",
                    image=image,
                    mode="l2-on",
                    l2_host=host_dir,
                )
                row["config"] = str(config_path)
                if not prompt.exists():
                    prepared, prompt, prompt_meta = self.prepare_prompt(
                        "cache-r25-r26-alternating",
                        target_tokens=TARGET_1M,
                        identity=identity,
                        kind="period",
                    )
                    if not prepared:
                        row["error"] = "prompt preparation failed"
                        self.gate(
                            f"cache:alternating:{index:02d}:{release}:{phase}",
                            False,
                            row,
                        )
                        continue
                code, request, request_path = self.request(
                    label,
                    prompt=prompt,
                    cache_salt=salt,
                    max_tokens=1,
                    deadline=1800,
                    expected_tokens=TARGET_1M,
                    ignore_eos=True,
                )
                time.sleep(STORE_DRAIN_SECONDS)
                timers = self.parse_connector_timers(label)
                row.update(
                    {
                        "returncode": code,
                        "request": str(request_path),
                        "api_wall_seconds": request.get("wall_seconds"),
                        "request_body_sha256": request.get("request_body_sha256"),
                        "visible_output_sha256": self.request_summary(request).get(
                            "visible_output_sha256"
                        ),
                        "full_output_sha256_diagnostic": self.request_summary(
                            request
                        ).get("full_output_sha256"),
                        "visible_final_status": self.request_summary(request).get(
                            "visible_final_status"
                        ),
                        "completion_tokens": self.request_summary(request).get(
                            "completion_tokens"
                        ),
                        "lmcache_hit_tokens": self.lmcache_hits(request),
                        "connector_timers": timers,
                        "config_effective": config.get("checks", {}).get("effective_mode"),
                        "config_passed": config.get("passed"),
                    }
                )
                cell_ok = (
                    request.get("passed") is True
                    and config.get("passed") is True
                    and (index == 0 or isinstance(row["lmcache_hit_tokens"], int))
                )
                self.gate(
                    f"cache:alternating:{index:02d}:{release}:{phase}",
                    cell_ok,
                    row,
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                self.gate(
                    f"cache:alternating:{index:02d}:{release}:{phase}:exception",
                    False,
                    row["error"],
                )
            finally:
                runtime.capture(label)
                self.stop(label)
                rows.append(row)
        warm_rows = [row for row in rows if row.get("phase") != "seed-cold"]
        release_walls: dict[str, list[float]] = {"R25": [], "R26": []}
        release_connector: dict[str, list[float]] = {"R25": [], "R26": []}
        for row in warm_rows:
            wall = row.get("api_wall_seconds")
            if isinstance(wall, (int, float)):
                release_walls[row["release"]].append(float(wall))
            timers = row.get("connector_timers", {}).get("matches", [])
            qualifying = [
                float(item["connector_reported_seconds"])
                for item in timers
                if int(item.get("retrieved_tokens", 0)) >= TARGET_1M - 4096
            ]
            if qualifying:
                release_connector[row["release"]].append(max(qualifying))
        request_hashes = {
            row.get("request_body_sha256") for row in rows if row.get("request_body_sha256")
        }
        visible_output_hashes = [
            row.get("visible_output_sha256") for row in rows
        ]
        full_output_hashes = [
            row.get("full_output_sha256_diagnostic") for row in rows
        ]
        warm_hits = [row.get("lmcache_hit_tokens") for row in warm_rows]
        summary = {
            "schema": PHASE_SCHEMA,
            "prompt": prompt_meta,
            "sequence": rows,
            "settings": {
                "tp": 4,
                "dcp": 4,
                "spec": "mtp0",
                "kv": "nvfp4_ds_mla",
                "max_tokens": 1,
                "temperature": 0.0,
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "cache_salt": salt,
            },
            "api_wall_seconds": release_walls,
            "api_wall_median_seconds": {
                release: statistics.median(values) if values else None
                for release, values in release_walls.items()
            },
            "connector_reported_seconds": release_connector,
            "connector_reported_median_seconds": {
                release: statistics.median(values) if values else None
                for release, values in release_connector.items()
            },
            "measurement_scope": (
                "Alternating one-token period requests compare API and connector "
                "latency only; they do not test recall or model-answer quality."
            ),
            "output_identity_diagnostics": {
                "visible_output_sha256": visible_output_hashes,
                "visible_output_equal_diagnostic": (
                    visible_output_hashes[0] is not None
                    and len(set(visible_output_hashes)) == 1
                ),
                "full_output_sha256": full_output_hashes,
                "full_output_equal_diagnostic": (
                    full_output_hashes[0] is not None
                    and len(set(full_output_hashes)) == 1
                ),
                "required_for_latency_result": False,
            },
            "checks": {
                "seed_completed": bool(rows and rows[0].get("api_wall_seconds")),
                "three_warm_samples_each": all(
                    len(release_walls[release]) == 3 for release in ("R25", "R26")
                ),
                "alternating_order": [row.get("release") for row in warm_rows]
                == ["R26", "R25", "R26", "R25", "R26", "R25"],
                "same_request_body": len(request_hashes) == 1,
                "exactly_one_completion_token_each": len(rows) == len(sequence)
                and all(row.get("completion_tokens") == 1 for row in rows),
                "all_warm_external_hits": len(warm_hits) == 6
                and all(isinstance(value, int) and value > 0 for value in warm_hits),
                "connector_timer_samples_each": all(
                    len(release_connector[release]) == 3 for release in ("R25", "R26")
                ),
            },
            "timer_scope_note": (
                "API wall medians and connector-emitted retrieve segments are separate series. "
                "No connector segment is relabelled as end-to-end restore latency."
            ),
        }
        summary["passed"] = all(summary["checks"].values())
        runtime.save_json("cache-r25-r26-alternating-summary.json", summary)
        self.gate(
            "cache:r25-r26:alternating-exact-one-token-shared-l2",
            summary["passed"],
            {"summary": str(runtime.ROOT / "cache-r25-r26-alternating-summary.json"), "checks": summary["checks"]},
        )
        self.cells["alternating-r25-r26"] = summary
        self.cleanup_owned_l2(host_dir, "cleanup-r25-r26-alternating")

    @staticmethod
    def event_args(port: int, replay_port: int, topic: str) -> list[str]:
        config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://*:{port}",
            "replay_endpoint": f"tcp://*:{replay_port}",
            "buffer_steps": 10000,
            "hwm": 100000,
            "max_queue_size": 100000,
            "topic": topic,
        }
        return ["--kv-events-config", json.dumps(config, separators=(",", ":"))]

    def focused_arm(self, arm: str, image: str) -> dict[str, Any]:
        label = f"cache-focused-{arm}"
        host_dir = self.owned_l2(f"focused-{arm}")
        namespace = f"r26-cache-{self.run_id}-focused-{arm}"
        event_port = runtime.PORT + 1100
        replay_port = runtime.PORT + 1101
        topic = f"r26-cache-{self.run_id}-{arm}"
        env = {
            "LMCACHE_L2_HOST_DIR": str(host_dir),
            "LMCACHE_L2_MAX_CAPACITY_GB": str(EVICTION_L2_GB),
            "LMCACHE_L1_SIZE_GB": str(EVICTION_L1_GB),
            "LMCACHE_L1_INIT_SIZE_GB": str(EVICTION_L1_GB),
            "LMCACHE_INSTANCE_ID": namespace,
            "LMCACHE_SHM_NAME": namespace,
            "LMCACHE_L2_ENABLED": "1",
        }
        result: dict[str, Any] = {
            "schema": PHASE_SCHEMA,
            "arm": arm,
            "image": image,
            "attempted": True,
            "l2_host": str(host_dir),
            "method": {
                "target_tokens": FOCUSED_TOKENS,
                "semantic_max_tokens": SEMANTIC_MAX_TOKENS,
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "visible_final_only_oracle": True,
                "method_changed_from_original": (
                    "compact 128-row unique preamble and 512-token semantic "
                    "budget replace the failed 18.5K-prefix/64-token method"
                ),
                "one_token_churn_scope": "cache pressure and latency only",
            },
        }
        if not self.boot(
            label,
            image=image,
            cache="lmcache",
            kv="nvfp4_ds_mla",
            spec="mtp3",
            extra_env=env,
            extra_args=self.event_args(event_port, replay_port, topic),
        ):
            result["booted"] = False
            self.gate(f"cache:focused:{arm}:attempted", False, "boot failed")
            self.stop(label)
            self.cleanup_owned_l2(host_dir, f"cleanup-focused-{arm}")
            return result
        result["booted"] = True
        try:
            _, config, config_path = self.config_snapshot(
                f"{label}-config-before",
                image=image,
                mode="l2-on",
                l2_host=host_dir,
                max_l2_gb=EVICTION_L2_GB,
                expected_l1_gb=EVICTION_L1_GB,
            )
            result["config_before"] = str(config_path)
            declaration_file = Path(
                "/home/josh/omp-workspace/drock-lmcache/"
                "release-review-20260905T154745Z/drock-temp.txt"
            )
            try:
                declaration_sha256 = hashlib.sha256(
                    declaration_file.read_bytes()
                ).hexdigest()
            except OSError:
                declaration_sha256 = None
            provenance = {
                "image_digest_observed": config.get("image_provenance", {}).get(
                    "expected_digest_observed"
                ),
                "source_lock_sha256": config.get("image_provenance", {}).get(
                    "source_lock_sha256"
                ),
                "source_lock_scope_note": config.get("image_provenance", {}).get(
                    "source_lock_scope_note"
                ),
                "declared_overlay_revision": (
                    "7db6a2d2f5680513ae1a396ff61169c4cacf8a95" if image == runtime.OVERLAY_IMAGE else None
                ),
                "declared_overlay_patches": (
                    ["#561", "remaining #574", "#643", "#645", "#648", "#599"]
                    if image == runtime.OVERLAY_IMAGE
                    else []
                ),
                "declaration_source": (
                    "/home/josh/omp-workspace/drock-lmcache/release-review-20260905T154745Z/drock-temp.txt:258-271"
                    if image == runtime.OVERLAY_IMAGE
                    else None
                ),
                "declaration_file_sha256": (
                    declaration_sha256 if image == runtime.OVERLAY_IMAGE else None
                ),
            }
            self.gate(
                f"cache:focused:{arm}:generated-cache-config",
                config.get("passed") is True,
                {"receipt": str(config_path), "checks": config.get("checks")},
            )
            runtime.save_json(f"{label}-provenance.json", provenance)
            self.gate(
                f"cache:focused:{arm}:image-provenance",
                provenance["image_digest_observed"] is True
                and (
                    image != runtime.OVERLAY_IMAGE
                    or provenance["declaration_file_sha256"] is not None
                ),
                provenance,
            )
            self.gate(
                f"cache:focused:{arm}:cpu-only-sidecar",
                config.get("sidecar", {}).get("cpu_only") is True,
                config.get("sidecar"),
            )
            generated_event_config = config.get("generated_kv_events_config")
            event_config_ok = (
                isinstance(generated_event_config, dict)
                and generated_event_config.get("enable_kv_cache_events") is True
                and generated_event_config.get("publisher") == "zmq"
                and generated_event_config.get("endpoint") == f"tcp://*:{event_port}"
                and generated_event_config.get("replay_endpoint")
                == f"tcp://*:{replay_port}"
                and generated_event_config.get("topic") == topic
            )
            self.gate(
                f"cache:focused:{arm}:generated-kv-event-config",
                event_config_ok,
                generated_event_config,
            )

            restart_identity = f"{self.run_id}-focused-matched-retention-16384"
            prepared, restart_prompt, restart_meta = self.prepare_prompt(
                f"{label}-retention",
                target_tokens=FOCUSED_TOKENS,
                identity=restart_identity,
                kind="reference",
            )
            restart_evidence: dict[str, Any] = {
                "prompt": restart_meta,
                "method": result["method"],
            }
            recurrent_cache_groups_observed = False
            if prepared:
                salt = f"{self.run_id}-focused-matched-retention-salt"
                reference = restart_meta.get("reference_code")
                _, first, first_path = self.request(
                    f"{label}-retention-cold",
                    prompt=restart_prompt,
                    cache_salt=salt,
                    max_tokens=SEMANTIC_MAX_TOKENS,
                    deadline=900,
                    expected_tokens=FOCUSED_TOKENS,
                    expected_reference=reference,
                )
                time.sleep(STORE_DRAIN_SECONDS)
                reset_code, reset_data, reset_path = self.reset_local(
                    f"{label}-retention-reset-local"
                )
                reset_ok = reset_code == 0 and reset_data.get("passed") is True
                _, warm, warm_path = self.request(
                    f"{label}-retention-warm",
                    prompt=restart_prompt,
                    cache_salt=salt,
                    max_tokens=SEMANTIC_MAX_TOKENS,
                    deadline=900,
                    expected_tokens=FOCUSED_TOKENS,
                    expected_reference=reference,
                )
                time.sleep(STORE_DRAIN_SECONDS)
                restarted = self.restart(f"{label}-retention")
                restart_data: dict[str, Any] = {}
                restart_path = runtime.ROOT / f"{label}-retention-restart.json"
                if restarted:
                    _, restart_data, restart_path = self.request(
                        f"{label}-retention-restart",
                        prompt=restart_prompt,
                        cache_salt=salt,
                        max_tokens=SEMANTIC_MAX_TOKENS,
                        deadline=900,
                        expected_tokens=FOCUSED_TOKENS,
                        expected_reference=reference,
                    )
                retention_timers = self.parse_connector_timers(
                    f"{label}-retention-restart"
                )
                connector_log_path = Path(retention_timers["raw_log"])
                connector_log_text = (
                    connector_log_path.read_text(errors="replace")
                    if connector_log_path.exists()
                    else ""
                )
                recurrent_cache_groups_observed = all(
                    re.search(pattern, connector_log_text, re.IGNORECASE)
                    is not None
                    for pattern in (
                        r"Detected recurrent KV cache groups",
                        r"KV cache group edits applied",
                    )
                )
                verdict = self.lifecycle_verdict(
                    FOCUSED_TOKENS, "reference", first, warm, restart_data
                )
                verdict["checks"]["local_prefix_reset"] = reset_ok
                restart_hit_tokens = self.lmcache_hits(restart_data)
                verdict["checks"]["restart_retention_depth"] = (
                    isinstance(restart_hit_tokens, int)
                    and restart_hit_tokens >= FOCUSED_TOKENS - 4096
                )
                verdict["checks"][
                    "recurrent_cache_group_protocol_observed"
                ] = recurrent_cache_groups_observed
                restart_evidence.update(
                    {
                        "cold": str(first_path),
                        "warm": str(warm_path),
                        "restart": str(restart_path),
                        "local_reset": str(reset_path),
                        "verdict": verdict,
                        "restart_request_observed": bool(restart_data),
                        "restart_lmcache_hit_tokens": restart_hit_tokens,
                        "connector_timers": retention_timers,
                        "recurrent_cache_group_protocol_observed": recurrent_cache_groups_observed,
                        "protocol_evidence_scope": (
                            "Startup logs show the pinned runtime's actual recurrent "
                            "cache-group protocol. They do not by themselves prove a "
                            "byte-identical memory handoff."
                        ),
                    }
                )
            else:
                verdict = {"checks": {"prepared": False}}
            result["retention_restart"] = restart_evidence
            retention_pass = all(verdict.get("checks", {}).values())
            if image == runtime.OVERLAY_IMAGE:
                self.gate(
                    "cache:drock-overlay:#574:exact-recurrent-retention-restart",
                    retention_pass,
                    restart_evidence,
                )
            else:
                self.gate(
                    "cache:stock-r26:#574:control-completed",
                    restart_evidence.get("restart_request_observed") is True,
                    {"semantic_pass": retention_pass, "evidence": restart_evidence},
                )

            code, focused, focused_path = self.run_probe(
                "hitmiss",
                label,
                [
                    "--port",
                    str(runtime.PORT),
                    "--model",
                    runtime.MODEL_NAME,
                    "--suite-identity",
                    f"{self.run_id}-focused-matched",
                    "--target-tokens",
                    str(FOCUSED_TOKENS),
                    "--churn-docs",
                    "40",
                    "--deadline",
                    "900",
                    "--store-drain-seconds",
                    "8",
                    "--eviction-settle-seconds",
                    "12",
                    "--event-settle-seconds",
                    "0.5",
                    "--event-endpoint",
                    f"tcp://127.0.0.1:{event_port}",
                    "--replay-endpoint",
                    f"tcp://127.0.0.1:{replay_port}",
                    "--event-topic",
                    topic,
                ],
                timeout=12_000,
            )
            _, after, after_path = self.config_snapshot(
                f"{label}-config-after",
                image=image,
                mode="l2-on",
                l2_host=host_dir,
                max_l2_gb=EVICTION_L2_GB,
                expected_l1_gb=EVICTION_L1_GB,
            )
            runtime.capture(label)
            result.update(
                {
                    "focused_returncode": code,
                    "focused_receipt": str(focused_path),
                    "focused_checks": focused.get("checks"),
                    "events": focused.get("events", {}).get("summary"),
                    "config_after": str(after_path),
                    "prompt_fingerprints": {
                        "serial": focused.get("scenarios", {})
                        .get("serial_shared_prefix", {})
                        .get("prompt", {})
                        .get("prompt_sha256"),
                        "divergent": focused.get("scenarios", {})
                        .get("divergent_continuation", {})
                        .get("shared_base_sha256"),
                        "concurrent": focused.get("scenarios", {})
                        .get("concurrent_shared_prefix", {})
                        .get("shared_base", {})
                        .get("prompt_sha256"),
                        "eviction": focused.get("scenarios", {})
                        .get("eviction_then_reuse", {})
                        .get("target_prompt", {})
                        .get("prompt_sha256"),
                    },
                }
            )
            checks = focused.get("checks") if isinstance(focused.get("checks"), dict) else {}
            scenario_names = set(focused.get("scenarios", {}))
            observation_complete = scenario_names == {
                "serial_shared_prefix",
                "divergent_continuation",
                "concurrent_shared_prefix",
                "eviction_then_reuse",
            }
            if image == runtime.OVERLAY_IMAGE:
                replay_checks = all(
                    checks.get(key) is True
                    for key in (
                        "all_scenarios_attempted",
                        "all_local_resets_succeeded",
                        "serial_cold_miss",
                        "serial_warm_hit",
                        "serial_visible_answers_complete_and_correct",
                        "serial_visible_output_byte_equal",
                        "divergent_continuation_partial_hit",
                        "divergent_visible_answers_complete_and_correct",
                        "concurrent_shared_prefix_hits",
                        "eviction_churn_requests_succeeded",
                        "concurrent_visible_answers_complete_and_correct",
                        "eviction_warm_control_hit",
                        "eviction_churn_complete",
                        "eviction_then_reuse_miss",
                        "eviction_visible_answers_complete_and_correct",
                        "eviction_recompute_visible_output_byte_equal",
                    )
                )
                event_checks = all(
                    checks.get(key) is True
                    for key in (
                        "event_live_and_replay",
                        "event_replay_payload_identity",
                        "truthful_sparse_event_shape",
                        "cached_prefix_replay_event",
                        "sparse_skipped_context_event",
                    )
                )
                self.gate(
                    "cache:drock-overlay:#643:sparse-retention-replay",
                    replay_checks,
                    {"receipt": str(focused_path), "checks": checks},
                )
                self.gate(
                    "cache:drock-overlay:#645:truthful-cache-events-and-replay",
                    event_checks,
                    {"receipt": str(focused_path), "events": result["events"]},
                )
            else:
                self.gate(
                    "cache:stock-r26:#643-#645:control-completed",
                    observation_complete,
                    {
                        "receipt": str(focused_path),
                        "semantic_checks": checks,
                        "events": result["events"],
                    },
                )
            eviction_metrics = {
                **self.metrics_matching(
                    after,
                    r"(?:l1|l2)_evicted.*(?:objects|chunks)|eviction.*triggered",
                ),
                **self.metrics_matching(after, r"l2.*deleted"),
            }
            log_path = runtime.ROOT / f"{label}.docker.log"
            log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            eviction_observed = any(value > 0 for value in eviction_metrics.values()) or bool(
                re.search(
                    r"above watermark|triggering eviction|evicted [1-9]|"
                    r"eviction_loop_triggered",
                    log_text,
                    re.IGNORECASE,
                )
            )
            result["eviction_metrics"] = eviction_metrics
            result["eviction_observed"] = eviction_observed
            self.gate(
                f"cache:focused:{arm}:eviction-observed",
                eviction_observed,
                {"metrics": eviction_metrics, "docker_log": str(log_path)},
            )
            result["observation_complete"] = observation_complete
            return result
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            self.gate(f"cache:focused:{arm}:exception", False, result["error"])
            return result
        finally:
            self.stop(label)
            self.cleanup_owned_l2(host_dir, f"cleanup-focused-{arm}")

    def focused_overlay_comparison(self) -> None:
        stock = self.focused_arm("stock-r26", runtime.IMAGE)
        overlay = self.focused_arm("drock-overlay", runtime.OVERLAY_IMAGE)
        comparison = {
            "schema": PHASE_SCHEMA,
            "stock-r26": stock,
            "drock-overlay": overlay,
            "comparison_contract": {
                "matched_prompt_identity": f"{self.run_id}-focused-matched",
                "tp": 4,
                "dcp": 4,
                "spec": "mtp3",
                "kv": "nvfp4_ds_mla",
                "cache": "lmcache",
                "l1_gb": EVICTION_L1_GB,
                "l2_gb": EVICTION_L2_GB,
                "focused_tokens": FOCUSED_TOKENS,
                "unique_preamble_rows": UNIQUE_PREAMBLE_ROWS,
                "minimum_unique_preamble_tokens": 4096,
                "semantic_max_tokens": SEMANTIC_MAX_TOKENS,
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "semantic_oracle": "naturally completed user-visible final only",
                "full_output_hash_scope": "diagnostic only",
                "scenarios": [
                    "serial shared-prefix",
                    "divergent continuation",
                    "concurrent shared-prefix",
                    "eviction then reuse",
                    "event live stream and replay",
                    "16,384-token restart retention",
                ],
            },
            "interpretation": (
                "Stock is the official R26 control. The D-Rock digest is reported as a "
                "separate candidate; a shared inherited source.lock is not treated as proof "
                "of overlay package identity or causality."
            ),
        }
        stock_retention_prompt = (
            stock.get("retention_restart", {}).get("prompt", {}).get("prompt_sha256")
        )
        overlay_retention_prompt = (
            overlay.get("retention_restart", {}).get("prompt", {}).get("prompt_sha256")
        )
        stock_fingerprints = stock.get("prompt_fingerprints")
        overlay_fingerprints = overlay.get("prompt_fingerprints")
        matched_prompts = (
            stock_retention_prompt is not None
            and stock_retention_prompt == overlay_retention_prompt
            and isinstance(stock_fingerprints, dict)
            and all(stock_fingerprints.values())
            and stock_fingerprints == overlay_fingerprints
        )
        comparison["matched_prompt_fingerprints"] = {
            "passed": matched_prompts,
            "stock_retention": stock_retention_prompt,
            "overlay_retention": overlay_retention_prompt,
            "stock_scenarios": stock_fingerprints,
            "overlay_scenarios": overlay_fingerprints,
        }
        runtime.save_json("cache-stock-vs-overlay-summary.json", comparison)
        both_attempted = stock.get("attempted") is True and overlay.get("attempted") is True
        self.gate(
            "cache:stock-vs-drock-overlay:#574-#643-#645:both-attempted",
            both_attempted,
            {
                "receipt": str(runtime.ROOT / "cache-stock-vs-overlay-summary.json"),
                "stock_complete": stock.get("observation_complete"),
                "overlay_complete": overlay.get("observation_complete"),
            },
        )
        self.gate(
            "cache:stock-vs-drock-overlay:matched-prompt-settings",
            matched_prompts,
            comparison["matched_prompt_fingerprints"],
        )
        self.cells["focused-stock-vs-overlay"] = comparison

    def source_and_storage_evidence(self) -> None:
        runtime.L2_HOST_ROOT.mkdir(parents=True, exist_ok=True)
        storage_code, storage, storage_path = self.run_probe(
            "storage",
            "cache-storage-filesystem-evidence",
            [
                "--paths",
                "/home",
                "/mnt/2king",
                str(runtime.L2_HOST_ROOT),
            ],
            timeout=120,
        )
        self.gate(
            "cache:storage:observable-nvme-scope",
            storage_code == 0 and storage.get("passed") is True,
            {
                "receipt": str(storage_path),
                "same_filesystem": storage.get("same_filesystem_device"),
                "non_rotational": storage.get("non_rotational"),
                "nvme_named": storage.get("nvme_named_in_observed_device_metadata"),
                "throttling_performed": storage.get("throttling_performed"),
            },
        )
        source_code, source, source_path = self.run_probe(
            "source-audit",
            "cache-byte-transfer-source-audit",
            ["--source-root", str(SOURCE_ROOT)],
            timeout=180,
        )
        self.gate(
            "cache:byte-level-external-transfer:source-scope-audited",
            source_code == 0 and source.get("passed") is True,
            {
                "receipt": str(source_path),
                "full_external_available": source.get(
                    "full_byte_level_external_kv_transfer_test_available"
                ),
                "invoked": source.get("invoked"),
                "supported_shapes": source.get("supported_shapes_observed"),
                "limitation": source.get("limitation"),
            },
        )
        self.cells["storage-evidence"] = storage
        self.cells["source-audit"] = source

    def finish(self) -> None:
        all_named_attempted = all(
            key in self.cells
            for key in (
                "official-fp8-lifecycles",
                "official-packed-nvfp4-lifecycles",
                "official-packed-nvfp4-needles",
                "eviction",
                "native-canaries",
                "prefill-overhead",
                "alternating-r25-r26",
                "focused-stock-vs-overlay",
            )
        )
        self.gate(
            "cache:phase:all-named-modes-attempted",
            all_named_attempted,
            {"cells": sorted(self.cells)},
        )
        failed = [row for row in self.checks if not row["passed"]]
        requirements = {
            "runtime": "scripts/r26/runtime.py with loopback-only supervised serial boot",
            "images": {
                "official_r26": runtime.IMAGE,
                "matched_r25": runtime.R25_IMAGE,
                "drock_overlay": runtime.OVERLAY_IMAGE,
            },
            "model": str(runtime.MODEL),
            "draft_model": str(runtime.DRAFT),
            "container": runtime.NAME,
            "port": runtime.PORT,
            "private_shm": "128 GiB for LMCache and native offload",
            "l2_root": str(runtime.L2_HOST_ROOT),
            "l2_policy": "cache-phase-owned child paths; normal cap 160 GiB; focused eviction cap 2 GiB; cleanup after restore receipts",
            "host_python": "Python 3 with pyzmq and msgpack for KV-event live/replay capture",
            "tools": ["docker", "nvidia-smi"],
        }
        upstream_blockers = [
            {
                "name": "full byte-level real GLM external transfer",
                "status": "not available in the inspected shipped-source probes",
                "detail": "Available bit-exact probes stop at CPU fallback, GPU copy planner, or mocked SHM; the real-model external test compares generated text.",
            },
            {
                "name": "overlay package identity from source.lock",
                "status": "source.lock is inherited base evidence only",
                "detail": "Use the immutable overlay digest plus declared revision 7db6a2d2f5680513ae1a396ff61169c4cacf8a95; do not infer the Python overlay from the base lock.",
            },
            {
                "name": "SATA latency reproduction",
                "status": "out of scope on this hardware",
                "detail": "The real paths are on the same NVMe filesystem and no disk is throttled.",
            },
        ]
        summary = {
            "schema": PHASE_SCHEMA,
            "run_id": self.run_id,
            "passed": not failed,
            "checks_passed": len(self.checks) - len(failed),
            "checks_total": len(self.checks),
            "failed": failed,
            "checks": self.checks,
            "cells": self.cells,
            "runtime_requirements": requirements,
            "upstream_blockers": upstream_blockers,
            "limitations": self.limitations,
            "owned_l2_dirs": [str(path) for path in sorted(self._owned_dirs)],
        }
        runtime.save_json("cache-phase-summary.json", summary)
        runtime.save_json(
            "cache-phase-complete.json",
            {
                "schema": PHASE_SCHEMA,
                "run_id": self.run_id,
                "completed_at": time.time(),
                "summary": str(runtime.ROOT / "cache-phase-summary.json"),
                "failed_gate_count": len(failed),
            },
        )

    def run_cell(self, name: str, operation: Any) -> None:
        try:
            operation()
        except Exception as error:
            self.gate(
                f"cache:cell:{name}:unhandled-exception",
                False,
                {"error": f"{type(error).__name__}: {error}"},
            )

    def main(self) -> None:
        runtime.note(f"R26 CACHE PHASE START run_id={self.run_id}")
        try:
            for name, operation in (
                ("source-and-storage", self.source_and_storage_evidence),
                ("official-lifecycles", self.official_lifecycles),
                ("eviction", self.eviction_cell),
                ("native-canaries", self.native_matrix),
                ("prefill-overhead", self.prefill_overhead),
                ("alternating-r25-r26", self.alternating_release_control),
                ("focused-stock-overlay", self.focused_overlay_comparison),
            ):
                self.run_cell(name, operation)
        finally:
            self.stop("phase-final")
            for path in sorted(self._owned_dirs):
                if path.exists():
                    self.cleanup_owned_l2(path, f"cleanup-final-{path.name}")
            self.finish()
            runtime.note("R26 CACHE PHASE COMPLETE")


def main() -> None:
    CachePhase().main()


if __name__ == "__main__":
    main()
