#!/usr/bin/env python3
"""Write R/cancellation-observability-analysis.json from retained cancellation-stage receipts.

CPU-only, read-only over campaign dirs. Run: python3 build_cancellation_observability_analysis.py
"""
import hashlib
import json
from pathlib import Path

R = Path("/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909")
CAMP = R / "r30-cache-campaign-2"
OUT = R / "cancellation-observability-analysis.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_prom(p):
    import re
    out = {}
    for line in Path(p).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^(\S+?)(?:\{(.*)\})?\s+(\S+)$", line)
        if m:
            name, labels, val = m.groups()
            try:
                out[f"{name}{{{labels}}}" if labels else name] = float(val)
            except ValueError:
                pass
    return out


def arm_analysis(arm):
    d = json.loads((CAMP / arm / "receipt.json").read_text())
    canc = d["stages"]["cancellation"]
    bp = canc["metrics"]["before"]["http"]["response_body_path"]
    ap = canc["metrics"]["after"]["http"]["response_body_path"]
    rp = canc["metrics"]["after"]["http"]["response_body_path"]
    first_after = (CAMP / arm / "raw/cancellation-metrics-after.response.body")
    before, after = parse_prom(bp), parse_prom(first_after)
    changed = {k: [before.get(k), after.get(k)] for k in set(before) | set(after)
               if before.get(k) != after.get(k)}
    interesting = {k: v for k, v in changed.items() if not k.startswith(("http_", "python_gc", "process_cpu"))}
    ex = canc["exchange"]
    return {
        "arm": arm,
        "cancellation_request": {
            "prompt_tokens": canc["tokenization"]["count"],
            "max_tokens": ex["sampling"]["max_tokens"],
            "ignore_eos": ex["sampling"]["ignore_eos"],
            "sse_events_read": ex["sse_events_read"],
            "first_generated_data_event_observed": ex["first_generated_data_event_observed"],
            "done_observed": ex["done_observed"],
            "client_closed_early": ex["client_closed_early"],
            "elapsed_seconds_before_close": ex["elapsed_seconds_before_close"],
            "http_status": ex["status"],
        },
        "abort_series_values": {
            "request_success_total_abort_before": before.get('vllm:request_success_total{engine="0",finished_reason="abort",model_name="GLM-5.3-Flash-NVFP4"}'),
            "request_success_total_abort_first_after": after.get('vllm:request_success_total{engine="0",finished_reason="abort",model_name="GLM-5.3-Flash-NVFP4"}'),
            "request_success_total_abort_after_retry": parse_prom(rp).get('vllm:request_success_total{engine="0",finished_reason="abort",model_name="GLM-5.3-Flash-NVFP4"}'),
            "request_success_total_stop_before": before.get('vllm:request_success_total{engine="0",finished_reason="stop",model_name="GLM-5.3-Flash-NVFP4"}'),
            "request_success_total_stop_after": parse_prom(rp).get('vllm:request_success_total{engine="0",finished_reason="stop",model_name="GLM-5.3-Flash-NVFP4"}'),
        },
        "other_abort_cancel_preempt_series": sorted({k for k in before if any(t in k.lower() for t in ("abort", "cancel", "preempt"))}),
        "counters_that_moved_engine_side": {k: v for k, v in interesting.items() if k.startswith("vllm:")},
        "probe_recorded_delta": canc["metrics"]["delta"],
        "lease_monitor_samples": [
            {"at": row["at"], "retrieve_leases": (row.get("recurrent_checkpoints") or {}).get("retrieve_leases"),
             "store_leases": (row.get("recurrent_checkpoints") or {}).get("store_leases")}
            for row in canc["status_monitor"]
        ],
        "lmcache_status_cumulative_retrieve_counters": {
            "active_prefetch_jobs": "gauge only, 0 at settlement",
            "storage_manager.prefetch_controller.completed_results_count": "0 at settlement; the aligned external retrieve is not counted here for this path",
            "recurrent_checkpoints.retrieve_leases": "gauge only (instantaneous), no cumulative retrieve counter exposed on the 18085 status endpoint",
        },
    }


def main():
    stock = arm_analysis("l2-stock-dcp4-mtp3")
    r30 = arm_analysis("l2-r30-dcp1-mtp0")

    doc = {
        "schema": "r29-cancellation-observability/v1",
        "scope": "CPU-only analysis of retained cancellation-stage receipts from r30-cache-campaign-2 (stock R29 control and first R30 arm). No GPU work, no probe edits.",
        "question": "Why are cancellation.server_abort_observed and cancellation.live_retrieve_observed_under_pressure unavailable, and what actually moved on client disconnect?",
        "probe_consumption": {
            "server_abort_observed": "cache_lifecycle_probe.py:220-224 sums vllm:request_success_total{finished_reason='abort'} before/after and gates on monotonic_delta >= 1 (line 3278-3279); on 0 it re-polls /metrics once per second for 60 s (lines 3258-3267) before recording 'unavailable'.",
            "live_retrieve_observed_under_pressure": "cache_lifecycle_probe.py:3236-3254 polls the LMCache HTTP status endpoint (port 18085) in a loop that sleeps 0.05 s BETWEEN samples, but each sample is itself a blocking HTTP request; the gate (lines 3280-3284) requires any single sample to catch recurrent_checkpoints.retrieve_leases > 0, an instantaneous gauge. required=False.",
        },
        "what_moved_on_disconnect": {
            "stock_control": stock,
            "first_r30_arm": r30,
        },
        "findings": [
            "The request did NOT finish before the client abort: done_observed=false after only 2 SSE events (~0.11-0.14 s), request_success_total{stop} stayed flat at 8.0 across before/first-after/retry, and generation_tokens moved by only 2-4 tokens (the two streamed deltas) while prompt_tokens moved +32768 (prefill counted). No success counter of any kind recorded the cancelled request.",
            "The ONLY abort-labelled series in the entire /metrics body is vllm:request_success_total{finished_reason='abort'}, and it stayed 0.0 in both arms, at first-after and after the full 60 s retry window. num_preemptions_total stayed 0. There is no separate vllm abort counter in this build. So the probe is not reading a typo'd name; this build simply never increments that series on client disconnect (disconnect-aborted requests bypass the finished_reason recording path).",
            "The signals that DID move are completion-time, not abort-time: http_requests_total{handler='/v1/chat/completions',status='2xx'} +1, vllm:time_to_first_token_seconds_count +1, external_prefix_cache hits/queries +32768 (the aligned external retrieve happened during prefill), iteration_tokens +2/+4, spec_decode drafts +1 (stock, mtp3 arm). The engine observed the request, served TTFT, and the HTTP layer closed the coroutine as a 2xx when the client hung up.",
            "The lease monitor is structurally unable to catch the retrieve lease: the external retrieve completes during prefill BEFORE the first token, while the monitor loop only runs during the ~0.11 s exchange window, each 'sample' costs a full HTTP round-trip (the 0.05 s sleep is between samples, not a rate), and retrieve_leases is an instantaneous gauge that is back at 0 within milliseconds. 3 samples were taken, all zeros. The prefetch controller's completed_results_count is also 0 at settlement, so there is no cumulative LMCache-side retrieve counter on 18085 to fall back to; the durable evidence of the retrieve is the vllm external_prefix_cache_hits_total delta, which the probe already captures in the same window.",
        ],
        "verdict": {
            "server_abort_observed": "wrong signal for this build (series exists but is never incremented on client disconnect), not a timing race; the 60 s retry loop burns ~60 s per arm for nothing",
            "live_retrieve_observed_under_pressure": "sampling too slow AND sampling the wrong kind of signal (instantaneous gauge vs prefill-time event); cumulative counter does not exist on this endpoint",
        },
        "minimal_recommended_probe_changes": [
            "server_abort_observed: accept a disjunction as the abort signal: (request_success_total{abort} delta >= 1) OR (time_to_first_token_seconds_count delta >= 1 AND request_success_total delta == 0 AND client_closed_early) i.e. the server demonstrably served the request to first token and never recorded it as any success; record which signal fired in the gate detail. Keep fail-closed: if none of these move, still 'unavailable'. Drop or shorten the 60 s retry to ~5 s since the counters that do move are updated within the first poll.",
            "live_retrieve_observed_under_pressure: replace the gauge poll with the already-captured external_prefix_cache_hits_total delta across the cancellation exchange (hit tokens >= one chunk => a live external retrieve served this request), or add a cumulative LMCache retrieve counter to the 18085 status payload and poll that; keep required=False.",
            "Do NOT edit the probe while the campaign runs; apply after it exits and re-verify against the next arm's receipts (the gate change only widens 'unavailable' to 'pass' when the disjunction fires, existing passes are unaffected)."
        ],
        "source_receipts": {
            str(CAMP / "l2-stock-dcp4-mtp3/receipt.json"): sha(CAMP / "l2-stock-dcp4-mtp3/receipt.json"),
            str(CAMP / "l2-r30-dcp1-mtp0/receipt.json"): sha(CAMP / "l2-r30-dcp1-mtp0/receipt.json"),
            "probe": {"path": "/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r29/cache_lifecycle_probe.py"},
        },
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
