#!/usr/bin/env python3
"""Write R/r30-dcp4-identity-analysis.json from retained r30-cache-campaign-2 receipts.

CPU-only, read-only over campaign dirs. Rerunnable; if the DCP4 identity follow-up
summary exists, its classification is attached.
"""
import hashlib
import json
from pathlib import Path

R = Path("/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909")
CAMP = R / "r30-cache-campaign-2"
FOLLOWUP = R / "r30-dcp4-identity-followup"
OUT = R / "r30-dcp4-identity-analysis.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "cold_token": x, "replay_token": y}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "cold_token": None, "replay_token": None}
    return {"index": None, "cold_token": None, "replay_token": None}


def arm_block(arm):
    d = json.loads((CAMP / arm / "receipt.json").read_text())
    cold = d["stages"]["cold"]["request"]
    csm = cold["summary"]
    cti = cold["token_identity"]
    stages = {}
    for prefix, stage_key, req in [
        ("replay", "exact_replay", d["stages"]["exact_replay"]["request"]),
        ("restart", "restart", d["stages"]["restart"]["replay"]["request"]),
    ]:
        rsm = req["summary"]
        rti = req["token_identity"]
        gate = next(g for g in d["gates"] if g["name"] == f"{prefix}.correct_and_identical")
        det = gate.get("detail") or {}
        md = (d["stages"][stage_key].get("request") or d["stages"][stage_key].get("replay") or {}).get("metrics", {}).get("delta") or {}
        stages[prefix] = {
            "gate_status": gate.get("status"),
            "required": bool(gate.get("required")),
            "gate_detail": det,
            "request_body_sha256_identical": cold["http"]["request_body_sha256"] == req["http"]["request_body_sha256"],
            "prompt_tokens": [csm["prompt_tokens"], rsm["prompt_tokens"]],
            "returned_prompt_token_ids_sha256_identical": cti["returned_prompt_token_ids_sha256"] == rti["returned_prompt_token_ids_sha256"],
            "completion_tokens": [csm["completion_tokens"], rsm["completion_tokens"]],
            "generated_token_ids_sha256": {"cold": cti["generated_token_ids_sha256"], "replay": rti["generated_token_ids_sha256"]},
            "first_generated_token_difference": first_diff(cti["generated_token_ids"], rti["generated_token_ids"]),
            "visible_content_sha256_identical": csm["visible_content_sha256"] == rsm["visible_content_sha256"],
            "visible_content": csm["visible_content"],
            "reasoning_content": {"cold": csm["reasoning_content"], "replay": rsm["reasoning_content"]},
            "reasoning_sha256_identical": csm["reasoning_content_sha256"] == rsm["reasoning_content_sha256"],
            "assistant_message_sha256_identical": csm["assistant_message_sha256"] == rsm["assistant_message_sha256"],
            "finish_reasons": [csm["finish_reason"], rsm["finish_reason"]],
            "visible_marker_exact_both": bool(csm["visible_marker_exact"] and rsm["visible_marker_exact"]),
            "external_hit_tokens": (rsm.get("cache_hit_observation") or {}).get("tokens"),
            "external_hit_source": (rsm.get("cache_hit_observation") or {}).get("source"),
            "engine_counter_deltas": {k: md.get(k) for k in (
                "vllm:num_preemptions_total", "vllm:num_requests_running", "vllm:num_requests_waiting",
                "vllm:request_success_total", "vllm:request_success_total:abort",
                "vllm:external_prefix_cache_hits_total") if k in md},
        }
    return {
        "sampling": cold.get("sampling"),
        "receipt": str(CAMP / arm / "receipt.json"),
        "receipt_sha256": sha(CAMP / arm / "receipt.json"),
        "stages": stages,
    }


def main():
    failing = {
        arm: arm_block(arm)
        for arm in ["l2-r30-dcp4-mtp0", "l2-r30-dcp4-dflash2"]
    }
    passing = {
        arm: arm_block(arm)
        for arm in ["l2-r30-dcp4-mtp3", "l2-stock-dcp4-mtp3"]
    }

    followup = None
    fsum = FOLLOWUP / "identity-followup-summary.json"
    if fsum.exists():
        followup = {"path": str(fsum), "sha256": sha(fsum), "content": json.loads(fsum.read_text())}

    doc = {
        "schema": "r30-dcp4-identity-analysis/v1",
        "generated_at": "2026-09-09T23:45:00+02:00",
        "scope": "CPU-only read of retained lifecycle receipts. Neutral description of what differed; no defect attribution beyond what the receipts state.",
        "observation": (
            "On the R30 DCP4 arms, the fixed deterministic 32768-token prompt replayed with byte-identical "
            "request bodies, identical returned prompt token ids, and a full external hit (32768 cached tokens "
            "via isolated_request.external_prefix_cache_hits_total). The visible final answer (the cache marker) "
            "was correct and byte-identical in every comparison, finish_reason=stop on both sides, no preemptions, "
            "no queue movement, exactly one request_success per replay. What differed was the generated token "
            "sequence BEFORE the visible marker, i.e. the short chain-of-thought reasoning preamble: it diverges "
            "at generated-token index 1 and re-converges on the marker. The gate "
            "correct_and_identical requires generated_token_ids and assistant_message sha equality, so a "
            "reasoning-only divergence fails it."
        ),
        "per_arm": {
            "failing_arms": failing,
            "passing_contrast_arms": passing,
        },
        "pattern": {
            "l2-r30-dcp4-mtp0": "replay identical (pass); after container restart the reasoning text differs (cold 'Return the marker exactly.' vs post-restart 'Return exactly the marker.'), same length 27/27, first token diff at index 1; persisted_checkpoint_boundary_same=false",
            "l2-r30-dcp4-dflash2": "replay itself already differs (cold 'Return exactly the marker.' 27 tok vs replay 'Return exact marker.' 26 tok, first diff index 1) with persisted boundary same=true; post-restart differs again ('Return the marker only.', 27 tok) with persisted boundary same=false",
            "l2-r30-dcp4-mtp3": "both stages fully identical (pass)",
            "l2-stock-dcp4-mtp3": "both stages fully identical (pass; R29 control)",
            "common": "divergence confined to reasoning content; visible marker always correct; temperature 0.0, top_p 1.0, seed 290064 pinned identically on every request",
        },
        "cannot_yet_classify": (
            "No stock R29 control exists for dcp4-mtp0 or dcp4-dflash2 in the completed campaign "
            "(the only R29 arm was dcp4-mtp3, which passes). Therefore it is not yet determined whether "
            "reasoning non-identity after DCP4 cache restore is an R30-specific regression, a DCP4 property "
            "present on both images, or run-level flakiness. A follow-up run (stock R29 dcp4-mtp0, stock R29 "
            "dcp4-dflash2, R30 dcp4-mtp0 repeat) is executing at R/r30-dcp4-identity-followup; its "
            "classification will be attached here when present."
        ),
        "classification": (followup or {}).get("content"),
        "source_receipts": {},
    }
    doc["source_receipts"] = {
        str(CAMP / a / "receipt.json"): sha(CAMP / a / "receipt.json")
        for a in ["l2-r30-dcp4-mtp0", "l2-r30-dcp4-dflash2", "l2-r30-dcp4-mtp3", "l2-stock-dcp4-mtp3"]
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); followup attached: {followup is not None}")


if __name__ == "__main__":
    main()
