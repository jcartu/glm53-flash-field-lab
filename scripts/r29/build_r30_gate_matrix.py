#!/usr/bin/env python3
"""Build R/r30-cache-gate-matrix.json from lifecycle receipts (read-only over campaign dirs).

Run any time; it reflects whatever arms have completed so far. Refresh after each arm
and once more when the campaign exits.
"""
import json
import sys
from pathlib import Path

R = Path("/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909")
CAMP = R / "r30-cache-campaign-2"
ABORTED = R / "r30-cache-campaign"
OUT = R / "r30-cache-gate-matrix.json"


def load_gates_jsonl(root):
    out = {}
    p = root / "gates.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = d.get("name", "")
        if name.startswith("cache-lifecycle:"):
            out[name.split(":", 1)[1]] = d.get("detail") or {}
        elif name.startswith("boot:"):
            out.setdefault("__boot__" + name.split(":", 1)[1], {})
            out["__boot__" + name.split(":", 1)[1]]["image"] = (d.get("detail") or {}).get("image")
    return out


def arm_entry(receipt_path, lifecycle_detail, boot_image):
    d = json.loads(receipt_path.read_text())
    gates = d.get("gates") or []
    matrix = {}
    chs = None
    for g in gates:
        matrix[g["name"]] = {
            "status": g.get("status") or ("pass" if g.get("passed") else "fail"),
            "required": bool(g.get("required")),
        }
        if g["name"].startswith("cold."):
            obs = (g.get("detail") or {}).get("cache_hit_observation")
            if isinstance(obs, dict) and obs.get("source"):
                chs = chs or {}
                chs[g["name"]] = obs["source"]
    return {
        "arm": receipt_path.parent.name,
        "image": lifecycle_detail.get("image") or boot_image,
        "dcp": lifecycle_detail.get("dcp"),
        "spec": lifecycle_detail.get("spec"),
        "complete": d.get("complete"),
        "returncode": lifecycle_detail.get("returncode"),
        "elapsed_seconds": d.get("elapsed_seconds"),
        "gate_summary": d.get("gate_summary"),
        "required_gate_count": d.get("required_gate_count"),
        "failed_gates": [k for k, v in matrix.items() if v["status"] == "fail"],
        "unavailable_gates": [k for k, v in matrix.items() if v["status"] == "unavailable"],
        "not_applicable_gates": [k for k, v in matrix.items() if v["status"] == "not_applicable"],
        "cache_hit_observation_source": chs,
        "gates": matrix,
        "receipt": str(receipt_path),
    }


def main():
    lifecycle = load_gates_jsonl(CAMP)
    arms = []
    for receipt in sorted(CAMP.glob("l2-*/receipt.json")):
        label = receipt.parent.name
        detail = lifecycle.get(label, {})
        boot_image = (lifecycle.get("__boot__" + label) or {}).get("image")
        arms.append(arm_entry(receipt, detail, boot_image))

    aborted = None
    ab_receipt = ABORTED / "l2-stock-dcp4-mtp3" / "receipt.json"
    if ab_receipt.exists():
        ab_lifecycle = load_gates_jsonl(ABORTED).get("l2-stock-dcp4-mtp3", {})
        aborted = arm_entry(ab_receipt, ab_lifecycle, None)
        aborted["note"] = (
            "aborted pilot root: harness race (fixed in cache_lifecycle_probe.py); "
            "receipt incomplete (complete=false), gates reflect the partial run only"
        )

    all_gate_names = sorted({g for a in arms for g in a["gates"]})
    doc = {
        "schema": "r30-gate-matrix/v1",
        "generated_by": str(Path(sys.argv[0]).resolve()),
        "campaign_root": str(CAMP),
        "planned_arms": "stock R29 control dcp4/mtp3 + R30 arms dcp{1,4} x {mtp0,mtp3,dflash2}",
        "arms_completed_so_far": [a["arm"] for a in arms if a["complete"] is True],
        "arms_in_progress": [a["arm"] for a in arms if a["complete"] is not True],
        "gate_columns": all_gate_names,
        "arms": arms,
        "aborted_pilot": aborted,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); arms: {[a['arm'] for a in arms]}")


if __name__ == "__main__":
    main()
