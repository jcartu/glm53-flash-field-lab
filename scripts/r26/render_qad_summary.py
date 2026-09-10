#!/usr/bin/env python3
"""Chart the completed matched checkpoint slice, independently of full runbook status."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import render_r26_summary as style
from run_qad_after_r27 import R26_IMAGE

CONFIGS = ("dcp1-dflash", "dcp4-mtp3")
LABELS = {"dcp1-dflash": "DCP1 · DFlash K7", "dcp4-mtp3": "DCP4 · MTP3"}
ARMS = ("published", "candidate")
COUNTS = ("attempted", "completed", "correct", "wrong", "errors", "exact", "near")


def collect(root: Path) -> dict:
    sources = {}

    def load(name):
        path = root / name
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    complete = load("qad-matched-completed.json")
    ledger = load("qad-matched-ledger.json")
    checkpoint = load("checkpoint-verification.json")
    templates = load("checkpoint-template-identity.json")
    if (
        not complete.get("complete")
        or complete.get("cells") != 4
        or len(ledger["cells"]) != 4
    ):
        raise ValueError("Matched checkpoint slice is incomplete")
    if (
        ledger["image"] != R26_IMAGE
        or not checkpoint.get("complete")
        or checkpoint.get("revision") != "3959f8a063b77cfdb22ab2e085a1f76fd38b195b"
    ):
        raise ValueError("Checkpoint/image provenance is incomplete")
    expected_templates = {
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    }
    if (
        templates.get("schema") != "checkpoint-template-identity/v1"
        or set(templates.get("files", {})) != expected_templates
    ):
        raise ValueError(
            "Tokenizer identity must include the standalone served chat template"
        )
    for filename, identity in templates["files"].items():
        if (
            identity.get("identical") is not True
            or identity["published"]["sha256"] != identity["candidate"]["sha256"]
        ):
            raise ValueError("Tokenizer, template or generation configuration differs")
        for arm in ("published", "candidate"):
            if identity[arm]["path"] != str(Path(ledger[arm]) / filename):
                raise ValueError(
                    "Template identity refers to a different checkpoint directory"
                )
    rows = []
    launches = {}
    for config in CONFIGS:
        for arm in ARMS:
            matches = [
                row
                for row in ledger["cells"]
                if row["config"] == config and row["weights"] == arm
            ]
            if len(matches) != 1 or matches[0].get("booted") is not True:
                raise ValueError(f"Missing completed checkpoint arm: {config}/{arm}")
            cell = matches[0]
            launch = load(cell["label"] + ".launch.json")
            if launch["image"] != R26_IMAGE or launch["model_dir"] != ledger[arm]:
                raise ValueError("Recorded checkpoint launch differs from the ledger")
            launches[(config, arm)] = {
                key: value
                for key, value in launch.items()
                if key not in ("label", "model_dir")
            }
            profiles = {}
            for profile in ("estonia", "lavd-test", "lavd-template-default"):
                receipt = load(f"{cell['label']}-{profile}.json")
                counts = {key: receipt["selected_summary"][key] for key in COUNTS}
                expected = 8 if profile == "lavd-template-default" else 24
                if (
                    counts["attempted"] != expected
                    or counts["completed"] + counts["errors"] != expected
                ):
                    raise ValueError("Unaccounted checkpoint profile requests")
                profiles[profile] = counts
            rows.append(
                {
                    "config": config,
                    "arm": arm,
                    "profiles": profiles,
                    "reasoning_scopes": cell["reasoning_scopes"],
                }
            )
        if launches[(config, "published")] != launches[(config, "candidate")]:
            raise ValueError(
                "Matched checkpoint launches differ beyond label and weights"
            )
    plan = load("phase-plan.json")
    executed = (
        load("qualification-executed.json")
        if (root / "qualification-executed.json").exists()
        else None
    )
    progress = (
        load("phase-progress.json") if (root / "phase-progress.json").exists() else []
    )
    interruption = (
        load("qualification-interrupted.json")
        if (root / "qualification-interrupted.json").exists()
        else None
    )
    phase_rows = executed["phases"] if executed else progress
    if (root / "recovery-plan.json").exists():
        recovery = load("recovery-plan.json")
        plan = recovery["phase_plan"]["phases"]
        first_phase = "qad-matched-published-vs-candidate"
        if not any(row["phase"] == first_phase for row in phase_rows):
            preserved = load(f"phase-{first_phase}.command.json")
            if preserved.get("returncode") != 0:
                raise ValueError(
                    "Preserved matched phase is not a successful invocation"
                )
            phase_rows = [
                {
                    "phase": first_phase,
                    "returncode": preserved["returncode"],
                    "execution": "preserved-pre-fault-invocation",
                },
                *phase_rows,
            ]
    full_complete = bool(
        executed
        and executed.get("all_phases_attempted") is True
        and [row["phase"] for row in phase_rows] == [row["name"] for row in plan]
    )
    interrupted = bool(
        interruption
        and (
            not executed
            or interruption.get("timestamp", 0) > executed.get("finished_at", 0)
        )
    )
    if interrupted:
        full_complete = False
    tail_recovery = (
        load("tail-recovery-plan.json")
        if (root / "tail-recovery-plan.json").exists()
        else None
    )
    return {
        "schema": "qad-matched-summary/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image": R26_IMAGE,
        "checkpoint_revision": checkpoint["revision"],
        "rows": rows,
        "sources": sources,
        "templates": templates,
        "full_runbook": {
            "complete_execution": full_complete,
            "recorded_phases": len(phase_rows),
            "interrupted": interrupted,
            "interruption": interruption if interrupted else None,
            "planned_phases": len(plan),
            "phase_returncodes": phase_rows,
            "nonzero_phase_exits": sum(row["returncode"] != 0 for row in phase_rows),
            "resumed_tail": tail_recovery is not None,
        },
        "limits": "Small matched checkpoint samples, not a general quality or fidelity score. "
        "LAVD verifier-accepted includes near answers. Explicit-low and default-template "
        "results are separate; full runbook execution and qualification pass are separate.",
    }


def render(data: dict, path: Path) -> None:
    cv = style.Canvas()
    cv.header(
        "QAD step2500: same runtime, different checkpoint",
        "4× RTX PRO 6000 · TP4 · FP8 KV · tokenizer/template/generation configs verified byte-identical",
        style.GREEN,
    )
    cv.card(150, 610)
    cv.panel_title(
        182,
        "Explicit low reasoning: verifier-accepted answers out of 24",
        "Compare checkpoint pairs within each workload only. LAVD acceptance includes near answers.",
    )
    cv.chip_row(
        76, 245, [(style.BLUE, "Published NVFP4"), (style.GREEN, "QAD step2500")]
    )
    ax = cv.axes(100, 285, 1200, 235)
    style.style_axis(ax)
    groups = [
        (config, profile) for config in CONFIGS for profile in ("estonia", "lavd-test")
    ]
    for arm, offset, color in (
        ("published", -0.18, style.BLUE),
        ("candidate", 0.18, style.GREEN),
    ):
        values = [
            next(
                row["profiles"][profile]["correct"]
                for row in data["rows"]
                if row["config"] == config and row["arm"] == arm
            )
            for config, profile in groups
        ]
        bars = ax.bar(
            [x + offset for x in range(4)], values, width=0.35, color=color, zorder=3
        )
        style.bar_labels(ax, bars, "{:.0f}", fontsize=12)
    ax.set_xticks(
        range(4),
        [
            f"{LABELS[config]}\n{'Estonia' if profile == 'estonia' else 'LAVD'}"
            for config, profile in groups
        ],
        fontsize=10,
    )
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 8, 16, 24])
    cv.text(
        76,
        582,
        "Small samples and mixed outcomes: neither a blanket checkpoint win nor evidence of equal fidelity.",
        9.5,
        color=style.MUTED,
    )

    cv.card(630, 1110)
    cv.panel_title(
        662,
        "Default-template LAVD: eight runs per configuration",
        "Exact, near and failed answers are shown separately. Do not pool these with the low-reasoning runs above.",
    )
    cv.chip_row(
        76,
        728,
        [
            (style.GREEN, "Exact"),
            (style.AMBER, "Near (verifier accepted)"),
            (style.RED, "Wrong/error"),
        ],
        pt=9.5,
    )
    ax = cv.axes(430, 778, 845, 238)
    style.style_axis(ax, "x")
    labels = []
    for index, row in enumerate(data["rows"]):
        counts = row["profiles"]["lavd-template-default"]
        labels.append(
            f"{LABELS[row['config']]}\n{'Published' if row['arm'] == 'published' else 'QAD step2500'}"
        )
        left = 0
        for value, color in (
            (counts["exact"], style.GREEN),
            (counts["near"], style.AMBER),
            (counts["wrong"] + counts["errors"], style.RED),
        ):
            ax.barh(index, value, left=left, height=0.62, color=color, zorder=3)
            if value:
                ax.text(
                    left + value / 2,
                    index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight=700,
                    color=style.BG,
                )
            left += value
        if left != 8:
            raise ValueError(
                "Default-template outcome categories do not account for all eight runs"
            )
    ax.set_yticks(range(4), labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 8)
    ax.set_xticks([0, 2, 4, 6, 8])
    cv.text(
        76,
        1075,
        "The visible result is not uniformly better: DFlash and MTP3 move differently in this sample.",
        9.5,
        color=style.MUTED,
    )

    cv.card(1130, 1430)
    cv.panel_title(1162, "Coverage and interpretation", None)
    runbook = data["full_runbook"]
    status = (
        "complete execution record"
        if runbook["complete_execution"]
        else "interrupted / not a final qualification"
        if runbook["interrupted"]
        else "still running / not a final qualification"
    )
    notes = [
        f"Serving runbook: {runbook['recorded_phases']}/{runbook['planned_phases']} phases recorded; {status}.",
        f"Nonzero phase exits: {runbook['nonzero_phase_exits']}. Completed execution does not mean every check passed.",
        "Matched checkpoint pairs share image, TP/DCP, KV, speculator and launch settings; only mounted weights differ.",
        "Bounded fixtures only; cache, topology, scheduling, capacity and long-generation findings are in the full report.",
    ]
    if runbook.get("resumed_tail"):
        notes.append(
            "Tail: core0; retained results: core+250. Memory+6000 unchanged. "
            "No matched-clock speed comparisons across this boundary."
        )
    y = 1215
    for note in notes:
        y = cv.wrapped(76, y, note, 9.5, 1230, color=style.TEXT) + 8
    cv.footer(
        [
            f"Image {data['image'].rsplit('sha256:', 1)[-1][:12]} · QAD revision {data['checkpoint_revision'][:12]}. No R27 runtime changes included.",
            "Counts, source hashes and runbook state: summary-qad.json. No production promotion implied.",
        ]
    )
    cv.fig.savefig(path, dpi=style.DPI, facecolor=style.BG)
    style.plt.close(cv.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results/qad-step2500",
    )
    args = parser.parse_args()
    data = collect(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / "summary-qad.png"
    render(data, image)
    data["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    data["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "summary-qad.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "image": str(image),
                "matched_arms": len(data["rows"]),
                "full_runbook": data["full_runbook"],
            }
        )
    )


if __name__ == "__main__":
    main()
