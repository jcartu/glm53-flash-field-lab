#!/usr/bin/env python3
"""Compare recorded published/QAD windows without mixing metric definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import qualification_report as qualification
import render_r26_summary as style
from run_qad_after_r27 import R26_IMAGE

QAD_REVISION = "3959f8a063b77cfdb22ab2e085a1f76fd38b195b"

NATURAL_ARMS = (
    ("r25-mtp3-bf16", "R25\nBF16 head"),
    ("r26-mtp3-bf16", "R26\nBF16 head"),
    ("r26-mtp3-default-nvfp4", "R26\nNVFP4 head"),
    ("drock-r26-overlay-mtp3-bf16", "D-Rock overlay\nBF16 head"),
)


def collect(root: Path, baseline: Path) -> dict:
    sources = {}
    documents = {}

    def load(path):
        path = path.resolve()
        if path not in documents:
            raw = path.read_bytes()
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            documents[path] = json.loads(raw)
        return documents[path]

    checkpoint = load(root / "checkpoint-verification.json")
    if not checkpoint.get("complete") or checkpoint.get("revision") != QAD_REVISION:
        raise ValueError("The chart requires the verified step2500 checkpoint")
    published_model = None

    speed_roots = {"published": baseline / "clean-reruns", "qad": root}
    isolation = {
        arm: qualification.Isolation(qualification.Root(folder))
        for arm, folder in speed_roots.items()
    }
    speed = []
    for dcp in (1, 4):
        stem = f"acceptance-dcp{dcp}-mtp3-r26-nvfp4-head"
        launches = {
            arm: load(folder / f"{stem}.launch.json")
            for arm, folder in speed_roots.items()
        }
        normalized = {
            arm: {
                key: value
                for key, value in launch.items()
                if key not in ("label", "model_dir")
            }
            for arm, launch in launches.items()
        }
        if (
            normalized["published"] != normalized["qad"]
            or launches["qad"]["image"] != R26_IMAGE
        ):
            raise ValueError("Checkpoint speed launch settings differ beyond weights")
        if launches["published"]["model_dir"] == launches["qad"]["model_dir"]:
            raise ValueError("Checkpoint speed comparison did not change weights")
        if launches["qad"]["model_dir"] != checkpoint["model_dir"]:
            raise ValueError("Speed launch does not use the verified QAD checkpoint")
        if published_model is None:
            published_model = launches["published"]["model_dir"]
        if launches["published"]["model_dir"] != published_model:
            raise ValueError("Published checkpoint differs across speed pairs")
        for launch in launches.values():
            if (
                launch["tp"],
                launch["dcp"],
                launch["spec"],
                launch["cache"],
                launch["kv"],
            ) != (4, dcp, "mtp3", "vram", "fp8_ds_mla") or launch["env"].get(
                "VLLM_GLM53_MTP_DRAFT_HEAD"
            ) != "nvfp4":
                raise ValueError(
                    "Speed launch differs from the stated MTP3/NVFP4-head configuration"
                )
        for concurrency in (1, 8):
            row = {
                "dcp": dcp,
                "concurrency": concurrency,
                "context_tokens": 0,
                "arms": {},
            }
            for arm, folder in speed_roots.items():
                values = []
                for repeat in (1, 2):
                    document = load(folder / f"{stem}-repeat{repeat}.json")
                    reference = load(
                        speed_roots["published"] / f"{stem}-repeat{repeat}.json"
                    )
                    metadata = {
                        key: value
                        for key, value in document["metadata"].items()
                        if key != "timestamp"
                    }
                    expected = {
                        key: value
                        for key, value in reference["metadata"].items()
                        if key != "timestamp"
                    }
                    if metadata != expected:
                        raise ValueError("Checkpoint benchmark metadata differs")
                    command = load(folder / f"{stem}-repeat{repeat}.bench.command.json")
                    window = {
                        "started_at": command["started_at"],
                        "finished_at": command["finished_at"],
                    }
                    verdict = isolation[arm].verdict(window)
                    if command["returncode"] != 0 or verdict["status"] != "clean":
                        raise ValueError(f"Unqualified {arm} speed window: {stem}")
                    cells = [
                        cell
                        for cell in document["results"]
                        if cell["concurrency"] == concurrency
                        and cell["context_tokens"] == 0
                    ]
                    if len(cells) != 1:
                        raise ValueError("Missing unique matched speed cell")
                    cell = cells[0]
                    if (
                        cell["num_errors"]
                        or cell["underfilled"]
                        or cell["warmup_timed_out"]
                        or cell["hardware_summary"]["gpu_count"] != 4
                        or cell["aggregate_source"] != "openai_continuous_usage"
                    ):
                        raise ValueError(
                            "Speed cell lacks complete four-GPU continuous-usage evidence"
                        )
                    values.append(cell["aggregate_tps"])
                row["arms"][arm] = {"mean": statistics.mean(values), "repeats": values}
            row["delta_percent"] = 100 * (
                row["arms"]["qad"]["mean"] / row["arms"]["published"]["mean"] - 1
            )
            speed.append(row)

    templates = load(root / "checkpoint-template-identity.json")
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
            raise ValueError(
                "Checkpoint pairs do not share identical tokenizer/template artifacts"
            )
        if identity["published"]["path"] != str(
            Path(published_model) / filename
        ) or identity["candidate"]["path"] != str(
            Path(checkpoint["model_dir"]) / filename
        ):
            raise ValueError(
                "Template identity refers to a different checkpoint directory"
            )

    natural = []
    natural_roots = {"published": baseline / "followups", "qad": root}
    natural_isolation = {
        arm: qualification.Isolation(qualification.Root(folder))
        for arm, folder in natural_roots.items()
    }
    for name, label in NATURAL_ARMS:
        pair = {
            arm: load(folder / f"realistic-acceptance-{name}.json")
            for arm, folder in natural_roots.items()
        }
        configs = {
            arm: {
                key: value
                for key, value in document["config"].items()
                if key != "model_dir"
            }
            for arm, document in pair.items()
        }
        if (
            configs["published"] != configs["qad"]
            or pair["published"]["image"] != pair["qad"]["image"]
            or pair["published"]["input_set"] != pair["qad"]["input_set"]
        ):
            raise ValueError(
                "Natural acceptance pair changed more than checkpoint weights"
            )
        if (
            pair["published"]["config"]["model_dir"] != published_model
            or pair["qad"]["config"]["model_dir"] != checkpoint["model_dir"]
        ):
            raise ValueError(
                "Natural acceptance pair does not use the same verified checkpoints"
            )
        row = {"arm": name, "label": label, "image": pair["qad"]["image"], "values": {}}
        for arm, document in pair.items():
            summary = document["summary"]
            measured = summary["all_measurements"]
            if (
                not summary["runtime_count_integrity"]["passed"]
                or measured["eligible_request_count"] != 32
            ):
                raise ValueError("Natural acceptance slice is incomplete")
            verdict = natural_isolation[arm].verdict(
                {
                    "started_at": document["started_at"],
                    "finished_at": document["finished_at"],
                }
            )
            if verdict["status"] != "clean":
                raise ValueError(
                    "Natural acceptance pair lacks an isolated measurement window"
                )
            row["values"][arm] = (
                100 * measured["token_weighted_pooled_acceptance_fraction"]
            )
        natural.append(row)
    for observer in [*isolation.values(), *natural_isolation.values()]:
        for path in observer.files:
            if path.is_file():
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": "qad-performance-comparison/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "baseline": str(baseline),
        "sources": sources,
        "speed": speed,
        "natural": natural,
        "checkpoint_revision": checkpoint["revision"],
        "templates": templates,
        "speed_definition": "Client-reported aggregate output tok/s, continuous-usage accounting, mean of two matched 60s windows, context 0. Not the interior steady-counter values in the separate R26 figure.",
        "scope": "Each pair matches runtime/configuration and inputs; checkpoint weights differ. Acceptance is token-weighted accepted/proposed draft tokens, not a quality score. Natural replies may differ across checkpoints.",
        "metadata_erratum": "Natural probe revision hardcodes weights.qad_checkpoint_used=false. Checkpoint identity is established by config.model_dir and recorded launch/mount paths, never by that stale boolean.",
    }


def render(data: dict, path: Path) -> None:
    cv = style.Canvas()
    cv.header(
        "QAD step2500: output speed and MTP3 acceptance",
        "Published vs QAD weights · 4× RTX PRO 6000 · pairs stay within the same runtime and launch settings",
        style.GREEN,
    )
    cv.card(150, 675)
    cv.panel_title(
        182,
        "Matched R26 runtime: MTP3 with NVFP4 proposal head",
        "Client-reported aggregate output tokens/s · context 0 · two 60-second windows per cell · all GPU windows eligible",
    )
    cv.chip_row(
        76, 248, [(style.BLUE, "Published NVFP4"), (style.GREEN, "QAD step2500")]
    )
    for panel, concurrency in enumerate((1, 8)):
        ax = cv.axes(100 + panel * 650, 320, 550, 225)
        style.style_axis(ax)
        ax.set_title(f"C{concurrency}", fontsize=13, color=style.TEXT)
        for arm, offset, color in (
            ("published", -0.18, style.BLUE),
            ("qad", 0.18, style.GREEN),
        ):
            values = [
                next(
                    row["arms"][arm]["mean"]
                    for row in data["speed"]
                    if row["dcp"] == dcp and row["concurrency"] == concurrency
                )
                for dcp in (1, 4)
            ]
            bars = ax.bar(
                [index + offset for index in (0, 1)],
                values,
                width=0.35,
                color=color,
                zorder=3,
            )
            style.bar_labels(ax, bars, "{:.0f}", fontsize=12)
        ax.set_xticks([0, 1], ["DCP1", "DCP4"], fontsize=11)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    deltas = " · ".join(
        f"DCP{row['dcp']}/C{row['concurrency']} {row['delta_percent']:+.1f}%"
        for row in data["speed"]
    )
    cv.text(76, 608, deltas, 10.5, color=style.GREEN, bold=True)
    cv.wrapped(
        76,
        642,
        "Full client windows, not the R26 figure's interior counters. Two repeats are not an equivalence test.",
        9,
        1250,
        color=style.MUTED,
    )

    cv.card(700, 1180)
    cv.panel_title(
        732,
        "Natural-chat draft acceptance: same 32-prompt/seed slice",
        "Token-weighted accepted ÷ proposed draft tokens · C1+C8 · 32 complete visible replies per arm · compare each pair only",
    )
    ax = cv.axes(105, 835, 1190, 245)
    style.style_axis(ax)
    for arm, offset, color in (
        ("published", -0.18, style.BLUE),
        ("qad", 0.18, style.GREEN),
    ):
        values = [row["values"][arm] for row in data["natural"]]
        bars = ax.bar(
            [index + offset for index in range(4)],
            values,
            width=0.35,
            color=color,
            zorder=3,
        )
        style.bar_labels(ax, bars, "{:.1f}%", fontsize=11)
    ax.set_xticks(range(4), [row["label"] for row in data["natural"]], fontsize=10)
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
    cv.text(
        76,
        1148,
        "This is natural-chat acceptance, not the repetition-probe ceiling and not a quality/fidelity score.",
        9.5,
        color=style.MUTED,
    )

    cv.card(1200, 1430)
    cv.panel_title(1232, "A speed gain does not settle the quality decision", None)
    y = 1270
    for note in [
        "Quality fixtures are mixed; see the separate exact/near/wrong-answer chart. Higher draft acceptance is not proof of better answers.",
        "No runtime, KV format, topology or proposal-head change is folded into a checkpoint pair. GPU health and foreign-process exclusions remain enforced.",
        "All values, repeated windows, input/configuration checks and source hashes are in performance-qad.json.",
    ]:
        y = cv.wrapped(76, y, note, 9.5, 1230, color=style.TEXT) + 8
    cv.footer(
        [
            f"Speed image: R26 d0592ea9d73c · target QAD revision {data['checkpoint_revision'][:12]}. Natural panels keep runtimes separate.",
            "Measured checkpoint comparison only. No candidate promotion implied.",
        ]
    )
    cv.fig.savefig(path, dpi=style.DPI, facecolor=style.BG)
    style.plt.close(cv.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "drock-lmcache/r26-battery",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results/qad-step2500",
    )
    args = parser.parse_args()
    data = collect(args.root.resolve(), args.baseline_root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / "performance-qad.png"
    render(data, image)
    data["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    data["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "performance-qad.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "image": str(image),
                "speed_pairs": len(data["speed"]),
                "natural_pairs": len(data["natural"]),
            }
        )
    )


if __name__ == "__main__":
    main()
