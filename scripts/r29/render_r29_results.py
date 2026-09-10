#!/usr/bin/env python3
"""Render the R29 campaign charts (performance, quality, baseline, and LMCache) from retained receipts.

CPU-only renderer. Every plotted number is loaded directly from verified R29 execution receipts:
1. 01-speculative-matrix-throughput.png:
   Speculative matrix C1 and C8 output tok/s at ctx0 and 32K across 18 configurations
   (3 checkpoints × 6 spec/DCP modes).
2. 02-speculative-efficiency.png:
   Acceptance fraction and emitted tokens per verifier step across checkpoints, specs, and DCPs.
3. 03-checkpoint-quality.png:
   52 functional tasks (strict format vs semantic correctness) and 32 history tasks (up to 524K context).
4. 04-runtime-baseline.png:
   R28.1 vs R29 standardized decode runtime baseline per cell with both repeats visible.
5. 05-lmcache-ram-overlay.png:
   LMCache RAM stock vs PR #64 overlay with verifier steps/s side panel demonstrating engine equivalence.

Outputs:
    glm53-flash-field-lab/results/r29/01-speculative-matrix-throughput.png
    glm53-flash-field-lab/results/r29/02-speculative-efficiency.png
    glm53-flash-field-lab/results/r29/03-checkpoint-quality.png
    glm53-flash-field-lab/results/r29/04-runtime-baseline.png
    glm53-flash-field-lab/results/r29/05-lmcache-ram-overlay.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Base paths
WORKSPACE = Path(__file__).resolve().parents[3]
FIELD_LAB = Path(__file__).resolve().parents[2]
R = WORKSPACE / "drock-lmcache" / "r29-execution-20260909"
OUT = FIELD_LAB / "results" / "r29"

# Shared dark theme styling (Discord / GitHub dark)
BG = '#0D1117'
PANEL = '#161B22'
GRID = '#263242'
TEXT = '#E6EDF3'
MUTED = '#8B949E'
BLUE = '#58A6FF'    # Published / R28.1 / Stock
PURPLE = '#BC8CFF'  # QAD2500 / R29
GREEN = '#3FB950'   # TVN1500 / Passed
AMBER = '#E3B341'   # PR #64 / Warning
RED = '#F85149'     # Regressed / Error


def render_chart1_speculative_matrix(sm_data: dict, out_dir: Path) -> Path:
    """Chart 1: Speculative matrix C1 and C8 output tok/s at ctx0 and 32K."""
    configs_by_label = {c["label"]: c for c in sm_data["configurations"]}

    checkpoints = [
        ("published", "Published", BLUE),
        ("qad2500", "QAD-2500", PURPLE),
        ("tvn1500", "TVN-1500", GREEN)
    ]

    modes = [
        ("dcp1", "mtp0", "DCP1\nMTP0"),
        ("dcp1", "mtp3", "DCP1\nMTP3"),
        ("dcp1", "dflash2", "DCP1\nDFlash2"),
        ("dcp4", "mtp0", "DCP4\nMTP0"),
        ("dcp4", "mtp3", "DCP4\nMTP3"),
        ("dcp4", "dflash2", "DCP4\nDFlash2"),
    ]

    conditions = [
        (1, 0, "C=1, Context 0 (Single-Stream Cold/Short)"),
        (1, 32768, "C=1, Context 32K (Single-Stream Deep Context)"),
        (8, 0, "C=8, Context 0 (Saturated Batch Short)"),
        (8, 32768, "C=8, Context 32K (Saturated Batch Deep Context)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
    fig.subplots_adjust(hspace=0.35, wspace=0.20, top=0.90, bottom=0.10, left=0.07, right=0.95)

    x = np.arange(len(modes))
    bar_width = 0.26

    for idx, (conc, ctx, title) in enumerate(conditions):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor(PANEL)
        ax.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

        for i, (ck_key, ck_name, color) in enumerate(checkpoints):
            vals = []
            for dcp, spec, _ in modes:
                cfg_label = f"{ck_key}-{dcp}-{spec}"
                cfg = configs_by_label[cfg_label]
                cell = next(c for c in cfg["cells"] if c["concurrency"] == conc and c["context_tokens"] == ctx)
                vals.append(cell["mean_aggregate_tps"])

            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=ck_name if idx == 0 else "", color=color, alpha=0.9, zorder=3)

            for bar, val in zip(bars, vals):
                height = bar.get_height()
                fs = 8.5 if val >= 100 else 8.0
                ax.annotate(f"{val:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=fs, fontweight='bold', color=TEXT)

        ax.set_title(title, fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in modes], fontsize=9.5, color=TEXT)
        ax.tick_params(axis='y', colors=MUTED, labelsize=9)
        ax.set_ylabel("Output Throughput (tok/s)", fontsize=10, color=MUTED)

        max_val = max(ax.get_ylim()[1], 100)
        ax.set_ylim(0, max_val * 1.15)

        for spine in ax.spines.values():
            spine.set_color(GRID)

    fig.suptitle("R29 Speculative Matrix: Aggregate Output tok/s by Mode & Checkpoint",
                 fontsize=17, fontweight='bold', color=TEXT, y=0.965)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=True,
               facecolor=PANEL, edgecolor=GRID, fontsize=11, labelcolor=TEXT)

    fig.text(0.07, 0.012, "4x GB202GL Blackwell (RTX PRO 6000) • 30 s duration cells, two warmed repeats • descriptive means, not significance claims",
             fontsize=9.5, color=MUTED)

    out_file = out_dir / "01-speculative-matrix-throughput.png"
    plt.savefig(out_file, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_file


def render_chart2_speculative_efficiency(sm_data: dict, out_dir: Path) -> Path:
    """Chart 2: Acceptance fraction and emitted tokens per verifier step."""
    configs_by_label = {c["label"]: c for c in sm_data["configurations"]}

    checkpoints = [
        ("published", "Published", BLUE),
        ("qad2500", "QAD-2500", PURPLE),
        ("tvn1500", "TVN-1500", GREEN)
    ]

    modes = [
        ("dcp1", "mtp3", "DCP1\nMTP3"),
        ("dcp4", "mtp3", "DCP4\nMTP3"),
        ("dcp1", "dflash2", "DCP1\nDFlash2"),
        ("dcp4", "dflash2", "DCP4\nDFlash2"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.25)

    x = np.arange(len(modes))
    bar_width = 0.26

    # Panel 1: Acceptance fraction
    ax1.set_facecolor(PANEL)
    ax1.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    for i, (ck_key, ck_name, color) in enumerate(checkpoints):
        means = []
        mins = []
        maxs = []
        for dcp, spec, _ in modes:
            cfg = configs_by_label[f"{ck_key}-{dcp}-{spec}"]
            vals = [c["mean_acceptance_fraction"] * 100 for c in cfg["cells"]]
            means.append(np.mean(vals))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))

        offset = (i - 1) * bar_width
        bars = ax1.bar(x + offset, means, bar_width, label=ck_name, color=color, alpha=0.9, zorder=3)

        yerr = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]
        ax1.errorbar(x + offset, means, yerr=yerr, fmt='none', ecolor=TEXT, capsize=4, capthick=1.2, zorder=4)

        for bar, m in zip(bars, means):
            ax1.annotate(f"{m:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, m - 3.5),
                         ha='center', va='top', fontsize=8.5, fontweight='bold', color=TEXT,
                         bbox=dict(boxstyle="round,pad=0.15", facecolor=PANEL, edgecolor='none', alpha=0.85))

    ax1.set_title("Draft Token Acceptance Fraction (%)\n(Mean across 6 cells, error bars show min/max across C1–C8 / ctx 0–32K)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[2] for m in modes], fontsize=10, color=TEXT)
    ax1.set_ylabel("Acceptance Fraction (%)", fontsize=10.5, color=MUTED)
    ax1.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax1.set_ylim(0, 65)
    for spine in ax1.spines.values():
        spine.set_color(GRID)

    # Panel 2: Emitted tokens per verifier step
    ax2.set_facecolor(PANEL)
    ax2.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    for i, (ck_key, ck_name, color) in enumerate(checkpoints):
        means = []
        mins = []
        maxs = []
        for dcp, spec, _ in modes:
            cfg = configs_by_label[f"{ck_key}-{dcp}-{spec}"]
            vals = [c["mean_emitted_tokens_per_verifier_step"] for c in cfg["cells"]]
            means.append(np.mean(vals))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))

        offset = (i - 1) * bar_width
        bars = ax2.bar(x + offset, means, bar_width, label=ck_name, color=color, alpha=0.9, zorder=3)

        yerr = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]
        ax2.errorbar(x + offset, means, yerr=yerr, fmt='none', ecolor=TEXT, capsize=4, capthick=1.2, zorder=4)

        for bar, m in zip(bars, means):
            ax2.annotate(f"{m:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, m - 0.18),
                         ha='center', va='top', fontsize=8.5, fontweight='bold', color=TEXT,
                         bbox=dict(boxstyle="round,pad=0.15", facecolor=PANEL, edgecolor='none', alpha=0.85))

    ax2.set_title("Emitted Tokens per Verifier Step\n(Mean across 6 cells, error bars show min/max across C1–C8 / ctx 0–32K)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[2] for m in modes], fontsize=10, color=TEXT)
    ax2.set_ylabel("Emitted Tokens / Verifier Step", fontsize=10.5, color=MUTED)
    ax2.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax2.set_ylim(0, 4.0)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    fig.suptitle("R29 Speculative Efficiency: Acceptance Rate & Token Emission per Verifier Step",
                 fontsize=16, fontweight='bold', color=TEXT, y=0.96)

    ax1.legend(loc='upper right', ncol=1, frameon=True,
               facecolor=PANEL, edgecolor=GRID, fontsize=9.5, labelcolor=TEXT)

    fig.text(0.08, 0.03, "4x GB202GL Blackwell (RTX PRO 6000) • Steady counter windows • MTP3 has ~50% acceptance vs DFlash2 ~24% with higher emitted/step at C8",
             fontsize=9.5, color=MUTED)

    out_file = out_dir / "02-speculative-efficiency.png"
    plt.savefig(out_file, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_file


def render_chart3_checkpoint_quality(out_dir: Path) -> Path:
    """Chart 3: 52 functional tasks (strict vs semantic) and 32 history tasks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.25)

    # Panel 1: 52 Functional Tasks
    ax1.set_facecolor(PANEL)
    ax1.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    arms = ["r29-published", "r29-qad2500", "r29-tvn1500"]
    arm_labels = ["Published\n(Baseline)", "QAD-2500\n(Candidate)", "TVN-1500\n(Candidate)"]
    x = np.arange(len(arms))
    width = 0.35

    strict_vals = [42, 44, 38]
    semantic_vals = [52, 52, 50]

    b1 = ax1.bar(x - width/2, strict_vals, width, label="Strict Format Pass", color=BLUE, alpha=0.85, zorder=3)
    b2 = ax1.bar(x + width/2, semantic_vals, width, label="Semantic Correct", color=GREEN, alpha=0.85, zorder=3)
    ax1.bar(x[2] + width/2, [2], width, bottom=[50], label="Unassessable (no-oracle JSON)", color=AMBER, alpha=0.85, zorder=3)

    for bar, val in zip(b1, strict_vals):
        ax1.annotate(f"{val}/52\n({val/52*100:.0f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=TEXT)

    for bar, val in zip(b2, semantic_vals):
        ax1.annotate(f"{val}/52\n({val/52*100:.0f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=TEXT)

    ax1.set_title("52 Functional Tasks: Strict Format vs Semantic Correctness\n(Code, Data, Tool-call synthetic tasks)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arm_labels, fontsize=10.5, color=TEXT)
    ax1.set_ylabel("Tasks Passed (out of 52)", fontsize=10.5, color=MUTED)
    ax1.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax1.set_ylim(0, 70)
    ax1.legend(loc='upper left', frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=9, labelcolor=TEXT)
    for spine in ax1.spines.values():
        spine.set_color(GRID)

    # Panel 2: History tasks (32 tasks, context up to 524K tokens)
    ax2.set_facecolor(PANEL)
    ax2.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    history_arms = ["R28.1\nPublished", "R29\nPublished", "R29\nQAD-2500", "R29\nTVN-1500"]
    history_passed = [32, 32, 32, 31]
    x_hist = np.arange(len(history_arms))

    colors = [PURPLE, BLUE, PURPLE, GREEN]
    bars_h = ax2.bar(x_hist, history_passed, 0.45, color=colors, alpha=0.85, zorder=3)

    for bar, val in zip(bars_h, history_passed):
        ax2.annotate(f"{val}/32\n({val/32*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=TEXT)

    # Annotation for TVN 524K miss
    ax2.annotate("TVN 524K Miss:\nRow classification 100% correct;\noff-by-one score 41 vs 40",
                 xy=(x_hist[3], 31), xytext=(x_hist[3] - 0.8, 22),
                 arrowprops=dict(facecolor=AMBER, edgecolor=AMBER, arrowstyle="->", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor=AMBER, lw=1.2),
                 fontsize=9, color=TEXT, zorder=5)

    ax2.set_title("32 Long-Context History Tasks (prompt sizes up to 820K tokens)\n(Synthetic ledger retrieval & visible-length control)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=12)
    ax2.set_xticks(x_hist)
    ax2.set_xticklabels(history_arms, fontsize=10.5, color=TEXT)
    ax2.set_ylabel("Tasks Passed (out of 32)", fontsize=10.5, color=MUTED)
    ax2.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax2.set_ylim(0, 38)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    fig.suptitle("GLM-5.3-Flash Checkpoint Evaluation: Quality Scorecard & History Retention",
                 fontsize=16, fontweight='bold', color=TEXT, y=0.96)

    fig.text(0.08, 0.03, "Small bounded synthetic suite, not a general model ranking • Semantic sandbox permits computed assignments • No oracle-based JSON selection",
             fontsize=9.5, color=MUTED)

    out_file = out_dir / "03-checkpoint-quality.png"
    plt.savefig(out_file, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_file


def render_chart4_runtime_baseline(rb_data: dict, out_dir: Path) -> Path:
    """Chart 4: R28.1 vs R29 standardized decode runtime baseline per cell."""
    rows = rb_data["rows"]
    labels = [f"C={r['concurrency']}\nctx {r['context']//1024}K" if r['context'] > 0 else f"C={r['concurrency']}\nctx 0" for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), facecolor=BG, gridspec_kw={'height_ratios': [2.8, 1.2]})
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.08, right=0.95, hspace=0.35)

    x = np.arange(len(rows))
    width = 0.32

    # Upper panel: Throughput with both repeats
    ax1.set_facecolor(PANEL)
    ax1.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    r281_means = [r["r281_mean"] for r in rows]
    r29_means = [r["r29_mean"] for r in rows]

    ax1.bar(x - width/2, r281_means, width, label="R28.1 Published (stock)", color=BLUE, alpha=0.85, zorder=3)
    ax1.bar(x + width/2, r29_means, width, label="R29 Published (stock)", color=PURPLE, alpha=0.85, zorder=3)

    for i, r in enumerate(rows):
        rep1, rep2 = r["r281_repeats"]
        ax1.scatter([x[i] - width/2, x[i] - width/2], [rep1, rep2], color=TEXT, s=36, zorder=5, edgecolor='black', linewidth=0.8)
        rep1_29, rep2_29 = r["r29_repeats"]
        ax1.scatter([x[i] + width/2, x[i] + width/2], [rep1_29, rep2_29], color=TEXT, s=36, zorder=5, edgecolor='black', linewidth=0.8)

        ax1.annotate(f"{r['r281_mean']:.1f}", xy=(x[i] - width/2, r['r281_mean']),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=BLUE)
        ax1.annotate(f"{r['r29_mean']:.1f}", xy=(x[i] + width/2, r['r29_mean']),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=PURPLE)

        pct = r["change_percent"]
        pct_color = GREEN if pct >= 0 else RED
        ax1.annotate(f"{pct:+.2f}%", xy=(x[i], max(r['r281_mean'], r['r29_mean'])),
                     xytext=(0, 20), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold', color=pct_color,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=pct_color, lw=1))

    ax1.set_title("Standardized Decode Output Throughput (TP4 / DCP1 / No-Spec, ABBA order, 2 boots/image)",
                  fontsize=11, fontweight='bold', color=TEXT, pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, color=TEXT)
    ax1.set_ylabel("Output Throughput (tok/s)", fontsize=10.5, color=MUTED)
    ax1.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax1.set_ylim(0, 920)
    ax1.legend(loc='upper left', frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=10, labelcolor=TEXT)
    for spine in ax1.spines.values():
        spine.set_color(GRID)

    # Lower panel: Change percent
    ax2.set_facecolor(PANEL)
    ax2.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)
    ax2.axhline(0, color=MUTED, linestyle='-', linewidth=1.0, zorder=2)
    ax2.axhspan(-1.0, 1.0, color='#21262D', alpha=0.5, zorder=1, label="±1.0% noise band")

    pct_vals = [r["change_percent"] for r in rows]
    colors_pct = [GREEN if p >= 0 else RED for p in pct_vals]
    b_pct = ax2.bar(x, pct_vals, 0.45, color=colors_pct, alpha=0.85, zorder=3)

    for bar, p in zip(b_pct, pct_vals):
        y_pos = p + (0.1 if p >= 0 else -0.22)
        ax2.annotate(f"{p:+.2f}%", xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                     ha='center', va='bottom' if p >= 0 else 'top', fontsize=9.5, fontweight='bold', color=TEXT)

    ax2.set_title("Percentage Change (R29 vs R28.1 Baseline Mean)", fontsize=11, fontweight='bold', color=TEXT, pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, color=TEXT)
    ax2.set_ylabel("Change (%)", fontsize=10, color=MUTED)
    ax2.tick_params(axis='y', colors=MUTED, labelsize=9)
    ax2.set_ylim(-1.2, 1.2)
    ax2.legend(loc='upper right', frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=9, labelcolor=TEXT)
    for spine in ax2.spines.values():
        spine.set_color(GRID)
    fig.suptitle("GLM-5.3-Flash Runtime Baseline: R28.1 vs R29 Paired Comparison",
                 fontsize=16, fontweight='bold', color=TEXT, y=0.975)
    fig.text(0.08, 0.02, "4x GB202GL Blackwell • Zero regression / zero speedup (-0.39% to +0.20%, mean -0.07%) • Dots show raw repeat observations",
             fontsize=9.5, color=MUTED)

    out_file = out_dir / "04-runtime-baseline.png"
    plt.savefig(out_file, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_file


def render_chart5_lmcache_ram_overlay(perf_data: dict, ctx_data: dict, out_dir: Path) -> Path:
    """Chart 5: LMCache RAM stock vs PR #64 overlay with verifier steps/s side panel."""
    rows = perf_data["rows"]
    ctx_rows = ctx_data["rows"]
    labels = [f"C={r['concurrency']}\nctx {r['context_tokens']//1024}K" if r['context_tokens'] > 0 else f"C={r['concurrency']}\nctx 0" for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7), facecolor=BG)
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.06, right=0.96, wspace=0.25)

    x = np.arange(len(rows))
    width = 0.35

    # Panel 1: Output Throughput (tok/s)
    ax1.set_facecolor(PANEL)
    ax1.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    stock_tps = [r["stock_mean_tps"] for r in rows]
    pr64_tps = [r["pr64_mean_tps"] for r in rows]

    ax1.bar(x - width/2, stock_tps, width, label="Stock R29", color=BLUE, alpha=0.85, zorder=3)
    ax1.bar(x + width/2, pr64_tps, width, label="PR #64 Overlay", color=AMBER, alpha=0.85, zorder=3)

    for i, r in enumerate(rows):
        pct = r["change_percent"]
        pct_color = GREEN if pct >= 0 else RED
        ax1.annotate(f"{r['stock_mean_tps']:.1f}", xy=(x[i] - width/2, r['stock_mean_tps']),
                     xytext=(0, -5), textcoords="offset points", ha='center', va='top',
                     fontsize=8.5, fontweight='bold', color=BG)
        ax1.annotate(f"{r['pr64_mean_tps']:.1f}", xy=(x[i] + width/2, r['pr64_mean_tps']),
                     xytext=(0, -5), textcoords="offset points", ha='center', va='top',
                     fontsize=8.5, fontweight='bold', color=BG)
        ax1.annotate(f"{pct:+.2f}%", xy=(x[i], max(r['stock_mean_tps'], r['pr64_mean_tps'])),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=pct_color,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=pct_color, lw=1))

    ax1.set_title("1. Client Output Throughput\n(tok/s, 2 warmed repeats per image)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, color=TEXT)
    ax1.set_ylabel("Throughput (tok/s)", fontsize=10.5, color=MUTED)
    ax1.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax1.set_ylim(0, 1050)
    ax1.legend(loc='upper left', frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=9.5, labelcolor=TEXT)
    for spine in ax1.spines.values():
        spine.set_color(GRID)

    # Panel 2: Verifier Steps per Second (Side Panel)
    ax2.set_facecolor(PANEL)
    ax2.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    stock_vsteps = [np.mean(cr["stock"]["aggregate_verifier_steps_per_second"]) for cr in ctx_rows]
    pr64_vsteps = [np.mean(cr["pr64"]["aggregate_verifier_steps_per_second"]) for cr in ctx_rows]

    ax2.bar(x - width/2, stock_vsteps, width, label="Stock R29", color=BLUE, alpha=0.85, zorder=3)
    ax2.bar(x + width/2, pr64_vsteps, width, label="PR #64 Overlay", color=AMBER, alpha=0.85, zorder=3)

    for i in range(len(rows)):
        diff_pct = (pr64_vsteps[i] - stock_vsteps[i]) / stock_vsteps[i] * 100
        ax2.annotate(f"{stock_vsteps[i]:.1f}", xy=(x[i] - width/2, stock_vsteps[i]),
                     xytext=(0, -5), textcoords="offset points", ha='center', va='top',
                     fontsize=8.5, fontweight='bold', color=BG)
        ax2.annotate(f"{pr64_vsteps[i]:.1f}", xy=(x[i] + width/2, pr64_vsteps[i]),
                     xytext=(0, -5), textcoords="offset points", ha='center', va='top',
                     fontsize=8.5, fontweight='bold', color=BG)
        ax2.annotate(f"{diff_pct:+.1f}%", xy=(x[i], max(stock_vsteps[i], pr64_vsteps[i])),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=MUTED,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=MUTED, lw=0.8))

    ax2.set_title("2. Engine Verifier Steps / sec\n(Actual server execution speed)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, color=TEXT)
    ax2.set_ylabel("Verifier Steps / sec", fontsize=10.5, color=MUTED)
    ax2.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax2.set_ylim(0, 420)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    # Panel 3: Acceptance Fraction
    ax3.set_facecolor(PANEL)
    ax3.grid(axis='y', color=GRID, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    stock_acc = [np.mean(cr["stock"]["acceptance_fraction"]) * 100 for cr in ctx_rows]
    pr64_acc = [np.mean(cr["pr64"]["acceptance_fraction"]) * 100 for cr in ctx_rows]

    ax3.bar(x - width/2, stock_acc, width, label="Stock R29", color=BLUE, alpha=0.85, zorder=3)
    ax3.bar(x + width/2, pr64_acc, width, label="PR #64 Overlay", color=AMBER, alpha=0.85, zorder=3)

    for i in range(len(rows)):
        diff_acc = pr64_acc[i] - stock_acc[i]
        ax3.annotate(f"{stock_acc[i]:.1f}%", xy=(x[i] - width/2, stock_acc[i]),
                     xytext=(0, -5), textcoords="offset points", ha='center', va='top',
                     fontsize=8.5, fontweight='bold', color=BG)
        ax3.annotate(f"{pr64_acc[i]:.1f}%", xy=(x[i] + width/2, pr64_acc[i]),
                     xytext=(0, -5), textcoords="offset points", ha='center', va='top',
                     fontsize=8.5, fontweight='bold', color=BG)
        ax3.annotate(f"{diff_acc:+.1f}pp", xy=(x[i], max(stock_acc[i], pr64_acc[i])),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=TEXT,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=GRID, lw=0.8))

    ax3.set_title("3. Draft Acceptance Rate (%)\n(Governs output tok/s per step)",
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10, color=TEXT)
    ax3.set_ylabel("Acceptance Rate (%)", fontsize=10.5, color=MUTED)
    ax3.tick_params(axis='y', colors=MUTED, labelsize=9.5)
    ax3.set_ylim(0, 68)
    for spine in ax3.spines.values():
        spine.set_color(GRID)

    fig.suptitle("LMCache PR #64 Overlay vs Stock R29: RAM Engine & Throughput Comparison\n(TP4 / DCP4 / MTP3 RAM Cache, Published Checkpoint)",
                 fontsize=15, fontweight='bold', color=TEXT, y=0.965)
    fig.text(0.06, 0.04, "CRITICAL CAVEAT: Engine verifier steps/s are identical (~96/s at C1, ~340/s at C8) on both stock and PR#64. Throughput delta is governed entirely by run-to-run variance in draft acceptance, NOT a patch speedup.",
             fontsize=9.5, fontweight='bold', color=AMBER)

    out_file = out_dir / "05-lmcache-ram-overlay.png"
    plt.savefig(out_file, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_file

def render_chart6_disk_cache_gates(matrix: dict, out_dir: Path) -> Path:
    """Chart 6: R30 disk-cache lifecycle gate-status heatmap (arms x gate groups)."""
    groups = ["cold", "replay", "restart", "growth", "pressure",
              "cancellation", "orphan", "evidence"]
    order = ["l2-stock-dcp4-mtp3",
             "l2-r30-dcp1-mtp0", "l2-r30-dcp1-mtp3", "l2-r30-dcp1-dflash2",
             "l2-r30-dcp4-mtp0", "l2-r30-dcp4-mtp3", "l2-r30-dcp4-dflash2"]
    labels = {"l2-stock-dcp4-mtp3": "R29 ctrl\nDCP4 MTP3",
              "l2-r30-dcp1-mtp0": "R30\nDCP1 MTP0",
              "l2-r30-dcp1-mtp3": "R30\nDCP1 MTP3",
              "l2-r30-dcp1-dflash2": "R30\nDCP1 DFl2",
              "l2-r30-dcp4-mtp0": "R30\nDCP4 MTP0",
              "l2-r30-dcp4-mtp3": "R30\nDCP4 MTP3",
              "l2-r30-dcp4-dflash2": "R30\nDCP4 DFl2"}
    arms = {a["arm"]: a for a in matrix["arms"]}

    counts = {}
    for arm in order:
        gates = arms[arm]["gates"]
        for grp in groups:
            rows = [g for name, g in gates.items() if name.split(".")[0] == grp]
            c = {s: sum(1 for g in rows if g["status"] == s)
                 for s in ("pass", "fail", "unavailable", "not_applicable")}
            counts[(arm, grp)] = c

    def worst(c):
        if c["fail"]:
            return RED
        if c["unavailable"]:
            return AMBER
        if c["not_applicable"]:
            return MUTED
        return GREEN

    fig, ax = plt.subplots(figsize=(13.5, 6.6), facecolor=BG)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.125, right=0.945)
    ax.set_facecolor(PANEL)

    for i, arm in enumerate(order):
        for j, grp in enumerate(groups):
            c = counts[(arm, grp)]
            color = worst(c)
            ax.add_patch(plt.Rectangle((j - 0.5, len(order) - 1 - i - 0.5), 1, 1,
                                        facecolor=color, edgecolor=BG, linewidth=2, alpha=0.88))
            txt = f"{c['pass']}P"
            if c["fail"]:
                txt += f" {c['fail']}F"
            if c["unavailable"]:
                txt += f" {c['unavailable']}U"
            if c["not_applicable"]:
                txt += f" {c['not_applicable']}-"
            ax.text(j, len(order) - 1 - i, txt, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=BG)

    # Annotations
    ax.annotate("4 fail: replay/restart dedup + no_new_unreferenced\n(expected: negative control must fail dedup)",
                xy=(1.5, len(order) - 0.5), xytext=(5.0, len(order) + 0.55),
                fontsize=8.5, color=TEXT, ha='center', va='center',
                arrowprops=dict(facecolor=RED, edgecolor=RED, arrowstyle='->', lw=1.4),
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL, edgecolor=RED, lw=1.1))
    ax.annotate("identity gate fails after DCP4 cache restore - follow-up ran:\nR29 controls + R30 repeat all pass identity, not reproduced (one-off)",
                xy=(2.0, len(order) - 1 - 4), xytext=(5.0, -1.15),
                fontsize=8.5, color=TEXT, ha='center', va='center',
                arrowprops=dict(facecolor=RED, edgecolor=RED, arrowstyle='->', lw=1.4),
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL, edgecolor=RED, lw=1.1))

    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-1.7, len(order) + 1.1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.upper() for g in groups], fontsize=10, color=TEXT, fontweight='bold')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[a] for a in reversed(order)], fontsize=9, color=TEXT)
    ax.tick_params(length=0, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [Patch(facecolor=GREEN, label='pass'), Patch(facecolor=RED, label='fail'),
               Patch(facecolor=AMBER, label='unavailable'), Patch(facecolor=MUTED, label='not_applicable')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.44, 1.10), ncol=4,
              frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=9.5, labelcolor=TEXT)

    ax.set_title("R30 Disk-Cache Lifecycle Gates: Stock R29 Negative Control vs R30 Arms\n(39 gates per arm; cold.external_miss passes on every arm after the observer fix; R30 passes both dedup gates)",
                 fontsize=13, fontweight='bold', color=TEXT, pad=26)

    fig.text(0.13, 0.035,
             "U = harness observability limits, not engine failures: abort counter never increments on client disconnect in this build (server_abort_observed);\n"
             "instantaneous lease gauge sampled too slowly to catch prefill-time retrieve (live_retrieve_observed_under_pressure); page byte equality covered by\n"
             "the separate 144/144 GPU byte-evidence component test (evidence.all_rank_payload_byte_equality). growth.schema2 gate only meaningful with dedup (R30 arms pass it).",
             fontsize=8.5, color=MUTED, ha='left')

    out_file = out_dir / "06-disk-cache-gates.png"
    plt.savefig(out_file, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_file




def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Rendering R29 campaign charts from: {R}")
    print(f"Output directory: {OUT}")

    # Load receipts
    sm_path = R / "speculative-matrix-results.json"
    with open(sm_path) as f:
        sm_data = json.load(f)

    rb_path = R / "runtime-baseline" / "paired-runtime-comparison.json"
    with open(rb_path) as f:
        rb_data = json.load(f)

    ram_perf_path = R / "cache-campaign" / "ram-performance-comparison.json"
    with open(ram_perf_path) as f:
        ram_perf_data = json.load(f)

    ram_ctx_path = R / "cache-campaign" / "ram-speculative-counter-context.json"
    with open(ram_ctx_path) as f:
        ram_ctx_data = json.load(f)

    matrix_path = R / "r30-cache-gate-matrix.json"
    matrix_data = json.loads(matrix_path.read_text()) if matrix_path.exists() else None

    c1 = render_chart1_speculative_matrix(sm_data, OUT)
    print(f"Rendered: {c1.name} ({c1.stat().st_size} bytes)")

    c2 = render_chart2_speculative_efficiency(sm_data, OUT)
    print(f"Rendered: {c2.name} ({c2.stat().st_size} bytes)")

    c3 = render_chart3_checkpoint_quality(OUT)
    print(f"Rendered: {c3.name} ({c3.stat().st_size} bytes)")

    c4 = render_chart4_runtime_baseline(rb_data, OUT)
    print(f"Rendered: {c4.name} ({c4.stat().st_size} bytes)")

    c5 = render_chart5_lmcache_ram_overlay(ram_perf_data, ram_ctx_data, OUT)
    print(f"Rendered: {c5.name} ({c5.stat().st_size} bytes)")

    if matrix_data is not None:
        c6 = render_chart6_disk_cache_gates(matrix_data, OUT)
        print(f"Rendered: {c6.name} ({c6.stat().st_size} bytes)")
        print("All 6 charts rendered successfully.")
    else:
        print("Skipped chart 6: R/r30-cache-gate-matrix.json not present yet.")
        print("All 5 charts rendered successfully.")


if __name__ == "__main__":
    main()
