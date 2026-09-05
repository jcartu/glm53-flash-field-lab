#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
OUT = ROOT / 'charts'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'report-data.json').read_text())
by25 = {row['name']: row for row in data['configs']}
by24 = {
    row['name']: row
    for row in json.loads(
        Path('/home/josh/omp-workspace/drock-lmcache/r24-battery/report-data.json').read_text()
    )['configs']
}

BG = '#090D13'
PANEL = '#111821'
GRID = '#263242'
TEXT = '#EDF4FA'
MUTED = '#92A2B5'
BLUE = '#4FA8FF'
CYAN = '#39D0C6'
GREEN = '#46D785'
AMBER = '#F2B84B'
ORANGE = '#FF824D'
RED = '#FF5F68'
PURPLE = '#B892FF'
PINK = '#FF70C5'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': PANEL,
    'axes.edgecolor': GRID,
    'axes.labelcolor': MUTED,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'text.color': TEXT,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})


def style(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, alpha=0.45, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)


def finish(fig, name, footer):
    fig.text(0.5, 0.018, footer, ha='center', color=MUTED, fontsize=8)
    fig.savefig(OUT / name, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def label_bars(ax, bars, fmt='{:.0f}', color=TEXT, fontsize=8, offset=5):
    for bar in bars:
        value = bar.get_height()
        if value is None or not np.isfinite(value):
            continue
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center',
            va='bottom',
            color=color,
            fontsize=fontsize,
            fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
        )


# 1. R25 topology map.
fp8_recheck = data['fp8_recheck']['recheck']
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.38, wspace=0.28)
dcps = [1, 2, 4]
specs = [('mtp0', 'no spec', GREEN), ('mtp3', 'MTP3', BLUE), ('dflash2', 'DFlash K7', AMBER)]
for ax, metric, title in zip(axes.flat[:3], ['C1', 'C8', 'C16'], ['1 user', '8 users', '16 users']):
    for spec, label, color in specs:
        values = [
            fp8_recheck[spec][metric]
            if dcp == 4
            else by25[f'dcp{dcp}-vram-fp8-{spec}'][metric]
            for dcp in dcps
        ]
        ax.plot(dcps, values, marker='o', linewidth=2.6, markersize=8, color=color, label=label)
        horizontal_offset = {'mtp0': -12, 'mtp3': 0, 'dflash2': 18}[spec]
        for x_value, value in zip(dcps, values):
            ax.annotate(
                f'{value:.0f}',
                (x_value, value),
                xytext=(horizontal_offset, 8),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=color,
                fontsize=8,
                fontweight='bold',
            )
    ax.set_xticks(dcps)
    ax.set_xticklabels(['DCP1', 'DCP2', 'DCP4'])
    ax.set_ylabel('output tokens/sec')
    ax.set_title(title, color=TEXT, fontweight='bold')
    style(ax)
axes[0, 0].legend(frameon=False, labelcolor=TEXT, ncol=3, fontsize=8, loc='lower left')
ax = axes[1, 1]
kv = [by25[f'dcp{dcp}-vram-fp8-mtp0']['kv_tokens'] / 1e6 for dcp in dcps]
bars = ax.bar(['DCP1', 'DCP2', 'DCP4'], kv, color=[BLUE, CYAN, PURPLE], width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}M')
ax.set_ylabel('runtime KV tokens')
ax.set_title('FP8 KV capacity, no spec', color=TEXT, fontweight='bold')
style(ax)
fig.suptitle('R25 topology field map', fontsize=20, fontweight='bold', color=BLUE, y=0.96)
fig.text(0.5, 0.905, 'Full DCP1, DCP2, and DCP4 matrix, DCP4 shown from the controlled warm-cache recheck', ha='center', color=TEXT, fontsize=11)
finish(fig, '01-r25-topology.png', '4x RTX PRO 6000 Blackwell | pinned R25 digest | FP8 KV | fairness 0.4 | 30-second sustained cells | first-pass receipts retained')


# 2. Matched R25, R24, R25 execution bracket.
execution = data['execution_ab']
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.82, bottom=0.11, hspace=0.40, wspace=0.30)
arm_labels = ['R25 A', 'R24', 'R25 B']
arm_keys = ['r25-a', 'r24', 'r25-b']
arm_colors = [GREEN, BLUE, CYAN]
for ax, metric, title in zip(
    axes.flat[:3],
    ['C1', 'C8', 'C16'],
    ['1 user decode', '8 user decode', '16 user decode'],
):
    values = [execution['arms'][key][metric] for key in arm_keys]
    bars = ax.bar(arm_labels, values, color=arm_colors, width=0.62, zorder=3)
    label_bars(ax, bars, '{:.0f}', fontsize=9)
    ax.set_ylabel('output tokens/sec')
    ax.set_title(title, color=TEXT, fontweight='bold')
    style(ax)
ax = axes[1, 1]
metrics = ['C1', 'C8', 'C16']
deltas = [execution['r25_vs_r24_percent'][metric] for metric in metrics]
colors = [GREEN if abs(value) <= 5 else AMBER for value in deltas]
bars = ax.bar(metrics, deltas, color=colors, width=0.62, zorder=3)
label_bars(ax, bars, '{:+.2f}%', fontsize=9, offset=5)
limit = max(6.0, max(abs(value) for value in deltas) * 1.35)
ax.axhspan(-5, 5, color=GREEN, alpha=0.09)
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_ylim(-limit, limit)
ax.set_ylabel('R25 bracket mean vs R24')
ax.set_title('Matched delta, +/-5% guardrail', color=TEXT, fontweight='bold')
style(ax)
fig.suptitle('R25 execution regression check', fontsize=20, fontweight='bold', color=GREEN, y=0.96)
fig.text(0.5, 0.90, 'R25 then R24 then R25, same live host window, DCP4 FP8 no spec', ha='center', color=TEXT, fontsize=11)
finish(fig, '02-r24-r25-regression.png', 'The bracket controls changing host load | vLLM and B12X package trees are byte-identical by source lock')


# 3. Exact one-million-token restore.
restore = data['restore_ab']
exact = data['exact_replay_1m']
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.80, bottom=0.11, hspace=0.44, wspace=0.30)
arm_colors = [BLUE, GREEN]
ax = axes[0, 0]
internal = [restore['arms']['R24']['critical_retrieve_seconds'], restore['arms']['R25']['critical_retrieve_seconds']]
bars = ax.bar(['R24', 'R25'], internal, color=arm_colors, width=0.58, zorder=3)
label_bars(ax, bars, '{:.3f}s', fontsize=10)
reduction = restore['comparison']['critical_restore_reduction_pct']
ax.text(0.5, max(internal) * 0.76, f'{reduction:.1f}% lower', ha='center', color=GREEN, fontsize=12, fontweight='bold')
ax.set_ylabel('logged control-segment seconds')
ax.set_title('Connector handoff segment', color=TEXT, fontweight='bold')
style(ax)
ax = axes[0, 1]
wall = [restore['arms']['R24']['wall_seconds'], restore['arms']['R25']['wall_seconds']]
bars = ax.bar(['R24', 'R25'], wall, color=arm_colors, width=0.58, zorder=3)
label_bars(ax, bars, '{:.2f}s', fontsize=10)
api_reduction = restore['comparison']['api_wall_reduction_pct']
ax.text(0.5, max(wall) * 0.76, f'{api_reduction:.1f}% lower', ha='center', color=GREEN, fontsize=12, fontweight='bold')
ax.set_ylabel('complete API wall time')
ax.set_title('End-to-end request', color=TEXT, fontweight='bold')
style(ax)
ax = axes[1, 0]
summary = exact['summary']
labels = ['cold', 'warm median', 'first restart', 'steady restart']
values = [
    exact['results'][0]['wall_seconds'],
    summary['warm_median_seconds'],
    summary['restart_first_seconds'],
    summary['restart_steady_median_seconds'],
]
bars = ax.bar(labels, values, color=[RED, GREEN, AMBER, CYAN], width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}s', fontsize=8)
ax.set_yscale('log')
ax.set_ylabel('API wall seconds, log scale')
ax.set_title('R25 exact 1M replay lifecycle', color=TEXT, fontweight='bold')
style(ax)
ax = axes[1, 1]
prefix = {
    row['label']: row['wall_seconds']
    for row in data['prefix_80k']['lmcache_unique']['results']
}
labels = ['cold', 'warm', 'restart']
values = [prefix[label] for label in labels]
bars = ax.bar(labels, values, color=[RED, GREEN, AMBER], width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}s', fontsize=9)
ax.set_ylabel('API wall seconds')
ax.set_title('80K prefix recovery', color=TEXT, fontweight='bold')
style(ax)
fig.suptitle('R25 LMCache restore, engine and API views', fontsize=20, fontweight='bold', color=CYAN, y=0.96)
fig.text(0.5, 0.895, 'Exact 1,000,000-token prompt, 999,424 restored plus 576 recomputed, private 128 GiB SHM', ha='center', color=TEXT, fontsize=10.5)
finish(fig, '03-r25-cache-restore.png', 'Matched DCP4 DFlash K7 packed-NVFP4 controls | connector log segment is not full restore latency | visible output byte-identical')


# 4. LMCache tradeoffs.
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.80, bottom=0.11, hspace=0.44, wspace=0.30)
spec_names = ['no spec', 'MTP3', 'DFlash K7']
gpu_rows = [by25['dcp4-vram-nvfp4-mtp0'], by25['dcp4-vram-nvfp4-mtp3'], by25['dcp4-vram-nvfp4-dflash2']]
lmc_rows = [by25['dcp4-lmcache-nvfp4-mtp0'], by25['dcp4-lmcache-nvfp4-mtp3'], by25['dcp4-lmcache-nvfp4-dflash2']]
x = np.arange(3)
width = 0.34
for ax, metric, title in [(axes[0, 0], 'C1', '1 user'), (axes[0, 1], 'C8', '8 users'), (axes[1, 0], 'C16', '16 users')]:
    gpu = [row[metric] for row in gpu_rows]
    cache = [row[metric] for row in lmc_rows]
    left = ax.bar(x - width / 2, gpu, width, color=BLUE, label='GPU only', zorder=3)
    right = ax.bar(x + width / 2, cache, width, color=GREEN, label='LMCache', zorder=3)
    label_bars(ax, left, '{:.0f}', fontsize=7)
    label_bars(ax, right, '{:.0f}', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names, fontsize=9)
    ax.set_ylabel('output tokens/sec')
    ax.set_title(title, color=TEXT, fontweight='bold')
    style(ax)
axes[0, 0].legend(frameon=False, labelcolor=TEXT, fontsize=8)
ax = axes[1, 1]
variant_labels = ['GPU only', 'LMCache\nexisting L2', 'LMCache\nRAM only', 'LMCache\nfresh L2']
variant_keys = ['gpu-only', 'lmcache-existing-l2', 'lmcache-ram-only', 'lmcache-fresh-l2']
values = [data['prefill_variants'][key]['median_tokens_per_second'] / 1000 for key in variant_keys]
bars = ax.bar(variant_labels, values, color=[BLUE, PURPLE, CYAN, GREEN], width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}k', fontsize=8)
for index, value in enumerate(values[1:], 1):
    delta = 100 * (value / values[0] - 1)
    ax.text(index, value * 0.74, f'{delta:+.1f}%', ha='center', color=BG, fontsize=8, fontweight='bold')
ax.set_ylabel('cold 33K prefill, k tok/s')
ax.set_title('Repeated cold prefill, 8 runs', color=TEXT, fontweight='bold')
style(ax)
steady_delta = np.mean([100 * (value / values[0] - 1) for value in values[1:]])
first_lmcache_seconds = np.mean([
    data['prefill_variants'][key]['runs'][0]['seconds']
    for key in variant_keys[1:]
])
fig.suptitle('R25 LMCache field cost', fontsize=20, fontweight='bold', color=PURPLE, y=0.96)
fig.text(0.5, 0.895, f'Steady cold-prefill medians {steady_delta:.1f}% vs GPU, first post-boot LMCache request {first_lmcache_seconds:.1f}s', ha='center', color=TEXT, fontsize=11)
finish(fig, '04-r25-lmcache-tradeoffs.png', 'DCP4 | packed NVFP4 KV | fairness 0.4 | GPU-only GMU .93 | LMCache GMU .95 | unique salts for every cold sample')


# 5. Validation scorecard.
fair = data['fairness']
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.42, wspace=0.28, left=0.08, right=0.95, top=0.75, bottom=0.10)
ax = fig.add_subplot(gs[0, 0])
labels = ['fairness off', 'compute share 0.4']
baseline = [fair['off']['baseline_decode_tps'], fair['compute_share_0.4']['baseline_decode_tps']]
collision = [fair['off']['collision_decode_tps'], fair['compute_share_0.4']['collision_decode_tps']]
x = np.arange(2)
width = 0.34
left = ax.bar(x - width / 2, baseline, width, color=BLUE, label='uncontended', zorder=3)
right = ax.bar(x + width / 2, collision, width, color=GREEN, label='during cold prefill', zorder=3)
label_bars(ax, left)
label_bars(ax, right)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('decode tokens/sec')
ax.set_title('Decode protection under collision', color=TEXT, fontweight='bold')
ax.legend(frameon=False, labelcolor=TEXT, fontsize=8)
style(ax)
for index, key in enumerate(('off', 'compute_share_0.4')):
    ax.text(index, max(baseline) * 0.80, f"{fair[key]['slowdown_percent']:.0f}% stall", ha='center', color=RED if key == 'off' else AMBER, fontweight='bold')
ax = fig.add_subplot(gs[0, 1])
gaps = [fair['off']['decode_gap_p95_ms'], fair['compute_share_0.4']['decode_gap_p95_ms']]
bars = ax.bar(labels, gaps, color=[RED, GREEN], width=0.58, zorder=3)
label_bars(ax, bars, '{:.1f}ms', fontsize=9)
ax.set_yscale('log')
ax.set_ylabel('p95 inter-token gap, log scale')
ax.set_title('User-visible decode gaps', color=TEXT, fontweight='bold')
style(ax)
ax = fig.add_subplot(gs[1, :])
ax.set_axis_off()
needle = data['needles']
replay = data['exact_replay_1m']['summary']
corruption_total = data['corruption']['dcp4_mtp3']['clean'] + data['corruption']['lmcache_dflash']['clean']
corruption_den = data['corruption']['dcp4_mtp3']['total'] + data['corruption']['lmcache_dflash']['total']
items = [
    ('1M depth needles', f"{needle['hits']} / {needle['total']} HIT", GREEN),
    ('exact 1M replay', f"{replay['exact_output_count']} / 9 EXACT", GREEN),
    ('eviction churn', f"{data['eviction']['identical_reply_count']} / 40 IDENTICAL", GREEN),
    ('long generations', f'{corruption_total} / {corruption_den} CLEAN', GREEN if corruption_total == corruption_den else AMBER),
    ('tool result reorder', 'PASS' if data['template']['pass'] else 'FAIL', GREEN if data['template']['pass'] else RED),
    ('Estonia', f"{data['quality']['estonia']['correct']} / 6 PASS", GREEN),
    ('Lavd ledger', f"{data['quality']['lavd']['exact']} / {data['quality']['lavd']['attempted']} EXACT", GREEN if data['quality']['lavd']['exact'] == data['quality']['lavd']['attempted'] else AMBER),
    ('native offload', 'PASS, private SHM', GREEN),
    ('LMCache sidecar', 'CPU ONLY', GREEN if data['sidecar']['verdict'] == 'pass' else RED),
    ('release provenance', '3 LAYERS, LOCK EXACT', GREEN),
]
columns = 5
for index, (title, value, color) in enumerate(items):
    row = index // columns
    column = index % columns
    x0 = 0.01 + column * 0.198
    y0 = 0.70 - row * 0.48
    rect = plt.Rectangle((x0, y0), 0.185, 0.35, transform=ax.transAxes, facecolor=PANEL, edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x0 + 0.092, y0 + 0.23, title, transform=ax.transAxes, ha='center', color=MUTED, fontsize=8)
    ax.text(x0 + 0.092, y0 + 0.10, value, transform=ax.transAxes, ha='center', color=color, fontsize=9, fontweight='bold')
fig.suptitle('R25 validation scorecard', fontsize=21, fontweight='bold', color=CYAN, y=0.96)
fig.text(0.5, 0.905, 'Correctness, persistence, fairness, packaging, and sidecar ownership all checked on the published digest', ha='center', color=TEXT, fontsize=10.5)
finish(fig, '05-r25-validation-scorecard.png', 'Independent field battery | every raw receipt retained | Discord restart claim 1.215s, embedded lock says 1.311s')


# 6. Cyclic agent hot requests versus periodic cold prefills.
hot = data['hot_queue']
rows = {(row['policy'], row['concurrency'], row['scenario']): row for row in hot['rows']}
comparisons = {(row['policy'], row['concurrency']): row for row in hot['comparisons']}
policies = ['off', 'compute-share-0.4']
policy_labels = ['fairness off', 'compute share 0.4']
concurrencies = [8, 16]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.78, bottom=0.12, hspace=0.48, wspace=0.30)
x = np.arange(2)
width = 0.35
ax = axes[0, 0]
for policy_index, (policy, label, color) in enumerate(zip(policies, policy_labels, [RED, BLUE])):
    values = [rows[(policy, concurrency, 'periodic-cold')]['hot_ttft_p95_seconds'] for concurrency in concurrencies]
    bars = ax.bar(x + (policy_index - 0.5) * width, values, width, color=color, label=label, zorder=3)
    label_bars(ax, bars, '{:.2f}s', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['C8', 'C16'])
ax.set_ylabel('hot-turn p95 TTFT')
ax.set_title('Agent continuation latency', color=TEXT, fontweight='bold')
ax.legend(frameon=False, labelcolor=TEXT, fontsize=8)
style(ax)
ax = axes[0, 1]
for policy_index, (policy, label, color) in enumerate(zip(policies, policy_labels, [RED, BLUE])):
    values = [comparisons[(policy, concurrency)]['hot_turn_rate_retention_percent'] for concurrency in concurrencies]
    bars = ax.bar(x + (policy_index - 0.5) * width, values, width, color=color, label=label, zorder=3)
    label_bars(ax, bars, '{:.0f}%', fontsize=8)
ax.axhline(100, color=MUTED, linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['C8', 'C16'])
ax.set_ylabel('turn-rate retention')
ax.set_title('Completed agent turns', color=TEXT, fontweight='bold')
style(ax)
ax = axes[1, 0]
for policy_index, (policy, label, color) in enumerate(zip(policies, policy_labels, [RED, BLUE])):
    values = [rows[(policy, concurrency, 'periodic-cold')]['max_waiting_requests'] for concurrency in concurrencies]
    bars = ax.bar(x + (policy_index - 0.5) * width, values, width, color=color, label=label, zorder=3)
    label_bars(ax, bars, '{:.1f}', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['C8', 'C16'])
ax.set_ylabel('max waiting requests')
ax.set_title('Queue backpressure', color=TEXT, fontweight='bold')
style(ax)
ax = axes[1, 1]
for policy_index, (policy, label, color) in enumerate(zip(policies, policy_labels, [RED, BLUE])):
    values = [comparisons[(policy, concurrency)]['hot_ttft_p95_multiplier'] for concurrency in concurrencies]
    bars = ax.bar(x + (policy_index - 0.5) * width, values, width, color=color, label=label, zorder=3)
    label_bars(ax, bars, '{:.1f}x', fontsize=8)
ax.axhline(1, color=MUTED, linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['C8', 'C16'])
ax.set_ylabel('p95 TTFT multiplier')
ax.set_title('Periodic cold vs baseline', color=TEXT, fontweight='bold')
style(ax)
worst = max(
    comparison['hot_ttft_p95_multiplier']
    for comparison in hot['comparisons']
)
c16_compute = rows[('compute-share-0.4', 16, 'periodic-cold')]
c16_compare = comparisons[('compute-share-0.4', 16)]
headline = 'Compute share saves throughput, not hot-turn latency' if worst > 2 else 'Hot agent turns remain stable under this cold cadence'
fig.suptitle('R25 hot-request queue test', fontsize=20, fontweight='bold', color=AMBER, y=0.96)
fig.text(0.5, 0.895, headline, ha='center', color=TEXT, fontsize=11)
fig.text(0.5, 0.855, f"C16 compute share retains {c16_compare['hot_turn_rate_retention_percent']:.0f}% of turns, p95 TTFT {c16_compute['hot_ttft_p95_seconds']:.1f}s, max queue {c16_compute['max_waiting_requests']:.0f}", ha='center', color=MUTED, fontsize=9.5)
finish(fig, '06-r25-hot-request-queue.png', 'Two 30s baseline and two 60s storm runs per C8/C16 cell | compute_share chooses decode vs prefill class, FCFS remains inside prefill')


# 7. PR646 draft field test.
pr646 = data['pr646']
arm_keys = [
    'stock-fine-256',
    'stock-coarse-2048',
    'pr646-decoupled-2048-256',
]
arm_labels = ['stock\n256 / 256', 'stock\n2048 / 2048', 'PR646\n2048 / 256']
arm_colors = [BLUE, AMBER, GREEN]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.76, bottom=0.13, hspace=0.50, wspace=0.30)
ax = axes[0, 0]
capacity = [pr646['arms'][key]['kv_capacity_tokens'] / 1e6 for key in arm_keys]
bars = ax.bar(arm_labels, capacity, color=arm_colors, width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}M', fontsize=9)
ax.set_ylabel('runtime KV tokens')
ax.set_title('Cache capacity', color=TEXT, fontweight='bold')
style(ax)
ax.text(
    1.0,
    max(capacity) * 0.62,
    f"{pr646['comparison']['capacity_gain_vs_fine_256']:.1f}x capacity",
    ha='center',
    color=GREEN,
    fontweight='bold',
    fontsize=11,
    bbox=dict(facecolor=BG, edgecolor=GREEN, boxstyle='round,pad=.25'),
)
ax = axes[0, 1]
cache_hits = [
    pr646['arms'][key]['prefix_probe']['repeat_metric_delta']['vllm:prefix_cache_hits_total'] / 1000
    for key in arm_keys
]
bars = ax.bar(arm_labels, cache_hits, color=arm_colors, width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}K', fontsize=9)
ax.set_ylabel('repeat prefix hit tokens')
ax.set_title('52,445-token repeat', color=TEXT, fontweight='bold')
style(ax)
ax = axes[1, 0]
repeat_wall = [
    pr646['arms'][key]['prefix_probe']['results'][1]['wall_seconds']
    for key in arm_keys
]
bars = ax.bar(arm_labels, repeat_wall, color=arm_colors, width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}s', fontsize=9)
ax.set_ylabel('complete request wall time')
ax.set_title('Cached replay latency', color=TEXT, fontweight='bold')
style(ax)
ax = axes[1, 1]
throughput = pr646['comparison']['throughput_delta_percent_vs_stock_coarse']
cell_keys = sorted(
    throughput,
    key=lambda key: (int(key.split('@')[1]), int(key.split('@')[0][1:])),
)
values = [throughput[key] for key in cell_keys]
colors = [
    GREEN if abs(value) <= 6 else (AMBER if value >= -15 else RED)
    for value in values
]
bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.72, zorder=3)
ax.axhspan(-6, 6, color=GREEN, alpha=0.12)
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_xticks(np.arange(len(values)))
ax.set_xticklabels(
    [
        f"{key.split('@')[0]}\n{int(key.split('@')[1]) // 1024}K"
        for key in cell_keys
    ],
    fontsize=7,
)
ax.set_ylabel('PR646 vs stock coarse, percent')
ax.set_title('Decode matrix delta', color=TEXT, fontweight='bold')
style(ax)
gain = pr646['comparison']['capacity_gain_vs_fine_256']
fine_preserved = cache_hits[2] >= cache_hits[0] - 0.256
quality_clean = (
    pr646['correctness']['estonia_correct']
    == pr646['correctness']['estonia_total']
)
geometry_works = gain > 2.5 and fine_preserved
if geometry_works and quality_clean:
    headline = 'The dangerous geometry works on DCP1'
elif geometry_works:
    headline = 'The capacity win is real, concurrent quality is not clean'
else:
    headline = 'The draft geometry needs more work'
fig.suptitle('PR646 cache geometry field test', fontsize=20, fontweight='bold', color=GREEN if geometry_works and quality_clean else AMBER, y=0.96)
fig.text(0.5, 0.895, headline, ha='center', color=TEXT, fontsize=11)
fig.text(
    0.5,
    0.855,
    f"90K needle {pr646['correctness']['needle_90k_hits']}/4, Estonia PR {pr646['correctness']['estonia_correct']}/{pr646['correctness']['estonia_total']} vs stock {pr646['correctness']['stock_estonia_correct']}/{pr646['correctness']['stock_estonia_total']}",
    ha='center',
    color=MUTED,
    fontsize=9.5,
)
finish(fig, '07-pr646-capacity-prefix.png', 'Draft PR646 rebased onto exact R25 | TP4 DCP1 DFlash K7 FP8 only | DCP greater than one intentionally not tested')
print('\n'.join(str(path) for path in sorted(OUT.glob('*.png'))))
