#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r24-battery')
OUT = ROOT / 'charts'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'report-data.json').read_text())
by = {row['name']: row for row in data['configs']}
sp = data['special']

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


def finish(fig, name, subtitle):
    fig.text(0.5, 0.018, subtitle, ha='center', color=MUTED, fontsize=8)
    fig.savefig(OUT / name, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def style(ax):
    ax.grid(axis='y', color=GRID, alpha=0.45, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)


def label_bars(ax, bars, fmt='{:.0f}', color=TEXT, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                    color=color, fontsize=fontsize, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground=BG)])


# 1. DCP scaling dashboard
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.38, wspace=0.28)
dcps = [1, 2, 4]
specs = [('mtp0', 'no spec', GREEN), ('mtp3', 'MTP3', BLUE), ('dflash2', 'DFlash K7', AMBER)]
for ax, metric, title in zip(axes.flat[:3], ['C1', 'C8', 'C16'], ['1 user', '8 users', '16 users']):
    for spec, label, color in specs:
        xs, ys = [], []
        for dcp in dcps:
            row = by.get(f'dcp{dcp}-vram-fp8-{spec}')
            if row and row.get(metric) is not None:
                xs.append(dcp); ys.append(row[metric])
        ax.plot(xs, ys, marker='o', linewidth=2.6, markersize=8, color=color, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f'{y:.0f}', (x, y), xytext=(0, 8), textcoords='offset points',
                        ha='center', color=color, fontsize=8, fontweight='bold')
    ax.set_xticks(dcps); ax.set_xticklabels(['DCP1', 'DCP2', 'DCP4'])
    ax.set_ylabel('tokens/sec'); ax.set_title(title, color=TEXT, fontweight='bold')
    style(ax)
axes[0, 0].legend(frameon=False, labelcolor=TEXT, ncol=3, fontsize=8, loc='lower left')
ax = axes[1, 1]
kv_m = [4.81, 10.93, 21.93]
bars = ax.bar(['DCP1', 'DCP2', 'DCP4'], kv_m, color=[BLUE, CYAN, PURPLE], width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}M')
ax.set_ylabel('usable KV tokens'); ax.set_title('FP8 KV capacity, no spec', color=TEXT, fontweight='bold')
style(ax)
fig.suptitle('R24 topology map', fontsize=20, fontweight='bold', color=BLUE, y=0.96)
fig.text(0.5, 0.905, 'DCP1 wins decode, DCP4 multiplies capacity, DCP2 is a real middle ground', ha='center', color=TEXT, fontsize=11)
finish(fig, '01-r24-dcp-scaling.png', '4× RTX PRO 6000 Blackwell · PCIe 5 single root · R24 pinned digest · FP8 KV · fairness 0.4 · 1M context')


# 2. LMCache tradeoff
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(top=0.76, bottom=0.11, hspace=0.48, wspace=0.28)
spec_names = ['no spec', 'MTP3', 'DFlash K7']
gpu_rows = [by['dcp4-vram-nvfp4-mtp0'], by['dcp4-vram-nvfp4-mtp3'], by['dcp4-vram-nvfp4-dflash2']]
lmc_rows = [by['dcp4-lmcache-nvfp4-mtp0'], by['dcp4-lmcache-nvfp4-mtp3'], by['dcp4-lmcache-nvfp4-dflash2']]
x = np.arange(3); w = 0.34
for ax, metric, title in [(axes[0,0], 'C1', '1 user'), (axes[0,1], 'C8', '8 users'), (axes[1,0], 'C16', '16 users')]:
    g = [r.get(metric) or 0 for r in gpu_rows]
    l = [r.get(metric) or 0 for r in lmc_rows]
    b1 = ax.bar(x-w/2, g, w, color=BLUE, label='GPU only', zorder=3)
    b2 = ax.bar(x+w/2, l, w, color=GREEN, label='LMCache', zorder=3)
    label_bars(ax, b1); label_bars(ax, b2)
    for i, (a, b) in enumerate(zip(g, l)):
        if a and b:
            delta = (b/a-1)*100
            ax.text(i, max(a,b)*1.08, f'{delta:+.1f}%', ha='center', color=GREEN if delta>=0 else AMBER, fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(spec_names, fontsize=9)
    ax.set_ylabel('tokens/sec'); ax.set_title(title, color=TEXT, fontweight='bold')
    style(ax)
axes[0,0].legend(frameon=False, labelcolor=TEXT, fontsize=9)
ax = axes[1,1]
labels = ['GPU only', 'LMCache\nRAM only', 'LMCache\nfresh L2']
values = [10.329, 8.812, 8.571]
bars = ax.bar(labels, values, color=[BLUE, CYAN, GREEN], width=0.62, zorder=3)
label_bars(ax, bars, '{:.2f}k')
for i, v in enumerate(values):
    if i:
        ax.text(i, v-1.15, f'{(v/values[0]-1)*100:+.1f}%', ha='center', color=BG, fontsize=9, fontweight='bold')
ax.set_ylim(0, 12.5); ax.set_ylabel('cold 33K prefill, k tok/s')
ax.set_title('Repeated cold prefill, 8 runs', color=TEXT, fontweight='bold')
style(ax)
fig.suptitle('R24 LMCache, decode is free, cold stores are not', fontsize=20, fontweight='bold', color=GREEN, y=0.97)
fig.text(0.5, 0.925, 'Decode stays flat, restart recovery is 9.5× faster, cold prefill stores cost 15% here', ha='center', color=TEXT, fontsize=11)
fig.text(0.5, 0.887, '951K replay: 77.1s cold  →  3.5s warm  →  12.7s first post-restart  →  3.7s next', ha='center', color=CYAN, fontsize=10, fontweight='bold')
finish(fig, '02-r24-lmcache-tradeoff.png', 'Matched DCP4 · packed NVFP4 KV · GPU-only GMU .93 · LMCache GMU .95 · unique cache salts for cold prefill A/B')


# 3. Capacity and backend choices
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
fig.subplots_adjust(top=0.80, bottom=0.17, wspace=0.33)
ax=axes[0]
labels=['DCP1\nFP8','DCP2\nFP8','DCP4\nFP8','DCP4\nNVFP4','DCP4\nB12X KDA']
vals=[4.81,10.93,21.93,36.51,40.58]
bars=ax.bar(labels,vals,color=[BLUE,CYAN,PURPLE,GREEN,AMBER],width=.64,zorder=3)
label_bars(ax,bars,'{:.2f}M',fontsize=8)
ax.set_ylabel('usable KV tokens'); ax.set_title('Capacity ladder',fontweight='bold',color=TEXT); style(ax)
ax=axes[1]
labels=['rank-local\nselection','complete-KV\nselection']
vals=[7974,9727]
bars=ax.bar(labels,vals,color=[RED,GREEN],width=.58,zorder=3)
label_bars(ax,bars,'{:.0f}')
ax.text(0.5,10250,'+22%',ha='center',color=GREEN,fontweight='bold',fontsize=13)
ax.set_ylim(0,11200); ax.set_ylabel('32K prefill tok/s'); ax.set_title('DCP4 complete-KV wins',fontweight='bold',color=TEXT); style(ax)
ax=axes[2]
labels=['FlashKDA','B12X KDA']
pre=[9727,9349]; kv=[36.51,40.58]
x=np.arange(2); w=.35
b1=ax.bar(x-w/2,[v/1000 for v in pre],w,color=BLUE,label='prefill k tok/s',zorder=3)
b2=ax.bar(x+w/2,kv,w,color=AMBER,label='KV million tokens',zorder=3)
label_bars(ax,b1,'{:.2f}k',fontsize=8); label_bars(ax,b2,'{:.2f}M',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title('KDA backend trade',fontweight='bold',color=TEXT); ax.legend(frameon=False,labelcolor=TEXT,fontsize=8); style(ax)
fig.suptitle('R24 memory and prefill choices',fontsize=20,fontweight='bold',color=PURPLE,y=.95)
fig.text(.5,.875,'Packed NVFP4 plus B12X KDA reaches 40.6M runtime KV tokens, complete-KV adds 22% prefill',ha='center',color=TEXT,fontsize=11)
finish(fig,'03-r24-capacity-backends.png','R24 · no speculation · TP4 · runtime KV budgets from benchmark metrics, not startup estimates')


# 4. Validation scorecard and fairness
fig = plt.figure(figsize=(13, 8))
gs=fig.add_gridspec(2,2,height_ratios=[1.15,1],hspace=.42,wspace=.28,left=.08,right=.95,top=.73,bottom=.10)
ax=fig.add_subplot(gs[0,0])
labels=['fairness off','compute share 0.4']
base=[194.3,180.6]; coll=[5.4,105.0]
x=np.arange(2); w=.34
b1=ax.bar(x-w/2,base,w,color=BLUE,label='uncontended',zorder=3)
b2=ax.bar(x+w/2,coll,w,color=GREEN,label='during cold prefill',zorder=3)
label_bars(ax,b1); label_bars(ax,b2)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('decode tok/s'); ax.set_title('Fairness changes the experience',fontweight='bold',color=TEXT); ax.legend(frameon=False,labelcolor=TEXT,fontsize=8); style(ax)
ax.text(0,176,'97% stall',ha='center',color=RED,fontweight='bold',bbox=dict(facecolor=BG,edgecolor=RED,boxstyle='round,pad=.25')); ax.text(1,160,'42% stall',ha='center',color=AMBER,fontweight='bold',bbox=dict(facecolor=BG,edgecolor=AMBER,boxstyle='round,pad=.25'))
ax=fig.add_subplot(gs[0,1])
labels=['cold','warm','post restart','next hit']; vals=[77.05,3.51,12.67,3.66]
bars=ax.bar(labels,vals,color=[RED,GREEN,AMBER,CYAN],width=.62,zorder=3)
label_bars(ax,bars,'{:.1f}s')
ax.set_ylabel('951K prompt wall time'); ax.set_title('LMCache survives restart',fontweight='bold',color=TEXT); style(ax)
ax.text(1.5,67,'6 / 6 exact hits',ha='center',color=GREEN,fontweight='bold',fontsize=12,bbox=dict(facecolor=BG,edgecolor=GREEN,boxstyle='round,pad=.3'))
ax=fig.add_subplot(gs[1,:]); ax.set_axis_off()
items=[
 ('1M needles','3 / 3 HIT',GREEN),('951K neutral replay','6 / 6 EXACT',GREEN),
 ('Eviction churn','40 / 40 IDENTICAL',GREEN),('Long generations','16 / 16 CLEAN',GREEN),
 ('Tool result reorder','PASS',GREEN),('Estonia','6 / 6 PASS',GREEN),
 ('Lavd ledger','8 / 8 EXACT',GREEN),('Native offload','PASS, private shm',AMBER),
 ('LMCache cold prefill','14 to 17% cost',ORANGE),('Image packaging','2 layers, source lock exact',GREEN),
]
cols=5
for i,(title,value,color) in enumerate(items):
    row=i//cols; col=i%cols
    x0=.01+col*.198; y0=.70-row*.48
    rect=plt.Rectangle((x0,y0),.185,.35,transform=ax.transAxes,facecolor=PANEL,edgecolor=color,linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x0+.092,y0+.23,title,transform=ax.transAxes,ha='center',color=MUTED,fontsize=8)
    ax.text(x0+.092,y0+.10,value,transform=ax.transAxes,ha='center',color=color,fontsize=9,fontweight='bold')
fig.suptitle('R24 validation scorecard',fontsize=21,fontweight='bold',color=CYAN,y=.96)
fig.text(.5,.905,'Everything important passes, two deployment gotchas are real and reproducible',ha='center',color=TEXT,fontsize=11)
finish(fig,'04-r24-validation-scorecard.png','Pinned digest ab4ff9d6 · source.lock exact · 4× RTX PRO 6000 Blackwell · all receipts preserved')

print('\n'.join(str(p) for p in sorted(OUT.glob('*.png'))))
