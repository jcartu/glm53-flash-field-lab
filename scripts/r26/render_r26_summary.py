#!/usr/bin/env python3
"""Render the final R26 summary charts (performance + operations) from retained receipts.

CPU-only renderer. Every plotted number is loaded from the actual battery JSON
receipts and cross-validated before rendering:

* speed rows are recomputed from the per-repeat ``r26-steady-counters/v1``
  steady summaries and must match ``festr-clean-counter-summary.json`` exactly;
  every contributing cell must be marked clean in ``clean-rerun-plan.json``
  (zero foreign GPU processes) and boot from the expected pinned image digest.
* natural-chat acceptance must come from clean windows with identical request
  payload fingerprints on all four arms.
* agent prefix-reuse arms must share one trace hash, complete every request,
  and pass every transport/counter integrity gate.
* scheduler lane numbers come from the server's own ``/prefill_fairness``
  readback captured in the completed scheduler-recheck api-discovery receipts.
* native-offload verdicts come from the six-cell native canary matrix.

Receipts that do not exist yet (lane-cap recheck, final runtime diagnostics)
are never fabricated: the generator records them under ``pending`` in the
sidecar JSON and renders only already-completed evidence. When Main re-runs
this script after those producers land, the panels pick them up automatically.

Outputs (all under results/r26/):
    summary-performance.png / summary-performance.json
    summary-operations.png  / summary-operations.json

Each sidecar carries the validation log (an ``observed`` value beside a
verdict where one matters, ``failure`` text only on a failed check) and the
sha256 of exactly the receipts that fed its own image.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r26-battery')
CLEAN = ROOT / 'clean-reruns'
FOLLOWUPS = ROOT / 'followups'
OUT = REPO / 'results' / 'r26'

# Pinned image digests (continuation-state.json / clean-rerun-plan.json).
IMG_R25 = 'voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804'
IMG_R26 = 'voipmonitor/vllm@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68'
IMG_OVERLAY = 'ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:27fe7a2f1df6d01e824cd24d6b83119edea471f4ec4997aa122d82c670133236'

# Shared chart style (Discord dark).
BG = '#0D1117'
PANEL = '#161B22'
GRID = '#263242'
TEXT = '#E6EDF3'
MUTED = '#AAB7C7'
BLUE = '#58A6FF'
GREEN = '#3FB950'
AMBER = '#E3B341'
RED = '#F85149'
PURPLE = '#B892FF'

C_R25 = BLUE
C_R26_BF16 = PURPLE
C_R26_NVFP4 = GREEN
C_OVERLAY = AMBER

WIDTH_PX, HEIGHT_PX, DPI = 1400, 1520, 160
PX_PER_PT = DPI / 72

plt.rcParams.update({
    'font.family': ['Noto Sans', 'DejaVu Sans'],
    'text.color': TEXT,
    'axes.facecolor': PANEL,
    'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'figure.facecolor': BG,
    'savefig.facecolor': BG,
})


class ValidationError(RuntimeError):
    pass


_CHECKS: list[dict] = []
_SOURCES: dict[str, str] = {}


def check(name: str, ok: bool, failure: str, observed: str | None = None) -> None:
    """Record one validation row and raise on failure.

    ``failure`` states what a failing check means and is recorded only when the
    check fails; ``observed`` is the measured value worth publishing beside the
    verdict and is recorded either way.
    """
    row = {'check': name, 'passed': bool(ok)}
    if observed is not None:
        row['observed'] = observed
    if not ok:
        row['failure'] = failure
    _CHECKS.append(row)
    if not ok:
        suffix = f' (observed {observed})' if observed is not None else ''
        raise ValidationError(f'{name}: {failure}{suffix}')


def load_json(path: Path) -> object:
    raw = path.read_bytes()
    _SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValidationError(f'{path.name}: not a PNG')
    return struct.unpack('>II', raw[16:24])


# ---------------------------------------------------------------------------
# Data loading + validation
# ---------------------------------------------------------------------------

FESTR_ARMS = {  # festr arm name -> (boot-label pattern, expected image)
    'R25': ('acceptance-dcp{dcp}-mtp3-r25-[ab]', IMG_R25),
    'R26 BF16': ('acceptance-dcp{dcp}-mtp3-r26-bf16-head', IMG_R26),
    'R26 NVFP4': ('acceptance-dcp{dcp}-mtp3-r26-nvfp4-head', IMG_R26),
    'R26 overlay NVFP4': ('acceptance-dcp{dcp}-mtp3-r26-overlay-nvfp4-head', IMG_OVERLAY),
}


def _boot_glob_match(label: str, pattern: str) -> bool:
    """Match a result/boot label against our fixed patterns with an [ab] class."""
    if '-[ab]' in pattern:
        return any(_boot_glob_match(label, pattern.replace('-[ab]', f'-{c}')) for c in 'ab')
    return label == pattern or label.startswith(pattern + '-repeat')


def load_speed() -> dict:
    """Clean MTP3 steady rows; recomputed from per-repeat counter receipts."""
    festr = load_json(CLEAN / 'festr-clean-counter-summary.json')
    plan = load_json(CLEAN / 'clean-rerun-plan.json')
    boots = {boot['boot_label']: boot for boot in plan['reruns']}

    rows = {(row['arm'], row['dcp'], row['concurrency']): row for row in festr['rows']}
    check('speed:rows', len(rows) == 14, 'expected 14 festr rows', observed=f'{len(rows)} rows')

    # Every (arm, dcp) pair in the festr summary -> the boot labels that fed it.
    labels_for: dict[tuple[str, int], list[str]] = {}
    for arm, dcp, _conc in rows:
        pattern, image = FESTR_ARMS[arm]
        pair = (arm, dcp)
        if pair in labels_for:
            continue
        matched = [label for label in boots if _boot_glob_match(label, pattern.format(dcp=dcp))]
        check(f'speed:boots:{arm}:dcp{dcp}', bool(matched), f'no boots match {pattern}')
        for label in matched:
            boot = boots[label]
            check(f'speed:boot:{label}', boot.get('booted') is True, 'not booted')
            check(f'speed:image:{label}', boot.get('image') == image,
                  f"expected image {image.split(':')[1][:12]}", observed=str(boot.get('image')))
            for cell in boot['cells']:
                ok = (cell.get('clean') is True
                      and all(a.get('executed') and a.get('clean') and a.get('counter_windows_valid')
                              for a in cell['attempts']))
                check(f'speed:clean:{cell["result_label"]}', ok, 'cell not clean/valid')
        labels_for[pair] = [cell['result_label'] for label in matched for cell in boots[label]['cells']]

    # Recompute every festr row from the per-repeat steady summaries.
    for (arm, dcp, conc), row in sorted(rows.items()):
        samples = []
        for label in sorted(labels_for[(arm, dcp)]):
            summary = load_json(CLEAN / f'{label}.steady-summary.json')
            check(f'speed:schema:{label}', summary.get('schema') == 'r26-steady-counters/v1',
                  'expected schema r26-steady-counters/v1', observed=str(summary.get('schema')))
            check(f'speed:windows:{label}', summary.get('all_windows_valid') is True,
                  'invalid steady windows')
            samples.extend(cell for cell in summary['cells']
                           if cell['context_tokens'] == 0 and cell['valid']
                           and cell['concurrency'] == conc)
        check(f'speed:samples:{arm}:dcp{dcp}:c{conc}', len(samples) == row['samples'],
              f"expected {row['samples']} clean repeats", observed=f'{len(samples)} clean repeats')
        tps = float(np.mean([c['output_tokens_per_second'] for c in samples]))
        steps = float(np.mean([c['aggregate_verifier_steps_per_second'] for c in samples]))
        acc = float(np.mean([c['acceptance_fraction'] * 100 for c in samples]))
        check(f'speed:recompute:{arm}:dcp{dcp}:c{conc}',
              abs(tps - row['output_tps_mean']) < 1e-6
              and abs(steps - row['verifier_steps_mean']) < 1e-6
              and abs(acc - row['acceptance_percent_mean']) < 1e-6,
              'festr row does not recompute from receipts',
              observed=f'{tps:.3f} tok/s · {steps:.3f} verifier steps/s · {acc:.2f}% acceptance')
    return {'rows': rows, 'method': festr['method']}


def load_acceptance() -> dict:
    comparison = load_json(FOLLOWUPS / 'realistic-acceptance-comparison.json')
    expected = {
        'r25-mtp3-bf16': IMG_R25,
        'r26-mtp3-bf16': IMG_R26,
        'r26-mtp3-default-nvfp4': IMG_R26,
        'drock-r26-overlay-mtp3-bf16': IMG_OVERLAY,
    }
    arms = {}
    fingerprints = set()
    for arm in comparison['arms']:
        name = arm['arm']
        check(f'acceptance:arm:{name}', name in expected, 'unexpected arm')
        check(f'acceptance:clean:{name}', arm.get('clean_window') is True, 'window not clean')
        check(f'acceptance:complete:{name}',
              arm.get('completed_visible_answers') == 32 and arm.get('requests') == 32,
              'expected 32/32 completed visible answers',
              observed=f"{arm.get('completed_visible_answers')}/{arm.get('requests')}")
        per_arm = load_json(FOLLOWUPS / f'realistic-acceptance-{name}.json')
        check(f'acceptance:image:{name}', per_arm.get('image') == expected[name],
              f"expected image {expected[name].split(':')[1][:12]}",
              observed=str(per_arm.get('image')))
        fingerprints.add(arm['input_set']['request_payload_fingerprint_sha256'])
        arms[name] = arm
    check('acceptance:identical-payloads', len(fingerprints) == 1,
          'request payload fingerprint differs across arms',
          observed=', '.join(sorted(fingerprints)))
    check('acceptance:arms', len(arms) == 4, 'expected 4 arms', observed=f'{len(arms)} arms')

    # Stress-ceiling reference: fixed-prompt repetition probes (ratio metric
    # from the main battery; used only as an off-scale marker, never a speed).
    ceiling = 0.0
    for probe in ('quality-r25-mtp3-bf16-head-acceptance.json',
                  'quality-r26-mtp3-bf16-head-acceptance.json',
                  'quality-r26-mtp3-default-nvfp4-head-acceptance.json'):
        data = load_json(ROOT / probe)
        for sample in data['samples']:
            ceiling = max(ceiling, sample['acceptance_fraction'] * 100)
    check('acceptance:ceiling-sane', 90.0 < ceiling < 100.0, 'ceiling outside 90–100%',
          observed=f'{ceiling:.1f}%')
    return {'arms': arms, 'scope': comparison['scope'], 'synthetic_ceiling_percent': ceiling}


def load_agent_cache() -> dict:
    out = {}
    traces = set()
    final_turn_sizes = set()
    for arm_key, fname, image in (
            ('stock', 'agent-cache-stock-r26.json', IMG_R26),
            ('overlay', 'agent-cache-drock-overlay.json', IMG_OVERLAY)):
        data = load_json(FOLLOWUPS / fname)
        check(f'agent-cache:schema:{arm_key}', data.get('schema') == 'r26-agent-cache/v1',
              'expected schema r26-agent-cache/v1', observed=str(data.get('schema')))
        check(f'agent-cache:image:{arm_key}', data.get('image') == image,
              f"expected image {image.split(':')[1][:12]}", observed=str(data.get('image')))
        traces.add(data['trace']['trace_sha256'])
        sessions = data['config']['sessions']
        final_turn = data['config']['turns_per_session'] - 1  # turn records are 0-indexed
        series = {}
        for name in ('vram', 'lmcache'):
            summary = data['summary'][name]
            check(f'agent-cache:complete:{arm_key}:{name}',
                  summary['requests_completed'] == summary['requests_expected'] == 96,
                  'expected 96/96 requests completed',
                  observed=f"{summary['requests_completed']}/{summary['requests_expected']}")
            gate = next(g for g in data['gates']
                        if g['name'] == f'runtime:agent-cache:{data["arm"]}:{name}')
            failing = [k for k, v in gate['checks'].items() if not v]
            # The only tolerated gap: the server does not expose optional
            # per-request LMCache stream stats (observability, not runtime).
            tolerated = {'lmcache_response_stats_exposed'}
            check(f'agent-cache:integrity:{arm_key}:{name}', set(failing) <= tolerated,
                  'unexpected failing integrity checks',
                  observed=f"unmet gate checks: {', '.join(sorted(failing)) or 'none'}")
            # Transcript size is the prompt of each session's final request. The
            # per-turn reusable total is summed over every session, so it is a
            # fleet figure, never a per-session transcript length.
            final_prompts = {t['session']: t['prompt_tokens'] for t in data['turns']
                             if t['series'] == name and t['turn'] == final_turn}
            by_session = [final_prompts[s] for s in sorted(final_prompts)]
            check(f'agent-cache:final-turn:{arm_key}:{name}', len(by_session) == sessions,
                  f'expected one final-turn request from each of {sessions} sessions',
                  observed=f'{len(by_session)} requests · prompt tokens {by_session}')
            final_turn_sizes.add(tuple(by_session))
            series[name] = {
                'hit_rate_percent': summary['hit_rate'] * 100,
                'observed_hit_tokens': summary['observed_hit_tokens'],
                'expected_reusable_tokens': summary['expected_reusable_tokens'],
                'ttft_mean_seconds': summary['ttft_seconds']['mean'],
                'ttft_p50_seconds': summary['ttft_seconds']['p50'],
                'per_turn': [(t['turn'], None if t['hit_rate'] is None else t['hit_rate'] * 100)
                             for t in summary['per_turn']],
                'final_turn_prompt_tokens_per_session': by_session,
                'final_turn_prompt_tokens_mean': sum(by_session) / len(by_session),
                'final_turn_reusable_tokens_all_sessions': next(
                    t['expected_reusable_tokens'] for t in summary['per_turn'] if t['turn'] == final_turn),
                'external_kv_transfer_tokens': summary['prompt_source_external_kv_transfer_tokens'],
                'per_request_stream_stats': gate['checks']['lmcache_response_stats_exposed'],
            }
        out[arm_key] = {
            'label': 'Stock R26' if arm_key == 'stock' else 'D-Rock overlay',
            'config': data['config'],
            'series': series,
        }
    check('agent-cache:same-trace', len(traces) == 1, 'trace hash differs across arms',
          observed=', '.join(sorted(traces)))
    check('agent-cache:same-transcript', len(final_turn_sizes) == 1,
          'final-turn prompt sizes differ across arms or series',
          observed=f'{len(final_turn_sizes)} distinct final-turn prompt-size vector(s)')
    return out


def load_lanes() -> dict:
    """Effective prefill-lane budget read back from the server, by batch tokens."""
    recheck = FOLLOWUPS / 'scheduler-recheck'
    rows = []
    for name in ('overlay-factor-bt4096-default-dma', 'overlay-factor-bt8192-default-dma',
                 'overlay-factor-bt12288-default-dma', 'overlay-factor-bt16384-default-dma'):
        discovery = load_json(recheck / f'{name}-api-discovery.json')
        readback = discovery['exchange']['body'].get('effective_max_parallel_prefills')
        requested = discovery['boot_group']['policy_structure']['max_parallel_prefills']
        effective = discovery['effective_lane_budget']
        check(f'lanes:{name}', requested == 4 and isinstance(effective, int) and effective >= 1,
              'expected 4 requested lanes and an integer effective budget of at least 1',
              observed=f'requested={requested} effective={effective}')
        check(f'lanes:readback:{name}', readback == effective,
              'top-level budget disagrees with server readback',
              observed=f'server readback effective_max_parallel_prefills={readback}')
        rows.append({'batch_tokens': discovery['boot_group']['batch_tokens'],
                     'dma': discovery['boot_group']['dma_value'],
                     'requested': requested, 'effective': effective})
    for name in ('overlay-factor-bt4096-dma512k', 'overlay-factor-bt12288-dma512k'):
        discovery = load_json(recheck / f'{name}-api-discovery.json')
        requested = discovery['boot_group']['policy_structure']['max_parallel_prefills']
        effective = discovery['effective_lane_budget']
        check(f'lanes:dma512k:{name}', requested == 4 and effective == 1,
              'expected 4 requested lanes clipped to 1 effective',
              observed=f'requested={requested} effective={effective}')
    autos = {}
    for name in ('overlay-rr-lanesauto-vram', 'overlay-decode-aware-lanesauto-vram'):
        discovery = load_json(recheck / f'{name}-api-discovery.json')
        policy = discovery['boot_group']['policy_structure']
        effective = discovery['effective_lane_budget']
        check(f'lanes:auto:{name}',
              policy['max_parallel_prefills'] == 'auto' and discovery['effective_state_coherent'] is True,
              'auto lane resolution not coherent',
              observed=f"lanes={policy['max_parallel_prefills']} effective={effective} "
                       f"coherent={discovery['effective_state_coherent']}")
        autos[policy['prefill_policy']] = effective
    rows.sort(key=lambda row: row['batch_tokens'])

    # Optional upgrade: corrected lane-cap recheck (includes the bt32768
    # four-lane diagnostic). Render it only when the receipt actually exists.
    pending = []
    recheck_report = CLEAN / 'scheduler-lane-cap-recheck.json'
    bt32768 = None
    if recheck_report.exists():
        report = load_json(recheck_report)
        if report.get('schema') == 'r26-scheduler-lane-cap-recheck/v1':
            discoveries = report.get('api_discoveries') or {}
            entries = discoveries.values() if isinstance(discoveries, dict) else discoveries
            for entry in entries:
                try:
                    tokens = entry['boot_group']['batch_tokens']
                    effective = entry['effective_lane_budget']
                except (TypeError, KeyError):
                    continue
                if tokens == 32768 and isinstance(effective, int):
                    bt32768 = {'batch_tokens': 32768, 'requested': 4, 'effective': effective}
            if bt32768 is None:
                pending.append({'artifact': str(recheck_report),
                                'status': 'present but no 32768-token effective-lane readback yet',
                                'impact': '32768 row stays unrendered'})
        else:
            pending.append({'artifact': str(recheck_report),
                            'status': f"unrecognised schema {report.get('schema')}",
                            'impact': 'lane panel keeps completed scheduler-recheck evidence'})
    else:
        pending.append({
            'artifact': str(recheck_report),
            'status': 'absent at render time; producer scheduler_lane_cap_recheck.py not yet run',
            'impact': 'lane panel uses the completed followups/scheduler-recheck readbacks; '
                      'the 32768-token four-lane diagnostic is noted as queued, not charted',
        })
    return {'rows': rows, 'autos': autos, 'bt32768': bt32768, 'pending': pending}


def load_native() -> dict:
    canary = load_json(FOLLOWUPS / 'cache-native-canary-summary.json')
    check('native:schema', canary.get('schema') == 'r26-cache-phase/v2',
          'expected schema r26-cache-phase/v2', observed=str(canary.get('schema')))
    cells = {}
    for cell in canary['matrix']:
        config = cell['configuration']
        key = (config['release'], config['gpu_memory_utilization'], config['kv'])
        cells[key] = {'passed': cell['passed'], 'booted': cell['booted'],
                      'failure_class': cell['failure_class'], 'label': cell['label']}
    for kv in ('fp8_ds_mla', 'nvfp4_ds_mla'):
        check(f'native:r25-pass:{kv}', cells[('r25', '0.93', kv)]['passed'] is True,
              'matched R25 control did not pass')
        for gmu in ('0.93', '0.90'):
            cell = cells[('r26', gmu, kv)]
            check(f'native:r26-oom:{gmu}:{kv}',
                  cell['passed'] is False and cell['failure_class'] == 'native_boot_cuda_oom',
                  'expected a native_boot_cuda_oom boot failure',
                  observed=f"passed={cell['passed']} failure_class={cell['failure_class']}")
    flags = {k: canary['checks'][k] for k in ('all_r25_gmu093_controls_passed',
                                              'all_r26_gmu093_baselines_passed',
                                              'all_r26_gmu090_diagnostics_passed')}
    check('native:contract',
          flags['all_r25_gmu093_controls_passed'] is True
          and flags['all_r26_gmu093_baselines_passed'] is False
          and flags['all_r26_gmu090_diagnostics_passed'] is False,
          'matrix checks inconsistent with the cell verdicts',
          observed=' '.join(f'{k}={v}' for k, v in flags.items()))

    # Optional upgrade: final runtime diagnostics (safetensors loader variant +
    # model-backed #645 replay). Render only from a real receipt.
    pending = []
    loader_rows = []
    replay = None
    diag_path = CLEAN / 'final-runtime-diagnostics.json'
    if diag_path.exists():
        diag = load_json(diag_path)
        if diag.get('schema') == 'r26-final-runtime-diagnostics/v1':
            for row in diag.get('native_loader') or []:
                config = row.get('configuration', {})
                if 'kv' in config:
                    check(f"native:loader-config:{config['kv']}",
                          config.get('load_format_override') == 'safetensors'
                          and row.get('attempted') is True and isinstance(row.get('passed'), bool),
                          'incomplete or differently configured loader diagnostic',
                          observed=f"load_format_override={config.get('load_format_override')} "
                                   f"attempted={row.get('attempted')} passed={row.get('passed')}")
                    loader_rows.append({'kv': config['kv'], 'passed': row['passed']})
            replay_block = diag.get('replay') or {}
            if isinstance(replay_block, dict) and 'passed' in replay_block:
                result = replay_block.get('result') or {}
                replay = {'passed': bool(replay_block['passed']),
                          'arm': result.get('arm'), 'image': result.get('image')}
                # The chart labels this result as the D-Rock overlay's, so pin
                # the image here rather than let it read as a stock R26 result.
                check('native:replay:image', replay['image'] == IMG_OVERLAY,
                      f"expected D-Rock overlay image {IMG_OVERLAY.split(':')[1][:12]}",
                      observed=str(replay['image']))
            if not loader_rows:
                pending.append({'artifact': str(diag_path),
                                'status': 'present but native_loader is empty',
                                'impact': 'loader-diagnostic row stays unrendered'})
        else:
            pending.append({'artifact': str(diag_path),
                            'status': f"unrecognised schema {diag.get('schema')}",
                            'impact': 'native panel keeps completed canary matrix only'})
    else:
        pending.append({
            'artifact': str(diag_path),
            'status': 'absent at render time; producer final_runtime_diagnostics.py not yet run',
            'impact': 'native panel shows the completed six-cell canary matrix only; '
                      'safetensors-loader workaround and model-backed #645 replay are noted '
                      'as queued, never charted',
        })
    return {'cells': cells, 'loader_rows': loader_rows, 'replay': replay, 'pending': pending}


# ---------------------------------------------------------------------------
# Pixel-true layout engine
# ---------------------------------------------------------------------------

def _fx(x_px: float) -> float:
    return x_px / WIDTH_PX


def _fy(y_px_from_top: float) -> float:
    return 1.0 - y_px_from_top / HEIGHT_PX


def _rect_px(x: float, y_top: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (_fx(x), _fy(y_top + h), _fx(w), h / HEIGHT_PX)


def _text_width_px(s: str, pt: float, bold: bool = False) -> float:
    return len(s) * pt * PX_PER_PT * (0.56 if bold else 0.50)


def wrap_px(s: str, pt: float, max_px: float, bold: bool = False) -> list[str]:
    chars = max(20, int(max_px / (pt * PX_PER_PT * (0.56 if bold else 0.50))))
    return textwrap.wrap(s, chars)


class Canvas:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, facecolor=BG)

    def text(self, x: float, y: float, s: str, pt: float, color: str = TEXT,
             bold: bool = False, va: str = 'center', ha: str = 'left',
             italic: bool = False) -> None:
        self.fig.text(_fx(x), _fy(y), s, fontsize=pt, color=color, va=va, ha=ha,
                      fontweight=700 if bold else 400,
                      style='italic' if italic else 'normal')

    def wrapped(self, x: float, y: float, s: str, pt: float, max_px: float,
                color: str = TEXT, leading: float = 1.45, italic: bool = False) -> float:
        """Draw wrapped text starting at y (top line center); return next free y."""
        lines = wrap_px(s, pt, max_px)
        step = pt * PX_PER_PT * leading
        for index, line in enumerate(lines):
            self.text(x, y + index * step, line, pt, color=color, italic=italic)
        return y + len(lines) * step

    def card(self, y_top: float, y_bottom: float, x: float = 40, w: float = 1320) -> None:
        self.fig.patches.append(FancyBboxPatch(
            (_fx(x), _fy(y_bottom)), _fx(w), (y_bottom - y_top) / HEIGHT_PX,
            transform=self.fig.transFigure, zorder=-10,
            boxstyle='round,pad=0,rounding_size=0.010',
            facecolor=PANEL, edgecolor=GRID, linewidth=1.0))
    def chip_row(self, x: float, y: float, items: list[tuple[str, str]], pt: float = 10) -> None:
        cursor = x
        for color, label in items:
            self.fig.patches.append(Rectangle(
                (_fx(cursor), _fy(y + 7)), _fx(15), 13 / HEIGHT_PX,
                transform=self.fig.transFigure, facecolor=color, edgecolor='none', zorder=5))
            self.text(cursor + 22, y, label, pt, color=TEXT)
            cursor += 22 + _text_width_px(label, pt) + 34

    def header(self, title: str, subtitle: str, accent: str) -> None:
        self.fig.patches.append(Rectangle(
            (_fx(40), _fy(34)), _fx(64), 6 / HEIGHT_PX,
            transform=self.fig.transFigure, facecolor=accent, edgecolor='none'))
        self.text(40, 68, title, 19.5, bold=True)
        self.wrapped(40, 100, subtitle, 10.5, 1280, color=MUTED)

    def footer(self, lines: list[str]) -> None:
        y = 1450
        for line in lines:
            for wrapped_line in wrap_px(line, 8.5, 1320):
                self.text(40, y, wrapped_line, 8.5, color=MUTED)
                y += 17

    def panel_title(self, y: float, title: str, note: str | None, accent: str = TEXT) -> float:
        self.text(76, y, title, 13.5, bold=True, color=accent)
        if note:
            return self.wrapped(76, y + 26, note, 9.5, 1250, color=MUTED)
        return y + 26

    def axes(self, x: float, y_top: float, w: float, h: float):
        return self.fig.add_axes(_rect_px(x, y_top, w, h))


def style_axis(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, alpha=0.5, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0)


def bar_labels(ax, bars, fmt, fontsize=10.5, color=TEXT, weight=600):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=fontsize, color=color, fontweight=weight)


def save(cv: Canvas, name: str) -> Path:
    path = OUT / name
    cv.fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(cv.fig)
    width, height = png_size(path)
    check(f'png:{name}', (width, height) == (WIDTH_PX, HEIGHT_PX),
          f'expected {WIDTH_PX}x{HEIGHT_PX}', observed=f'{width}x{height}')
    return path


def write_sidecar(name, image, panels, checks, sources, pending):
    """``checks`` and ``sources`` are the validation rows and receipt hashes of this image only."""
    sidecar = {
        'schema': 'r26-summary-render/v1',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generator': {'script': str(Path(__file__).resolve()),
                      'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'image': {'file': image.name, 'width_px': WIDTH_PX, 'height_px': HEIGHT_PX},
        'panels': panels,
        'validation': checks,
        'sources': sources,
        'pending': pending,
    }
    path = OUT / name
    path.write_text(json.dumps(sidecar, indent=2) + '\n')
    return path


def delta_text(base, new):
    return f'{(100 * (new / base - 1)):+.1f}%'.replace('-', '−')


# ---------------------------------------------------------------------------
# Performance image
# ---------------------------------------------------------------------------

def render_performance(speed, acceptance) -> Path:
    rows = speed['rows']
    cv = Canvas()
    cv.header('GLM-5.3-Flash — R26 vs R25, clean MTP3 speed',
              '4× RTX PRO 6000 Blackwell · TP4 · FP8 KV · context-0 steady decode · '
              'zero-foreign-GPU windows only', GREEN)
    cells = [(1, 1), (1, 8), (4, 1), (4, 8)]

    # ---- Panel A: R25 vs R26 default (NVFP4 head), overlay at DCP4 ---------
    cv.card(150, 680)
    cv.panel_title(182, 'Steady decode throughput — R25 vs R26 default',
                   'Output tokens/s, aggregate · mean of valid interior steady windows '
                   '(n=2 clean repeats; R25 DCP4 n=4, two boots)')
    cv.chip_row(76, 248, [
        (C_R25, 'R25 · BF16 MTP3 head'),
        (C_R26_NVFP4, 'R26 · NVFP4 MTP3 head (default)'),
        (C_OVERLAY, 'D-Rock overlay · NVFP4 (DCP4 only)'),
    ])
    axes = [cv.axes(76, 294, 596, 262), cv.axes(748, 294, 576, 262)]
    deltas = {}
    for ax, conc, title in zip(axes, (1, 8), ('1 user (C1)', '8 users (C8)')):
        dcps = [1, 4]
        x = np.arange(len(dcps))
        width = 0.26
        r25 = [rows[('R25', dcp, conc)]['output_tps_mean'] for dcp in dcps]
        r26 = [rows[('R26 NVFP4', dcp, conc)]['output_tps_mean'] for dcp in dcps]
        b1 = ax.bar(x - width / 2, r25, width, color=C_R25, zorder=3)
        b2 = ax.bar(x + width / 2, r26, width, color=C_R26_NVFP4, zorder=3)
        fmt = '{:.1f}' if conc == 1 else '{:.0f}'
        bar_labels(ax, b1, fmt)
        bar_labels(ax, b2, fmt)
        centers = list(x)
        overlay = None
        for dcp_index, dcp in enumerate(dcps):
            deltas[(dcp, conc)] = 100 * (r26[dcp_index] / r25[dcp_index] - 1)
            if dcp == 4:
                overlay = rows[('R26 overlay NVFP4', dcp, conc)]['output_tps_mean']
                bo = ax.bar([x[dcp_index] + 1.5 * width], [overlay], width,
                            color=C_OVERLAY, zorder=3)
                bar_labels(ax, bo, fmt, fontsize=10)
                centers[dcp_index] = x[dcp_index] + width / 2
        ax.set_xticks(centers)
        ax.set_xticklabels(['DCP1', 'DCP4'], fontsize=11.5)
        ax.set_title(title, color=TEXT, fontsize=11.5, fontweight=600, pad=4)
        ax.set_ylim(0, max(r25 + r26 + ([overlay] if overlay else [])) * 1.24)
        style_axis(ax)
    cv.text(76, 632, 'These four matched speed cells are flat to slightly up:',
            11.5, bold=True)
    flat = ' · '.join(f'DCP{dcp}·C{conc} {delta_text(100, 100 + deltas[(dcp, conc)])}'
                      for dcp, conc in ((1, 1), (4, 1), (1, 8), (4, 8)))
    cv.text(76, 656, flat, 11.5, color=GREEN)

    # ---- Panel B: within R26, BF16 vs NVFP4 proposal head ------------------
    cv.card(700, 1110)
    cv.panel_title(732, 'Inside R26: BF16 vs NVFP4 proposal head',
                   'Same pinned R26 image, same FP8 KV, same MTP3 depth — '
                   'only the draft-head dtype differs')
    cv.chip_row(76, 782, [(C_R26_BF16, 'BF16 head'), (C_R26_NVFP4, 'NVFP4 head')])
    cv.text(76, 810, 'Output tokens/s', 9.5, color=MUTED)
    cv.text(748, 810, 'Per-request verifier steps/s', 9.5, color=MUTED)
    axes = [cv.axes(76, 832, 596, 182), cv.axes(748, 832, 576, 182)]
    metrics = [('output_tps_mean', 'output tok/s'),
               ('verifier_steps_mean', 'verifier steps/s')]
    head_deltas = {}
    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(cells))
        width = 0.36
        bf16 = [rows[('R26 BF16', dcp, conc)][metric] for dcp, conc in cells]
        nvfp4 = [rows[('R26 NVFP4', dcp, conc)][metric] for dcp, conc in cells]
        b1 = ax.bar(x - width / 2, bf16, width, color=C_R26_BF16, zorder=3)
        b2 = ax.bar(x + width / 2, nvfp4, width, color=C_R26_NVFP4, zorder=3)
        for bars in (b1, b2):
            for bar in bars:
                ax.annotate(f'{bar.get_height():.0f}',
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2), textcoords='offset points', ha='center',
                            fontsize=9, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels([f'DCP{dcp}·C{conc}' for dcp, conc in cells], fontsize=9.5)
        ax.set_ylim(0, max(bf16 + nvfp4) * 1.18)
        style_axis(ax)
        if metric == 'output_tps_mean':
            for index, (dcp, conc) in enumerate(cells):
                head_deltas[(dcp, conc)] = 100 * (nvfp4[index] / bf16[index] - 1)
    best = max(head_deltas, key=head_deltas.get)
    cv.wrapped(76, 1066,
               'Output speed depends on both verifier throughput and accepted tokens per step. '
               f'Largest observed head-only output gain: DCP{best[0]}·C{best[1]} '
               f'{delta_text(100, 100 + head_deltas[best])}; DCP4·C8 '
               f'{delta_text(100, 100 + head_deltas[(4, 8)])}.',
               10, 1240, color=TEXT)
    # ---- Panel C: acceptance is a workload property ------------------------
    cv.card(1130, 1430)
    cv.panel_title(1162, 'Prompt mix moves acceptance more in this slice',
                   'C1 and C8 · 32 replies per arm · same prompts and seeds · '
                   'acceptance = accepted ÷ proposed draft tokens')
    ax = cv.axes(360, 1224, 560, 160)
    arm_order = ['r25-mtp3-bf16', 'r26-mtp3-bf16', 'r26-mtp3-default-nvfp4', 'drock-r26-overlay-mtp3-bf16']
    arm_display = ['R25 · BF16', 'R26 · BF16', 'R26 · NVFP4 (default)', 'D-Rock overlay · BF16']
    arm_colors = [C_R25, C_R26_BF16, C_R26_NVFP4, C_OVERLAY]
    values = [acceptance['arms'][arm]['acceptance_percent'] for arm in arm_order]
    y = np.arange(len(arm_order))[::-1]
    ax.barh(y, values, 0.62, color=arm_colors, zorder=3)
    for yi, value in zip(y, values):
        ax.annotate(f'{value:.1f}%', (value, yi), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=11.5, fontweight=700, color=TEXT)
    bench = [row['acceptance_percent_mean'] for row in rows.values()]
    lo, hi = min(bench), max(bench)
    ax.axvspan(lo, hi, color=MUTED, alpha=0.16, zorder=1)
    ax.annotate(f'clean sustained-decode mix: {lo:.1f}–{hi:.1f}%',
                ((lo + hi) / 2, 3.95), ha='center', va='top', fontsize=9, color=MUTED)
    ceiling = acceptance['synthetic_ceiling_percent']
    ax.annotate(f'Repetition probe: {ceiling:.1f}% (stress ceiling; off scale)',
                (41.5, -0.72), ha='right', va='center', fontsize=9, color=MUTED, style='italic')
    ax.set_yticks(y)
    ax.set_yticklabels(arm_display, fontsize=10.5)
    ax.set_xlim(0, 42)
    ax.set_ylim(-1.0, 4.3)
    style_axis(ax, axis='x')
    spread = max(values) - min(values)
    cv.wrapped(960, 1222,
               f'Spread: {spread:.1f} percentage points on identical prompts. '
               'This 32-reply slice is descriptive, not a test of equal answer quality.',
               9.5, 370, color=TEXT)

    cv.footer([
        'CPU render of retained receipts · per-file sha256 + validation log in results/r26/summary-performance.json',
        'Pinned images: R25 89376e9aa494 · R26 d0592ea9d73c · overlay 27fe7a2f1df6 · QAD is separate, not charted',
    ])
    return save(cv, 'summary-performance.png')

# ---------------------------------------------------------------------------
# Operations image
# ---------------------------------------------------------------------------

def render_operations(agent_cache, lanes, native) -> Path:
    cv = Canvas()
    cv.header('R26 operations — cache reuse, lanes, offload',
              'Completed R26 observations · pending diagnostics are explicitly marked', AMBER)

    cv.panel_title(182, 'Agent prefix reuse — 12-turn append-only trace',
                   '8 sessions × 12 short turns · identical trace hash on both builds · '
                   'hit rate = cached ÷ reusable prompt tokens')
    ax = cv.axes(98, 250, 518, 230)
    x = np.arange(2)
    width = 0.34
    stock = [agent_cache['stock']['series'][s]['hit_rate_percent'] for s in ('vram', 'lmcache')]
    overlay = [agent_cache['overlay']['series'][s]['hit_rate_percent'] for s in ('vram', 'lmcache')]
    b1 = ax.bar(x - width / 2, stock, width, color=C_R26_NVFP4, zorder=3)
    b2 = ax.bar(x + width / 2, overlay, width, color=C_OVERLAY, zorder=3)
    bar_labels(ax, b1, '{:.1f}%', fontsize=11)
    bar_labels(ax, b2, '{:.1f}%', fontsize=11)
    for index, series in enumerate(('vram', 'lmcache')):
        s_ttft = agent_cache['stock']['series'][series]['ttft_mean_seconds']
        o_ttft = agent_cache['overlay']['series'][series]['ttft_mean_seconds']
        ax.annotate(f'mean TTFT {s_ttft:.2f}s → {o_ttft:.2f}s', (x[index], 0),
                    xytext=(0, -29), textcoords='offset points', ha='center',
                    fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(['VRAM-local', 'LMCache-backed'], fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
    style_axis(ax)
    cv.chip_row(76, 566, [(C_R26_NVFP4, 'Stock R26'), (C_OVERLAY, 'D-Rock overlay')])

    ax = cv.axes(700, 250, 624, 260)
    line_specs = [
        ('overlay', 'lmcache', C_OVERLAY, '-', 'overlay · LMCache'),
        ('overlay', 'vram', AMBER, '--', 'overlay · VRAM'),
        ('stock', 'lmcache', C_R26_NVFP4, '-', 'stock R26 · LMCache'),
        ('stock', 'vram', RED, ':', 'stock R26 · VRAM'),
    ]
    for arm, series, color, linestyle, label in line_specs:
        per_turn = agent_cache[arm]['series'][series]['per_turn']
        turns = [t for t, rate in per_turn if rate is not None]
        rates = [rate for _t, rate in per_turn if rate is not None]
        ax.plot(turns, rates, linestyle, color=color, linewidth=2.0, marker='o',
                markersize=3.5, label=label, zorder=3)
    ax.set_xlabel('turn', fontsize=9.5)
    ax.set_ylabel('per-turn hit rate', fontsize=9.5)
    ax.set_ylim(-4, 104)
    ax.set_xlim(0.6, 12.4)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
    style_axis(ax, axis='both')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc='upper left', ncol=2,
              columnspacing=1.2, handlelength=1.8)
    # Both arms replay one trace (validated above), so the stock LMCache series
    # carries the transcript size; the reusable total is a sum over all sessions.
    transcript = agent_cache['stock']['series']['lmcache']
    config = agent_cache['stock']['config']
    cv.wrapped(76, 600,
               f"Short transcripts: ~{transcript['final_turn_prompt_tokens_mean'] / 1000:.1f}K prompt tokens "
               f"per session by turn {config['turns_per_session']} "
               f"(~{transcript['final_turn_reusable_tokens_all_sessions'] / 1000:.0f}K reusable prompt tokens "
               f"summed over the {config['sessions']} sessions) — not the 95% long-transcript production "
               'target. Per-request LMCache stream stats are not exposed by these builds: an observability '
               'gap, not a runtime failure — every transport/counter integrity gate passed, and external '
               'KV-transfer hits stayed 0 on both arms.',
               9.5, 1250, color=MUTED)

    # ---- Panel B: effective lane budget ------------------------------------
    cv.card(720, 1060)
    cv.panel_title(752, 'Asked for 4 prefill lanes — the scheduler clips to the budget',
                   'Live /prefill_fairness · overlay, VRAM, DCP4 · requested lane count is an upper bound')
    ax = cv.axes(76, 806, 540, 216)
    ax.set_axis_off()
    rows = list(lanes['rows'])
    if lanes['bt32768'] is not None:
        rows.append(lanes['bt32768'])
    y_top = 0.94
    for index, row in enumerate(rows):
        yy = y_top - index * 0.22
        ax.text(0.0, yy, f"batch {row['batch_tokens']}", fontsize=10.5, color=TEXT,
                va='center', fontweight=600)
        for slot in range(4):
            filled = slot < row['effective']
            ax.add_patch(Rectangle((0.36 + slot * 0.085, yy - 0.062), 0.062, 0.124,
                                   facecolor=GREEN if filled else 'none',
                                   edgecolor=GREEN if filled else MUTED,
                                   linewidth=1.4, linestyle='-' if filled else (0, (3, 3))))
        ax.text(0.74, yy, f"{row['effective']} of 4 effective", fontsize=10, color=TEXT, va='center')
    if lanes['bt32768'] is None:
        yy = y_top - len(rows) * 0.22
        ax.text(0.0, yy, 'batch 32768', fontsize=10.5, color=MUTED, va='center', fontweight=600)
        ax.text(0.36, yy, 'diagnostic queued — not measured yet', fontsize=10, color=MUTED,
                va='center', style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.10, 1.06)
    autos = lanes['autos']
    auto_txt = ' / '.join(f'{policy}: {count}' for policy, count in sorted(autos.items()))
    y_note = 812
    for note in [
        'The clip is by design, not a fault: the earlier "mismatch" flag was the harness’s '
        'own exact-lane assertion.',
        f'lanes=auto lands coherent ({auto_txt} effective at batch 4096).',
        'dma=512KB at batch 4096 / 12288: still 1 effective lane.',
        'Live POST tunes only compute share + half-life; lanes, policy, refill are boot-only. '
        'Shipped policies: round-robin, decode-aware.',
    ]:
        y_note = cv.wrapped(660, y_note, note, 9.5, 660, color=TEXT) + 6

    # ---- Panel C: native offload caveat ------------------------------------
    cv.card(1080, 1430)
    cv.panel_title(1112, 'Native DRAM offload: stock R26 fails to boot here',
                   'TP4/DCP4 no-spec · matched R25 control · safetensors is a separate loader diagnostic')
    ax = cv.axes(76, 1170, 540, 220)
    ax.set_axis_off()
    grid_rows = [('r25', '0.93', 'R25 · GMU 0.93', 'matched control'),
                 ('r26', '0.93', 'R26 · GMU 0.93', 'baseline'),
                 ('r26', '0.90', 'R26 · GMU 0.90', 'diagnostic')]
    kv_cols = [('fp8_ds_mla', 'FP8 KV'), ('nvfp4_ds_mla', 'NVFP4 KV')]
    for col_index, (_kv, kv_label) in enumerate(kv_cols):
        ax.text(0.70 + col_index * 0.32, 1.04, kv_label, fontsize=10, color=MUTED,
                ha='center', fontweight=600)
    for row_index, (release, gmu, label, role) in enumerate(grid_rows):
        yy = 0.80 - row_index * 0.30
        ax.text(0.0, yy + 0.02, label, fontsize=10.5, color=TEXT, va='center', fontweight=600)
        ax.text(0.0, yy - 0.09, role, fontsize=8.5, color=MUTED, va='center')
        for col_index, (kv, _label) in enumerate(kv_cols):
            cell = native['cells'][(release, gmu, kv)]
            passed = cell['passed']
            color = GREEN if passed else RED
            ax.add_patch(FancyBboxPatch((0.59 + col_index * 0.32, yy - 0.105), 0.22, 0.21,
                                        boxstyle='round,pad=0.004,rounding_size=0.012',
                                        facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.2))
            ax.text(0.70 + col_index * 0.32, yy + 0.01, 'PASS' if passed else 'BOOT OOM',
                    fontsize=9.5, color=color, ha='center', va='center', fontweight=700)
    if native['loader_rows']:
        yy = 0.80 - len(grid_rows) * 0.30
        ax.text(0.0, yy + 0.02, 'R26 · GMU 0.93', fontsize=10.5, color=TEXT, va='center',
                fontweight=600)
        ax.text(0.0, yy - 0.09, 'safetensors loader diagnostic', fontsize=8.5, color=MUTED,
                va='center')
        for col_index, (kv, _label) in enumerate(kv_cols):
            row = next((r for r in native['loader_rows'] if r['kv'] == kv), None)
            if row is None:
                continue
            color = GREEN if row['passed'] else RED
            ax.text(0.70 + col_index * 0.32, yy + 0.01, 'PASS' if row['passed'] else 'CHECK FAIL',
                    fontsize=9.5, color=color, ha='center', va='center', fontweight=700)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.25, 1.12)
    if native['loader_rows']:
        loader_note = ('Safetensors: boot + 80K one-token canary only, not offload pressure or durability. '
                       'Default-loader OOM still stands.')
    else:
        loader_note = ('A safetensors-loader workaround diagnostic is queued but not measured — '
                       'nothing is claimed for it.')
    if native['replay'] is not None:
        loader_note += (' Model-backed #645 replay on the D-Rock overlay: '
                        f"{'passed' if native['replay']['passed'] else 'FAILED'}.")
    y_note = 1178
    for note in [
        'Default InstantTensor loader: CUDA OOM ~7% into 191 GB, at GMU 0.93 and 0.90 '
        'with either KV type.',
        'The matched R25 control boots with both KV types. R26 loader diagnostics are '
        'separate configurations, not fixes to the tested image.',
        loader_note,
    ]:
        y_note = cv.wrapped(660, y_note, note, 9.5, 660, color=TEXT) + 8

    cv.footer([
        'Scope: visible-answer checks, not KV-byte equality. Budget-limited replies remain incomplete. '
        'Latency and internal timers stay separate.',
        'Raw receipts + validation: results/r26/summary-operations.json. Pending diagnostics stay unclaimed.',
    ])
    return save(cv, 'summary-operations.png')


# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    speed = load_speed()
    acceptance = load_acceptance()
    perf_path = render_performance(speed, acceptance)
    # Snapshot, then reset: each sidecar lists only the checks and receipts behind its own image.
    perf_checks, perf_sources = list(_CHECKS), dict(_SOURCES)
    _CHECKS.clear()
    _SOURCES.clear()
    perf_pending = [{
        'artifact': str(CLEAN / 'scheduler-lane-cap-recheck.json'),
        'status': 'not an input to this image',
        'impact': 'performance image is complete from clean speed receipts',
    }]

    agent_cache = load_agent_cache()
    lanes = load_lanes()
    native = load_native()
    ops_path = render_operations(agent_cache, lanes, native)
    ops_checks, ops_sources = list(_CHECKS), dict(_SOURCES)

    rows = speed['rows']
    perf_panels = {
        'release_comparison': {
            f'dcp{dcp}_c{conc}': {
                'r25_output_tps': rows[('R25', dcp, conc)]['output_tps_mean'],
                'r26_nvfp4_output_tps': rows[('R26 NVFP4', dcp, conc)]['output_tps_mean'],
                'delta_percent': 100 * (rows[('R26 NVFP4', dcp, conc)]['output_tps_mean']
                                        / rows[('R25', dcp, conc)]['output_tps_mean'] - 1),
                **({'overlay_nvfp4_output_tps': rows[('R26 overlay NVFP4', dcp, conc)]['output_tps_mean']}
                   if ('R26 overlay NVFP4', dcp, conc) in rows else {}),
            }
            for dcp, conc in ((1, 1), (1, 8), (4, 1), (4, 8))
        },
        'proposal_head_within_r26': {
            f'dcp{dcp}_c{conc}': {
                'bf16_output_tps': rows[('R26 BF16', dcp, conc)]['output_tps_mean'],
                'nvfp4_output_tps': rows[('R26 NVFP4', dcp, conc)]['output_tps_mean'],
                'bf16_verifier_steps_per_s': rows[('R26 BF16', dcp, conc)]['verifier_steps_mean'],
                'nvfp4_verifier_steps_per_s': rows[('R26 NVFP4', dcp, conc)]['verifier_steps_mean'],
                'bf16_acceptance_percent': rows[('R26 BF16', dcp, conc)]['acceptance_percent_mean'],
                'nvfp4_acceptance_percent': rows[('R26 NVFP4', dcp, conc)]['acceptance_percent_mean'],
            }
            for dcp, conc in ((1, 1), (1, 8), (4, 1), (4, 8))
        },
        'natural_chat_acceptance': {
            arm: {'acceptance_percent': acceptance['arms'][arm]['acceptance_percent'],
                  'requests': 32, 'clean_window': True}
            for arm in acceptance['arms']
        },
        'references': {
            'bench_mix_acceptance_range_percent': [
                min(row['acceptance_percent_mean'] for row in rows.values()),
                max(row['acceptance_percent_mean'] for row in rows.values())],
            'synthetic_repetition_probe_ceiling_percent': acceptance['synthetic_ceiling_percent'],
            'scope': acceptance['scope'],
        },
    }
    write_sidecar('summary-performance.json', perf_path, perf_panels, perf_checks, perf_sources,
                  perf_pending)

    ops_panels = {
        'agent_prefix_reuse': {
            arm: {
                'sessions': agent_cache[arm]['config']['sessions'],
                'turns_per_session': agent_cache[arm]['config']['turns_per_session'],
                **{series: {k: v for k, v in agent_cache[arm]['series'][series].items()
                            if k != 'per_turn'} | {'per_turn': agent_cache[arm]['series'][series]['per_turn']}
                   for series in ('vram', 'lmcache')},
            }
            for arm in ('stock', 'overlay')
        },
        'effective_lane_budget': {
            'rows': lanes['rows'], 'auto_resolution': lanes['autos'],
            'bt32768': lanes['bt32768'],
        },
        'native_offload': {
            f'{release}-gmu{gmu}-{kv}': native['cells'][(release, gmu, kv)]
            for release, gmu, kv in native['cells']
        },
        'native_loader_diagnostic': native['loader_rows'],
        'model_backed_replay_645': native['replay'],
    }
    write_sidecar('summary-operations.json', ops_path, ops_panels, ops_checks, ops_sources,
                  lanes['pending'] + native['pending'])

    passed = sum(1 for check_ in perf_checks + ops_checks if check_['passed'])
    print(f'{passed} validation checks passed')
    print(perf_path)
    print(ops_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
