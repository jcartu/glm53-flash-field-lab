#!/usr/bin/env python3
"""Render pinned R27 findings without pooling source, policy, or checkpoint arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import render_r26_summary as style

MODES = (
    ('mtp0', 1, 'no spec\nDCP1 FP8'),
    ('mtp3', 1, 'MTP3\nDCP1 FP8'),
    ('mtp3', 4, 'MTP3\nDCP4 FP8'),
    ('dflash2', 4, 'DFlash K7\nDCP4 NVFP4'),
)


def collect(root: Path) -> dict:
    sources = {}

    def load(name: str | Path) -> dict:
        path = root / name
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    completion = load('r27-qualification-completed.json')
    if completion.get('all_phase_reports_final') is not True:
        raise ValueError('R27 evidence is not final; no publication chart generated')
    speed = load('r27-speed-summary.json')
    if speed.get('all_cells_attempted') is not True or len(speed['cells']) != 8:
        raise ValueError('Incomplete R27 matched speed matrix')
    rows = []
    for boot in speed['cells']:
        if boot.get('speed_eligible') is not True or boot.get('exclusive_gpu_window') is not True:
            raise ValueError(f"Unqualified speed window: {boot['label']}")
        steady = load(boot['steady_summary'])
        for concurrency in (1, 8):
            matches = [cell for cell in steady['cells']
                       if cell['concurrency'] == concurrency and cell['context_tokens'] == 0]
            if len(matches) != 1 or matches[0].get('valid') is not True:
                raise ValueError(f"Missing valid steady window: {boot['label']} C{concurrency}")
            cell = matches[0]
            rows.append({key: boot[key] for key in ('arm', 'image', 'spec', 'dcp', 'kv')} | {
                'concurrency': concurrency,
                'context_tokens': 0,
                'output_tokens_per_second': cell['output_tokens_per_second'],
                'acceptance_fraction': cell['acceptance_fraction'] if boot['spec'] != 'mtp0' else None,
                'aggregate_verifier_steps_per_second': (
                    cell['aggregate_verifier_steps_per_second'] if boot['spec'] != 'mtp0' else None),
                'started_at': cell['started_at'], 'finished_at': cell['finished_at'],
            })
    boundary = load('r27-boundary-phase-summary.json')
    counts = boundary['case_accounting']
    if not boundary['source_verified'] or counts['measured_http'] != counts['planned_http']:
        raise ValueError('Incomplete/source-unverified boundary measurements')
    images = {row['arm']: row['image'] for row in boundary['scalar_results']}
    if len(boundary['scalar_results']) != 3 or set(images) != {'stock', 'patched', 'auto'}:
        raise ValueError('Expected one scalar result per pinned arm')
    for row in speed['cells'] + boundary['http_results']:
        if row['image'] != images[row['arm']]:
            raise ValueError('Mixed image provenance within an R27 arm')
    scalar = []
    for arm in boundary['scalar_results']:
        summary = arm['summary']
        if not summary['matrix_execution_complete']:
            raise ValueError('Incomplete scalar matrix')
        scalar.append({'arm': arm['arm'], 'total': summary['measured_cases'],
                       'passed': summary['measured_cases'] - len(summary['failed_case_ids'])})
    cache = load('r27-cache-summary.json')
    if cache.get('all_cases_attempted') is not True:
        raise ValueError('Incomplete cache lifecycle evidence')
    for row in cache['lifecycles'] + cache['native']:
        if row['image'] != images[row['case']['arm']]:
            raise ValueError('Cache image differs from the reported R27 arm')
    cache_rows = []
    for arm in ('stock', 'patched'):
        lifecycles = [row for row in cache['lifecycles'] if row['case']['arm'] == arm]
        native = [row for row in cache['native'] if row['case']['arm'] == arm]
        stage_results = [
            {'label': row['label'], 'stages': {
                stage: receipt.get('returncode') == 0
                and receipt.get('answer_evidence', {}).get('accepted_visible_recall') is True
                for stage, receipt in row['requests'].items()
            }} for row in lifecycles
        ]
        cache_rows.append({'arm': arm, 'lifecycle_pass': sum(row.get('passed') is True for row in lifecycles),
                           'lifecycle_total': len(lifecycles),
                           'native_pass': sum(row['booted'] and row['canary_passed'] for row in native),
                           'native_total': len(native), 'stage_results': stage_results,
                           'restore_answers_pass': sum(row['stages'].get(stage) is True
                                                       for row in stage_results for stage in ('l1', 'l2')),
                           'restore_answers_total': 2 * len(lifecycles)})
    scheduler = load('r27-scheduler/phase-summary.json')
    policy = scheduler['comparisons_and_deltas']['configured_auto_policy_effect']
    policy_rows = []
    for row in policy['mixed_policy_qos']:
        if row['measurement_classification'] != 'measured':
            raise ValueError('Missing matched patched-source policy measurement')
        policy_rows.append({'identity': row['identity'],
                            'fixed': row['baseline']['observed_values'],
                            'auto': row['candidate']['observed_values']})
    if len(policy_rows) != 4:
        raise ValueError('Expected four matched policy fixtures')
    mismatches = {arm: [] for arm in images}
    for row in boundary['http_results']:
        mismatches[row['arm']].extend(
            {'cell': row['cell_id'], 'case': case}
            for case in row['summary'].get('cache_oracle_failures', []))
    return {
        'schema': 'r27-summary-render/v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'root': str(root), 'sources': sources, 'images': images, 'speed': rows, 'scalar': scalar,
        'cache': cache_rows, 'boundary_counts': counts,
        'cache_oracle_mismatches_by_arm': mismatches,
        'policy': policy_rows, 'scheduler_findings': scheduler['findings'],
        'scope': 'Source-matched steady decode, packaged scalar kernels, visible-answer lifecycle gates, '
                 'and same-source fixed/auto workload fixtures. No promotion, production hit-rate, '
                 'KV-byte equality, or QAD checkpoint claim.',
    }


def render(data: dict, path: Path) -> None:
    cv = style.Canvas()
    cv.header('R27: matched speed and the failures that matter',
              '4× RTX PRO 6000 · TP4 · published NVFP4 weights · D-Rock eval images stay separate from stock',
              style.BLUE)
    cv.card(150, 580)
    cv.panel_title(182, 'Steady decode: stock R27 vs matched D-Rock patch',
                   'Context 0 · aggregate output tokens/s · 60-second cells, interior counter windows · fixed policy on both')
    cv.chip_row(76, 245, [(style.BLUE, 'Stock R27'), (style.AMBER, 'Patched eval')])
    for index, concurrency in enumerate((1, 8)):
        ax = cv.axes(90 + index * 650, 295, 550, 200)
        style.style_axis(ax)
        ax.set_title(f'{concurrency} concurrent request' + ('s' if concurrency > 1 else ''),
                     color=style.TEXT, fontsize=12)
        for arm, offset, color in (('stock', -0.18, style.BLUE), ('patched', 0.18, style.AMBER)):
            values = [next(row['output_tokens_per_second'] for row in data['speed']
                           if (row['arm'], row['spec'], row['dcp'], row['concurrency'])
                           == (arm, spec, dcp, concurrency)) for spec, dcp, _ in MODES]
            bars = ax.bar([x + offset for x in range(4)], values, width=0.35, color=color, zorder=3)
            style.bar_labels(ax, bars, '{:.0f}', fontsize=10)
        ax.set_xticks(range(4), [label for _, _, label in MODES], fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    cv.wrapped(76, 551, 'Matched within each mode only · output includes acceptance effects · verifier counters stay in the sidecar.',
               9, 1240, color=style.MUTED)

    cv.card(600, 970)
    cv.panel_title(632, 'Scalar kernel fix; cache outcomes shown separately',
                   '#674 packaged scalar kernel cases · LMCache cold/APC/L1/restart-L2 · native default-loader canaries')
    ax = cv.axes(195, 716, 400, 166)
    style.style_axis(ax, 'x')
    scalar = data['scalar']
    arm_labels = {'stock': 'Stock', 'patched': 'Patch', 'auto': 'Patch auto'}
    arm_colors = {'stock': style.BLUE, 'patched': style.AMBER, 'auto': style.GREEN}
    bars = ax.barh(range(len(scalar)), [row['passed'] for row in scalar],
                   color=[arm_colors[row['arm']] for row in scalar], height=0.55, zorder=3)
    ax.set_yticks(range(len(scalar)), [arm_labels[row['arm']] for row in scalar])
    ax.invert_yaxis()
    ax.set_xlim(0, 21)
    ax.set_xticks([0, 6, 12, 18])
    for bar, row in zip(bars, scalar):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{row['passed']}/{row['total']}", color=style.TEXT, va='center', fontsize=11, fontweight=700)
    cv.text(690, 718, 'Cache outcomes · separate prompts', 11, bold=True)
    for index, row in enumerate(data['cache']):
        cv.text(690, 756 + 34 * index,
                f"{row['arm'].title()}: L1/L2 {row['restore_answers_pass']}/{row['restore_answers_total']} · "
                f"whole-cycle {row['lifecycle_pass']}/{row['lifecycle_total']}", 10.5)
    stock_stages = next(row['stage_results'] for row in data['cache'] if row['arm'] == 'stock')
    failures = '/'.join(stage.upper() for stage in ('cold', 'apc', 'l1', 'l2')
                        if any(row['stages'].get(stage) is False for row in stock_stages))
    native = '; '.join(f"{arm_labels[row['arm']]} {row['native_pass']}/{row['native_total']}"
                       for row in data['cache'])
    cv.wrapped(690, 821, f'Stock answer failures: {failures}. Native default-loader boots: {native}. '
               'No cross-arm quality claim: different prompts.', 9, 610, color=style.MUTED)
    mismatch_counts = ', '.join(f"{arm_labels[arm]} {len(rows)}"
                                for arm, rows in data['cache_oracle_mismatches_by_arm'].items())
    cv.wrapped(76, 928, f"Boundary serving: {data['boundary_counts']['measured_http']} measured requests. "
               f"Cache-count oracle mismatches: {mismatch_counts}. "
               'Visible-answer quality is separate; no KV-byte equality claim.', 9, 1250, color=style.MUTED)

    cv.card(990, 1430)
    cv.panel_title(1022, 'Same patched source: fixed policy vs automatic policy',
                   '60-second mixed fixtures · p95 TTFT in seconds · synthetic workload evidence, not production validation')
    cv.text(76, 1110, 'Workload', 10, bold=True)
    cv.text(560, 1110, 'Hot requests: fixed → auto', 10, bold=True)
    cv.text(955, 1110, 'Cold requests: fixed → auto', 10, bold=True)
    for index, row in enumerate(data['policy']):
        ident = row['identity']
        label = f"{'128K arrivals' if ident['profile'] == 'periodic-128k' else 'Short prefill'} · C{ident['concurrency']}"
        y = 1160 + index * 43
        cv.text(76, y, label, 10.5)
        for x, metric in ((560, 'hot_ttft_p95_seconds'), (955, 'cold_ttft_p95_seconds')):
            cv.text(x, y, f"{row['fixed'][metric]:.2f} → {row['auto'][metric]:.2f}", 11, bold=True)
    cv.wrapped(76, 1362, 'Stock hit the scalar-restore crash during the 128K fixture. Later stock cells are '
               'blocked by that crash, not matched measurements. Above: patched fixed vs patched auto only.',
               9, 1240, color=style.MUTED)
    pins = ' · '.join(f"{arm_labels[arm]} {image.rsplit('sha256:', 1)[-1][:12]}"
                      for arm, image in data['images'].items())
    cv.footer([f'Pins: {pins}. Newer API/image and #685 are not included.',
               'Raw source hashes, numerical values and scopes: summary-r27.json. Qualification is not promotion.'])
    cv.fig.savefig(path, dpi=style.DPI, facecolor=style.BG)
    style.plt.close(cv.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2]/'results/r27')
    args = parser.parse_args()
    data = collect(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out/'summary-r27.png'
    render(data, image)
    data['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
    data['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out/'summary-r27.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'image': str(image), 'sources': len(data['sources']),
                      'matched_speed_rows': len(data['speed']), 'policy_rows': len(data['policy'])}))


if __name__ == '__main__':
    main()
