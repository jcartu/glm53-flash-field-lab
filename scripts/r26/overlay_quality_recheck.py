#!/usr/bin/env python3
"""Retry the overlay quality arms using its documented native fairness CLI."""
from __future__ import annotations

from dataclasses import replace

import quality_phase as quality
import runtime as rt


def semantic_controls() -> None:
    # The first matched low-reasoning DFlash LAVD run was 3/24 on R26
    # versus 11/24 on R25. These controls separate speculation from
    # target execution and distinguish low from the earlier default-mode run.
    controls = []
    for release, image in [('r25', rt.R25_IMAGE), ('r26', rt.IMAGE)]:
        for spec, reasoning, runs in [('mtp0', 'explicit-low', 24), ('dflash2', 'template-default', 8)]:
            label = f'semantic-{release}-{spec}-{reasoning}-lavd'
            row = {'label': label, 'image': image, 'spec': spec, 'reasoning': reasoning,
                   'runs': runs, 'dcp': 1, 'kv': 'fp8_ds_mla'}
            try:
                if not rt.boot(label, image=image, dcp=1, spec=spec, kv='fp8_ds_mla'):
                    row['booted'] = False
                    continue
                row['booted'] = True
                if reasoning == 'explicit-low':
                    quality.run_profile(rt, quality.ProfilePlan(label, 'lavd-test', runs))
                else:
                    # Original R25 invocation: omit template kwargs and use
                    # the pinned template default; do not mix it with low.
                    rt.profile(label, 'lavd-test', runs)
                row['receipt'] = label + '.json'
            except Exception as error:
                row['error'] = repr(error)
                rt.record_gate('semantic-control:' + label, False, row)
            finally:
                rt.capture(label)
                rt.stop()
                controls.append(row)
                rt.save_json('semantic-controls.json', {'trigger': 'Matched low-reasoning LAVD difference; not yet attributed to a component.', 'controls': controls})






def main() -> None:
    mappings = []
    for original in quality.build_plan(rt):
        if original.image != rt.OVERLAY_IMAGE:
            continue
        label = original.label + '-native-recheck'
        arm = replace(original, label=label)
        entry = {'original_arm': original.label, 'recheck_arm': label, 'image': original.image,
                 'reason': 'The inherited legacy --fairness-engine flag is not accepted by the overlay. Native fixed0.4 preserves the intended compute share.',
                 'native_args': ['--prefill-compute-share', '0.4'], 'receipts': {}}
        try:
            settings = {**original.extra_env, 'FAIRNESS_ENGINE': 'none', 'PREFILL_COMPUTE_SHARE': '0.4'}
            if not rt.boot(label, image=original.image, tp=4, dcp=original.dcp, spec=original.spec,
                           cache=original.cache, kv=original.kv, extra_env=settings,
                           extra_args=entry['native_args']):
                entry['booted'] = False
                continue
            entry['booted'] = True
            if original.acceptance_probe:
                quality.run_acceptance_probe(rt, arm)
                entry['receipts'][original.label + '-acceptance.json'] = label + '-acceptance.json'
            for probe in original.probes:
                revised = replace(probe, label=probe.label + '-native-recheck')
                quality.run_probe(rt, revised)
                entry['receipts'][probe.label + '.json'] = revised.label + '.json'
            for profile in original.profiles:
                revised = replace(profile, label=profile.label + '-native-recheck')
                quality.run_profile(rt, revised)
                entry['receipts'][profile.label + '.json'] = revised.label + '.json'
        except Exception as error:
            entry['error'] = repr(error)
            rt.record_gate('overlay-quality-recheck:' + original.label, False, entry)
        finally:
            rt.capture(label)
            rt.stop()
            mappings.append(entry)
            rt.save_json('overlay-quality-recheck-map.json', {'original_attempts_preserved': True, 'rechecks': mappings})
    rt.save_json('overlay-quality-rechecks-completed.json', {'all_overlay_arms_attempted': True, 'arms': mappings})
    semantic_controls()


if __name__ == '__main__':
    main()
