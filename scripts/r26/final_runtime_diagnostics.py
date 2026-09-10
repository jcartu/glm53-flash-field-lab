#!/usr/bin/env python3
"""Close remaining evidence gaps without changing the pinned runtime artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cache_phase
import runtime as rt


def replay_verdict(result: dict) -> dict:
    """Classify the measured overlay by its image, independent of output labels."""
    path = Path(result['focused_receipt'])
    raw = path.read_bytes()
    probe = json.loads(raw)
    required = (
        'event_live_and_replay', 'event_replay_payload_identity',
        'truthful_sparse_event_shape', 'cached_prefix_replay_event',
        'sparse_skipped_context_event',
    )
    checks = {key: probe.get('checks', {}).get(key) is True for key in required}
    passed = (result.get('image') == rt.OVERLAY_IMAGE
              and result.get('focused_returncode') == 0
              and probe.get('schema') == 'r26-cache-probe/v2'
              and all(checks.values()))
    gate = {'name': 'cache:drock-overlay:#645:truthful-cache-events-and-replay',
            'passed': passed, 'detail': {'receipt': str(path),
                                       'sha256': hashlib.sha256(raw).hexdigest(),
                                       'image': result.get('image'), 'checks': checks}}
    return {'result': result, 'gates': [gate], 'passed': passed,
            'classification_basis': 'Pinned image and measured v2 probe checks, not an arm-label spelling'}


def main() -> None:
    rt.stop()
    phase = cache_phase.CachePhase()
    results = {
        'schema': 'r26-final-runtime-diagnostics/v1',
        'replay_client_source_sha256': hashlib.sha256(Path(__file__).with_name('cache_probe.py').read_bytes()).hexdigest(),
        'scope': 'Model-backed replay recheck after fixing the REQ/ROUTER client mismatch, plus an explicitly different weight-loader diagnostic for native-offload startup OOM.',
        'replay': None,
        'native_loader': [],
        'metadata561': None,
    }
    try:
        import metadata561_phase
        try:
            results['metadata561'] = metadata561_phase.main()
        except Exception as error:
            results['metadata561'] = {'passed': False, 'error': repr(error)}
            rt.record_gate('final-diagnostic:metadata561', False, results['metadata561'])
        finally:
            rt.stop()
            rt.save_json('final-runtime-diagnostics.json', results)
        try:
            replay = phase.focused_arm('drock-overlay-replay', rt.OVERLAY_IMAGE)
            results['replay'] = replay_verdict(replay)
        except Exception as error:
            results['replay'] = {'passed': False, 'error': repr(error)}
            rt.record_gate('final-diagnostic:replay', False, results['replay'])
        finally:
            rt.stop()
            rt.save_json('final-runtime-diagnostics.json', results)
        for kv in ('fp8_ds_mla', 'nvfp4_ds_mla'):
            try:
                row = phase.native_cell(release='r26', image=rt.IMAGE,
                                        gpu_memory_utilization='0.93', kv=kv,
                                        role='loader-diagnostic', load_format='safetensors')
            except Exception as error:
                row = {'kv': kv, 'error': repr(error), 'passed': False}
                rt.record_gate('final-diagnostic:native-loader:' + kv, False, row)
            finally:
                rt.stop()
            results['native_loader'].append(row)
            rt.save_json('final-runtime-diagnostics.json', results)
        import scheduler_lane_cap_recheck
        scheduler_lane_cap_recheck.main()
        results['lane_cap_recheck'] = str(rt.ROOT / 'scheduler-lane-cap-recheck.json')
        results['all_diagnostics_attempted'] = True
        results['interpretation'] = 'Any safetensors-loader success is a separately qualified workaround. It does not erase the default InstantTensor/cumem OOM on R26 at either tested GMU.'
        rt.save_json('final-runtime-diagnostics.json', results)
    finally:
        rt.stop()


if __name__ == '__main__':
    main()
