#!/usr/bin/env python3
"""Run inside the pinned image: real CUDA IPC pairs or isolated policy faults."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def policy_probe() -> dict:
    import torch
    from vllm.v1.attention.ops import cp_common as cp
    required = ('direct_cp_peer_access_enabled', '_cuda_p2p_spans_group')
    if not all(hasattr(cp, name) for name in required):
        return {'passed': False, 'scope': 'isolated policy faults, not physical bad-topology hardware', 'error': 'Image lacks the #599 peer-access policy functions'}
    cases = [
        {'name': 'all-directed-peers-available', 'expected': True},
        {'name': 'forward-peer-unavailable', 'failed_pair': (0, 1), 'expected': False},
        {'name': 'reverse-peer-unavailable', 'failed_pair': (1, 0), 'expected': False},
        {'name': 'distant-peer-unavailable', 'failed_pair': (3, 2), 'expected': False},
        {'name': 'peer-probe-error', 'probe_error': True, 'expected': False},
        {'name': 'incomplete-rank-collection', 'incomplete': True, 'expected': False},
        {'name': 'rank-collection-error', 'gather_error': True, 'expected': False},
        {'name': 'non-cuda-platform', 'cuda': False, 'expected': False},
        {'name': 'unsupported-dtype', 'supported': False, 'expected': False},
        {'name': 'symmetric-memory-unavailable', 'symmetric': False, 'expected': False},
        {'name': 'explicit-off-preserved', 'override': False, 'expected': False},
        {'name': 'explicit-on-preserved-despite-unavailable-peer', 'override': True, 'failed_pair': (0, 1), 'expected': True},
    ]
    results = []
    class Group:
        world_size = 4
        local_rank = 0
        cpu_group = None
    for case in cases:
        group = Group()
        observed_pairs = []
        def gather(slots, local_rank, group=None):
            if case.get('gather_error'):
                raise RuntimeError('injected rank-collection failure')
            slots[:] = [0, 1, 2, None if case.get('incomplete') else 3]
        def peer(src, dst):
            observed_pairs.append([src, dst])
            if case.get('probe_error'):
                raise RuntimeError('injected CUDA IPC probe failure')
            return (src, dst) != case.get('failed_pair')
        cp._cuda_p2p_spans_group.cache_clear()
        with patch.object(cp, 'current_platform', SimpleNamespace(is_cuda=lambda: case.get('cuda', True))), \
             patch.object(cp, 'symm_mem_available', case.get('symmetric', True)), \
             patch.object(cp, 'in_the_same_node_as', return_value=[True] * 4), \
             patch.object(cp.torch.distributed, 'all_gather_object', side_effect=gather), \
             patch.object(cp, 'gpu_p2p_access_check', side_effect=peer):
            observed = cp.direct_cp_peer_access_enabled(group, torch.bfloat16, case.get('override'),
                                                       (torch.bfloat16,) if case.get('supported', True) else (torch.float32,))
        results.append({'name': case['name'], 'expected': case['expected'], 'observed': observed,
                        'passed': observed is case['expected'], 'directed_pairs_consulted': observed_pairs})
    return {'passed': all(row['passed'] for row in results), 'scope': 'Isolated fault injection against the shipped Python policy; not evidence from physically broken PCIe hardware.',
            'source': str(Path(cp.__file__)), 'source_sha256': hashlib.sha256(Path(cp.__file__).read_bytes()).hexdigest(),
            'cases': results, 'override_caveat': 'The explicit VLLM_USE_DIRECT_DCP_A2A=1 override intentionally bypasses automatic peer gating; never force it on an unverified topology.'}


def physical_probe() -> dict:
    import torch
    from vllm.distributed.device_communicators.all_reduce_utils import can_actually_p2p
    count = torch.cuda.device_count()
    if count != 4:
        return {'passed': False, 'error': f'Expected 4 physical GPUs, found {count}'}
    pairs = [(src, dst) for src in range(count) for dst in range(count) if src != dst]
    capability = {f'{src}->{dst}': bool(torch.cuda.can_device_access_peer(src, dst)) for src, dst in pairs}
    # This is the real upstream cross-process probe, not the cached driver
    # capability result. A consumer modifies 1KiB of a producer's CUDA allocation
    # through an IPC handle, and both processes validate the modified bytes.
    actual = can_actually_p2p([src for src, _ in pairs], [dst for _, dst in pairs])
    transfers = {f'{src}->{dst}': bool(ok) for (src, dst), ok in zip(pairs, actual)}
    return {'passed': len(transfers) == 12 and all(transfers.values()), 'scope': 'Real cross-process CUDA IPC read/write checks for every directed pair on this four-card host.',
            'device_names': [torch.cuda.get_device_name(index) for index in range(count)],
            'driver_capability': capability, 'fresh_ipc_transfer': transfers, 'cached_probe_results_used': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('policy', 'physical'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    try:
        result = policy_probe() if args.mode == 'policy' else physical_probe()
    except Exception as error:
        result = {'passed': False, 'error': repr(error), 'mode': args.mode}
    result['started_at'] = started
    result['finished_at'] = time.time()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
