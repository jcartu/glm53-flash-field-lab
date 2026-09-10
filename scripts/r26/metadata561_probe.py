#!/usr/bin/env python3
"""Exercise packaged #561 metadata refresh on a real GPU without target weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def owner_count(length: int, rank: int, interleave: int) -> int:
    return sum((position // interleave) % 4 == rank for position in range(length))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = {'world_size_parameter': 4, 'ranks': [0, 1, 2, 3], 'interleaves': [1, 2, 4],
            'active_requests': 129, 'buffer_capacity': 160, 'graph_replay_increments': [1, 7],
            'checks': ['enumerated ownership lengths', 'zero padded tail', 'stable storage through graph replay',
                       'actual B12X builder refresh', 'missing-global fail closed', 'accepted-count refresh'],
            'scope': 'One GPU with all four DCP rank parameters; no inter-GPU collective or native indexer-planner claim'}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if args.out is None:
        parser.error('--out is required for GPU execution')
    import torch
    import vllm.v1.attention.backends.utils as utils
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseMetadataBuilder

    report = {'schema': 'metadata561-gpu-probe/v1', 'plan': plan, 'cases': [],
              'source_sha256': hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest(),
              'implementation_present': hasattr(utils, 'refresh_dcp_local_seq_lens_')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save() -> None:
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.out)
    if not report['implementation_present']:
        report.update({'passed': False, 'classification': 'implementation_not_present'})
        save()
        return
    for interleave in plan['interleaves']:
        for rank in plan['ranks']:
            row = {'rank': rank, 'interleave': interleave, 'passed': False}
            try:
                global_lens = torch.arange(129, dtype=torch.int32, device='cuda')
                local_lens = torch.full((160,), -77, dtype=torch.int32, device='cuda')
                accepted = torch.full((160,), 7, dtype=torch.int32, device='cuda')
                builder = SimpleNamespace(dcp_world_size=4, dcp_rank=rank,
                                          cp_kv_cache_interleave_size=interleave,
                                          requires_glm_next_selector_metadata=True)
                metadata = SimpleNamespace(dcp_global_seq_lens=global_lens,
                                           seq_lens=local_lens, num_reqs=129,
                                           selector_num_accepted_tokens=accepted)
                pointers = (global_lens.data_ptr(), local_lens.data_ptr(), accepted.data_ptr())
                def update() -> None:
                    B12xMLASparseMetadataBuilder.update_draft_decode_metadata(builder, metadata)
                def verify() -> None:
                    expected = [owner_count(length, rank, interleave) for length in global_lens.cpu().tolist()] + [0] * 31
                    assert local_lens.cpu().tolist() == expected
                    assert accepted.cpu().tolist() == [1] * 160
                    assert pointers == (global_lens.data_ptr(), local_lens.data_ptr(), accepted.data_ptr())
                warm_stream = torch.cuda.Stream()
                warm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm_stream):
                    update()
                warm_stream.synchronize()
                verify()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    update()
                for increment in plan['graph_replay_increments']:
                    global_lens.add_(increment)
                    accepted.fill_(23)
                    graph.replay()
                    verify()
                metadata.dcp_global_seq_lens = None
                rejected = False
                try:
                    update()
                except RuntimeError as error:
                    rejected = 'global sequence lengths' in str(error)
                assert rejected
                row.update({'passed': True, 'cuda_graph_replays': 2, 'padded_tail_zero': True,
                            'persistent_addresses': True, 'missing_global_rejected': True})
            except Exception as error:
                row['error'] = repr(error)
            report['cases'].append(row)
            save()
    report['passed'] = len(report['cases']) == 12 and all(row['passed'] for row in report['cases'])
    save()
    print(json.dumps({'out': str(args.out), 'cases': len(report['cases']), 'passed': report['passed']}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
