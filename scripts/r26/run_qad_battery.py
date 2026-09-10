#!/usr/bin/env python3
"""After the R26 followups finish, run the full runbook against the QAD checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil

PUBLISHED = Path('/mnt/2king/models/GLM-5.3-Flash-NVFP4')


def wait_for_process(pid: int, created: float) -> None:
    while True:
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - created) > 0.01 or process.status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(15)


def checkpoint_complete(model_dir: Path, revision: str) -> dict:
    index = model_dir / 'model.safetensors.index.json'
    report = {'model_dir': str(model_dir), 'revision': revision, 'index_present': index.exists(),
              'missing_shards': [], 'incomplete_downloads': [], 'revision_metadata_errors': []}
    if not index.exists():
        return report
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    report['shards'] = len(shards)
    report['missing_shards'] = [name for name in shards if not (model_dir / name).exists() or (model_dir / name).stat().st_size == 0]
    report['incomplete_downloads'] = [str(path) for path in model_dir.rglob('*.incomplete')]
    configuration_files = ['model.safetensors.index.json', 'config.json', 'tokenizer.json',
                           'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja']
    required = [*configuration_files, *shards]
    for name in required:
        metadata = model_dir / '.cache' / 'huggingface' / 'download' / (name + '.metadata')
        observed = metadata.read_text().splitlines()[0] if metadata.exists() else None
        if observed != revision:
            report['revision_metadata_errors'].append({'file': name, 'expected': revision, 'observed': observed})
    report['config_present'] = all((model_dir / name).is_file() for name in configuration_files)
    report['verification_scope'] = 'Completed pinned HF download and per-file revision metadata; no second whole-checkpoint hash scan.'
    report['complete'] = (report['config_present'] and not report['missing_shards']
                          and not report['incomplete_downloads'] and not report['revision_metadata_errors'])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-created', type=float, required=True)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-tag', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print('QAD BATTERY QUEUED: waiting for complete R26 qualification and clean reruns', flush=True)
    wait_for_process(args.wait_pid, args.wait_created)
    restored_path = args.upstream_root / 'production-restored.json'
    completed_path = args.upstream_root / 'clean-rerun-completed.json'
    if not restored_path.exists() or not completed_path.exists():
        raise RuntimeError('R26 clean reruns did not finish and restore production')
    restored = json.loads(restored_path.read_text())
    completed = json.loads(completed_path.read_text())
    if not restored.get('healthy') or not completed.get('complete') or completed.get('clean') != completed.get('planned'):
        raise RuntimeError('R26 clean reruns are incomplete; QAD will not overtake them')
    if restored.get('timestamp', 0) < args.wait_created or completed.get('finished_at', 0) < args.wait_created:
        raise RuntimeError('Upstream completion or restoration receipt is stale')
    if args.model_dir.resolve() == PUBLISHED.resolve():
        raise RuntimeError('Candidate directory is the published checkpoint')
    weights = checkpoint_complete(args.model_dir, args.revision)
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / 'checkpoint-verification.json').write_text(json.dumps(weights, indent=2) + '\n')
    if not weights.get('complete'):
        raise RuntimeError(f'Candidate checkpoint incomplete: {weights}')
    os.environ.update({
        'BATTERY_ROOT': str(args.output_root),
        'BATTERY_MODEL_DIR': str(args.model_dir),
        'BATTERY_MODEL_CACHE_TAG': args.model_tag,
        'BATTERY_L2_SHARED': f'/mnt/2king/lmcache-r26-battery/shared-{args.model_tag}',
    })
    import run_qualification as coordinator
    # Checkpoint-dependent comparisons lead. The entire serving runbook follows:
    # changed weights can alter memory capacity, acceptance, and workload shape.
    coordinator.PHASES = [
        ('qad-matched-published-vs-candidate', 'qad_matched_phase.py', [], 14400),
        ('quality-and-mtp-dcp-correctness', 'quality_phase.py', [], 21600),
        ('festr-matched-acceptance', 'matrix_phase.py', ['--section', 'priority'], 18000),
        ('realistic-mtp3-acceptance', 'realistic_acceptance_phase.py', [], 10800),
        ('cache-lifecycle-and-boundaries', 'cache_phase.py', [], 21600),
        ('complete-topology-matrix', 'matrix_phase.py', ['--section', 'matrix'], 21600),
        ('agentic-prefix-cache-reuse', 'agent_cache_probe.py', ['--arm', 'all'], 7200),
        ('batch-dma-and-overlay-controls', 'matrix_phase.py', ['--section', 'tuning'], 10800),
        ('tp2-capacity', 'matrix_phase.py', ['--section', 'tp2'], 7200),
        ('mixed-agent-scheduling', 'scheduler_recheck_phase.py', [], 21600),
        ('direct-dcp-peer-guard', 'peer_phase.py', [], 7200),
    ]
    coordinator.main()
    print('QAD BATTERY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
