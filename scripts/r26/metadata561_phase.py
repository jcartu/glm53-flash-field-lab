#!/usr/bin/env python3
"""Isolated #561 GPU metadata checks, after the R26 model measurement boundary."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import runtime as rt

PROBE = Path(__file__).with_name('metadata561_probe.py')
R27 = 'voipmonitor/vllm@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512'
SOURCES = Path('/home/josh/omp-workspace/drock-lmcache/r26-battery/metadata-561-source')


def main() -> dict:
    rt.stop()
    summary = {'schema': 'metadata561-qualification/v1', 'scope': 'Actual packaged B12X metadata update and CUDA-graph replay, DCP4 rank parameters on one GPU; model E2E remains separate',
               'source_manifest': str(SOURCES / 'manifest.json'),
               'probe_sha256': hashlib.sha256(PROBE.read_bytes()).hexdigest(),
               'stock_r26_note': 'Source snapshot lacks refresh_dcp_local_seq_lens_; stock R26 is not labeled as carrying #561.',
               'arms': [], 'all_arms_attempted': False}
    for arm, image in (('overlay-r26', rt.OVERLAY_IMAGE), ('stock-r27', R27)):
        label = 'metadata561-' + arm
        output_dir = rt.ROOT / label
        output_dir.mkdir(exist_ok=False)
        command = ['docker', 'create', '--label', 'field-lab.battery=r26', '--init',
                   '--gpus', 'device=0', '--network', 'none', '--shm-size', '1g',
                   '--entrypoint', 'python', '-v', f'{PROBE}:/probe.py:ro',
                   '-v', f'{output_dir}:/results', image, '/probe.py', '--out', '/results/receipt.json']
        row = {'arm': arm, 'image': image, 'create_args': command, 'passed': False}
        container_id = None
        try:
            created = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
            candidate_id = created.stdout.strip()
            if len(candidate_id) != 64 or any(char not in '0123456789abcdef' for char in candidate_id):
                raise RuntimeError('Docker did not return an immutable container ID')
            container_id = candidate_id
            row['container_id'] = container_id
            row['returncode'] = rt.run(['docker', 'start', '-a', container_id], label=label + '-gpu', timeout=900)
            receipt_path = output_dir / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            expected_source = SOURCES / arm / 'vllm/v1/attention/backends/utils.py'
            row['source_matches_snapshot'] = receipt.get('source_sha256') == hashlib.sha256(expected_source.read_bytes()).hexdigest()
            row['receipt'] = str(receipt_path)
            row['passed'] = row['returncode'] == 0 and receipt.get('passed') is True and row['source_matches_snapshot']
        except Exception as error:
            row['error'] = repr(error)
        finally:
            if container_id is not None:
                removed = subprocess.run(['docker', 'rm', '-f', container_id], capture_output=True, text=True, timeout=60)
                row['removed_by_created_id'] = removed.returncode == 0
                row['passed'] = row['passed'] and row['removed_by_created_id']
            summary['arms'].append(row)
            rt.record_gate('metadata561:' + arm, row['passed'], row)
            rt.save_json('metadata561-qualification.json', summary)
    summary['all_arms_attempted'] = len(summary['arms']) == 2
    summary['passed'] = all(row['passed'] for row in summary['arms'])
    rt.save_json('metadata561-qualification.json', summary)
    return summary


if __name__ == '__main__':
    main()
