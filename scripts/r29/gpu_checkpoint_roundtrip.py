#!/usr/bin/env python3
"""Real four-GPU checkpoint DMA, atomic publication, and native-FS restart proof.

Runs inside the pinned validation image without a model server. It uses the
shipped checkpoint/SHM/RPC implementations and known byte patterns, not model
outputs as a substitute for cache-byte equality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

import torch
import zmq

sys.path.insert(0, '/opt/lmcache64-validation')
from tests.v1.multiprocess import test_checkpoint_storage as support
from lmcache.integration.vllm.checkpoint_copy import CheckpointPageCopier
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest, CheckpointPrefix
from lmcache.v1.multiprocess.checkpoint_transfer import CheckpointTransferJob, UnsafeCheckpointCopyError
from lmcache.v1.multiprocess.modules.checkpoint import CheckpointModule
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import EngineDrivenTransferContext

PAGE_BYTES = 65536
SOURCE_IDS = ((1, 2), (3,), (4, 5), (6,))
DESTINATION_IDS = ((11, 12), (13,), (14, 15), (16,))
LAYOUT = {'page_bytes': PAGE_BYTES, 'component_probe': 'target-recurrent-draft-aux-v1'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def service(path):
    name = 'lmcache_l1_pool_gpu_roundtrip_' + uuid.uuid4().hex
    with support.open_store(path=path, native=True, shm_name=name) as (_, _, storage, mapping):
        module = CheckpointModule(SimpleNamespace(storage_manager=storage,
            shm_pool_info={'shm_name': name, 'pool_size': 4 * 1024 * 1024}),
            index_path=path / 'atomic-manifests.sqlite3')
        url = 'inproc://gpu-checkpoint-' + uuid.uuid4().hex
        context = zmq.Context.instance()
        server = support.MessageQueueServer(url, context)
        blocking = []
        for spec in module.get_handlers():
            server.add_handler(spec.request_type, support.get_payload_classes(spec.request_type),
                               support.get_handler_type(spec.request_type), spec.handler)
            if spec.request_type != RequestType.CHECKPOINT_CAPABILITIES:
                blocking.append(spec.request_type)
        server.add_normal_thread_pool(blocking, max_workers=4)
        server.start()
        client = support.MessageQueueClient(url, context)
        try:
            yield client, module, mapping
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = storage.report_status()['store_controller']
                if state['pending_keys_count'] == state['in_flight_task_count'] == 0:
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('Filesystem checkpoint writes did not drain before restart')
        finally:
            client.close()
            server.close()
            module.close()


def rpc(client, kind, *args):
    return client.submit_request(kind, list(args)).result(timeout=20)


def manifest(schema, variant, namespace):
    specs = [('target.attention.0', [0, 1]), ('target.recurrent.0', [16]),
             ('draft.context.0', [0, 1]), ('target-draft-auxiliary', [0])]
    groups = []
    for name, positions in specs:
        group = {'name': name, 'page_bytes': PAGE_BYTES, 'positions': positions}
        if schema == 2:
            group['content_keys'] = [hashlib.sha256(
                f'{name}:{position}:{"shared" if name.startswith("target.attention") else variant}'.encode()
            ).hexdigest() for position in positions]
        groups.append(group)
    return CheckpointManifest(f'gpu-schema{schema}-{variant}',
        CheckpointPrefix(namespace, 8192, hashlib.sha256(b'fixed-prefix').digest(), (11, 12, variant)),
        4, json.dumps({'schema_version': schema, 'worker_layout': LAYOUT, 'page_groups': groups}).encode())


def expected_pool(rank, variant):
    base = (torch.arange(32 * PAGE_BYTES, dtype=torch.int64).reshape(32, PAGE_BYTES) + rank * 37) % 251
    if variant == 2:
        for block in (3, 4, 5, 6):
            base[block] = (base[block] + 73) % 251
    return base.to(torch.uint8).pin_memory()


def bind_copiers(client, capability, pools):
    transport = EngineDrivenTransferContext()
    transport._engine_driven_context = EngineDrivenContextShm(
        metadata=EngineDrivenContextMetadata(
            layout_desc=MemoryLayoutDesc([torch.Size([PAGE_BYTES])], [torch.uint8]),
            block_size=1, use_mla=False),
        mq_client=client, mq_timeout=20, shm_name=capability.shm_name,
        pool_size=capability.pool_size)
    def validate(layout):
        require(layout == LAYOUT, 'Worker layout changed')
    copiers = [CheckpointPageCopier(pool, LAYOUT, validate, transport, capability) for pool in pools]
    return transport, copiers


def checked_copy(copier, job, lease):
    try:
        copier(job, lease)
    except UnsafeCheckpointCopyError as error:
        # The library forbids reclaiming SHM after an undrained CUDA copy.
        # Terminate this owned component process before Python closes mappings.
        print(json.dumps({'fatal': 'unsafe-undrained-checkpoint-copy', 'error': str(error)}), file=sys.stderr, flush=True)
        os._exit(2)


def store(client, entry, pools, copiers, expected):
    require(rpc(client, RequestType.CHECKPOINT_BEGIN, entry), 'Manifest admission failed')
    omitted = 0
    for rank in range(4):
        require(rpc(client, RequestType.CHECKPOINT_FIND, (entry.prefix,)) is None,
                'Partial rank set became visible before atomic publication')
        with torch.cuda.device(rank):
            producer = torch.cuda.Stream(device=rank)
            with torch.cuda.stream(producer):
                pools[rank].copy_(expected[rank], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(producer)
        lease = rpc(client, RequestType.CHECKPOINT_PREPARE_STORE, entry, rank)
        require(lease.status == 'ready', 'Store lease unavailable')
        omitted += sum(slot == (-1, 0) for group in lease.slots for slot in group)
        checked_copy(copiers[rank], CheckpointTransferJob(entry, rank, 'STORE', SOURCE_IDS, ready), lease)
        complete = rpc(client, RequestType.CHECKPOINT_FINISH_STORE, lease.lease_id, True)
        require(complete == (rank == 3), 'Publication did not wait for all four ranks')
    require(rpc(client, RequestType.CHECKPOINT_FIND, (entry.prefix,)) == entry,
            'Complete manifest was not discoverable')
    return omitted


def restore(client, entry, pools, copiers, expected, label):
    results = []
    for rank in range(4):
        with torch.cuda.device(rank):
            for group in DESTINATION_IDS:
                for block in group:
                    pools[rank][block].zero_()
            torch.cuda.synchronize(rank)
        lookup = rpc(client, RequestType.CHECKPOINT_BEGIN_RETRIEVE, entry, rank)
        require(lookup.status == 'pending', 'Retrieve admission failed')
        deadline = time.monotonic() + 30
        lease = lookup
        while lease.status == 'pending' and time.monotonic() < deadline:
            lease = rpc(client, RequestType.CHECKPOINT_POLL_RETRIEVE, lookup.lease_id)
            if lease.status == 'pending':
                time.sleep(.01)
        require(lease.status == 'ready', 'Complete payload did not become readable')
        checked_copy(copiers[rank], CheckpointTransferJob(entry, rank, 'RETRIEVE', DESTINATION_IDS), lease)
        for group_index, (source_ids, destination_ids) in enumerate(zip(SOURCE_IDS, DESTINATION_IDS)):
            for source, destination in zip(source_ids, destination_ids):
                actual = pools[rank][destination].cpu()
                require(torch.equal(actual, expected[rank][source]),
                        f'Byte mismatch after {label}: rank={rank} group={group_index} page={source}')
                results.append({'rank': rank, 'group': group_index, 'source_page': source,
                    'destination_page': destination, 'bytes': PAGE_BYTES,
                    'sha256': hashlib.sha256(actual.numpy().tobytes()).hexdigest(), 'matched': True})
        require(rpc(client, RequestType.CHECKPOINT_FINISH_RETRIEVE, lease.lease_id), 'Read lease did not finish')
    return results


def run(output):
    require(torch.cuda.device_count() == 4, 'Exactly four GPUs are required')
    output.mkdir(parents=True, exist_ok=False)
    pools = [torch.empty((32, PAGE_BYTES), dtype=torch.uint8, device=f'cuda:{rank}') for rank in range(4)]
    reports = []
    for schema in (1, 2):
        directory = output / f'schema{schema}'
        directory.mkdir()
        namespace = 'gpu-transport-' + uuid.uuid4().hex
        first = manifest(schema, 1, namespace)
        second = manifest(schema, 2, namespace)
        expected_first = [expected_pool(rank, 1) for rank in range(4)]
        expected_second = [expected_pool(rank, 2) for rank in range(4)]
        report = {'schema_version': schema, 'stages': {}}
        with service(directory) as (client, module, _):
            capability = rpc(client, RequestType.CHECKPOINT_CAPABILITIES)
            require(capability.durable_index, 'Manifest directory is not durable')
            transport, copiers = bind_copiers(client, capability, pools)
            try:
                report['first_omitted_pages'] = store(client, first, pools, copiers, expected_first)
                report['stages']['ram_first'] = restore(client, first, pools, copiers, expected_first, 'RAM first')
                report['second_omitted_pages'] = store(client, second, pools, copiers, expected_second)
                require(report['second_omitted_pages'] == (8 if schema == 2 else 0), 'Unexpected shared-attention no-copy count')
                report['stages']['ram_second'] = restore(client, second, pools, copiers, expected_second, 'RAM second')
            finally:
                copiers.clear()
                transport.close()
            state = module.report_status()['recurrent_checkpoints']
            require(state['store_leases'] == state['retrieve_leases'] == 0, 'Leases remain before storage restart')
        with service(directory) as (client, module, _):
            capability = rpc(client, RequestType.CHECKPOINT_CAPABILITIES)
            require(rpc(client, RequestType.CHECKPOINT_FIND, (second.prefix,)) == second, 'Manifest did not survive restart')
            transport, copiers = bind_copiers(client, capability, pools)
            try:
                report['stages']['native_fs_restart'] = restore(client, second, pools, copiers, expected_second, 'native FS restart')
            finally:
                copiers.clear()
                transport.close()
            state = module.report_status()['recurrent_checkpoints']
            require(state['store_leases'] == state['retrieve_leases'] == 0, 'Leases remain after restart restore')
        reports.append(report)
        (output / 'progress.json').write_text(json.dumps(reports, indent=2) + '\n')
    result = {'passed': True, 'gpu_count': 4, 'reports': reports,
        'scope': 'One worker process uses all four physical GPUs. Real checkpoint copier, pinned SHM, typed MQ, atomic all-rank publication, native filesystem storage and directory restart. Known-byte component fixtures, not a dump/equality claim for a live model KV cache.'}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'schemas': 2, 'page_comparisons': sum(len(rows) for report in reports for rows in report['stages'].values())}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)
