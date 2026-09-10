#!/usr/bin/env python3
"""Pinned R27 arms using the existing serial field-lab runtime."""
from __future__ import annotations

import re
from pathlib import Path

import runtime as rt

IMAGES = {
    'stock': 'voipmonitor/vllm@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512',
    'patched': 'ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:5dcd9c0ee44485f1bf733af4387144425deb87e9349154396457b0e2b3bab5e5',
    'auto': 'ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:da9ca76623f951a5de982cf7bc297d1a077de746b707a1fa420342b3af9d86e1',
}
SOURCE_ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r27-source-20260906')
SOURCE_DIRS = {'stock': 'stock-r27', 'patched': 'patched-r27', 'auto': 'patched-auto-r27'}
AUTO_ENV = {
    'PREFILL_COMPUTE_SHARE': 'auto',
    'PREFILL_COMPUTE_HALF_LIFE': 'responsive',
    'MAX_PARALLEL_PREFILLS': 'auto',
    'PREFILL_POLICY': 'decode-aware',
    'DECODE_REFILL_TARGET': 'auto',
    'VLLM_SERVER_DEV_MODE': '1',
}


def ensure_scope() -> None:
    root = rt.ROOT.resolve()
    if root.parent != SOURCE_ROOT.parent or not root.name.startswith('r27-'):
        raise RuntimeError('R27 execution requires its own BATTERY_ROOT under drock-lmcache/r27-*')
    if rt.NAME != 'r27-test':
        raise RuntimeError('R27 execution requires BATTERY_CONTAINER=r27-test')
    if not (SOURCE_ROOT / 'manifest.json').is_file():
        raise RuntimeError('Pinned R27 source snapshot is missing')


def boot(label: str, *, arm: str, spec: str = 'mtp3', dcp: int = 1,
         cache: str = 'vram', kv: str = 'fp8_ds_mla', batch: int = 4096,
         extra_env: dict[str, str] | None = None,
         extra_args: list[str] | None = None) -> bool:
    ensure_scope()
    if arm not in IMAGES or not re.fullmatch(r'r27-[A-Za-z0-9_-]+', label):
        raise ValueError('Unknown R27 arm or unsafe result label')
    settings = {'MAX_NUM_BATCHED_TOKENS': str(batch)}
    if cache == 'lmcache':
        settings.update({
            'LMCACHE_INSTANCE_ID': 'r27-field-qualification',
            'LMCACHE_SHM_NAME': 'r27-field-qualification',
            'LMCACHE_L2_HOST_DIR': str(rt.L2_HOST_ROOT / ('cache-phase-' + label)),
        })
    settings.update(extra_env or {})
    return rt.boot(label, image=IMAGES[arm], tp=4, dcp=dcp, spec=spec,
                   cache=cache, kv=kv, extra_env=settings, extra_args=extra_args)
