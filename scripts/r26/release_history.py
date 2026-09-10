#!/usr/bin/env python3
"""R26 phase: historical release comparison (CPU only, no GPU work).

Inventories every GLM-5.3-Flash serving build measured on this host — the lab
repo's ``results/`` tree, the local ``drock-lmcache`` result directories, and
the live R26 battery root — then writes an auditable manifest and renders
dark-theme comparison charts.

Outputs (``results/r26/``):

* ``history-manifest.json`` — every tested build, every decode receipt, every
  cell, the provenance evidence behind each row, and the comparability gaps.
* ``history-cells.csv`` — flat decode/prefill cells for spreadsheet auditing.
* ``history-main.png`` — compact JJ no-spec C16/C1 comparisons inside explicit
  configuration segments; no line crosses a break. R27 C1/C8 controls are on
  the matched-lineage chart, never squeezed into a C16 strip.
* ``history-matched.png`` — every matched lineage (speculators, prefill, KV,
  collision probe) with soft differences and inferred launches printed.
* ``history-inventory.png`` — appendix: every build we actually ran, with the
  launch it was measured under. A colour tag is a launch family, not a
  controlled comparison; the matched chart is the comparison.
* ``history-kv-evidence.json`` — server-logged KV pool vs the bench-derived
  figure, with the exact log line for every battery receipt.

R26 receipts are ingested from ``BATTERY_ROOT`` (default
``/home/josh/omp-workspace/drock-lmcache/r26-battery``) whenever they exist;
configuration is read from the ``<label>.inspect.json`` docker inspect the
runtime captures, so R26 cells join a matched lineage only when their launch
really matches. Nothing here mutates historical receipts.

R26 speed windows are additionally validated against the GPU isolation sampler
(``gpu-isolation-events.jsonl``) and the recorded ``<label>.bench.command.json``
times, mirroring ``clean_rerun_phase.py``'s rules: a window with foreign GPU
work (or without sampler coverage) is never charted. Verified-quiet reruns in
``BATTERY_ROOT/clean-reruns`` supersede the same-label primary receipt only
when image digest, model, normalised config and bench parameters all agree;
the superseded primary stays in the manifest with its values for audit.

R27 (stock / patched / patched-auto) is ingested from ``R27_BATTERY_ROOT``
(default ``/home/josh/omp-workspace/drock-lmcache/r27-battery``) through the
two completed producers only: ``r27-speed-summary.json`` (r27-speed/v1, the
stock/patched C1/C8 · ctx0+32k · 60 s plan) and
``r27-scheduler/phase-summary.json`` (glm-r27-scheduler-phase/v1, the
fixed-vs-auto policy attribution). A speed cell is charted only when the
producer gate, this generator's own isolation recheck (fully covered,
foreign-free window) and the pinned source-snapshot image all agree; an
absent or partial root renders no R27 datapoints at all, and the patched-auto
arm is always a policy control, never a release performance gain. No C16 or
QAD points exist in the R27 plan and none are invented here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import fnmatch
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/home/josh/omp-workspace/glm53-flash-field-lab')
RESULTS = REPO / 'results'
LOCAL = Path('/home/josh/omp-workspace/drock-lmcache')
DEFAULT_R26_ROOT = LOCAL / 'r26-battery'
DEFAULT_R27_ROOT = LOCAL / 'r27-battery'
R27_SOURCE_MANIFEST = LOCAL / 'r27-source-20260906' / 'manifest.json'
OUT_DIR = RESULTS / 'r26'
HOST = 'rasputin'

# --------------------------------------------------------------------------- releases
# One entry per build that produced a receipt on this host (or that we tried to
# run). ``lineage``: jj = official Jovian Judgement community image, pre-jj =
# the community r5 predecessor, drock = D-Rock pre-bake on a JJ parent,
# patch = local rebase, other = different engine and/or checkpoint.
# ``digest_evidence`` names where the digest was read from.
RELEASES = [
    dict(id='r5', name='community r5', lineage='pre-jj', date='2026-08-30',
         tag='voipmonitor/vllm:glm53-flash-nvfp4-dflash2-community-20260830-r5',
         digest='sha256:1cb556807752cff49ec3383e193a7ffb4f0ccf5e58aa9c08996417899d536fe7',
         digest_evidence=['docker images --digests (local pull)', 'scripts/serve-lmcache-glm53.sh header'],
         note='Pre-Jovian community image. Measured through the local r5+LMCache-branch build '
              '(local/lmcache-glm53:drock-6e479686) with the connector off, so the runtime is r5.'),
    dict(id='r7', name='JJ r7', lineage='jj', date='2026-08-30',
         tag='voipmonitor/vllm:jovian-judgement-community-20260830-r7',
         digest='sha256:488ddf752938b5ab17e3083dd7d5bb84f418bc3f8856f93cc514c8b66abbe4c6',
         digest_evidence=['drock-lmcache/glm53-runbook.md (rtx6kpro glm-5.3-flash.md, commit 790db5a7)',
                          'scripts/serve-r7.sh BASE', 'docker images --digests'],
         note='The digest scripts/spec-matrix.sh labels IMG_R8 is this r7 digest; the 2 Sep ctx0 chart '
              'cell called "r8 mtp0" therefore ran on r7. Recorded here under r7.'),
    dict(id='r8', name='JJ r8', lineage='jj', date='2026-08-31',
         tag='voipmonitor/vllm:jovian-judgement-community-20260831-r8',
         digest='sha256:827a64ce0cea267aad843b3d521a47d742a6e78b502eaec7c05b4ae8bf403194',
         digest_evidence=['docker images --digests (tag present locally)', 'drock-lmcache/tipping-point.sh IMG'],
         note='Control matrix + >16-stream cliff hunt + speculator tipping-point sweep.'),
    dict(id='r10', name='JJ r10', lineage='jj', date='2026-08-31',
         tag='voipmonitor/vllm:jovian-judgement-community-20260831-r10',
         digest='sha256:488b05f81b4eb821c287102b536601ece86fa5a8249a18661e12d7d528dc26a2',
         digest_evidence=['docker images --digests (tag present locally, pulled 17:45Z; receipt 19:36Z)'],
         note='Launch script not retained; receipt file name plus the same-evening Discord post '
              '("matched config vs r8: DFlash2 MXFP8, DCP4, seqs 16").'),
    dict(id='r12', name='JJ r12', lineage='jj', date='2026-09-01',
         tag='voipmonitor/vllm:jovian-judgement-community-20260901-r12',
         digest='sha256:80dc3c3481255c123b3fe9ff164a879b7a141292389d29b0fd04a8472e6bf15d',
         digest_evidence=['docker images --digests', 'scripts/spec-matrix.sh IMG_R12'],
         note='First official image with LMCache wiring (LMCACHE_ENABLED=1).'),
    dict(id='r15', name='JJ r15', lineage='jj', date='2026-09-02',
         tag='voipmonitor/vllm:jovian-judgement-community-20260902-r15',
         digest='sha256:d9ca4c299fdb8e6176d70f36c8f98a687474eef807b7a71fe6fee9f154208fe0',
         digest_evidence=['docker images --digests', 'scripts/spec-matrix.sh IMG_R15',
                          'drock-lmcache/r15-lmcache-integration.md'],
         note='Full 0-128k matrix (no spec) plus ctx0 chart cells.'),
    dict(id='r17', name='JJ r17', lineage='jj', date='2026-09-02',
         tag='voipmonitor/vllm:jovian-judgement-community-20260902-r17',
         digest='sha256:6d33da5f57dad4d68bc7331917bf7b5d4fb5e46d65826d3737aabe8605b78dc6',
         digest_evidence=['docker images --digests', 'drock-lmcache/r17-matrix.sh'],
         note='ctx0 chart cells, acceptance repeats, Lavd quality profiles, and the _destroyed/reaper '
              'custom-config arms (docker-compose-glm53-r17-ourconfig-*.yml).'),
    dict(id='r18', name='JJ r18', lineage='jj', date='2026-09-02',
         tag='voipmonitor/vllm:jovian-judgement-community-20260902-r18',
         digest='sha256:0d1c2d4bce7052e2f7346e93ae87b90b6f4bf31f5cb6c980eecc1cf282b821bc',
         digest_evidence=['docker images --digests (tag present locally, created 19:40Z; receipt 03 Sep 01:23Z)'],
         note='Launch script not retained. KV pool 1,998,848 (3904 x 512) equals the r17 DCP1 no-spec '
              'geometry, so the launch was almost certainly the r17-matrix.sh cell; treated as inferred.'),
    dict(id='r20', name='JJ r20', lineage='jj', date='2026-09-03',
         tag='voipmonitor/vllm:jovian-judgement-community-20260903-r20',
         digest='sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340',
         digest_evidence=['docker images --digests', 'drock-lmcache/r20-battery.sh', 'drock-lmcache/overnight-r20.sh'],
         note='Full DCP1 battery, fairness engine, production candidate (DCP4 / 1M / DFlash K7).'),
    dict(id='r22', name='JJ r22', lineage='jj', date='2026-09-04',
         tag='voipmonitor/vllm:jovian-judgement-community-20260904-r22',
         digest='sha256:284784e685aa0377f1cf63a312a364fc884b02beb98949d6886624edbddb3806',
         digest_evidence=['docker images --digests', 'drock-lmcache/r22-battery.sh'],
         note='DCP1 battery, corruption hunt, chat-template probe, production cutover.'),
    dict(id='r24', name='JJ r24', lineage='jj', date='2026-09-04',
         tag='voipmonitor/vllm:jovian-judgement-community-20260904-r24',
         digest='sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd',
         digest_evidence=['docker inspect receipts (*.inspect.json, label local-inference.release.name)',
                          'results/r24/report-data.json artifact block'],
         note='18-config battery (DCP1/2/4, FP8/NVFP4 KV, GPU/LMCache/native).'),
    dict(id='r25', name='JJ r25', lineage='jj', date='2026-09-04',
         tag='voipmonitor/vllm:jovian-judgement-community-20260904-r25',
         digest='sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804',
         digest_evidence=['docker inspect receipts', 'results/r25/provenance.json', 'results/r25/source.lock'],
         note='Same battery as r24; vLLM/B12X byte-identical to r24, LMCache retrieve path changed.'),
    dict(id='r26', name='JJ R26', lineage='jj', date='2026-09-05',
         tag='voipmonitor/vllm@sha256:d0592ea9…',
         digest='sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68',
         digest_evidence=['worker-contract.txt', 'r26-battery/provenance.json', 'docker images --digests'],
         note='Ingested from the R26 battery root when receipts exist; no numbers are shown before that.'),
    dict(id='r27', name='JJ r27', lineage='jj', date='2026-09-06',
         tag='voipmonitor/vllm:jovian-judgement-community-20260906-r27',
         digest='sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512',
         digest_evidence=['r27-source-20260906/manifest.json (pinned run image)', 'r26-battery/testing-pivot-r27-20260906.json'],
         note='Stock R27. Not yet GPU-tested on this host: cells appear only from producer-validated R27 battery receipts.'),
    dict(id='r27-patched', name='R27 patched eval 185b4aaa', lineage='patch', date='2026-09-06',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache:r27-patched-eval-185b4aaa',
         digest='sha256:5dcd9c0ee44485f1bf733af4387144425deb87e9349154396457b0e2b3bab5e5',
         digest_evidence=['r27-source-20260906/manifest.json (pinned run image)', 'r26-battery/testing-pivot-r27-20260906.json'],
         note='#674 restore fix + #676 DCP reuse fix + a338ca71 (contention-only timing, geometry-independent auto lanes) on the exact R27 '
              'source. Eval candidate, not a release; compared against stock only inside its own matched lineages.'),
    dict(id='r27-auto', name='R27 patched auto (as-shipped)', lineage='patch', date='2026-09-06',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache:r27-patched-auto-eval-185b4aaa',
         digest='sha256:da9ca76623f951a5de982cf7bc297d1a077de746b707a1fa420342b3af9d86e1',
         digest_evidence=['r27-source-20260906/manifest.json (pinned run image)', 'r26-battery/testing-pivot-r27-20260906.json'],
         note='Same patched source with the auto prefill-policy defaults baked as shipped. Auto-vs-fixed policy differences are '
              'scheduler attribution controls, never release performance gains.'),
    # ---- D-Rock pre-bakes (JJ parents, separate lineage)
    dict(id='drock-bdb5bdc2', name='D-Rock LMCache candidate bdb5bdc2', lineage='drock', date='2026-08-31',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:bdb5bdc2…',
         digest='sha256:bdb5bdc2dca4e196b5f9239c08d543635ca6f5ed4b0a435635fedc92e937f58a',
         digest_evidence=['drock-lmcache/drock-pedigree.md', 'configs/drock-tp4-dcp4-production.yaml'],
         status='no_receipt',
         note='r7 parent + LMCache. Private (unauthorized) during the 31 Aug field test; the public '
              'assembly attempts are in results/REPORT-lmcache-field-test.md. No benchmark receipt exists '
              'for this digest; superseded the same day by 353d7.'),
    dict(id='drock-353d7', name='D-Rock 353d7 (r7 + LMCache)', lineage='drock', date='2026-08-31',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:353d7e15…',
         digest='sha256:353d7e1597559b7928baf4bfa749b3bf4e7309ef239eb285f32531d4c1a4d13b',
         digest_evidence=['configs/new-glm53-lmcache-mtp3-tp4-dcp4-production.yaml',
                          'drock-lmcache/new-glm53-lmcache-dflash2-production-pedigree.md'],
         note='MTP3 + LMCache on the r7 parent, D-Rock compose verbatim.'),
    dict(id='drock-r12-auto1024', name='D-Rock r12-cache-auto1024', lineage='drock', date='2026-09-02',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache:20260901-r12-cache-auto1024-47f86087-d6e402b2',
         digest='sha256:87ec2835b07f0fc03dbb36b6a7c2a79bae0842f82eead76f3b03c13dafad1d67',
         digest_evidence=['configs/glm53-jj-r12-cache-auto1024-production-intent.yaml', 'docker images --digests'],
         note='r12 parent, automatic 1024-token split pages, LMCache.'),
    dict(id='drock-r15-cache', name='D-Rock r15-cache-complete', lineage='drock', date='2026-09-02',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache:20260902-r15-cache-complete-7e35913b-d6e402b2',
         digest='sha256:d36eb52e626e8fccecb80a77d7ba4ab8850027d2a3f7a26d160944ab300a3a92',
         digest_evidence=['drock-lmcache/glm53-jj-r15-cache-complete-production-v7.yaml', 'docker images --digests'],
         note='r15 parent + PR 549/575 + native offload fixes + LMCache d6e402b2.'),
    dict(id='drock-r18-fairness', name='D-Rock r18 fairness test images', lineage='drock', date='2026-09-03',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache:20260903-r18-fairness-{engines-5d42195d,auto-6862ff3e}-test',
         digest='sha256:036ad54e878edc7f72a13060371499033df7acad5690f31f68f1922c5070b7cf / sha256:b3904e5fd4128c5e4cdbfdf0e44b64581047d35ac13380407ed7c68a3093cc7d',
         digest_evidence=['drock-lmcache/glm53-jj-r18-fairness-api.md', 'docker images --digests'],
         note='Fairness-engine pre-bakes. Only the collision probe (collision-auto-c1.json) is attributable; '
              'no decode matrix was recorded on these images.'),
    dict(id='r26-overlay', name='D-Rock R26 Python overlay', lineage='drock', date='2026-09-05',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:27fe7a2f…',
         digest='sha256:27fe7a2f1df6d01e824cd24d6b83119edea471f4ec4997aa122d82c670133236',
         digest_evidence=['worker-contract.txt', 'r26-battery/provenance.json'],
         note='Separately labelled candidate, not stock R26 (#561, #574, #643/#645, #648, #599; '
              'source 7db6a6d2). Ingested from the R26 battery root when receipts exist.'),
    # ---- local patches
    dict(id='r25-pr646', name='r25 + draft PR646 (local rebase)', lineage='patch', date='2026-09-05',
         tag='local/glm53-r25-pr646:test',
         digest='sha256:70a158cd6825074c8a335b8783c06326132f1cb5a2a167efa47ea136da0e43b4',
         digest_evidence=['scripts/r25/run_pr646.py PATCHED', 'results/r25/pr646-image.json'],
         note='Draft PR rebased on exact r25; three-arm field test. Not a release.'),
    # ---- other engines / checkpoints (comparison notes only)
    dict(id='sglang-accel', name='SGLang accel (local/sglang:glm-5.3-flash-accel)', lineage='other', date='2026-08-31',
         tag='local/sglang:glm-5.3-flash-accel',
         digest='sha256:721aafb443325b313ae0e181d346a0e99ab7a7495a48a903b8acbf32cedf3685',
         digest_evidence=['scripts/serve-glm53-flash-nvfp4.sh IMAGE', 'docker images --digests'],
         note='Different engine (SGLang, EAGLE/DFlash draft path). Same NVFP4 target checkpoint.'),
    dict(id='tr3-exl3', name="brandon's TR3 EXL3 4bpw (TP2)", lineage='other', date='2026-09-02',
         tag='verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2',
         digest='sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692',
         digest_evidence=['configs/tr3-compose.yaml', 'docker images --digests'],
         note='Different checkpoint (EXL3 4bpw), different vLLM build (0.1.dev20111), TP2/DCP2, 4 slots, '
              '98k context. Single-stream specialist profile.'),
    dict(id='drock-nonflash-exl3', name='D-Rock nonflash EXL3 (TR3 3.0bpw)', lineage='other', date='2026-09-03',
         tag='ghcr.io/yatesdr/jovian-judgement-glm53-nonflash-exl3:20260902-r15-dcp4-mtp3-fp8',
         digest='sha256:7d0b33cbbb47e0f1da78cfa6858d7d2e766b94544bdcb118b3ec3082eb8de682',
         digest_evidence=['drock-lmcache/glm53-nonflash-exl3-r15-dcp4-mtp3-fp8-production.yaml', 'docker images --digests'],
         note='JJ r15-derived runtime but a different checkpoint (GLM-5.3-EXL3-TR3-3.0bpw), 64-token pages, '
              '900k context. Not a Flash-NVFP4 measurement.'),
    dict(id='r22-qad', name='r22 image + QAD-Step1750 checkpoint', lineage='other', date='2026-09-04',
         tag='voipmonitor/vllm:jovian-judgement-community-20260904-r22 + /mnt/2king/models/GLM-5.3-Flash-NVFP4-QAD-Step1750',
         digest='sha256:284784e685aa0377f1cf63a312a364fc884b02beb98949d6886624edbddb3806',
         digest_evidence=['drock-lmcache/qad-hunt-fixed.sh', 'drock-lmcache/isource-repro-qad.sh'],
         note='Same r22 runtime, different (QAD) target weights. Speed differences are checkpoint effects.'),
]
RELEASE_BY_ID = {r['id']: r for r in RELEASES}
RELEASE_BY_DIGEST = {r['digest']: r['id'] for r in RELEASES if '/' not in r['digest']}
RELEASE_BY_DIGEST[RELEASE_BY_ID['drock-r18-fairness']['digest'].split(' / ')[0]] = 'drock-r18-fairness'
RELEASE_BY_DIGEST[RELEASE_BY_ID['drock-r18-fairness']['digest'].split(' / ')[1]] = 'drock-r18-fairness'

# --------------------------------------------------------------------------- launch configs
# Normalised launch configuration. Keys in MATCH_FIELDS must agree for two
# receipts to sit in one matched lineage; SOFT_FIELDS are reported as caveats.
MATCH_FIELDS = ('tp', 'dcp', 'seqs', 'maxlen', 'batched', 'gmu', 'kv', 'cache', 'fairness', 'gather', 'kda')
SOFT_FIELDS = ('capture', 'interval', 'alloc', 'ipc')


def cfg(base: dict | None = None, **kw):
    """Normalised launch config: defaults, then a launcher base (SR7/SM/…), then per-receipt overrides."""
    c = dict(tp=4, dcp=1, seqs=16, maxlen=262144, batched=4096, gmu='0.90', kv='fp8', cache='vram',
             fairness='off', gather='default', kda='flashkda', draft='none', spec='none',
             capture='default', interval='default', alloc='default', ipc='host', mtp_head='default')
    c.update(base or {})
    c.update(kw)
    return c


# serve-r7.sh defaults (the r7/r8/r10 DCP4 CKV-gather control launcher).
SR7 = dict(dcp=4, gather='1', seqs=16, capture='128', spec='dflash2-k7', draft='mxfp8', alloc='expandable_off')
# spec-matrix.sh / r17-matrix.sh / r20-battery.sh DCP1 launcher (16 slots, GMU 0.90).
SM = dict(dcp=1, seqs=16)
# D-Rock cache overlay composes (DCP4, NVFP4 KV, LMCache, 96 slots, GMU 0.93, 202752 context).
DROCK_CACHE = dict(dcp=4, seqs=96, maxlen=202752, gmu='0.93', kv='nvfp4', cache='lmcache', gather='auto', interval='8', ipc='private-shm')

# Receipts whose launch is known from a retained script/compose rather than a docker inspect.
# grade: script = retained launcher proves the config; inferred = launcher not retained, config
# reconstructed from receipt geometry + same-day scripts/posts; filename = only the file name says so.
def R(path, release, config, grade, evidence, note='', kind='decode'):
    return dict(path=path, release=release, config=config, grade=grade, evidence=evidence, note=note, kind=kind)


CURATED = [
    # ---- community r5 (pre-JJ), via local/lmcache-glm53:drock-6e479686 with the connector off
    R('bench-arm0-lmcache-off.json', 'r5', cfg(dcp=4, seqs='image-default', spec='dflash2-k?', draft='mxfp8', gather='default',
                                             alloc='expandable_on', capture='default'),
      'script', ['scripts/serve-lmcache-glm53.sh (LMCACHE=0, DCP=4, prefix caching off)'],
      'Prefix caching disabled by the launcher; MAX_NUM_SEQS left at the image default.'),
    R('bench-arm0b-dcp1-lmcache-off.json', 'r5', cfg(dcp=1, seqs='image-default', spec='dflash2-k?', draft='mxfp8', alloc='expandable_on'),
      'script', ['scripts/serve-lmcache-glm53.sh (LMCACHE=0, DCP=1)'], 'Same arm at DCP1.'),
    # ---- r7 / r8 / r10 controls through serve-r7.sh
    R('bench-r7-control-mxfp8.json', 'r7', cfg(SR7), 'script',
      ['scripts/serve-r7.sh defaults', 'results/REPORT-lmcache-field-test.md §1'],
      'Server-reported KV pool 3,915,860 tokens (REPORT §1); the receipt figure is bench metrics x CP.'),
    R('bench-r7-c24.json', 'r7', cfg(SR7, seqs='>=24'), 'filename', ['file name; C17/C24 probe during the r8 cliff hunt'],
      'MAX_NUM_SEQS raised for the probe; exact value not retained.'),
    R('bench-r8-control-mxfp8.json', 'r8', cfg(SR7), 'inferred',
      ['scripts/serve-r7.sh with IMAGE=r8 tag (pattern used by drock-lmcache/tipping-point.sh)',
       'Discord 31 Aug 19:36Z post: matched config vs r8 (DFlash2 MXFP8, DCP4, seqs 16)'],
      'Launch command line not retained.'),
    R('bench-r8-cliff-hunt.json', 'r8', cfg(SR7, seqs='>=22'), 'filename', ['file name; C17-C22 cliff hunt'], ''),
    R('bench-r8-capture288.json', 'r8', cfg(SR7, seqs='>=27', capture='288'), 'filename', ['file name; capture-size experiment'], ''),
    R('bench-r8-c27-probe.json', 'r8', cfg(SR7, seqs='>=27'), 'filename', ['file name; 60-second C24/C27 cells'], ''),
    R('tip-dflash3.json', 'r8', cfg(SR7, seqs=32, capture='288', spec='dflash2-k3'), 'script', ['drock-lmcache/tipping-point.sh'], ''),
    R('tip-dflash5.json', 'r8', cfg(SR7, seqs=32, capture='288', spec='dflash2-k5'), 'script', ['drock-lmcache/tipping-point.sh'], ''),
    R('tip-dflash7.json', 'r8', cfg(SR7, seqs=32, capture='288', spec='dflash2-k7'), 'inferred',
      ['drock-lmcache/tipping-point.sh pattern (dflash7 cell not in the retained script)'], ''),
    R('tip-mtp3.json', 'r8', cfg(SR7, seqs=32, capture='288', spec='mtp3', draft='none'), 'inferred',
      ['drock-lmcache/tipping-point.sh pattern (retained script runs C16/24/32; this receipt is C1/4/8/16)'], ''),
    R('bench-r10-control-mxfp8.json', 'r10', cfg(SR7), 'inferred',
      ['scripts/serve-r7.sh with IMAGE=r10 tag; Discord 31 Aug 19:36Z post'], 'Launch command line not retained.'),
    # ---- r12
    R('bench-r12-lmcache-on.json', 'r12', cfg(SR7, cache='lmcache'), 'inferred',
      ['serve-r7.sh geometry (KV 2877 x 512 x CP4, model name GLM-5.3) + LMCACHE_ENABLED=1', 'results/prefix-r12.json same session'],
      'Official r12 with its own LMCache switch; DFlash2 K7.'),
    R('chart-r12-mtp0.json', 'r12', cfg(SM, capture='128'), 'script', ['scripts/spec-matrix.sh'], '', ),
    R('chart-r12-mtp3.json', 'r12', cfg(SM, capture='128', spec='mtp3'), 'script', ['scripts/spec-matrix.sh'], ''),
    # ---- r7 digest run as the "r8" chart cell
    R('chart-r8-mtp0.json', 'r7', cfg(SM, capture='128'), 'script',
      ['scripts/spec-matrix.sh IMG_R8 = sha256:488ddf75… = the r7 digest (glm53-runbook.md)'],
      'Published on 2 Sep as "r8 mtp0"; the digest is r7.'),
    # ---- r15
    R('bench-r15.json', 'r15', cfg(SM, capture='128'), 'inferred',
      ['KV pool 2,000,384 = 3907 x 512 = the DCP1 no-spec geometry of the same-day spec-matrix.sh cells',
       'Discord 2 Sep 11:43Z: "r15 mtp0 from this week\'s batteries"'],
      'Full 0-128k matrix with prefill; launcher not retained.'),
    R('chart-r15-mtp3.json', 'r15', cfg(SM, capture='128', spec='mtp3'), 'script', ['scripts/spec-matrix.sh'], ''),
    R('chart-r15-dflash2.json', 'r15', cfg(SM, capture='128', spec='dflash2-k7', draft='mxfp8'), 'script', ['scripts/spec-matrix.sh'], ''),
    # ---- r17
    R('chart-r17-mtp0.json', 'r17', cfg(SM, capture='256'), 'script', ['drock-lmcache/r17-matrix.sh'], ''),
    R('chart-r17-mtp3.json', 'r17', cfg(SM, capture='256', spec='mtp3'), 'script', ['drock-lmcache/r17-matrix.sh'], ''),
    R('chart-r17-dflash2.json', 'r17', cfg(SM, capture='256', spec='dflash2-k7', draft='mxfp8'), 'script', ['drock-lmcache/r17-matrix.sh'], ''),
    R('accept-r17-mtp1.json', 'r17', cfg(SM, capture='256', spec='mtp1'), 'script', ['drock-lmcache/r17-accept.sh'], 'Acceptance repeat.'),
    R('accept-r17-mtp3.json', 'r17', cfg(SM, capture='256', spec='mtp3'), 'script', ['drock-lmcache/r17-accept.sh'], 'Acceptance repeat.'),
    R('accept-r17-dflash2.json', 'r17', cfg(SM, capture='256', spec='dflash2-k7', draft='mxfp8'), 'script', ['drock-lmcache/r17-accept.sh'], 'Acceptance repeat.'),
    R('destroyed-mtp3.json', 'r17', cfg(dcp=1, seqs=32, maxlen=1048576, batched=8192, gmu='0.95', spec='mtp3', capture='128', interval='8',
                                        kv='fp8 (kv-cache-memory-bytes override, 2048-token pages)'),
      'script', ['drock-lmcache/docker-compose-glm53-r17-ourconfig-mtp3.yml'],
      "reaper/_destroyed custom serving config on the r17 image (30 GiB KV per rank override, B12X tuning env)."),
    R('destroyed-dfk7.json', 'r17', cfg(dcp=1, seqs=32, maxlen=1048576, batched=8192, gmu='0.95', spec='dflash2-k7', draft='mxfp8', capture='256', interval='8',
                                        kv='fp8 (kv-cache-memory-bytes override, 2048-token pages)'),
      'script', ['drock-lmcache/docker-compose-glm53-r17-ourconfig-dflash-k7.yml'],
      'reaper/_destroyed custom serving config on the r17 image (25 GiB KV per rank override).'),
    # ---- r18
    R('bench-r18.json', 'r18', cfg(SM, capture='256'), 'inferred',
      ['KV pool 1,998,848 = 3904 x 512 = r17-matrix.sh DCP1 no-spec geometry; launcher not retained'],
      'Treated as inferred; excluded from strict matched lineages.'),
    # ---- r20
    R('r20-mtp0-flashkda.json', 'r20', cfg(SM), 'script', ['drock-lmcache/r20-battery.sh'], ''),
    R('r20-dflash2.json', 'r20', cfg(SM, spec='dflash2-k7', draft='mxfp8'), 'script', ['drock-lmcache/r20-battery.sh'], ''),
    R('r20-mtp3.json', 'r20', cfg(SM, spec='mtp3'), 'script', ['drock-lmcache/r20-battery.sh'], ''),
    R('r20-mtp0-b12xkda.json', 'r20', cfg(SM, kda='b12x'), 'script', ['drock-lmcache/r20-battery.sh'], ''),
    R('overnight-r20/decode-ctx-matrix.json', 'r20',
      cfg(dcp=4, seqs=16, maxlen=1048576, capture='256', interval='1', spec='dflash2-k7', draft='mxfp8', fairness='compute_share 0.4'),
      'script', ['drock-lmcache/overnight-r20.sh config A', 'overnight-r20/run.log: GPU KV cache size 13,026,309'],
      'Production candidate: long-context C1/C4 cells to 1M.'),
    # ---- r22
    R('r22-battery/r22-mtp0-flashkda.json', 'r22', cfg(SM, capture='256', interval='1'), 'script', ['drock-lmcache/r22-battery.sh'], ''),
    R('r22-battery/r22-dflash2.json', 'r22', cfg(SM, capture='256', interval='1', spec='dflash2-k7', draft='mxfp8'), 'script', ['drock-lmcache/r22-battery.sh'], ''),
    R('r22-battery/r22-mtp3.json', 'r22', cfg(SM, capture='256', interval='1', spec='mtp3'), 'script', ['drock-lmcache/r22-battery.sh'], ''),
    R('r22-battery/decode-ctx-spot.json', 'r22',
      cfg(dcp=4, seqs=16, maxlen=1048576, capture='256', interval='1', spec='dflash2-k7', draft='mxfp8', fairness='compute_share 0.4'),
      'script', ['drock-lmcache/r22-battery.sh P4', 'r22-battery/run.log: GPU KV cache size 13,026,309'],
      'Production cutover spot cells (C1/C4 at ctx0 and 1M).'),
    # ---- D-Rock pre-bakes
    R('bench-drock353d7-mtp3-lmcache.json', 'drock-353d7',
      cfg(dcp=4, gather='1', seqs=16, capture='128', spec='mtp3', cache='lmcache'),
      'script', ['configs/new-glm53-lmcache-mtp3-tp4-dcp4-production.yaml (compose verbatim)'], ''),
    R('bench-auto1024.json', 'drock-r12-auto1024', cfg(DROCK_CACHE), 'script',
      ['configs/glm53-jj-r12-cache-auto1024-production-intent.yaml defaults (SPECULATOR=mtp, MTP_DEPTH=0)'], ''),
    R('bench-auto1024-dflash2.json', 'drock-r12-auto1024', cfg(DROCK_CACHE, spec='dflash2-k?'), 'filename',
      ['file name only; compose default DFLASH_DEPTH=2'],
      'C1 148 vs 148 for the no-spec run and an identical KV pool: speculator activation unverified.'),
    R('bench-r15cache.json', 'drock-r15-cache', cfg(DROCK_CACHE), 'script',
      ['drock-lmcache/glm53-jj-r15-cache-complete-production-v7.yaml defaults'], ''),
    # ---- other engines / checkpoints
    R('bench-sglang-mxfp8-draft.json', 'sglang-accel', cfg(tp=4, dcp='n/a', seqs='sglang default', kv='fp8 (sglang)', spec='dflash (sglang draft)', draft='mxfp8 attempted'),
      'script', ['scripts/serve-glm53-flash-nvfp4.sh', 'results/REPORT-lmcache-field-test.md §4'],
      'SGLang engine; the MXFP8 draft failed to load (REPORT §4), receipt shows the fallback run.'),
    R('bench-tr3-4bpw.json', 'tr3-exl3', cfg(tp=2, dcp=2, seqs=4, maxlen=98304, batched=2072, gmu='0.986', kv='nvfp4', spec='dflash2-k7', draft='bf16?', ipc='host'),
      'script', ['configs/tr3-compose.yaml'], 'EXL3 4bpw checkpoint, different vLLM build.'),
    R('bench-tr3-4bpw-c1-long.json', 'tr3-exl3', cfg(tp=2, dcp=2, seqs=4, maxlen=98304, batched=2072, gmu='0.986', kv='nvfp4', spec='dflash2-k7', draft='bf16?'),
      'script', ['configs/tr3-compose.yaml'], 'C1 at 64k/96k; the 96k cell errored.'),
    R('bench-nonflash-works.json', 'drock-nonflash-exl3', cfg(dcp=4, seqs='compose default', maxlen=900000, kv='fp8 (64-token pages)', spec='none'),
      'script', ['drock-lmcache/nf-serve.sh'], 'TR3 3.0bpw checkpoint on the nonflash EXL3 runtime.'),
    R('bench-nonflash-mtp3.json', 'drock-nonflash-exl3', cfg(dcp=4, seqs='compose default', maxlen=900000, kv='fp8 (64-token pages)', spec='mtp3'),
      'script', ['drock-lmcache/nf-serve-mtp.sh'], ''),
    R('qad-bench/qad-dflash2.json', 'r22-qad', cfg(SM, capture='256', interval='1', spec='dflash2-k7', draft='mxfp8'),
      'script', ['drock-lmcache/qad-hunt-fixed.sh'], 'r22 runtime, QAD-Step1750 target weights.'),
]

# Battery receipts that have no docker inspect sidecar: stem -> (release, config, evidence).
BATTERY_STEM_OVERRIDES = {
    'dcp4-vram-nvfp4-mtp0': ('r24', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', kv='nvfp4', fairness='compute_share 0.4', interval='1', gather='auto'),
                             ['drock-lmcache/r24-smoke.sh', 'dcp4-vram-nvfp4-mtp0.docker.log: "Using full-CKV gather" (launcher default = auto)']),
    'dcp4-vram-nvfp4-mtp0-extra': ('r24', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', kv='nvfp4', fairness='compute_share 0.4', interval='1', gather='auto'),
                                   ['file name + r24-smoke.sh geometry (KV 4457 x 2048 x CP4 identical to the smoke receipt); launcher not retained']),
    'native-retry': ('r24', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', kv='nvfp4', cache='native', fairness='compute_share 0.4', interval='1', gather='default', ipc='private-shm'),
                     ['scripts/r24/native_retry.sh']),
    'r25-dflash-smoke': ('r25', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.95', kv='nvfp4', cache='lmcache', spec='dflash2-k7', draft='mxfp8',
                                    fairness='compute_share 0.4', interval='1', gather='auto', ipc='private-shm'),
                         ['file name; KV geometry (5277 x 1024 x CP4) matches the R25 DCP4 LMCache DFlash arm; launcher not retained']),
    'execution-ab-r24': ('r24', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', fairness='compute_share 0.4', interval='1', gather='auto'),
                         ['scripts/r25/run_execution_ab.py (IMAGES["r24"] = ab4ff9d6…)']),
    'execution-ab-r25-a': ('r25', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', fairness='compute_share 0.4', interval='1', gather='auto'),
                           ['scripts/r25/run_execution_ab.py']),
    'execution-ab-r25-b': ('r25', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', fairness='compute_share 0.4', interval='1', gather='auto'),
                           ['scripts/r25/run_execution_ab.py']),
    'recheck-dcp4-vram-fp8-mtp0': ('r25', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', fairness='compute_share 0.4', interval='1', gather='auto'),
                                   ['scripts/r25/run_fp8_recheck.py']),
    'recheck-dcp4-vram-fp8-mtp3': ('r25', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', fairness='compute_share 0.4', interval='1', gather='auto', spec='mtp3'),
                                   ['scripts/r25/run_fp8_recheck.py']),
    'recheck-dcp4-vram-fp8-dflash2': ('r25', cfg(dcp=4, seqs=32, maxlen=1048576, gmu='0.93', fairness='compute_share 0.4', interval='1', gather='auto', spec='dflash2-k7', draft='mxfp8'),
                                      ['scripts/r25/run_fp8_recheck.py']),
    'pr646-stock-fine-256': ('r25', cfg(dcp=1, seqs=32, gmu='0.93', kv='fp8 (256/256-token pages)', spec='dflash2-k7', draft='mxfp8', fairness='compute_share 0.4', interval='1'),
                             ['scripts/r25/run_pr646.py ARMS[stock-fine-256] (stock r25 image, --prefix-match-unit 256)']),
    'pr646-stock-coarse-2048': ('r25', cfg(dcp=1, seqs=32, gmu='0.93', kv='fp8 (2048/2048-token pages)', spec='dflash2-k7', draft='mxfp8', fairness='compute_share 0.4', interval='1'),
                                ['scripts/r25/run_pr646.py ARMS[stock-coarse-2048] (stock r25 image)']),
    'pr646-pr646-decoupled-2048-256': ('r25-pr646', cfg(dcp=1, seqs=32, gmu='0.93', kv='fp8 (2048/256-token pages, PR646)', spec='dflash2-k7', draft='mxfp8', fairness='compute_share 0.4', interval='1'),
                                       ['scripts/r25/run_pr646.py ARMS[pr646-decoupled-2048-256] (local/glm53-r25-pr646:test)']),
}

# Battery directories: (root, default release for receipts whose inspect is missing and whose stem is not overridden)
BATTERIES = [
    (LOCAL / 'r24-battery', 'r24', RESULTS / 'r24'),
    (LOCAL / 'r25-battery', 'r25', RESULTS / 'r25'),
]

# Non-decode receipts kept visible in the manifest (file name pattern -> release).
OTHER_RECEIPTS = [
    ('prefix-arm0-control.json', 'r5', 'prefix'),
    ('prefix-drock353d7.json', 'drock-353d7', 'prefix'), ('eviction-drock353d7.json', 'drock-353d7', 'eviction'),
    ('prefix-r12.json', 'r12', 'prefix'),
    ('prefix-auto1024.json', 'drock-r12-auto1024', 'prefix'), ('eviction-auto1024.json', 'drock-r12-auto1024', 'eviction'),
    ('prefix-r15.json', 'r15', 'prefix'), ('collision-r15.json', 'r15', 'collision'),
    ('collision-r15-microchunk.json', 'r15', 'collision (local micro-chunk scheduler experiment; patch not retained)'),
    ('prefix-r15cache.json', 'drock-r15-cache', 'prefix'),
    ('collision-r15cache-cold.json', 'drock-r15-cache', 'collision'), ('collision-r15cache-warm.json', 'drock-r15-cache', 'collision (warm)'),
    ('collision-r15cache-warm2.json', 'drock-r15-cache', 'collision (warm)'),
    ('prefix-tr3.json', 'tr3-exl3', 'prefix'), ('collision-tr3.json', 'tr3-exl3', 'collision (32k prefill)'),
    ('lavd-lavd-mtp0.json', 'r17', 'lavd quality'), ('lavd-lavd-mtp1.json', 'r17', 'lavd quality'),
    ('lavd-lavd-mtp3.json', 'r17', 'lavd quality'), ('lavd-lavd-dflash2.json', 'r17', 'lavd quality'),
    ('collision-r18.json', 'r18', 'collision (32k prefill)'),
    ('collision-r18-guard512.json', 'r18', 'collision (32k prefill, local scheduler patch results/scheduler-r18-patched.py)'),
    ('collision-auto-c1.json', 'drock-r18-fairness', 'collision (inferred from the fairness-auto chart of the same session)'),
    ('estonia-nonflash.json', 'drock-nonflash-exl3', 'estonia quality'), ('lavd-nonflash.json', 'drock-nonflash-exl3', 'lavd quality'),
    ('r20-collision-mtp0-flashkda.json', 'r20', 'collision'), ('r20-collision-dflash2.json', 'r20', 'collision'),
    ('r20-collision-mtp3.json', 'r20', 'collision'), ('r20-collision-mtp0-b12xkda.json', 'r20', 'collision'),
    ('r20-collision-clean.json', 'r20', 'collision (label "clean"; policy not recorded in the retained script)'),
]
LOCAL_OTHER = [
    ('r22-battery/r22-collision-mtp0-flashkda.json', 'r22', 'collision'), ('r22-battery/r22-collision-dflash2.json', 'r22', 'collision'),
    ('r22-battery/r22-collision-mtp3.json', 'r22', 'collision'), ('r22-battery/needle-1m.jsonl', 'r22', '1M needles'),
    ('r22-battery/corruption-hunt.json', 'r22', 'corruption hunt'), ('r22-battery/template-probe.json', 'r22', 'chat-template tool probe'),
    ('r22-battery/fairness-c4/c4-r22.json', 'r22', 'fairness sweep C4'),
    ('overnight-r20/needle-1m.jsonl', 'r20', '1M needles'), ('overnight-r20/prefix-lmcache-mtp3.json', 'r20', 'prefix (LMCache + MTP3)'),
    ('overnight-r20/fairness-c8/c8-validation.json', 'r20', 'fairness sweep C8'),
    ('fairness-c4/c4-validation.json', 'r20', 'fairness sweep C4 (sweep-c4.sh: r20-fair container)'),
    ('fairness-sweep-c8/c8-remaining.json', 'r20', 'fairness sweep C8 (inferred: same-day r20-fair container)'),
    ('fairness-decode-heavy/chat.json', 'r20', 'fairness sweep decode-heavy (inferred)'),
    ('fairness-prefill-heavy/analytics.json', 'r20', 'fairness sweep prefill-heavy (inferred)'),
    ('qad-bench/qad-corruption-hunt.json', 'r22-qad', 'corruption hunt'), ('qad-bench/qad-hunt-fixed.json', 'r22-qad', 'corruption hunt (fixed prompts)'),
    ('qad-bench/stock-hunt-fixed.json', 'r22', 'corruption hunt (fixed prompts, stock checkpoint)'),
    ('isource-repro/metrics-trace.jsonl', 'r22', 'iSource fairness repro metrics trace'),
    ('collision-r15-cadence2.json', 'r15', 'collision (local cadence experiment; patch not retained)'),
    ('collision-r15-cadence2-budget2072.json', 'r15', 'collision (local cadence experiment; patch not retained)'),
]

# Collision probe lineage (D-Rock's decode_prefill_collision.py, 65,535-token cold prefill against one 4k decode stream).
COLLISION_SERIES = [
    ('r15', 'off', RESULTS / 'collision-r15.json', 'DCP1 · no spec · 16 slots'),
    ('r20', 'launcher default', RESULTS / 'r20-collision-mtp0-flashkda.json', 'DCP1 · no spec · 16 slots; FAIRNESS_ENGINE not passed (r20-battery.sh); 87% stall matches fairness off'),
    ('r22', 'launcher default', LOCAL / 'r22-battery/r22-collision-mtp0-flashkda.json', 'DCP1 · no spec · 16 slots; FAIRNESS_ENGINE not passed (r22-battery.sh); 42% stall and 3.7k prefill match the compute-share 0.4 signature, so the r22 image default is the likely explanation; unverified'),
    ('r24', 'off', RESULTS / 'r24/collision-dcp1-fair-off.json', 'DCP1 · no spec · 32 slots'),
    ('r24', 'compute_share 0.4', RESULTS / 'r24/collision-dcp1-fair04.json', 'DCP1 · no spec · 32 slots'),
    ('r25', 'off', RESULTS / 'r25/collision-dcp1-fair-off.json', 'DCP1 · no spec · 32 slots'),
    ('r25', 'compute_share 0.4', RESULTS / 'r25/collision-dcp1-fair04.json', 'DCP1 · no spec · 32 slots'),
]

# --------------------------------------------------------------------------- receipt parsing
KV_RE = re.compile(r'KV cache budget from vLLM metrics: ([\d,]+) tokens \((\d+) blocks x (\d+)(?:; local ([\d,]+) \u00d7 CP (\d+))?')
SERVER_KV_RE = re.compile(r'GPU KV cache size: ([\d,]+) tokens')
LABEL_RE = re.compile(r'^(?:recheck-|execution-ab-|final-)?dcp(\d)-(vram|lmcache|native)-(fp8|nvfp4)-(mtp\d|dflash2)(?:-(b12xkda|ranklocal))?')


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def is_bench_receipt(d) -> bool:
    return isinstance(d, dict) and isinstance(d.get('metadata'), dict) and isinstance(d.get('results'), list) \
        and 'concurrency_levels' in d['metadata']


def cell_status(r: dict) -> str:
    conc = r.get('concurrency') or 0
    eff = r.get('effective_concurrency') or 0
    if r.get('num_errors') or not r.get('aggregate_tps'):
        return 'error'
    if r.get('timeout_reason') or r.get('warmup_timed_out') or eff < 0.85 * conc:
        return 'underfilled'
    if r.get('capacity_limited'):
        return 'queued'
    return 'ok'


def parse_bench(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    if not is_bench_receipt(d):
        return None
    md = d['metadata']
    ev = d.get('event_log') or []
    kv = dict(bench_total=md.get('max_total_tokens'), blocks=None, block_size=None, local=None, cp=None, formula='metadata.max_total_tokens')
    for line in ev:
        m = KV_RE.search(line)
        if m:
            kv.update(bench_total=int(m.group(1).replace(',', '')), blocks=int(m.group(2)), block_size=int(m.group(3)),
                      local=int(m.group(4).replace(',', '')) if m.group(4) else None, cp=int(m.group(5)) if m.group(5) else 1,
                      formula=f'{m.group(2)} blocks x {m.group(3)}' + (f' x CP {m.group(5)}' if m.group(5) else ''))
            break
    ctxlen = next((int(l.split('model context length: ')[1].split(' ')[0].replace(',', '')) for l in ev if 'model context length' in l), None)
    engine = next((l.split('startup engine ')[1].split(' models=')[0] for l in ev if 'startup engine ' in l), None)
    cells = []
    for r in d['results']:
        cells.append(dict(conc=r['concurrency'], ctx=r['context_tokens'], tps=round(float(r.get('aggregate_tps') or 0), 1),
                          status=cell_status(r), effective=r.get('effective_concurrency'),
                          spec_accept=r.get('server_spec_accept_rate'), errors=r.get('num_errors', 0),
                          per_user=round(float(r.get('output_tps_per_user_avg') or 0), 1)))
    prefill = {}
    for k, v in (d.get('prefill') or {}).items():
        tps = v.get('client_tok_per_sec') or v.get('tok_per_sec')
        if tps:
            prefill[int(k)] = round(float(tps))
    args = (d.get('startup_diagnostics') or {}).get('args') or {}
    hostname = (d.get('startup_diagnostics') or {}).get('hostname')
    bench = dict(concurrency=md.get('concurrency_levels'), contexts=md.get('context_lengths'), duration=md.get('duration_per_test'),
                 max_tokens=md.get('max_tokens'), skip_prefill=md.get('skip_prefill'), version=md.get('version'),
                 prefill_contexts=args.get('prefill_contexts'))
    return dict(timestamp=md.get('timestamp'), model=md.get('model'), engine=engine, hostname=hostname,
                kv=kv, model_context=ctxlen, bench=bench, cells=cells, prefill=prefill,
                bench_key=bench_key(bench))


def bench_key(b: dict) -> str:
    conc = ','.join(str(c) for c in (b.get('concurrency') or []))
    ctx = ','.join(fmt_ctx(c) for c in (b.get('contexts') or []))
    return f"C{conc}·ctx{ctx}·{b.get('max_tokens')}tok·{b.get('duration'):g}s"


def fmt_ctx(c) -> str:
    return '0' if not c else (f'{c // 1024}k' if c % 1024 == 0 else str(c))


def family_key(c: dict) -> str:
    return '·'.join(f'{k}={c.get(k)}' for k in MATCH_FIELDS)


def parse_inspect(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    if isinstance(d, list):
        d = d[0] if d else None
    if not isinstance(d, dict):
        return None
    conf = d.get('Config') or {}
    env = {}
    for item in conf.get('Env') or []:
        k, _, v = item.partition('=')
        env[k] = v
    labels = conf.get('Labels') or {}
    image_ref = conf.get('Image') or ''
    digest = image_ref.split('@')[1] if '@' in image_ref else None
    host = d.get('HostConfig') or {}
    return dict(env=env, image_ref=image_ref, image_digest=digest, image_id=d.get('Image'), created=d.get('Created'),
                release_label=labels.get('local-inference.release.name'), ipc=host.get('IpcMode'), shm=host.get('ShmSize'))


def config_from_env(env: dict, ipc_mode: str | None = None) -> dict:
    c = cfg()
    c['tp'] = int(env.get('TP', 4))
    c['dcp'] = int(env.get('DCP', 1))
    c['seqs'] = int(env.get('MAX_NUM_SEQS', 16)) if env.get('MAX_NUM_SEQS') else 'image-default'
    c['maxlen'] = int(env['MAX_MODEL_LEN']) if env.get('MAX_MODEL_LEN') else 'image-default'
    c['batched'] = int(env['MAX_NUM_BATCHED_TOKENS']) if env.get('MAX_NUM_BATCHED_TOKENS') else 'image-default'
    c['gmu'] = env.get('GPU_MEMORY_UTILIZATION', 'image-default')
    c['kv'] = {'fp8_ds_mla': 'fp8', 'nvfp4_ds_mla': 'nvfp4'}.get(env.get('KV_CACHE_QUANT', ''), env.get('KV_CACHE_QUANT') or 'image-default')
    c['cache'] = env.get('CACHE_MODE', 'vram')
    eng = env.get('FAIRNESS_ENGINE', '')
    c['fairness'] = 'off' if eng in ('', 'none', 'off') else f"{eng} {env.get('PREFILL_COMPUTE_SHARE', '')}".strip()
    c['gather'] = env.get('DCP_CKV_GATHER', 'default')
    c['kda'] = env.get('GLM53_KDA_PREFILL_BACKEND', 'flashkda')
    spec = env.get('SPECULATOR', 'mtp')
    depth = env.get('MTP_DEPTH') or env.get('MTP') or env.get('NUM_SPECULATIVE_TOKENS') or '0'
    if spec.startswith('dflash'):
        k = env.get('DFLASH_DEPTH') or env.get('NUM_SPECULATIVE_TOKENS') or '?'
        c['spec'] = f'dflash2-k{k}'
        c['draft'] = 'mxfp8' if 'mxfp8' in env.get('DFLASH_MODEL', '').lower() else 'bf16?'
    else:
        c['spec'] = 'none' if str(depth) == '0' else f'mtp{depth}'
    c['capture'] = env.get('MAX_CUDAGRAPH_CAPTURE_SIZE', 'default')
    c['interval'] = env.get('PREFILL_SCHEDULE_INTERVAL', 'default')
    c['alloc'] = 'default'
    c['ipc'] = 'host' if ipc_mode == 'host' else 'private-shm'
    c['mtp_head'] = env.get('VLLM_GLM53_MTP_DRAFT_HEAD', 'default')
    return c


def config_from_label(stem: str) -> dict | None:
    m = LABEL_RE.match(stem)
    if not m:
        return None
    dcp, cache, kv, spec, variant = m.groups()
    c = cfg(dcp=int(dcp), seqs=32, maxlen=1048576, gmu='0.95' if cache == 'lmcache' else '0.93', kv=kv, cache=cache,
            fairness='compute_share 0.4', interval='1', gather='0' if variant == 'ranklocal' else 'auto',
            kda='b12x' if variant == 'b12xkda' else 'flashkda', ipc='host' if cache == 'vram' else 'private-shm')
    c['spec'] = 'none' if spec == 'mtp0' else ('dflash2-k7' if spec == 'dflash2' else spec)
    c['draft'] = 'mxfp8' if spec == 'dflash2' else 'none'
    return c


def server_kv_evidence(path: Path) -> dict | None:
    """First 'GPU KV cache size' line of a server log: value, 1-based line number, verbatim text."""
    try:
        with open(path, errors='replace') as f:
            for lineno, line in enumerate(f, 1):
                m = SERVER_KV_RE.search(line)
                if m:
                    return dict(value=int(m.group(1).replace(',', '')), line=lineno, text=line.rstrip()[:240])
    except OSError:
        return None
    return None


def server_kv_from_log(path: Path) -> int | None:
    try:
        with open(path, errors='replace') as f:
            for line in f:
                m = SERVER_KV_RE.search(line)
                if m:
                    return int(m.group(1).replace(',', ''))
    except OSError:
        return None
    return None


def resolve_local(rel: str) -> Path:
    """Historical receipts live in the repo results tree; some only in drock-lmcache."""
    for base in (RESULTS, LOCAL):
        p = base / rel
        if p.exists():
            return p
    return RESULTS / rel


# --------------------------------------------------------------------------- inventory
def collect_curated() -> list[dict]:
    rows = []
    for spec in CURATED:
        path = resolve_local(spec['path'])
        entry = dict(id=spec['path'], release=spec['release'], path=str(path), exists=path.exists(), kind=spec['kind'],
                     config=spec['config'], grade=spec['grade'], evidence=spec['evidence'], note=spec['note'])
        if path.exists():
            parsed = parse_bench(path)
            if parsed is None:
                entry['error'] = 'not an llm_decode_bench receipt'
            else:
                entry.update(parsed)
            twin = (LOCAL / spec['path']) if path.is_relative_to(RESULTS) else (RESULTS / spec['path'])
            if twin.exists() and twin != path:
                entry['duplicate_copy'] = dict(path=str(twin), identical=sha256_of(twin) == sha256_of(path))
            entry['sha256'] = sha256_of(path)
        else:
            entry['error'] = 'missing'
        entry['family'] = family_key(spec['config'])
        if 'kv' in entry:
            entry['kv']['server_total'] = None
        rows.append(entry)
    # server-reported KV where a log/report proves it
    known_server_kv = {
        'bench-r7-control-mxfp8.json': (3915860, 'results/REPORT-lmcache-field-test.md §1'),
        'overnight-r20/decode-ctx-matrix.json': (13026309, 'overnight-r20/run.log'),
        'r22-battery/decode-ctx-spot.json': (13026309, 'r22-battery/run.log'),
    }
    for row in rows:
        if row['id'] in known_server_kv and 'kv' in row:
            row['kv']['server_total'], row['kv']['server_source'] = known_server_kv[row['id']]
    return rows


def parse_launch(path: Path) -> dict | None:
    """runtime.py writes <label>.launch.json before docker run: intended env + image reference."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get('env'), dict):
        return None
    image = d.get('image') or ''
    return dict(env=d['env'], image_ref=image, image_digest=image.split('@')[1] if '@' in image else None,
                extra_args=d.get('extra_args') or [], l2_host=d.get('l2_host'))


def excluded_stems(root: Path, patterns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Stems Main has ruled out (contaminated runs): fixed patterns plus an optional excluded.json list."""
    pats = list(patterns)
    marker = root / 'excluded.json'
    if marker.exists():
        try:
            listed = json.load(open(marker))
            pats += [str(x) for x in (listed if isinstance(listed, list) else listed.get('stems', []))]
        except (OSError, ValueError, AttributeError):
            pats.append('excluded.json:unreadable')
    hits = sorted({p.name.split('.')[0] for p in root.iterdir() if any(fnmatch.fnmatch(p.name.split('.')[0], pat) for pat in pats)})
    return pats, hits


def collect_battery(root: Path, default_release: str | None, repo_copy_dir: Path | None, log_hint: str,
                    exclude: tuple[str, ...] = ()) -> tuple[list[dict], list[dict]]:
    """Every bench receipt in a battery root; configuration from docker inspect, then launch.json, then the label.
    Receipts whose stem matches ``exclude`` (or the root's excluded.json) are skipped but reported."""
    rows, boots = [], []
    if not root.exists():
        return rows, boots
    _, banned = excluded_stems(root, exclude) if exclude or (root / 'excluded.json').exists() else ([], [])
    stems_seen = set(banned)
    for p in sorted(root.glob('*.json')):
        if p.name.endswith(('.inspect.json', '.launch.json')):
            continue
        parsed = parse_bench(p)
        if parsed is None:
            continue
        stem = p.name[:-5]
        if stem in banned:
            continue
        stems_seen.add(stem)
        inspect_path = root / f'{stem}.inspect.json'
        launch_path = root / f'{stem}.launch.json'
        insp = parse_inspect(inspect_path) if inspect_path.exists() else None
        launch = parse_launch(launch_path) if launch_path.exists() else None
        entry = dict(id=f'{root.name}/{p.name}', path=str(p), exists=True, sha256=sha256_of(p),
                     kind='acceptance' if stem.startswith('acceptance-') else 'decode', note='')
        entry.update(parsed)
        if insp or launch:
            src = insp or launch
            entry['config'] = config_from_env(src['env'], insp['ipc'] if insp else 'private')
            entry['grade'] = 'inspect' if insp else 'launch'
            entry['evidence'] = [f'{inspect_path.name}: Env + Config.Image'] if insp else [f'{launch_path.name}: env + image (container inspect missing)']
            if launch:
                entry['launch'] = dict(image=launch['image_ref'], extra_args=launch['extra_args'], l2_host=launch['l2_host'])
                if insp and launch['image_digest'] and insp['image_digest'] and launch['image_digest'] != insp['image_digest']:
                    entry['note'] += f"launch.json image {launch['image_digest'][:19]} != inspect image {insp['image_digest'][:19]}; "
            digest = (insp['image_digest'] if insp else None) or launch and launch['image_digest']
            image_id = insp['image_id'] if insp else None
            entry['image'] = dict(ref=src['image_ref'], digest=digest, id=image_id,
                                  release_label=insp['release_label'] if insp else None, created=insp['created'] if insp else None)
            rel = RELEASE_BY_DIGEST.get(digest or '') or RELEASE_BY_DIGEST.get(image_id or '')
            entry['release'] = rel or f"unknown-image-{(digest or image_id or '?')[7:19]}"
        elif stem in BATTERY_STEM_OVERRIDES:
            rel, c, evidence = BATTERY_STEM_OVERRIDES[stem]
            entry.update(config=c, grade='script' if any('.sh' in e or '.py' in e for e in evidence) else 'inferred',
                         evidence=evidence, release=rel)
        else:
            c = config_from_label(stem)
            if c is None:
                entry.update(config=cfg(), grade='unknown', evidence=['no inspect/launch sidecar; label did not parse'],
                             release=default_release or 'unknown', note='configuration unknown: excluded from matched lineages')
            else:
                entry.update(config=c, grade='label', release=default_release or 'unknown',
                             evidence=[f'label pattern {stem} + {log_hint} defaults; no inspect/launch sidecar'])
        entry['family'] = family_key(entry['config'])
        dlog = root / f'{stem}.docker.log'
        if dlog.exists():
            ev = server_kv_evidence(dlog)
            entry['kv']['server_total'] = ev['value'] if ev else None
            entry['kv']['server_source'] = dlog.name
            entry['kv']['server_evidence'] = dict(log=str(dlog), **ev) if ev else None
        else:
            entry['kv']['server_total'] = None
        if repo_copy_dir is not None:
            twin = repo_copy_dir / p.name
            if twin.exists():
                entry['duplicate_copy'] = dict(path=str(twin), identical=sha256_of(twin) == entry['sha256'])
        rows.append(entry)
    # containers that booted (or failed to) without leaving a bench receipt stay visible
    SIDECARS = ('.inspect.json', '.launch.json', '.boot.log', '.docker.log')
    for side in sorted(p for p in root.iterdir() if p.name.endswith(SIDECARS)):
        stem = next(side.name[:-len(s)] for s in SIDECARS if side.name.endswith(s))
        if stem in stems_seen or stem.endswith('-final') or any(b['label'] == stem for b in boots):
            continue
        dlog = root / f'{stem}.docker.log'
        launch = parse_launch(root / f'{stem}.launch.json') if (root / f'{stem}.launch.json').exists() else None
        boots.append(dict(label=stem, release=RELEASE_BY_DIGEST.get((launch or {}).get('image_digest') or '', default_release),
                          evidence=[s.name for s in root.glob(f'{stem}.*')],
                          server_kv=server_kv_from_log(dlog) if dlog.exists() else None))
    failures = []
    for logf in sorted(root.glob('*.log')):
        try:
            for line in open(logf, errors='replace'):
                m = re.search(r'BOOT FAIL (\S+)', line)
                if m:
                    failures.append(dict(label=m.group(1), source=logf.name, line=line.strip()[:200]))
        except OSError:
            pass
    for b in boots:
        b['boot_failures'] = [f for f in failures if f['label'] == b['label']]
    for f in failures:
        if not any(b['label'] == f['label'] for b in boots):
            boots.append(dict(label=f['label'], release=default_release, evidence=[f['source']], server_kv=None, boot_failures=[f]))
    return rows, boots


# Receipts Main ruled out of the R26 record (the initial smoke pass ran while an unrelated task had
# GPU browsers open). Anything matching is skipped and listed under r26.excluded, never charted.
R26_EXCLUDE = ('smoke*',)

# --------------------------------------------------------------------------- GPU isolation validation
# clean_rerun_phase.py's coverage rules, mirrored read-only: a bench window is "covered" when the
# sampler has >= 2 rows inside [start, end], the first/last samples land within 10 s of the edges,
# and no gap between samples exceeds 10 s. Covered + no foreign process => clean.
def isolation_rows(root: Path) -> list[dict]:
    path = root / 'gpu-isolation-events.jsonl'
    if not path.exists():
        return []
    rows = []
    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def covered_window(rows: list[dict], start: float, end: float) -> list[dict] | None:
    window = [row for row in rows if start <= row.get('timestamp', 0) <= end]
    if len(window) < 2 or window[0]['timestamp'] - start > 10 or end - window[-1]['timestamp'] > 10:
        return None
    if any(b['timestamp'] - a['timestamp'] > 10 for a, b in zip(window, window[1:])):
        return None
    return window


def isolation_verdict(events: list[dict], command_path: Path) -> dict:
    """Verdict for one bench receipt from the sampler log and the recorded command times."""
    try:
        cmd = json.load(open(command_path))
    except (OSError, ValueError):
        return dict(verdict='no-command-record', evidence=f'{command_path.name} missing or unreadable')
    start, end = cmd.get('started_at'), cmd.get('finished_at')
    if not start or not end:
        return dict(verdict='no-command-record', evidence=f'{command_path.name} has no timestamps')
    base = dict(window=dict(started_at=start, finished_at=end), command=command_path.name)
    if not events:
        return dict(base, verdict='no-coverage', evidence='gpu-isolation-events.jsonl missing or empty')
    window = covered_window(events, start, end)
    if window is None:
        return dict(base, verdict='no-coverage', evidence='sampler coverage has a gap > 10 s across the bench window')
    unhealthy = [row for row in window if row.get('speed_eligible', True) is not True and not row.get('foreign')]
    if unhealthy:
        return dict(base, verdict='gpu-health-fault', samples=len(window), unhealthy_samples=len(unhealthy),
                    evidence='producer marked GPU recovery-state samples ineligible')
    dirty = [row for row in window if row.get('foreign')]
    out = dict(base, verdict='contaminated' if dirty else 'clean', samples=len(window), dirty_samples=len(dirty))
    if dirty:
        names = sorted({p.get('name', '?') for row in dirty for p in row['foreign']})
        out['foreign'] = names
        out['evidence'] = f"{len(dirty)}/{len(window)} samples show foreign GPU work ({', '.join(names)})"
    else:
        out['evidence'] = f'{len(window)}/{len(window)} samples show no foreign GPU work'
    return out


def steady_summary_for(receipt: Path) -> dict | None:
    """Producer's r26-steady-counters/v1 sidecar: interior steady windows for the same bench."""
    p = receipt.with_name(receipt.name[:-len('.json')] + '.steady-summary.json')
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        return None
    if d.get('schema') != 'r26-steady-counters/v1':
        return dict(path=str(p), schema=d.get('schema'), note='unknown schema; recorded but not used as a gate')
    cells = [dict(conc=c.get('concurrency'), ctx=c.get('context_tokens'), valid=c.get('valid'),
                  seconds=c.get('seconds'), output_tps=c.get('output_tokens_per_second'),
                  verifier_steps_per_s=c.get('aggregate_verifier_steps_per_second'),
                  acceptance_fraction=c.get('acceptance_fraction')) for c in d.get('cells') or []]
    return dict(path=str(p), schema=d['schema'], all_windows_valid=d.get('all_windows_valid'),
                method=d.get('method'), step_units=d.get('step_units'), cells=cells)


def clean_rerun_plan(root: Path) -> dict:
    """Producer ledger (clean-rerun-plan.json): result_label -> clean flag + attempts.

    The resume ledger can carry one result_label in several boot groups (the prior
    group plus appended retries). The canonical entry is unique per result_label and
    prefers the latest eligible clean cell — matching the producer's own resume
    count — falling back to the latest cell when no cell for the label is clean."""
    plan = dict(path=str(root / 'clean-rerun-plan.json'), planned=0, cells={})
    try:
        d = json.load(open(plan['path']))
    except (OSError, ValueError):
        return plan
    plan['planned'] = d.get('planned') or 0
    latest, latest_clean = {}, {}
    for entry in d.get('reruns') or []:
        for cell in entry.get('cells') or []:
            label = cell.get('result_label')
            if not label:
                continue
            rec = dict(clean=bool(cell.get('clean')), boot_label=entry.get('boot_label'), image=entry.get('image'),
                       attempts=[dict(attempt=a.get('attempt'), clean=a.get('clean'),
                                      counter_windows_valid=a.get('counter_windows_valid')) for a in cell.get('attempts') or []],
                       resume_validation_failed=cell.get('resume_validation_failed'))
            latest[label] = rec
            if rec['clean']:
                latest_clean[label] = rec
    plan['cells'] = {label: (latest_clean.get(label) or rec) for label, rec in latest.items()}
    return plan


def planned_rerun_labels(parent: Path, parent_events: list[dict]) -> list[str]:
    """The producer's planned set recomputed read-only: every primary bench command whose window was
    dirty or uncovered, plus every acceptance cell (rerun for complete steady verifier counters)."""
    labels = []
    for command_path in sorted(parent.glob('*.bench.command.json')):
        stem = command_path.name[:-len('.bench.command.json')]
        if stem.startswith('smoke'):
            continue
        try:
            command = json.load(open(command_path))
        except (OSError, ValueError):
            continue
        if command.get('returncode') != 0:
            continue
        if isolation_verdict(parent_events, command_path)['verdict'] != 'clean' or stem.startswith('acceptance-'):
            labels.append(stem)
    return labels


def collect_clean_reruns(root: Path, parent: Path) -> dict:
    """Verified-quiet reruns: producer gate (plan ledger) + our own isolation recheck + steady sidecar."""
    out = dict(root=str(root), present=root.exists(), speed=[], acceptance=[], boots=[],
               plan=clean_rerun_plan(root), planned_labels=[], pending_labels=[])
    if not root.exists():
        return out
    rows, boots = collect_battery(root, None, None, 'clean_rerun_phase.py')
    events = isolation_rows(root)
    for r in rows:
        stem = Path(r['path']).name[:-len('.json')]
        r['isolation'] = isolation_verdict(events, root / f'{stem}.bench.command.json')
        r['steady'] = steady_summary_for(Path(r['path']))
        gate = out['plan']['cells'].get(stem)
        r['clean_gate'] = dict(plan_clean=gate['clean'] if gate else None,
                               plan_attempts=gate['attempts'] if gate else None,
                               steady_all_valid=(r['steady'] or {}).get('all_windows_valid'),
                               isolation=r['isolation']['verdict'])
        r['clean_gate']['ok'] = bool(gate and gate['clean']) and r['isolation']['verdict'] == 'clean'
        if r['kind'] == 'acceptance':
            out['acceptance'].append(r)
        else:
            r['speed_usable'] = bool(r['clean_gate']['ok'])
            out['speed'].append(r)
    out['boots'] = boots
    parent_events = isolation_rows(parent)
    planned = planned_rerun_labels(parent, parent_events)
    out['planned_labels'] = planned
    clean_labels = {label for label, c in out['plan']['cells'].items() if c['clean']}
    out['pending_labels'] = [l for l in planned if l not in clean_labels]
    return out


# A clean rerun supersedes the primary receipt with the same result label only when the image
# digest, the model, the normalised launch config and the bench parameters all agree.
SUPERSEDE_FIELDS = MATCH_FIELDS + ('spec', 'draft', 'mtp_head')


def apply_supersessions(primaries: list[dict], clean_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_name = {}
    for r in primaries:
        by_name.setdefault(Path(r['path']).name, []).append(r)
    supersessions, conflicts = [], []
    for clean in clean_rows:
        if not clean.get('clean_gate', {}).get('ok'):
            continue
        cands = by_name.get(Path(clean['path']).name) or []
        if not cands:
            continue
        pri = cands[0]
        problems = []
        c_digest = (clean.get('image') or {}).get('digest')
        p_digest = (pri.get('image') or {}).get('digest')
        if c_digest and p_digest and c_digest != p_digest:
            problems.append(f'image {p_digest[:19]}… vs {c_digest[:19]}…')
        if clean.get('model') and pri.get('model') and clean['model'] != pri['model']:
            problems.append(f"model {pri['model']} vs {clean['model']}")
        for k in SUPERSEDE_FIELDS:
            if str(clean['config'].get(k)) != str(pri['config'].get(k)):
                problems.append(f"{k}: {pri['config'].get(k)} vs {clean['config'].get(k)}")
        if clean.get('bench_key') != pri.get('bench_key'):
            problems.append(f"bench {pri.get('bench_key')} vs {clean.get('bench_key')}")
        if problems:
            conflicts.append(dict(clean=clean['id'], primary=pri['id'], problems=problems))
            continue
        verdict = pri.get('isolation', {}).get('verdict')
        reason = ('primary window had foreign GPU work' if verdict == 'contaminated'
                  else 'primary window lacked sampler coverage' if verdict != 'clean'
                  else 'rerun adds complete steady-window verifier counters')
        pri['superseded_by'] = clean['id']
        pri['speed_usable'] = False
        clean['supersedes'] = pri['id']
        rec = dict(label=Path(clean['path']).name[:-len('.json')], primary=pri['id'], clean=clean['id'],
                   reason=reason, image=c_digest, primary_verdict=verdict)
        for conc in (16, 1):
            pc, cc = cell(pri, conc, 0), cell(clean, conc, 0)
            if pc and cc:
                rec[f'c{conc}'] = dict(primary=pc['tps'], clean=cc['tps'])
        supersessions.append(rec)
    return supersessions, conflicts


def oom_note(path: Path) -> str | None:
    """Root-cause line for a boot that died loading weights (native-offload arms)."""
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return None
    if 'CUDA Error: out of memory' in text:
        where = ' during InstantTensor weight load' if 'InstantTensor' in text else ''
        return f'CUDA out of memory{where}; boot failed, no benchmark was run'
    return None


def collect_r26(root: Path) -> dict:
    rows, boots = collect_battery(root, 'r26', None, 'runtime.py', exclude=R26_EXCLUDE)
    events = isolation_rows(root)
    acceptance = []
    decode = []
    for r in rows:
        stem = Path(r['path']).name[:-len('.json')]
        r['isolation'] = isolation_verdict(events, root / f'{stem}.bench.command.json')
        if r['kind'] == 'acceptance':
            acceptance.append(r)
            continue
        r['speed_usable'] = r['isolation']['verdict'] == 'clean'
        decode.append(r)
    rows = decode
    patterns, banned = excluded_stems(root, R26_EXCLUDE) if root.exists() else (list(R26_EXCLUDE), [])
    for r in rows + acceptance:
        if r['grade'] not in ('inspect', 'launch') and r['release'] == 'r26':
            r['note'] = (r['note'] + ' image not verified by inspect/launch sidecar; ').strip()
            if 'overlay' in r['id']:
                r['release'] = 'r26-overlay'
    gates, excluded_gates = [], []
    gp = root / 'gates.jsonl'
    if gp.exists():
        for line in open(gp, errors='replace'):
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except ValueError:
                g = dict(raw=line[:200])
            label = str(g.get('name', '')).split(':', 1)[-1]
            banned_gate = any(label == b or label.startswith(b + '.') for b in banned) or any(fnmatch.fnmatch(label, pat) for pat in patterns)
            (excluded_gates if banned_gate else gates).append(g)
    failed = [g for g in gates if g.get('passed') is False]
    for b in boots:
        b['boot_failures'] = b.get('boot_failures', []) + [g for g in failed if str(g.get('name', '')) == f"boot:{b['label']}"]
        if b.get('boot_failures'):
            note = oom_note(root / f"{b['label']}.docker.log")
            if note:
                b['failure_note'] = note
    interrupted = None
    marker = root / 'qualification-interrupted.json'
    if marker.exists():
        try:
            interrupted = json.load(open(marker))
        except (OSError, ValueError):
            interrupted = dict(raw='unreadable')
    return dict(root=str(root), present=root.exists(), receipts=rows, acceptance=acceptance, boots=boots, gates=gates,
                isolation_events=str(root / 'gpu-isolation-events.jsonl'), isolation_samples=len(events),
                excluded=dict(patterns=patterns, stems=banned, gates=len(excluded_gates),
                              reason='initial smoke pass ran with unrelated GPU browsers open; ruled out by Main, archived, never reported as qualified'),
                interrupted=interrupted,
                failed_gates=[dict(name=g.get('name'), detail=g.get('detail')) for g in failed],
                gate_summary=dict(total=len(gates), passed=sum(1 for g in gates if g.get('passed') is True), failed=len(failed)))


# --------------------------------------------------------------------------- R27 ingestion
# The R27 phase is produced by r27_workload_phase.py (speed plan) and r27_scheduler_phase.py
# (scheduler attribution). This generator ingests both read-only. A speed cell becomes a
# charted decode row only when every gate agrees (R27_SPEED_POLICY); the auto arm is always a
# policy control, never a release performance gain.
R27_RELEASES = ('r27', 'r27-patched', 'r27-auto')
R27_ARM_RELEASE = {'stock': 'r27', 'patched': 'r27-patched', 'auto': 'r27-auto'}
R27_SPEC = {'mtp0': 'none', 'mtp3': 'mtp3', 'dflash2': 'dflash2-k7'}
R27_KV = {'fp8_ds_mla': 'fp8', 'nvfp4_ds_mla': 'nvfp4'}
R27_IMAGE_DIGESTS = {arm: RELEASE_BY_ID[rel]['digest']
                     for arm, rel in (('stock', 'r27'), ('patched', 'r27-patched'), ('auto', 'r27-auto'))}

R27_SPEED_POLICY = ('A R27 speed cell plots only when all of these hold: the producer cell booted and executed; the producer gates '
                    'passed (counter_windows_valid and exclusive_gpu_window, i.e. speed_eligible); this generator\'s own isolation '
                    'recheck is clean (the sampler covers the recorded bench command window with >= 2 samples, first/last within 10 s '
                    'of the edges, no gap > 10 s, zero foreign GPU work); the launch/inspect image digest equals the pinned '
                    'source-snapshot run image; and the launch/inspect environment agrees with the producer arm (spec/dcp/kv). '
                    'Anything short of that stays in the manifest with its reasons and never plots.')


def r27_read_json(path: Path):
    try:
        return json.load(open(path)), None
    except (OSError, ValueError) as error:
        return None, f'{type(error).__name__}: {error}'


def r27_source_provenance() -> dict:
    """Read-only digest evidence from the pinned extracted-source snapshot manifest."""
    out = dict(path=str(R27_SOURCE_MANIFEST), present=R27_SOURCE_MANIFEST.exists(), images=[])
    if not R27_SOURCE_MANIFEST.exists():
        return out
    out['manifest_sha256'] = sha256_of(R27_SOURCE_MANIFEST)
    d, err = r27_read_json(R27_SOURCE_MANIFEST)
    if d is None:
        out['error'] = err
        return out
    if not isinstance(d, dict) or d.get('schema') != 'r27-source-snapshot/v1':
        out['schema'] = d.get('schema') if isinstance(d, dict) else None
        out['error'] = 'unexpected schema; recorded but not used as evidence'
        return out
    arm_key_of = {'stock-r27': 'stock', 'patched-r27': 'patched', 'patched-auto-r27': 'auto'}
    for image in d.get('images') or []:
        arm = image.get('arm')
        arm_key = arm_key_of.get(arm)
        pinned = R27_IMAGE_DIGESTS.get(arm_key)
        digest = str(image.get('image') or '').split('@')[-1] or None
        out['images'].append(dict(
            arm=arm, arm_key=arm_key, image=image.get('image'), gpu_used=image.get('gpu_used'),
            source_lock_sha256=None, files={}, image_digest_matches_pinned=bool(pinned and digest and digest == pinned)))
    for image, entry in zip(d.get('images') or [], out['images']):
        files = {f.get('container_path'): f.get('sha256') for f in image.get('files') or [] if isinstance(f, dict)}
        entry['files'] = files
        entry['source_lock_sha256'] = files.get('/opt/glm53-flash/source.lock')
    launcher = {e['arm']: e['files'].get('/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh')
                for e in out['images']}
    out['identical_leaf_launcher_sha256'] = bool(launcher) and len(set(launcher.values())) == 1
    return out


def collect_r27(root: Path) -> dict:
    """R27 speed + scheduler ingestion from the completed producer contracts.

    Speed cells (r27-speed/v1) become decode rows only when every gate in R27_SPEED_POLICY
    agrees; failed cells stay as gate records with their reasons. Scheduler cases
    (glm-r27-scheduler-phase/v1) become kind='scheduler' evidence rows: fixed-vs-auto and
    as-shipped-default attribution, never speed points. An absent or partial root yields no
    receipts at all — nothing zero-filled, nothing estimated."""
    out = dict(root=str(root), present=root.exists(), receipts=[], scheduler_rows=[],
               speed=dict(summary_path=str(root / 'r27-speed-summary.json'), producer_schema='r27-speed/v1',
                          present=False, cells=[], eligible=[], policy=R27_SPEED_POLICY),
               scheduler=dict(summary_path=str(root / 'r27-scheduler' / 'phase-summary.json'),
                              producer_schema='glm-r27-scheduler-phase/v1', present=False, cases=[], receipts=[]),
               source_provenance=r27_source_provenance())
    if not root.exists():
        return out
    events = isolation_rows(root)
    speed_summary, err = r27_read_json(root / 'r27-speed-summary.json')
    if speed_summary is None:
        out['speed']['error'] = err or 'missing'
    elif not isinstance(speed_summary, dict) or speed_summary.get('schema') != 'r27-speed/v1':
        out['speed']['error'] = f"unexpected schema {speed_summary.get('schema') if isinstance(speed_summary, dict) else type(speed_summary).__name__}"
    else:
        out['speed']['present'] = True
        out['speed']['all_cells_attempted'] = bool(speed_summary.get('all_cells_attempted'))
        for cell in speed_summary.get('cells') or []:
            if not isinstance(cell, dict) or not cell.get('label'):
                continue
            label = cell['label']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', str(label)):
                out['speed']['cells'].append(dict(label=str(label), arm=cell.get('arm'), skipped='unsafe label'))
                continue
            gate = dict(label=label, arm=cell.get('arm'), image=cell.get('image'), spec=cell.get('spec'),
                        dcp=cell.get('dcp'), kv=cell.get('kv'), booted=bool(cell.get('booted')),
                        executed=bool(cell.get('executed')), speed_eligible=bool(cell.get('speed_eligible')),
                        counter_windows_valid=cell.get('counter_windows_valid'),
                        exclusive_gpu_window=cell.get('exclusive_gpu_window'),
                        unavailable=cell.get('unavailable'), error=cell.get('error'))
            out['speed']['cells'].append(gate)
            if not cell.get('executed'):
                continue
            receipt_path = root / f'{label}.json'
            inspect_path = root / f'{label}.inspect.json'
            launch_path = root / f'{label}.launch.json'
            insp = parse_inspect(inspect_path) if inspect_path.exists() else None
            launch = parse_launch(launch_path) if launch_path.exists() else None
            entry = dict(id=f'{root.name}/{receipt_path.name}', path=str(receipt_path), exists=receipt_path.exists(),
                         sha256=sha256_of(receipt_path) if receipt_path.exists() else None,
                         kind='decode', note='', r27=gate, grade='unknown', evidence=[])
            parsed = parse_bench(receipt_path)
            problems = []
            if parsed is None:
                problems.append('bench receipt missing or unreadable')
                if cell.get('executed'):
                    entry['error'] = 'producer recorded executed=True but the bench receipt is missing or unreadable'
            else:
                entry.update(parsed)
            if insp or launch:
                src = insp or launch
                entry['config'] = config_from_env(src['env'], insp['ipc'] if insp else 'private')
                entry['grade'] = 'inspect' if insp else 'launch'
                entry['evidence'] = [f'{inspect_path.name}: Env + Config.Image'] if insp else [f'{launch_path.name}: env + image']
            else:
                entry['config'] = cfg()
                problems.append('no inspect/launch sidecar')
            arm = cell.get('arm')
            entry['release'] = R27_ARM_RELEASE.get(arm, f'unknown-r27-arm-{arm}')
            if arm not in R27_ARM_RELEASE:
                problems.append(f'unknown arm {arm}')
            # the producer's arm/spec/dcp/kv must agree with the launch environment
            if 'bench' in entry:
                c = entry['config']
                want_spec, want_dcp, want_kv = R27_SPEC.get(str(cell.get('spec'))), cell.get('dcp'), R27_KV.get(str(cell.get('kv')))
                if want_spec is not None and str(c.get('spec')) != want_spec:
                    problems.append(f"spec env {c.get('spec')} vs producer {cell.get('spec')}")
                if want_dcp is not None and c.get('dcp') != want_dcp:
                    problems.append(f"dcp env {c.get('dcp')} vs producer {want_dcp}")
                if want_kv is not None and str(c.get('kv')) != want_kv:
                    problems.append(f"kv env {c.get('kv')} vs producer {cell.get('kv')}")
            digest = (insp['image_digest'] if insp else None) or (launch['image_digest'] if launch else None)
            if insp or launch:
                entry['image'] = dict(ref=src['image_ref'], digest=digest, id=insp['image_id'] if insp else None,
                                      release_label=insp['release_label'] if insp else None, created=insp['created'] if insp else None)
            pinned = R27_IMAGE_DIGESTS.get(arm)
            if pinned and digest != pinned:
                problems.append(f"image digest {str(digest)[:19]} vs pinned {pinned[:19]}")
            if not pinned:
                problems.append('arm has no pinned image digest in the source snapshot')
            entry['isolation'] = isolation_verdict(events, root / f'{label}.bench.command.json')
            entry['steady'] = steady_summary_for(receipt_path)
            entry.setdefault('kv', {})
            dlog = root / f'{label}.docker.log'
            if dlog.exists():
                ev = server_kv_evidence(dlog)
                entry['kv']['server_total'] = ev['value'] if ev else None
                entry['kv']['server_source'] = dlog.name
                entry['kv']['server_evidence'] = dict(log=str(dlog), **ev) if ev else None
            else:
                entry['kv']['server_total'] = None
            entry['family'] = family_key(entry['config'])
            producer_ok = bool(cell.get('booted')) and bool(cell.get('speed_eligible')) \
                and bool(cell.get('counter_windows_valid')) and bool(cell.get('exclusive_gpu_window'))
            entry['speed_usable'] = bool(producer_ok and entry['isolation']['verdict'] == 'clean' and not problems)
            gate['generator_problems'] = problems
            gate['generator_isolation'] = entry['isolation']['verdict']
            if problems:
                entry['note'] += 'not speed-eligible: ' + '; '.join(problems)
            if entry['speed_usable']:
                out['speed']['eligible'].append(entry['id'])
            out['receipts'].append(entry)
    sched_summary, serr = r27_read_json(root / 'r27-scheduler' / 'phase-summary.json')
    if sched_summary is None:
        out['scheduler']['error'] = serr or 'missing'
    elif not isinstance(sched_summary, dict) or sched_summary.get('schema') != 'glm-r27-scheduler-phase/v1':
        out['scheduler']['error'] = f"unexpected schema {sched_summary.get('schema') if isinstance(sched_summary, dict) else type(sched_summary).__name__}"
    else:
        out['scheduler']['present'] = True
        out['scheduler']['summary'] = dict(
            status=sched_summary.get('status'), started_at_utc=sched_summary.get('started_at_utc'),
            finished_at_utc=sched_summary.get('finished_at_utc'), plan_path=sched_summary.get('plan_path'),
            plan_sha256=sched_summary.get('plan_sha256'), source_contract_sha256=sched_summary.get('source_contract_sha256'),
            source_contract_status=sched_summary.get('source_contract_status'),
            expected_case_counts=sched_summary.get('expected_case_counts'), observed=sched_summary.get('observed'),
            actual_policy_and_control_configs=sched_summary.get('actual_policy_and_control_configs'),
            lane_sweep=sched_summary.get('lane_sweep'), trace_and_prompt_identity=sched_summary.get('trace_and_prompt_identity'),
            comparisons_and_deltas=sched_summary.get('comparisons_and_deltas'), findings=sched_summary.get('findings'),
            interpretation=sched_summary.get('interpretation'), execution_complete=sched_summary.get('execution_complete'),
            evidence_complete=sched_summary.get('evidence_complete'),
            summary_sha256_without_self=sched_summary.get('summary_sha256_without_self'))
        for key, case in (sched_summary.get('actual_policy_and_control_configs') or {}).items():
            if not isinstance(case, dict):
                continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', str(key)):
                continue
            arm = case.get('arm')
            release = R27_ARM_RELEASE.get(arm, f'unknown-r27-arm-{arm}')
            out['scheduler']['cases'].append(dict(case=key, release=release, arm=arm, role=case.get('role')))
            row = dict(id=f'{root.name}/r27-scheduler/raw/{key}-decode.json',
                       path=str(root / 'r27-scheduler' / 'raw' / f'{key}-decode.json'), exists=False, kind='scheduler',
                       release=release, speed_usable=False,
                       scheduler=dict(case=key, role=case.get('role'), arm=arm,
                                      requested_policy=case.get('requested_policy'), observed_policy=case.get('observed_policy'),
                                      image=case.get('image'), policy_sha256=case.get('policy_sha256'),
                                      launch_config_sha256=case.get('launch_config_sha256'), api_receipt=case.get('api_receipt'),
                                      policy='scheduler decode cells are policy attribution evidence (fixed vs auto, as-shipped '
                                             'defaults); they are never release speed points and never join speed lineages'))
            launch = parse_launch(root / f'r27-scheduler-{key}.launch.json') if (root / f'r27-scheduler-{key}.launch.json').exists() else None
            row['config'] = config_from_env(launch['env'], 'private') if launch else cfg()
            row['grade'] = 'launch' if launch else 'unknown'
            row['evidence'] = [f'r27-scheduler-{key}.launch.json: env + image'] if launch else \
                ['r27-scheduler/phase-summary.json case record only']
            bench_path = Path(row['path'])
            if bench_path.exists():
                parsed = parse_bench(bench_path)
                if parsed is None:
                    row['error'] = 'decode receipt unreadable'
                else:
                    row.update(parsed)
                row['exists'] = True
                row['sha256'] = sha256_of(bench_path)
                rec, _ = r27_read_json(root / 'r27-scheduler' / 'raw' / f'{key}-decode-receipt.json')
                if isinstance(rec, dict) and isinstance(rec.get('isolation'), dict):
                    row['isolation'] = rec['isolation']  # the producer's own verdict (5 s edges/gaps rule)
                else:
                    row['isolation'] = isolation_verdict(events, root / 'r27-scheduler' / 'raw' / f'{key}-decode.bench.command.json')
            row['family'] = family_key(row['config'])
            row.setdefault('kv', {})
            out['scheduler_rows'].append(row)
            out['scheduler']['receipts'].append(row['id'])
    return out


def collect_other() -> list[dict]:
    rows = []
    for rel, release, kind in OTHER_RECEIPTS:
        p = resolve_local(rel)
        rows.append(dict(path=str(p), release=release, kind=kind, exists=p.exists()))
    for rel, release, kind in LOCAL_OTHER:
        p = LOCAL / rel
        rows.append(dict(path=str(p), release=release, kind=kind, exists=p.exists()))
    for d, release in ((RESULTS / 'r24', 'r24'), (RESULTS / 'r25', 'r25')):
        for p in sorted(d.iterdir()):
            if p.is_dir() or p.suffix not in ('.json', '.jsonl', '.txt', '.lock', '.md'):
                continue
            if parse_bench(p) is not None:
                continue
            kind = re.split(r'[-.]', p.name)[0]
            rows.append(dict(path=str(p), release=release, kind=kind, exists=True))
    return rows


def collect_collisions() -> list[dict]:
    out = []
    for release, policy, path, config in COLLISION_SERIES:
        row = dict(release=release, policy=policy, path=str(path), config=config, exists=path.exists())
        if path.exists():
            try:
                d = json.load(open(path))
                s = d.get('summary') or {}
                md = d.get('metadata') or {}
                row.update(prefill_tokens=md.get('prefill_prompt_tokens'), runs=s.get('runs'),
                           baseline_tps=round(s['baseline_decode_tokens_per_second']['mean'], 1),
                           collision_tps=round(s['collision_decode_tokens_per_second']['mean'], 1),
                           slowdown_pct=round(s['decode_slowdown_percent']['mean'], 1),
                           retained_pct=round(100 - s['decode_slowdown_percent']['mean'], 1),
                           gap_p95_ms=round(1000 * s['max_decode_inter_chunk_gap_seconds']['p95']),
                           prefill_tps=round(s['prefill_tokens_per_second']['mean']))
            except (OSError, ValueError, KeyError, TypeError) as e:
                row['error'] = f'unreadable: {e}'
        out.append(row)
    return out


def docker_image_snapshot() -> list[str]:
    """Read-only evidence: local image tags and digests. Never pulls or runs anything."""
    try:
        out = subprocess.run(['docker', 'images', '--digests', '--format', '{{.Repository}}:{{.Tag}} {{.Digest}} {{.ID}} {{.CreatedAt}}'],
                             capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    keep = ('voipmonitor/vllm', 'ghcr.io/yatesdr', 'local/', 'verdictai/', 'glm53-r18')
    return sorted(l for l in out.stdout.splitlines() if l.startswith(keep))


# --------------------------------------------------------------------------- lookups
def cell(receipt: dict, conc: int, ctx: int) -> dict | None:
    for c in receipt.get('cells', []):
        if c['conc'] == conc and c['ctx'] == ctx:
            return c
    return None


def cell_value(receipt: dict, conc: int, ctx: int):
    c = cell(receipt, conc, ctx)
    if c is None:
        return None, 'absent'
    return c['tps'], c['status']


def first_receipt(rows: list[dict], release: str, spec: str, family: str | None = None, bkey: str | None = None) -> dict | None:
    hits = [r for r in rows if r.get('release') == release and r.get('exists') and 'cells' in r and r['config'].get('spec') == spec
            and (family is None or r['family'] == family) and (bkey is None or r['bench_key'] == bkey)]
    hits.sort(key=lambda r: r.get('timestamp') or '')
    return hits[0] if hits else None


# --------------------------------------------------------------------------- matched lineages
# Each lineage names its anchor receipt: every other member must match the anchor
# on MATCH_FIELDS and on the bench key. Members are declared by (release, spec)
# and resolved from the inventory, so R26 joins automatically when its launch
# and bench arguments really match the R25 anchor.
LINEAGES = [
    dict(id='ctx0-chart', title='DCP1 \u00b7 16 slots \u00b7 ctx0 \u00b7 1024-token cells',
         subtitle='festr\u2019s ctx0 chart series (spec-matrix.sh / r17-matrix.sh) \u00b7 GMU 0.90 \u00b7 C1 and C16 \u00b7 30 s cells',
         anchor=('r12', 'none'), releases=['r7', 'r12', 'r15', 'r17'],
         specs=[('none', 'no spec'), ('mtp3', 'MTP3'), ('dflash2-k7', 'DFlash K7')],
         metrics=[(16, 0, 'C16'), (1, 0, 'C1')],
         caveat='r7 is the 488ddf75 digest the original chart called "r8". r17 cells used MAX_CUDAGRAPH_CAPTURE_SIZE 256 vs 128 (not expected to move C\u226416).'),
    dict(id='dcp1-32k', title='DCP1 \u00b7 16 slots \u00b7 ctx0 + 32k \u00b7 8192-token cells',
         subtitle='r20-battery.sh \u2192 r22-battery.sh \u00b7 GMU 0.90 \u00b7 C1 and C16 at ctx0 \u00b7 30 s cells',
         anchor=('r20', 'none'), releases=['r20', 'r22'],
         specs=[('none', 'no spec'), ('mtp3', 'MTP3'), ('dflash2-k7', 'DFlash K7')],
         metrics=[(16, 0, 'C16'), (1, 0, 'C1')],
         caveat='r22 cells added MAX_CUDAGRAPH_CAPTURE_SIZE 256 and PREFILL_SCHEDULE_INTERVAL 1 (soft fields; listed, not hidden).'),
    dict(id='dcp4-ckv', title='DCP4 CKV-gather \u00b7 DFlash K7 \u00b7 16 slots',
         subtitle='serve-r7.sh control launcher \u00b7 MXFP8 draft \u00b7 GMU 0.90 \u00b7 C1 / C8 / C16 at ctx0 \u00b7 32k client prefill',
         anchor=('r7', 'dflash2-k7'), releases=['r7', 'r8', 'r10'],
         specs=[('dflash2-k7', 'DFlash K7')],
         metrics=[(16, 0, 'C16'), (8, 0, 'C8'), (1, 0, 'C1')], prefill=32768,
         caveat='r8 and r10 launch lines were not retained; geometry and the same-evening post match the r7 launcher.'),
    dict(id='battery', title='R24 \u2192 R25 \u2192 R26 \u00b7 DCP1 \u00b7 FP8 \u00b7 32 slots \u00b7 1M',
         subtitle='run_r2x_battery.py / runtime.py \u00b7 compute-share 0.4 \u00b7 docker-inspect verified \u00b7 C1 and C16 at ctx0 \u00b7 30 s cells',
         anchor=('r25', 'none'), releases=['r24', 'r25', 'r26'],
         specs=[('none', 'no spec'), ('mtp3', 'MTP3'), ('dflash2-k7', 'DFlash K7')],
         metrics=[(16, 0, 'C16'), (1, 0, 'C1')],
         caveat='R26 joins only from an inspect-verified launch in a GPU-isolation-clean window that matches the R25 anchor; contaminated or superseded primary cells never plot.'),
    dict(id='battery-dcp4', title='R24 \u2192 R25 \u2192 R26 \u00b7 DCP4 \u00b7 packed NVFP4 \u00b7 GPU-only',
         subtitle='the cache-room configuration \u00b7 no spec \u00b7 32 slots \u00b7 1M \u00b7 C1 / C16 at ctx0 \u00b7 32k client prefill \u00b7 server-reported KV pool',
         anchor=('r25', 'none'), releases=['r24', 'r25', 'r26'],
         specs=[('none', 'no spec')], metrics=[(16, 0, 'C16'), (1, 0, 'C1')], prefill=32768, kv=True,
         family_override=dict(dcp=4, kv='nvfp4', cache='vram', kda='flashkda', gather='auto'),
         caveat='r24\u2019s C16 comes from the -extra receipt (same geometry, launcher not retained).'),
    # ---- R27 stock-vs-patched candidates (r27_workload_phase.py speed plan)
    # One lineage per launch family: MATCH_FIELDS must agree inside a lineage, so DCP and KV
    # differences between the four plan modes are separate panels/blocks, never one line.
    dict(id='r27-dcp1', title='R27 stock vs patched \u00b7 DCP1 \u00b7 FP8 \u00b7 32 slots \u00b7 1M',
         subtitle='r27_workload_phase.py speed plan \u00b7 compute-share 0.4 \u00b7 C1/C8 at ctx0 + 32k \u00b7 60 s cells \u00b7 8192-token cells',
         anchor=('r27', 'none'), releases=['r27', 'r27-patched'],
         specs=[('none', 'no spec'), ('mtp3', 'MTP3')],
         metrics=[(1, 0, 'C1'), (8, 0, 'C8'), (1, 32768, 'C1 32k'), (8, 32768, 'C8 32k')], prefill=32768,
         family_override=dict(dcp=1, kv='fp8'),
         caveat='R27 cells join only from a producer-gated, fully covered clean window whose launch matches this selector and whose image '
                 'digest is the pinned source snapshot. The patched arm is an eval candidate, not a release; the R27 plan has no C16.'),
    dict(id='r27-dcp4-mtp3', title='R27 stock vs patched \u00b7 DCP4 \u00b7 FP8 \u00b7 MTP3 \u00b7 32 slots \u00b7 1M',
         subtitle='r27_workload_phase.py speed plan \u00b7 compute-share 0.4 \u00b7 C1/C8 at ctx0 + 32k \u00b7 60 s cells',
         anchor=('r27', 'mtp3'), releases=['r27', 'r27-patched'],
         specs=[('mtp3', 'MTP3')],
         metrics=[(1, 0, 'C1'), (8, 0, 'C8'), (1, 32768, 'C1 32k'), (8, 32768, 'C8 32k')], prefill=32768,
         family_override=dict(dcp=4, kv='fp8'),
         caveat='DCP4 MTP3 with FP8 KV is its own launch family; it is never drawn against the DCP1 or DFlash cells.'),
    dict(id='r27-dcp4-dflash', title='R27 stock vs patched \u00b7 DCP4 \u00b7 NVFP4 \u00b7 DFlash K7 \u00b7 32 slots \u00b7 1M',
         subtitle='r27_workload_phase.py speed plan \u00b7 compute-share 0.4 \u00b7 C1/C8 at ctx0 + 32k \u00b7 60 s cells',
         anchor=('r27', 'dflash2-k7'), releases=['r27', 'r27-patched'],
         specs=[('dflash2-k7', 'DFlash K7')],
         metrics=[(1, 0, 'C1'), (8, 0, 'C8'), (1, 32768, 'C1 32k'), (8, 32768, 'C8 32k')], prefill=32768,
         family_override=dict(dcp=4, kv='nvfp4'),
         caveat='Speculative decode: acceptance moves these numbers. The patched-auto image never joins this lineage \u2014 auto '
                 'policy defaults are scheduler attribution controls, not release performance gains.'),
]


def launch_deviations(c: dict, anchor: dict) -> tuple[list[str], list[str]]:
    hard = [f"{k}: {c.get(k)} vs {anchor.get(k)}" for k in MATCH_FIELDS if str(c.get(k)) != str(anchor.get(k))]
    soft = [f"{k}: {c.get(k)} vs {anchor.get(k)}" for k in SOFT_FIELDS if str(c.get(k)) != str(anchor.get(k))]
    return hard, soft


def same_cells(b: dict, anchor: dict) -> bool:
    """A cell is comparable when its token budget and window equal the anchor's; cell lists may differ."""
    return b.get('max_tokens') == anchor.get('max_tokens') and b.get('duration') == anchor.get('duration')


def lineage_rows(lineage: dict, rows: list[dict], pending_note=None) -> dict:
    """Resolve a lineage: the anchor fixes launch fields and bench parameters; every member metric
    is taken from the earliest receipt of that release/spec whose launch matches on MATCH_FIELDS,
    whose bench parameters match, and which actually contains the cell. Nothing is interpolated.
    ``pending_note(release, spec, selector)`` explains an empty R26 slot (e.g. clean rerun pending)."""
    anchor_release, anchor_spec = lineage['anchor']
    selector = lineage.get('family_override') or {}

    def selected(r):
        return r.get('exists') and 'cells' in r and all(str(r['config'].get(k)) == str(v) for k, v in selector.items())

    anchor = next((r for r in rows if r.get('release') == anchor_release and r['config'].get('spec') == anchor_spec and selected(r)), None)
    out = dict(id=lineage['id'], title=lineage['title'], subtitle=lineage['subtitle'], caveat=lineage['caveat'],
               metrics=[dict(conc=conc, ctx=ctx, label=label) for conc, ctx, label in lineage['metrics']],
               has_prefill=bool(lineage.get('prefill')), has_kv=bool(lineage.get('kv')), anchor=anchor['id'] if anchor else None,
               members=[], gaps=[])
    if anchor is None:
        out['gaps'].append(f'anchor receipt for {anchor_release}/{anchor_spec} not found')
        for release in lineage['releases']:
            for spec, spec_label in lineage['specs']:
                member = dict(release=release, spec=spec, spec_label=spec_label, status='pending')
                if pending_note:
                    note = pending_note(release, spec, selector)
                    if note:
                        member['note'] = note
                out['members'].append(member)
        return out
    acfg = anchor['config']
    out['family'] = anchor['family']
    out['bench'] = dict(max_tokens=anchor['bench']['max_tokens'], duration=anchor['bench']['duration'])
    for release in lineage['releases']:
        for spec, spec_label in lineage['specs']:
            cands = sorted((r for r in rows if r.get('release') == release and r['config'].get('spec') == spec and selected(r)),
                           key=lambda r: r.get('timestamp') or '')
            member = dict(release=release, spec=spec, spec_label=spec_label)
            if not cands:
                pending = release in ('r26', 'r26-overlay') or release in R27_RELEASES
                member['status'] = 'pending' if pending else 'absent'
                if pending and pending_note:
                    note = pending_note(release, spec, selector)
                    if note:
                        member['note'] = note
                if not pending:
                    out['gaps'].append(f'{release}/{spec_label}: no receipt')
                out['members'].append(member)
                continue
            usable = []
            for r in cands:
                hard, soft = launch_deviations(r['config'], acfg)
                if not same_cells(r['bench'], anchor['bench']):
                    hard = hard + [f"bench {r['bench_key']} vs {anchor['bench_key']}"]
                if not hard:
                    usable.append((r, soft))
            if not usable:
                closest = min(cands, key=lambda r: len(launch_deviations(r['config'], acfg)[0]) + (0 if same_cells(r['bench'], anchor['bench']) else 1))
                hard, soft = launch_deviations(closest['config'], acfg)
                if not same_cells(closest['bench'], anchor['bench']):
                    hard.append(f"bench {closest['bench_key']} vs {anchor['bench_key']}")
                if release in ('r26', 'r26-overlay') or release in R27_RELEASES:
                    member.update(status='pending', receipt=closest['id'], hard_deviations=hard, soft_deviations=soft,
                                  note='receipts so far do not match this lineage')
                else:
                    member.update(status='mismatch', receipt=closest['id'], hard_deviations=hard, soft_deviations=soft)
                    out['gaps'].append(f"{release}/{spec_label}: launch/bench does not match anchor ({'; '.join(hard)})")
                out['members'].append(member)
                continue
            member.update(values={}, statuses={}, sources={}, soft_deviations=sorted({d for _, s in usable for d in s}))
            for conc, ctx, label in lineage['metrics']:
                for r, _ in usable:
                    v, st = cell_value(r, conc, ctx)
                    if v is not None:
                        member['values'][label], member['statuses'][label], member['sources'][label] = v, st, r['id']
                        break
                else:
                    member['values'][label], member['statuses'][label] = None, 'absent'
                    out['gaps'].append(f'{release}/{spec_label}: no {label} cell in a matched receipt')
            if lineage.get('prefill'):
                src = next(((r['prefill'][lineage['prefill']], r['id']) for r, _ in usable if r['prefill'].get(lineage['prefill'])), (None, None))
                member['values']['prefill32k'], member['sources']['prefill32k'] = src
            if lineage.get('kv'):
                src = next(((r['kv']['server_total'], r['id']) for r, _ in usable if r['kv'].get('server_total')), (None, None))
                member['values']['kv_server'], member['sources']['kv_server'] = src
                member['values']['kv_bench'] = usable[0][0]['kv'].get('bench_total')
            grades = {r['grade'] for r, _ in usable if r['id'] in member['sources'].values()}
            member['grades'] = sorted(grades)
            member['status'] = 'matched' if grades <= {'script', 'inspect', 'launch'} else 'matched-inferred'
            if member['status'] == 'matched-inferred':
                out['gaps'].append(f"{release}/{spec_label}: launch config {'/'.join(sorted(grades - {'script', 'inspect', 'launch'}))}, not proven by a retained script or inspect")
            out['members'].append(member)
    # a speculator lineage must use one draft across releases (K7 MXFP8 vs BF16 would be a spec change)
    for spec, spec_label in lineage['specs']:
        drafts = {}
        for m in out['members']:
            if m['spec'] == spec and m.get('status', '').startswith('matched'):
                r = next(r for r in rows if r['id'] == list(m['sources'].values())[0])
                drafts[m['release']] = r['config'].get('draft')
        if len(set(drafts.values())) > 1:
            out['gaps'].append(f'{spec_label}: draft differs across releases {drafts}')
            for m in out['members']:
                if m['spec'] == spec and m.get('status', '').startswith('matched'):
                    m['status'] = 'mismatch'
                    m['hard_deviations'] = [f"draft: {drafts[m['release']]}"]
    return out


# --------------------------------------------------------------------------- overview rows
FAMILY_STYLES = [
    ('dcp1-16-chart', 'DCP1 \u00b7 16 slots \u00b7 GMU 0.90 \u00b7 1024-token ctx0 cells', '#58A6FF'),
    ('dcp1-16', 'DCP1 \u00b7 16 slots \u00b7 GMU 0.90 \u00b7 8192-token cells', '#1F6FEB'),
    ('dcp4-ckv', 'DCP4 CKV \u00b7 DFlash K7 \u00b7 16 slots', '#E3B341'),
    ('battery-dcp1', 'DCP1 \u00b7 FP8 \u00b7 32 slots \u00b7 1M \u00b7 share 0.4', '#3FB950'),
    ('battery-dcp4', 'DCP4 \u00b7 NVFP4 \u00b7 32 slots \u00b7 1M \u00b7 share 0.4', '#56D4DD'),
    ('battery-dcp4-fp8', 'DCP4 \u00b7 FP8 \u00b7 MTP3 \u00b7 32 slots \u00b7 1M \u00b7 share 0.4', '#F0883E'),
    ('battery-lmcache', 'DCP4 \u00b7 LMCache \u00b7 32 slots \u00b7 1M \u00b7 share 0.4', '#FF9E64'),
    ('prod-dcp4', 'DCP4 \u00b7 1M \u00b7 DFlash K7 \u00b7 share 0.4', '#F778BA'),
    ('drock-cache', 'D-Rock cache overlays', '#BC8CFF'),
    ('other', 'other launch / engine / checkpoint', '#8B949E'),
]
FAMILY_COLOR = {k: c for k, _, c in FAMILY_STYLES}
SHORT = {
    'r5': 'community r5', 'r7': 'JJ r7', 'r8': 'JJ r8', 'r10': 'JJ r10', 'r12': 'JJ r12', 'r15': 'JJ r15', 'r17': 'JJ r17',
    'r18': 'JJ r18', 'r20': 'JJ r20', 'r22': 'JJ r22', 'r24': 'JJ r24', 'r25': 'JJ r25', 'r26': 'JJ R26',
    'r27': 'JJ r27', 'r27-patched': 'R27 patch', 'r27-auto': 'R27 auto',
    'drock-bdb5bdc2': 'D-Rock bdb5bdc2', 'drock-353d7': 'D-Rock 353d7', 'drock-r12-auto1024': 'D-Rock r12 auto',
    'drock-r15-cache': 'D-Rock r15 cache', 'drock-r18-fairness': 'D-Rock r18 fair', 'r26-overlay': 'R26 overlay',
    'r25-pr646': 'r25 + PR646', 'sglang-accel': 'SGLang accel', 'tr3-exl3': 'EXL3 4bpw TP2',
    'drock-nonflash-exl3': 'EXL3 TR3 3bpw', 'r22-qad': 'r22 + QAD weights',
}


def family_group(r: dict) -> str:
    c = r['config']
    if r['release'] in ('sglang-accel', 'tr3-exl3', 'drock-nonflash-exl3', 'r22-qad'):
        return 'other'
    if c['cache'] == 'lmcache' and r['release'].startswith('drock'):
        return 'drock-cache'
    if c['dcp'] == 4 and c['gather'] == '1' and c['seqs'] == 16 and c['spec'] == 'dflash2-k7' and c['maxlen'] == 262144:
        return 'dcp4-ckv'
    if c['dcp'] == 1 and c['seqs'] == 16 and c['gmu'] == '0.90' and c['maxlen'] == 262144 and c['cache'] == 'vram':
        return 'dcp1-16-chart' if r['bench'].get('max_tokens') == 1024 else 'dcp1-16'
    if c['seqs'] == 32 and c['maxlen'] == 1048576 and c['fairness'] == 'compute_share 0.4' and c['cache'] == 'vram' and c['kv'] == 'fp8' and c['dcp'] == 1:
        return 'battery-dcp1'
    if c['seqs'] == 32 and c['maxlen'] == 1048576 and c['fairness'] == 'compute_share 0.4' and c['dcp'] == 4 and c['kv'] == 'nvfp4' and c['cache'] == 'vram':
        return 'battery-dcp4'
    if c['seqs'] == 32 and c['maxlen'] == 1048576 and c['fairness'] == 'compute_share 0.4' and c['dcp'] == 4 and c['kv'] == 'fp8' and c['cache'] == 'vram':
        return 'battery-dcp4-fp8'
    if c['seqs'] == 32 and c['maxlen'] == 1048576 and c['fairness'] == 'compute_share 0.4' and c['dcp'] == 4 and c['cache'] == 'lmcache':
        return 'battery-lmcache'
    if c['dcp'] == 4 and c['maxlen'] == 1048576 and c['spec'] == 'dflash2-k7' and c['fairness'] == 'compute_share 0.4' and c['seqs'] == 16:
        return 'prod-dcp4'
    return 'other'


def config_chip(c: dict) -> str:
    spec = {'none': 'no spec'}.get(c['spec'], c['spec'].replace('dflash2-k', 'DFlash K').replace('mtp', 'MTP'))
    kv = str(c['kv']).split(' ')[0].upper() + ('\u2020' if ' ' in str(c['kv']) else '')
    parts = ([f"TP{c['tp']}"] if c['tp'] != 4 else []) + [f"DCP{c['dcp']}", f"{c['seqs']} slots", kv, spec]
    if c['cache'] != 'vram':
        parts.append(c['cache'])
    if c['fairness'] != 'off':
        parts.append('share ' + c['fairness'].replace('compute_share ', ''))
    if c['maxlen'] == 1048576:
        parts.append('1M')
    return ' \u00b7 '.join(str(p) for p in parts)


# Shortest bench window a headline receipt may have; the inventory header states this window and
# every headline row measured with a different one says so on the row.
HEADLINE_WINDOW_S = 30


def pick_headline(rows: list[dict]) -> dict | None:
    """The most complete ctx0 decode receipt of at least HEADLINE_WINDOW_S seconds for a release: prefer
    no-spec, then C16 coverage. Expects pre-filtered plot rows (decode kind, clean-window, not superseded)."""
    def score(r):
        c = r['config']
        return (c['spec'] == 'none', cell(r, 16, 0) is not None, cell(r, 8, 0) is not None, len(r['cells']),
                r.get('grade') in ('script', 'inspect', 'launch'))
    rows = [r for r in rows if r.get('exists') and 'cells' in r and r.get('kind') == 'decode'
            and (r['bench'].get('duration') or 0) >= HEADLINE_WINDOW_S]
    return max(rows, key=score) if rows else None


def overview_rows(all_rows: list[dict], r26: dict, r26_status=None, r27_status=None) -> list[dict]:
    """One inventory row per release. ``all_rows`` must be the plot-filtered set (superseded and
    contaminated R26 windows already removed); r26_status(rid) returns the R26 validation line,
    r27_status(rid) the R27 one."""
    out = []
    by_release = {}
    for r in all_rows:
        by_release.setdefault(r['release'], []).append(r)
    for rel in RELEASES:
        rid = rel['id']
        rows = by_release.get(rid, [])
        head = pick_headline(rows)
        row = dict(release=rid, name=rel['name'], lineage=rel['lineage'], date=rel['date'], status=rel.get('status', 'measured'))
        if rid in ('r26', 'r26-overlay') and r26_status:
            row['validation'] = r26_status(rid)
        elif rid in R27_RELEASES and r27_status:
            row['validation'] = r27_status(rid)
        if head is None:
            short_cells = [r for r in rows if 'cells' in r and r.get('kind') == 'decode']
            if short_cells:
                row['status'] = 'smoke-only'
                row['smoke'] = ', '.join(f"{r['id'].split('/')[-1]} ({r['bench']['duration']:g} s cells)" for r in short_cells[:3])
            elif rid == 'r27-auto':
                row['status'] = 'control-only'
            elif rid in ('r26', 'r26-overlay') or rid in R27_RELEASES:
                row['status'] = 'pending'
            elif rid == 'drock-r18-fairness':
                row['status'] = 'probe-only'
            elif rel.get('status') != 'no_receipt':
                row['status'] = 'no-decode-receipt'
            row['others'] = []
            out.append(row)
            continue
        c1 = cell(head, 1, 0)
        c8 = cell(head, 8, 0)
        c16 = cell(head, 16, 0)
        row.update(status='measured', headline=head['id'], family=family_group(head), chip=config_chip(head['config']),
                   grade=head['grade'], note=head.get('note', ''), c1=c1, c8=c8, c16=c16, duration=head['bench']['duration'],
                   kv_server=head['kv'].get('server_total'), kv_bench=head['kv'].get('bench_total'),
                   prefill32k=head['prefill'].get(32768), timestamp=head['timestamp'])
        others = []
        seen = {config_chip(head['config'])}
        for r in sorted(rows, key=lambda r: r.get('timestamp') or ''):
            chip = config_chip(r['config'])
            if chip not in seen and 'cells' in r and r.get('kind') == 'decode':
                seen.add(chip)
                others.append(chip)
        row['others'] = others
        row['receipt_count'] = len([r for r in rows if 'cells' in r and r.get('kind') == 'decode'])
        out.append(row)
    return out


# --------------------------------------------------------------------------- rendering
BG, CARD, GRID, TEXT, MUTED = '#0D1117', '#161B22', '#21262D', '#E6EDF3', '#AAB7C7'
BLUE, AMBER, RED, GREEN = '#58A6FF', '#E3B341', '#F85149', '#3FB950'
DPI = 140
W = 1400


def _fig(w, h):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    return fig, ax


def _text(ax, x, y, s, size, color=TEXT, weight='normal', ha='left', va='top', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def _card(ax, x, y, w, h, fc=CARD, ec=GRID, lw=1.2):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=12', facecolor=fc, edgecolor=ec, linewidth=lw))


def _rect(ax, x, y, w, h, fc, ec='none', lw=0, zorder=2):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))


def short_name(release: str) -> str:
    rel = RELEASE_BY_ID.get(release)
    if rel is None:
        return release
    if rel['lineage'] == 'jj':
        return 'R' + release[1:]
    return SHORT.get(release, rel['name'])


def fmt_k(v) -> str:
    if v is None:
        return '\u2014'
    return f'{v / 1e6:.2f}M' if v >= 1e6 else f'{v / 1e3:.0f}K'


def fmt_tps(c: dict | None) -> tuple[str, str]:
    if c is None:
        return '\u2014', MUTED
    if c['status'] == 'error':
        return 'err', RED
    v = f"{c['tps']:,.0f}"
    if c['status'] == 'underfilled':
        return v + '*', AMBER
    return v, TEXT


def footer(ax, h, stamp):
    _text(ax, W / 2, h - 42, f'4x RTX PRO 6000 Blackwell · rasputin · rendered {stamp}', 12, MUTED, ha='center')


def tw(s: str, size: float) -> float:
    """Approximate rendered width in px for DejaVu Sans at DPI (0.6 em average advance)."""
    return len(s) * size * DPI / 72 * 0.6


def wrap_chip(chip: str, size: float, width: float) -> list[str]:
    """Wrap a ' · '-joined chip into lines that fit the column. Never truncates: the row grows."""
    parts = chip.split(' \u00b7 ')
    lines, cur = [], ''
    for p in parts:
        cand = p if not cur else f'{cur} \u00b7 {p}'
        if cur and tw(cand, size) > width:
            lines.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def render_overview(ov: list[dict], out: Path, stamp: str) -> None:
    lanes = [('jj', 'Jovian Judgement community images (voipmonitor/vllm)'), ('pre-jj', 'Pre-JJ community image'),
             ('drock', 'D-Rock pre-bakes on JJ parents \u2014 separate lineage'), ('patch', 'Local patches \u2014 not releases'),
             ('other', 'Other engines / checkpoints \u2014 comparison notes only, never release gains')]
    lanes = [(k, t) for k, t in lanes if any(r['lineage'] == k for r in ov)]
    LANE_H = 50
    cols = dict(name=88, chip=350, chip_w=370, c1=810, c8=918, c16=1032, kv=1180, pf=1322)
    messages = {'pending': 'no clean-window decode cells yet \u2014 nothing this image produced has passed GPU-isolation validation',
                'no_receipt': 'no benchmark receipt on this host (private image during the field test)',
                'probe-only': 'collision probe only \u2014 no decode matrix recorded',
                'no-decode-receipt': 'no decode receipt',
                'control-only': 'auto-policy control: the patched-auto arm is measured only through the scheduler attribution phase \u2014 '
                                'auto policy defaults are controls, never release performance gains'}

    def layout(r: dict) -> dict:
        """Row content fully wrapped at the column width; the row grows, notes are never cut."""
        if r['status'] == 'measured':
            chip_lines = wrap_chip(r['chip'], 11, cols['chip_w'])
            sub = []
            if r['duration'] != HEADLINE_WINDOW_S:
                sub.append(f"{r['duration']:g} s cells")
            if r.get('note'):
                sub.append(r['note'].rstrip('.'))
            if r['grade'] not in ('script', 'inspect', 'launch'):
                sub.append(f"launch {r['grade']}")
            if r.get('validation'):
                sub.append(r['validation'])
            if r['others']:
                sub.append(f"+{len(r['others'])} other launch{'es' if len(r['others']) > 1 else ''} in the manifest")
            sub_lines = wrap_text(' \u00b7 '.join(sub), 9.5, cols['c1'] - 96 - cols['chip']) if sub else []
            return dict(chip_lines=chip_lines, sub_lines=sub_lines, h=max(74, 20 + 20 * len(chip_lines) + 17 * len(sub_lines) + 18))
        msg = messages.get(r['status'], r['status'])
        if r['status'] == 'smoke-only':
            msg = f"smoke cells only so far: {r['smoke']} \u2014 no {HEADLINE_WINDOW_S} s decode matrix yet"
        if r.get('validation'):
            msg = f"{msg} \u2014 {r['validation']}"
        msg_lines = wrap_text(msg, 12, W - 96 - cols['chip'])
        return dict(msg_lines=msg_lines, h=max(74, 26 + 21 * len(msg_lines) + 14))

    lane_rows = [(lane_id, lane_title, [(r, layout(r)) for r in ov if r['lineage'] == lane_id]) for lane_id, lane_title in lanes]
    legend_rows, lx = 1, 70
    for key, label, color in FAMILY_STYLES:
        width = 24 + tw(label, 11) + 30
        if lx + width > W - 60:
            lx, legend_rows = 70, legend_rows + 1
        lx += width
    notes = ('KV pool: "server" = the GPU KV cache size the server logged; "\u2248 bench \u00d7 CP" = benchmark metrics \u00d7 DCP degree, which overstates DCP4 pools. '
             '* = cell ran under-filled (fewer streams than requested). err = the cell errored. \u2020 = non-standard KV page geometry or budget override. '
             'Speculative modes change acceptance, so cross-spec numbers are workload-dependent. Grey bar = no comparable decode matrix. '
             'R26 cells additionally require a GPU-isolation-clean window; contaminated windows and superseded primaries stay in the manifest, never on this chart. '
             'R27 cells additionally require the producer gate (steady counters + exclusive window), a fully covered clean window and the pinned source-snapshot '
             'image; the patched-auto row is a policy control, and absent R27 data stays blank \u2014 never estimated.')
    note_lines = wrap_text(notes, 10.5, W - 140)
    hdr_lines = wrap_text('One row per image, with the launch it was measured under. Numbers are the headline receipt only. '
                          'A colour tag is a launch family (same geometry and cell size), not a controlled comparison; the matched chart '
                          'is the comparison. Grey = a different engine, checkpoint or launch.', 13.5, W - 132)
    legend_y = 142 + 26 * len(hdr_lines) + 20
    head_h = legend_y + 24 * (legend_rows - 1) + 42 + 48
    body_h = sum(LANE_H + 12 + sum(lay['h'] for _, lay in items) for _, _, items in lane_rows)
    fig_h = head_h + body_h + 12 + 21 * len(note_lines) + 100

    fig, ax = _fig(W, fig_h)
    _text(ax, 70, 52, ' '.join('GLM-5.3 FLASH / FIELD NOTES'), 12.5, BLUE, 'bold')
    _text(ax, 66, 84, 'Appendix: every build we ran.', 30, TEXT, 'bold')
    for j, line in enumerate(hdr_lines):
        _text(ax, 66, 142 + 26 * j, line, 13.5, MUTED)
    lx, ly = 70, legend_y
    for key, label, color in FAMILY_STYLES:
        width = 24 + tw(label, 11) + 30
        if lx + width > W - 60:
            lx, ly = 70, ly + 24
        _rect(ax, lx, ly - 7, 14, 14, color)
        _text(ax, lx + 22, ly, label, 11, MUTED, va='center')
        lx += width
    y = ly + 42
    _text(ax, cols['name'], y, 'build', 11.5, MUTED, 'bold')
    _text(ax, cols['chip'], y, 'launch measured under', 11.5, MUTED, 'bold')
    for k, lab in (('c1', 'C1'), ('c8', 'C8'), ('c16', 'C16')):
        _text(ax, cols[k], y, lab, 11.5, MUTED, 'bold', ha='right')
    _text(ax, cols['kv'], y, 'KV pool', 11.5, MUTED, 'bold', ha='right')
    _text(ax, cols['pf'], y, '32k prefill', 11.5, MUTED, 'bold', ha='right')
    _text(ax, cols['pf'], y + 21, f'decode = aggregate output tok/s at ctx0 \u00b7 {HEADLINE_WINDOW_S} s cells unless noted', 10, MUTED, ha='right')
    y += 48
    for lane_id, lane_title, items in lane_rows:
        _text(ax, 70, y + 10, lane_title, 13, TEXT, 'bold')
        y += LANE_H
        for r, lay in items:
            h = lay['h']
            _card(ax, 56, y, W - 112, h - 8)
            measured = r['status'] == 'measured'
            _rect(ax, 56, y + 8, 5, h - 24, FAMILY_COLOR.get(r.get('family', 'other'), MUTED) if measured else GRID, zorder=3)
            _text(ax, cols['name'], y + 13, SHORT.get(r['release'], r['name'])[:17], 12, TEXT, 'bold')
            _text(ax, cols['name'], y + 42, r['date'], 10, MUTED)
            if not measured:
                for i, line in enumerate(lay['msg_lines']):
                    _text(ax, cols['chip'], y + 22 + 21 * i, line, 12, MUTED, style='italic')
                y += h
                continue
            for i, line in enumerate(lay['chip_lines']):
                _text(ax, cols['chip'], y + 11 + i * 20, line, 11, TEXT)
            for i, line in enumerate(lay['sub_lines']):
                _text(ax, cols['chip'], y + 13 + len(lay['chip_lines']) * 20 + 17 * i, line, 9.5, MUTED)
            for k in ('c1', 'c8', 'c16'):
                s, col = fmt_tps(r.get(k))
                _text(ax, cols[k], y + 16, s, 16, col, 'bold', ha='right')
            if r.get('kv_server'):
                _text(ax, cols['kv'], y + 16, fmt_k(r['kv_server']), 14, AMBER, 'bold', ha='right')
                _text(ax, cols['kv'], y + 44, 'server', 9, MUTED, ha='right')
            elif r.get('kv_bench'):
                _text(ax, cols['kv'], y + 16, '\u2248' + fmt_k(r['kv_bench']), 13, MUTED, 'bold', ha='right')
                _text(ax, cols['kv'], y + 44, 'bench \u00d7 CP', 9, MUTED, ha='right')
            pf = r.get('prefill32k')
            _text(ax, cols['pf'], y + 16, f'{pf:,.0f}' if pf else '\u2014', 14, TEXT if pf else MUTED, 'bold', ha='right')
            y += h
        y += 12
    y += 4
    for line in note_lines:
        _text(ax, 70, y, line, 10.5, MUTED)
        y += 21
    footer(ax, fig_h, stamp)
    fig.savefig(out, dpi=DPI, facecolor=BG)
def wrap_text(s: str, size: float, width: float) -> list[str]:
    """Greedy word wrap. Never truncates a note: callers size their layout to the returned lines."""
    words, lines, cur = s.split(' '), [], ''
    for w in words:
        cand = w if not cur else f'{cur} {w}'
        if cur and tw(cand, size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines

def slope_panel(ax, x0, y0, w, h, title, releases, series, note_lines=(), pending=()):
    """series: list of (label, color, values, statuses); values/statuses aligned with releases.
    The title sits top-left inside the card; the plot area starts below it. Series labels sit left
    of the first point; value labels ride above each marker and alternate below it when the points
    are too close for the label width. note_lines are pre-wrapped by the caller, who grows h so
    every line fits; notes are never cut. ``pending`` names the release labels whose empty slot
    means 'pending' rather than 'n/a'."""
    _card(ax, x0, y0, w, h, fc=BG)
    _text(ax, x0 + 16, y0 + 14, title, 10.5, TEXT)
    pl, pr, pt = x0 + 70, x0 + w - 16, y0 + 62
    pb = y0 + h - 74 - 14 * max(0, len(note_lines) - 2)
    n = max(len(releases), 1)
    xs = [pl + (pr - pl) * (i + 0.5) / n for i in range(n)]
    spacing = (pr - pl) / n
    vals = [v for _, _, values, _ in series for v in values if v is not None]
    vmax = max(vals) if vals else 1.0
    scale = (pb - pt) / (vmax * 1.12)
    ax.plot([pl - 36, pr], [pb, pb], color=MUTED, lw=1.2)
    for i, rel in enumerate(releases):
        _text(ax, xs[i], pb + 8, rel, 12, TEXT, 'bold', ha='center')
    for label, color, values, statuses in series:
        pts = [(xs[i], pb - v * scale) for i, v in enumerate(values) if v is not None]
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.2, alpha=0.9, zorder=3)
        first = next((i for i, v in enumerate(values) if v is not None), None)
        if first is not None:
            _text(ax, pl - 40, pb - values[first] * scale, label, 10.5, color, 'bold', ha='left', va='center')
        labels = [(fmt_k(v) if v >= 1e6 else f'{v:,.0f}' + ('*' if statuses[i] == 'underfilled' else '')) if v is not None else '' for i, v in enumerate(values)]
        size = 12.5 if max((tw(t, 12.5) for t in labels), default=0) <= spacing * 0.9 else 11
        stagger = max((tw(t, size) for t in labels), default=0) > spacing * 0.9
        for i, v in enumerate(values):
            if v is None:
                continue
            yv = pb - v * scale
            _rect(ax, xs[i] - 5, yv - 5, 10, 10, BG if statuses[i] == 'underfilled' else color, ec=color, lw=2, zorder=4)
            if stagger and i % 2:
                _text(ax, xs[i], yv + 9, labels[i], size, color, 'bold', ha='center', va='top', zorder=5)
            else:
                _text(ax, xs[i], yv - 9, labels[i], size, color, 'bold', ha='center', va='bottom', zorder=5)
    for i, rel in enumerate(releases):
        if all(values[i] is None for _, _, values, _ in series):
            _text(ax, xs[i], pb - 16, 'pending' if rel in pending else 'n/a', 10.5, MUTED, ha='center', va='bottom', style='italic')
    for j, line in enumerate(note_lines):
        _text(ax, x0 + 16, y0 + h - 12 - 14 * (len(note_lines) - 1 - j), line, 9, MUTED, va='bottom')


REASONS = {'bench': 'different cells', 'dcp': 'DCP differs', 'cache': 'cache mode differs', 'gmu': 'GMU differs',
           'seqs': 'slot count differs', 'maxlen': 'context limit differs', 'kv': 'KV format differs', 'fairness': 'fairness differs',
           'gather': 'CKV gather differs', 'kda': 'KDA backend differs', 'batched': 'scheduler budget differs', 'draft': 'draft differs'}


def lineage_panels(L: dict) -> tuple[list[str], list[tuple], set]:
    """Panels for one resolved lineage: one per speculator and client context, plus prefill and KV
    when the lineage carries them. A panel only ever holds the concurrency levels of one context,
    so a release's near-identical ctx0 and 32k values never overdraw each other and the palette
    restarts per panel."""
    releases = []
    for m in L['members']:
        if m['release'] not in releases:
            releases.append(m['release'])
    xlabels = [short_name(r) for r in releases]
    pending_x = {short_name(r) for r in releases if r in ('r26', 'r26-overlay') or r in R27_RELEASES}

    def member(rel, spec_label):
        m = next((m for m in L['members'] if m['release'] == rel and m['spec_label'] == spec_label), None)
        return m if m and m.get('status', '').startswith('matched') else None

    specs = []
    for m in L['members']:
        if m['spec_label'] not in specs:
            specs.append(m['spec_label'])
    palette = [BLUE, AMBER, GREEN]
    contexts = []
    for mt in L['metrics']:
        if mt['ctx'] not in contexts:
            contexts.append(mt['ctx'])
    panels = []
    for spec in specs:
        notes = []
        for m in L['members']:
            if m['spec_label'] != spec:
                continue
            if m.get('status') == 'mismatch':
                notes.append(f"{short_name(m['release'])}: {REASONS.get(m['hard_deviations'][0].split(':')[0].split(' ')[0], 'launch differs')}")
            elif m.get('status') == 'absent':
                notes.append(f"{short_name(m['release'])}: not run")
            elif m.get('status') == 'pending':
                notes.append(f"{short_name(m['release'])}: {m.get('note') or 'not run yet'}")
            elif m.get('status') == 'matched-inferred':
                notes.append(f"{short_name(m['release'])}: launch inferred")
            if m.get('status', '').startswith('matched') and m.get('soft_deviations'):
                notes.append(f"{short_name(m['release'])}: soft diff {', '.join(m['soft_deviations'])}")
        note = ' \u00b7 '.join(notes) if notes else None
        for ctx in contexts:
            series = []
            for mt in L['metrics']:
                if mt['ctx'] != ctx:
                    continue
                ml = mt['label']
                values = [member(r, spec)['values'].get(ml) if member(r, spec) else None for r in releases]
                statuses = [member(r, spec)['statuses'].get(ml, 'ok') if member(r, spec) else 'absent' for r in releases]
                series.append((ml, palette[len(series) % len(palette)], values, statuses))
            panels.append((f'{spec} \u00b7 decode tok/s \u00b7 ctx{fmt_ctx(ctx)}', series, note))
    if L.get('has_prefill'):
        spec = specs[0]
        values = [member(r, spec)['values'].get('prefill32k') if member(r, spec) else None for r in releases]
        panels.append(('32k client prefill \u00b7 tok/s', [('32k', GREEN, values, ['ok'] * len(releases))], None))
    if L.get('has_kv'):
        spec = specs[0]
        values = [member(r, spec)['values'].get('kv_server') if member(r, spec) else None for r in releases]
        panels.append(('server-reported KV pool \u00b7 tokens', [('pool', AMBER, values, ['ok'] * len(releases))], None))
    return xlabels, panels, pending_x


def render_matched(lins: list[dict], collisions: list[dict], out: Path, stamp: str) -> None:
    PANEL_H, PER_ROW = 262, 3
    text_w = W - 112 - 100
    pw = (W - 112 - 40 - (PER_ROW - 1) * 16) / PER_ROW
    blocks = []
    for L in lins:
        xlabels, panels, pending_x = lineage_panels(L)
        wrapped = [(title, series, wrap_text(note, 9, pw - 32) if note else []) for title, series, note in panels]
        row_hs = []
        for i in range(0, len(wrapped), PER_ROW):
            extra = max((len(lines) for _, _, lines in wrapped[i:i + PER_ROW]), default=0)
            row_hs.append(PANEL_H + 14 * max(0, extra - 2))
        sub = wrap_text(L['subtitle'], 12, text_w)
        cav = wrap_text(L['caveat'], 10.5, text_w)
        blocks.append((L, xlabels, wrapped, row_hs, sub, cav, pending_x, 80 + 20 * len(sub) + sum(row_hs) + 20 + 19 * len(cav)))
    caption = ('R15/R20/R22 ran 16 slots and 262k context, R24/R25 32 slots and 1M; the probe is one decode + one prefill, so slots should not move it. '
               'Grey = FAIRNESS_ENGINE was not passed to the launcher: R20 behaves like "off", R22 like compute-share 0.4 (image default, unverified).')
    cap_lines = wrap_text(caption, 10.5, W - 200)
    COLL_H = 360 + 19 * max(0, len(cap_lines) - 3)
    hdr_lines = wrap_text('Within each block the launch family and benchmark parameters are the same. Known soft differences (capture size, schedule '
                          'interval, shm) and inferred launches are printed under each panel, not hidden. No line crosses a change of DCP, slots, KV '
                          'format, cache mode or fairness. Hollow marker = under-filled cell.', 13.5, W - 132)
    blocks_y = 142 + 26 * len(hdr_lines) + 24
    H = blocks_y + sum(b[7] + 14 for b in blocks) + COLL_H + 90
    fig, ax = _fig(W, H)
    _text(ax, 70, 52, ' '.join('GLM-5.3 FLASH / FIELD NOTES'), 12.5, BLUE, 'bold')
    _text(ax, 66, 84, 'Matched runs, newer image.', 30, TEXT, 'bold')
    for j, line in enumerate(hdr_lines):
        _text(ax, 66, 142 + 26 * j, line, 13.5, MUTED)
    y = blocks_y
    idx = 1
    for L, xlabels, panels, row_hs, sub, cav, pending_x, bh in blocks:
        _card(ax, 56, y, W - 112, bh)
        _text(ax, 88, y + 22, f'{idx:02d}', 20, MUTED, 'bold')
        _text(ax, 148, y + 22, L['title'], 18, TEXT, 'bold')
        for j, line in enumerate(sub):
            _text(ax, 148, y + 58 + j * 20, line, 12, MUTED)
        top = y + 74 + 20 * len(sub)
        for i, (title, series, lines) in enumerate(panels):
            px = 76 + (i % PER_ROW) * (pw + 16)
            py = top + sum(row_hs[:i // PER_ROW])
            slope_panel(ax, px, py, pw, row_hs[i // PER_ROW] - 12, title, xlabels, series, lines, pending_x)
        for j, line in enumerate(cav):
            _text(ax, 88, y + bh - 14 - 19 * (len(cav) - 1 - j), line, 10.5, MUTED, va='bottom')
        y += bh + 14
        idx += 1
    _card(ax, 56, y, W - 112, COLL_H)
    _text(ax, 88, y + 22, f'{idx:02d}', 20, MUTED, 'bold')
    _text(ax, 148, y + 22, 'Decode kept alive during one cold 65k prefill', 18, TEXT, 'bold')
    _text(ax, 148, y + 58, 'D-Rock\u2019s decode_prefill_collision.py, unchanged \u00b7 one 4k decode stream + one 65,535-token cold prefill', 12, MUTED)
    rows = [c for c in collisions if c.get('retained_pct') is not None]
    bx, bw, base = 110, W - 112 - 140, y + 226
    slot = bw / max(len(rows), 1)
    for i, c in enumerate(rows):
        x = bx + i * slot
        policy = c['policy']
        color = AMBER if policy.startswith('compute_share') else (BLUE if policy == 'off' else MUTED)
        h = c['retained_pct'] / 100 * 120
        _rect(ax, x + slot * 0.22, base - h, slot * 0.56, h, color)
        _text(ax, x + slot / 2, base - h - 6, f"{c['retained_pct']:.0f}%", 15, color, 'bold', ha='center', va='bottom')
        _text(ax, x + slot / 2, base + 8, short_name(c['release']), 12, TEXT, 'bold', ha='center')
        _text(ax, x + slot / 2, base + 27, {'off': 'fairness off', 'launcher default': 'default'}.get(policy, 'share 0.4'), 10.5, MUTED, ha='center')
        _text(ax, x + slot / 2, base + 44, f"{c['collision_tps']:.0f} / {c['baseline_tps']:.0f} tok/s", 9.5, MUTED, ha='center')
    ax.plot([bx, bx + bw], [base, base], color=MUTED, lw=1.2)
    for j, line in enumerate(cap_lines):
        _text(ax, 88, y + COLL_H - 14 - 19 * (len(cap_lines) - 1 - j), line, 10.5, MUTED, va='bottom')
    footer(ax, H, stamp)
    fig.savefig(out, dpi=DPI, facecolor=BG)

# --------------------------------------------------------------------------- main chart
# Segments of the compact chart. Every release inside a segment was launched with the
# same geometry (selector on the normalised config) and measured with the same cell
# parameters (max_tokens); nothing is drawn across a segment break.
MAIN_SEGMENTS = [
    dict(title='DCP1 \u00b7 16 slots', sub='1024-token ctx0 chart cells \u00b7 GMU 0.90',
         releases=['r7', 'r12', 'r17'], spec='none', max_tokens=1024,
         selector=dict(dcp=1, seqs=16, gmu='0.90', maxlen=262144, cache='vram', kv='fp8', fairness='off'),
         note='r7 = the 488ddf75 digest the 2 Sep chart called "r8"; r15 has no 1024-token no-spec cell'),
    dict(title='DCP1 \u00b7 16 slots', sub='8192-token cells \u00b7 GMU 0.90',
         releases=['r15', 'r18', 'r20', 'r22'], spec='none', max_tokens=8192,
         selector=dict(dcp=1, seqs=16, gmu='0.90', maxlen=262144, cache='vram', kv='fp8', fairness='off'),
         note='r15 and r18 launch lines were not retained (geometry-inferred); r22 added capture 256 + schedule interval 1'),
    dict(title='DCP1 \u00b7 32 slots', sub='1M context \u00b7 compute-share 0.4 \u00b7 GMU 0.93 \u00b7 inspect-verified',
         releases=['r24', 'r25', 'r26'], spec='none', max_tokens=8192,
         selector=dict(dcp=1, seqs=32, gmu='0.93', maxlen=1048576, cache='vram', kv='fp8', fairness='compute_share 0.4'),
         note='different slot count, context limit, GMU and fairness from the segments on the left'),
]
MAIN_DCP4 = dict(title='DCP4 CKV-gather \u00b7 DFlash K7 (MXFP8 draft) \u00b7 16 slots \u00b7 GMU 0.90',
                 releases=['r7', 'r8', 'r10'], spec='dflash2-k7', max_tokens=8192,
                 selector=dict(dcp=4, gather='1', seqs=16, gmu='0.90', maxlen=262144, cache='vram', kv='fp8'),
                 note='speculative decode: acceptance moves these numbers; r8 and r10 launch lines were not retained')


def segment_points(seg: dict, rows: list[dict]) -> list[dict]:
    """One point per release: earliest receipt matching the segment selector, spec and cell size,
    with the segment's metric cells (default ctx0 C16 + C1; the R27 segment carries C8 + C1).
    ``rows`` is the plot-filtered set, so a superseded, contaminated or gate-failed cell can
    never become a point."""
    keys = seg.get('metrics', ['c16', 'c1'])
    concs = tuple(int(k[1:]) for k in keys)
    pts = []
    for rel in seg['releases']:
        cands = sorted((r for r in rows if r.get('release') == rel and 'cells' in r and r['config']['spec'] == seg['spec']
                        and r['bench'].get('max_tokens') == seg['max_tokens'] and (r['bench'].get('duration') or 0) >= 30
                        and all(str(r['config'].get(k)) == str(v) for k, v in seg['selector'].items())),
                       key=lambda r: r.get('timestamp') or '')
        pick = next((r for r in cands if all(cell(r, c, 0) for c in concs)), None)
        if pick is None:
            pts.append(dict(release=rel, status='pending' if rel in ('r26', 'r26-overlay') or rel in R27_RELEASES else 'absent'))
            continue
        pt = dict(release=rel, status='ok', receipt=pick['id'], inferred=pick['grade'] not in ('script', 'inspect', 'launch'),
                  provenance='clean-rerun' if pick['id'].startswith('clean-reruns/')
                  else ('isolation-validated primary' if (pick.get('isolation') or {}).get('verdict') == 'clean' else 'historical'))
        for k, c in zip(keys, concs):
            cl = cell(pick, c, 0)
            pt[k] = cl['tps']
            pt[f'{k}_status'] = cl['status']
        pts.append(pt)
    return pts


def draw_strip(ax, x0, y0, w, h, label, segs_pts, key, color, vmax):
    """One metric strip across all segments: separate polylines per segment, dashed breaks between them."""
    _text(ax, x0, y0 + 2, label, 12, TEXT, 'bold')
    pl, pr, pt, pb = x0 + 150, x0 + w, y0 + 30, y0 + h - 34
    total = sum(len(p) for p in segs_pts)
    gap = 1.0
    unit = (pr - pl) / (total + gap * (len(segs_pts) - 1))
    scale = (pb - pt) / (vmax * 1.12)
    ax.plot([pl, pr], [pb, pb], color=MUTED, lw=1.2)
    x = pl
    for si, pts in enumerate(segs_pts):
        xs = [x + unit * (i + 0.5) for i in range(len(pts))]
        line = [(xs[i], pb - p[key] * scale) for i, p in enumerate(pts) if p['status'] == 'ok' and key in p]
        if len(line) >= 2:
            ax.plot([q[0] for q in line], [q[1] for q in line], color=color, lw=2.4, alpha=0.9, zorder=3)
        for i, p in enumerate(pts):
            _text(ax, xs[i], pb + 8, short_name(p['release']), 12, TEXT, 'bold', ha='center')
            if p['status'] != 'ok':
                _text(ax, xs[i], pb - 14, 'pending' if p['status'] == 'pending' else 'n/a', 10.5, MUTED, ha='center', va='bottom', style='italic')
                continue
            if key not in p:
                _text(ax, xs[i], pb - 14, f'no {key.upper()} in plan', 10, MUTED, ha='center', va='bottom', style='italic')
                continue
            yv = pb - p[key] * scale
            hollow = p[f'{key}_status'] == 'underfilled' or p['inferred']
            if p.get('provenance') == 'clean-rerun':
                ax.plot([xs[i]], [yv], marker='o', ms=8, mfc=BG if hollow else color, mec=color, mew=2, zorder=4)
            else:
                _rect(ax, xs[i] - 5, yv - 5, 10, 10, BG if hollow else color, ec=color, lw=2, zorder=4)
            _text(ax, xs[i], yv - 9, f"{p[key]:,.0f}" + ('*' if p[f'{key}_status'] == 'underfilled' else ''), 13, color, 'bold', ha='center', va='bottom', zorder=5)
        x += unit * len(pts)
        if si < len(segs_pts) - 1:
            bx = x + unit * gap / 2
            ax.plot([bx, bx], [pt - 6, pb + 40], color=GRID, lw=1.5, ls=(0, (4, 4)))
            x += unit * gap
    return pl, unit, gap


def render_main(rows: list[dict], clean: dict, r27: dict, out: Path, stamp: str) -> None:
    segs = [(seg, segment_points(seg, rows)) for seg in MAIN_SEGMENTS]
    dcp4 = segment_points(MAIN_DCP4, rows)
    intro_lines = wrap_text('No speculative decode \u00b7 context 0 \u00b7 aggregate output tok/s \u00b7 30-second cells. '
                            'Dashed dividers mark different settings; never read across a divider as a release gain.',
                            13.5, W - 140)
    notes = ('Hollow marker = launch line not retained (config inferred from geometry). R26 square = primary cell the GPU isolation log proves clean; '
             'round = verified-quiet rerun; contaminated windows never plot. ' + ' \u00b7 '.join(seg['note'] for seg, _ in segs))
    note_lines = wrap_text(notes, 9.5, W - 200)
    # R26 clean-rerun status: landed cells with real values, plus the honest pending list.
    status = []
    plan = clean['plan']
    n_clean = sum(1 for c in plan['cells'].values() if c['clean'])
    if clean['present'] and plan['planned']:
        status.append((f"Verified-quiet rerun cells clean so far: {n_clean} of {plan['planned']} planned. All speed cells below ran at 32 slots, 1M context, "
                       'compute-share 0.4. Contaminated primary windows are excluded and rerun; whatever is still pending stays empty above \u2014 never estimated.', MUTED))
        for r in clean['speed']:
            c16, c1 = cell(r, 16, 0), cell(r, 1, 0)
            if not (c16 and c1):
                continue
            sup = ' \u2014 supersedes the contaminated primary' if r.get('supersedes') else ''
            status.append((f"{r['id'].split('/')[-1][:-5]} \u2014 C16 {c16['tps']:,.0f} \u00b7 C1 {c1['tps']:,.0f} tok/s{sup}", TEXT))
        if clean['pending_labels']:
            status.append(('Still pending: ' + ', '.join(clean['pending_labels']), MUTED))
        status.append(('These exact R26 settings have no r24/r25 twin on this box (their LMCache arms ran GMU 0.95, their DCP2 arms FP8 KV), so they are '
                       'single-release rows, not a comparison. Acceptance reruns are in the manifest, not on this speed chart.', MUTED))
    else:
        status.append(('No clean reruns have landed yet; any R26 point above comes from a primary window the GPU isolation log proves clean.', MUTED))
    # R27 status: only producer-gated, clean-window, pinned-image cells may plot; absent root stays empty.
    if not r27['present']:
        status.append(('R27 stock / patched / auto have not been GPU-tested on this host yet \u2014 the battery root has no producer receipts, so no R27 '
                       'datapoint appears anywhere. Nothing is estimated.', MUTED))
    else:
        attempted = len(r27['speed']['cells'])
        eligible = len(r27['speed']['eligible'])
        if eligible:
            status.append((f"R27 speed cells eligible so far: {eligible} of {attempted} attempted (producer gate + clean window + pinned image); stock-vs-"
                           'patched values plot on the matched chart \u2014 the R27 plan has no C16.', TEXT))
        else:
            status.append((f"R27 speed cells attempted: {attempted}; none are eligible yet (producer gate + fully covered clean window + pinned image). "
                           'No R27 point appears until one is.', MUTED))
        status.append(('The patched-auto R27 image ships the auto prefill-policy defaults; fixed-vs-auto policy differences are scheduler attribution '
                       'controls, never release performance gains.', MUTED))
    status_lines = [(line, color) for text, color in status for line in wrap_text(text, 10.5, W - 200)]
    # layout: segment card grows with the notes, then the DCP4 strip, then the status card
    seg_y = max(206, 140 + 22 * len(intro_lines) + 18)
    seg_card_h = 534 + 16 * len(note_lines) + 22
    dcp4_y = seg_y + seg_card_h + 14
    dcp4_h = 176
    stat_y = dcp4_y + dcp4_h + 14
    stat_h = 52 + 21 * len(status_lines) + 18
    H = stat_y + stat_h + 76
    fig, ax = _fig(W, H)
    _text(ax, 70, 52, ' '.join('GLM-5.3 FLASH / FIELD NOTES'), 12.5, BLUE, 'bold')
    _text(ax, 66, 84, 'JJ releases: matched comparisons.', 30, TEXT, 'bold')
    for j, line in enumerate(intro_lines):
        _text(ax, 66, 140 + 22 * j, line, 13.5, MUTED)
    # segment headers
    _card(ax, 56, seg_y, W - 112, seg_card_h)
    pl, unit, gap = 56 + 24 + 150, 0, 1.0
    total = sum(len(p) for _, p in segs)
    unit = (W - 112 - 48 - 150) / (total + gap * (len(segs) - 1))
    x = pl
    for seg, pts in segs:
        cx = x + unit * len(pts) / 2
        _text(ax, cx, seg_y + 16, seg['title'], 12, TEXT, 'bold', ha='center')
        for j, line in enumerate(wrap_text(seg['sub'], 9.5, unit * len(pts) + unit * 0.7)):
            _text(ax, cx, seg_y + 38 + 14 * j, line, 9.5, MUTED, ha='center')
        x += unit * (len(pts) + gap)
    draw_strip(ax, 80, seg_y + 86, W - 160, 220, '16 callers\nC16 \u00b7 tok/s', [p for _, p in segs], 'c16', BLUE,
               max([p['c16'] for _, pts in segs for p in pts if p['status'] == 'ok' and 'c16' in p] or [1]))
    draw_strip(ax, 80, seg_y + 320, W - 160, 196, '1 caller\nC1 \u00b7 tok/s', [p for _, p in segs], 'c1', AMBER,
               max([p['c1'] for _, pts in segs for p in pts if p['status'] == 'ok' and 'c1' in p] or [1]))
    for j, line in enumerate(note_lines):
        _text(ax, 80, seg_y + 534 + 16 * j, line, 9.5, MUTED)
    # DCP4 CKV strip
    _card(ax, 56, dcp4_y, W - 112, dcp4_h)
    _text(ax, 80, dcp4_y + 14, MAIN_DCP4['title'], 12.5, TEXT, 'bold')
    _text(ax, 80, dcp4_y + 36, 'serve-r7.sh control launcher \u00b7 ' + MAIN_DCP4['note'], 10, MUTED)
    px = 80 + 130
    unit4 = 150
    pb = dcp4_y + 138
    vmax = max([p['c16'] for p in dcp4 if p['status'] == 'ok'] or [1])
    ax.plot([px, px + unit4 * len(dcp4)], [pb, pb], color=MUTED, lw=1.2)
    for i, p in enumerate(dcp4):
        cx = px + unit4 * (i + 0.5)
        _text(ax, cx, pb + 8, short_name(p['release']), 12, TEXT, 'bold', ha='center')
        if p['status'] != 'ok':
            _text(ax, cx, pb - 14, 'n/a', 10.5, MUTED, ha='center', va='bottom', style='italic')
            continue
        for key, color in (('c16', BLUE), ('c1', AMBER)):
            yv = pb - (p[key] / vmax) * 62
            hollow = p[f'{key}_status'] == 'underfilled' or p['inferred']
            _rect(ax, cx - 4, yv - 4, 8, 8, BG if hollow else color, ec=color, lw=2, zorder=4)
            _text(ax, cx + 12, yv, f"{p[key]:,.0f}", 12, color, 'bold', ha='left', va='center')
    for key, color, ly in (('c16', BLUE, pb - 62), ('c1', AMBER, pb - 10)):
        _text(ax, px - 8, ly, {'c16': 'C16', 'c1': 'C1'}[key], 11, color, 'bold', ha='right', va='center')
    side = ('Separate speculative lineage: DFlash K7 output depends on draft acceptance, so it is never compared with the no-spec strips above. '
            'D-Rock pre-bakes, local patches, SGLang, EXL3 and QAD runs live in the inventory appendix and the manifest.')
    sx = px + unit4 * len(dcp4) + 40
    for j, line in enumerate(wrap_text(side, 10, W - 80 - sx)):
        _text(ax, sx, pb - 70 + 16 * j, line, 10, MUTED)
    # R26 clean-rerun + R27 producer status card
    _card(ax, 56, stat_y, W - 112, stat_h)
    _text(ax, 80, stat_y + 16, 'R26 verified-quiet reruns \u00b7 R27 producer-gated cells \u2014 landed so far', 12.5, TEXT, 'bold')
    for j, (line, color) in enumerate(status_lines):
        _text(ax, 80, stat_y + 48 + 21 * j, line, 10.5, color)
    footer(ax, H, stamp)
    fig.savefig(out, dpi=DPI, facecolor=BG)


# --------------------------------------------------------------------------- kv evidence
def write_kv_evidence(rows: list[dict], out: Path) -> None:
    """Server-logged KV pool vs bench-derived figure for every battery receipt with a docker log,
    plus the prior summary.json claims so old bench-estimate numbers can be corrected."""
    claims = {}
    for rel in ('r24', 'r25'):
        p = RESULTS / rel / 'summary.json'
        if p.exists():
            try:
                for row in json.load(open(p)):
                    claims[(rel, row.get('name'))] = dict(path=str(p), kv_tokens=row.get('kv_tokens', row.get('kv')))
            except (OSError, ValueError):
                pass
    records = []
    for r in rows:
        ev = r.get('kv', {}).get('server_evidence')
        if not ev:
            continue
        stem = r['id'].split('/')[-1][:-5]
        rec = dict(release=r['release'], receipt=r['path'], receipt_sha256=r.get('sha256'), inspect=None,
                   server_log=ev['log'], server_log_line=ev['line'], server_log_text=ev['text'],
                   server_kv_tokens=ev['value'], bench_kv_tokens=r['kv'].get('bench_total'), bench_formula=r['kv'].get('formula'),
                   bench_over_server=round(r['kv']['bench_total'] / ev['value'], 4) if r['kv'].get('bench_total') and ev['value'] else None,
                   prior_summary_claim=claims.get((r['release'], stem)))
        img = r.get('image') or {}
        if img:
            rec['inspect'] = dict(path=str(Path(r['path']).with_name(f'{stem}.inspect.json')), image_id=img.get('id'), digest=img.get('digest'),
                                  release_label=img.get('release_label'))
        records.append(rec)
    corrections = [dict(release=x['release'], receipt=Path(x['receipt']).name, claimed=x['prior_summary_claim']['kv_tokens'], server=x['server_kv_tokens'], claim_path=x['prior_summary_claim']['path'])
                   for x in records if x['prior_summary_claim'] and x['prior_summary_claim']['kv_tokens'] is not None and x['prior_summary_claim']['kv_tokens'] != x['server_kv_tokens']]
    out.write_text(json.dumps(dict(
        generated_by='scripts/r26/release_history.py',
        method='server_kv_tokens = first "GPU KV cache size: N tokens" line in the captured docker log; bench_kv_tokens = llm_decode_bench '
               'metadata.max_total_tokens (num_blocks x block_size x DCP as the bench multiplies), which overstates DCP4 pools',
        records=records, summary_claims_to_correct=corrections), indent=2) + '\n')
# --------------------------------------------------------------------------- csv
def write_cells_csv(rows: list[dict], out: Path) -> int:
    n = 0
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['release', 'receipt', 'timestamp', 'grade', 'family', 'bench_key', 'spec', 'kind', 'conc', 'ctx', 'value', 'status', 'spec_accept',
                    'kv_bench', 'kv_server', 'isolation', 'speed_usable', 'superseded_by', 'supersedes', 'policy'])
        for r in rows:
            if 'cells' not in r:
                continue
            iso = (r.get('isolation') or {}).get('verdict', '')
            policy = (r.get('scheduler') or {}).get('requested_policy') or ''
            policy = policy.get('key') if isinstance(policy, dict) else policy
            for c in r['cells']:
                w.writerow([r['release'], r['id'], r['timestamp'], r['grade'], r['family'], r['bench_key'], r['config']['spec'], r['kind'],
                            c['conc'], c['ctx'], c['tps'], c['status'], c['spec_accept'], r['kv'].get('bench_total'), r['kv'].get('server_total'),
                            iso, r.get('speed_usable', True), r.get('superseded_by', ''), r.get('supersedes', ''), policy])
                n += 1
            for ctx, v in sorted(r['prefill'].items()):
                w.writerow([r['release'], r['id'], r['timestamp'], r['grade'], r['family'], r['bench_key'], r['config']['spec'], 'prefill', 1, ctx, v, 'ok',
                            '', '', '', iso, r.get('speed_usable', True), r.get('superseded_by', ''), r.get('supersedes', ''), policy])
                n += 1
    return n


# --------------------------------------------------------------------------- main
def build(r26_root: Path, out_dir: Path, render: bool = True, r27_root: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
    curated = collect_curated()
    battery_rows, battery_boots = [], []
    for root, rel, repo_dir in BATTERIES:
        rows, boots = collect_battery(root, rel, repo_dir, f'run_{rel}_battery.py')
        battery_rows += rows
        battery_boots += [dict(b, battery=root.name) for b in boots]
    r26 = collect_r26(r26_root)
    clean = collect_clean_reruns(r26_root / 'clean-reruns', r26_root)
    r27 = collect_r27(r27_root or DEFAULT_R27_ROOT)
    sup_speed, conflicts = apply_supersessions(r26['receipts'], clean['speed'])
    sup_acc, acc_conflicts = apply_supersessions(r26['acceptance'], clean['acceptance'])
    all_rows = curated + battery_rows + r26['receipts'] + clean['speed'] + r27['receipts']
    for r in all_rows:
        r.setdefault('kv', {})
    # Charts and lineages see only plot-eligible rows: decode kind, never superseded, and (for the
    # R26/R27 phases) a GPU-isolation-clean window plus the phase gate. Historical rows predate the
    # sampler and stay as graded. R27 scheduler rows are kind='scheduler': manifest/CSV only.
    plot_rows = [r for r in all_rows if r.get('kind') == 'decode' and not r.get('superseded_by') and r.get('speed_usable', True)]
    excluded_r26 = [r for r in r26['receipts'] if not r.get('speed_usable', True)]

    def pending_note(release: str, spec: str, selector: dict) -> str | None:
        for r in excluded_r26:
            if r.get('superseded_by') or r['release'] != release or r['config'].get('spec') != spec:
                continue
            if not all(str(r['config'].get(k)) == str(v) for k, v in selector.items()):
                continue
            verdict = (r.get('isolation') or {}).get('verdict')
            if verdict == 'contaminated':
                return 'primary window had foreign GPU work \u2014 clean rerun pending'
            if verdict == 'no-coverage':
                return 'primary window lacked sampler coverage \u2014 clean rerun pending'
        if release in R27_RELEASES:
            for r in r27['receipts']:
                if r.get('speed_usable') or r.get('release') != release or r['config'].get('spec') != spec:
                    continue
                if not all(str(r['config'].get(k)) == str(v) for k, v in selector.items()):
                    continue
                verdict = (r.get('isolation') or {}).get('verdict')
                problems = (r.get('r27') or {}).get('generator_problems') or []
                if verdict == 'contaminated':
                    return 'window had foreign GPU work \u2014 nothing plots until a clean window lands'
                if verdict in ('no-coverage', 'no-command-record'):
                    return 'window lacked full sampler coverage \u2014 nothing plots without a covered window'
                if problems:
                    return 'producer/generator gate failed: ' + '; '.join(problems)[:160]
                return 'not speed-eligible \u2014 pending a clean producer cell'
        return None

    def r26_status(rid: str) -> str:
        prim = [r for r in r26['receipts'] if r['release'] == rid]
        reruns = [r for r in clean['speed'] if r['release'] == rid]
        usable = [r for r in prim if r.get('speed_usable')]
        superseded = [r for r in prim if r.get('superseded_by')]
        dirty = [r for r in prim if not r.get('speed_usable') and not r.get('superseded_by')]
        parts = []
        if usable:
            parts.append(f"{len(usable)} primary cells proven clean by the isolation log")
        if superseded:
            parts.append(f"{len(superseded)} contaminated cells superseded by verified-quiet reruns")
        if dirty:
            parts.append(f"{len(dirty)} contaminated cells excluded, reruns pending")
        if reruns:
            parts.append(f"{len(reruns)} clean rerun cells landed")
        if clean['plan']['planned']:
            parts.append(f"rerun plan {sum(1 for c in clean['plan']['cells'].values() if c['clean'])}/{clean['plan']['planned']} clean")
        return ' \u00b7 '.join(parts) if parts else 'no validated cells yet'

    def r27_status(rid: str) -> str:
        if not r27['present']:
            return 'R27 battery root absent \u2014 the R27 images have not been GPU-tested on this host yet'
        parts = []
        cells = [r for r in r27['receipts'] if r['release'] == rid]
        eligible = [r for r in cells if r.get('speed_usable')]
        if eligible:
            parts.append(f"{len(eligible)} producer-gated clean-window speed cells eligible")
        elif cells:
            parts.append(f"{len(cells)} speed cells attempted, none eligible yet (producer gate + clean window + pinned image)")
        elif rid != 'r27-auto' and r27['speed']['present']:
            parts.append('speed cells attempted: 0')
        cases = [c for c in r27['scheduler']['cases'] if c['release'] == rid]
        if cases:
            roles = sorted({c['role'] for c in cases if c.get('role')})
            parts.append(f"scheduler attribution: {len(cases)} cases ({'; '.join(roles)})")
        if rid == 'r27-auto':
            parts.append('auto prefill-policy defaults are a control, never a release performance gain')
        return ' \u00b7 '.join(parts) if parts else 'no producer receipts yet'

    lins = [lineage_rows(L, plot_rows, pending_note) for L in LINEAGES]
    collisions = collect_collisions()
    ov = overview_rows(plot_rows, r26, r26_status, r27_status)
    missing = [r['id'] for r in all_rows if not r.get('exists') or r.get('error')]
    missing += [r['id'] for r in r27['scheduler_rows'] if r.get('error')]
    n_clean_cells = sum(1 for c in clean['plan']['cells'].values() if c['clean'])
    contaminated_primary = [r for r in r26['receipts'] if (r.get('isolation') or {}).get('verdict') == 'contaminated']
    oom_boots = [b for b in r26['boots'] if b.get('failure_note')]
    gaps = [
        'Engine version string is identical (0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1\u2026) on every JJ image, so image identity comes from scripts/inspect, never from receipts.',
        'r7/r8/r10 controls, r12 LMCache, r15 full matrix, r18 and the tip-* sweeps have no retained launch line: their configs are inferred from geometry + same-day scripts and are flagged, not silently trusted.',
        'Receipts before 4 Sep report KV as bench metrics x CP; only r24/r25/r26 (docker logs), r7 (report) and the r20/r22 production boots (run.log) have the server-reported pool.',
        'ctx0 chart cells (1024-token requests) flag momentary queueing as capacity_limited; those cells are shown as "queued" and kept, true under-fills are starred.',
        'The r5 arms ran with prefix caching disabled and image-default slots; they are pre-JJ context, not part of any matched lineage.',
        'r17 custom-config arms (_destroyed/reaper compose) and the r22 QAD checkpoint runs are shown with their own labels and never compared against stock cells.',
        'R26 official and the D-Rock R26 overlay are separated by the image digest in each inspect sidecar; receipts without a sidecar are marked unverified.',
        'Collision probe: r20-battery.sh and r22-battery.sh never passed FAIRNESS_ENGINE, so those cells ran the launcher default; r20 behaves like fairness off (87% stall), r22 like compute-share 0.4 (42% stall, 3.7k prefill). No r22 test-container log was kept, so the r22 default is inferred from the signature.',
        'KV pool figures from bench metrics x CP disagree with the server figure on DCP4 (e.g. R25 DCP4 NVFP4 no-spec: 36.51M bench-derived vs 32.68M server); the charts prefer the server figure wherever a docker log exists.',
        f'R26 speed windows are gated on the GPU isolation sampler plus the recorded command times: {len(contaminated_primary)} primary cells had foreign GPU work in their window; '
        f'{len(sup_speed)} of them are superseded by verified-quiet same-label reruns (image, model, config and bench parameters all re-checked), the rest stay unplotted until their rerun lands '
        f'({n_clean_cells}/{clean["plan"]["planned"]} rerun cells clean so far).',
        'R26 native-offload arms failed at boot, never benchmarked: ' + ('; '.join(b['label'] + ': ' + b['failure_note'] for b in oom_boots) if oom_boots else 'none observed') + '.',
        f'Acceptance arms are a separate workload: all {len(clean["acceptance"])} R26-phase acceptance windows rerun in verified-quiet boots and supersede the primary repeats; '
        'natural-chat acceptance (32 chats/arm, descriptive, not a semantic-equivalence proof) is sourced from followups/realistic-acceptance-comparison.json.',
        f'R27 speed cells attempted: {len(r27["speed"]["cells"])}; eligible: {len(r27["speed"]["eligible"])}. A cell needs the producer gate, a fully covered clean '
        'GPU window and the pinned source-snapshot image; until one lands the R27 slots stay empty, never zero-filled or estimated.',
        'The R27 speed plan is C1/C8 at ctx0 and 32k, 60 s, 8192-token cells: no C16 exists in the plan, so R27 never joins the R24\u2192R26 C1/C16 battery lineage '
        'and nothing is compared across that break. QAD checkpoints are a separate runbook and no QAD point is invented here.',
        'The patched-auto R27 image ships the auto prefill-policy defaults; fixed-vs-auto and configured-auto-vs-as-shipped deltas are scheduler attribution '
        'controls (manifest r27.scheduler), never release performance gains.',
        'R27 stock exact recurrent checkpoint claims cover DCP1 no-spec/MTP only; DCP4/DFlash and external cache retention are aligned, not promised by the speed cells.',
        'R27 stock/patched/auto identity and patches come from the pinned extracted-source snapshot (r27-source-20260906/manifest.json): per-arm file SHA-256s '
        'prove the patch set; the inherited source.lock and the leaf cache-complete launcher are identical across the three arms.',
    ]
    for L in lins:
        gaps += [f"{L['id']}: {g}" for g in L['gaps']]
    # acceptance evidence: Main's clean counter means + the followups natural-chat comparison, embedded read-only
    acceptance_evidence = {}
    festr_path = r26_root / 'clean-reruns' / 'festr-clean-counter-summary.json'
    if festr_path.exists():
        try:
            festr = json.load(open(festr_path))
            acceptance_evidence['festr_clean_counter_summary'] = dict(path=str(festr_path), method=festr.get('method'), rows=festr.get('rows'))
        except (OSError, ValueError):
            acceptance_evidence['festr_clean_counter_summary'] = dict(path=str(festr_path), error='unreadable')
    rac_path = r26_root / 'followups' / 'realistic-acceptance-comparison.json'
    if rac_path.exists():
        try:
            rac = json.load(open(rac_path))
            acceptance_evidence['realistic_acceptance_comparison'] = dict(
                path=str(rac_path), scope=rac.get('scope'),
                arms=[dict(arm=a.get('arm'), requests=a.get('requests'), completed_visible_answers=a.get('completed_visible_answers'),
                           acceptance_percent=a.get('acceptance_percent'), clean_window=a.get('clean_window')) for a in rac.get('arms') or []])
        except (OSError, ValueError):
            acceptance_evidence['realistic_acceptance_comparison'] = dict(path=str(rac_path), error='unreadable')
    manifest = dict(
        generated_at=stamp, generator='scripts/r26/release_history.py', host=HOST,
        sources=dict(repo_results=str(RESULTS), local_results=str(LOCAL), r26_battery=str(r26_root),
                     r27_battery=str(r27['root']), r27_source_manifest=r27['source_provenance'].get('path'),
                     docker_images_snapshot=docker_image_snapshot()),
        releases=[dict(r, status=r.get('status', 'measured')) for r in RELEASES],
        match_fields=list(MATCH_FIELDS), soft_fields=list(SOFT_FIELDS),
        receipts=all_rows + r26['acceptance'] + clean['acceptance'] + r27['scheduler_rows'],
        battery_boots_without_receipt=battery_boots,
        other_receipts=collect_other(),
        collision_series=collisions,
        lineages=lins,
        overview=ov,
        r27=dict(root=r27['root'], present=r27['present'],
                 source_provenance=r27['source_provenance'],
                 speed=r27['speed'],
                 scheduler=r27['scheduler'],
                 receipts=[r['id'] for r in r27['receipts']],
                 scheduler_receipts=[r['id'] for r in r27['scheduler_rows']],
                 speed_usable=[r['id'] for r in r27['receipts'] if r.get('speed_usable')]),
        r26=dict(root=r26['root'], present=r26['present'], decode_receipts=[r['id'] for r in r26['receipts']],
                 acceptance_receipts=[r['id'] for r in r26['acceptance']],
                 boots=r26['boots'], excluded=r26['excluded'], interrupted=r26['interrupted'], failed_gates=r26['failed_gates'],
                 gate_summary=r26['gate_summary'], gates=r26['gates'],
                 isolation_policy=dict(
                     events=r26['isolation_events'], samples=r26['isolation_samples'],
                     method='a receipt plots only when the sampler log covers its recorded bench.command window (>= 2 samples, edges and gaps within 10 s) '
                            'with zero foreign GPU processes; verdicts per receipt are in receipts[].isolation',
                     speed_usable=[r['id'] for r in r26['receipts'] if r.get('speed_usable')],
                     excluded=[dict(id=r['id'], verdict=r['isolation']['verdict'], evidence=r['isolation'].get('evidence'),
                                    superseded_by=r.get('superseded_by')) for r in excluded_r26]),
                 clean_reruns=dict(root=clean['root'], present=clean['present'], planned=clean['plan']['planned'],
                                   planned_labels=clean['planned_labels'],
                                   clean_labels=sorted(l for l, c in clean['plan']['cells'].items() if c['clean']),
                                   pending_labels=clean['pending_labels'],
                                   gate='producer plan cell clean flag (unique result_label across resume ledger entries; latest eligible '
                                        'canonical result preferred) + this generator\'s own isolation recheck; both required',
                                   supersessions=sup_speed, acceptance_supersessions=sup_acc,
                                   conflicts=conflicts + acc_conflicts,
                                   speed=[r['id'] for r in clean['speed']], acceptance=[r['id'] for r in clean['acceptance']]),
                 acceptance_evidence=acceptance_evidence),
        missing_receipts=missing,
        comparability_gaps=gaps,
        comparison_notes=[
            dict(release='sglang-accel', note='SGLang runs the same NVFP4 checkpoint on a different engine; C1 69 / C16 504 is an engine comparison, not a release gain.'),
            dict(release='tr3-exl3', note='EXL3 4bpw on TP2 with 4 slots: single-stream specialist (C1 120-134), concurrency cells under-filled by design.'),
            dict(release='drock-nonflash-exl3', note='TR3 3.0bpw checkpoint on a JJ r15-derived nonflash runtime; 900k context, 64-token pages, ~46-73 tok/s single stream.'),
            dict(release='r22-qad', note='Stock r22 runtime with QAD-Step1750 weights; DFlash K7 C1 214 vs 228 stock is a checkpoint effect.'),
            dict(release='r25-pr646', note='Draft PR646 rebased on r25: capacity/prefix behaviour, not a speed release; 10/12 concurrent Estonia vs 12/12 stock.'),
            dict(release='r27-patched', note='Eval candidate on the exact R27 source (#674/#676/a338ca71); any delta is a candidate finding, not a release.'),
            dict(release='r27-auto', note='Patched source with auto prefill-policy defaults baked as shipped; auto-vs-fixed differences are controls, never release performance gains.'),
        ],
    )
    (out_dir / 'history-manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    audit_rows = all_rows + r26['acceptance'] + clean['acceptance'] + r27['scheduler_rows']
    n_cells = write_cells_csv(audit_rows, out_dir / 'history-cells.csv')
    manifest['cell_rows'] = n_cells
    if render:
        render_main(plot_rows, clean, r27, out_dir / 'history-main.png', stamp)
        render_matched(lins, collisions, out_dir / 'history-matched.png', stamp)
        render_overview(ov, out_dir / 'history-inventory.png', stamp)
    write_kv_evidence(audit_rows, out_dir / 'history-kv-evidence.json')
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--battery-root', default=os.environ.get('BATTERY_ROOT', str(DEFAULT_R26_ROOT)))
    ap.add_argument('--r27-root', default=os.environ.get('R27_BATTERY_ROOT', str(DEFAULT_R27_ROOT)),
                    help='R27 battery root (default: drock-lmcache/r27-battery); absent or partial roots never produce datapoints')
    ap.add_argument('--out', default=str(OUT_DIR))
    ap.add_argument('--no-render', action='store_true', help='write the manifest/CSV only')
    ap.add_argument('--gate', action='store_true',
                    help='record a release-history gate through scripts/r26/runtime.py (use when run as a supervised phase)')
    args = ap.parse_args(argv)
    try:
        m = build(Path(args.battery_root), Path(args.out), render=not args.no_render, r27_root=Path(args.r27_root))
    except Exception as error:  # a broken generator must fail loudly, never leave a stale chart behind
        if args.gate:
            _gate(False, dict(error=repr(error)))
        raise
    decode = [r for r in m['receipts'] if 'cells' in r]
    measured = [o for o in m['overview'] if o['status'] == 'measured']
    print(f"releases inventoried: {len(m['releases'])} · with decode receipts: {len(measured)} · decode receipts: {len(decode)} · cells: {m['cell_rows']}")
    for L in m['lineages']:
        ok = [x for x in L['members'] if x.get('status', '').startswith('matched')]
        print(f"lineage {L['id']}: anchor={L['anchor']} matched={len(ok)} gaps={len(L['gaps'])}")
        for g in L['gaps']:
            print(f'   gap: {g}')
    iso = m['r26']['isolation_policy']
    print(f"r26 root {m['r26']['root']}: present={m['r26']['present']} decode receipts={len(m['r26']['decode_receipts'])} "
          f"acceptance receipts={len(m['r26']['acceptance_receipts'])} clean primary windows={len(iso['speed_usable'])} excluded={len(iso['excluded'])} "
          f"excluded stems={m['r26']['excluded']['stems']} gates={m['r26']['gate_summary']} failed={[g['name'] for g in m['r26']['failed_gates']]}")
    cr = m['r26']['clean_reruns']
    print(f"clean reruns: planned={cr['planned']} clean={len(cr['clean_labels'])} pending={len(cr['pending_labels'])} "
          f"supersessions={len(cr['supersessions'])} speed + {len(cr['acceptance_supersessions'])} acceptance, conflicts={len(cr['conflicts'])}")
    if cr['pending_labels']:
        print('   rerun pending:', ', '.join(cr['pending_labels']))
    r27 = m['r27']
    print(f"r27 root {r27['root']}: present={r27['present']} speed cells={len(r27['speed']['cells'])} eligible={len(r27['speed_usable'])} "
          f"scheduler receipts={len(r27['scheduler_receipts'])} scheduler summary={r27['scheduler']['present']} "
          f"source snapshot={r27['source_provenance'].get('present')}")
    if r27['speed'].get('error'):
        print(f"   r27 speed summary: {r27['speed']['error']}")
    if r27['scheduler'].get('error'):
        print(f"   r27 scheduler summary: {r27['scheduler']['error']}")
    if m['missing_receipts']:
        print('MISSING/UNREADABLE receipts:', ', '.join(m['missing_receipts']))
    print('wrote', Path(args.out) / 'history-manifest.json')
    ok = not m['missing_receipts']
    if args.gate:
        _gate(ok, dict(manifest=str(Path(args.out) / 'history-manifest.json'), decode_receipts=len(decode), cells=m['cell_rows'],
                       r26_decode_receipts=len(m['r26']['decode_receipts']), r26_excluded=m['r26']['excluded']['stems'],
                       r27_present=r27['present'], r27_speed_attempted=len(r27['speed']['cells']),
                       r27_speed_eligible=len(r27['speed_usable']), r27_scheduler_present=r27['scheduler']['present'],
                       lineage_gaps={L['id']: len(L['gaps']) for L in m['lineages']}, missing=m['missing_receipts']))
    return 0 if ok else 1


def _gate(passed: bool, detail: dict) -> None:
    """Record through Main's runtime when it is importable; the generator itself never needs it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import runtime  # noqa: WPS433 - sibling module owned by Main
    except ImportError as error:
        print(f'gate not recorded: runtime.py unavailable ({error})')
        return
    runtime.record_gate('release-history', passed, detail)


if __name__ == '__main__':
    sys.exit(main())
