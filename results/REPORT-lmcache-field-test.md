# GLM-5.3-Flash LMCache field test — 2026-08-31 (independent, 4× RTX PRO 6000)

Test host: 4× RTX PRO 6000 Blackwell (97.9 GB each), single-root PCIe (NODE topology), 251 GB host RAM, NVMe L2. Baselines via llm_decode_bench (C1/4/8/16 × 0–128k ctx, 30 s cells, aggregate t/s).

## 1. Independent cross-host validation of the unchanged-r7 control

Image `voipmonitor/vllm@sha256:488ddf75…` (JJ community r7), DFlash2/DCP4+CKV_GATHER, MXFP8 draft, GMU 0.90, per the r7 runbook. Markers verified (B12X PCIe, B12xMxfp8LinearKernel, FA2, split cache pages, graphs). KV pool 3,915,860 tokens (CN4 qualification: 3,974,266, −1.5%).

| ctx\conc | C1 | C4 | C8 | C16 |
|---|---:|---:|---:|---:|
| ours, ctx 0 | 149.0 | 377.8 | 538.2 | 769.5 |
| CN4 published, ctx 0 | 147.7 | 384.6 | 516.7 | 728.7 |
| ours, ctx 100–128k | 142.6–164.8 | 367.2–369.7 | 522.5–543.5 | 767.8–772.6 |
| CN4 published, ctx 100k | 136.8 | 362.4 | 532.6 | cap-limited |

Prefill: ours 7,302–7,623 t/s at 32k–128k (CN4 canonical 32,320-token: 7,562). All cells within ±5% of CN4 → the r7 control reproduces on independent hardware. (C16 32k cell was capacity-limited 15/16, consistent with CN4's cap-limited 100k C16.)

## 2. Public-assembly failure map (r7 + LMCache `integration/glm53-upstream-consolidation` @ 6e479686)

We rebuilt your overlay from public pieces (r7 parent + your branch, CUDA-built in-image, `--no-build-isolation`, verified `lmcache_native`/`cuda_ops` import) and could not reach the qualified TP4/DCP4 config through any public wiring. Five exact gates, all reproducible:

1. **Allocator gate (any connector).** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (image default) → pydantic hard error at VllmConfig validation: connector incompatible unless `enable_cumem_allocator`. Fix: expandable off or cumem on.
2. **cumem path OOMs at weight load.** With `--enable-cumem-allocator` (any GMU 0.80/0.90, CUDAGRAPH_MODE FULL or NONE): OOM in `load_weights`, invariant footprint — 85.64 GiB PyTorch + 39.68 GiB "private pools" (CuMem VMM reservation), 95.01 GiB card. The VMM pool does not scale with GMU as configured.
3. **Builtin `LMCacheConnectorV1` is not HMA.** vLLM turns off the hybrid KV manager; GLM-5.3-Flash then hard-fails: `Hybrid KV cache manager cannot be disabled when attention layers mix DCP-replicated and DCP-sharded KV cache specs`. Fatal at DCP4 *and* DCP1.
4. **Branch `LMCacheConnectorV1Dynamic` via external `kv_connector_module_path`** — resolves and loads, but is not `SupportsHMA` either → identical gate-3 failure.
5. **`LMCacheMPConnector` (SupportsHMA)** — resolves via external module path, validates geometry, starts compiling, and is the only path that keeps HMA alive; but it requires the cumem/VMM route (REGISTER_KV_CACHE maps remote buffers) → gate 2.

Sidecar notes (MP server runs cleanly in a separate container: L1/L2 eviction, store, prefetch controllers up on ZMQ :5555): under DCP4 the connector demands **chunk size multiple of 9216** (vLLM block 256 × DCP4 scaling); MP keys must nest under `kv_connector_extra_config` (top-level keys rejected by pydantic).

**Conclusion:** the qualified TP4/DCP4 config is reachable only through `ghcr.io/yatesdr/jovian-judgement-glm53-lmcache@sha256:bdb5bdc2…`, which is still **private** (unauthorized as of 07:55 UTC). Until visibility flips or read tokens go out, nobody outside can reproduce the qualification — the failure map above is what every public assembler will hit.

## 3. Community-r5 reference (LMCache-off): DCP4 vs DCP1

Image `glm53-flash-nvfp4-dflash2-community-20260830-r5`, DFlash2, MXFP8 draft, no LMCache:

| | DCP4 | DCP1 |
|---|---:|---:|
| KV pool | 7.55M tokens | 3.53M tokens |
| Prefill (16k–128k) | 4.6k–5.0k t/s | 5.0k–5.2k t/s |
| C1 decode | 148–174 | 152–183 |
| C4 decode | 395–417 | 428–485 |
| C8 decode | 589–622 | 649–691 |

DCP4 = 2.14× KV pool at ~8–10% decode cost at C4/C8 on this host. (Also: r7 control at 3.92M KV shows r7's page separation reclaimed most of r5-DCP4's 7.55M-vs-3.53M gap at *lower* waste — but note r7 ran GMU 0.90 fp8-KV with the separate-pages PR #535.)

## 4. For Festr — SGLang accel image cannot load the MXFP8 draft

`local/sglang:glm-5.3-flash-accel` + `--speculative-draft-model-quantization mxfp8` → `ModelOptFp8Config.from_config: only supports regular FP8 quantization, but found 'MXFP8'. Use the native 'mxfp8' quantization method…`. The pure-MXFP8 ModelOpt draft path evidently ships with your next image; backporting it (or native-mxfp8 dequant for DFlash2DraftModel) would let SGLang users halve draft VRAM like vLLM does. On vLLM the MXFP8 draft runs natively (r5 + r7): KV pool 3.87M → 3.92M (+1.5%) vs BF16 draft, markers clean.

## 5. Improvement ideas (grounded in tonight's numbers)

1. **Ship the MTP3+LMCache profile as production** — your own matrix shows it dominates r7+DFlash2 everywhere (C1 180.7 vs 147.7, C16 1,005.7 vs 728.7, prefill 8,296 vs 7,562, KV 130.61 vs 113.55 GiB). Channel data agrees dflash2 wins coding-style acceptance but loses general/mixed concurrency — so publish both profiles with a one-line "pick by workload" note rather than defaulting dflash2.
2. **Cudagraph capture size:** with MTP3 × 16 seqs, (max_num_seqs+1)×spec_depth = 64, not the 128 default (also raised by timricese tonight) — frees VRAM → more KV at equal GMU.
3. **Reclaim hybrid-layout padding waste** (up to 29.41% of pool per the pedigree): packing the ten padding layers could return up to ~1.1M effective tokens on TP4/DCP4.
4. **DCP2 bench cell:** our DCP1-vs-DCP4 data hints DCP2 may sit near the DCP1 decode curve with a mid pool (tuna6975 measured 2.58M KV at DCP2) — worth one matrix run before locking profiles.
5. **L1 sizing on ≥256 GB-RAM hosts:** 64 GiB lazy L1 is conservative with 251 GB installed; 128 GiB lazy + `use_odirect: true` on NVMe L2 should cut restore latency materially.
6. **r8 (#541 prefill-cadence fix) matrix rerun** — the C8/C16 cells under mixed prefill+decode load should move; our r7 numbers above are the pre-fix reference.

Artifacts: bench JSONs (`bench-r7-control-mxfp8.json`, `bench-arm0-lmcache-off.json`, `bench-arm0b-dcp1-lmcache-off.json`), serve/repro scripts, and both of your compose files + pedigree archived at the test host.
