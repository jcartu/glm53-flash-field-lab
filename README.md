# GLM-5.3-Flash Field Lab

Independent validation harness and results for GLM-5.3-Flash serving stacks on
4x RTX PRO 6000 Blackwell (SM120, single-root PCIe 5, 256 GB host RAM).

Everything here was built to answer one question the hard way: **what actually
holds up when you run it for real?** Speed matrices, restart persistence,
cache-corruption batteries under eviction churn, and decode/prefill collision
measurements — all reproducible from this repo.

## Harness

| Script | What it does |
|---|---|
| `scripts/bench.sh` | Standard decode/prefill matrix (wraps llm_decode_bench, C1-16 x 0-128k) |
| `scripts/prefix_test.py` | Cold/warm/restart TTFT probe — proves cache survives restarts |
| `scripts/eviction_test.py` | Corruption battery: 40 docs forced through L1/L2 eviction, byte-identical verification |
| `scripts/decode_prefill_collision.py` | Scheduler-fairness probe (authored by D-Rock, included unmodified — measures decode stall during cold prefill) |
| `scripts/serve-r7.sh` | Parameterized launcher for the JJ community vLLM images (image, TP/DCP, speculator, KV dtype, capture size all overridable) |
| `scripts/spec-matrix.sh` | Image x speculator x concurrency sweep for comparison charts |
| `scripts/serve-lmcache-glm53.sh`, `scripts/serve-mp-sidecar.sh` | Two-arm LMCache test rigs (in-process + MP sidecar arrangements) |
| `scripts/serve-glm53-flash-nvfp4.sh` | Production SGLang launcher (NVFP4 + DFlash2) |

## Results index

| File | What it shows |
|---|---|
| `results/r24/` | Full R24 battery: 16 runtime configs, DCP1/2/4, three speculators, FP8/NVFP4 KV, GPU/LMCache/native modes, 1M needles and replay, fairness, corruption, tool ordering, Estonia/Lavd, charts |
| `results/bench-r15.json` | JJ r15 — C16 1,130 t/s, decode flat to 128k (record on this host) |
| `results/collision-r15.json` | 90% decode stall during one cold 65k prefill (fairness finding) |
| `results/bench-r7-control-mxfp8.json` | r7 control — cross-validated vs CN4 within +-5% |
| `results/eviction-drock353d7.json`, `results/eviction-auto1024.json` | LMCache corruption batteries — 40/40 identical under churn |
| `results/prefix-r12.json`, `results/prefix-auto1024.json` | Cross-restart L2 recovery proofs |
| `results/bench-tr3-4bpw*.json` | brandon's EXL3 4bpw TP2 — single-stream specialist profile |
| `results/tip-*.json`, `results/chart-*.json` | Speculator tipping-point data |
| `results/REPORT-lmcache-field-test.md` | Full written report of the LMCache field test |
| `results/chart-decode-configs.png` | The progression chart |

## Method notes

- All decode numbers: aggregate tokens/sec, 30s cells, temp 0, same host
  throughout every comparison in a table.
- Concurrency cells only run when the config's max-num-seqs admits them
  (the >16-stream cliff on JJ images is a cudagraph-capture issue:
  set MAX_CUDAGRAPH_CAPTURE_SIZE = concurrency x 8 for dflash2).
- Corruption verdicts require byte-identical replies across eviction churn,
  not just "looks fine".

## Attribution

Built on top of the RTX6kPRO / Local Inference Lab community's work, with
thanks: festr. (JJ vLLM image line, B12X), D-Rock (LMCache consolidation
branch, the collision script, production composes), brandonmusic (TR3 EXL3
4bpw artifact + runtime composes), luke (NVFP4 quants), voipmonitor (image
publishing). Their repos and channels are the source; this is just the field
data.
