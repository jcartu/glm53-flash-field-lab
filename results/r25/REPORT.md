# R25 GLM-5.3-Flash field report

## Scope

Published image:

```text
voipmonitor/vllm:jovian-judgement-community-20260904-r25
sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804
```

The embedded `/opt/glm53-flash/source.lock` hashes to
`8d99c847855c32bb2348cd58f8a3a72f97df9c5fbeb38223f94a3bdfb590d9d0`.
The image has three filesystem layers. R25 changes the LMCache Python retrieve
path; its vLLM and B12X package trees are byte-identical to R24.

Field host: four RTX PRO 6000 Blackwell 96 GB GPUs on one PCIe 5 root, TP4,
1,048,576-token model limit, 4,096 target tokens per scheduler step, max 32
sequences, and compute-share fairness at 0.4 unless a control says otherwise.
LMCache and native-offload arms use private 128 GiB container shared memory.

## Verdict

R25's intended LMCache improvement is real. In an exact matched R24/R25
one-million-token filesystem replay, R25 reduced complete API wall time from
4.65 to 3.82 seconds, 17.9%. The connector's logged handoff segment dropped
from 150 to 29 ms, but that segment is not the complete restore latency. Both
arms restored 999,424 tokens, recomputed 576, returned byte-identical visible
output, and produced zero unpinned-memory fallback warnings.

The matched R25, R24, R25 no-spec execution bracket was -1.81%, -0.93%, and
-1.49% at C1, C8, and C16. That supports the source-lock claim that execution
is unchanged. A slow first DCP4 FP8 pass coincided with elevated host CPU load
and low PCIe traffic; a controlled recheck returned to the normal range. Both
receipts are retained.

The strict acceptance ledger is 44/47. The three misses are reported rather
than hidden:

1. LMCache DFlash long-generation stress was 23/24 clean. The single Apollo
   guidance computer request ran away at the 8,192-token limit in wave one;
   the same topic was clean in both repeats.
2. Lavd was 14/16 exact across two waves, with two coherent but wrong final
   answers and zero transport/runtime errors.
3. Draft PR646 scored 10/12 on concurrent Estonia while its matched stock arm
   scored 12/12. Its capacity and prefix-cache results are strong, but it is
   not clean enough for production.

## Topology and capacity

The DCP4 values below use the controlled recheck, not the transient first pass.
All decode values are aggregate output tokens per second from 30-second cells.

```text
mode             DCP   C1     C8     C16    32K prefill   KV tokens
no spec            1   194    833    1226       7,874       4.77M
MTP3               1   200    687     966       7,898       3.84M
DFlash K7          1   242    706    1042       8,243       4.06M
no spec            2   176    763    1133       9,111      10.93M
MTP3               2   199    668     938       9,225       8.87M
DFlash K7          2   198    640     868       9,203       9.37M
no spec            4   164    704    1059       9,430      21.93M
MTP3               4   179    625     880       9,377      17.85M
DFlash K7          4   200    643     909       9,253      18.60M
```

Packed NVFP4, DCP4, no spec reached 36.51M runtime KV tokens. B12X KDA reached
40.58M with C1/C8/C16 at 172/727/1066 and 8,925 tok/s client 32K prefill.
Complete-KV selection measured 9,488 tok/s versus 7,807 for the rank-local
control, a 21.5% gain. Native offload booted with private SHM and measured
11.38 seconds cold versus 0.99 seconds warm on the 80K probe.

## LMCache

```text
DCP4 packed NVFP4       GPU-only C1/C8/C16    LMCache C1/C8/C16
no spec                       172 / 724 / 1071          172 / 725 / 1074
MTP3                          176 / 639 /  883          192 / 622 /  862
DFlash K7                     182 / 612 /  889          199 / 635 /  899
```

Speculative output differences track acceptance variation; no-spec is the
stable execution comparison.

Exact independent cache probes:

```text
80K API wall       cold 11.87s   warm 0.68s   restart 0.72s
1M API wall        cold 124.63s  warm median 2.55s
1M after restart   first 3.14s   steady median 2.55s
1M output          9/9 byte-identical visible answers
Eviction churn     40/40 byte-identical replies
Cold 1M needles    3/3 hits at 10%, 50%, and 90% depth
```

The persisted R24 L2 objects were also readable directly by R25. The original
80K and depth-needle probes therefore arrived already hot, completing in about
one second and 4-9 seconds respectively. Unique first-block identities and
cache salts were added for the true cold measurements above.

Eight-run 33K cold-prefill medians were 10.24K tok/s GPU-only, 9.73K with the
existing L2, 9.81K RAM-only, and 9.77K with a fresh L2. Steady medians are
4.2-5.0% below GPU-only. The first post-boot LMCache request took about 8.1
seconds, so aggregate eight-run throughput is lower than the steady median.
The filesystem is not the steady-state bottleneck.

The LMCache sidecar created no GPU process. GPU work stayed in the four vLLM
workers.

## Fairness and the agent hot-request question

The original collision probe still shows why compute-share matters:

```text
policy              baseline decode   during 65K cold   slowdown   p95 gap
fairness off             181 tok/s          4.8 tok/s      97.4%    558 ms
compute share 0.4        194 tok/s        112.4 tok/s      42.0%      5.8 ms
```

A separate workload models the question iSource actually asked. Each session
cycles decode, a roughly 358-token incremental prefill, then decode again.
Unique 128K cold requests arrive every 15 seconds. Every C8/C16 cell has two
30-second baseline runs and two 60-second storm runs.

```text
policy              C   hot p95 TTFT   turn-rate retained   max queue
fairness off         8       4.36s             18%               2
compute share 0.4    8      15.92s             32%             8.5
fairness off        16      16.91s             28%               8
compute share 0.4   16      18.86s             39%              17
```

The answer is yes: small continuation prefills still queue behind periodic
large cold prefills. Compute-share preserves more aggregate turn throughput,
but it does not protect hot-turn latency and produces deeper queues. Source
inspection agrees with D-Rock's description: compute-share chooses the decode
or prefill service class, while the default FCFS request queue remains inside
the prefill class. Adaptive short/long priority or proper chunk interleaving is
a separate scheduler feature, not something R25 already provides.

## Draft PR646

PR646 was rebased locally onto exact R25 commit
`d49385468458cf97dff0fc8d9c8863f8082abf4f`. The PR was draft and GitHub
reported a dirty merge state. Two source conflicts and one launcher guard were
resolved for the isolated field image. The focused rebased prefix-cache file
passed 46 tests. Nothing was pushed or deployed.

TP4/DCP1, DFlash K7, FP8 KV:

```text
geometry                 KV tokens   repeat hit   repeat wall
stock 256 / 256             1.01M       52,224        0.15s
stock 2048 / 2048           3.04M            0        6.70s
PR646 2048 / 256            3.04M       52,224        0.17s
```

PR646 therefore keeps the stock coarse-page capacity, gives 3.01x the
fine-page capacity, and restores the complete 256-token-granularity repeat
hit. Across the matched 0/16K/32K decode matrix its mean absolute delta versus
stock coarse was 2.48%; the worst cell was -5.57%. The 90K repeated needle was
4/4.

The caution is concurrent quality: patched Estonia was 10/12, versus 12/12 on
the matched stock control, with one miss in each patched wave. That is enough
to keep this in the "interesting draft, do not run in production" bucket.
DCP greater than one was intentionally not tested because the PR explicitly
leaves it unsupported/unmeasured.

## Receipts

- `summary.json`: all 18 primary runtime configurations.
- `report-data.json`: normalized chart/report input.
- `acceptance.json`: all 47 acceptance checks and the three failures.
- `hot-off.json`, `hot-compute-share-0.4.json`: raw cyclic agent workload data.
- `pr646-summary.json`, `pr646-rebase-note.txt`, `pr646-unit-tests.log`: draft
  PR provenance and results.
- `charts/`: seven 160-DPI dark-theme charts.

The mutable Discord R25 announcement says the restart check was 1.215 seconds;
the immutable embedded source lock says 1.311 seconds. Both source claims are
preserved in `provenance.json`, and the field measurements above are reported
independently.
