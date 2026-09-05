#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path('/home/josh/omp-workspace/drock-lmcache/r25-battery')
MAIN_DONE = ROOT / 'DONE'
STAGES = [
    ('dcp2-mtp3', ROOT / 'run_dcp2_mtp3.py', 1800),
    ('fp8-recheck', ROOT / 'run_fp8_recheck.py', 4200),
    ('execution-ab', ROOT / 'run_execution_ab.py', 3600),
    ('extended', ROOT / 'run_r25_extended.py', 7200),
    ('restore-ab', ROOT / 'run_restore_ab.py', 3600),
    ('prefill-variants', ROOT / 'run_prefill_variants.py', 7200),
    ('agent-hot-queue', ROOT / 'run_agent_hot_queue.py', 18000),
    ('pr646', ROOT / 'run_pr646.py', 18000),
    ('matrix-summary', ROOT / 'build_summary.py', 300),
    ('hot-summary', ROOT / 'summarize_hot_queue.py', 300),
    ('report-data', ROOT / 'build_report_data.py', 300),
    ('charts', ROOT / 'render_r25_charts.py', 300),
    ('acceptance', ROOT / 'validate_acceptance.py', 300),
]

print('R25 COORDINATOR WAITING', flush=True)
deadline = time.time() + 12 * 60 * 60
while not MAIN_DONE.exists():
    if time.time() >= deadline:
        raise TimeoutError('main R25 battery did not finish within 12 hours')
    time.sleep(20)

print('R25 COORDINATOR MAIN DONE', flush=True)
for label, script, timeout in STAGES:
    print(f'R25 COORDINATOR START {label}', flush=True)
    with (ROOT / f'coordinator-{label}.log').open('w') as output:
        subprocess.run(
            ['python3', str(script)],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    print(f'R25 COORDINATOR DONE {label}', flush=True)
(ROOT / 'PIPELINE_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
print('R25 COORDINATOR COMPLETE', flush=True)
