#!/usr/bin/env python3
"""Resume the interrupted QAD step2500 tail with the nine recorded phases preserved.

The second GPU fault (Xid120 on GPU3, host reboot) interrupted the recovered
QAD run during the mixed-agent-scheduling phase.  This driver validates the
immutable source evidence, stages a fresh output root from it, and delegates
the two remaining phases to the guarded run_qualification coordinator:

* the nine recorded phases are preserved verbatim, including the measured
  agentic-prefix-cache-reuse failure (returncode 1);
* the scheduler phase is invoked as ``scheduler_recheck_phase.py --resume`` so
  the interrupted plan resumes from the copied evidence;
* the direct-dcp-peer-guard phase runs exactly as planned;
* the merged eleven-phase ledger is written only when both tail phases have
  actual invocation receipts, and the original phase plan is restored.

Plan only (CPU-only: no filesystem mutation, no services, no devices):
    python3 continue_qad_tail.py --plan-only \
        --source-root /home/josh/omp-workspace/drock-lmcache/r26-qad2500-recovered \
        --output-root /home/josh/omp-workspace/drock-lmcache/r26-qad2500-core0-tail

Execute (Main guarantees the original glm53-prod production container is
healthy; the coordinator drains it, runs the tail phases, and restores it):
    python3 continue_qad_tail.py \
        --source-root /home/josh/omp-workspace/drock-lmcache/r26-qad2500-recovered \
        --output-root /home/josh/omp-workspace/drock-lmcache/r26-qad2500-core0-tail
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continue_qad_battery as base

require = base.require

# The tail plan is the pinned original plan with one deliberate adjustment:
# the interrupted scheduler phase gains --resume so it reuses the complete
# saved plan outputs and resumes the partial plan from the copied evidence.
TAIL_PHASES = [
    ('mixed-agent-scheduling', 'scheduler_recheck_phase.py', ['--resume'], 21600),
    ('direct-dcp-peer-guard', 'peer_phase.py', [], 7200),
]
EXPECTED_RECORDED_PHASES = 9
MEASURED_FAILURE_ROWS = {'agentic-prefix-cache-reuse': 1}
SCHEDULER_DIR = 'scheduler-recheck'
PARTIAL_SCHEDULER_PLAN = 'official-static07-coverage'
ARCHIVE_DIR = 'pre-tail-interruption-archive'
ARCHIVED_ARTIFACTS = (
    'current-phase.json',
    'gpu-health-interruption.json',
    'gpu-health-preflight.json',
    'phase-mixed-agent-scheduling.log',
    'phase-progress.json',
    'qualification-interrupted.json',
    'tail-resume-hold.json',
    'test-final.docker.log',
    'test-final.gpu.txt',
    'test-final.inspect.json',
    'test-final.metrics.txt',
)
APPEND_ONLY_STREAMS = ('battery.log', 'gates.jsonl', 'gpu-isolation-events.jsonl')
WEIGHT_SUFFIXES = ('.safetensors', '.gguf', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.h5')

require(len(base.PINNED_PHASES) == 11, 'The pinned QAD plan must have eleven phases')
require(
    base.PINNED_PHASES[9] == ('mixed-agent-scheduling', 'scheduler_recheck_phase.py', [], 21600)
    and base.PINNED_PHASES[10] == ('direct-dcp-peer-guard', 'peer_phase.py', [], 7200),
    'The pinned plan tail changed',
)
require(
    TAIL_PHASES[0][:2] == base.PINNED_PHASES[9][:2]
    and TAIL_PHASES[0][3] == base.PINNED_PHASES[9][3]
    and TAIL_PHASES[0][2] == ['--resume'],
    'The scheduler tail phase must match the pinned plan except for the resume flag',
)
require(TAIL_PHASES[1] == base.PINNED_PHASES[10], 'The peer guard tail phase must match the pinned plan')


def clock_boundary_state(hold: dict, core_policy_path: Path) -> dict:
    """Validate and record the core+250 -> core0 clock boundary around the reboot."""
    core_policy = base.read_json(core_policy_path)
    require(
        isinstance(core_policy, dict)
        and core_policy.get('schema') == 'gpu-core-policy-change/v1'
        and core_policy.get('requested_core_vf_offset_mhz') == 0
        and core_policy.get('preserved_memory_vf_offset_mhz') == 6000
        and core_policy.get('preserved_power_limit_w') == 600,
        'Core policy change receipt does not pin the core0/memory6000/600W boundary',
    )
    diagnosis_root = core_policy_path.parent.parent
    pre_path = diagnosis_root / 'current-readonly-nvml-settings.json'
    pre = base.read_json(pre_path)
    require(isinstance(pre, dict) and isinstance(pre.get('stdout'), str), 'Pre-change clock readback receipt is incomplete')
    pre_rows = json.loads(pre['stdout'])
    require(isinstance(pre_rows, list) and len(pre_rows) == 4
            and all(isinstance(row, dict) for row in pre_rows),
            'Pre-change clock readback is not a four-GPU record')
    readable_core = {row['core_vf_offset_mhz'] for row in pre_rows if isinstance(row.get('core_vf_offset_mhz'), int)}
    readable_memory = {row['memory_vf_offset_mhz'] for row in pre_rows if isinstance(row.get('memory_vf_offset_mhz'), int)}
    readable_power = {row['power_limit_mw'] for row in pre_rows if isinstance(row.get('power_limit_mw'), int)}
    require(readable_core == {250}, 'Pre-change readback does not document the core VF offset +250 profile')
    require(readable_memory == {6000}, 'Pre-change readback does not document the preserved memory VF offset +6000')
    require(readable_power == {600000}, 'Pre-change readback does not document the preserved 600W power limit')

    post_path = diagnosis_root / 'post-reboot' / 'clock-readback.json'
    post = base.read_json(post_path)
    require(
        isinstance(post, dict)
        and isinstance(post.get('rows'), list)
        and len(post['rows']) == 4
        and post.get('requested_profile_verified') is True
        and post.get('settings_changed') is False,
        'Post-reboot clock readback does not verify the requested profile',
    )
    indices = set()
    for row in post['rows']:
        require(
            isinstance(row, dict)
            and row.get('core_offset') == 0
            and row.get('memory_offset') == 6000
            and row.get('power_limit_mw') == 600000
            and row.get('index') in {0, 1, 2, 3},
            'Post-reboot clock readback row is not core0/memory6000/600W',
        )
        indices.add(row.get('index'))
    require(indices == {0, 1, 2, 3}, 'Post-reboot clock readback does not cover distinct GPUs 0-3')
    cuda_path = diagnosis_root / 'post-reboot' / 'four-gpu-cuda-proof.json'
    cuda = base.read_json(cuda_path)
    require(isinstance(cuda, dict) and cuda.get('returncode') == 0 and isinstance(cuda.get('stdout'), str),
            'Post-reboot CUDA proof did not run cleanly')
    cuda_result = json.loads(cuda['stdout'])
    require(cuda_result.get('cuda_devices') == 4 and cuda_result.get('passed') is True,
            'Post-reboot CUDA proof did not exercise exactly four GPUs')
    return {
        'schema': 'qad-clock-boundary/v1',
        'pre_reboot_profile': {
            'core_vf_offset_mhz': 250,
            'memory_vf_offset_mhz': 6000,
            'power_limit_w': 600,
            'evidence': base.file_identity(pre_path),
            'note': 'GPU3 offset readback was unavailable before the reboot (pending FLR reset); every readable GPU reported core +250.',
        },
        'change_receipt': base.file_identity(core_policy_path),
        'post_reboot_profile': {
            'core_vf_offset_mhz': 0,
            'memory_vf_offset_mhz': 6000,
            'power_limit_w': 600,
            'evidence': base.file_identity(post_path),
            'cuda_proof': base.file_identity(cuda_path),
        },
        'core_vf_offset_boundary': '+250 -> 0',
        'memory_vf_offset_unchanged': True,
        'power_limit_unchanged': True,
        'speed_claim_policy': (
            'No matched-speed claim spans the clock boundary: preserved measurements ran at core VF '
            'offset +250; tail measurements run at core 0 with memory VF offset +6000 and 600W unchanged.'
        ),
    }


def scheduler_plan_state(source_root: Path, hold: dict) -> dict:
    """Cross-check the hold receipt against the recorded scheduler plans and count cells."""
    plan_path = source_root / SCHEDULER_DIR / 'phase-plan.json'
    plan = base.read_json(plan_path)
    require(isinstance(plan, dict) and isinstance(plan.get('plans'), list),
            'Recorded scheduler phase plan is not a plan object')
    plans = plan['plans']
    hold_plans = hold.get('scheduler_plans')
    require(isinstance(hold_plans, list) and len(hold_plans) == len(plans),
            'Hold scheduler plan list size disagrees with the recorded phase plan')
    require([row.get('name') for row in hold_plans] == [row.get('name') for row in plans],
            'Hold scheduler plan order disagrees with the recorded phase plan')

    per_plan = []
    totals = {
        'plans': len(plans),
        'plans_recorded_terminal': 0,
        'plans_partial': 0,
        'plans_not_attempted': 0,
        'cells_expected': 0,
        'cells_recorded': 0,
        'cells_complete': 0,
        'cells_failed_recorded': 0,
        'cells_missing': 0,
    }
    for plan_row, hold_row in zip(plans, hold_plans, strict=True):
        name = plan_row['name']
        expected = plan_row.get('scenario_count')
        require(isinstance(expected, int) and not isinstance(expected, bool) and expected == hold_row.get('scenarios'),
                f'Scenario count disagreement for scheduler plan: {name}')
        disposition = hold_row.get('disposition')
        require(isinstance(disposition, str), f'Hold disposition missing for scheduler plan: {name}')
        receipt_path = source_root / SCHEDULER_DIR / f'{name}.json'
        command_path = source_root / SCHEDULER_DIR / f'{name}.command.json'
        expected_identities = {
            (profile, concurrency, repeat)
            for profile in plan_row['profiles']
            for concurrency in plan_row['concurrencies']
            for repeat in range(1, plan_row['repeats'] + 1)
        }
        require(len(expected_identities) == expected, f'Plan scenario set does not match scenario count: {name}')

        receipt_identity = None
        command_identity = None
        returncode = None
        cells = []
        if receipt_path.exists():
            receipt = base.read_json(receipt_path)
            raw_cells = receipt.get('cells')
            require(isinstance(raw_cells, list), f'Plan receipt cells are not a list: {name}')
            seen = set()
            for cell in raw_cells:
                require(isinstance(cell, dict), f'Plan receipt has a non-object cell: {name}')
                identity = (cell.get('profile'), cell.get('concurrency'), cell.get('repeat'))
                require(identity not in seen, f'Plan receipt records a duplicate cell: {name}')
                seen.add(identity)
                cells.append(cell)
            require(seen <= expected_identities, f'Plan receipt records foreign cells: {name}')
            receipt_identity = base.file_identity(receipt_path)
            recorded_hash = hold_row.get('receipt_sha256')
            require(recorded_hash == receipt_identity['sha256'], f'Hold receipt hash drifted for scheduler plan: {name}')
            require(hold_row.get('receipt') == str(receipt_path), f'Hold receipt path drifted for scheduler plan: {name}')

        if command_path.exists():
            command = base.read_json(command_path)
            require(isinstance(command, dict) and isinstance(command.get('returncode'), int)
                    and not isinstance(command.get('returncode'), bool),
                    f'Scheduler plan command receipt is invalid: {name}')
            returncode = command['returncode']
            command_identity = base.file_identity(command_path)
            require(hold_row.get('command_receipt') == str(command_path),
                    f'Hold command receipt path drifted for scheduler plan: {name}')

        recorded = len(cells)
        complete = sum(1 for cell in cells if cell.get('status') == 'complete')
        if disposition == 'recorded-terminal-plan':
            require(
                receipt_identity is not None and command_identity is not None and recorded == expected,
                f'Terminal scheduler plan is not fully recorded: {name}',
            )
            require(hold_row.get('returncode') == returncode,
                    f'Hold returncode drifted for terminal scheduler plan: {name}')
            totals['plans_recorded_terminal'] += 1
        elif disposition == 'partial-hardware-interrupted':
            require(
                receipt_identity is not None and command_identity is None
                and 0 < recorded < expected and hold_row.get('returncode') is None,
                f'Partial scheduler plan does not match its interrupted state: {name}',
            )
            totals['plans_partial'] += 1
        elif disposition == 'not-attempted':
            require(
                receipt_identity is None and command_identity is None and recorded == 0,
                f'Scheduler plan marked not-attempted has recorded evidence: {name}',
            )
            totals['plans_not_attempted'] += 1
        else:
            raise RuntimeError(f'Unknown hold disposition for scheduler plan {name}: {disposition}')

        if name == PARTIAL_SCHEDULER_PLAN:
            require(disposition == 'partial-hardware-interrupted',
                    f'The partial coverage plan changed disposition: {name}')
            c8_identities = {identity for identity in expected_identities if identity[1] == 8}
            require({(cell.get('profile'), cell.get('concurrency'), cell.get('repeat')) for cell in cells} == c8_identities,
                    f'The partial coverage plan is not exactly the completed c8 cell: {name}')
            require(all(cell.get('status') == 'complete' for cell in cells),
                    f'The completed c8 cell is not complete: {name}')

        totals['cells_expected'] += expected
        totals['cells_recorded'] += recorded
        totals['cells_complete'] += complete
        totals['cells_failed_recorded'] += recorded - complete
        totals['cells_missing'] += expected - recorded
        per_plan.append({
            'name': name,
            'disposition': disposition,
            'expected_cells': expected,
            'recorded_cells': recorded,
            'complete_cells': complete,
            'missing_cells': expected - recorded,
            'returncode': returncode,
            'receipt': receipt_identity,
            'command_receipt': command_identity,
            'recorded_cell_identities': [
                {'profile': cell.get('profile'), 'concurrency': cell.get('concurrency'),
                 'repeat': cell.get('repeat'), 'status': cell.get('status')}
                for cell in cells
            ],
        })
    return {'plan_receipt': base.file_identity(plan_path), 'totals': totals, 'plans': per_plan}
def isolation_state(source_root: Path, hold_epoch: float) -> dict:
    """Validate the preserved GPU isolation stream that bootstraps the new run."""
    path = source_root / 'gpu-isolation-events.jsonl'
    data = path.read_bytes()
    samples = []
    for line in data.splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        require(isinstance(sample, dict) and isinstance(sample.get('timestamp'), (int, float))
                and not isinstance(sample.get('timestamp'), bool),
                'Preserved isolation stream has an invalid sample')
        samples.append(sample)
    require(len(samples) >= 2, 'Preserved isolation stream lacks samples')
    last = samples[-1]
    require(last['timestamp'] == hold_epoch, 'Preserved isolation stream does not end at the hold fault timestamp')
    require(last.get('gpu_health', {}).get('healthy') is False,
            'The final preserved isolation sample does not record the GPU fault')
    return {
        'source': base.file_identity(path),
        'sample_count': len(samples),
        'sha256': hashlib.sha256(data).hexdigest(),
        'final_sample_timestamp': hold_epoch,
        'policy': (
            'The preserved sampler stream is staged at the output root and new coordinator samples append '
            'to it. Fault-era terminal samples are historical records of the interrupted attempt; '
            'consumers qualify each measurement window.'
        ),
    }


def validate_source(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    require(source_root.is_dir() and not source_root.is_symlink(),
            f'Source evidence root is missing or a symlink: {source_root}')
    require(not output_root.exists(), f'Refusing to overwrite output root: {output_root}')
    for inside, container, message in (
        (output_root, source_root, 'Output root must not be inside the immutable source root'),
        (source_root, output_root, 'Source root must not be inside the output root'),
    ):
        try:
            inside.relative_to(container)
        except ValueError:
            pass
        else:
            raise RuntimeError(message)

    expected_plan = base.phase_rows(base.PINNED_PHASES)
    require(base.source_driver_phases() == base.PINNED_PHASES,
            'run_qad_battery.py no longer has the pinned 11-phase plan')
    source_plan_path = source_root / 'phase-plan.json'
    source_plan = base.read_json(source_plan_path)
    require(source_plan == expected_plan, 'Source phase-plan.json is not the pinned original 11-phase plan')

    previous_plan = base.read_json(source_root / 'recovery-plan.json')
    require(
        isinstance(previous_plan, dict)
        and previous_plan.get('schema') == 'r26-qad-recovery-plan/v1'
        and previous_plan.get('output_root') == str(source_root),
        'Source root is not the recorded output of the previous QAD recovery plan',
    )

    hold_path = source_root / 'tail-resume-hold.json'
    hold = base.read_json(hold_path)
    require(isinstance(hold, dict) and hold.get('schema') == 'qad-tail-hold/v1',
            'Source hold receipt is not a qad-tail-hold/v1 object')
    fault = hold.get('second_gpu_fault')
    require(
        isinstance(fault, dict)
        and isinstance(fault.get('timestamp'), (int, float))
        and not isinstance(fault.get('timestamp'), bool),
        'Hold receipt lacks the second GPU fault timestamp',
    )
    hold_epoch = float(fault['timestamp'])
    require(0 < hold_epoch < time.time(), 'Hold fault timestamp is not a past epoch')
    require(hold.get('recorded_phase_count') == EXPECTED_RECORDED_PHASES,
            'Hold receipt does not record exactly nine preserved phases')
    kernel_evidence = hold.get('kernel_evidence')
    require(isinstance(kernel_evidence, list) and kernel_evidence
            and all(isinstance(item, str) for item in kernel_evidence),
            'Hold receipt lacks kernel fault evidence')
    core_policy_receipt = hold.get('core_policy_change_receipt')
    require(isinstance(core_policy_receipt, str) and core_policy_receipt,
            'Hold receipt lacks the core policy change receipt')

    progress_path = source_root / 'phase-progress.json'
    progress = base.read_json(progress_path)
    require(isinstance(progress, list) and len(progress) == EXPECTED_RECORDED_PHASES,
            'Source phase-progress.json is not exactly the nine preserved rows')
    expected_names = [phase[0] for phase in base.PINNED_PHASES[:EXPECTED_RECORDED_PHASES]]
    observed = []
    for row in progress:
        require(
            isinstance(row, dict)
            and isinstance(row.get('phase'), str)
            and isinstance(row.get('returncode'), int)
            and not isinstance(row.get('returncode'), bool),
            'Preserved progress has an invalid invocation row',
        )
        observed.append(row['phase'])
    require(observed == expected_names, 'Preserved progress is not the original nine-phase prefix')
    reused = progress[0]
    require(
        reused.get('execution') == 'reused-successful-pre-fault-invocation' and reused.get('returncode') == 0,
        'The first preserved row is not the reused pre-fault matched invocation',
    )
    for row in progress[1:]:
        expected_code = MEASURED_FAILURE_ROWS.get(row['phase'], 0)
        require(row['returncode'] == expected_code,
                f'Preserved phase returncode changed: {row["phase"]}')
    require(hold.get('completed_phases') == progress, 'Hold receipt phases disagree with phase-progress.json')
    require(hold.get('remaining_phases') == base.phase_rows(base.PINNED_PHASES[EXPECTED_RECORDED_PHASES:]),
            'Hold receipt remaining phases disagree with the pinned plan')

    phase_receipts = {}
    for row, (name, script, args, _timeout) in zip(progress, base.PINNED_PHASES[:EXPECTED_RECORDED_PHASES], strict=True):
        receipt_path = source_root / f'phase-{name}.command.json'
        command = base.read_json(receipt_path)
        argv = command.get('args') if isinstance(command, dict) else None
        valid = (
            isinstance(command, dict)
            and isinstance(command.get('returncode'), int)
            and not isinstance(command.get('returncode'), bool)
            and command.get('returncode') == row['returncode']
            and isinstance(argv, list)
            and len(argv) >= 2
        )
        if valid:
            valid = (
                Path(argv[1]).resolve() == HERE / script
                and argv[2:] == args
                and isinstance(command.get('started_at'), (int, float))
                and not isinstance(command.get('started_at'), bool)
                and isinstance(command.get('finished_at'), (int, float))
                and not isinstance(command.get('finished_at'), bool)
                and command['started_at'] <= command['finished_at'] <= hold_epoch
            )
            if row is not reused:
                valid = valid and command.get('log') == str(source_root / f'phase-{name}.log')
        require(valid, f'Preserved phase invocation receipt does not back the recorded row: {name}')
        require((source_root / f'phase-{name}.log').is_file(), f'Preserved phase log is missing: {name}')
        phase_receipts[name] = base.file_identity(receipt_path)
    require(progress[0].get('invocation_receipt_sha256') == phase_receipts[base.FIRST_PHASE]['sha256'],
            'The reused matched invocation receipt hash drifted')

    interrupted = base.read_json(source_root / 'qualification-interrupted.json')
    require(
        isinstance(interrupted, dict)
        and isinstance(interrupted.get('timestamp'), (int, float))
        and not isinstance(interrupted.get('timestamp'), bool)
        and interrupted['timestamp'] >= hold_epoch,
        'Interruption receipt predates the hold fault timestamp',
    )
    require(interrupted.get('completed_phases') == progress[1:],
            'Interruption receipt disagrees with the preserved progress rows')
    current_phase = base.read_json(source_root / 'current-phase.json')
    require(
        isinstance(current_phase, dict)
        and current_phase.get('phase') == 'mixed-agent-scheduling'
        and current_phase.get('script') == 'scheduler_recheck_phase.py'
        and isinstance(current_phase.get('started_at'), (int, float))
        and not isinstance(current_phase.get('started_at'), bool)
        and current_phase['started_at'] < hold_epoch,
        'The interrupted attempt was not inside the scheduler phase before the hold fault',
    )

    checkpoint_path = source_root / 'checkpoint-verification.json'
    checkpoint = base.read_json(checkpoint_path)
    require(
        isinstance(checkpoint, dict)
        and checkpoint.get('model_dir') == str(base.PINNED_MODEL)
        and checkpoint.get('revision') == base.PINNED_REVISION
        and checkpoint.get('complete') is True,
        'Source checkpoint receipt does not verify the pinned QAD step2500 checkpoint',
    )
    current_checkpoint = base.checkpoint_complete(base.PINNED_MODEL, base.PINNED_REVISION)
    require(current_checkpoint.get('complete') is True,
            f'Pinned checkpoint is no longer complete: {current_checkpoint}')

    template_receipt = base.read_json(source_root / 'checkpoint-template-identity.json')
    require(isinstance(template_receipt, dict) and template_receipt.get('schema') == 'checkpoint-template-identity/v1',
            'Preserved template identity receipt is not a checkpoint-template-identity/v1 object')
    template_identity = {
        'schema': 'qad-tail-template-identity/v1',
        'verified_at': time.time(),
        'files': {},
        'scope': 'Fresh verification of the served config, tokenizer, generation and standalone template '
                 'artifacts against the published checkpoint; the preserved receipt must still match.',
    }
    for filename in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja'):
        published = base.file_identity(base.PUBLISHED_MODEL / filename)
        candidate = base.file_identity(base.PINNED_MODEL / filename)
        require(published['sha256'] == candidate['sha256'],
                f'Checkpoint config/template artifact differs: {filename}')
        recorded = template_receipt.get('files', {}).get(filename)
        if recorded is not None:
            require(isinstance(recorded, dict) and isinstance(recorded.get('candidate'), dict)
                    and recorded['candidate'].get('sha256') == candidate['sha256'],
                    f'Preserved template receipt disagrees with the fresh candidate hash: {filename}')
        template_identity['files'][filename] = {'published': published, 'candidate': candidate, 'identical': True}

    clock_boundary = clock_boundary_state(hold, Path(core_policy_receipt))
    scheduler_state = scheduler_plan_state(source_root, hold)
    isolation = isolation_state(source_root, hold_epoch)

    source_files: dict[str, dict] = {}
    source_dirs: list[str] = []
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(source_root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for entry in dirnames:
            if (Path(dirpath) / entry).is_symlink():
                raise RuntimeError(f'Source tree contains a symlinked directory: {Path(dirpath) / entry}')
        rel_dir = Path(dirpath).relative_to(source_root)
        if rel_dir != Path('.'):
            source_dirs.append(str(rel_dir))
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_symlink():
                raise RuntimeError(f'Source tree contains a symlinked file: {path}')
            require(path.is_file(), f'Source tree contains a non-regular file: {path}')
            require(path.suffix.lower() not in WEIGHT_SUFFIXES,
                    f'Refusing to stage model weight artifact from the evidence root: {path}')
            rel = str(path.relative_to(source_root))
            identity = base.file_identity(path)
            source_files[rel] = identity
            total_bytes += identity['size_bytes']
    archived_identities = {}
    for name in ARCHIVED_ARTIFACTS:
        require(name in source_files, f'Mutable operational boundary artifact is missing: {name}')
        archived_identities[name] = source_files[name]

    drivers = {
        'tail_driver': base.file_identity(Path(__file__)),
        'environment_contract': base.file_identity(HERE / 'continue_qad_battery.py'),
        'original_qad_driver': base.file_identity(base.ORIGINAL_DRIVER),
        'coordinator': base.file_identity(base.COORDINATOR),
        'scheduler_recheck_phase': base.file_identity(HERE / 'scheduler_recheck_phase.py'),
        'agent_workload_recheck': base.file_identity(HERE / 'agent_workload_recheck.py'),
        'peer_phase': base.file_identity(HERE / 'peer_phase.py'),
    }
    append_streams = {name: base.file_identity(source_root / name) for name in APPEND_ONLY_STREAMS}
    return {
        'source_root': source_root,
        'output_root': output_root,
        'hold': hold,
        'hold_identity': base.file_identity(hold_path),
        'hold_epoch': hold_epoch,
        'kernel_evidence': kernel_evidence,
        'previous_plan': previous_plan,
        'previous_plan_identity': base.file_identity(source_root / 'recovery-plan.json'),
        'source_plan_bytes': source_plan_path.read_bytes(),
        'source_plan_identity': base.file_identity(source_plan_path),
        'preserved_progress_rows': progress,
        'progress_identity': base.file_identity(progress_path),
        'phase_receipts': phase_receipts,
        'checkpoint': checkpoint,
        'current_checkpoint': current_checkpoint,
        'template_identity': template_identity,
        'clock_boundary': clock_boundary,
        'scheduler_state': scheduler_state,
        'isolation': isolation,
        'append_streams': append_streams,
        'drivers': drivers,
        'source_identities': source_files,
        'source_dirs': source_dirs,
        'archived_identities': archived_identities,
        'file_count': len(source_files),
        'total_bytes': total_bytes,
    }


def plan_receipt(state: dict, copied: dict[str, dict] | None = None) -> dict:
    manifest = copied if copied is not None else base.compact_hashes(state['source_identities'])
    archived = {name: manifest[name] for name in ARCHIVED_ARTIFACTS}
    return {
        'schema': 'r26-qad-tail-recovery-plan/v1',
        'created_at': time.time(),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(state['source_root']),
        'output_root': str(state['output_root']),
        'hold': {
            'receipt': state['hold_identity'],
            'schema': state['hold']['schema'],
            'fault_epoch': state['hold_epoch'],
            'gpu_health': state['hold'].get('second_gpu_fault', {}).get('gpu_health'),
            'fault_result': state['hold'].get('second_gpu_fault', {}).get('result'),
            'kernel_evidence': state['kernel_evidence'],
            'scope': state['hold'].get('scope'),
            'production_restored': state['hold'].get('production_restored'),
            'host_reboot_authorized': state['hold'].get('host_reboot_authorized'),
        },
        'interruption_chain': {
            'first_fault_timestamp': state['previous_plan'].get('fault_timestamp'),
            'first_recovery_plan': state['previous_plan_identity'],
            'second_fault_epoch': state['hold_epoch'],
            'second_fault_kernel_evidence': state['kernel_evidence'],
        },
        'pinned': {
            'checkpoint': str(base.PINNED_MODEL),
            'checkpoint_revision': base.PINNED_REVISION,
            'image': base.PINNED_IMAGE,
            'model_cache_tag': base.MODEL_TAG,
            'l2_namespace': str(base.L2_SHARED),
        },
        'clock_boundary': state['clock_boundary'],
        'phase_plan': {
            'phase_count': len(base.PINNED_PHASES),
            'source_receipt': state['source_plan_identity'],
            'phases': base.phase_rows(base.PINNED_PHASES),
        },
        'preserved_phases': {
            'count': EXPECTED_RECORDED_PHASES,
            'rows': state['preserved_progress_rows'],
            'progress_receipt': state['progress_identity'],
            'phase_invocation_receipts': state['phase_receipts'],
            'measured_failure_policy': (
                'Recorded failures stay failures: the agentic-prefix-cache-reuse returncode 1 row is '
                'preserved verbatim and never rerun or requalified.'
            ),
            'execution_window': (
                'All nine preserved phases completed before the second GPU fault, at core VF offset +250 '
                'with memory VF offset +6000.'
            ),
        },
        'tail_phases': [
            {
                'name': name,
                'script': script,
                'planned_args': base.PINNED_PHASES[EXPECTED_RECORDED_PHASES + index][2],
                'actual_args': args,
                'timeout_seconds': timeout,
                'adjustment': (
                    'The tail recovery adds --resume so the interrupted scheduler plan is resumed from the '
                    'copied evidence; the invocation receipt keeps the actual argv.'
                    if args else None
                ),
            }
            for index, (name, script, args, timeout) in enumerate(TAIL_PHASES)
        ],
        'scheduler_state': state['scheduler_state'],
        'stages': [
            {
                'stage': 1,
                'disposition': 'nine-preserved-phases-staged-verbatim',
                'phase_count': EXPECTED_RECORDED_PHASES,
                'copy_verified': copied is not None,
            },
            {
                'stage': 2,
                'disposition': 'fresh-post-reboot-execution-through-run_qualification',
                'phases': [name for name, _script, _args, _timeout in TAIL_PHASES],
                'scheduler_invocation': (
                    'scheduler_recheck_phase.py --resume reuses the complete saved plan outputs, resumes '
                    'official-static07-coverage from the copied c8 cell, and boots only groups with work remaining.'
                ),
            },
            {
                'stage': 3,
                'disposition': 'merge-nine-preserved-and-two-fresh-rows-and-restore-original-phase-plan',
                'all_phases_attempted_policy': (
                    'written only after both tail phases have actual invocation receipts'
                ),
            },
        ],
        'copy_manifest': {
            'policy': (
                'Recursive shutil.copy2 of the immutable source evidence; every file re-hashed after copy; '
                'paths and mtimes preserved; symlinks and model weights refused; no hard links.'
            ),
            'file_count': state['file_count'],
            'total_bytes': state['total_bytes'],
            'copy_verified': copied is not None,
            'identities': manifest,
        },
        'archived_artifacts': {
            'directory': ARCHIVE_DIR,
            'names': list(ARCHIVED_ARTIFACTS),
            'policy': (
                'Mutable operational-boundary receipts of the interrupted attempt are archived instead of '
                'staged at the output root, so copied old interruption receipts cannot supersede the new '
                'complete run. The immutable source root retains every original byte.'
            ),
            'identities': archived,
        },
        'append_only_streams': {
            'gpu-isolation-events.jsonl': {**state['isolation'], 'bootstrap': 'staged-by-copy'},
            'battery.log': {
                'identity': state['append_streams']['battery.log'],
                'policy': 'Preserved run log; new coordinator notes append chronologically.',
            },
            'gates.jsonl': {
                'identity': state['append_streams']['gates.jsonl'],
                'policy': 'Preserved gate stream; new coordinator gates append.',
            },
        },
        'drivers': state['drivers'],
        'environment': base.battery_environment(state['output_root']),
        'checkpoint_revalidation': state['current_checkpoint'],
        'template_identity': state['template_identity'],
        'source_immutability': {
            'policy': 'The source root is never written; provenance is recorded by identity and the raw '
                      'embedded paths of preserved receipts are kept untouched.',
            'raw_embedded_paths_rewritten': False,
        },
        'gpu_jobs_started_by_planning': False,
        'services_touched_by_planning': False,
    }


def copy_evidence(state: dict) -> dict[str, dict]:
    output_root = state['output_root']
    archive_root = output_root / ARCHIVE_DIR
    archive_root.mkdir(parents=True, exist_ok=False)
    for rel in state['source_dirs']:
        (output_root / rel).mkdir(parents=True, exist_ok=True)
    copied = {}
    for rel in sorted(state['source_identities']):
        source = state['source_root'] / rel
        if rel in ARCHIVED_ARTIFACTS:
            destination = archive_root / rel
        else:
            destination = output_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
        identity = base.file_identity(destination)
        source_identity = state['source_identities'][rel]
        require(
            identity['sha256'] == source_identity['sha256']
            and identity['size_bytes'] == source_identity['size_bytes'],
            f'Staged evidence changed during relocation: {rel}',
        )
        copied[rel] = identity
    for rel in sorted(state['source_dirs'], key=lambda item: (len(item.split('/')), item), reverse=True):
        shutil.copystat(state['source_root'] / rel, output_root / rel)
    return copied


def fresh_progress_rows(state: dict) -> list[dict]:
    path = state['output_root'] / 'phase-progress.json'
    if not path.exists():
        return []
    rows = base.read_json(path)
    require(isinstance(rows, list), 'Tail phase-progress.json is not a list')
    if rows == state['preserved_progress_rows']:
        return []
    expected = [name for name, _script, _args, _timeout in TAIL_PHASES]
    observed = []
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row.keys()) == {'phase', 'returncode'}
            and isinstance(row['phase'], str)
            and isinstance(row['returncode'], int)
            and not isinstance(row['returncode'], bool),
            'Fresh tail progress has an invalid invocation row',
        )
        observed.append(row['phase'])
    require(observed == expected[:len(observed)], 'Fresh tail progress is not a tail-phase prefix')
    return rows


def validate_tail_execution(state: dict) -> tuple[dict, list[dict]]:
    execution_path = state['output_root'] / 'qualification-executed.json'
    execution = base.read_json(execution_path)
    require(isinstance(execution, dict), 'Coordinator execution receipt is not an object')
    rows = execution.get('phases')
    require(
        execution.get('all_phases_attempted') is True and isinstance(rows, list),
        'Coordinator did not record both tail phase attempts',
    )
    require(
        [row.get('phase') for row in rows if isinstance(row, dict)] == [name for name, _s, _a, _t in TAIL_PHASES],
        'Coordinator execution ledger is not the two-phase tail',
    )
    for row, (name, script, args, _timeout) in zip(rows, TAIL_PHASES, strict=True):
        code = row.get('returncode') if isinstance(row, dict) else None
        require(isinstance(code, int) and not isinstance(code, bool), f'Invalid return code for {name}')
        command_path = state['output_root'] / f'phase-{name}.command.json'
        command = base.read_json(command_path)
        argv = command.get('args') if isinstance(command, dict) else None
        require(
            isinstance(command, dict)
            and command.get('returncode') == code
            and isinstance(argv, list)
            and len(argv) >= 2
            and Path(argv[1]).resolve() == HERE / script
            and argv[2:] == args
            and isinstance(command.get('started_at'), (int, float))
            and not isinstance(command.get('started_at'), bool)
            and isinstance(command.get('finished_at'), (int, float))
            and not isinstance(command.get('finished_at'), bool),
            f'Actual invocation receipt does not match the recorded phase: {name}',
        )
    return execution, rows


def write_tail_progress(state: dict, coordinator_returned: bool) -> None:
    fresh = fresh_progress_rows(state)
    interrupted_path = state['output_root'] / 'qualification-interrupted.json'
    base.write_json_new(state['output_root'] / 'tail-progress.json', {
        'schema': 'qad-tail-progress/v1',
        'timestamp': time.time(),
        'coordinator_returned': coordinator_returned,
        'preserved_phase_count': EXPECTED_RECORDED_PHASES,
        'fresh_phase_count': len(fresh),
        'phases': [*state['preserved_progress_rows'], *fresh],
        'all_phases_attempted': False,
        'policy': (
            'The full eleven-phase ledger is written only after both tail phases have actual invocation '
            'receipts. Preserved rows are pre-reboot core+250 measurements; fresh rows are post-reboot '
            'core0 measurements.'
        ),
        'source_root': str(state['source_root']),
        'output_root': str(state['output_root']),
        'tail_recovery_plan': base.file_identity(state['output_root'] / 'tail-recovery-plan.json'),
        'coordinator_interrupted_receipt': (
            base.file_identity(interrupted_path) if interrupted_path.exists() else None
        ),
    })


def merge_execution(state: dict, coordinator_returned: bool) -> list[dict] | None:
    execution_path = state['output_root'] / 'qualification-executed.json'
    if not execution_path.exists():
        require(not coordinator_returned, 'Coordinator returned without an execution receipt')
        write_tail_progress(state, coordinator_returned)
        return None
    execution, rows = validate_tail_execution(state)
    raw_execution = execution_path.read_bytes()
    remaining_path = state['output_root'] / 'tail-coordinator-executed.json'
    if not remaining_path.exists():
        with remaining_path.open('xb') as handle:
            handle.write(raw_execution)
    remaining_identity = base.file_identity(remaining_path)
    plan_identity = base.file_identity(state['output_root'] / 'tail-recovery-plan.json')
    merged = {
        **execution,
        'all_phases_attempted': True,
        'phases': [*state['preserved_progress_rows'], *rows],
        'tail_recovery': {
            'schema': 'r26-qad-tail-recovery-execution/v1',
            'coordinator_returned': coordinator_returned,
            'preserved_phase_count': EXPECTED_RECORDED_PHASES,
            'fresh_phase_count': len(TAIL_PHASES),
            'coordinator_execution': remaining_identity,
            'tail_recovery_plan': plan_identity,
            'source_root': str(state['source_root']),
            'raw_embedded_paths_rewritten': False,
            'invocation_adjustment': [{
                'phase': TAIL_PHASES[0][0],
                'planned_args': base.PINNED_PHASES[EXPECTED_RECORDED_PHASES][2],
                'actual_args': TAIL_PHASES[0][2],
                'reason': (
                    'The tail recovery resumes the interrupted scheduler plan; the --resume flag stays in '
                    'the preserved invocation receipt instead of pretending the planned CLI was unchanged.'
                ),
            }],
            'clock_boundary': state['clock_boundary'],
            'provenance_policy': (
                'The first nine merged rows are preserved pre-reboot measurements taken at core VF offset '
                '+250; the last two rows are post-reboot core0 measurements. No matched-speed claim spans '
                'the clock boundary.'
            ),
        },
    }
    base.atomic_write_json(execution_path, merged)
    return rows


def restore_plan_and_progress(state: dict, fresh_rows: list[dict] | None = None) -> None:
    base.atomic_write(state['output_root'] / 'phase-plan.json', state['source_plan_bytes'])
    if fresh_rows is None:
        fresh_rows = fresh_progress_rows(state)
    else:
        require(fresh_progress_rows(state) == fresh_rows,
                'Coordinator progress file disagrees with the execution ledger')
    base.atomic_write_json(state['output_root'] / 'phase-progress.json',
                           [*state['preserved_progress_rows'], *fresh_rows])


def execute(state: dict) -> None:
    output_root = state['output_root']
    output_root.mkdir(parents=True, exist_ok=False)
    copied = copy_evidence(state)
    base.write_json_new(output_root / 'tail-recovery-plan.json', plan_receipt(state, copied))
    base.atomic_write(output_root / 'phase-plan.json', state['source_plan_bytes'])
    base.write_json_new(output_root / 'phase-progress.json', state['preserved_progress_rows'])

    require('runtime' not in sys.modules and 'run_qualification' not in sys.modules,
            'Runtime/coordinator was imported before the pinned recovery environment was installed')
    os.environ.update(base.battery_environment(output_root))
    coordinator = importlib.import_module('run_qualification')
    coordinator.PHASES = list(TAIL_PHASES)
    executed_rows: list[dict] | None = None
    try:
        coordinator.main()
    except BaseException as error:
        try:
            merge_execution(state, coordinator_returned=False)
        except Exception as merge_error:
            error.add_note(f'Could not merge a post-attempt tail ledger: {merge_error}')
        raise
    else:
        executed_rows = merge_execution(state, coordinator_returned=True)
    finally:
        restore_plan_and_progress(state, fresh_rows=executed_rows)

    print('QAD TAIL RECOVERY COMPLETE', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()

    state = validate_source(args.source_root, args.output_root)
    if args.plan_only:
        receipt = plan_receipt(state)
        receipt.update({
            'plan_only': True,
            'filesystem_mutation_performed': False,
            'gpu_jobs_started': False,
            'services_touched': False,
        })
        print(json.dumps(receipt, indent=2))
        return
    execute(state)


if __name__ == '__main__':
    main()
