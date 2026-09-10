"""Focused regressions for R29 cache-campaign qualification gates."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cache_campaign as campaign
from cache_lifecycle_probe import LifecycleProbe, ProbeAbort, inventory_l2


def ram_records():
    return [
        {'label': 'ram-stock-mtp3-dcp4', 'tag': 'stock', 'passed': True},
        {'label': 'ram-pr64-mtp3-dcp4', 'tag': 'pr64', 'passed': True},
    ]


def candidate_records():
    records = []
    for dcp, spec in campaign.REQUIRED_CANDIDATE_ARMS:
        record = {
            'label': f'l2-pr64-dcp{dcp}-{spec}',
            'tag': 'pr64',
            'dcp': dcp,
            'spec': spec,
            'returncode': 0,
            'complete': True,
            'receipt_passed': True,
        }
        if (dcp, spec) == campaign.MIXED_RESTORE_ARM:
            record['mixed_restore'] = {
                'returncode': 0,
                'complete': True,
                'passed': True,
            }
        record['passed'] = campaign.lifecycle_arm_passed(record)
        records.append(record)
    return records


def stock_control(status='fail', *, complete=True, include_gate=True):
    gates = []
    if include_gate:
        gates.append({
            'name': campaign.EXACT_PAYLOAD_DEDUP_GATE,
            'status': status,
            'passed': status == 'pass' if status in ('pass', 'fail') else None,
            'required': True,
        })
    return {
        'label': 'l2-stock-dcp4-mtp3',
        'tag': 'stock',
        'dcp': 4,
        'spec': 'mtp3',
        'returncode': 0 if status == 'pass' else 1 if complete else 2,
        'complete': complete,
        'receipt_passed': status == 'pass' if complete else None,
        'fatal_error': None if complete else 'aborted fixture',
        'exact_payload_dedup_gates': gates,
        'failed_gates': [campaign.EXACT_PAYLOAD_DEDUP_GATE] if status == 'fail' else [],
        'unavailable_gates': [campaign.EXACT_PAYLOAD_DEDUP_GATE] if status == 'unavailable' else [],
    }


def serial_cache_request(hits):
    before = {
        "vllm:external_prefix_cache_hits_total": 0.0,
        "vllm:external_prefix_cache_queries_total": 0.0,
        "vllm:external_prefix_cache_hits_created": 1000.0,
        "vllm:request_success_total": 0.0,
        "vllm:num_requests_running": 0.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:num_preemptions_total": 0.0,
    }
    after = dict(before)
    after.update({
        "vllm:external_prefix_cache_hits_total": float(hits),
        "vllm:external_prefix_cache_queries_total": 32768.0,
        "vllm:request_success_total": 1.0,
    })
    return {
        "passed": True,
        "summary": {"prompt_tokens": 32768, "cache_stats": None},
        "metrics": {
            "before": {"http": {"ok": True}, "parsed": {"totals": before}},
            "after": {"http": {"ok": True}, "parsed": {"totals": after}},
        },
    }


def idle_cache_status():
    healthy = {'is_healthy': True}
    return {
        'is_healthy': True,
        'storage_manager': {
            'store_controller': {
                **healthy, 'pending_keys_count': 0, 'in_flight_task_count': 0,
            },
            'prefetch_controller': {
                **healthy, 'submission_queue_size': 0, 'pending_queue_size': 0,
                'in_flight_request_count': 0,
            },
            'l1_manager': {
                **healthy, 'write_locked_count': 0, 'read_locked_count': 0,
            },
        },
        'recurrent_checkpoints': {
            **healthy, 'pending_generations': 0, 'store_leases': 0,
            'retrieve_leases': 0,
        },
    }


class VanishingScanEntry:
    """DirEntry stand-in that is listed, then raises ENOENT on stat like a
    temp file renamed away between listing and stat."""

    def __init__(self, directory, name):
        self.path = str(Path(directory) / name)
        self.name = name

    def is_symlink(self):
        return False

    def is_dir(self, follow_symlinks=False):
        return False

    def is_file(self, follow_symlinks=False):
        return True

    def stat(self, follow_symlinks=False):
        raise FileNotFoundError(2, 'No such file or directory', self.path)


def vanishing_scandir(target, name, vanish_passes=None):
    """os.scandir stand-in: `name` is listed on the passes in `vanish_passes`
    (None = every pass) but raises ENOENT when stat'ed."""
    real_scandir = os.scandir
    passes = {'count': 0}

    def fake(path):
        if Path(path) == Path(target):
            passes['count'] += 1
            entries = list(real_scandir(path))
            if vanish_passes is None or passes['count'] in vanish_passes:
                entries.append(VanishingScanEntry(path, name))
            return iter(entries)
        return real_scandir(path)

    return fake


class CacheCampaignGateTests(unittest.TestCase):
    def test_capacity_limited_ram_cell_is_not_valid_timing(self):
        rows = [
            {
                'concurrency': concurrency,
                'context_tokens': context,
                'aggregate_tps': 1.0,
                'num_errors': 0,
                'underfilled': False,
                'warmup_timed_out': False,
                'capacity_limited': False,
                'measurement_seconds': 30,
            }
            for concurrency in (1, 8)
            for context in (0, 32768)
        ]
        self.assertTrue(campaign.ram_trial_valid(rows))
        rows[-1]['capacity_limited'] = True
        self.assertFalse(campaign.ram_trial_valid(rows))

    def test_mixed_restore_requires_outcome_completeness_and_command_success(self):
        baseline = next(
            row for row in candidate_records()
            if (row['dcp'], row['spec']) == campaign.MIXED_RESTORE_ARM
        )
        self.assertTrue(campaign.lifecycle_arm_passed(baseline))
        for field, value in (('passed', False), ('complete', False), ('returncode', 1)):
            with self.subTest(field=field):
                record = deepcopy(baseline)
                record['mixed_restore'][field] = value
                self.assertFalse(campaign.lifecycle_arm_passed(record))

    def test_failed_mixed_restore_cannot_leave_candidate_or_campaign_passed(self):
        candidates = candidate_records()
        mixed = next(
            row for row in candidates
            if (row['dcp'], row['spec']) == campaign.MIXED_RESTORE_ARM
        )
        mixed['mixed_restore']['passed'] = False
        mixed['passed'] = True  # Reproduce the stale lifecycle-only result from the old campaign.
        summary = campaign.build_campaign_summary(
            ram_records(), [stock_control(), *candidates]
        )
        self.assertFalse(summary['candidate_passed'])
        self.assertFalse(summary['passed'])

    def test_missing_candidate_arms_do_not_pass_vacuously(self):
        summary = campaign.build_campaign_summary(ram_records(), [stock_control()])
        self.assertFalse(summary['candidate_coverage']['complete'])
        self.assertEqual(
            len(summary['candidate_coverage']['missing_arms']),
            len(campaign.REQUIRED_CANDIDATE_ARMS),
        )
        self.assertFalse(summary['candidate_passed'])
        self.assertFalse(summary['passed'])

    def test_r30_candidate_tag_is_honoured(self):
        records = [
            dict(row, tag='r30', label=row['label'].replace('pr64', 'r30'))
            for row in candidate_records()
        ]
        coverage = campaign.candidate_coverage(records, candidate_tag='r30')
        self.assertTrue(coverage['complete'])
        summary = campaign.build_campaign_summary(
            ram_records(), [stock_control(), *records], candidate_tag='r30'
        )
        self.assertTrue(summary['candidate_coverage']['complete'])
        # The default pr64 filter must not silently adopt r30 rows.
        self.assertFalse(campaign.candidate_coverage(records)['complete'])


    def test_vanished_temporary_l2_scan_tolerates_in_flight_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            l2_dir = Path(directory)
            (l2_dir / 'fixture.txt').write_bytes(b'payload-fixture')
            probe = LifecycleProbe.__new__(LifecycleProbe)
            probe.deadline = time.monotonic() + 60
            probe.l2_dir = l2_dir
            probe.inventory_dir = l2_dir / 'inventories'
            probe.raw_dir = l2_dir / 'raw'
            probe.cache_status = lambda label, capture=True: {
                'status': idle_cache_status()
            }

            # A .lmcache-write.tmp.* entry listed on the first scan pass and on
            # the first settle poll vanishes before stat; the scan must record
            # it, re-scan to a clean pass, and settle without aborting.
            temp_fake = vanishing_scandir(
                l2_dir, '.lmcache-write.tmp.527.794', vanish_passes={1, 3}
            )
            with patch('os.scandir', temp_fake):
                snapshot = inventory_l2(l2_dir)
            self.assertEqual(snapshot['errors'], [])
            self.assertEqual(
                snapshot['vanished_temporary'], ['.lmcache-write.tmp.527.794']
            )
            self.assertEqual(snapshot['inventory_passes'], 2)
            self.assertEqual(
                [row['relative_path'] for row in snapshot['files']],
                ['fixture.txt'],
            )
            with patch('os.scandir', temp_fake):
                settled = probe.settle('growth-temporary-vanished')
            self.assertTrue(settled['settled'])

            # A vanished non-temporary payload stays an unreadable inventory
            # and still aborts the settle loop.
            payload_fake = vanishing_scandir(
                l2_dir,
                'local-inference-lab-SEP-GLM-5.3-Flash-NVFP4@00000000@0@cafe.data',
            )
            with patch('os.scandir', payload_fake):
                snapshot = inventory_l2(l2_dir)
            self.assertEqual(snapshot['vanished_temporary'], [])
            self.assertEqual(len(snapshot['errors']), 1)
            self.assertIn('@cafe.data', snapshot['errors'][0])
            with patch('os.scandir', payload_fake):
                with self.assertRaises(ProbeAbort) as caught:
                    probe.settle('growth-payload-vanished')
            self.assertIn('unsafe or unreadable L2 inventory', str(caught.exception))
            self.assertIn('@cafe.data', str(caught.exception))

    def test_isolated_external_counts_preserve_integer_hit_contract(self):
        request = serial_cache_request(4096)
        self.assertEqual(LifecycleProbe.request_cache_hits(request), 4096)
        self.assertEqual(
            request["summary"]["cache_hit_observation"]["source"],
            "isolated_request.external_prefix_cache_hits_total",
        )
        cold = serial_cache_request(0)
        self.assertEqual(LifecycleProbe.request_cache_hits(cold), 0)
        direct = serial_cache_request(4096)
        direct["summary"]["cache_stats"] = {"num_lmcache_cached_tokens": 8192}
        self.assertEqual(LifecycleProbe.request_cache_hits(direct), 8192)

    def test_unattributable_external_counters_remain_unavailable(self):
        changes = (
            ("vllm:external_prefix_cache_hits_total", None),
            ("vllm:external_prefix_cache_hits_total", -1),
            ("vllm:external_prefix_cache_hits_total", 1.5),
            ("vllm:external_prefix_cache_hits_created", 2000),
            ("vllm:request_success_total", 2),
            ("vllm:num_requests_running", 1),
            ("vllm:num_requests_waiting", 1),
            ("vllm:num_preemptions_total", 1),
            ("vllm:external_prefix_cache_queries_total", 0),
        )
        for name, value in changes:
            with self.subTest(metric=name, value=value):
                request = serial_cache_request(4096)
                request["metrics"]["after"]["parsed"]["totals"][name] = value
                self.assertIsNone(LifecycleProbe.request_cache_hits(request))
        request = serial_cache_request(4096)
        request["summary"]["cache_stats"] = {"num_lmcache_cached_tokens": True}
        self.assertIsNone(LifecycleProbe.request_cache_hits(request))

    def test_stock_control_requires_completed_specific_gate_failure(self):
        expected = campaign.assess_stock_negative_control([stock_control()])
        self.assertTrue(expected['valid'])
        self.assertEqual(expected['classification'], 'expected_failure_observed')

        cases = {
            'missing arm': [],
            'missing gate': [stock_control(include_gate=False)],
            'unavailable gate': [stock_control('unavailable')],
            'aborted probe': [stock_control(complete=False)],
            'inverted gate': [stock_control('pass')],
        }
        for name, records in cases.items():
            with self.subTest(name=name):
                assessment = campaign.assess_stock_negative_control(records)
                self.assertFalse(assessment['valid'])
        inverted = campaign.assess_stock_negative_control(cases['inverted gate'])
        self.assertTrue(inverted['inverted'])
        self.assertEqual(inverted['classification'], 'inverted')

    def test_inverted_control_is_saved_and_drives_phase_exit(self):
        saved = {}
        disks = [stock_control('pass'), *candidate_records()]

        def save_json(name, value):
            saved[name] = value

        with (
            patch.object(campaign, 'artifacts', return_value={'runtime_image': 'sha256:candidate'}),
            patch.object(campaign, 'gpu_components'),
            patch.object(campaign, 'ram_performance', return_value=ram_records()),
            patch.object(campaign, 'lifecycle', return_value=disks),
            patch.object(campaign.rt, 'save_json', side_effect=save_json),
            patch.object(campaign.rt, 'record_gate'),
        ):
            with self.assertRaises(SystemExit) as caught:
                campaign.phase()

        summary = saved['cache-campaign-summary.json']
        self.assertEqual(caught.exception.code, 1)
        self.assertFalse(summary['passed'])
        self.assertTrue(summary['stock_negative_control']['inverted'])
        self.assertEqual(
            summary['stock_negative_control']['classification'], 'inverted'
        )


if __name__ == '__main__':
    unittest.main()
