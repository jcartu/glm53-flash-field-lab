"""Safety boundaries for GPU reservation across Docker worker restarts."""
import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_qualification as guard

HEALTHY_GPU_OUTPUT = """0, GPU-0, None
1, GPU-1, None
2, GPU-2, None
3, GPU-3, None
"""


class GPUIsolationTests(unittest.TestCase):
    def monitor(self, samples, gpu_observations=None, mode='strict'):
        if gpu_observations is None:
            gpu_observations = [HEALTHY_GPU_OUTPUT] * len(samples)
        query_results = [
            observation if isinstance(observation, BaseException)
            else SimpleNamespace(stdout=observation, returncode=0)
            for observation in gpu_observations
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = SimpleNamespace(ROOT=root, save_json=Mock(), note=Mock())
            stopped = Mock()
            stopped.wait.side_effect = [False] * len(samples) + [True]
            with patch.object(guard, 'rt', runtime), patch.object(guard, 'ISOLATION_MODE', mode), \
                    patch.object(guard, 'foreign_gpu_processes', side_effect=samples), \
                    patch.object(guard.subprocess, 'run', side_effect=query_results), \
                    patch.object(guard.os, 'kill') as interrupt:
                guard.watch_gpu_isolation(stopped)
            events = [json.loads(line) for line in (root/'gpu-isolation-events.jsonl').read_text().splitlines()]
            return interrupt, events, runtime.save_json, runtime.note

    def test_persistent_foreign_work_interrupts_and_records(self):
        foreign = [{'pid': 424242, 'name': 'external-worker', 'created_at': 10.0}]
        interrupt, events, receipts, _ = self.monitor([foreign, foreign])
        interrupt.assert_called_once_with(guard.os.getpid(), signal.SIGINT)
        self.assertEqual(events[-1]['foreign'], foreign)
        self.assertEqual(receipts.call_args.args[0], 'gpu-isolation-interruption.json')

    def test_one_restart_transient_does_not_interrupt(self):
        foreign = [{'pid': 424242, 'name': 'draining-worker', 'created_at': 10.0}]
        interrupt, events, _, _ = self.monitor([foreign, [], []])
        interrupt.assert_not_called()
        self.assertEqual([event['foreign'] for event in events], [foreign, [], []])

    def test_healthy_four_card_state_does_not_interrupt(self):
        interrupt, events, receipts, _ = self.monitor([[], []])
        interrupt.assert_not_called()
        receipts.assert_not_called()
        self.assertTrue(all(event['gpu_health']['healthy'] for event in events))
        self.assertTrue(all(event['speed_eligible'] for event in events))

    def test_strict_preflight_rejects_bad_health_before_stopping_production(self):
        unhealthy = {
            'healthy': False,
            'gpus': [
                {'index': 3, 'uuid': 'GPU-3', 'recovery_action': 'Reset'},
            ],
            'errors': ["GPU 3 recovery action is 'Reset', not 'None'"],
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(ROOT=Path(directory), save_json=Mock(), note=Mock())
            with patch.object(guard, 'rt', runtime), \
                    patch.object(guard, 'ISOLATION_MODE', 'strict'), \
                    patch.object(guard, 'production_idle', return_value=True), \
                    patch.object(guard, 'gpu_recovery_health', return_value=unhealthy), \
                    patch.object(guard.signal, 'signal'), \
                    patch.object(guard.subprocess, 'run') as command:
                with self.assertRaisesRegex(RuntimeError, 'Refusing unhealthy GPU recovery state'):
                    guard.main()
        command.assert_not_called()
        receipt_names = [call.args[0] for call in runtime.save_json.call_args_list]
        self.assertIn('gpu-health-interruption.json', receipt_names)

    def test_reset_required_gpu_interrupts_without_foreign_processes(self):
        reset_required = HEALTHY_GPU_OUTPUT.replace('3, GPU-3, None', '3, GPU-3, Reset')
        interrupt, events, receipts, _ = self.monitor([[]], [reset_required])
        interrupt.assert_called_once_with(guard.os.getpid(), signal.SIGINT)
        self.assertEqual(events[-1]['foreign'], [])
        self.assertFalse(events[-1]['gpu_health']['healthy'])
        self.assertFalse(events[-1]['speed_eligible'])
        self.assertEqual(receipts.call_args.args[0], 'gpu-health-interruption.json')
        self.assertEqual(receipts.call_args.args[1]['stage'], 'monitor')

    def test_malformed_missing_unsupported_and_unreadable_health_interrupt(self):
        cases = {
            'malformed': HEALTHY_GPU_OUTPUT.replace('3, GPU-3, None', '3, GPU-3'),
            'missing': '\n'.join(HEALTHY_GPU_OUTPUT.splitlines()[:3]),
            'unsupported': HEALTHY_GPU_OUTPUT.replace('3, GPU-3, None', '3, GPU-3, N/A'),
            'unreadable': guard.subprocess.CalledProcessError(1, ['nvidia-smi']),
        }
        for name, observation in cases.items():
            with self.subTest(name=name):
                interrupt, events, receipts, _ = self.monitor([[]], [observation])
                interrupt.assert_called_once_with(guard.os.getpid(), signal.SIGINT)
                self.assertFalse(events[-1]['gpu_health']['healthy'])
                self.assertEqual(receipts.call_args.args[0], 'gpu-health-interruption.json')

    def test_record_mode_records_bad_health_without_speed_eligibility(self):
        reset_required = HEALTHY_GPU_OUTPUT.replace('3, GPU-3, None', '3, GPU-3, Reset')
        interrupt, events, receipts, notes = self.monitor([[]], [reset_required], mode='record')
        interrupt.assert_not_called()
        receipts.assert_not_called()
        self.assertFalse(events[-1]['speed_eligible'])
        notes.assert_called_once_with(
            'GPU RECOVERY STATE UNHEALTHY AND RECORDED; speed measurements are not eligible'
        )

    def test_verified_restart_owner_does_not_exempt_reused_pid(self):
        state = {'listed': True, 'birth': 10.0}

        def command(args, **kwargs):
            if args[0] == 'nvidia-smi':
                text = '424242\n'
            elif args[1] == 'ps':
                text = 'owned-container\n' if state['listed'] else ''
            else:
                text = 'PID\n424242\n'
            return SimpleNamespace(stdout=text, returncode=0)

        def process(pid):
            return SimpleNamespace(create_time=lambda: state['birth'], status=lambda: 'running',
                                   name=lambda: 'worker')

        with patch.object(guard, '_OWNED_GPU_IDENTITIES', {}), \
                patch.object(guard.subprocess, 'run', side_effect=command), \
                patch.object(guard.psutil, 'Process', side_effect=process):
            self.assertEqual(guard.foreign_gpu_processes(), [])
            state['listed'] = False
            self.assertEqual(guard.foreign_gpu_processes(), [])
            state['birth'] = 20.0
            self.assertEqual(guard.foreign_gpu_processes(),
                             [{'pid': 424242, 'name': 'worker', 'created_at': 20.0}])



if __name__ == '__main__':
    unittest.main()
