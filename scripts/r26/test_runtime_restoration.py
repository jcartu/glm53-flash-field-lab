"""Regression coverage for accepted stops, cleanup failures and restoration."""
from contextlib import ExitStack
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runtime
import run_qualification as coordinator


class DockerState:
    def __init__(self, stop_failure=None):
        self.production_id = 'original-production-id'
        self.test_id = 'owned-test-id'
        self.production_running = True
        self.test_running = False
        self.pending_stop = False
        self.stop_failure = stop_failure
        self.stop_calls = 0

    def run(self, args, **kwargs):
        if args[:2] == ['docker', 'inspect']:
            target = args[-1]
            if target in (coordinator.PRODUCTION, self.production_id):
                data = {'Id': self.production_id, 'Image': 'original-image', 'Name': '/glm53-prod',
                        'State': {'Running': self.production_running}, 'Config': {'Labels': {}}}
            elif target in (runtime.NAME, self.test_id) and self.test_running:
                data = {'Id': self.test_id, 'Image': 'candidate-image', 'Name': '/' + runtime.NAME,
                        'State': {'Running': True}, 'Config': {'Labels': {'field-lab.battery': 'r26'}}}
            else:
                return SimpleNamespace(returncode=1, stdout='', stderr='not found')
            return SimpleNamespace(returncode=0, stdout=json.dumps([data]), stderr='')
        if args[:2] == ['docker', 'stop']:
            self.stop_calls += 1
            if self.stop_calls == 1 and self.stop_failure:
                self.pending_stop = True
                if self.stop_failure == 'interrupt':
                    raise KeyboardInterrupt('stop accepted before client interruption')
                raise subprocess.TimeoutExpired(args, 90)
            self.pending_stop = False
            self.production_running = False
        elif args[:2] == ['docker', 'start']:
            self.production_running = True
        elif args[:2] == ['docker', 'rm']:
            self.test_running = False
        else:
            raise AssertionError('Unexpected external operation: ' + repr(args))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def complete_daemon_work(self):
        if self.pending_stop:
            self.production_running = False
            self.pending_stop = False

    def phase(self, *args, **kwargs):
        self.test_running = True
        return 0


def healthy_response(*args, **kwargs):
    return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                           json=lambda: {'data': [{'id': 'original-model'}]})


class RuntimeRestorationTests(unittest.TestCase):
    def exercise(self, state, capture_error=None):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(runtime, 'ROOT', Path(directory)))
            stack.enter_context(patch.object(runtime, '_CURRENT_LABEL', None))
            stack.enter_context(patch.object(runtime, 'capture', side_effect=capture_error))
            stack.enter_context(patch.object(runtime, 'note'))
            stack.enter_context(patch.object(runtime, 'run', side_effect=state.phase))
            stack.enter_context(patch.object(coordinator, 'PHASES', [('fixture', 'unused.py', [], 1)]))
            stack.enter_context(patch.object(coordinator, 'production_idle', return_value=True))
            stack.enter_context(patch.object(coordinator, 'gpu_recovery_health', return_value={'healthy': True, 'gpus': [], 'errors': []}))
            stack.enter_context(patch.object(coordinator, 'foreign_gpu_processes', return_value=[]))
            stack.enter_context(patch.object(coordinator, 'watch_gpu_isolation', return_value=None))
            stack.enter_context(patch.object(coordinator.signal, 'signal'))
            stack.enter_context(patch.object(coordinator.subprocess, 'run', side_effect=state.run))
            stack.enter_context(patch.object(coordinator.requests, 'get', side_effect=healthy_response))
            stack.enter_context(patch.object(coordinator.time, 'sleep', return_value=None))
            caught = None
            try:
                coordinator.main()
            except (KeyboardInterrupt, subprocess.TimeoutExpired, OSError) as error:
                caught = error
            state.complete_daemon_work()
            return caught

    def test_interrupt_after_stop_acceptance_restores_original(self):
        state = DockerState('interrupt')
        error = self.exercise(state)
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertTrue(state.production_running)
        self.assertFalse(state.test_running)
        self.assertFalse(state.pending_stop)

    def test_timed_out_stop_client_is_reconciled_before_restore(self):
        state = DockerState('timeout')
        error = self.exercise(state)
        self.assertIsInstance(error, subprocess.TimeoutExpired)
        self.assertTrue(state.production_running)
        self.assertFalse(state.test_running)
        self.assertFalse(state.pending_stop)

    def test_diagnostic_timeout_does_not_leave_test_owning_gpus(self):
        state = DockerState()
        self.exercise(state, subprocess.TimeoutExpired(['nvidia-smi'], 20))
        self.assertFalse(state.test_running)
        self.assertTrue(state.production_running)

    def test_diagnostic_filesystem_failure_does_not_prevent_restoration(self):
        state = DockerState()
        self.exercise(state, OSError('evidence volume unavailable'))
        self.assertFalse(state.test_running)
        self.assertTrue(state.production_running)


if __name__ == '__main__':
    unittest.main()
