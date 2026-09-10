"""Deterministic regression coverage for speculative-matrix baseline reuse."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import speculative_matrix as matrix


class BaselineEvidence:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.benchmark = self.root / 'llm_decode_bench.py'
        self.benchmark.write_text('# synthetic benchmark identity\n')
        self.rates = {}
        self.summary = {'complete': True, 'passed': True, 'arms': []}
        self.gates = []
        self._build()

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, indent=2) + '\n')
        return path

    def read_json(self, name):
        return json.loads((self.root / name).read_text())

    def rewrite_gates(self):
        (self.root / 'gates.jsonl').write_text(
            ''.join(json.dumps(gate) + '\n' for gate in self.gates))

    def _build(self):
        benchmark_sha256 = hashlib.sha256(self.benchmark.read_bytes()).hexdigest()
        self.write_json('experiment-plan.json', {
            'arms': [[arm, image] for arm, image, _ in matrix.BASELINE_ARMS],
            'model': str(matrix.MODELS['published']),
            'tp': 4,
            'dcp': 1,
            'contexts': [0, 32768],
            'concurrency': [1, 4, 8],
            'duration_seconds': 30,
            'sampling': {
                'temperature': 1.0,
                'top_p': 0.95,
                'reasoning_effort': 'max',
                'clear_thinking': False,
            },
            'benchmark_sha256': benchmark_sha256,
        })

        timestamp = 1.0
        for arm_index, (arm, image, _) in enumerate(matrix.BASELINE_ARMS):
            benchmark_path = (self.root / f'{arm}.json').resolve()
            launch_path = (self.root / f'{arm}.launch.json').resolve()
            counter_path = (self.root / f'{arm}.steady-summary.json').resolve()
            cells = []
            rates = {}
            for concurrency, context in sorted(matrix.EXPECTED):
                rate = 100.0 + arm_index * 10 + concurrency + context / 32768
                rates[(concurrency, context)] = rate
                cells.append({
                    'concurrency': concurrency,
                    'context_tokens': context,
                    'benchmark_mode': 'duration',
                    'request_count_target': 0,
                    'measurement_seconds': 30.0,
                    'aggregate_source': 'openai_continuous_usage',
                    'aggregate_tps': rate,
                    'num_errors': 0,
                    'underfilled': False,
                    'warmup_timed_out': False,
                    'capacity_limited': False,
                })
            self.rates[arm] = rates
            self.write_json(f'{arm}.json', {
                'metadata': {
                    'engine': 'vllm',
                    'model': 'GLM-5.3-Flash-NVFP4',
                    'decode_mode': 'duration',
                    'primary_decode_layer': 'sustained_decode',
                    'duration_per_test': 30.0,
                    'request_count': 0,
                    'warmup_request_count': 0,
                    'run_burst': False,
                    'standalone_prefill': False,
                    'prefill_only': False,
                    'skip_prefill': False,
                    'max_tokens': 8192,
                    'ignore_eos': True,
                    'concurrency_levels': [1, 4, 8],
                    'context_lengths': [0, 32768],
                },
                'results': cells,
            })
            self.write_json(f'{arm}.launch.json', {
                'label': arm,
                'image': image,
                'tp': 4,
                'dcp': 1,
                'spec': 'mtp0',
                'cache': 'vram',
                'kv': 'fp8_ds_mla',
                'env': dict(matrix.BASELINE_LAUNCH_ENV),
                'extra_args': list(matrix.BASELINE_EXTRA_ARGS),
                'l2_host': None,
                'gpus': '0,1,2,3',
                'model_dir': str(matrix.MODELS['published']),
            })
            self.write_json(f'{arm}.bench.command.json', {
                'args': [
                    'python3', str(self.benchmark), '--port', '5002',
                    '--model', 'GLM-5.3-Flash-NVFP4', '--concurrency', '1,4,8',
                    '--contexts', '0,32k', '--duration', '30', '--max-tokens', '8192',
                    '--output', str(benchmark_path),
                ],
                'returncode': 0,
                'started_at': timestamp,
                'finished_at': timestamp + 1,
                'elapsed_seconds': 1.0,
            })
            self.write_json(f'{arm}.steady-summary.json', {
                'schema': 'r26-steady-counters/v1',
                'label': arm,
                'source_benchmark': str(benchmark_path),
                'cells': [
                    {'concurrency': concurrency, 'context_tokens': context, 'valid': True}
                    for concurrency, context in sorted(matrix.EXPECTED)
                ],
                'all_windows_valid': True,
            })
            summary_arm = {
                'arm': arm,
                'image': image,
                'passed': True,
                'issues': [],
                'benchmark': str(benchmark_path),
                'counter_windows_valid': True,
                'counter_summary': str(counter_path),
            }
            self.summary['arms'].append(summary_arm)
            self.gates.extend([
                {
                    'name': f'boot:{arm}',
                    'passed': True,
                    'detail': {
                        'returncode': 0,
                        'image': image,
                        'launch': str(launch_path),
                    },
                    'timestamp': timestamp,
                },
                {
                    'name': f'benchmark-execution:{arm}',
                    'passed': True,
                    'detail': {
                        'returncode': 0,
                        'result': str(benchmark_path),
                        'concurrency': '1,4,8',
                        'contexts': '0,32k',
                        'duration_seconds': 30,
                    },
                    'timestamp': timestamp + 0.25,
                },
                {
                    'name': f'runtime-baseline:{arm}',
                    'passed': True,
                    'detail': dict(summary_arm),
                    'timestamp': timestamp + 0.5,
                },
            ])
            timestamp += 2

        self.write_json('runtime-baseline-summary.json', self.summary)
        self.rewrite_gates()
        comparison = {'rows': []}
        for concurrency, context in sorted(matrix.EXPECTED):
            cell = (concurrency, context)
            r281_repeats = [self.rates[arm][cell] for arm in ('r281-ab', 'r281-ba')]
            r29_repeats = [self.rates[arm][cell] for arm in ('r29-ab', 'r29-ba')]
            r281_mean = sum(r281_repeats) / 2
            r29_mean = sum(r29_repeats) / 2
            comparison['rows'].append({
                'concurrency': concurrency,
                'context': context,
                'r281_repeats': r281_repeats,
                'r29_repeats': r29_repeats,
                'r281_mean': r281_mean,
                'r29_mean': r29_mean,
                'change_percent': (r29_mean / r281_mean - 1.0) * 100,
            })
        self.write_json('paired-runtime-comparison.json', comparison)


class SpecBaselineReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.evidence = BaselineEvidence(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self):
        return matrix.verify_baseline_reuse(
            self.evidence.root, benchmark_path=self.evidence.benchmark)

    def test_accepts_consistent_abba_evidence_with_auditable_sources(self):
        receipt = self.verify()

        self.assertTrue(receipt['client_timing_valid'])
        self.assertEqual(receipt['repeat_arms'], {
            'r281': ['r281-ab', 'r281-ba'],
            'r29': ['r29-ab', 'r29-ba'],
        })
        self.assertEqual(len(receipt['derived_comparison']), 6)
        sources = {source['role']: source for source in receipt['sources']}
        raw = self.evidence.root / 'r29-ab.json'
        self.assertEqual(sources['r29-ab_raw_benchmark']['path'], str(raw))
        self.assertEqual(
            sources['r29-ab_raw_benchmark']['sha256'],
            hashlib.sha256(raw.read_bytes()).hexdigest(),
        )

    def test_rejects_missing_raw_arm_receipt(self):
        (self.evidence.root / 'r29-ba.json').unlink()

        with self.assertRaises(matrix.BaselineReuseError):
            self.verify()

    def test_rejects_partial_cell_matrix(self):
        receipt = self.evidence.read_json('r29-ba.json')
        receipt['results'].pop()
        self.evidence.write_json('r29-ba.json', receipt)

        with self.assertRaises(matrix.BaselineReuseError):
            self.verify()

    def test_rejects_stale_derived_rate(self):
        comparison = self.evidence.read_json('paired-runtime-comparison.json')
        comparison['rows'][0]['r29_mean'] += 1.0
        self.evidence.write_json('paired-runtime-comparison.json', comparison)

        with self.assertRaises(matrix.BaselineReuseError):
            self.verify()

    def test_rejects_failed_source_cell(self):
        receipt = self.evidence.read_json('r29-ab.json')
        receipt['results'][0]['capacity_limited'] = True
        self.evidence.write_json('r29-ab.json', receipt)

        with self.assertRaises(matrix.BaselineReuseError):
            self.verify()

    def test_rejects_failed_recorded_source_gate(self):
        gate = next(
            gate for gate in self.evidence.gates
            if gate['name'] == 'runtime-baseline:r29-ab'
        )
        gate['passed'] = False
        self.evidence.rewrite_gates()

        with self.assertRaises(matrix.BaselineReuseError):
            self.verify()

    def test_rejects_mismatched_launch_identity(self):
        launch = self.evidence.read_json('r29-ab.launch.json')
        launch['dcp'] = 4
        self.evidence.write_json('r29-ab.launch.json', launch)

        with self.assertRaises(matrix.BaselineReuseError):
            self.verify()

    def test_keeps_supplemental_counter_validity_separate(self):
        counter = self.evidence.read_json('r29-ab.steady-summary.json')
        counter['cells'][0]['valid'] = False
        counter['all_windows_valid'] = False
        self.evidence.write_json('r29-ab.steady-summary.json', counter)
        summary_arm = next(
            arm for arm in self.evidence.summary['arms'] if arm['arm'] == 'r29-ab'
        )
        summary_arm['counter_windows_valid'] = False
        self.evidence.write_json('runtime-baseline-summary.json', self.evidence.summary)
        source_gate = next(
            gate for gate in self.evidence.gates
            if gate['name'] == 'runtime-baseline:r29-ab'
        )
        source_gate['detail']['counter_windows_valid'] = False
        self.evidence.rewrite_gates()

        receipt = self.verify()

        self.assertTrue(receipt['client_timing_valid'])
        self.assertFalse(receipt['supplemental_counter_windows_valid'])
        self.assertFalse(receipt['supplemental_counter_windows_by_arm']['r29-ab'])


if __name__ == '__main__':
    unittest.main()
