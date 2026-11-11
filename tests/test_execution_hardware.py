import unittest
from unittest.mock import patch

from dci_bench.results.hardware import execution_hardware, host_hardware


class ExecutionHardwareTest(unittest.TestCase):
    @patch('dci_bench.results.hardware.host_hardware', return_value={'gpus': [{'name': 'runner GPU'}]})
    def test_remote_server_does_not_inherit_runner_gpus(self, _host):
        result = execution_hardware({'kind': 'remote_api'}, 8)
        self.assertIsNone(result['model_server']['gpus'])
        self.assertIsNone(result['model_server']['gpu_count'])
        self.assertEqual(result['model_server']['source'], 'unknown')
        self.assertEqual(result['execution']['concurrency'], 8)

    @patch('dci_bench.results.hardware.host_hardware', return_value={})
    def test_server_snapshot_keeps_device_memory_and_timestamp(self, _host):
        gpus = [{'name': 'fixture', 'total_memory_bytes': 32000000000}]
        runtime = {'captured_at': 'fixture-time', 'cuda': {'gpus': gpus},
                   'host_hardware': {'cpu_models': ['fixture CPU'], 'memory_total_bytes': 64000000000}}
        result = execution_hardware({'kind': 'vllm', 'runtime': runtime, 'tensor_parallel_size': 1}, 4)
        self.assertEqual(result['model_server']['gpus'], gpus)
        self.assertEqual(result['model_server']['gpu_count'], 1)
        self.assertEqual(result['model_server']['captured_at'], 'fixture-time')
        self.assertEqual(result['model_server']['host'], runtime['host_hardware'])

    @patch('dci_bench.results.hardware.shutil.which', return_value=None)
    def test_cpu_only_host_remains_reportable(self, _which):
        result = host_hardware()
        self.assertIsNone(result['gpus'])
        self.assertIn('logical_cpu_count', result)
        self.assertIn('memory_total_bytes', result)
