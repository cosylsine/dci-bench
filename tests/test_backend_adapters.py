"""Adapter extension tests; all providers and HTTP responses are synthetic."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request

from dci_bench.backends.adapters import ADAPTERS, get_backend_adapter, inspect_model_kwargs
from dci_bench.backends.adapters.openai_chat import OpenAIChatAdapter
from dci_bench.backends.manifest import build_api_manifest, inspect_generation_kwargs, validate_backend_manifest


class FixtureAdapter(OpenAIChatAdapter):
    def validate(self, backend):
        if backend['version'] != 'fixture-native-v1':
            raise ValueError('fixture transport version mismatch')

    def inspect_kwargs(self, backend, model, *, api_key, disable_retries):
        return {'model': 'fixture-native/' + model['served_model_name'],
                'model_base_url': backend['base_url'],
                'model_args': {'api_key': api_key, 'native_retries': 0 if disable_retries else 2}}

    def warmup_request(self, base_url, body, api_key):
        return Request(base_url + '/generate', data=json.dumps({'request': body}).encode(),
                       headers={'Content-Type': 'application/json', 'X-Fixture-Key': api_key}, method='POST')

    def normalize_warmup_response(self, payload):
        return payload['response']


class BackendAdapterTest(unittest.TestCase):
    def fixture_manifest(self, base_url='http://127.0.0.1:1'):
        return build_api_manifest(
            model_key='MiniCPM5-2B', served_model_name='fixture-model', provider='fixture', revision='v1',
            base_url=base_url, thinking_policy={'mode': 'not_supported', 'evidence': 'synthetic fixture'},
            context_length=32768, adapter='fixture-native', transport_version='fixture-native-v1',
        )

    def test_default_route_and_retry_policy_preserve_existing_wire(self):
        kwargs = inspect_model_kwargs({'openai_service': 'vllm', 'base_url': 'http://localhost/v1'},
                                      {'served_model_name': 'fixture'}, api_key='secret', disable_retries=True)
        self.assertEqual(kwargs, {'model': 'openai-api/vllm/fixture', 'model_base_url': 'http://localhost/v1',
                                 'model_args': {'api_key': 'secret', 'responses_api': False, 'max_retries': 0}})

    def test_unknown_adapter_fails_without_importing_manifest_code(self):
        for name in ('missing', 'os:system', [], None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unsupported backend adapter'):
                get_backend_adapter({'adapter': name})

    def test_registered_adapter_owns_route_but_cannot_change_contract_via_manifest(self):
        with patch.dict(ADAPTERS, {'fixture-native': FixtureAdapter()}):
            manifest = self.fixture_manifest()
            kwargs = inspect_model_kwargs(manifest['backend'], manifest['model'], api_key='secret', disable_retries=True)
            self.assertEqual(kwargs['model'], 'fixture-native/fixture-model')
            self.assertEqual(kwargs['model_args'], {'api_key': 'secret', 'native_retries': 0})
            self.assertEqual(inspect_generation_kwargs(manifest['backend'])['seed'], 0)
            manifest['backend']['generation_config']['temperature'] = 1
            with self.assertRaisesRegex(ValueError, 'shared protocol contract'):
                validate_backend_manifest(manifest)

    def test_local_nonreasoning_mapping_does_not_need_a_runner_change(self):
        generation = inspect_generation_kwargs({'kind': 'vllm', 'thinking_policy': {
            'mode': 'not_supported', 'evidence': 'synthetic nonreasoning model'}})
        self.assertNotIn('extra_body', generation)
        self.assertEqual(generation['max_tokens'], 1024)

    def test_phase4_and_legacy_batch_use_registered_provider_without_overrides(self):
        from test_phase4_runner import Phase4RunnerTest
        from test_phase4_batch_task import load_run_task_module
        from dci_bench.results.phase4_runner import execute_phase4

        fixture = Phase4RunnerTest()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(ADAPTERS, {'fixture-native': FixtureAdapter()}):
            root = Path(tmp)
            args = fixture._fixture(root)
            manifest = self.fixture_manifest()
            args.backend_manifest.write_text(json.dumps(manifest))
            fake_eval = fixture._fake_eval()

            def evaluate(task, **kwargs):
                self.assertEqual(kwargs['model'], 'fixture-native/fixture-model')
                self.assertEqual(kwargs['model_args'], {'api_key': args.api_key, 'native_retries': 2})
                return fake_eval(task, **kwargs)

            execution = execute_phase4(args, selected_query_ids=['Q1'], inspect_eval_fn=evaluate,
                                       task_factory=fixture._task_factory)
            self.assertEqual(execution.exit_code, 0)
            module = load_run_task_module()
            cli = module.parse_args(['--task', args.task, '--model-key', args.model_key, '--query-ids', 'Q1'])
            kwargs = module._inspect_eval_kwargs(cli, inspect_eval_fn=evaluate, manifest=manifest,
                                                  log_dir=root / 'logs', sample_count=1)
            self.assertEqual(kwargs['model'], 'fixture-native/fixture-model')
            self.assertNotIn('responses_api', kwargs['model_args'])

    def test_warmup_uses_adapter_url_auth_body_and_response(self):
        from scripts.harness_pilot_warmup import run_warmup

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))['request']
                requests.append((self.path, self.headers.get('X-Fixture-Key'), self.headers.get('Authorization')))
                message = {'content': 'synthetic final'}
                finish = 'stop'
                if body.get('tools'):
                    named = isinstance(body['tool_choice'], dict)
                    name = 'read' if named else 'bash'
                    arguments = {'path': '/workspace/corpus/__warmup__.md'} if named else {'command': 'printf warmup'}
                    message = {'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
                    finish = 'tool_calls'
                response = {'response': {'choices': [{'message': message, 'finish_reason': finish}],
                                         'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}}
                raw = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(ADAPTERS, {'fixture-native': FixtureAdapter()}):
                manifest = self.fixture_manifest(f'http://127.0.0.1:{server.server_port}')
                result = run_warmup(base_url=manifest['backend']['base_url'], model='fixture-model',
                                    concurrency=1, rounds=1, deadline_seconds=10, http_timeout_seconds=5,
                                    api_key='fixture-key', backend=manifest['backend'])
            self.assertEqual(result['completed_requests'], 3)
            self.assertEqual(result['usage']['total_tokens'], 36)
            self.assertEqual(requests, [('/generate', 'fixture-key', None)] * 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
