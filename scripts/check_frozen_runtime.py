#!/usr/bin/env python3
"""Verify a prepared RootFS and exercise the actual Pi sandbox without a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--profile', default='document-pool-v2-lite',
                        choices=('v3', 'document-pool-v1-small', 'document-pool-v1-medium',
                                 'document-pool-v1-large', 'document-pool-v2-lite', 'document-pool-v2-full'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--startup-timeout-seconds', type=float, default=240,
                        help='Operational guard for Node/source loading and sandbox startup (default: 240)')
    args = parser.parse_args()
    if args.startup_timeout_seconds <= 0:
        parser.error('--startup-timeout-seconds must be positive')
    from dci_bench.runtime import verify_runtime
    runtime = verify_runtime(args.runtime_config)
    environment = dict(os.environ)
    for key in ('DCI_PROTOCOL_PROFILE', 'DCI_PROTOCOL_CONTRACT_SHA256', 'DCI_CODE_RUNTIME_CONTRACT'):
        environment.pop(key, None)
    environment['DCI_EVALUATION_TRACK'] = 'document'
    environment['DCI_PROTOCOL_PROFILE'] = args.profile
    with tempfile.TemporaryDirectory(prefix='dci-frozen-preflight-') as temporary:
        workspace = Path(temporary)
        (workspace/'corpus').mkdir()
        (workspace/'corpus/probe.md').write_text('Frozen runtime synthetic probe.\n')
        result = subprocess.run(
            ['node', '--import', 'tsx', 'scripts/dci-run.ts', '--preflight-only',
             '--workspace', str(workspace), '--runtime-json', json.dumps(runtime.to_payload())],
            cwd=ROOT/'pi-dci', env=environment, capture_output=True, text=True,
            timeout=args.startup_timeout_seconds)
        if result.returncode:
            raise RuntimeError(f'Frozen Pi preflight failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
    report = {'status': 'passed', 'scope': 'rootfs-and-pi-sandbox', 'protocol_profile': args.profile,
              'agent_runtime': runtime.identity, 'external_model_calls': 0, 'gpu_started': False,
              'pi_preflight_stdout': result.stdout}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
