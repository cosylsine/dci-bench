#!/usr/bin/env python3
"""Print frozen expansion commands; never starts a model or evaluation."""
import argparse
import json
import os
from pathlib import Path
import shlex

from prepare_selective_expansion import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--stage', choices=['smoke', 'pilot', 'remaining', 'full_new'], default='smoke')
    p.add_argument('--backend-manifest', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    args = p.parse_args()
    plan = args.plan.resolve()
    verify(plan)
    backend = args.backend_manifest.resolve(strict=True)
    results = args.results_root.resolve()
    if results.exists():
        p.error('--results-root must be a new directory; stage outputs must remain separate')
    commands = json.loads((plan / 'commands.json').read_text())
    order = json.loads((plan / 'plan.json').read_text())['task_order']
    print('# Commands only: a fresh backend manifest does not establish endpoint/tool acceptance.')
    print('# Run the smoke stage before pilot; inspect CPU sandbox status in verification.json.')
    for task in order:
        cmd = commands[args.stage + '/' + task]
        argv = [str(backend) if x == '<ACTUAL_BACKEND_MANIFEST>' else
                str(results) if x == '<NEW_RESULTS_ROOT>' else x for x in cmd['argv']]
        env = dict(cmd['environment'])
        env['PATH'] = env.pop('PATH_prefix') + os.pathsep + os.environ.get('PATH', '')
        env['LD_LIBRARY_PATH'] = env.pop('LD_LIBRARY_PATH_prefix') + (
            os.pathsep + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        invocation = ['env', *[f'{k}={v}' for k, v in sorted(env.items())], *argv]
        print('(cd ' + shlex.quote(cmd['cwd']) + ' && ' + shlex.join(invocation) + ')')


if __name__ == '__main__':
    main()
