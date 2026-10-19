#!/usr/bin/env python3
"""Register a frozen expansion catalog locally, then use the existing runner."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.expansion import sha256_file
from dci_bench.data.registry import TASKS, TaskSpec

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('--suite', required=True, type=Path)
args, remaining = parser.parse_known_args()
catalog_path = args.suite / 'catalog.json'
completion = json.loads((args.suite / '_COMPLETE.json').read_text())
if sha256_file(catalog_path) != completion['catalog_sha256']:
    raise ValueError('Expansion catalog hash mismatch')
catalog = json.loads(catalog_path.read_text())
for row in catalog['tasks']:
    if sha256_file(args.suite / 'metadata' / row['task'] / '_COMPLETE.json') != catalog['task_completion_hashes'][row['task']]:
        raise ValueError('Task completion hash mismatch')
    spec = TaskSpec(row['task'], row['dataset'], row['revision'], row['task'], row['task_instruction'])
    if row['task'] in TASKS:
        previous = TASKS[row['task']]
        if (previous.hf_dataset, previous.revision, previous.task_instruction) != (spec.hf_dataset, spec.revision, spec.task_instruction):
            raise ValueError('Expansion task collides with a different registered task')
    else:
        TASKS[row['task']] = spec
if any(arg.split('=')[0] in ('--prepare', '--prepare-data', '--workspace-root', '--metadata-root') for arg in remaining):
    raise ValueError('Expansion uses its already prepared frozen workspace and metadata')
sys.argv = [str(ROOT / 'scripts/run_task.py'), '--workspace-root', str(args.suite / 'workspaces'),
            '--metadata-root', str(args.suite / 'metadata'), *remaining]
runpy.run_path(str(ROOT / 'scripts/run_task.py'), run_name='__main__')
