"""CPU-only acceptance of paired selection, budget isolation and context checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import size_budget_experiment as S
import run_size_budget_experiment as B


class SizeBudgetTests(unittest.TestCase):
    def test_tiers_are_exhaustive_and_boundary_stable(self):
        for n, name, total, reserve in [(1,'small',128000,32000),(1000,'small',128000,32000),
                (1001,'medium',256000,48000),(100000,'medium',256000,48000),
                (100001,'large',512000,64000),(511954,'large',512000,64000)]:
            self.assertEqual(S.tier(n),dict(name=name,total=total,reserve=reserve))
        for n in [0,-1,True,1.5]:
            with self.assertRaises(ValueError): S.tier(n)

    def test_selection_uses_only_id_hash_and_retains_source_order(self):
        ids = [f'q{i}' for i in range(10)]
        selected = S.choose('task',ids)
        self.assertEqual(len(selected),2)
        self.assertEqual(selected,[q for q in ids if q in selected])
        self.assertEqual(set(selected),set(S.choose('task',list(reversed(ids)))))
        with self.assertRaises(ValueError): S.choose('task',ids[:9])

    def test_paired_batches_same_ids_counterbalanced_and_bounded(self):
        plan = B.E.read(B.pilot.PLAN / 'plan.json')
        splits = B.E.read(B.pilot.PLAN / 'query-splits.json')
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            info = {k:dict(runtime='/runtime/'+k,profile=k,contract_sha256=k,total=1)
                    for k in ['baseline','small','medium','large']}
            (bundle/'runtimes.json').write_text(json.dumps(info))
            work = S.batches(plan,splits,8,bundle)
        self.assertEqual(len(work),14)
        self.assertEqual(sum(len(b['query_ids']) for b in work),28)
        self.assertEqual(len({(b['task'],q) for b in work for q in b['query_ids']}),14)
        for i in range(7):
            a,b = work[2*i:2*i+2]
            self.assertEqual(a['query_ids'],b['query_ids'])
            self.assertEqual(a['stage'],'baseline' if i%2==0 else 'tiered')
            self.assertNotEqual(a['stage'],b['stage'])
        locked = threading.Lock()
        active = peak = 0
        def run(batch):
            nonlocal active,peak
            with locked:
                active += batch['max_concurrency']; peak=max(peak,active)
            time.sleep(.01)
            with locked: active -= batch['max_concurrency']
            return [batch]
        self.assertEqual(len(S.execute_batches(work,run,8)),14)
        self.assertLessEqual(peak,8)

    def test_live_context_must_be_explicit_and_exact(self):
        S.check_context({'data':[dict(id='Qwen3.8-27B',max_model_len=S.CONTEXT)]})
        for data in [[],[dict(id='Qwen3.8-27B')],
                     [dict(id='Qwen3.8-27B',max_model_len=S.CONTEXT//2)]]:
            with self.assertRaises(ValueError): S.check_context({'data':data})

    def test_redirected_dependency_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle'
            frozen=Path(tmp)/'frozen'
            frozen.mkdir()
            for sub in ['packages','node_modules']: (frozen/sub).mkdir()
            info={'baseline':{'runtime':str(frozen)}}
            for _,name,_,_ in S.TIERS:
                runtime=bundle/'runtimes'/name
                (runtime/'pi-dci').mkdir(parents=True)
                for sub in ['packages','node_modules']:
                    (runtime/'pi-dci'/sub).symlink_to(frozen/sub,target_is_directory=True)
                info[name]={'runtime':str(runtime)}
            (bundle/'runtimes.json').write_text(json.dumps(info))
            (bundle/'bindings.json').write_text(json.dumps(dict(baseline_runtime=str(frozen),
                frozen_pi=str(frozen),anchors={})))
            S.verify_bindings(bundle)
            link=bundle/'runtimes/large/pi-dci/packages'
            link.unlink()
            link.symlink_to(bundle,target_is_directory=True)
            with self.assertRaises(ValueError): S.verify_bindings(bundle)

    def test_driver_derivation_keeps_startup_but_replaces_seventy_query_loop(self):
        source=(ROOT/'scripts/run_selective_expansion_pilot.py').read_text()
        result=B.adapt_driver(source,Path('/new/bundle'))
        self.assertIn('experiment.check_context(models)',result)
        self.assertIn('verify_batch_in_child',result)
        self.assertIn('len(records) != 28',result)
        self.assertNotIn('len(records) != 70',result)
        self.assertIn("summary['valid'] == 28",result)
        self.assertIn("summary['failed_samples']",result)
        self.assertIn('budget_policy=experiment.RESERVE_MODE',result)
        for key in ['inspect-parallel','enterprise-startup','build_vllm_manifest','check_ports(port, work)']:
            self.assertIn(key,result)
        with self.assertRaises(ValueError): B.replace_once('x x','x','y')

    def test_v2_keeps_frozen_pi_and_funds_long_history_finalization(self):
        source = B.E.code_path(B.E.read(B.pilot.PLAN / 'plan.json')['execution_runtime'])
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / 'small'
            info = B.make_runtime(source, runtime, S.tier(100))
            runner = runtime / 'pi-dci/scripts/dci-run.ts'
            self.assertEqual(runner.read_bytes(), (source / 'pi-dci/scripts/dci-run.ts').read_bytes())
            self.assertEqual((runtime / 'budget-runtime.patch').read_text(), '')
            contract = B.E.read(runtime / 'resolved-contract.json')
            self.assertEqual(contract['sample_timeout_seconds'], 600)
            self.assertEqual(contract['max_total_model_tokens'], 128000)
            self.assertIn('hard_timeout_only', contract['budget_control']['wall_time_policy'])
            self.assertEqual(info['profile'], 'budgeted-selective-size-v2-small')
            # Execute the actual frozen forecast, not a Python reimplementation.
            script = Path(tmp) / 'forecast.mjs'
            script.write_text('import {estimateBudgetReserve} from ' + json.dumps(runner.as_uri()) + ';\n' + '''
// Observed final request estimates from the two 20261021 failed samples.
console.log(JSON.stringify([[20401,17216,32000], [28165,22738,48000]].map(
    ([nextInput, remaining, reserve_tokens]) => {
        const message = {role: 'assistant', content: [{type: 'text', text: ''}],
            usage: {input: nextInput - 1, output: 1, totalTokens: nextInput}};
        return {remaining, ...estimateBudgetReserve({messages: [message]}, {reserve_tokens})};
    })));
''')
            result = subprocess.run(['node', str(script)], env=S.arm_environment(B.E.environment(), info),
                                    text=True, capture_output=True, check=True, timeout=60)
            for forecast in json.loads(result.stdout):
                self.assertGreater(forecast['reserve'], 64000)
                self.assertGreater(forecast['reserve'], forecast['nextRequest'] * 2)
                self.assertLess(forecast['remaining'], forecast['nextRequest'])

    def test_failures_are_exposed_without_reclassifying_them_as_valid(self):
        records = [dict(sample=dict(run_id='tiered', task='task', query_id='q', valid_output=False,
                                   failure=dict(kind='timeout', message='deadline'), usage={}, timing={}),
                        run='evaluation/task/tiered', experiment=dict(tier='medium'))]
        report = S.summarize(records, ['task'], lambda rows, tasks: dict(completed=len(rows)))
        self.assertEqual(report['valid'], 0)
        self.assertEqual(report['failed_samples'][0]['failure']['kind'], 'timeout')

    def test_backend_accepts_only_declared_context_change(self):
        source=(ROOT/'scripts/run_selective_expansion_pilot.py').read_text()
        namespace={'__name__':'derived_driver_test', '__file__':str(ROOT/'scripts/driver.py')}
        exec(B.adapt_driver(source,Path('/new/bundle')), namespace)
        expected=B.E.read(B.E.SOURCE/'expected-backend.json')
        actual=deepcopy(expected)
        actual['backend']['context_length']=S.CONTEXT
        confirmation=Mock()
        namespace['check_backend_identity'](confirmation,expected,actual,4)
        confirmation.check_backend.assert_called_once_with(B.E.SOURCE, expected)
        self.assertEqual(actual['backend']['context_length'],S.CONTEXT)
        actual['backend']['runtime']['launch_environment']['VLLM_EXTRA_ARGS']='--different'
        with self.assertRaises(ValueError):
            namespace['check_backend_identity'](confirmation,expected,actual,4)
        actual['backend']['context_length']=65536
        with self.assertRaises(ValueError):
            namespace['check_backend_identity'](confirmation,expected,actual,4)

    def test_summary_preserves_pairs_and_rejects_data_drift(self):
        def record(arm,total):
            return dict(sample=dict(run_id=arm,task='task',query_id='q',query_sha256='same',
                valid_output=True,metrics=dict(recall_at_10=.5,f1_at_10=.4,ndcg_at_10=.6),
                usage=dict(total_tokens=total),timing=dict(wall_time_seconds=3)),
                data_identity={'corpus':'same'},experiment=dict(tier='large',total=total))
        records=[record('baseline',128000),record('tiered',512000)]
        report=S.summarize(records,['task'],lambda rows,tasks:dict(completed=len(rows)))
        self.assertEqual(report['by_tier']['large']['tiered']['completed'],1)
        self.assertEqual(report['paired_samples'][0]['deltas']['recall_at_10'],0)
        records[1]['data_identity']['corpus']='different'
        with self.assertRaises(ValueError):
            S.summarize(records,['task'],lambda rows,tasks:{})


if __name__ == '__main__': unittest.main()
