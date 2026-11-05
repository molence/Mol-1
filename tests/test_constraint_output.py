# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-free checks of the current inference loop and candidate writer."""
import ast
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'mol-1-dev').is_dir():
    ROOT = ROOT / 'mol-1-dev'

def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]

class Coordinates(list):
    @property
    def shape(self):
        return (len(self), 2, 3)

class NoConstraintFilterTest(unittest.TestCase):
    def test_violating_candidates_reach_base_dumper_unchanged(self):
        # Every source/partner atom overlaps: all five would fail the old 6 A rule.
        coordinates = Coordinates([[[0., 0., 0.], [0., 0., 0.]] for _ in range(5)])
        prediction = dict(coordinate=coordinates,
            summary_confidence=[{'ranking_score':i} for i in range(5)],
            full_data=[{'candidate':i} for i in range(5)])
        class Loader(list):
            dataset = [None]
        data = dict(sample_name='violating', sample_index=0, entity_poly_type={})
        data.update({key:SimpleNamespace(item=lambda:2)
            for key in ('N_asym', 'N_token', 'N_atom', 'N_msa')})
        ns = dict(json=json, os=os, time=time, traceback=traceback, opjoin=os.path.join,
            opexists=os.path.exists, logger=logging.getLogger('test'),
            DIST_WRAPPER=SimpleNamespace(rank=0), seed_everything=lambda **kw:None,
            update_inference_configs=lambda cfg,n:cfg,
            fix_cterminal_carboxyl_oxygens=lambda xyz,atoms:xyz,
            _peak_memory_mib=lambda device:0, _failure_hint=lambda error:'',
            torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
        infer = extract(ROOT/'mol/runner/inference.py', 'infer_predict', ns)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json'; path.write_text('[{"name":"violating"}]')
            cfg=SimpleNamespace(input_json_path=str(path), use_seeds_in_json=False,
                seeds=[101], deterministic=False, dump_dir=d)
            runner=SimpleNamespace(error_dir=str(Path(d)/'ERR'),
                device=SimpleNamespace(type='cpu'), predict=lambda data:prediction,
                update_model_configs=lambda cfg:None,
                build_dataloader=lambda **kw:Loader([[(data, [None,None], '')]]),
                dumper=SimpleNamespace(dump=Mock()),
                postprocess_prediction=Mock(side_effect=AssertionError('filter hook invoked')))
            infer(runner,cfg)
            runner.postprocess_prediction.assert_not_called()
            runner.dumper.dump.assert_called_once()
            emitted=runner.dumper.dump.call_args.kwargs['pred_dict']
            self.assertIs(emitted,prediction)
            self.assertEqual([len(emitted[k]) for k in ('coordinate','summary_confidence','full_data')],[5,5,5])

    def test_base_ranking_and_structure_writer_keep_all_candidates(self):
        def argsort(values, descending=False):
            return sorted(range(len(values)), key=lambda i:values[i], reverse=descending)
        save=Mock()
        ns=dict(torch=SimpleNamespace(tensor=lambda v:v,argsort=argsort),
                os=os, save_structure_cif=save)
        rank=extract(ROOT/'mol/runner/dumper.py','_get_ranker_indices',ns)
        write=extract(ROOT/'mol/runner/dumper.py','_save_structure',ns)
        dumper=SimpleNamespace(sorted_by_ranking_score=True)
        indices=rank(dumper,{'summary_confidence':[{'ranking_score':x} for x in (2,0,4,1,3)]})
        self.assertEqual(sorted(indices),list(range(5)))
        xyz=Coordinates([[[0.,0.,0.],[0.,0.,0.]] for _ in range(5)])
        write(dumper,xyz,'/unused','target',object(),{},101,indices)
        self.assertEqual(save.call_count,5)
        self.assertEqual(len({call.kwargs['output_fpath'] for call in save.call_args_list}),5)

if __name__=='__main__':
    unittest.main()
