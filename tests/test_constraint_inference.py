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

"""Public CLI, JSON projection and regular/EMA checkpoint regressions."""
import json
import contextlib
import io
from copy import deepcopy
import os
import pickle
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from click.testing import CliRunner
from ml_collections import ConfigDict

from mol.constraint import checkpoint, config, encoders, inference, inputs, schema
from mol.runner import batch_inference as batch
from mol.data.inference.infer_dataloader import InferenceDataset


class ConstraintInferenceTest(unittest.TestCase):
    def test_input_precheck_preserves_normal_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            payload = [dict(name='normal', modelSeeds=[101], sequences=[
                {'proteinChain': {'sequence': 'ACDE', 'count': 1}}],
                constraints={'contact': {'token_pairs': [[0, 2]]}})]
            original = json.dumps(payload, indent=2).encode()
            path.write_bytes(original)
            prepared = Path(directory) / 'prepared.json'
            runner = SimpleNamespace(configs={}, error_dir=str(Path(directory) / 'ERR'))

            def preprocess(source, **kwargs):
                self.assertEqual(source, str(path))
                self.assertEqual(path.read_bytes(), original)
                self.assertTrue(kwargs['use_msa'])
                self.assertTrue(kwargs['use_template'])
                prepared.write_bytes(original)
                return str(prepared)

            with patch.object(batch, 'get_default_runner', return_value=runner), \
                 patch.object(batch, 'preprocess_input', side_effect=preprocess) as prepare, \
                 patch.object(batch, 'infer_predict') as predict:
                result = CliRunner().invoke(batch.mol_cli, ['cpred', '-i', str(path),
                    '-o', directory, '--use_template', 'true'])
            self.assertEqual(result.exit_code, 0, result.output)
            prepare.assert_called_once()
            predict.assert_called_once_with(runner, runner.configs)
            self.assertEqual(runner.configs['input_json_path'], str(prepared))
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_raw_json_fails_before_search_and_other_inputs_continue(self):
        invalid_inputs = (
            ('[{"name":"bad","constraints":{"contact":{"token_pairs":[[0,1]]},'
             '"contact":{"token_pairs":[[2,3]]}}}]', 'duplicate JSON key: contact'),
            ('[{"name":"bad","value":NaN}]', 'invalid JSON constant: NaN'),
            ('[{"name":', 'Expecting value'),
        )
        for raw, error in invalid_inputs:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                input_dir = Path(directory) / 'inputs'
                input_dir.mkdir()
                bad, good = input_dir / 'bad.json', input_dir / 'good.json'
                bad.write_text(raw)
                good.write_text('[{"name":"good","sequences":[]}]')
                runner = SimpleNamespace(configs={}, error_dir=str(Path(directory) / 'ERR'))
                with patch.object(batch, 'get_default_runner', return_value=runner), \
                     patch.object(batch, 'preprocess_input',
                                  side_effect=lambda path, **kwargs: path) as prepare, \
                     patch.object(batch, 'infer_predict') as predict:
                    result = CliRunner().invoke(batch.mol_cli,
                        ['cpred', '-i', str(input_dir), '-o', directory])
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn('1 of 2 input(s) failed', result.output)
                prepare.assert_called_once()
                self.assertEqual(prepare.call_args.args[0], str(good))
                predict.assert_called_once()
                self.assertIn(error, (Path(runner.error_dir) / 'bad.txt').read_text())
                self.assertEqual(bad.read_text(), raw)

    def test_default_checkpoint_download_and_reuse_through_shared_runner(self):
        from mol.runner import inference as shared

        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / 'mol-1-constraint.pt'
            configs_seen = []

            def check_cache(cfg):
                configs_seen.append(cfg)
                self.assertFalse(cfg.load_checkpoint_path)
                return shared.resolve_checkpoint(cfg)

            def download(url, path, **kwargs):
                self.assertEqual(url, 'https://mol-1-release.oss-cn-shanghai.aliyuncs.com/'
                                 'checkpoint/mol-1-constraint.pt')
                self.assertEqual(Path(path), expected)
                self.assertEqual(kwargs, {'check_weight': True})
                expected.touch()

            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps({'load_checkpoint_dir': directory}))
            with patch.object(batch, 'download_inference_cache', side_effect=check_cache), \
                 patch.object(shared, 'download_from_url', side_effect=download) as fetch, \
                 patch.object(inference, 'ConstraintInferenceRunner',
                              side_effect=lambda cfg: SimpleNamespace(configs=cfg)):
                for _ in range(2):
                    batch.get_default_runner(model_name='mol-1-constraint',
                                             constraint_config=str(config_path), device='cpu')
            fetch.assert_called_once()
            self.assertEqual(len(configs_seen), 2)

    def test_explicit_missing_constraint_checkpoint_never_downloads(self):
        from mol.runner import inference as shared

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.pt'
            cfg = config.load_inference_config(None, ['--load_checkpoint_path', str(path)])
            with patch.object(shared, 'download_from_url') as fetch:
                with self.assertRaises(FileNotFoundError):
                    shared.resolve_checkpoint(cfg)
            fetch.assert_not_called()

    def test_demo_omits_default_checkpoint_and_preserves_explicit_paths(self):
        script = Path(__file__).resolve().parents[1] / 'inference_demo.sh'
        env = dict(os.environ, DRY_RUN='1')
        env.pop('CHECKPOINT_PATH', None)
        default = shlex.split(subprocess.check_output(['bash', str(script)], env=env, text=True))
        self.assertNotIn('--load_checkpoint_path', default)
        env['CHECKPOINT_PATH'] = '/custom weights/mol-1-constraint.pt'
        explicit = shlex.split(subprocess.check_output(['bash', str(script)], env=env, text=True))
        self.assertEqual(explicit[explicit.index('--load_checkpoint_path') + 1],
                         env['CHECKPOINT_PATH'])

    def test_shared_cli_and_default_budget_override_warning(self):
        pred, cpred = (batch.mol_cli.commands[n] for n in ('pred', 'cpred'))
        self.assertIs(cpred.callback, pred.callback)
        a,b=({p.name:p for p in cmd.params} for cmd in (pred,cpred))
        self.assertEqual(set(b)-set(a), {'constraint_config'})
        for command in ('pred', 'cpred'):
            for use_defaults, expected in ((False, (3, 7)), (True, (20, 200))):
                with self.subTest(command=command, use_defaults=use_defaults), \
                     patch.object(batch, 'inference_jsons') as invoke, \
                     patch.object(batch.logger, 'warning') as warning:
                    result = CliRunner().invoke(batch.mol_cli, [command, '-i', 'input.json',
                        '-c', '3', '-p', '7', '--use_default_params', str(use_defaults),
                        '--device', 'cpu', '--load_checkpoint_path', 'weights.pt'])
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertEqual((invoke.call_args.kwargs['n_cycle'],
                                      invoke.call_args.kwargs['n_step']), expected)
                    if use_defaults:
                        warning.assert_called_once()
                        message = warning.call_args.args[0] % warning.call_args.args[1:]
                        self.assertIn('--use_default_params=true', message)
                        self.assertIn('--cycle 3 -> 20', message)
                        self.assertIn('--step 7 -> 200', message)
                    else:
                        warning.assert_not_called()

    def test_default_budget_overrides_config_without_spurious_argument_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'model': {'N_cycle': 12},
                                        'sample_diffusion': {'N_step': 30}}))
            for extra in ([], ['--cycle', '20', '--step', '200']):
                with patch.object(batch, 'inference_jsons') as invoke, \
                     patch.object(batch.logger, 'warning') as warning:
                    result = CliRunner().invoke(batch.mol_cli, ['cpred', '-i', 'input.json',
                        '--config', str(path), '--use_default_params', 'true', *extra])
                self.assertEqual(result.exit_code, 0, result.output)
                warning.assert_not_called()
                kwargs = invoke.call_args.kwargs
                with patch.object(inference, 'ConstraintInferenceRunner',
                                  side_effect=lambda c: SimpleNamespace(configs=c)), \
                     patch.object(batch, 'download_inference_cache'):
                    actual = batch.get_default_runner(model_name='mol-1-constraint',
                        constraint_config=str(path), provided_options=kwargs['provided_options'],
                        n_cycle=kwargs['n_cycle'], n_step=kwargs['n_step']).configs
                self.assertEqual((actual.model.N_cycle, actual.sample_diffusion.N_step), (20, 200))

    def test_python_defaults_need_no_config_file(self):
        with patch.object(Path, 'open', side_effect=AssertionError('unexpected config file read')):
            defaults = config.load_config_overrides()
            defaults['model']['N_cycle'] = 999
            actual = config.load_inference_config(None, ['--device', 'cpu'])
        self.assertEqual(actual.model.N_cycle, 20)
        self.assertEqual(actual.sample_diffusion.N_step, 200)
        self.assertTrue(actual.constraint_contact_enable)
        self.assertTrue(actual.constraint_exclusion_enable)
        self.assertEqual(config.DEFAULT_CONFIG['model']['N_cycle'], 20)
        self.assertIsNone(next(p for p in batch.mol_cli.commands['cpred'].params
                               if p.name == 'constraint_config').default)

    def test_config_precedence_without_mutating_base_defaults(self):
        from copy import deepcopy
        from mol.config.base import configs as base
        before=pickle.dumps(base)
        cfg=config.load_config_overrides()
        cfg.update(device='cpu', use_msa=False)
        cfg['model']['N_cycle']=12;cfg['sample_diffusion']['N_step']=30
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';path.write_text(json.dumps(cfg))
            with patch.object(inference,'ConstraintInferenceRunner',side_effect=lambda c:SimpleNamespace(configs=c)), patch.object(batch,'download_inference_cache'):
                inherited=batch.get_default_runner(model_name='mol-1-constraint',constraint_config=str(path),
                    provided_options=set(),n_cycle=20,n_step=200).configs
                explicit=batch.get_default_runner(model_name='mol-1-constraint',constraint_config=str(path),
                    provided_options={'cycle','step','dtype'},n_cycle=3,n_step=7,dtype='fp32').configs
        self.assertEqual((inherited.model.N_cycle,inherited.sample_diffusion.N_step),(12,30))
        self.assertEqual((explicit.model.N_cycle,explicit.sample_diffusion.N_step,explicit.dtype),(3,7,'fp32'))
        self.assertEqual(pickle.dumps(base),before)

    def test_inline_contact_canonicalizes_without_versions(self):
        cfg = dict(constraint_contact_enable=True, constraint_exclusion_enable=True)
        for contact, expected in (
            ({'pairs': [[['B', 2], ['A', 1]], [['A', 1], ['B', 2]],
                       [['B', 1], ['A', 2]]]},
             [(('A', 1), ('B', 2)), (('A', 2), ('B', 1))]),
            ({'token_pairs': [[3, 1], [1, 3], [2, 0]]}, [(0, 2), (1, 3)]),
        ):
            payload = [dict(name='x', constraints={'contact': contact})]
            before = pickle.dumps(payload)
            native, rows = inputs.prepare_inputs(payload, config=cfg)
            self.assertEqual(rows[0]['contact'], expected)
            self.assertNotIn('constraints', native[0])
            self.assertEqual(pickle.dumps(payload), before)

    def test_inline_only_rejects_old_task_fields_and_versions(self):
        cfg = dict(constraint_contact_enable=True, constraint_exclusion_enable=True)
        for field in ('constraint', 'custom_site_context'):
            for value in (None, {}, {'target_id': 'x'}):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'inline'):
                    inputs.prepare_inputs([{'name': 'x', field: value}], config=cfg)
        for condition in ({'version': '1.0.0'},
                          {'site': {'version': '1.0.0'}},
                          {'contact': {'version': '2.0.0', 'token_pairs': [[0, 1]]}}):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                inputs.prepare_inputs([{'name': 'x', 'constraints': condition}], config=cfg)

    def test_constraint_config_extends_base_without_mutation(self):
        from mol.config.base import configs as base
        from mol.config.data import data_configs
        from mol.config.inference import inference_configs
        from mol.config.parser import parse_configs

        originals = (base, data_configs, inference_configs, config.DEFAULT_CONFIG)
        before = pickle.dumps(originals)
        inherited = parse_configs(deepcopy({**base, 'data': data_configs, **inference_configs}),
                                  fill_required_with_null=True)
        actual = config.load_inference_config(None, [])
        self.assertEqual(actual.device, inherited.device)
        self.assertEqual(actual.data.to_dict(), inherited.data.to_dict())
        self.assertEqual(actual.inference_noise_scheduler.to_dict(),
                         inherited.inference_noise_scheduler.to_dict())
        self.assertEqual(actual.model_name, 'mol-1-constraint')
        self.assertTrue(actual.constraint_contact_enable)
        self.assertTrue(actual.constraint_exclusion_enable)
        self.assertEqual(actual.constraint_exclusion_d_gap, 1.5)
        actual.data.template.prot_template_cache_dir = '/changed-in-this-call'
        self.assertEqual(pickle.dumps(originals), before)

        # A base extension becomes available without a second field registry.
        with patch.dict(base, {'base_extension_probe': 7}):
            actual = config.load_inference_config(None, ['--base_extension_probe', '9'])
            self.assertEqual(actual.base_extension_probe, 9)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit_info:
                config.load_inference_config(None, ['--help'])
            self.assertEqual(exit_info.exception.code, 0)
            self.assertIn('--base_extension_probe', output.getvalue())
            self.assertIn('--constraint_contact_enable', output.getvalue())
        self.assertEqual(pickle.dumps(originals), before)

    def test_inference_config_inheritance_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / 'parent.json'
            child = Path(directory) / 'child.json'
            parent.write_text(json.dumps({'model': {'N_cycle': 3},
                                         'sample_diffusion': {'N_step': 7},
                                         'data': {'msa': {'max_size': {'test': 128}}}}))
            child.write_text(json.dumps({'extends': 'parent.json',
                                        'model': {'N_cycle': 4},
                                        'constraint_numerics': {'detailed': True}}))
            actual = config.load_inference_config(child, ['--model.N_cycle', '5'])
            self.assertEqual(actual.model.N_cycle, 5)
            self.assertEqual(actual.sample_diffusion.N_step, 7)
            self.assertEqual(actual.data.msa.max_size.test, 128)
            self.assertTrue(actual.constraint_numerics.detailed)

    def test_json_to_native_dataset_projects_three_constraints(self):
        cfg=ConfigDict(dict(constraint_contact_enable=True,constraint_exclusion_enable=True))
        site=dict(source_chains=['A'],partner_chains=['B'],residue_ids=[['A',1]])
        payload=[dict(name='x',constraints=dict(site=site,exclusion=dict(site,residue_ids=[['A',2]]),
                                              contact={'pairs':[[['A',1],['B',1]]]}))]
        native,conditions=inputs.prepare_inputs(payload,config=cfg)
        dataset=object.__new__(inference.ConstraintInferenceDataset)
        dataset.configs=cfg;dataset.conditions={'x':conditions[0]}
        features=dict(token_index=torch.arange(4),atom_to_token_idx=torch.arange(4),is_protein=torch.ones(4,dtype=torch.long))
        class Atoms(SimpleNamespace):
            def __len__(self): return len(self.chain_id)
        atoms=Atoms(chain_id=np.array(['A','A','B','B']),label_seq_id=np.array(['1','2','1','2']))
        with patch.object(InferenceDataset,'process_one',return_value=({'input_feature_dict':features},atoms,{})):
            data,actual_atoms,_=dataset.process_one(native[0])
        f=data['input_feature_dict']
        self.assertIs(actual_atoms,atoms)
        self.assertEqual(f['site_residue_mask'].tolist(),[True,False,False,False])
        self.assertEqual(f['exclusion_residue_mask'].tolist(),[False,True,False,False])
        self.assertEqual(schema.ContactMask.from_features(f).pair_indices.tolist(),[[0,2]])
        cfg.constraint_contact_enable=False
        with self.assertRaises(ValueError): inputs.prepare_inputs(payload,config=cfg)

    def test_site_exclusion_partner_scopes_through_projection(self):
        cfg = ConfigDict(dict(constraint_contact_enable=True, constraint_exclusion_enable=True))
        cases = [
            (['B'], ['C'], 1, False),  # same residue, disjoint partners
            (['B'], ['C'], 2, False),  # disjoint residues and partners
            (['B', 'C'], ['C'], 1, True),  # partial partner overlap
            (['B'], ['B'], 1, True),
            (['B', 'C'], ['C'], 2, False),  # shared partners, disjoint residues
            (['B', 'C'], ['B', 'C'], 2, False),  # both scopes have mixed partners
        ]
        class Atoms(SimpleNamespace):
            def __len__(self): return len(self.chain_id)
        atoms = Atoms(chain_id=np.array(['A', 'A', 'B', 'C']),
                      label_seq_id=np.array(['1', '2', '1', '1']))
        for site_partners, exclusion_partners, position, conflict in cases:
            with self.subTest(site=site_partners, exclusion=exclusion_partners, position=position):
                payload = [dict(name='x', constraints=dict(
                    site=dict(source_chains=['A'], partner_chains=site_partners, residue_ids=[['A', 1]]),
                    exclusion=dict(source_chains=['A'], partner_chains=exclusion_partners,
                                   residue_ids=[['A', position]])))]
                if conflict:
                    with self.assertRaisesRegex(ValueError, 'conflict'):
                        inputs.prepare_inputs(payload, config=cfg)
                    continue
                native, rows = inputs.prepare_inputs(payload, config=cfg)
                dataset = object.__new__(inference.ConstraintInferenceDataset)
                dataset.configs = cfg
                dataset.conditions = {'x': rows[0]}
                features = dict(token_index=torch.arange(4), atom_to_token_idx=torch.arange(4),
                                is_protein=torch.tensor([1, 1, 1, 0]))
                with patch.object(InferenceDataset, 'process_one',
                                  return_value=({'input_feature_dict': features}, atoms, {})):
                    data, _, _ = dataset.process_one(native[0])
                actual = data['input_feature_dict']
                self.assertEqual(actual['site_partner_token_mask'].tolist(),
                                 [False, False, 'B' in site_partners, 'C' in site_partners])
                self.assertEqual(actual['exclusion_partner_token_mask'].tolist(),
                                 [False, False, 'B' in exclusion_partners, 'C' in exclusion_partners])

    def test_four_branch_combinations_load_regular_and_ema_strictly(self):
        for contact in (False, True):
            for exclusion in (False, True):
                flags = ['--constraint_contact_enable', str(contact),
                         '--constraint_exclusion_enable', str(exclusion)]
                cfg = config.load_inference_config(None, ['--device', 'cpu', *flags])
                model = torch.nn.Module()
                model.backbone = torch.nn.Linear(8, 8)
                model.constraint_conditioner = encoders.ResidueSiteConstraintEncoder(8)
                if exclusion:
                    model.constraint_conditioner.exclusion = encoders.ResidueSiteConstraintEncoder(8)
                if contact:
                    model.constraint_conditioner.contact = encoders.ContactMaskEncoder(8)
                base = {k: v.clone() for k, v in model.state_dict().items()
                        if not k.startswith('constraint_conditioner.')}
                for exported in (False, True):
                    with self.subTest(contact=contact, exclusion=exclusion, ema=exported), \
                         tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'weights.pt'
                        weights = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
                        saved = dict(model=weights, model_name='mol-1-constraint', ema_export=exported)
                        # Literal export contracts keep this test independent of writers.
                        if contact:
                            saved['contact'] = {'contract': {
                                'cutoff_angstrom': 4.5, 'geometry': 'minimum_heavy_atom',
                                'scope': 'token_pairs', 'positive_only': True, 'zero': 'unknown'}}
                        if exclusion:
                            saved['exclusion'] = {'contract': {
                                'distance': 'minimum-heavy-atom', 'positive_cutoff': 4.5,
                                'exclusion_cutoff': 6.0, 'd_gap': 1.5, 'scope': 'protein_source',
                                'violation': 'any_distance_le_cutoff'}}
                        torch.save(saved, path)
                        cfg.load_checkpoint_path = str(path)
                        runner = object.__new__(inference.ConstraintInferenceRunner)
                        runner.configs, runner.model = cfg, model
                        runner.load_checkpoint()
                        self.assertFalse(model.training)
                        for name, value in model.state_dict().items():
                            self.assertTrue(torch.equal(value, weights[name]), name)
                        # Old labels are irrelevant; actual contracts still determine compatibility.
                        labelled = deepcopy(saved)
                        labelled['constraint_schema'] = 'historical-unused'
                        for branch in ('contact', 'exclusion'):
                            if branch in labelled:
                                labelled[branch]['contract']['version'] = 'historical-unused'
                        if exclusion:
                            labelled['exclusion']['training_protocol'] = 'historical-unused'
                        torch.save(labelled, path)
                        runner.load_checkpoint()
                        for branch in ('contact', 'exclusion'):
                            if branch not in labelled:
                                continue
                            broken_contract = deepcopy(labelled)
                            broken_contract[branch]['contract']['scope'] = 'incorrect'
                            torch.save(broken_contract, path)
                            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                                runner.load_checkpoint()
                        # Missing branch parameters must fail before copying any weights.
                        broken = dict(weights)
                        broken.pop('constraint_conditioner.output_projection.weight')
                        torch.save(dict(saved, model=broken), path)
                        with self.assertRaisesRegex(RuntimeError, 'tensor keys mismatch'):
                            runner.load_checkpoint()
                        for name, value in model.state_dict().items():
                            self.assertTrue(torch.equal(value, weights[name]), name)
                        torch.save(dict(model=base), path)
                        with self.assertRaises((RuntimeError, ValueError)):
                            runner.load_checkpoint()

    def test_incomplete_and_conflicting_inputs_fail(self):
        cfg=config.load_inference_config(None,['--device','cpu'])
        for condition in ({'site':{'source_chains':['A']}},
                          {'contact':{'token_pairs':[[0,0]]}},
                          {'site':{'source_chains':['A'],'partner_chains':['A'],'residue_ids':[['A',1]]}}):
            with self.assertRaises(ValueError): inputs.prepare_inputs([{'name':'x','constraints':condition}],config=cfg)

    def test_rank_sharding_has_no_padding_or_duplicates(self):
        class Dataset:
            def __len__(self):return 5
        runner=object.__new__(inference.ConstraintInferenceRunner)
        cfg=ConfigDict(dict(num_workers=0));assigned=[]
        with patch.object(inference,'ConstraintInferenceDataset',return_value=Dataset()):
            for rank in range(2):
                with patch.object(inference,'DIST_WRAPPER',SimpleNamespace(rank=rank,world_size=2)):
                    assigned+=list(runner.build_dataloader(cfg).sampler)
        self.assertEqual(sorted(assigned),list(range(5)))


if __name__ == '__main__': unittest.main()
