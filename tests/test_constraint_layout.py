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

"""Release boundaries and unchanged base extension contracts."""
import ast
import json
import pickle
import unittest
from pathlib import Path

from mol.constraint import config

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'mol/constraint'


class ConstraintLayoutTest(unittest.TestCase):
    def test_import_direction(self):
        for name in ('schema', 'features', 'encoders', 'model', 'runtime'):
            tree = ast.parse((PACKAGE / (name + '.py')).read_text())
            for node in ast.walk(tree):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
                for mod in mods:
                    if mod and mod.startswith('mol.constraint'):
                        self.assertNotIn(mod, {'mol.constraint.train', 'mol.constraint.inference'})
                        self.assertFalse(mod.startswith('mol.constraint.data'), (name,mod))

    def test_private_model_preset_does_not_mutate_base(self):
        from mol.config.model_type import model_configs
        before = pickle.dumps(model_configs)
        private = config.constraint_model_configs()
        self.assertEqual(set(private), {'mol-1-constraint'})
        private['mol-1-constraint']['c_z'] = 1
        self.assertEqual(pickle.dumps(model_configs), before)

    def test_base_hook_preserved(self):
        from mol.model.mol import Mol
        self.assertIsNone(Mol.get_constraint_update(None, {}))

    def test_single_packaged_config_and_current_entrypoints(self):
        import tomllib
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        patterns = project['tool']['setuptools']['package-data']['mol']
        included = {p.resolve() for pattern in patterns for p in (ROOT / 'mol').glob(pattern)}
        self.assertEqual(list(PACKAGE.rglob('*.json')), [])
        self.assertNotIn('constraint/config.json', patterns)
        cfg = config.load_config_overrides()
        self.assertNotIn('training', cfg)
        self.assertFalse((ROOT / 'posttrain_constraint_demo.sh').exists())
        for removed in ('train.py', 'trainability.py', 'losses.py', 'geometry.py'):
            self.assertFalse((PACKAGE / removed).exists())
        self.assertFalse(list((PACKAGE / 'data').glob('*.py')))
        self.assertIn('mol/constraint/inference.py', (ROOT / 'inference_demo.sh').read_text())


if __name__ == '__main__': unittest.main()
