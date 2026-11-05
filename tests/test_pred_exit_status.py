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

"""mol pred exits zero only when every input was predicted."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner

from mol.runner import batch_inference, inference


class PredExitStatusTest(unittest.TestCase):
    def run_pred(self, directory, preprocess=None, predict=None):
        out_dir = Path(directory) / "out"
        runner = SimpleNamespace(configs={}, error_dir=str(out_dir / "ERR"))
        with patch.object(batch_inference, "get_default_runner", return_value=runner), \
             patch.object(batch_inference, "preprocess_input",
                          side_effect=preprocess or (lambda path, **kwargs: path)), \
             patch.object(batch_inference, "infer_predict", side_effect=predict) as infer:
            result = CliRunner().invoke(
                batch_inference.mol_cli,
                ["pred", "-i", str(Path(directory) / "in"), "-o", str(out_dir)],
            )
        return result, infer, out_dir / "ERR"

    def write_inputs(self, directory, *names):
        (Path(directory) / "in").mkdir()
        for name in names:
            (Path(directory) / "in" / f"{name}.json").write_text("[]")

    def test_all_inputs_predicted_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_inputs(d, "a", "b")
            result, infer, _ = self.run_pred(d)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(infer.call_count, 2)

    def test_failed_preparation_is_reported_and_the_rest_still_run(self):
        def preprocess(path, **kwargs):
            if path.endswith("a.json"):
                raise FileNotFoundError("first line\nsecond line")
            return path

        with tempfile.TemporaryDirectory() as d:
            self.write_inputs(d, "a", "b")
            result, infer, error_dir = self.run_pred(d, preprocess=preprocess)
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("1 of 2 input(s) failed", result.output)
            self.assertEqual(infer.call_count, 1)
            self.assertIn("first line\nsecond line", (error_dir / "a.txt").read_text())

    def test_failed_prediction_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_inputs(d, "a")
            result, _, _ = self.run_pred(d, predict=RuntimeError("1 of 1 prediction(s) failed"))
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("1 of 1 input(s) failed", result.output)

    def test_input_without_json_files_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "in").mkdir()
            (Path(d) / "in" / "notes.txt").write_text("")
            result, infer, _ = self.run_pred(d)
            self.assertNotEqual(result.exit_code, 0)
            infer.assert_not_called()

    def test_dataloader_failure_raises_and_leaves_a_report(self):
        with tempfile.TemporaryDirectory() as d:
            input_json = os.path.join(d, "input.json")
            with open(input_json, "w") as f:
                json.dump([{"name": "x", "sequences": []}], f)
            runner = SimpleNamespace(error_dir=os.path.join(d, "ERR"))
            configs = SimpleNamespace(input_json_path=input_json, use_seeds_in_json=False,
                                      seeds=[101])
            runner.build_dataloader = Mock(side_effect=ValueError("unreadable input"))
            with patch.object(inference, "get_inference_dataloader") as base_loader:
                with self.assertRaisesRegex(RuntimeError, "Dataloader initialization failed"):
                    inference.infer_predict(runner, configs)
            runner.build_dataloader.assert_called_once_with(configs=configs)
            base_loader.assert_not_called()
            self.assertIn("unreadable input",
                          Path(runner.error_dir, "error.txt").read_text())

    def test_a_sample_that_fails_featurization_fails_the_run(self):
        class Loader(list):
            dataset = [None]

        with tempfile.TemporaryDirectory() as d:
            input_json = os.path.join(d, "input.json")
            with open(input_json, "w") as f:
                json.dump([{"name": "x", "sequences": []}], f)
            runner = SimpleNamespace(error_dir=os.path.join(d, "ERR"))
            configs = SimpleNamespace(input_json_path=input_json, use_seeds_in_json=False,
                                      seeds=[101], deterministic=False)
            batch = [({"sample_name": "x"}, None, "kalign exited with status 1")]
            runner.build_dataloader = Mock(return_value=Loader([batch]))
            with patch.object(inference, "get_inference_dataloader") as base_loader:
                with self.assertRaisesRegex(RuntimeError, r"1 of 1 prediction\(s\) failed: x"):
                    inference.infer_predict(runner, configs)
            runner.build_dataloader.assert_called_once_with(configs=configs)
            base_loader.assert_not_called()
            self.assertIn("kalign exited with status 1",
                          Path(runner.error_dir, "x.txt").read_text())


if __name__ == "__main__":
    unittest.main()
