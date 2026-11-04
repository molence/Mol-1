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

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mol.runner import msa_search
from mol.web_service import colab_request_parser


def fake_service(x, prefix, use_pairing=False, **kwargs):
    """Write what the ColabFold client writes, without contacting a server."""
    if use_pairing:
        sequences = x.splitlines()[1::2]
        for i, sequence in enumerate(sequences):
            os.makedirs(os.path.join(prefix, str(i)), exist_ok=True)
            with open(os.path.join(prefix, str(i), "pairing.a3m"), "w") as f:
                f.write(f">query\n{sequence}\n>UniRef100_A_1/\n{sequence}\n")
        return None
    query_id = str(int(x.split("\n")[0].split("_")[-1]))
    out = os.path.join(prefix, query_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "non_pairing.a3m"), "w") as f:
        f.write(f">query\n{x.split(chr(10))[1]}\n")
    return os.path.abspath(out)


class MsaSearchTest(unittest.TestCase):
    def test_complex_gets_its_paired_msa_next_to_each_unpaired_one(self):
        task = {"name": "t", "sequences": [
            {"proteinChain": {"sequence": "MKAAA", "count": 1}},
            {"proteinChain": {"sequence": "MKCCC", "count": 1}},
        ]}
        with tempfile.TemporaryDirectory() as d, \
             patch.object(colab_request_parser, "run_mmseqs2_service", side_effect=fake_service):
            msa_search.update_seq_msa(task, d)
            for entity in task["sequences"]:
                chain = entity["proteinChain"]
                self.assertTrue(os.path.isfile(chain["unpairedMsaPath"]))
                self.assertTrue(os.path.isfile(chain["pairedMsaPath"]))
                self.assertEqual(os.path.dirname(chain["pairedMsaPath"]),
                                 os.path.dirname(chain["unpairedMsaPath"]))

    def test_single_chain_has_no_paired_msa(self):
        task = {"name": "t", "sequences": [{"proteinChain": {"sequence": "MKAAA", "count": 1}}]}
        with tempfile.TemporaryDirectory() as d, \
             patch.object(colab_request_parser, "run_mmseqs2_service", side_effect=fake_service) as service:
            msa_search.update_seq_msa(task, d)
            chain = task["sequences"][0]["proteinChain"]
            self.assertTrue(os.path.isfile(chain["unpairedMsaPath"]))
            self.assertNotIn("pairedMsaPath", chain)
            self.assertFalse(any(c.kwargs.get("use_pairing") for c in service.call_args_list))

    def test_fasta_results_are_named_after_their_records(self):
        from click.testing import CliRunner

        from mol.runner import batch_inference

        with tempfile.TemporaryDirectory() as d:
            fasta = os.path.join(d, "in.fasta")
            with open(fasta, "w") as f:
                f.write(">B|protein\nMKCCC\n>A|protein\nMKAAA\n>A|protein\nMKGGG\n")
            out = os.path.join(d, "out")
            with patch.object(colab_request_parser, "run_mmseqs2_service",
                              side_effect=fake_service):
                result = CliRunner().invoke(batch_inference.mol_cli,
                                            ["msa", "-i", fasta, "-o", out])
            self.assertEqual(result.exit_code, 0, result.output)
            # Only the named result directories and the index remain.
            self.assertEqual(sorted(os.listdir(out)),
                             ["A_protein", "A_protein_2", "B_protein", "msa_index.json"])
            with open(os.path.join(out, "msa_index.json")) as f:
                index = json.load(f)
            self.assertEqual([e["id"] for e in index], ["B|protein", "A|protein", "A|protein"])
            self.assertEqual([e["sequence"] for e in index], ["MKCCC", "MKAAA", "MKGGG"])
            self.assertEqual([os.path.basename(e["dir"]) for e in index],
                             ["B_protein", "A_protein", "A_protein_2"])
            for entry in index:
                with open(os.path.join(entry["dir"], "non_pairing.a3m")) as f:
                    self.assertIn(entry["sequence"], f.read())
                self.assertTrue(os.path.isfile(os.path.join(entry["dir"], "pairing.a3m")))

    def test_dir_names_are_safe_and_unique(self):
        from mol.runner.batch_inference import msa_dir_names

        self.assertEqual(msa_dir_names(["sp|P1|X Y", "a/b", "", "msa_index.json", "a/b"]),
                         ["sp_P1_X_Y", "a_b", "seq_2", "msa_index.json_2", "a_b_2"])


if __name__ == "__main__":
    unittest.main()
