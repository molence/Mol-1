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

"""Constraint feature conflicts and empty encoder behavior."""
import itertools
import unittest

import torch

from mol.constraint.schema import ContactMask, TokenSiteConstraintBatch
from mol.constraint.encoders import ContactMaskEncoder
from mol.constraint.features import attach_site_constraint_features, exclusion_attach


class ConstraintFeatureTest(unittest.TestCase):


    def test_site_exclusion_conflicts_use_per_sample_partner_intersection(self):
        for shared_residue, shared_partner in itertools.product((False, True), repeat=2):
            with self.subTest(residue=shared_residue, partner=shared_partner):
                positive = TokenSiteConstraintBatch(
                    torch.tensor(True), torch.ones(4, dtype=torch.bool),
                    torch.tensor([False, False, True, False]),
                    torch.tensor([True, False, False, False]))
                features = attach_site_constraint_features({'token_index': torch.arange(4)}, positive)
                selected = torch.tensor([shared_residue, not shared_residue, False, False])
                partner = torch.tensor([False, False, shared_partner, not shared_partner])
                if shared_residue and shared_partner:
                    with self.assertRaisesRegex(ValueError, 'conflict'):
                        exclusion_attach(features, selected, partner)
                else:
                    result = exclusion_attach(features, selected, partner)
                    self.assertTrue(torch.equal(result['exclusion_partner_token_mask'], partner))

        # Intersections in different batch items must not produce a false conflict.
        positive = TokenSiteConstraintBatch(
            torch.ones(2, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool),
            torch.tensor([[False, False, True, False]] * 2),
            torch.tensor([[True, False, False, False]] * 2))
        features = attach_site_constraint_features({}, positive)
        selected = torch.tensor([[True, False, False, False], [False, True, False, False]])
        partner = torch.tensor([[False, False, False, True], [False, False, True, False]])
        exclusion_attach(features, selected, partner)
        selected[1] = selected[0]
        with self.assertRaisesRegex(ValueError, 'conflict'):
            exclusion_attach(features, selected, partner)

    def test_contact_exclusion_conflict_still_rejected(self):
        features = attach_site_constraint_features(
            {'token_index': torch.arange(4)},
            TokenSiteConstraintBatch.null(4, protein_token_mask=torch.ones(4, dtype=torch.bool)))
        features = exclusion_attach(features, torch.tensor([1, 0, 0, 0], dtype=torch.bool),
                                     torch.tensor([0, 0, 0, 1], dtype=torch.bool))
        features.update(ContactMask.from_pairs([(0, 3)], 4).as_features())
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            ContactMask.from_features(features, exclusion_cutoff=6)


    def test_empty_encoders_produce_zero_output(self):
        encoder = ContactMaskEncoder(8)
        with torch.no_grad(): encoder.output_projection.weight.normal_()
        output = encoder(ContactMask.empty(4))
        self.assertEqual(output.shape, (4, 4, 8))
        self.assertEqual(output.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
