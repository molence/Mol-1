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

"""Tests for the Apple MPS triangle-attention and Pairformer backends."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from mol.model.mol import Mol
from mol.model.triangular import mps_attention
from mol.model.triangular.layers import _attention
from mol.model.triangular.mps_attention import mps_triangle_attention
from mol.runner.batch_inference import predict

HAS_MPS = torch.backends.mps.is_available()


def _qkv(outer, length, heads, dim=32):
    # Match Attention._prep_qkv: physical [I,N,H,D] storage, [I,H,N,D] view.
    q, k, v = (
        torch.randn(outer, length, heads, dim, device="mps").transpose(1, 2)
        for _ in range(3)
    )
    return q / dim**0.5, k, v


class TestPairformerBackendFlag(unittest.TestCase):
    def _backend(self, backend, device="cpu", training=False):
        stub = SimpleNamespace(
            configs=SimpleNamespace(pairformer_backend=backend),
            training=training,
            _device=lambda: torch.device(device),
        )
        return Mol._pairformer_backend(stub)

    def test_cli_default_is_torch(self):
        option = next(p for p in predict.params if p.name == "pairformer_backend")
        self.assertEqual(option.default, "torch")

    def test_torch_is_accepted_everywhere(self):
        self.assertEqual(self._backend("torch"), "torch")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self._backend("cuda")

    def test_mlx_requires_mps_inference(self):
        with self.assertRaises(RuntimeError):
            self._backend("mlx", device="cpu")
        with self.assertRaises(RuntimeError):
            self._backend("mlx", device="mps", training=True)
        self.assertEqual(self._backend("mlx", device="mps"), "mlx")


@unittest.skipUnless(HAS_MPS, "Apple MPS required")
class TestMpsTriangleAttention(unittest.TestCase):
    def test_masked_kernel_matches_torch(self):
        torch.manual_seed(7)
        for length in (1, 15, 16, 17, 31, 32, 33, 128):
            with self.subTest(length=length), torch.inference_mode():
                heads = 4
                q, k, v = _qkv(2, length, heads)
                mask = torch.zeros(2, 1, 1, length, device="mps")
                if length > 1:
                    mask[0, ..., -1] = -1e9
                triangle = torch.randn(1, heads, length, length, device="mps")

                expected = _attention(q, k, v, [mask, triangle]).transpose(1, 2)
                actual = mps_triangle_attention(q, k, v, [mask, triangle])

                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_unmasked_sdpa_matches_torch(self):
        with torch.inference_mode():
            q, k, v = _qkv(2, 33, 4)
            mask = torch.zeros(2, 1, 1, 33, device="mps")
            triangle = torch.randn(1, 4, 33, 33, device="mps")
            expected = _attention(q, k, v, [mask, triangle]).transpose(1, 2)
            actual = mps_triangle_attention(q, k, v, [triangle])
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_chunked_sdpa_matches_torch(self):
        with torch.inference_mode(), mock.patch.object(
            mps_attention, "_SDPA_MAX_SCORE_ELEMENTS", 4 * 33 * 33
        ):
            q, k, v = _qkv(7, 33, 4)
            mask = torch.zeros(7, 1, 1, 33, device="mps")
            triangle = torch.randn(1, 4, 33, 33, device="mps")
            expected = _attention(q, k, v, [mask, triangle]).transpose(1, 2)
            actual = mps_triangle_attention(q, k, v, [triangle])
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_rejects_autograd(self):
        q, k, v = _qkv(1, 4, 4)
        q = q.detach().requires_grad_()
        triangle = torch.zeros(1, 4, 4, 4, device="mps")
        with self.assertRaisesRegex(RuntimeError, "inference only"):
            mps_triangle_attention(q, k, v, [triangle])


@unittest.skipUnless(
    HAS_MPS and importlib.util.find_spec("mlx"), "Apple MPS and MLX required"
)
class TestMlxPairformer(unittest.TestCase):
    def test_matches_torch_stack(self):
        from mol.model.modules.pairformer import PairformerStack
        from mol.model.triangular.mlx_pairformer import run_mlx_pairformer_stack

        torch.manual_seed(0)
        stack = PairformerStack(
            n_blocks=2, c_z=32, c_s=64, n_heads=4, n_pairmixer_blocks=1
        ).to("mps").eval()
        s = torch.randn(24, 64, device="mps")
        z = torch.randn(24, 24, 32, device="mps")
        with torch.inference_mode():
            s_ref, z_ref = stack(s, z, pair_mask=None)
            s_out, z_out = run_mlx_pairformer_stack(stack, s, z)
        torch.testing.assert_close(s_out, s_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(z_out, z_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
