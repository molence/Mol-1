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

"""Inference fast paths agree with the reference implementations they replace."""

import unittest

import torch

CUDA = torch.cuda.is_available()


def _cueq_available() -> bool:
    try:
        import cuequivariance_torch.primitives.triangle  # noqa: F401
    except Exception:
        return False
    return True


def _init(module: torch.nn.Module, std: float = 0.05) -> torch.nn.Module:
    torch.manual_seed(0)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=std)
    return module.cuda().eval()


@unittest.skipUnless(CUDA, "needs CUDA")
class TritonLayerNormTest(unittest.TestCase):
    def test_matches_extension(self):
        from mol.model.layer_norm import triton_layer_norm as tln
        from mol.model.triangular.layers import LayerNorm

        for dtype, atol in ((torch.bfloat16, 3e-2), (torch.float32, 1e-5)):
            ln = _init(LayerNorm(320), std=0.5)
            x = torch.randn(70000, 320, device="cuda", dtype=dtype)
            with torch.no_grad():
                self.assertTrue(tln.enabled(x))
                out = tln.layer_norm(x, ln.weight, ln.bias, ln.eps)
                ref = torch.nn.functional.layer_norm(
                    x.float(), (320,), ln.weight.to(dtype).float(), ln.bias.to(dtype).float(), ln.eps
                )
            torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)


@unittest.skipUnless(CUDA and _cueq_available(), "needs CUDA and cuEquivariance")
class TriangleFastPathTest(unittest.TestCase):
    def _mask(self, n, dtype):
        mask = torch.ones(n, n, device="cuda", dtype=dtype)
        mask[:, -3:] = 0
        mask[-3:, :] = 0
        return mask

    def test_triangle_attention_fused_matches_mha_path(self):
        import mol.model.triangular.triangular as tri

        n = 72
        for starting in (True, False):
            mod = _init(tri.TriangleAttention(c_in=64, c_hidden=32, no_heads=2, starting=starting))
            z = torch.randn(n, n, 64, device="cuda", dtype=torch.bfloat16)
            mask = self._mask(n, z.dtype)
            call = lambda: mod(z, mask=mask, chunk_size=1024, triangle_attention="cuequivariance")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tri._FUSED_TRI_ATT = False
                try:
                    ref = call().float()
                finally:
                    tri._FUSED_TRI_ATT = True
                out = call().float()
            keep = mask.bool()
            torch.testing.assert_close(out[keep], ref[keep], atol=2e-3, rtol=2e-2)

    def test_triangle_attention_residual_update_matches_transposed_path(self):
        """Ending-node attention on the untransposed pair tensor, residual fused."""
        import mol.model.triangular.triangular as tri

        n = 72
        mod = _init(tri.TriangleAttention(c_in=64, c_hidden=32, no_heads=2))
        z = torch.randn(n, n, 64, device="cuda", dtype=torch.bfloat16)
        mask = self._mask(n, z.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ending in (False, True):
                zz, mm = (z.transpose(0, 1).contiguous(), mask.t()) if ending else (z, mask)
                ref = zz + mod(zz, mask=mm, triangle_attention="cuequivariance")
                if ending:
                    ref = ref.transpose(0, 1)
                out = mod.fused_residual_update(z, mask, "cuequivariance", None, ending=ending)
                keep = mask.bool()
                torch.testing.assert_close(out.float()[keep], ref.float()[keep], atol=2e-2, rtol=2e-2)

    def test_fast_trimul_matches_cuequivariance(self):
        from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update

        from mol.model.triangular.fast_trimul import triangle_multiplicative_update as fast

        n, c, dt = 64, 64, torch.bfloat16
        torch.manual_seed(0)
        x = torch.randn(n, n, c, device="cuda", dtype=dt)
        res = torch.randn_like(x)
        mask = self._mask(n, dt)
        w = lambda *s: (torch.randn(*s, device="cuda") / s[-1] ** 0.5).to(dt)
        niw, nib, now, nob = w(c) + 1, w(c), w(c) + 1, w(c)
        p_in, g_in, p_out, g_out = w(2 * c, c), w(2 * c, c), w(c, c), w(c, c)
        for direction in ("outgoing", "incoming"):
            with torch.no_grad():
                ref = triangle_multiplicative_update(
                    x[None], direction=direction, mask=mask[None], norm_in_weight=niw,
                    norm_in_bias=nib, p_in_weight=p_in, g_in_weight=g_in,
                    norm_out_weight=now, norm_out_bias=nob, p_out_weight=p_out,
                    g_out_weight=g_out, eps=1e-5,
                )[0].float()
                out = fast(x, direction, mask, niw, nib, torch.cat([g_in, p_in]), now, nob,
                           p_out, g_out, 1e-5, residual=res).float()
            keep = mask.bool()
            torch.testing.assert_close(out[keep], (ref + res.float())[keep], atol=5e-2, rtol=2e-2)

    def test_pairformer_stack_padding_is_transparent(self):
        """Padding the pair tensor to a multiple of 8 leaves real tokens unchanged."""
        import mol.model.triangular.triangular as tri
        from mol.model.modules.pairformer import PairformerStack

        n = 61
        stack = _init(PairformerStack(n_blocks=1, n_heads=4, c_z=64, c_s=32, dropout=0.0), std=0.02)
        s = torch.randn(n, 32, device="cuda")
        z = torch.randn(n, n, 64, device="cuda")
        kwargs = dict(pair_mask=None, triangle_multiplicative="cuequivariance",
                      triangle_attention="cuequivariance", inplace_safe=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            old = tri._TRI_PAD_MULTIPLE
            tri._TRI_PAD_MULTIPLE = 0
            try:
                s_ref, z_ref = stack(s.clone(), z.clone(), **kwargs)
            finally:
                tri._TRI_PAD_MULTIPLE = old
            s_out, z_out = stack(s.clone(), z.clone(), **kwargs)
        self.assertEqual(z_out.shape, z.shape)
        torch.testing.assert_close(z_out.float(), z_ref.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(s_out.float(), s_ref.float(), atol=5e-2, rtol=5e-2)


@unittest.skipUnless(CUDA, "needs CUDA")
class TransitionAndPairBiasFastPathTest(unittest.TestCase):
    def test_fused_transition_matches_reference(self):
        import mol.model.modules.primitives as prim

        mod = _init(prim.Transition(c_in=64, n=4))
        x = torch.randn(300, 300, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertTrue(mod._fused_eval_ok(x))
            out = mod(x, chunk_size=64).float()
            prim._FUSED_TRANSITION = False
            try:
                ref = mod(x, chunk_size=64).float()
            finally:
                prim._FUSED_TRANSITION = True
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)

    def test_fused_pair_bias_matches_reference(self):
        import mol.model.modules.transformer as tr

        apb = _init(tr.AttentionPairBias(has_s=False, n_heads=16, c_a=64, c_z=64), std=0.1)
        zp = torch.randn(262, 262, 64, device="cuda", dtype=torch.bfloat16)
        z = zp[:260, :260]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = apb.project_pair_bias(z).float()
            tr._FUSED_PAIR_BIAS = False
            try:
                ref = apb.project_pair_bias(z).float()
            finally:
                tr._FUSED_PAIR_BIAS = True
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(CUDA, "needs CUDA")
class TF32AttentionTest(unittest.TestCase):
    def test_matches_fp32_sdpa(self):
        from mol.model.modules.triton_attention import tf32_attention, tf32_attention_ok

        torch.manual_seed(0)
        # Diffusion transformer layout: strided heads, bias broadcast over samples.
        q, k, v = (torch.randn(3, 200, 4, 48, device="cuda").transpose(1, 2) * 0.4 for _ in range(3))
        bias = torch.randn(1, 4, 200, 200, device="cuda") * 2
        bias[..., 150:] = -1e9
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with torch.no_grad():
                self.assertTrue(tf32_attention_ok(q, k, v, bias))
                out = tf32_attention(q, k, v, bias)
                ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)


    def test_local_attention_matches_trunked_reference(self):
        import mol.model.modules.primitives as prim
        import mol.model.modules.triton_attention as ta

        torch.manual_seed(0)
        n, d = 300, 32  # not a multiple of the 32-query window
        q, k, v = (torch.randn(2, 4, n, d, device="cuda") * 0.5 for _ in range(3))
        n_trunks = (n + 31) // 32
        bias = torch.randn(4, n_trunks, 32, 128, device="cuda")  # broadcast over the batch
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            for tf32, atol in ((False, 1e-5), (True, 5e-3)):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                with torch.no_grad():
                    self.assertEqual(ta.local_attention_ok(q, k, v, None, bias, 32, 128), tf32)
                    # Exercise the kernel in full FP32 too, though only used with TF32.
                    out = ta.local_attention(q, k, v, bias, 32, 128)
                    ta._ENABLED = False
                    try:
                        ref = prim._local_attention(q, k, v, 32, 128, trunked_attn_bias=bias)
                    finally:
                        ta._ENABLED = True
                torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


@unittest.skipUnless(CUDA, "needs CUDA")
class GraphedDenoiseStepTest(unittest.TestCase):
    def test_replay_matches_eager(self):
        from mol.model.cuda_graph import GraphedDenoiseStep
        from mol.model.triangular.layers import LayerNorm

        ln = _init(LayerNorm(16))
        lin = _init(torch.nn.Linear(16, 3))
        feats = torch.randn(50, 16, device="cuda")

        def net(x_noisy, t_hat_noise_level, feats):
            return lin(ln(feats + x_noisy.sum(-1, keepdim=True))) * t_hat_noise_level[..., None, None] + x_noisy

        step = GraphedDenoiseStep(net, feats=feats)
        with torch.no_grad():
            for i in range(4):
                x = torch.randn(2, 50, 3, device="cuda")
                t = torch.full((2,), float(i + 1), device="cuda")
                torch.testing.assert_close(step(x, t), net(x, t, feats), atol=1e-5, rtol=1e-5)
        self.assertIsNotNone(step.graph)


if __name__ == "__main__":
    unittest.main()
