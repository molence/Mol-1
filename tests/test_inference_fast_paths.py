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

    def test_triangle_attention_fp32_tf32_kernel(self):
        """FP32 with TF32 enabled (the confidence head): the TMA attention
        kernel with the mask and gate fused, both directions."""
        import mol.model.modules.triton_attention as ta
        import mol.model.triangular.triangular as tri

        n = 136
        used = []
        orig = ta.gated_attention
        ta.gated_attention = lambda *a, **k: used.append(orig(*a, **k)) or used[-1]
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            for starting in (True, False):
                mod = _init(tri.TriangleAttention(c_in=64, c_hidden=32, no_heads=4, starting=starting))
                z = torch.randn(n, n, 64, device="cuda")
                mask = self._mask(n, z.dtype)
                call = lambda: mod(z, mask=mask, chunk_size=1024, triangle_attention="cuequivariance")
                with torch.no_grad():
                    tri._FUSED_TRI_ATT = False
                    try:
                        ref = call()
                    finally:
                        tri._FUSED_TRI_ATT = True
                    out = call()
                keep = mask.bool()
                torch.testing.assert_close(out[keep], ref[keep], atol=5e-3, rtol=1e-2)
        finally:
            ta.gated_attention = orig
            torch.backends.cuda.matmul.allow_tf32 = old
        self.assertTrue(used and all(used))

    def _check_tri_kernel(self, kernel, other):
        """BF16 on Blackwell: `kernel` (bias on the tensor cores, gate fused),
        both directions, against the cuEquivariance path, with `other`
        disabled; a mask that is not padding falls back."""
        import mol.model.triangular.triangular as tri

        n = 136
        used = []
        orig = kernel.attention
        kernel.attention = lambda *a, **k: used.append(orig(*a, **k)) or used[-1]
        other._ENABLED = False
        try:
            for starting in (True, False):
                mod = _init(tri.TriangleAttention(c_in=64, c_hidden=32, no_heads=4, starting=starting))
                z = torch.randn(n, n, 64, device="cuda", dtype=torch.bfloat16)
                mask = self._mask(n, z.dtype)
                call = lambda: mod(z, mask=mask, chunk_size=1024, triangle_attention="cuequivariance")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    kernel._ENABLED = False
                    try:
                        ref = call().float()
                    finally:
                        kernel._ENABLED = True
                    out = call().float()
                keep = mask.bool()
                torch.testing.assert_close(out[keep], ref[keep], atol=2e-2, rtol=2e-2)
            self.assertTrue(used and all(used))
            used.clear()
            mask = (torch.rand(n, n, device="cuda") > 0.1).to(torch.bfloat16)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mod(z, mask=mask, chunk_size=1024, triangle_attention="cuequivariance")
            self.assertFalse(used)
        finally:
            kernel.attention = orig
            other._ENABLED = True

    def test_triangle_attention_bf16_tri_fmha(self):
        import mol.model.triangular.tri_fa4 as fa4
        import mol.model.triangular.tri_fmha as tf

        if torch.cuda.get_device_capability()[0] != 10 or tf._cutlass_dir() is None:
            self.skipTest("needs Blackwell and CUTLASS headers")
        self._check_tri_kernel(tf, fa4)

    def test_triangle_attention_bf16_tri_fa4(self):
        import mol.model.triangular.tri_fa4 as fa4
        import mol.model.triangular.tri_fmha as tf

        if torch.cuda.get_device_capability()[0] != 10 or not fa4._load():
            self.skipTest("needs Blackwell and the CuTe DSL with flash-attn-4")
        self._check_tri_kernel(fa4, tf)

    def test_key_prefix(self):
        import mol.model.triangular.triangular as tri

        n = 16
        valid = torch.arange(n, device="cuda") < 13
        mask = (valid[:, None] & valid[None, :]).float()
        self.assertEqual(tri._key_prefix(mask, False), 13)
        self.assertEqual(tri._key_prefix(mask, True), 13)
        mask[2, 5] = 0  # not separable
        self.assertIsNone(tri._key_prefix(mask, False))
        holes = valid.clone()
        holes[4] = False  # not a prefix
        self.assertIsNone(tri._key_prefix((holes[:, None] & holes[None, :]).float(), False))

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
        self._fast_trimul(torch.bfloat16, atol=5e-2, rtol=2e-2)

    def test_fast_trimul_fp32_matches_cuequivariance(self):
        # strict FP32 on both sides (the module takes this path in FP32 only with TF32)
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            self._fast_trimul(torch.float32, atol=1e-4, rtol=1e-4)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def _fast_trimul(self, dt, atol, rtol):
        from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update

        from mol.model.triangular.fast_trimul import triangle_multiplicative_update as fast

        n, c = 64, 64
        torch.manual_seed(0)
        x = torch.randn(n, n, c, device="cuda", dtype=dt)
        res = torch.randn_like(x)
        mask = self._mask(n, dt)
        w = lambda *s: (torch.randn(*s, device="cuda") / s[-1] ** 0.5).to(dt)
        niw, nib, now, nob = w(c) + 1, w(c), w(c) + 1, w(c)
        p_in, g_in, p_out, g_out = w(2 * c, c), w(2 * c, c), w(c, c), w(c, c)
        from mol.model.triangular.fast_trimul import interleave_gate_weight

        w_in = torch.cat([g_in, p_in])
        for direction, w_il in (("outgoing", None), ("incoming", None), ("outgoing", interleave_gate_weight(w_in)),
                                ("incoming", interleave_gate_weight(w_in))):
            with torch.no_grad():
                ref = triangle_multiplicative_update(
                    x[None], direction=direction, mask=mask[None], norm_in_weight=niw,
                    norm_in_bias=nib, p_in_weight=p_in, g_in_weight=g_in,
                    norm_out_weight=now, norm_out_bias=nob, p_out_weight=p_out,
                    g_out_weight=g_out, eps=1e-5,
                )[0].float()
                out = fast(x, direction, mask, niw, nib, w_in, now, nob,
                           p_out, g_out, 1e-5, residual=res, in_weight_interleaved=w_il).float()
            keep = mask.bool()
            torch.testing.assert_close(out[keep], (ref + res.float())[keep], atol=atol, rtol=rtol)

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
        import mol.model.modules.triton_attention as ta

        torch.manual_seed(0)
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            # (tokens, bias shape): the diffusion transformer layout (strided
            # heads, bias broadcast over samples), a bias per sample, and a key
            # count whose bias rows are not 16-byte aligned (no TMA).
            for n, bias_lead in ((200, (1, 4)), (200, (3, 4)), (198, (1, 4))):
                q, k, v = (torch.randn(3, n, 4, 48, device="cuda").transpose(1, 2) * 0.4 for _ in range(3))
                bias = torch.randn(*bias_lead, n, n, device="cuda") * 2
                bias[..., 150:] = -1e9
                for tma in (True, False):
                    ta._TMA = tma
                    with torch.no_grad():
                        self.assertTrue(ta.tf32_attention_ok(q, k, v, bias))
                        out = ta.tf32_attention(q, k, v, bias)
                        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
                    torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
            ta._TMA = True


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
class OuterProductMeanTest(unittest.TestCase):
    def test_swap12_matches_permute(self):
        from mol.model.triangular.triton_permute import swap12

        x = torch.randn(37, 32, 45, 32, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.equal(swap12(x), x.permute(0, 2, 1, 3).contiguous()))

    def test_opm_matches_reference(self):
        from mol.model.triangular.layers import OuterProductMean

        mod = _init(OuterProductMean(c_m=16, c_z=24, c_hidden=8))
        a = torch.randn(30, 20, 8, device="cuda")
        b = torch.randn(30, 20, 8, device="cuda")
        with torch.no_grad():
            out = mod._opm(a, b)
            ref = mod.linear_out(torch.einsum("...bac,...dae->...bdce", a, b).reshape(30, 30, 64))
        torch.testing.assert_close(out, ref)


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


@unittest.skipUnless(CUDA, "needs CUDA")
class FusedDiffusionTransformerTest(unittest.TestCase):
    def _compare(self, n, local, tf32, s_shape=None, cross=False, c_a=64):
        import mol.model.modules.diffusion_fast as dfast
        from mol.model.modules.transformer import DiffusionTransformer

        torch.manual_seed(0)
        dt = DiffusionTransformer(c_a=c_a, c_s=32, c_z=16, n_blocks=3, n_heads=4, cross_attention_mode=cross).cuda().eval()
        with torch.no_grad():
            for p in dt.parameters():  # the zero-initialized gates and outputs too
                p.copy_(torch.randn_like(p) * 0.2 + (1.0 if p.dim() == 1 and "weight" in str(p.shape) else 0.0))
        a = torch.randn(3, n, c_a, device="cuda")
        s = torch.randn(*(s_shape or (3,)), n, 32, device="cuda")
        a0 = a.clone()
        kwargs = {}
        if local:
            z = torch.randn(1, (n + 31) // 32, 32, 128, 16, device="cuda")
            kwargs = dict(n_queries=32, n_keys=128)
        else:
            z = torch.randn(1, n, n, 16, device="cuda")  # leading batch dim, as in sampling
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            with torch.no_grad():
                if not local:
                    kwargs = dict(projected_pair_biases=dt.prepare_pair_biases(z))
                self.assertTrue(dfast.fused_ok(dt, a, s))
                out = dt(a, s, z, **kwargs)
                dfast._ENABLED = False
                try:
                    ref = dt(a, s, z, **kwargs)
                finally:
                    dfast._ENABLED = True
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
        torch.testing.assert_close(a, a0, rtol=0, atol=0)  # the input is not updated in place
        tol = 2e-2 if tf32 else 1e-4
        torch.testing.assert_close(out, ref, atol=tol * ref.abs().max().item(), rtol=tol)

    def test_global_attention(self):
        # 160 tokens: TMA attention with the gate fused; 150: bias rows not
        # 16-byte aligned, SDPA and a separate gate.
        for n in (160, 150):
            for tf32 in (True, False):
                with self.subTest(n=n, tf32=tf32):
                    self._compare(n, local=False, tf32=tf32)

    def test_global_attention_fa4(self):
        """With TF32: the token attention in FP16 on FlashAttention-4 (bias
        and gate fused); head dims 16 and 48, 150 tokens (keys padded for
        TMA and masked), against the unfused FP32 blocks and the Triton
        kernel."""
        import mol.model.modules.fa4_attention as fa

        if torch.cuda.get_device_capability()[0] != 10 or fa._kernel(48) is None:
            self.skipTest("needs Blackwell and the CuTe DSL with flash-attn-4")
        used = []
        orig = fa.attention
        fa.attention = lambda *a, **k: used.append(orig(*a, **k)) or used[-1]
        try:
            for n in (160, 150):
                for c_a in (64, 192):
                    with self.subTest(n=n, c_a=c_a):
                        self._compare(n, local=False, tf32=True, c_a=c_a)
                        self.assertTrue(used and all(used))
                        used.clear()
                        fa._ENABLED = False
                        try:
                            self._compare(n, local=False, tf32=True, c_a=c_a)
                        finally:
                            fa._ENABLED = True
                        self.assertFalse(used)
        finally:
            fa.attention = orig

    def test_local_attention(self):
        for tf32 in (True, False):
            with self.subTest(tf32=tf32):
                self._compare(200, local=True, tf32=tf32)

    def test_shared_conditioning(self):
        # s shared by the samples, as in the atom transformers
        for s_shape in ((), (1,)):
            for local in (True, False):
                with self.subTest(s_shape=s_shape, local=local):
                    self._compare(160, local=local, tf32=True, s_shape=s_shape)

    def test_cross_attention_mode(self):
        # the atom transformers: keys/values from their own AdaLN, s shared
        for local in (True, False):
            for tf32 in (True, False):
                with self.subTest(local=local, tf32=tf32):
                    self._compare(160, local=local, tf32=tf32, s_shape=(1,), cross=True)



@unittest.skipUnless(CUDA and _cueq_available(), "needs CUDA and cuEquivariance")
class TemplateDedupTest(unittest.TestCase):
    def test_matches_loop_over_templates(self):
        """Identical templates run once and the projection is reused across
        cycles; the output is bit-identical to the reference loop."""
        from mol.model.modules.pairformer import TemplateEmbedder

        n, t = 40, 4
        te = _init(TemplateEmbedder(n_blocks=2, c=64, c_z=128, dropout=0.0), std=0.02)
        g = torch.Generator(device="cuda").manual_seed(0)
        # As the featurizer builds them without templates: slot 0 all gaps,
        # slots 1-3 zero padding, every mask and geometric feature zero.
        feats = {
            "asym_id": torch.tensor([0] * 25 + [1] * 15, device="cuda"),
            "template_aatype": torch.zeros(t, n, dtype=torch.long, device="cuda"),
            "template_distogram": torch.zeros(t, n, n, 39, device="cuda"),
            "template_pseudo_beta_mask": torch.zeros(t, n, n, device="cuda"),
            "template_unit_vector": torch.zeros(t, n, n, 3, device="cuda"),
            "template_backbone_frame_mask": torch.zeros(t, n, n, device="cuda"),
        }
        feats["template_aatype"][0] = 31
        with_hit = dict(feats)  # one real template in slot 2
        for k in ("template_distogram", "template_pseudo_beta_mask"):
            with_hit[k] = feats[k].clone()
            with_hit[k][2] = torch.rand(with_hit[k][2].shape, device="cuda", generator=g).round()
        kwargs = dict(triangle_multiplicative="cuequivariance",
                      triangle_attention="cuequivariance", inplace_safe=True)
        for ifd, groups in ((feats, [0, 1, 1, 1]), (with_hit, [0, 1, 2, 1])):
            cache = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(2):  # second call reuses the cached projections
                    z = torch.randn(n, n, 128, device="cuda", generator=g)
                    ref = te(ifd, z.clone(), **kwargs)
                    out = te(ifd, z.clone(), cycle_cache=cache, **kwargs)
                    self.assertEqual(cache["rep"], groups)
                    self.assertTrue(torch.equal(out, ref))

if __name__ == "__main__":
    unittest.main()
