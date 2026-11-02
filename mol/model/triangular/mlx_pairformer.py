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

"""Inference-only MLX implementation of the Pairformer stack for Apple silicon."""

from __future__ import annotations

import weakref

import torch

try:
    import mlx.core as mx
except ImportError as exc:
    raise ImportError(
        "pairformer_backend='mlx' requires MLX: install the `mlx` extra"
    ) from exc

_LAYER_NORM_EPS = 1e-5
_CACHE_LIMIT_BYTES = 1 << 30


def _linear(x, weight):
    return mx.matmul(x, mx.swapaxes(weight, -1, -2))


def _layer_norm(x, weights, prefix):
    return mx.fast.layer_norm(
        x, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"], _LAYER_NORM_EPS
    )


def _heads(x, heads):
    return mx.transpose(mx.reshape(x, (*x.shape[:-1], heads, -1)), (0, 2, 1, 3))


def _triangle_multiplication(z, weights, prefix, outgoing):
    norm = _layer_norm(z, weights, f"{prefix}.layer_norm_in")

    def projection(name):
        gate = mx.sigmoid(_linear(norm, weights[f"{prefix}.linear_{name}_g.weight"]))
        return gate * _linear(norm, weights[f"{prefix}.linear_{name}_p.weight"])

    a, b = projection("a"), projection("b")
    if outgoing:
        product = mx.matmul(mx.transpose(a, (2, 0, 1)), mx.transpose(b, (2, 1, 0)))
    else:
        product = mx.matmul(mx.transpose(a, (2, 1, 0)), mx.transpose(b, (2, 0, 1)))
    product = _layer_norm(mx.transpose(product, (1, 2, 0)), weights, f"{prefix}.layer_norm_out")
    gate = mx.sigmoid(_linear(norm, weights[f"{prefix}.linear_g.weight"]))
    return _linear(product, weights[f"{prefix}.linear_z.weight"]) * gate


def _triangle_attention(z, weights, prefix):
    norm = _layer_norm(z, weights, f"{prefix}.layer_norm")
    bias_weight = weights[f"{prefix}.linear.weight"]
    heads = bias_weight.shape[0]
    head_dim = weights[f"{prefix}.mha.linear_q.weight"].shape[0] // heads
    bias = mx.transpose(_linear(norm, bias_weight), (2, 0, 1))[None]
    q, k, v = (
        _heads(_linear(norm, weights[f"{prefix}.mha.linear_{name}.weight"]), heads)
        for name in "qkv"
    )
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=bias)
    gate = mx.sigmoid(_linear(norm, weights[f"{prefix}.mha.linear_g.weight"]))
    attended = mx.transpose(attended, (0, 2, 1, 3)) * mx.reshape(gate, (*z.shape[:2], heads, head_dim))
    attended = mx.reshape(attended, (*z.shape[:2], heads * head_dim))
    return _linear(attended, weights[f"{prefix}.mha.linear_o.weight"])


def _transition(x, weights, prefix):
    norm = _layer_norm(x, weights, f"{prefix}.layernorm1")
    a = _linear(norm, weights[f"{prefix}.linear_no_bias_a.weight"])
    b = _linear(norm, weights[f"{prefix}.linear_no_bias_b.weight"])
    return _linear(a * mx.sigmoid(a) * b, weights[f"{prefix}.linear_no_bias.weight"])


def _single_attention(s, z, weights):
    prefix = "attention_pair_bias"
    s_norm = _layer_norm(s, weights, f"{prefix}.layernorm_a")
    z_norm = _layer_norm(z, weights, f"{prefix}.layernorm_z")
    bias_weight = weights[f"{prefix}.linear_nobias_z.weight"]
    heads = bias_weight.shape[0]
    head_dim = weights[f"{prefix}.attention.linear_q.weight"].shape[0] // heads
    bias = mx.transpose(_linear(z_norm, bias_weight), (2, 0, 1))[None]
    q = _linear(s_norm, weights[f"{prefix}.attention.linear_q.weight"])
    q = q + weights[f"{prefix}.attention.linear_q.bias"]
    k = _linear(s_norm, weights[f"{prefix}.attention.linear_k.weight"])
    v = _linear(s_norm, weights[f"{prefix}.attention.linear_v.weight"])
    q, k, v = (_heads(x[None], heads) for x in (q, k, v))
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=bias)
    attended = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)), (s.shape[0], heads * head_dim))
    gate = mx.sigmoid(_linear(s_norm, weights[f"{prefix}.attention.linear_g.weight"]))
    return _linear(attended * gate, weights[f"{prefix}.attention.linear_o.weight"])


def _pair_mixer_block(s, z, weights):
    z = z + _triangle_multiplication(z, weights, "tri_mul_out", True)
    z = z + _triangle_multiplication(z, weights, "tri_mul_in", False)
    z = z + _transition(z, weights, "pair_transition")
    s = s + _single_attention(s, z, weights)
    return s + _transition(s, weights, "single_transition"), z


def _pairformer_block(s, z, weights):
    z = z + _triangle_multiplication(z, weights, "tri_mul_out", True)
    z = z + _triangle_multiplication(z, weights, "tri_mul_in", False)
    z = z + _triangle_attention(z, weights, "tri_att_start")
    z = z + mx.transpose(
        _triangle_attention(mx.transpose(z, (1, 0, 2)), weights, "tri_att_end"), (1, 0, 2)
    )
    z = z + _transition(z, weights, "pair_transition")
    s = s + _single_attention(s, z, weights)
    return s + _transition(s, weights, "single_transition"), z


def _to_mlx(tensor: torch.Tensor):
    return mx.from_dlpack(tensor.detach().contiguous())


def _adapter_weight(stack, name):
    weight = stack.adapters[name].state_dict().get("pair_adapter.weight")
    return None if weight is None else _to_mlx(weight)


class MLXPairformerRunner:
    """MLX copy of one loaded PairformerStack's weights."""

    def __init__(self, stack: torch.nn.Module) -> None:
        if next(stack.parameters()).device.type != "mps":
            raise RuntimeError("the MLX Pairformer backend requires an MPS model")
        torch.mps.synchronize()
        mx.set_cache_limit(_CACHE_LIMIT_BYTES)
        self.n_pairmixer_blocks = stack.n_pairmixer_blocks
        self.blocks = [
            {name: _to_mlx(value) for name, value in block.state_dict().items()}
            for block in stack.blocks
        ]
        self.adapters_before = {
            index: _adapter_weight(stack, name)
            for index, name in stack._adapter_before_block.items()
        }
        output_adapter = stack._output_adapter_name
        self.adapter_after = (
            None if output_adapter is None else _adapter_weight(stack, output_adapter)
        )
        mx.eval(self.blocks, self.adapters_before, self.adapter_after)
        mx.synchronize()
        self.pair_mixer = mx.compile(_pair_mixer_block)
        self.pairformer = mx.compile(_pairformer_block)

    def __call__(self, s: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if s.device.type != "mps" or z.device.type != "mps":
            raise RuntimeError("the MLX Pairformer backend requires MPS inputs")
        if torch.is_grad_enabled():
            raise RuntimeError("the MLX Pairformer backend supports inference only")
        if s.ndim != 2 or z.ndim != 3:
            raise RuntimeError("the MLX Pairformer backend requires unbatched inputs")

        torch.mps.synchronize()
        s_mlx, z_mlx = _to_mlx(s), _to_mlx(z)
        for index, weights in enumerate(self.blocks):
            adapter = self.adapters_before.get(index)
            if adapter is not None:
                z_mlx = _linear(z_mlx, adapter)
            block = self.pair_mixer if index < self.n_pairmixer_blocks else self.pairformer
            s_mlx, z_mlx = block(s_mlx, z_mlx, weights)
            # Evaluating per block lets MLX recycle the large pair intermediates.
            mx.eval(s_mlx, z_mlx)
        if self.adapter_after is not None:
            z_mlx = _linear(z_mlx, self.adapter_after)
        mx.eval(s_mlx, z_mlx)
        mx.synchronize()
        # Copy out of MLX-owned buffers before they are released.
        s_out, z_out = torch.from_dlpack(s_mlx).clone(), torch.from_dlpack(z_mlx).clone()
        mx.clear_cache()
        return s_out, z_out


_runners: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def run_mlx_pairformer_stack(
    stack: torch.nn.Module, s: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if stack not in _runners:
        _runners[stack] = MLXPairformerRunner(stack)
    return _runners[stack](s, z)
