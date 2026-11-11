# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
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
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: protenix/model/layer_norm/layer_norm.py
# Changes: Fast LayerNorm is built on first use and cached per PyTorch, CUDA and GPU architecture; tensors off CUDA use PyTorch LayerNorm.

# Copyright 2024 ByteDance and/or its affiliates.
#
# Copyright 2021- HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numbers
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch.nn.parameter import Parameter

from mol.model.layer_norm import triton_layer_norm
from mol.model.layer_norm.build_utils import (
    ExtensionBuildInfo,
    collect_extension_build_info,
)


logger = logging.getLogger(__name__)
current_dir = Path(__file__).resolve().parent
kernel_dir = current_dir / "kernel"
extension_sources = [
    kernel_dir / "layer_norm_cuda.cpp",
    kernel_dir / "layer_norm_cuda_kernel.cu",
]
extension_build_inputs = [
    *extension_sources,
    *sorted(kernel_dir.glob("*.h")),
    current_dir / "torch_ext_compile.py",
]


def _load_fast_layer_norm_extension() -> tuple[Any, ExtensionBuildInfo, str]:
    from torch.utils.cpp_extension import CUDA_HOME

    build_info = collect_extension_build_info(
        torch,
        extension_build_inputs,
        cuda_home=CUDA_HOME,
    )
    logger.info(
        "Fast LayerNorm extension: device=%s (%s), capability=%s, "
        "torch=%s, torch_cuda=%s, nvcc=%s, module=%s",
        build_info.device_index,
        build_info.device_name,
        build_info.capability_tag,
        torch.__version__,
        torch.version.cuda,
        build_info.nvcc_tag,
        build_info.name,
    )

    from mol.model.layer_norm.torch_ext_compile import compile

    try:
        extension = compile(
            name=build_info.name,
            sources=[str(path) for path in extension_sources],
            extra_include_paths=[str(kernel_dir)],
            build_directory=None,
        )
    except (ImportError, OSError, RuntimeError) as compile_error:
        raise RuntimeError(
            "Unable to load or build the Fast LayerNorm CUDA extension for "
            f"{build_info.capability_tag} with torch {torch.__version__} "
            f"(CUDA {torch.version.cuda}). Set LAYERNORM_TYPE=torch to use "
            "the PyTorch LayerNorm implementation."
        ) from compile_error

    build_directory = str(Path(extension.__file__).resolve().parent)
    logger.info("Fast LayerNorm extension cache: %s", build_directory)
    return extension, build_info, build_directory


# Loaded on first use rather than at import: every `mol` subcommand imports the
# model code, and none but prediction and training should wait for a CUDA build.
_extension_state: Optional[tuple[Any, ExtensionBuildInfo, str]] = None

_LAZY_ATTRIBUTES = {
    "fast_layer_norm_cuda": 0,
    "FAST_LAYER_NORM_BUILD_INFO": 1,
    "FAST_LAYER_NORM_BUILD_DIRECTORY": 2,
}


def _fast_layer_norm_extension() -> Any:
    """Return the Fast LayerNorm extension, loading or building it once per process."""
    global _extension_state
    if _extension_state is None:
        _extension_state = _load_fast_layer_norm_extension()
    return _extension_state[0]


def __getattr__(name: str) -> Any:
    # The extension and its build metadata keep their module-level names.
    if name in _LAZY_ATTRIBUTES:
        _fast_layer_norm_extension()
        return _extension_state[_LAZY_ATTRIBUTES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        normalized_shape: torch.Size,
        eps: float,
    ) -> torch.Tensor:
        fast_layer_norm_cuda = _fast_layer_norm_extension()
        d = input.dtype

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()

        if weight is None:
            if bias is None:
                output, mean, invvar = fast_layer_norm_cuda.forward_none_affine(
                    input_, ctx.normalized_shape, ctx.eps
                )
            else:
                output, mean, invvar = fast_layer_norm_cuda.forward_with_bias_affine(
                    input_, ctx.normalized_shape, bias.to(d), ctx.eps
                )
        else:
            if bias is None:
                (
                    output,
                    mean,
                    invvar,
                ) = fast_layer_norm_cuda.forward_with_weight_affine(
                    input_, ctx.normalized_shape, weight.to(d), ctx.eps
                )
            else:
                output, mean, invvar = fast_layer_norm_cuda.forward_with_both_affine(
                    input_,
                    ctx.normalized_shape,
                    weight.to(d),
                    bias.to(d),
                    ctx.eps,
                )
        ctx.save_for_backward(input_, weight, bias, mean, invvar)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        fast_layer_norm_cuda = _fast_layer_norm_extension()
        d = grad_output.dtype
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if weight_ is None:
            if bias_ is None:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda.backward_none_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    ctx.eps,
                )
            else:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda.backward_with_bias_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        else:
            if bias_ is None:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda.backward_with_weight_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    ctx.eps,
                )
            else:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda.backward_with_both_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        return (
            grad_input,
            None if weight_ is None else grad_weight,
            None if bias_ is None else grad_bias,
            None,
            None,
            None,
        )


class FusedLayerNorm(torch.nn.Module):
    """
    Args:
        normalized_shape (int or list or torch.Size) input shape from an expected input of size
        create_scale (bool) If set to False, the layer will not learn an additive weight, Default: True
        create_offset (bool) If set to False, the layer will not learn an additive bias, Default: True
        eps (float) a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        normalized_shape: Union[int, list[int], torch.Size],
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super(FusedLayerNorm, self).__init__()
        # On a CUDA host, build (or load) the extension now, so a missing
        # toolchain fails at model construction rather than in the middle of
        # the first forward. Without CUDA every call takes the PyTorch path.
        if torch.cuda.is_available():
            _fast_layer_norm_extension()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        if create_scale:
            self.weight = Parameter(torch.ones(*normalized_shape))
        else:
            self.weight = None

        if create_offset:
            self.bias = Parameter(torch.zeros(*normalized_shape))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda:
            return _torch_layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if len(self.normalized_shape) == 1 and triton_layer_norm.enabled(input):
            return triton_layer_norm.layer_norm(input, self.weight, self.bias, self.eps)
        return FusedLayerNormAffineFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps
        )


def _torch_layer_norm(
    x: torch.Tensor,
    normalized_shape: torch.Size,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """LayerNorm for tensors the CUDA kernel cannot take (CPU, Apple MPS),
    computed as the LAYERNORM_TYPE=torch path computes it."""
    if x.dtype is torch.bfloat16:
        weight = weight.to(dtype=x.dtype) if weight is not None else None
        bias = bias.to(dtype=x.dtype) if bias is not None else None
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)


if __name__ == "__main__":
    dtype = torch.float32
    data = torch.rand(10, 10).cuda().to(dtype=dtype)
    data1 = data * 1
    data.requires_grad = True
    data1.requires_grad = True
    layer_norm = (
        FusedLayerNorm(10, create_scale=True, create_offset=True).cuda().to(dtype=dtype)
    )
    layer_norm_torch = torch.nn.LayerNorm(10).cuda().to(dtype=dtype)
    out = layer_norm(data)
    out1 = layer_norm_torch(data1)
    # print(out - out1)
    loss = out.sum()
    loss.backward()
    loss1 = out1.sum()
    loss1.backward()
    print(data.grad - data1.grad)
    print(layer_norm.weight.grad - layer_norm_torch.weight.grad)
    print(layer_norm.bias.grad - layer_norm_torch.bias.grad)
    print(layer_norm.weight.grad, layer_norm.bias.grad)

    # layer_norm = FusedLayerNorm(10, create_scale=True, create_offset=False).cuda()
    # out = layer_norm(data)
    # loss = out.sum()
    # loss.backward()

    # layer_norm = FusedLayerNorm(10, create_scale=False, create_offset=True).cuda()
    # out = layer_norm(data)
    # loss = out.sum()
    # loss.backward()

    # layer_norm = FusedLayerNorm(10, create_scale=True, create_offset=True).cuda()
    # out = layer_norm(data)
    # loss = out.sum()
    # loss.backward()
