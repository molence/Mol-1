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

"""CUDA Graph replay for the diffusion denoising step at inference.

One denoising step launches on the order of a thousand small kernels on a
few thousand token rows, so the sampling loop is bound by launch latency, not
by the GPU. Every step has the same shapes and reads the same request-constant
tensors (trunk outputs, caches), so the step is captured once and replayed.
Only the noisy coordinates and the noise level change; they are copied into
the captured input buffers before each replay.

Under autocast the step runs with the autocast weight-cast cache disabled, so
the casts are captured as ordinary kernels. ``MOL_DIFFUSION_CUDA_GRAPH=0``
disables it, and ``MOL_DIFFUSION_CUDA_GRAPH=check`` also runs every replayed
step eagerly and prints the difference (debugging only).
"""

import os
from typing import Any, Callable, Optional

import torch


def diffusion_graph_enabled(x: torch.Tensor) -> bool:
    return (
        os.getenv("MOL_DIFFUSION_CUDA_GRAPH", "1") != "0"
        and x.is_cuda
        and not torch.is_grad_enabled()
    )


class GraphedDenoiseStep:
    """Wraps ``denoise_net(x_noisy=..., t_hat_noise_level=..., **constant_kwargs)``.

    The first call for a given input shape runs eagerly (its result is used
    as-is) and then captures a graph; later calls replay it. The returned
    tensor is the graph's output buffer, valid until the next call.
    """

    def __init__(self, denoise_net: Callable, **constant_kwargs: Any) -> None:
        self.denoise_net = denoise_net
        self.constant_kwargs = constant_kwargs
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.key = None

    def _run(self, x_noisy: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            "cuda",
            dtype=torch.get_autocast_dtype("cuda"),
            enabled=torch.is_autocast_enabled("cuda"),
            cache_enabled=False,
        ):
            return self.denoise_net(
                x_noisy=x_noisy, t_hat_noise_level=t_hat, **self.constant_kwargs
            )

    def __call__(self, x_noisy: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        key = (x_noisy.shape, x_noisy.dtype, t_hat.shape, t_hat.dtype)
        if self.graph is not None and key == self.key:
            self.static_x.copy_(x_noisy)
            self.static_t.copy_(t_hat)
            self.graph.replay()
            if os.getenv("MOL_DIFFUSION_CUDA_GRAPH") == "check":
                ref = self._run(x_noisy, t_hat)
                print(
                    "[graph check] t=%.4g max|graph-eager|=%.3g max|eager|=%.3g nan=%d"
                    % (t_hat.flatten()[0].item(), (self.static_out - ref).abs().max().item(),
                       ref.abs().max().item(), self.static_out.isnan().sum().item()),
                    flush=True,
                )
            return self.static_out

        # Eager step on a side stream (also the warm-up capture requires),
        # then capture the same step for replay.
        self.static_x = x_noisy.clone()
        self.static_t = t_hat.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out = self._run(self.static_x, self.static_t)
        torch.cuda.current_stream().wait_stream(stream)
        out.record_stream(torch.cuda.current_stream())

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self._run(self.static_x, self.static_t)
        self.key = key
        return out
