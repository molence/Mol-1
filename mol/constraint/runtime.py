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
"""Numerical policies and checks for constraint inference."""
from __future__ import annotations

from collections.abc import Mapping
from math import prod
from pathlib import Path
from types import MethodType
import json
import logging
import os
import socket
import torch


class NonFiniteTensorError(FloatingPointError):
    """A diagnosed NaN/Inf, distinct from programming or device failures."""


def require_finite(value, location):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise NonFiniteTensorError(
                f"Non-finite tensor at {location}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            require_finite(child, f"{location}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            require_finite(child, f"{location}[{index}]")


def reference_layer_norms(model):
    """Replace only this model's CUDA LayerNorms; retain identical parameter keys."""
    for name, child in list(model.named_children()):
        if (type(child).__name__ == "FusedLayerNorm"
                and type(child).__module__.startswith("mol.model.layer_norm")):
            replacement = torch.nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=False)
            replacement.weight = child.weight
            replacement.bias = child.bias
            setattr(model, name, replacement)
        else:
            reference_layer_norms(child)


def install_finite_hooks(model, *, detailed=False):
    """Check model outputs for non-finite values."""
    boundaries = {"input_embedder", "constraint_conditioner", "template_embedder",
                  "msa_module", "pairformer_stack", "diffusion_module",
                  "confidence_head", "distogram_head"}
    handles = []
    for name, module in model.named_modules():
        if not name or (name not in boundaries and not detailed):
            continue

        def check(_module, _inputs, output, name=name):
            require_finite(output, f"model.{name}.output")

        handles.append(module.register_forward_hook(check))
    return handles


_REPORTED = set()


def _get(config, path, default=None):
    for key in path.split("."):
        config = config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)
        if config is default:
            return default
    return config


def resolve_n_cycle(configs):
    """Use the same recycle count as native Mol-1 inference."""
    key = "inference_max_cycle"
    value = _get(configs, "model.N_cycle")
    if value is None or isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{key} must resolve to a positive integer, got {value!r}")
    return int(value)


def runtime_snapshot(config):
    return {
        "mc_dropout_apply_rate": _get(config, "mc_dropout_apply_rate"),
        "mc_dropout_rate": _get(config, "mc_dropout_rate"),
        "steps": _get(config, "sample_diffusion.N_step"),
        "cycles": _get(config, "model.N_cycle"),
        "diffusion_chunk_size": _get(config, "infer_setting.sample_diffusion_chunk_size"),
        "enable_tf32": _get(config, "enable_tf32"),
        "dtype": _get(config, "dtype"),
    }


def verify_runtime_config(config, expected, *, phase, execution_plan=None,
                          mc_dropout_apply_rate=None, cuda_execution=False, model=None):
    """Check resolved values again after all runtime/shape-dependent updates."""
    if expected is None:
        return None
    actual = runtime_snapshot(config)
    if mc_dropout_apply_rate is not None:
        actual["mc_dropout_apply_rate"] = mc_dropout_apply_rate
    mismatches = {key: {"expected": value, "actual": actual[key]}
                  for key, value in expected.items() if actual[key] != value}
    chunk = actual["diffusion_chunk_size"]
    if execution_plan is not None:
        chunk = execution_plan.diffusion_sample_chunk
        if chunk != expected["diffusion_chunk_size"]:
            mismatches["execution_plan.diffusion_sample_chunk"] = {
                "expected": expected["diffusion_chunk_size"], "actual": chunk}
    model_cycles = getattr(model, "inference_max_cycle", None)
    if model_cycles is not None and model_cycles != expected["cycles"]:
        mismatches["model.inference_max_cycle"] = {"expected": expected["cycles"], "actual": model_cycles}
    runtime = {}
    if cuda_execution:
        import torch
        runtime = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "autocast_enabled": torch.is_autocast_enabled("cuda"),
                   "autocast_dtype": str(torch.get_autocast_dtype("cuda"))}
        if runtime["matmul_allow_tf32"] != expected["enable_tf32"]:
            mismatches["torch.matmul.allow_tf32"] = runtime["matmul_allow_tf32"]
        dtype = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}.get(expected["dtype"])
        if dtype and (not runtime["autocast_enabled"] or runtime["autocast_dtype"] != dtype):
            mismatches["torch.autocast"] = runtime
    record = {"phase": phase, "host": socket.gethostname(), "pid": os.getpid(),
              "rank": os.getenv("RANK", "0"), "expected": expected, "actual": actual,
              "effective_diffusion_chunk_size": chunk, "model_inference_max_cycle": model_cycles,
              "skip_amp": {key: _get(config, "skip_amp." + key)
                           for key in ("sample_diffusion", "confidence_head")},
              "runtime": runtime, "passed": not mismatches, "mismatches": mismatches}
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    audit_path = os.getenv("CONSTRAINT_INFERENCE_AUDIT_PATH")
    if audit_path:
        path = Path(audit_path.replace("{rank}", os.getenv("RANK", "0")))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(encoded + "\n")
    signature = (phase, json.dumps(actual, sort_keys=True), str(runtime))
    if signature not in _REPORTED:
        logging.getLogger(__name__).info("Inference runtime verification: %s", encoded)
        _REPORTED.add(signature)
    if mismatches:
        raise RuntimeError("Inference runtime config changed after resolution: " + json.dumps(mismatches))
    return record


def configure_inference_transition_chunking(model, *, min_tokens, chunk_rows):
    """Bound large pair-transition intermediates without changing parameters."""
    from mol.model.modules.primitives import Transition
    if min_tokens <= 0 or chunk_rows <= 0:
        raise ValueError("inference transition chunk settings must be positive")

    def chunked_forward(module, x, chunk_size=None):
        if module.training or x.ndim < 3 or x.shape[-2] < min_tokens:
            return Transition.forward(module, x, chunk_size=chunk_size)
        leading = prod(x.shape[:-3])
        rows_per_chunk = max(1, chunk_rows // (leading * x.shape[-2]))
        if x.shape[-3] <= rows_per_chunk:
            return Transition.forward(module, x, chunk_size=chunk_size)
        output = torch.empty_like(x)
        for start in range(0, x.shape[-3], rows_per_chunk):
            end = min(start + rows_per_chunk, x.shape[-3])
            output[..., start:end, :, :] = Transition.forward(
                module, x[..., start:end, :, :], chunk_size=chunk_size
            )
        return output

    for module in model.modules():
        if isinstance(module, Transition):
            module.forward = MethodType(chunked_forward, module)
