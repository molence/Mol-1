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

"""Tuned cuEquivariance Triton configs for Mol-1's pair widths.

cuEquivariance ships tuned tile configs for its gated dual GEMM only for pair
width 128/256, so at Mol-1's widths (64, 320, 448) the kernel runs on its
default tiles. The JSON files here hold configs tuned on Mol-1's shapes
(``bench/tune_cueq.py``). Lookups that miss cuEquivariance's own cache fall
back to the entry for the same GEMM shape and precision with the nearest row
count. Anything cuEquivariance finds itself, and any explicit
``CUEQ_TRITON_TUNING`` mode, takes precedence. ``MOL_CUEQ_TUNING=0`` disables.
"""

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
_installed = False


def _load_table(major: int, minor: int) -> dict:
    table = {}
    for name in os.listdir(_DIR):
        if not name.endswith(f".{major}.{minor}.json"):
            continue
        fn_key = name[: -len(f".{major}.{minor}.json")]
        with open(os.path.join(_DIR, name), encoding="utf-8") as f:
            entries = json.load(f)
        by_shape = table.setdefault(fn_key, {})
        for key, value in entries.items():
            m, shape = key.split("_", 1)
            by_shape.setdefault(shape, []).append((int(m), value))
    return table


def install(device: torch.device) -> None:
    """Install the fallback lookup once per process (CUDA devices only)."""
    global _installed
    if _installed or device.type != "cuda" or os.getenv("MOL_CUEQ_TUNING", "1") == "0":
        return
    _installed = True
    try:
        from cuequivariance_ops.triton.cache_manager import get_cache_manager
    except ImportError:
        return
    major, minor = torch.cuda.get_device_capability(device)
    table = _load_table(major, minor)
    if not table:
        return
    cache_manager = get_cache_manager()
    original_get = cache_manager.get

    def get(fn_key, inp_key):
        value = original_get(fn_key, inp_key)
        if value is not None or cache_manager.aot_mode is not None:
            return value
        m, shape = inp_key.split("_", 1)
        candidates = table.get(fn_key, {}).get(shape)
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c[0] - int(m)))[1]

    cache_manager.get = get
    logger.info("Using Mol-1 cuEquivariance tuning for compute capability %d.%d", major, minor)
