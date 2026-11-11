// Copyright 2026 Molence Bio
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Gated triangle attention for Blackwell (SM100/SM103), BF16, head dim 32:
//   out = softmax(scale * q k^T + bias, keys < n_keys) v * sigmoid(gate)
// on the CUTLASS Blackwell FMHA forward kernel (examples/77_blackwell_fmha,
// headers under collective/ kernel/ device/), with the [H, Q, K] pair bias
// added by the tensor cores and the gate applied in the epilogue.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cute/tensor.hpp"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "kernel/fmha_tile_scheduler.hpp"

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;

namespace {

using Element = cutlass::bfloat16_t;
using TileShape = Shape<_256, _128, _32>;
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;  // Q K D ((H_R, H_K), B)
using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

using Mainloop = Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, float, float, TileShape, StrideQ, StrideK, StrideK, ResidualMask>;
using Operation = cutlass::fmha::device::FMHA<Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape, Mainloop,
    Sm100FmhaFwdEpilogueTmaWarpspecialized<Element, float, typename Mainloop::TileShapePV, StrideQ, StrideLSE>,
    PersistentTileScheduler>>;

bool aligned(torch::Tensor const& t) {
  // TMA / 16-byte vector access: 16-byte aligned base and strides, unit stride last
  if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0 || t.stride(-1) != 1) {
    return false;
  }
  for (int i = 0; i < t.dim() - 1; i++) {
    if (t.stride(i) % 8 != 0) {
      return false;
    }
  }
  return true;
}

StrideQ stride_q(torch::Tensor const& t) {
  return {(int)t.stride(2), _1{}, make_tuple(make_tuple((int)t.stride(1), (int)t.stride(1)), (int)t.stride(0))};
}

StrideK stride_k(torch::Tensor const& t) {
  return {(int)t.stride(2), _1{}, make_tuple(make_tuple(_0{}, (int)t.stride(1)), (int)t.stride(0))};
}

}  // namespace

// q, gate, out: [B, H, Q, 32]; k, v: [B, H, K, 32] (any strides, unit in the
// last dim, 16-byte aligned); bias: [H, Q, K] contiguous. Keys at or past
// n_keys are masked.
void fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor gate,
         torch::Tensor out, int64_t n_keys, double scale) {
  for (auto const* t : {&q, &k, &v, &bias, &gate, &out}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16, "tri_fmha: BF16 CUDA tensors expected");
    TORCH_CHECK(aligned(*t), "tri_fmha: 16-byte aligned strides expected");
  }
  int b = q.size(0), h = q.size(1), nq = q.size(2), d = q.size(3), nk = k.size(2);
  TORCH_CHECK(d == 32 && k.size(3) == 32 && v.size(3) == 32, "tri_fmha: head dim 32 expected");
  TORCH_CHECK(k.size(0) == b && k.size(1) == h && v.sizes() == k.sizes(), "tri_fmha: k/v shape");
  TORCH_CHECK(gate.sizes() == q.sizes() && out.sizes() == q.sizes(), "tri_fmha: gate/out shape");
  TORCH_CHECK(bias.is_contiguous() && bias.dim() == 3 && bias.size(0) == h && bias.size(1) == nq && bias.size(2) == nk,
              "tri_fmha: bias [H, Q, K] expected");
  TORCH_CHECK(0 < n_keys && n_keys <= nk, "tri_fmha: n_keys");

  ProblemShape shape{nq, (int)n_keys, d, make_tuple(make_tuple(1, h), b)};
  typename Operation::Arguments args;
  args.problem_shape = shape;
  args.mainloop.load = {reinterpret_cast<Element*>(q.data_ptr()), stride_q(q),
                        reinterpret_cast<Element*>(k.data_ptr()), stride_k(k),
                        reinterpret_cast<Element*>(v.data_ptr()), stride_k(v),
                        reinterpret_cast<Element*>(bias.data_ptr()), nq, nk, h};
  args.mainloop.scale_softmax = (float)scale;
  args.mainloop.bias_diag = (float)(1.0 / scale);
  args.mainloop.ptr_gate = reinterpret_cast<Element*>(gate.data_ptr());
  args.mainloop.gate_sq = gate.stride(2);
  args.mainloop.gate_sh = gate.stride(1);
  args.mainloop.gate_sb = gate.stride(0);
  args.epilogue = {reinterpret_cast<Element*>(out.data_ptr()), stride_q(out), nullptr, StrideLSE{}};
  args.hw_info.device_id = q.device().index();
  static int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
  args.hw_info.sm_count = sm_count;

  Operation op;
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "tri_fmha: unsupported problem");
  TORCH_CHECK(op.initialize(args, nullptr, stream) == cutlass::Status::kSuccess, "tri_fmha: initialize failed");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess, "tri_fmha: launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd); }
