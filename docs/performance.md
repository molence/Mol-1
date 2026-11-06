# Inference performance

Measured on one NVIDIA GB300 (compute capability 10.3), 9fm7 (644 tokens,
4,372 atoms), 20 cycles, 200 diffusion steps, 5 samples, no templates.
"Forward" is the model alone, median of repeated runs in one process
(`bench/mol_bench.py`); a single `mol pred` run adds about 7 s of one-time
library initialization (cuEquivariance and cuDNN attention plans) to its
first prediction.

| Configuration | Trunk | Diffusion | Confidence | Forward |
| --- | --- | --- | --- | --- |
| Before, defaults | 26.0 s | 31.8 s | 3.5 s | 61.3 s |
| Now, defaults | 9.1 s | 14.7 s | 2.9 s | 26.8 s |
| Before, `--enable_tf32 true --infer_setting.sample_diffusion_chunk_size 5` | 25.9 s | 6.5 s | 1.45 s | 34.0 s |
| Now, same flags | 9.0 s | 2.1 s | 0.91 s | 12.1 s |
| Now, same flags plus `--bf16_confidence_head true` | 9.1 s | 2.1 s | 0.24 s | 11.4 s |

Other inputs, one `mol pred` process per input file, same flags as the
second pair of rows (`--use_msa false` for the FoldBench inputs). The first
input of each process includes the one-time initialization (with Triton's
kernel cache already populated; the very first run on a machine compiles
the Triton kernels, about 2.5 s more).

| Input | Tokens | Before | Now | Peak memory before / now |
| --- | --- | --- | --- | --- |
| 7r6r (first) | 245 | 24.4 s | 12.4 s | 4.2 / 4.8 GiB |
| 7pzb | 600 | 24.3 s | 11.4 s | 15.6 / 17.6 GiB |
| 7wux | 1,218 | 134.8 s | 50.5 s | 38.0 / 41.3 GiB |
| 8cnh (first) | 509 | 32.4 s | 16.9 s | 9.3 / 12.9 GiB |
| 8j2w | 1,000 | 50.7 s | 26.7 s | 28.7 / 31.0 GiB |
| 8h0q | 1,499 | 241.4 s | 78.7 s | 58.4 / 59.3 GiB |

The fused paths keep larger temporaries alive inside a module, so peak memory
rises by up to a few GiB. Ranking scores agreed within the spread of repeated runs of the
old code, which is large on the FoldBench inputs without MSAs (8cnh: 0.29 to
0.66 across three runs of the old code; 8h0q: 0.69 to 0.77).

`mol pred` on 9fm7 also starts about 45 s sooner: the model used to be built
with SciPy's truncated-normal initializer before the checkpoint overwrote
every weight.

## Recommended flags

```bash
mol pred -i input.json -o ./output \
  --enable_tf32 true --infer_setting.sample_diffusion_chunk_size 5
```

- `--enable_tf32 true`: up to 2,560 tokens, diffusion sampling and the
  confidence head run in FP32; TF32 tensor cores make them several times
  faster. Results differ slightly.
- `--infer_setting.sample_diffusion_chunk_size 5` denoises the five samples
  together. Diffusion memory per sample is small next to the trunk's, so
  this rarely changes peak memory (it did not on 9fm7), but samples are drawn
  in a different order, so a seed gives different (equally valid) samples
  than with the default of 1.
- `--bf16_confidence_head true` / `--bf16_sample_diffusion true` run those
  stages under BF16 autocast at every size, as the defaults already do above
  2,560 tokens. On 9fm7 the confidence head drops from 0.91 s to 0.24 s and
  the summary metrics move by less than run-to-run noise. They are off by
  default.

With `--enable_tf32 true`, attention on FP32 operands (diffusion sampling,
the confidence head) also runs on TF32 tensor cores (Triton kernels below);
without it those stay on PyTorch's FP32 SDPA, so the default configuration
gains from the trunk changes only.

## What changed

All of these apply at inference only (no autograd) on CUDA, keep the
existing path for training, and can be switched off with the environment
variable listed, for comparison or if one misbehaves.

| Change | Where | Off switch |
| --- | --- | --- |
| Pad the pair tensor to a multiple of 8 tokens, once per Pairformer stack and once around each MSA block's pair stack, so cuEquivariance triangle attention takes its Blackwell (SM100f) kernel, about 3x faster than the fallback; padding is masked out | `triangular.py`, `pairformer.py` | `MOL_TRI_PAD=0` |
| Triangle attention: one GEMM for q/k/v/gate/bias, a Triton kernel for sigmoid gating plus the head transpose, the residual added in the output GEMM; ending-node attention reads the untransposed pair tensor through strides instead of transposing it twice per block | `triangular.py`, `fused_gate.py`, `pairformer.py` | `MOL_FUSED_TRI_ATT=0` |
| Triangle multiplication: cuEquivariance's pipeline with the projections on cuBLAS (about 2.4x the throughput of its Triton GEMM at Mol-1's widths) and gating, mask and the residual add fused | `fast_trimul.py` | `MOL_FAST_TRIMUL=0` |
| Within it, the input projection gated in the epilogue of a persistent Triton GEMM, so the [4C, N^2] projection never reaches memory (1.4-1.5x) | `fast_trimul.py` | `MOL_GATED_PROJECTION=0` |
| Tuned tiles for cuEquivariance's gated GEMM at Mol-1's pair widths on cc 10.3 (still used by FP32 trimul) | `cueq_tuning/` | `MOL_CUEQ_TUNING=0` |
| Triton LayerNorm for pair tensors (2.4x faster than the extension at width 320/448 in BF16) | `triton_layer_norm.py` | `MOL_TRITON_LN=0` |
| Transition: one GEMM for both up-projections, fused SwiGLU, contiguous row chunks | `primitives.py`, `triton_swiglu.py` | `MOL_FUSED_TRANSITION=0` |
| Within it, the up-projections and SwiGLU as one persistent, warp-specialized Triton GEMM (Blackwell, BF16): the [N^2, 8C] intermediate never reaches memory (1.8-2.0x) | `triton_swiglu.py` | `MOL_SWIGLU_MATMUL=0` |
| Outer product mean: the [N, C, N, C] -> [N, N, C, C] permute at memory bandwidth (4x PyTorch's strided copy) | `triton_permute.py` | — |
| With TF32 enabled: FP32 attention (diffusion token transformer, confidence head) as a TF32 flash-attention Triton kernel (2.9x SDPA's FP32 kernel), and the atom transformer's windowed attention read in place instead of materializing padded key windows | `triton_attention.py`, `primitives.py` | `MOL_TF32_ATTENTION=0` |
| Pair bias: LayerNorm + head projection + transpose in one pass | `pair_bias.py` | `MOL_FUSED_PAIR_BIAS=0` |
| Diffusion: each denoising step is replayed from a CUDA Graph | `cuda_graph.py` | `MOL_DIFFUSION_CUDA_GRAPH=0` |
| Fast LayerNorm extension launches on PyTorch's current stream (it used the legacy default stream, which escaped graph capture) | `layer_norm_cuda_kernel.cu` | — |
| Skip random weight init before a strict checkpoint load | `inference.py` | — |

Each was checked against the run-to-run spread of the unmodified code on 9fm7
(per-sample RMSD between two identical runs is 0.1-0.5 A; pTM differs by
about 5e-4): every change stays within it. `tests/test_inference_fast_paths.py`
compares each fast path with the code it replaces.

## Measuring

```bash
# median timings per stage, outputs saved for comparison
python bench/mol_bench.py --tag mine --repeats 3 -- --enable_tf32 true \
  --infer_setting.sample_diffusion_chunk_size 5
python bench/compare.py bench/results/<reference> bench/results/mine
MOL_BENCH_MODTIME=1 MOL_DIFFUSION_CUDA_GRAPH=0 python bench/mol_bench.py ...  # per-module times
MOL_BENCH_PROFILE=trace.json python bench/mol_bench.py ...                   # torch profiler
python bench/tune_cueq.py <shard> <n_shards> <dir>                            # regenerate cueq_tuning
python bench/peaks.py                                                         # GEMM / DRAM ceilings
```

Nsight Systems: `bench/nsys.sh <gpu> <tag> -- <runner args>` (Slurm
allocation in `JOBID`) traces the last repeat with an NVTX range per stage
and per module (`MOL_BENCH_NVTX=1`; diffusion graph replays are traced per
node). Then

```bash
nsys export -t sqlite -o trace.sqlite bench/results/nsys/<tag>.nsys-rep
python bench/nsys_attr.py trace.sqlite --depth 9 --top 8   # kernel time per module path
python bench/sol.py trace.sqlite                            # vs the speed-of-light model
```

`bench/micro_*.py` and `bench/check_*.py` time and check single kernels.

## Speed-of-light model and where the time goes now

`bench/sol.py` bounds each pair module by its FLOPs at the measured BF16
GEMM ceiling (2.05 PFLOPS, `bench/peaks.py`), its minimum DRAM traffic
(read and write the pair tensor once, 6.8 TB/s) and, for attention, the exp
throughput of the special-function units. 9fm7 (648 padded tokens), kernel
time per call from nsys, `--enable_tf32 true`:

| Module (calls per prediction) | SOL | Start of this round | Now | Now / SOL |
| --- | --- | --- | --- | --- |
| Pairformer triangle attention, C=320, 10 heads (1,600) | 0.59 ms (exp) | 2.03 ms + 0.92 ms of transposes per block pair | 2.06 ms | 3.5x |
| Pairformer triangle multiplication, C=320 (1,600) | 0.34 ms (GEMM) | 1.25 ms | 1.10 ms | 3.3x |
| Pairformer transition, C=320 (800) | 0.50 ms (GEMM) | 1.36 ms | 0.84 ms | 1.7x |
| PairMixer triangle multiplication, C=448 (320) | 0.61 ms | 1.86 ms | 1.58 ms | 2.6x |
| PairMixer transition, C=448 (160) | 0.99 ms | 2.07 ms | 1.44 ms | 1.5x |
| Diffusion token transformer, per step (200) | 1.45 ms (TF32) | about 10 ms | 7.5 ms | 5x |

Kernel time on 9fm7 is now 11.9 s: Pairformer stack 7.0 s, diffusion 1.9 s,
MSA module 1.8 s, confidence head 0.9 s, template embedder 0.3 s. Remaining
gaps, largest first:

- Triangle attention: the cuEquivariance/cuDNN kernel takes 1.27 ms of the
  2.06 ms. At head dim 32 the softmax arithmetic (exp, bias add,
  reductions) rather than the MMAs bounds it, at roughly 1.0 ms in
  practice, so the kernel is near its floor; the other 0.8 ms (projection
  GEMM 0.36 ms writing 1.1 GB of q/k/v/gate/bias, gating 0.15 ms, LayerNorm
  0.12 ms, output GEMM 0.12 ms) would need the gating moved into the
  output GEMM's prologue and the LayerNorm into the projection.
- Triangle multiplication: gated projection 0.34 ms, triangular einsum
  0.23 ms (cuBLAS batched GEMM at 0.8 PFLOPS), two LayerNorm transposes
  0.21 ms, output GEMMs 0.17 ms, output gating 0.16 ms. Fusing the output
  gating into a dual-input GEMM was tried: with Triton 3.8 the
  warp-specialized form fails to compile with two accumulators, and without
  warp specialization it is slower than cuBLAS.
- Diffusion: about 1,290 kernels per step for a few GFLOP each; the token
  attention kernel (98 us per call vs about 10 us) is latency-bound in
  Triton's TF32 flash attention, and LayerNorm, adaptive-LayerNorm gating
  and residual adds are separate elementwise kernels (about 0.6 s in all).

Tried and dropped: cuEquivariance gated-GEMM tile configs beyond its tuning
space (no gain), torch.compile'd SwiGLU (slower than the Triton kernel),
row chunks under 256k rows in the transition (slower), non-persistent
Triton GEMMs for the SwiGLU and gating fusions (slower than cuBLAS plus a
separate kernel), the windowed atom attention in full FP32 (Triton's IEEE
FP32 dot is slower than SDPA plus the window copies, so that kernel is used
with TF32 only).
