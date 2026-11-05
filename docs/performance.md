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
| Now, defaults | 11.0 s | 14.7 s | 2.9 s | 28.7 s |
| Before, `--enable_tf32 true --infer_setting.sample_diffusion_chunk_size 5` | 25.9 s | 6.5 s | 1.45 s | 34.0 s |
| Now, same flags | 10.9 s | 3.2 s | 0.94 s | 15.2 s |
| Now, same flags plus `--bf16_confidence_head true` | 11.0 s | 3.2 s | 0.28 s | 14.6 s |

Other inputs, one `mol pred` process per input file, same flags as the
second pair of rows (`--use_msa false` for the FoldBench inputs). The first
input of each process includes the one-time initialization.

| Input | Tokens | Before | Now | Peak memory before / now |
| --- | --- | --- | --- | --- |
| 7r6r (first) | 245 | 24.4 s | 14.0 s | 4.2 / 4.7 GiB |
| 7pzb | 600 | 24.3 s | 14.0 s | 15.6 / 17.4 GiB |
| 7wux | 1,218 | 134.8 s | 59.8 s | 38.0 / 41.1 GiB |
| 8cnh (first) | 509 | 32.4 s | 21.7 s | 9.3 / 12.7 GiB |
| 8j2w | 1,000 | 50.7 s | 32.4 s | 28.7 / 30.7 GiB |
| 8h0q | 1,499 | 241.4 s | 89.4 s | 58.4 / 59.1 GiB |

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
  2,560 tokens. On 9fm7 the confidence head drops from 0.95 s to 0.32 s and
  the summary metrics move by less than run-to-run noise. They are off by
  default.

## What changed

All of these apply at inference only (no autograd) on CUDA, keep the
existing path for training, and can be switched off with the environment
variable listed, for comparison or if one misbehaves.

| Change | Where | Off switch |
| --- | --- | --- |
| Pad the pair tensor to a multiple of 8 tokens, once per Pairformer stack, so cuEquivariance triangle attention takes its Blackwell (SM100f) kernel, about 3x faster than the fallback; padding is masked out | `triangular.py`, `pairformer.py` | `MOL_TRI_PAD=0` |
| Triangle attention: one GEMM for q/k/v/gate/bias, and a Triton kernel for sigmoid gating plus the head transpose | `triangular.py`, `fused_gate.py` | `MOL_FUSED_TRI_ATT=0` |
| Triangle multiplication: cuEquivariance's pipeline with the projections on cuBLAS (about 2.4x the throughput of its Triton GEMM at Mol-1's widths) and gating, mask and the residual add fused | `fast_trimul.py` | `MOL_FAST_TRIMUL=0` |
| Tuned tiles for cuEquivariance's gated GEMM at Mol-1's pair widths on cc 10.3 (still used by FP32 trimul) | `cueq_tuning/` | `MOL_CUEQ_TUNING=0` |
| Triton LayerNorm for pair tensors (2.4x faster than the extension at width 320/448 in BF16) | `triton_layer_norm.py` | `MOL_TRITON_LN=0` |
| Transition: one GEMM for both up-projections, fused SwiGLU, contiguous row chunks | `primitives.py`, `triton_swiglu.py` | `MOL_FUSED_TRANSITION=0` |
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
```

`bench/micro_*.py` and `bench/check_*.py` time and check single kernels.

## Where the time goes now

Per Pairformer block at 648 (padded) tokens, pair width 320: triangle
attention 2 x 2.0 ms, of which 1.3 ms is the cuEquivariance attention kernel
itself (about 270 TFLOPS); triangle multiplication 2 x 1.2 ms; transition
1.3 ms; pair bias and single attention 0.6 ms. Ideas not yet done:

- Triangle attention is limited by cuEquivariance's kernel (cuDNN SDPA with a
  broadcast bias is 2.4x slower at head dim 32).
- The two transposes around the ending-node triangle attention in each block
  are full copies of the pair tensor.
- The MSA module (about 2 s) and template embedder (about 0.6 s) still pad
  per op and use unfused paths.

Tried and dropped: cuEquivariance gated-GEMM tile configs beyond its tuning
space (no gain), torch.compile'd SwiGLU (slower than the Triton kernel),
row chunks under 256k rows in the transition (slower).
