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
| Now, defaults | 8.6 s | 12.2 s | 2.9 s | 23.8 s |
| Before, `--enable_tf32 true --infer_setting.sample_diffusion_chunk_size 5` | 25.9 s | 6.5 s | 1.45 s | 34.0 s |
| Now, same flags | 8.6 s | 1.39 s | 0.55 s | 10.6 s |
| Now, same flags, FA4 triangle attention installed (below) | 8.1 s | 1.39 s | 0.55 s | 10.2 s |
| Now, same flags plus `--bf16_confidence_head true` | 8.6 s | 1.38 s | 0.23 s | 10.3 s |
| Now, same flags, FA4 triangle and token attention (below) | 8.2 s | 1.11 s | 0.55 s | 9.9 s |

The last row was measured on a shared node against 10.25 s (diffusion 1.40 s)
for the same code with `MOL_FA4_ATTENTION=0` in the same session.

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
rises by up to a few GiB (on 9fm7, 0.8 GiB of it is the fused diffusion
weights, kept for the life of the model). Ranking scores agreed within the spread of repeated runs of the
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
| Triangle attention on Blackwell (BF16, head dim 32): the CUTLASS Blackwell FMHA kernel extended so the pair bias is loaded by TMA and added by the tensor cores (S starts as bias @ identity, then q k^T accumulates on it) and the sigmoid gate is applied in the epilogue, which writes the [I, J, H, D] output directly: 1.17 ms per call at N=648 against 1.44 ms for cuEquivariance's cuDNN kernel plus the gating kernel. Compiled at first use against CUTLASS headers (`MOL_CUTLASS_DIR`, or the copy in NVIDIA's PyTorch containers); masks other than padding fall back | `tri_fmha/`, `triangular.py` | `MOL_TRI_FMHA=0` |
| Triangle attention on FlashAttention-4 when the CuTe DSL is installed: the FA4 SM100 forward kernel (CuTe DSL) with the same two changes (bias tiles by TMA, S = bias @ I / scale on the tensor cores before q k^T; gate tiles by TMA in the epilogue warp, sigmoid applied in the correction epilogue): 0.91 ms per call at N=648 against 1.13 ms for the CUTLASS kernel, which stays the fallback. Needs flash-attn-4 (in NVIDIA's PyTorch containers) and `pip install "nvidia-cutlass-dsl[cu13]==4.6.0"` (or `MOL_CUTE_DSL_DIR`); compiled once (about 30 s) into a shared library that later processes load through tvm_ffi in about 0.5 s, without importing the DSL (importing it costs about 20 s) | `tri_fa4/`, `triangular.py` | `MOL_TRI_FA4=0` |
| Triangle multiplication: cuEquivariance's pipeline with the projections on cuBLAS (about 2.4x the throughput of its Triton GEMM at Mol-1's widths) and gating, mask and the residual add fused; in FP32 (the confidence head) only with TF32 enabled, where it is 1.8x faster than cuEquivariance's FP32 kernels | `fast_trimul.py` | `MOL_FAST_TRIMUL=0` |
| Within it, the input projection gated in the epilogue of a persistent Triton GEMM, so the [4C, N^2] projection never reaches memory (1.4-1.5x) | `fast_trimul.py` | `MOL_GATED_PROJECTION=0` |
| Tuned tiles for cuEquivariance's gated GEMM at Mol-1's pair widths on cc 10.3 (still used by strict-FP32 trimul) | `cueq_tuning/` | `MOL_CUEQ_TUNING=0` |
| Triton LayerNorm for pair tensors (2.4x faster than the extension at width 320/448 in BF16) | `triton_layer_norm.py` | `MOL_TRITON_LN=0` |
| Transition: one GEMM for both up-projections, fused SwiGLU, contiguous row chunks | `primitives.py`, `triton_swiglu.py` | `MOL_FUSED_TRANSITION=0` |
| Within it, the up-projections and SwiGLU as one persistent, warp-specialized Triton GEMM (Blackwell, BF16): the [N^2, 8C] intermediate never reaches memory (1.8-2.0x) | `triton_swiglu.py` | `MOL_SWIGLU_MATMUL=0` |
| Outer product mean: the [N, C, N, C] -> [N, N, C, C] permute at memory bandwidth (4x PyTorch's strided copy) | `triton_permute.py` | — |
| With TF32 enabled: FP32 attention (diffusion token transformer, confidence head) as a TF32 flash-attention Triton kernel (2.9x SDPA's FP32 kernel), and the atom transformer's windowed attention read in place instead of materializing padded key windows | `triton_attention.py`, `primitives.py` | `MOL_TF32_ATTENTION=0` |
| Within it, on Blackwell: q/k/v/bias tiles loaded by TMA with the key loop pipelined (95 -> 67 us per call at the diffusion shape), the attention module's sigmoid gate in the epilogue, and an optional key mask; the confidence head's FP32 triangle attention uses it too (1.6 vs 3.0 ms per call for cuEquivariance's FP32 kernel) | `triton_attention.py`, `triangular.py` | `MOL_TMA_ATTENTION=0` |
| Diffusion token attention on FlashAttention-4 (with TF32 enabled, on Blackwell with the CuTe DSL installed, as for `tri_fa4`): the q/k/v/gate projection, the attention and the output projection in FP16 with FP32 accumulation (FP16 has TF32's 10-bit mantissa; softmax stays FP32) on the FA4 SM100 kernel of `tri_fa4`, compiled for FP16 at head dim 48: the pair bias added by the tensor cores, the sigmoid gate in the correction epilogue (gate tiles read in 16-byte loads, the product rounded once to f16x2), three bias stages, 80 registers for the correction warps. 21.1 us per call at [5, 16, 644, 48] against 74 us for the Triton TMA kernel; the 24-block token transformer 3.0 vs 4.5 ms per step. On 9fm7, against two runs of the Triton path: per-sample RMSD 0.08-0.18 A (0.08-0.50 A between those two runs), mean pLDDT within 0.04 (0.12), pTM within 7e-4 (4e-4). The pair biases are converted to FP16 (keys padded to a multiple of 8) once per prediction, 0.3 GB at 9fm7 | `fa4_attention.py`, `diffusion_fast.py`, `tri_fa4/` | `MOL_FA4_ATTENTION=0` |
| Diffusion transformer blocks: the s-side projections of all blocks (both adaptive LayerNorms, both output gates) as one LayerNorm and two wide GEMMs per group of blocks, AdaLN of the activations as one Triton kernel that also applies the previous sublayer's gated residual and takes the gate sigmoids, q/k/v/g as one GEMM: about 45 kernels per block become 9. The atom encoder and decoder transformers take the same path (cross-attention mode: keys and values from a second AdaLN; their conditioning is shared by the samples and computed once per atom). Diffusion 2.08 -> 1.39 s with TF32, 14.7 -> 12.2 s in full FP32 | `diffusion_fast.py`, `transformer.py` | `MOL_FUSED_DIFFUSION_BLOCKS=0` |
| Template embedder: templates with identical features go through its Pairformer once (without templates the featurizer still emits four, all masked: slot 0 all gaps, slots 1-3 zero padding, so 4 runs become 2), linear_no_bias_z(z) is shared by the templates and the z-independent template projection is computed once per prediction instead of every cycle. Bit-identical output; 16.0 -> 7.5 ms per cycle on 9fm7, trunk 8.64 -> 8.46 s. Skipping the embedder entirely is not equivalent: masked templates still pass linear_no_bias_z(LayerNorm(z)) through the stack, so the output depends on z | `pairformer.py`, `mol.py` | `MOL_DEDUP_TEMPLATES=0` |
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
| Pairformer triangle attention, C=320, 10 heads (1,600) | 0.59 ms (exp) | 2.03 ms + 0.92 ms of transposes per block pair | 1.86 ms | 3.1x |
| Pairformer triangle multiplication, C=320 (1,600) | 0.34 ms (GEMM) | 1.25 ms | 1.10 ms | 3.3x |
| Pairformer transition, C=320 (800) | 0.50 ms (GEMM) | 1.36 ms | 0.84 ms | 1.7x |
| PairMixer triangle multiplication, C=448 (320) | 0.61 ms | 1.86 ms | 1.58 ms | 2.6x |
| PairMixer transition, C=448 (160) | 0.99 ms | 2.07 ms | 1.44 ms | 1.5x |
| Diffusion stage, per step (200) | 1.45 ms (TF32, token transformer) | about 10 ms | 5.5 ms (whole stage, wall) | |

Kernel time on 9fm7 is now 10.3 s: Pairformer stack 6.5 s, MSA module
1.7 s, diffusion 1.2 s, confidence head 0.54 s, template embedder 0.29 s.
Remaining gaps, largest first:

- Triangle attention: the CUTLASS kernel takes 1.21 ms of the 1.86 ms. At
  head dim 32 it is bound by the per-step synchronization between its
  warps rather than by the exponentials (see below). With the CuTe DSL
  installed, the FA4-based kernel replaces it (0.91 ms in isolation against
  1.13 ms; FA4 itself, without bias or gate, takes 0.81 ms, and CUTLASS's
  example kernel 1.16 ms, so most of the gain is FA4's pipeline); the rest is the
  projection GEMM (0.36 ms, writing 1.1 GB of q/k/v/gate/bias), LayerNorm
  0.12 ms, output GEMM 0.11 ms and the bias gather into [H, N, N] (a Triton
  kernel, 9-12 us; `permute().contiguous()` took 19 us starting-node and
  50 us ending-node at C=320, 30 / 91 us at C=448).
- Triangle multiplication: gated projection 0.34 ms, triangular einsum
  0.23 ms (cuBLAS batched GEMM at 0.8 PFLOPS), two LayerNorm transposes
  0.21 ms, output GEMMs 0.17 ms, output gating 0.16 ms. Fusing the output
  gating into a dual-input GEMM was tried: with Triton 3.8 the
  warp-specialized form fails to compile with two accumulators, and without
  warp specialization it is slower than cuBLAS.
- Diffusion: about 420 kernels per step (was 1,290) after fusing the token
  and atom transformer blocks. The token attention takes 21.1 us per call on
  FA4 (about 0.1 s per prediction; it was 67-74 us in Triton) against a
  SOL of about 10 us. FA4 at head dim 48 is bound by the latency of its
  per-key-block chain (about 1.2 us per 2 x 128 x 128 step, twice what its
  exponentials need: GB300 runs ex2 at 30 per clock per SM, measured), and
  the shape quantizes badly: 644 = 5 x 128 + 4, so a sixth key block and a
  sixth query tile per head are almost empty (16% each), and the 240 work
  units (80 heads x 3 x 256 rows) take two rounds on 152 SMs (one round of
  one unit per CTA takes about 5 us plus 1.2 us per key block). The
  attention is now 21 of the 125 us per token transformer block.

Kernel work that did not pay off (fourth round, FA4 token attention at
[5, 16, 644, 48], 21.1 us per call; 20.5 us with the tanh gate below):

- Tile shapes: 96-key blocks (7 blocks, 672 keys instead of 768) 26.6 us,
  192-key blocks 40.6 us, one 128-row query tile per CTA (480 units) 27.6
  us, the non-persistent scheduler 23.3 us, the CLC scheduler 20.5 us. With
  the bias and gate both on, FA4's four bias stages leave shared memory for a
  single K/V stage, which hung; two stages 24.0 us, three 22.0 us.
- FA4's partial exp2 emulation on the FMA pipe (off by default on sm_103):
  22.2-23.4 us against 21.9 us; the kernel is not exp-bound.
- P in its own TMEM columns (S0 P0 S1 P1 O0 O1), so the MMA warp computes
  S(i+1) as soon as the softmax warps hold S(i) in registers instead of after
  P(i) V: 21.1 vs 20.6 us, no gain (and the correction warps would then need
  to wait for P(i-1) V before rescaling O).
- Releasing each softmax warp group's statistics buffer as soon as the
  correction warps read it (FA4 releases it after processing the other warp
  group, which keeps the two in alternation): the per-step chain shortened
  in an instrumented run (clock stamps per event), but no change uninstrumented.
- The gate: FA4's sigmoid (exp2 + rcp per element, FP32) in the correction
  epilogue cost 2.8 us per call. Precomputing sigmoid(g) in place in shared
  memory by the epilogue warp was slower (+1.5 us: one warp of tanh competes
  for issue with the softmax warps); what helped was 16-byte gate loads and 80
  registers for the correction warps (2.8 -> 1.5 us). The sigmoid as f16x2
  tanh.approx (one MUFU op per two elements) saved another 0.6 us but lowered
  pLDDT on 9fm7 by 0.2 in two runs (run-to-run spread 0.1): small gates
  cancel in 0.5 + 0.5 tanh, so the gate stays FP32 ex2 + rcp.

Kernel work that did not pay off (fourth round, FA4 triangle attention at
N=648, 10 heads, 0.93 ms per call; all correct, none faster):

- A clock64 trace of one CTA shows the two softmax warpgroups strictly
  alternating: FA4's correction warps release the softmax-stats barrier
  crosswise (stage 0's release frees stage 1), which makes stage 1 wait for
  stage 0 every key block. Releasing per stage lets both run at once and
  makes the traced CTA 18% faster, but the kernel time does not move
  (0.925 ms).
- The pair-bias MMAs as one inline-PTX block instead of eight `cute.gemm`
  calls: 0.928 ms. Skipping the bias MMAs entirely: 0.85 ms (the bias costs
  about 80 us, TMA plus tensor core).
- Two S/P buffers per q stage in TMEM (n_block 96 or 64, so 4 S + 2 O fit in
  512 columns), the QK MMA of block i + 2 issued right after the PV MMA of
  block i, O rescaled by the softmax warps themselves (no correction warps
  in the loop), one MMA warp per q stage: 1.02 ms. S is then always ready,
  and the softmax warps become the limit.
- On top of that, split keys: two softmax warpgroups per q stage (even and
  odd key blocks, each with its own O accumulator and running max/sum,
  merged in the epilogue), 16 softmax warps at 104 registers: 1.08 ms. Twice
  the warps, the same per-SM softmax throughput, about 12 elements per clock
  per SM, while ncu shows MUFU at 38%, the tensor pipe at 46-56% and issue
  slots at 45%. The exponentials are not the limit (ex2 runs at 32 per clock
  per SM on GB300, measured; FA4's ex2 emulation for 1/6 to 1/2 of the
  elements, `tcgen05.ld.red` row max and FA4's s0/s1 exp-phase barrier all
  made it no faster or slower). The code is on the local branch
  `wip-tri-fa4-splitkey`.

Kernel work that did not pay off (third round):

- In the CUTLASS triangle attention kernel, skipping the exponentials of the
  mostly padded last key tile (648 keys = 5 x 128 + 8) and of query rows past
  N: no gain, and the row skip slowed it down. At head dim 32 the kernel is
  bound by the hand-offs between the MMA, softmax and correction warps per
  128-key step, not by the exponentials (replacing exp2 with an FMA does not
  make it faster). Larger key tiles do not fit: two 128x128 FP32 S tiles and
  two O tiles already fill the 512 TMEM columns.
- Adding the bias in the softmax warps from global memory: 2.6 ms
  (uncoalesced, one row per thread). As one 128x128 identity MMA per tile:
  1.19 ms; as eight 128x16x16 MMAs: 1.15 ms (without the gate).
- BF16 triangle attention in Triton: 2.25 ms vs 1.47 ms for cuDNN.
- FA4's own way of adding a bias (a `score_mod` reading the bias tensor per
  element, as generated by `flex_attention(..., kernel_options={"BACKEND":
  "FLASH"})`): 3.1 ms, against 0.87 ms with the bias tiles on the tensor
  cores.
- The triangle attention LayerNorm folded into its projection (row
  statistics kernel + persistent TMA GEMM with the normalization in the
  epilogue): 428 vs 473 us per call; about 0.07 s per prediction, not taken.
- flash_attn_4 would need the CuTe DSL from PyPI, not in the container.

Kernel work that did not pay off (second round):

- Triton attention for the diffusion shape plateaus at 64-68 us per call
  (SOL about 10 us): TMA loads, ping-pong subtiles with warp specialization
  and register-resident mma.sync all land there. ncu shows it latency-bound
  (one softmax row per thread, about 10% occupancy, long-scoreboard and
  barrier stalls). Going further needs a hand-written CuTe kernel.
- Triangle multiplication output side (LayerNorm of the einsum, output
  projection, gating, residual) as one Triton GEMM, with the LayerNorm
  applied in registers (613 vs 360 us) or folded into the GEMM epilogue via
  per-position statistics (418-438 us): the separate kernels are already
  near their bandwidth bound (about 310 us); with g and the residual still
  read, the fused floor is about 200 us.
- flash_attn_4 for triangle attention: the container lacks its CuTe DSL
  dependency, and its bias would be a per-element gather in `score_mod`.

Tried and dropped earlier: cuEquivariance gated-GEMM tile configs beyond its tuning
space (no gain), torch.compile'd SwiGLU (slower than the Triton kernel),
row chunks under 256k rows in the transition (slower), non-persistent
Triton GEMMs for the SwiGLU and gating fusions (slower than cuBLAS plus a
separate kernel), the windowed atom attention in full FP32 (Triton's IEEE
FP32 dot is slower than SDPA plus the window copies, so that kernel is used
with TF32 only).
