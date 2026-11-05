*The following applies to evaluations of the Mol-1 base model.*

# Inference memory controls

## Sizing a GPU

Every successful prediction logs its token count next to the peak device
memory it used:

```
[Rank 0] 8j2w-assembly1 [seed:101] succeeded. Model forward time: 181.62s. 
N_token 1000, peak device memory 29427 MiB. Results saved to ./output
```

Run the three bundled systems in `examples/example_without_msa.json` and you
have a curve for your own card. For orientation, three FoldBench inputs with
the `mol pred` defaults (20 cycles, 200 diffusion steps, 5 samples,
cuEquivariance kernels, BF16 with the diffusion module and confidence head in
FP32) but TF32 on, on one 95 GiB NVIDIA Blackwell GPU:

| N_token | peak | forward |
| ------- | ---- | ------- |
| 509     | 9.3 GiB | 66 s |
| 1,000   | 28.7 GiB | 182 s |
| 1,499   | 58.5 GiB | 461 s |

The forward time is the model alone: loading the model, the first run's CUDA
build, featurization and any MSA or template search come on top.
[kernels.md](kernels.md#measurements) has the same inputs with every kernel
combination.

Peak grows close to quadratically in token count. Fit `peak ≈ a + b·N + c·N²` to your
own points; a pure `a + c·N²` fits small inputs poorly, so keep the linear
term.

**Do not reuse these numbers for another GPU.** The coefficients are
device-specific. Peak also shifts with every chunk setting below, so measure
in the configuration you will actually run.

Running out of memory is reported with the knobs to turn, in the order that
helps most: `--infer_setting.chunk_size`,
`--model.msa_module.msa_chunk_size`, then the sample count. `mol pred`
and `mol/runner/inference.py` accept every setting in the table below under the
same name, except `infer_setting.lddt_metrics_chunk_size`, which only applies
when labels are given; [`inference_demo.sh`](../inference_demo.sh) forwards
extra arguments to the runner. The sample count is `--sample` in `mol pred` and
`--sample_diffusion.N_sample` in the runner.

## What the controls are

Chunk sizes and caches are set explicitly. The runner makes one prediction attempt per
input and does not retry at smaller chunks after an out-of-memory error; if a
prediction does not fit, turn the knobs below and run it again.

| Setting | Controls | Default |
| --- | --- | --- |
| `infer_setting.chunk_size` | global attention chunk | 1024 |
| `infer_setting.dynamic_chunk_size` | picking the attention chunk from the token count instead: none up to 1024 tokens, then 512, 256 and 128 up to 1536, 2048 and 2560, and 32 above | off |
| `model.msa_module.msa_chunk_size` | MSA row chunk | 2048 |
| `infer_setting.msa_pair_transition_chunk_size` | pair transition inside the MSA module | 1024 |
| `infer_setting.trunk_pairmixer_transition_chunk_size`, `infer_setting.trunk_pairformer_transition_chunk_size` | pair transitions in the trunk | 128 |
| `infer_setting.confidence_pair_transition_chunk_size` | pair transitions in the confidence head | 128 |
| `infer_setting.sample_diffusion_chunk_size` | diffusion samples per batch | 1 |
| `infer_setting.diffusion_conditioning_row_chunk_size` | pair-conditioning rows | 768 |
| `infer_setting.lddt_metrics_chunk_size` | metric computation | 1 |
| `diffusion_cache_mode` | what the diffusion module reuses across steps | `full` |

`infer_setting.sample_diffusion_chunk_size` is 1, so the diffusion module
denoises the samples one at a time and its memory does not grow with the
sample count. When an input fits with room to spare, set it to the sample count
(`--infer_setting.sample_diffusion_chunk_size 5` with the default five samples)
to denoise them together, which is faster: on 9fm7 (644 tokens), the forward
pass took 43 s instead of 62 s at the same peak memory. Go back to 1 if the
input no longer fits.

The four stage-specific transition chunks are independent of the global
attention chunk. They are `ValueMaybeNone`: `None` inherits the effective pair
chunk for that stage, and if that is also `None`, Transition chunks
automatically. `diffusion_conditioning_row_chunk_size` is also
`ValueMaybeNone`, where `None` means one whole tensor rather than row chunks;
completed chunks stay on the device the input was on. All five accept `None`,
`none` and `null` from the CLI.

Precision also changes with the token count, and no setting overrides it: up
to 2560 tokens the confidence head runs in FP32, above that under BF16
autocast, and above 3840 tokens the diffusion module does too. Inputs just
under 2560 tokens are therefore the hardest to fit — a 2456-token input with
default settings runs out of memory in the confidence head on a 95 GiB GPU.
The settings under [Large inputs](#large-inputs) fit it.

`diffusion_cache_mode` selects how much work the diffusion module reuses
across denoising steps:

| Mode | Reuses |
| --- | --- |
| `full` (default) | pair conditioning, atom encoder shared features, per-block projected pair biases |
| `shared` | pair conditioning and atom encoder shared features |
| `off` | nothing; shared features are recomputed per denoising call |

Projected biases are detached and only cached under evaluation with gradients
disabled. During training, `enable_diffusion_shared_vars_cache` governs caching
instead, and this mode has no effect.

Above 2000 tokens, the PAE and PDE logits are moved to CPU as soon as they are
computed during evaluation, which frees that device memory before the next
sample; pLDDT and resolved logits stay on the compute device. Training keeps
all four on it. This is not configurable.

Example:

```bash
mol pred -i input.json -o ./output --infer_setting.chunk_size 512
bash inference_demo.sh --infer_setting.chunk_size 512
```

## Large inputs

The defaults fit inputs up to about 1,500 tokens on a 95 GiB GPU. Larger ones
need smaller chunks, the PyTorch triangle multiplication, whose update runs
chunked and in place where the cuEquivariance and FlashPairformer kernels have
no such path, and PyTorch's expandable memory segments. With these settings,
the two largest FoldBench inputs, of 2,456 and 2,532 tokens, complete on a
95 GiB GPU, peaking at 83 and 87 GiB:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mol pred -i input.json -o ./output \
  --trimul_kernel torch \
  --infer_setting.chunk_size 256 \
  --model.msa_module.msa_chunk_size 512 \
  --infer_setting.msa_pair_transition_chunk_size 256 \
  --infer_setting.trunk_pairmixer_transition_chunk_size 32 \
  --infer_setting.trunk_pairformer_transition_chunk_size 32 \
  --infer_setting.confidence_pair_transition_chunk_size 32 \
  --infer_setting.diffusion_conditioning_row_chunk_size 384
```

The allocator setting is not optional: without it, the same command fails on
a 2,456-token input with 20 GiB of device memory reserved but too fragmented to
hold one more 10 GiB pair tensor. The settings trade speed for memory, so keep
them for inputs that do not fit otherwise. For `mol/runner/inference.py`, the
kernel option is `--triangle_multiplicative torch`.
