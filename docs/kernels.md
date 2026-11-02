# Kernels

Three operations have more than one implementation: layer normalization, the
triangle multiplicative update and triangle attention. The choice changes how
fast a prediction runs and how much memory it needs, not what the model
computes, although backends round differently, so outputs are not bitwise
identical across them.

## Which to use

| Situation | Settings |
| --- | --- |
| NVIDIA GPU, up to about 1,500 tokens | the defaults |
| NVIDIA GPU, faster | `--trimul_kernel flashpairformer --triatt_kernel triattention`, after installing `opt_core` (below): 11–22% faster at the same memory |
| NVIDIA GPU, more than about 1,500 tokens | the [large-input settings](inference_memory.md#large-inputs), which include `--trimul_kernel torch` |
| CPU | the defaults; `auto` resolves to PyTorch |
| GPU of compute capability 7.x, such as the V100 | the defaults; both operations run on PyTorch in FP32 |
| Any GPU | never `--triatt_kernel torch`: it is slower and runs out of memory from about 1,000 tokens |
| Fine-tuning | the defaults; `flashpairformer` is inference only |

`mol pred` defaults both triangle operations to `auto`, which resolves to
cuEquivariance on CUDA when its operator wheels import and to PyTorch
otherwise. `mol doctor` prints what `auto` resolves to on the machine it runs
on.

## Measurements

Measured with the `mol pred` defaults (200 diffusion steps, 5 samples
,no templates) but with 10 cycles and  TF32 on, on one 95 GiB NVIDIA 
Blackwell GPU (compute capability 12.0), for three FoldBench inputs 
in `examples/example_without_msa.json`. Each cell is the model forward 
time and the PyTorch peak memory.

| `--trimul_kernel` / `--triatt_kernel` | 8cnh, 509 tokens | 8j2w, 1,000 tokens | 8h0q, 1,499 tokens |
| --- | --- | --- | --- |
| `cuequivariance` / `cuequivariance` (default) | 44 s, 9.3 GiB | 109 s, 28.1 GiB | 259 s, 56.0 GiB |
| `cuequivariance` / `triattention` | 43 s, 9.3 GiB | 107 s, 27.9 GiB | 248 s, 56.0 GiB |
| `flashpairformer` / `cuequivariance` | 39 s, 9.5 GiB | 89 s, 28.3 GiB | 211 s, 56.3 GiB |
| `flashpairformer` / `triattention` | 39 s, 9.5 GiB | 87 s, 28.2 GiB | 201 s, 56.3 GiB |
| `torch` / `cuequivariance` | 46 s, 9.3 GiB | 119 s, 28.1 GiB | 278 s, 51.9 GiB |
| `torch` / `triattention` | 45 s, 9.3 GiB | 114 s, 27.9 GiB | 268 s, 51.6 GiB |
| any / `torch` | 59–66 s, 21.3–21.5 GiB | out of memory | out of memory |

- FlashPairformer is the backend that pays: 12–19% faster than the default on
  its own and 11–22% with `triattention`, more so the larger the input, at the
  same memory. `triattention` alone adds nothing at 509 tokens and 4–5% at
  1,499.
- The PyTorch triangle multiplication costs 5–9% and, at 1,499 tokens, saves
  4 GiB; past about 1,500 tokens it is what fits. With the
  [large-input settings](inference_memory.md#large-inputs), the two largest
  FoldBench inputs completed: 9f37 (2,456 tokens) in 1,371 s at 83.3 GiB and
  7yvb (2,532 tokens) in 1,568 s at 87.1 GiB.
- The PyTorch triangle attention is 40–50% slower already at 509 tokens and
  runs out of memory at 1,000.
- TF32 is off by default. Without it, the default combination is 4–8% slower
  at the same memory; `--enable_tf32 true` buys that back, with slightly
  different results.

## Selecting a backend

| Setting | `mol pred` option | Values | Config default |
| --- | --- | --- | --- |
| `triangle_multiplicative` | `--trimul_kernel` | `auto`, `cuequivariance`, `flashpairformer`, `torch` | `cuequivariance` |
| `triangle_attention` | `--triatt_kernel` | `auto`, `cuequivariance`, `triattention`, `metal`, `torch` | `cuequivariance` |
| `pairformer_backend` | `--pairformer_backend` | `torch`, `mlx` | `torch` |

## Apple silicon

`--triatt_kernel metal` and `--pairformer_backend mlx` are opt-in, FP32-only,
inference-only options for `--device mps`. The MLX backend needs the `mlx`
extra (`pip install "molence[mlx]"`).

```bash
mol pred -i input.json -o ./output --device mps -d fp32 --triatt_kernel metal --pairformer_backend mlx
```

The config defaults in [mol/config/base.py](../mol/config/base.py) are what
`inference_demo.sh`, `finetune_demo.sh` and `mol/runner/train.py` use;
`mol pred` defaults to `auto` instead, because it is also the entry point on
machines without a GPU.

```bash
mol pred -i input.json -o ./output --trimul_kernel flashpairformer --triatt_kernel triattention
bash inference_demo.sh --triangle_multiplicative torch --triangle_attention triattention
TRIANGLE_MULTIPLICATIVE=flashpairformer bash inference_demo.sh
```

`auto` never picks `flashpairformer`, which has to be installed on purpose. A
requested backend that cannot be used is an error, not a downgrade. Two
substitutions are made by design: on compute capability 7.x both operations
run on PyTorch in FP32, and the log says so; and `flashpairformer` hands the
calls it cannot take to cuEquivariance, as described below.

## Triangle multiplicative update

**`cuequivariance`**: fused kernels from NVIDIA's
[cuEquivariance](https://github.com/NVIDIA/cuEquivariance), installed by the
`cuda` extra (`cuda12` for a CUDA 12 PyTorch). CUDA only; training and
inference. The operator wheels are tied to a CUDA major version: a `-cu12`
wheel on a CUDA 13 PyTorch fails at import.

**`flashpairformer`**: the fused TMK3 kernel from `opt_core`. Inference only,
and rejected during training. It serves CUDA calls computed in BF16 on inputs
longer than cuEquivariance's short-sequence threshold (at least 100 tokens);
every other call goes to cuEquivariance on CUDA, or to PyTorch elsewhere. The
confidence head runs in FP32 up to 2,560 tokens, so below that size its
triangle multiplications fall back to cuEquivariance.
`flashpairformer_stats(model)` counts the calls served and the reasons for
each fallback. A call it should serve whose dependency is missing, or whose
kernel fails, raises an error.

`opt_core` comes from
[anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
(Apache-2.0; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) and is not
on PyPI. Install the revision `requirements.txt` pins, or point `OPT_CORE_DIR`
at a source checkout:

```bash
pip install "opt_core @ git+https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e#subdirectory=common/opt_core"
# or
OPT_CORE_DIR=/path/to/opt_core mol pred -i input.json -o ./output --trimul_kernel flashpairformer
```

It also uses the cuEquivariance operator package, so it needs the `cuda` or
`cuda12` extra as well.

**`torch`**: the PyTorch implementation. Every device, training and
inference. During inference it updates the pair representation in place, one
chunk at a time, which no fused kernel does; that is why the large-input
settings use it.

## Triangle attention

**`cuequivariance`**: as above.

**`triattention`**: the kernel in
[mol/model/tri_attention/](../mol/model/tri_attention/). The first time it
meets a shape, it picks a tile size by timing the candidates and caches the
choice in `mol/model/tri_attention/TriAttentionCache/<rank>/` inside the
installed package; that first call is slower, and the choice can differ
between machines, so bitwise-identical results across machines need the same
cache.

**`torch`**: the PyTorch implementation. Every device, but on a GPU it needs
far more memory than the other two (see [Measurements](#measurements)).

## Layer normalization

A fused CUDA LayerNorm, adapted from [FastFold](https://github.com/hpcaitech/FastFold)
and [OneFlow](https://github.com/Oneflow-Inc/oneflow), is used by default on
CUDA. It is compiled when a model is first built on a CUDA machine, which takes
a couple of minutes and prints only compiler output, and cached under
`TORCH_EXTENSIONS_DIR` (by default PyTorch's extension cache, in a `mol`
subdirectory), keyed on the sources, the compiler flags, the GPU architecture
and the Python, PyTorch, CUDA and `nvcc` versions. Tensors that are not on a
CUDA device use the PyTorch implementation, and a machine without CUDA
compiles nothing.

```bash
export LAYERNORM_TYPE=torch   # use the PyTorch implementation instead
```
