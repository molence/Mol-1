# Third-party notices

Mol-1 is released under the Apache License 2.0; see [LICENSE](LICENSE).

## Protenix

Most of this repository derives from
[Protenix](https://github.com/bytedance/Protenix) by the ByteDance AML
AI4Science team, also under Apache-2.0, with the upstream copyright notice:
"Copyright 2024 ByteDance and/or its affiliates." Those files keep their original
copyright notices and license headers, unchanged. Files modified here keep the
original notices and add a `Modifications Copyright 2026 Molence Bio`
line, followed by a statement that the file was modified, the upstream revision
and the upstream path; significant changes are summarized in a `Changes:` line.
Files added here, or rewritten so that no upstream code remains, carry only a
Molence Bio copyright. Renaming the product does not remove source attribution.

The upstream revision these files are based on is
`bd54a05d047b8925a241056f36d619700604068a`.

Protenix itself builds on ideas and components from the AlphaFold 3 ecosystem,
including AlphaFold 3, OpenFold and ColabFold.

### OneFlow, FastFold and OpenFold

The LayerNorm operators in `mol/model/layer_norm/` include code adapted from
[OneFlow](https://github.com/Oneflow-Inc/oneflow) and
[FastFold](https://github.com/hpcaitech/FastFold), via Protenix. Other
implementations derive from [OpenFold](https://github.com/aqlaboratory/openfold).
These projects use Apache-2.0; third-party components retain their
respective licenses and copyright notices.

## FlashPairformer TMK3 (`opt_core`)

`mol/model/triangular/flashpairformer.py` is an adapter written for
Mol-1. It is not part of the Protenix code imported here.
Copyright 2026 Molence Bio, licensed under Apache-2.0.

The adapter calls the separately installed `opt_core` package, from
[anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
(`common/opt_core`). Its original code is Copyright 2026 Anthropic, PBC and
licensed under Apache-2.0. Kernels, headers and code portions it adapts from
third parties are listed in that package's own `THIRD_PARTY_NOTICES.md`, with
their license texts under `third_party_licenses/`, and keep their upstream
licenses.

No copy of `opt_core` is vendored here. `requirements.txt` pins the revision
used to verify this backend (`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`). It is
deliberately not a dependency in `pyproject.toml`: PyPI refuses packages whose
dependencies are direct URLs, and `opt_core` is not published there.

Anyone distributing `opt_core` -- including inside a container image -- must
carry its `LICENSE`, its `THIRD_PARTY_NOTICES.md` and its
`third_party_licenses/` with it. Its license notes that model weights are not
part of that library or of any image built from it.

Selecting this kernel is optional: `triangle_multiplicative=flashpairformer`.
The default path does not require `opt_core`.

## cuEquivariance

The fused triangle kernels come from NVIDIA's `cuequivariance` packages,
installed through the `cuda` or `cuda12` extra and not vendored here. They are optional:
`triangle_* = auto` resolves to the PyTorch path when they are absent.

## CUTLASS

`mol/model/triangular/tri_fmha/csrc/{collective,kernel,device}/` are copied
from NVIDIA CUTLASS 4.6 (`examples/77_blackwell_fmha`), Copyright (c) 2024 -
2026 NVIDIA CORPORATION & AFFILIATES, under the BSD-3-Clause license whose text
each file keeps in its header. Three of them were modified for Mol-1's
triangle attention (pair bias on the tensor cores, output gate in the
epilogue); each says so below its license. The CUTLASS library headers they
include are not vendored: the kernel is compiled at first use against a
CUTLASS checkout (`MOL_CUTLASS_DIR`, or the one in NVIDIA's PyTorch
containers) and is skipped when none is found.

## Data and models

No structural data or MSA databases are distributed from this repository. The
CCD and template caches and the precomputed MSAs and template hits fetched by
`scripts/database/download_mol_data.sh` derive from their upstream publishers'
data under those publishers' terms, as does any search database used for MSA or
template preprocessing. Check those terms before redistributing anything they
produce.
