# Installation

Mol-1 supports CPython 3.12 and 3.13. (Biotite 1.7, which the structure
parser is pinned to, requires 3.12 or later.)

## Tested configuration

Mol-1 is tested on Linux x86_64 with PyTorch 2.12.0+cu132, cuEquivariance
0.10.0 and Triton 3.7.0 on an NVIDIA GPU, the versions `requirements.txt`
pins.
The extras accept a range: `cuda` resolves to the newest PyTorch 2.13 and its
Triton, `cuda12` to PyTorch 2.10 with Triton 3.6. PyTorch 2.14 is excluded:
with these cuEquivariance wheels, fine-tuning fails in the backward pass.
`mol doctor` reports what the installed build can do.

On CPU, Mol-1 runs the PyTorch triangle path.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[cuda]"            # NVIDIA GPU, CUDA 13
pip install -e ".[cuda12]"          # NVIDIA GPU, driver too old for CUDA 13
pip install -e ".[cpu]"             # CPU
```

Nothing has to be matched up by hand. The fused cuEquivariance kernels come as
CUDA-major-version-specific wheels, and PyTorch's default wheel on PyPI is too:
torch 2.10 and earlier ship CUDA 12, 2.11 and later ship CUDA 13. Each extra
pins the torch range whose default build matches its operator wheels, so a
mismatch is a resolver error rather than an import failure with no explanation.

To pick a specific PyTorch build anyway — a different patch version, or a wheel
from PyTorch's own index — install it before this package and the extra will
accept it as long as it falls in range:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[cuda]"
```

Optional, only for `triangle_multiplicative=flashpairformer`: `opt_core`, from
[anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
under Apache-2.0. If it is already importable -- preinstalled in an image, say --
there is nothing to do; otherwise install the revision `requirements.txt` pins,
or point `OPT_CORE_DIR` at a source checkout. [kernels.md](kernels.md) has the
details.

The `cpu` extra installs nothing — it names the intent, since the PyTorch
triangle path is the only backend on CPU.

## Verify

```bash
mol doctor
```

`mol doctor` loads no model and compiles nothing. The first prediction or
training run on a CUDA machine compiles the fused LayerNorm extension, which
takes a couple of minutes with no output but compiler lines; it is cached per
environment afterwards. On CPU the PyTorch LayerNorm is used and nothing is
compiled.

The report names the PyTorch build, every CUDA device with its compute
capability, whether the build covers that architecture, which backend
`--device auto` and `triangle_* = auto` resolve to, the preprocessing binaries,
and the runtime caches it found:

```text
Mol-1 environment
- Python: 3.12.3 (/usr/bin/python3)
- Platform: Linux-6.8.0-71-generic-x86_64-with-glibc2.39
- PyTorch: 2.12.0+cu132
- torch CUDA version: 13.2
- torch.cuda.is_available: True
- CUDA device count: 1
- torch CUDA arch list: sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
- CUDA device 0: NVIDIA ..., compute capability 12.0, 95 GiB -- covered by this build
- Apple MPS: built=False available=False
- nvidia-smi: NVIDIA ..., 580.159.04, 97887 MiB
- GPU optional packages:
  - cuequivariance_torch: 0.10.0
  - cuequivariance_ops_torch: 0.10.0
  - triton: 3.7.0
- device=auto resolves to: cuda:0
- triangle_*=auto resolves to: cuequivariance
- Preprocessing binaries (MSA and templates):
  - hmmsearch: /usr/bin/hmmsearch
  - hmmbuild: /usr/bin/hmmbuild
  - nhmmer: /usr/bin/nhmmer
  - hmmalign: /usr/bin/hmmalign
  - kalign: /usr/bin/kalign
- Downloader for the shared caches: none (the download script falls back to Python)
- MOL_ROOT_DIR: /path/to/data
  - checkpoint/mol-1-base.pt: present
  - common/components.cif: present
  - common/components.cif.rdkit_mol.pkl: present
- TORCH_EXTENSIONS_DIR: unset (PyTorch default)
- LAYERNORM_TYPE: unset (fast LayerNorm)
- opt_core (FlashPairformer TMK3): unavailable: ModuleNotFoundError: No module named 'opt_core'
  - OPT_CORE_DIR: unset
```

Two lines to read first:

- **`torch CUDA arch list`** against your device's compute capability. If the
  architecture is absent and there is no `compute_XX` PTX entry to JIT from,
  inference fails on a bare matmul with `no kernel image is available`. That is
  the wrong wheel, not a broken install.
- **GPU optional packages**. `installed but not importable` usually means a
  cu12 operator wheel on a CUDA 13 torch, or the reverse.

## Runtime data and weights

Set `MOL_ROOT_DIR` to a data root (it defaults to the home directory) and fetch
the reference data into it once with `bash scripts/database/download_mol_data.sh`.
`mol cpred` uses `$MOL_ROOT_DIR/checkpoint/mol-1-constraint.pt`, downloading
the weights there on first prediction if missing. Custom weights are passed with
`--load_checkpoint_path`. [data_sources.md](data_sources.md) lists every archive
and file. The scripts under `scripts/` are not included in the wheel; run them
from a checkout.

## Preprocessing binaries

Protein MSAs come from the ColabFold MMseqs2 service and need no local
programs. The rest shell out to:

- `kalign`, whenever `--use_template` is on, including for an input JSON that
  already carries `templatesPath`: every template hit is aligned to the query.
- HMMER's `hmmsearch` and `hmmbuild`, to search for templates for an input
  without `templatesPath`.
- HMMER's `nhmmer`, `hmmalign` and `hmmbuild`, with `--use_rna_msa` on, for RNA
  chains without an MSA.

On Debian or Ubuntu, `sudo apt-get install hmmer kalign` installs both. In a
conda environment,
`conda install -c conda-forge -c bioconda --override-channels hmmer kalign3=3.3.5`
does; `--override-channels` keeps conda to those two channels, so Miniconda
does not stop to ask for the Anaconda channels' terms of service. Keep the
`kalign3` pin: it installs the same Kalign as the Ubuntu 24.04 package, while
later Kalign releases align template hits differently.
`mol doctor` reports which are on `PATH`.
