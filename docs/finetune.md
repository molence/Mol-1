# Fine-tuning on your own structures

Fine-tuning starts from structures you have — mmCIF files from the PDB or
from your own experiments — which the scripts in this repository turn into a
training set:

1. [Structures to bioassembly dicts](#1-structures-to-bioassembly-dicts) — `scripts/prepare_training_data.py`
2. [MSAs](#2-msas) (recommended) — `scripts/prepare_finetune_msa.py`
3. [Fine-tune](#3-fine-tune) — `finetune_demo.sh`
4. [Predict with the result](#4-predict-with-the-fine-tuned-weights)

The commands below run the whole flow from the repository root on six example
structures: the five under `scripts/msa/data/mmcif/` (single proteins, three with
ligands) and the barnase–barstar complex in `examples/finetune/1brs.cif`.

## Before you start

- **A checkpoint.** `CHECKPOINT_PATH`, or `$MOL_ROOT_DIR/checkpoint/mol-1-base.pt`.
- **`MOL_ROOT_DIR` with `common/`**, the same reference files a prediction
  uses: `bash scripts/database/download_mol_data.sh` fetches them
  ([data_sources.md](data_sources.md)). Preparing structures reads the
  chemical component dictionary from it; `common/clusters-by-entity-40.txt`
  provides the sequence clusters that weight sampling. Those clusters cover PDB
  entries only: the protein chains of 10 or more residues in your own
  structures all share one cluster, so sampling does not balance them by
  sequence.
- **A CUDA GPU and the `cuda` (or `cuda12`) extra** for step 3, which uses the
  cuEquivariance kernels ([installation.md](installation.md)). The default recipe (384-token crops, 48 diffusion
  samples per structure) fits on a single 95 GiB GPU; step 3 says what to lower
  on a smaller one.

```bash
export MOL_ROOT_DIR=/path/to/data
export CHECKPOINT_PATH=/path/to/mol-1-base.pt
```

## 1. Structures to bioassembly dicts

```bash
mkdir -p finetune_data
ls scripts/msa/data/mmcif/*.cif examples/finetune/*.cif > finetune_data/structures.txt
python3 scripts/prepare_training_data.py \
  -i finetune_data/structures.txt \
  -b finetune_data/bioassembly \
  -o finetune_data/indices.csv \
  -c "$MOL_ROOT_DIR/common/clusters-by-entity-40.txt" \
  -n 4
```

`-i` takes a directory of `.cif` files or a text file listing their paths.
Each structure that passes the filters becomes `finetune_data/bioassembly/<pdb_id>.pkl.gz`,
and `indices.csv` lists its chains and interfaces, which are the units
training samples from. For the six examples this takes about 15 seconds and
gives 17 entries: seven protein chains, four ligand chains and six interfaces,
one of them the barnase–barstar protein–protein interface.

By default the PDB training filters apply (waters, hydrogens and clashing
chains removed, assembly 1 expanded, large assemblies cut down to the chains
around one interface). For predicted
or designed structures that should be used as they are, add `-d`.
[prepare_training_data.md](prepare_training_data.md) lists the filters and
describes both output formats, and covers refreshing the component dictionary
for ligands newer than its cutoff.

## 2. MSAs

The model was trained with MSAs and depends on them, so fine-tuning without
them changes what it learns. This step searches every protein sequence of the
dataset with the same service `mol msa` uses at inference, and writes the index
training reads:

```bash
python3 scripts/prepare_finetune_msa.py \
  -b finetune_data/bioassembly \
  -o finetune_data/msa
```

The sequences are searched on the public ColabFold MMseqs2 service
([data_sources.md](data_sources.md#msa-search-service)). The six distinct
sequences of the examples take about two minutes. Each sequence gets a
directory under `finetune_data/msa/`, and `finetune_data/msa/seq_to_msa_dir.json`
maps the sequence to it. Training reads `pairing.a3m` and `non_pairing.a3m`
from that directory; each sequence is searched on its own, so only the
unpaired file is written and chains are not paired across species.
A failed search is reported and can be retried by running the command again —
sequences already indexed are skipped.

Skipping this step is allowed: without `seq_to_msa_dir.json`, step 3 trains
without MSAs, each chain's MSA holding only its own sequence.

Only protein MSAs are prepared. RNA MSAs and templates stay off during
fine-tuning: the training loader reads template structures from a local PDB
mirror and RNA MSAs in a layout these scripts do not produce. The model was
trained with templates whenever they were available, so a long run without
them may weaken how it uses templates at prediction time; check predictions
with templates on before relying on such a run.

## 3. Fine-tune

```bash
bash finetune_demo.sh --max_steps 6 --checkpoint_interval 3 --log_interval 1
```

`finetune_demo.sh` loads the checkpoint strictly and starts a new run: fresh
optimizer and scheduler, EMA initialized from the loaded weights. It reads the
dataset from `FINETUNE_DATA_DIR` (default `./finetune_data`) and uses the MSA
index when it is there. Flags after the script override its defaults, which are
the full recipe:

| Setting | Default | Flag |
| --- | --- | --- |
| Crop / diffusion samples | 384 tokens / 48 | `--train_crop_size`, `--diffusion_batch_size` |
| Learning rate / warm-up | 0.001 / 2000 steps | `--lr`, `--warmup_steps` |
| Steps / checkpoint every | 100000 / 400 | `--max_steps`, `--checkpoint_interval` |
| Recycles (sampled per step) | up to 4 | `--model.N_cycle` |
| EMA decay | 0.999 | `--ema_decay` |

On a smaller GPU, lower `--diffusion_batch_size` first, then
`--train_crop_size`. `DRY_RUN=1 bash finetune_demo.sh` prints the full command
without running it.

Output goes to `$BASE_DIR/<run_name>_<timestamp>/` (default `./output`):

```text
output/mol_finetune_<timestamp>/checkpoints/
├── 2.pt              # weights, optimizer and scheduler at step 2
├── 2_ema_0.999.pt    # EMA weights at step 2
├── 5.pt
└── 5_ema_0.999.pt
```

The script configures no test set, so evaluation is off. To
evaluate during training, prepare a held-out set the same way, add an entry
for it to the `data` section of [mol/config/data.py](../mol/config/data.py),
and pass `--data.test_sets <name> --eval_interval <steps>`. Alternatively,
predict held-out structures with each saved checkpoint (step 4).

### Part of the dataset

To train on some of the prepared structures only, list their PDB IDs in a text
file, one per line, and pass it:

```bash
bash finetune_demo.sh \
  --data.weightedPDB_before210930_v20260101.base_info.pdb_list /path/to/subset.txt
```

### Several GPUs

Training uses PyTorch DDP. `finetune_demo.sh` runs on one GPU; to use several
GPUs on one machine, print its command with `DRY_RUN=1 bash finetune_demo.sh`
and launch `mol/runner/train.py` with `torchrun --nproc_per_node=<gpus>` in
place of `python3`. Each rank samples its own shard of the training set, only
rank 0 writes checkpoints, and at the end of the run `mol/runner/train.py`
checks that a sample of parameters agrees across ranks (`assert_replicas_agree`
in [mol/utils/distributed.py](../mol/utils/distributed.py)).

cuEquivariance's backward pass is not guaranteed to be bitwise deterministic,
so two training runs on identical inputs can differ. Forward inference is
unaffected.

## 4. Predict with the fine-tuned weights

The EMA checkpoint is the one to predict with:

```bash
mol pred -i examples/example.json -o ./output/finetuned \
  --load_checkpoint_path output/mol_finetune_<timestamp>/checkpoints/5_ema_0.999.pt
```

Use `mol pred` for base fine-tuned weights. `inference_demo.sh` runs constraint
inference and requires matching constraint weights; it cannot load a base checkpoint.

Whether fine-tuning on a given dataset improves the model depends on the data:
compare predictions on held-out structures from the fine-tuned and the base
checkpoint before relying on the result.
