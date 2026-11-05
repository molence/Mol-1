# MSA and template pipeline

How Mol-1 stores MSAs and templates, the MSA header format that pairing needs,
and the template and RNA MSA searches.

## MSA files

If you downloaded the precomputed MSAs with `bash scripts/database/download_mol_data.sh --full` (see [data_sources.md](./data_sources.md)), your data directory is organized as follows:

```bash
├── common
│   ├── seq_to_pdb_index.json # Mapping from sequence to directory ID
│   └── ...                   # Other metadata (CCD, release dates, etc.)
├── mmcif_msa_template                 # Root directory for MSA files
│   ├── 0
│   │   ├── pairing.a3m       # MSAs paired by taxonomy ID
│   │   ├── non_pairing.a3m   # Unpaired MSAs
│   │   └── hmmsearch.a3m     # Templates from HMMER search
│   ├── 1
│   │   ├── pairing.a3m
│   │   ├── non_pairing.a3m
│   │   └── hmmsearch.a3m
│   └── ...

```

Each unique protein sequence in the dataset is assigned an integer index (e.g., `0`, `1`, `2`). These indices correspond to the subdirectories under `mmcif_msa_template`. The `seq_to_pdb_index.json` file is used by the data loader to resolve a sequence to its corresponding directory ID.

The released MSAs were built with the scripts in `scripts/msa/`, around a local
ColabFold MMseqs2 search with a pinned UniRef30. To build MSAs for your own
structures, use `scripts/prepare_finetune_msa.py` instead, as
[finetune.md](finetune.md#2-msas) describes: it uses the same service as
inference and writes its own index, so the released `seq_to_pdb_index.json`
stays as it is.

### MSA header format

To enable correct taxonomic pairing, the headers in `pairing.a3m` must follow specific patterns. Mol-1 supports both UniRef and UniProt header formats.

**UniRef Format:**
```text
>UniRef100_{hit_name}_{taxonomy_id}/
```

**UniProt Format:**
```text
>tr|ACCESSION|ID_SPECIES/START-END [optional description] OS=Species Name OX=TaxonomyID ...
```

The system uses the following regex patterns to extract `SpeciesId`:

```python
_UNIPROT_REGEX = re.compile(
    r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)
_UNIREF_REGEX = re.compile(r"^UniRef100_[^_]+_([^_/]+)")
```

## Template search

Template search is used to find structural templates for protein sequences. It uses `hmmsearch` to search against the `pdb_seqres` database based on the profile of existing MSAs.

At prediction time (`--use_template true`), the hits released after 2021-09-30 are dropped and up to four are used. Each is aligned to the query with Kalign, and its structure is read from `$MOL_ROOT_DIR/mmcif/`, downloaded from PDBe on first use.

### Usage

The template search can be executed via:

```bash
python3 -m mol.runner.template_search
```

Run from the repository root, that searches the bundled example in `examples/5sak/1` and writes `hmmsearch.a3m` there; `mol prep` runs the search for an input JSON.

The core function `run_template_search` performs the following:
- Takes a directory containing `pairing.a3m` and `non_pairing.a3m` as input.
- Builds an HMM profile from these MSAs and searches the database.
- Outputs a merged `hmmsearch.a3m` file in the same directory.

### Database

The default database is `MOL_ROOT_DIR/search_database/pdb_seqres_20261001.fasta`, from the public `search_database.tar.gz` (`bash scripts/database/download_mol_data.sh`); another one can be passed with `--seqres_database_path` — see [data_sources.md](data_sources.md).

## RNA MSA search

Mol-1 provides a specialized pipeline for RNA MSA search using `nhmmer`.

### Usage

The module can be run on its own:

```bash
python3 -m mol.runner.rna_msa_search
```

That searches one bundled example sequence and writes the result under `./output`; `mol prep --use_rna_msa true` and `mol cpred --use_rna_msa true` run the search for the RNA chains of an input JSON that have no `unpairedMsaPath`.

The pipeline searches across three major RNA databases. You can use the clustered versions of the databases released by AlphaFold 3, available zstd-compressed (decompress with `zstd -d`) from `https://storage.googleapis.com/alphafold-databases/v3.0/<file name>.zst`, or provide your own custom databases:
1. **Rfam**: `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta`
2. **RNAcentral**: `rnacentral_active_seq_id_90_cov_80_linclust.fasta`
3. **NT-RNA**: `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta`

The databases are read from `$MOL_ROOT_DIR/search_database/` under the file names above; a database elsewhere can be passed with `--rfam_database_path`, `--rna_central_database_path` and `--ntrna_database_path`. See [data_sources.md](data_sources.md).

### Output

The search results from all databases are merged, deduplicated, and saved as `rna_msa.a3m`. `mol prep` and `mol cpred` write it to `<out_dir>/<job>/rna_msa/<entity index>/rna_msa.a3m`, where the entity index is the 0-based position in `sequences`, and fill it in as the chain's `unpairedMsaPath`; it is used only with `--use_rna_msa true`.

### Prerequisites

- **HMMER**: Ensure `hmmer` is installed and binaries (`nhmmer`, `hmmalign`, `hmmbuild`) are in your PATH.
- **Storage**: The RNA databases take tens of gigabytes.
