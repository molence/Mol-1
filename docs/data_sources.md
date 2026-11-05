# Reference data

Besides the weights, Mol-1 needs reference data that is not in this
repository. It is published as archives next to the weights, and
[scripts/database/download_mol_data.sh](../scripts/database/download_mol_data.sh)
fetches and extracts them into `$MOL_ROOT_DIR`. Every run needs `common/`: the
chemical component dictionary and its RDKit cache, PDB sequence clusters and
release tables. Template search, for inputs without `templatesPath`, also needs
`search_database/`. Template structures are not in the archives: each hit's
mmCIF is downloaded from PDBe
(`https://www.ebi.ac.uk/pdbe/entry-files/download/<id>.cif`) into
`$MOL_ROOT_DIR/mmcif/` the first time it is used.

| Archive | Size | Extracts to | Needed for |
| --- | --- | --- | --- |
| [common.tar.gz](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/common.tar.gz) | 229 MB (706 MB extracted) | `common/` — the six `common/` files in the table below plus `seq_to_pdb_index.json` | every run |
| [search_database.tar.gz](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/search_database.tar.gz) | 64 MB (391 MB) | `search_database/pdb_seqres_20261001.fasta` | template search |
| [mmcif_msa_template.tar.gz](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/mmcif_msa_template.tar.gz) | 9.8 GB (46 GB) | `mmcif_msa_template/<id>/{pairing,non_pairing,hmmsearch}.a3m` for 7,074 PDB sequences | training |

To fetch them:

```bash
export MOL_ROOT_DIR=/path/to/data
bash scripts/database/download_mol_data.sh          # common + search_database
bash scripts/database/download_mol_data.sh --full   # also mmcif_msa_template
```

The script uses `wget` or `curl` when either is installed and Python otherwise;
each resumes an interrupted download when the command is run again.

`mmcif_msa_template/` is what training reads by default: `common/seq_to_pdb_index.json`
maps each sequence to its directory, holding the paired and unpaired MSAs and
the template hits (`hmmsearch.a3m`). A sequence missing from the index has no
precomputed MSA; [finetune.md](finetune.md#2-msas) builds MSAs for your own
structures instead. The structures themselves — mmCIF files, bioassembly
dicts and training indices — are not in the archives.

## Individual files

These are the files a run looks for under `$MOL_ROOT_DIR`. The names on the
left are the keys in [mol/data/sources.py](../mol/data/sources.py). A run does
not download them: a missing file stops it with an error that lists every
missing file and points to the download script.

| Key | Path under `MOL_ROOT_DIR` | Needed for | Derived from |
| --- | --- | --- | --- |
| `ccd_components_file` | `common/components.cif` | every prediction | wwPDB Chemical Component Dictionary |
| `ccd_components_rdkit_mol_file` | `common/components.cif.rdkit_mol.pkl` | every prediction | the CIF above, converted to RDKit molecules with 3D conformers by [scripts/gen_ccd_cache.py](../scripts/gen_ccd_cache.py) |
| `pdb_cluster_file` | `common/clusters-by-entity-40.txt` | every prediction (checked at start-up); antibody handling, training weights | RCSB 40% sequence-identity entity clusters |
| `obsolete_release_data_csv` | `common/obsolete_release_date.csv` | every prediction (checked at start-up); release-date filtering | wwPDB entry status history |
| `obsolete_pdbs_path` | `common/obsolete_to_successor.json` | templates | wwPDB obsolete-entry mapping |
| `release_dates_path` | `common/release_date_cache.json` | templates | wwPDB entry release dates |
| `template_search_database` | `search_database/pdb_seqres_20261001.fasta` | template search | wwPDB `pdb_seqres`, snapshot 2026-10-01 |
| `rfam_search_database` | `search_database/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta` | RNA MSA | Rfam 14.9, clustered at 90% identity / 80% coverage |
| `rnacentral_search_database` | `search_database/rnacentral_active_seq_id_90_cov_80_linclust.fasta` | RNA MSA | RNAcentral active set, linclust at 90% / 80% |
| `nt_rna_search_database` | `search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta` | RNA MSA | NCBI `nt` RNA subset, snapshot 2023-02-23, clustered at 90% / 80% |

The archives do not include the three RNA databases. They are the clustered
sets released with AlphaFold 3, zstd-compressed at
`https://storage.googleapis.com/alphafold-databases/v3.0/<file name>.zst`
(decompress with `zstd -d`); the file names encode the clustering thresholds.
Any database in the same format can be substituted by passing its path
explicitly.

The CCD cache has a cutoff date. A component released after it is simply
missing, and an input using it fails rather than triggering a download; give
such a ligand as SMILES or a structure file instead, or regenerate the cache. [prepare_training_data.md](prepare_training_data.md) covers
regenerating the cache from a newer CIF.

## MSA search service

Protein MSAs are searched on the public ColabFold MMseqs2 service at
`https://api.colabfold.com`: each protein chain on its own, which gives its
unpaired MSA, and, when an input has several distinct protein chains, all of
them together, which gives the MSA paired across chains. No search runs when
the input JSON already carries MSA paths —
[infer_json_format.md](infer_json_format.md) describes those fields.

## Model weights

`mol cpred` uses `mol-1-constraint.pt`; `mol pred` uses `mol-1-base.pt`.
Weights resolve in this order, the first that applies winning:

1. `--load_checkpoint_path` — used as given. It must exist: naming a file says
   which weights to use, so fetching different ones instead would be wrong.
2. `$MOL_ROOT_DIR/checkpoint/<model_name>.pt` — used if it is there.
3. Otherwise they are downloaded to that path from
   [constraint checkpoint](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt)
   or [base checkpoint](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt),
   according to the selected model.

A download is verified by loading it before use; a truncated file is deleted
rather than left to fail later as an unintelligible unpickling error. Nothing
is ever re-downloaded once in place.
