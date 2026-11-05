# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locations of the released weights and reference data.

:data:`RELATIVE_PATH` is the layout under ``MOL_ROOT_DIR``. The checkpoint is
also served at that path under :data:`RELEASE_URL` and downloaded when missing;
reference data is never downloaded by a run. ``docs/data_sources.md``
describes the files and how to get them.
"""

from typing import Iterable, Mapping

__all__ = [
    "RELATIVE_PATH",
    "RELEASE_URL",
    "ReferenceDataMissing",
    "checkpoint_key",
    "checkpoint_url",
    "missing_reference_data",
]

#: Host of the public release: the checkpoint and the archives
#: ``common.tar.gz``, ``search_database.tar.gz`` and ``mmcif_msa_template.tar.gz``.
RELEASE_URL = "https://mol-1-release.oss-cn-shanghai.aliyuncs.com"

#: Logical name -> path of that file relative to ``MOL_ROOT_DIR``. The
#: checkpoint is served at the same path under :data:`RELEASE_URL`.
RELATIVE_PATH: Mapping[str, str] = {
    # Released weights, one entry per registered model configuration.
    "checkpoint_mol-1-base": "checkpoint/mol-1-base.pt",
    "checkpoint_mol-1-constraint": "checkpoint/mol-1-constraint.pt",
    # Always needed: chemical component definitions and PDB bookkeeping.
    "ccd_components_file": "common/components.cif",
    "ccd_components_rdkit_mol_file": "common/components.cif.rdkit_mol.pkl",
    "pdb_cluster_file": "common/clusters-by-entity-40.txt",
    "obsolete_release_data_csv": "common/obsolete_release_date.csv",
    # Needed only when template features are switched on.
    "obsolete_pdbs_path": "common/obsolete_to_successor.json",
    "release_dates_path": "common/release_date_cache.json",
    # Sequence databases for the optional on-the-fly searches.
    "template_search_database": "search_database/pdb_seqres_20261001.fasta",
    "rfam_search_database": (
        "search_database/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
    ),
    "rnacentral_search_database": (
        "search_database/rnacentral_active_seq_id_90_cov_80_linclust.fasta"
    ),
    "nt_rna_search_database": (
        "search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
    ),
}


class ReferenceDataMissing(FileNotFoundError):
    """Reference files a run needs are not under ``MOL_ROOT_DIR``."""


def missing_reference_data(paths: Iterable[str]) -> ReferenceDataMissing:
    """Build the error for missing reference files, saying how to get them.

    Args:
        paths (Iterable[str]): The files that are missing.

    Returns:
        ReferenceDataMissing: The error to raise.
    """
    listed = "".join(f"  {path}\n" for path in paths)
    return ReferenceDataMissing(
        f"Reference data is missing:\n{listed}"
        "Fetch it into MOL_ROOT_DIR once with "
        "`bash scripts/database/download_mol_data.sh` (archives at "
        f"{RELEASE_URL}), or put the files there yourself."
    )


def checkpoint_key(model_name: str) -> str:
    """Return the :data:`RELATIVE_PATH` key holding a model's weights.

    Args:
        model_name (str): A registered model configuration name.

    Returns:
        str: The corresponding key.
    """
    return f"checkpoint_{model_name}"


def checkpoint_url(model_name: str) -> str:
    """Return the URL of a registered model's released weights.

    Args:
        model_name (str): A registered model configuration name.

    Returns:
        str: Absolute URL of the checkpoint.

    Raises:
        KeyError: If no weights are released for that model.
    """
    try:
        relative = RELATIVE_PATH[checkpoint_key(model_name)]
    except KeyError:
        raise KeyError(
            f"No released weights for model {model_name!r}; pass "
            "--load_checkpoint_path."
        ) from None
    return f"{RELEASE_URL}/{relative}"
