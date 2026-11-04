# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
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
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: protenix/web_service/colab_request_parser.py
# Changes: Reduced to protein MSA search against the ColabFold MMseqs2 API; a complex's paired MSA is used.

import os
import shutil
from pathlib import Path
from typing import Sequence

from mol.web_service.colab_request_utils import (
    parse_fasta_string,
    run_mmseqs2_service,
    validate_msa_query,
)

# Public MMseqs2 endpoint of ColabFold.
COLABFOLD_SERVICE_HOST_URL = "https://api.colabfold.com"


class RequestParser(object):
    """Protein MSA search through a ColabFold MMseqs2 service."""

    @staticmethod
    def msa_search(
        seqs_pending_msa: Sequence[str],
        tmp_fasta_fpath: str,
        msa_res_dir: str,
        email: str = "",
    ) -> list:
        """Search each sequence, and the sequences together as a complex.

        Every sequence gets ``<msa_res_dir>/<i>/<i>/non_pairing.a3m``. With more
        than one sequence, the paired search of the complex is written beside
        it as ``pairing.a3m``, one row per hit that pairs across the chains.

        Returns:
            list: The result directory of every sequence that was searched.
        """
        lines = []
        for idx, seq in enumerate(seqs_pending_msa):
            lines.append(f">query_{idx}\n")
            lines.append(f"{seq}\n")
        if (last_line := lines[-1]).endswith("\n"):
            lines[-1] = last_line.rstrip("\n")
        with open(tmp_fasta_fpath, "w") as f:
            for line in lines:
                f.write(line)

        with open(tmp_fasta_fpath, "r") as f:
            query_seqs = f.read()
        host_url = COLABFOLD_SERVICE_HOST_URL
        chain_dirs = []
        fasta_dict = parse_fasta_string(query_seqs)
        for i, (seq_name, seq) in enumerate(fasta_dict.items()):
            print(f"Searching MSA for {seq_name} with the sequence itself.")
            chain_dirs.append(
                run_mmseqs2_service(
                    f">{seq_name}\n{seq}",
                    os.path.join(msa_res_dir, str(i)),
                    use_env=True,
                    use_filter=True,
                    use_templates=False,
                    filter=None,
                    use_pairing=False,
                    pairing_strategy="greedy",
                    host_url=host_url,
                    user_agent="colabfold/1.5.5",
                    email=email,
                )
            )
        if len(fasta_dict) > 1:
            complex_dir = os.path.join(msa_res_dir, "complex")
            run_mmseqs2_service(
                query_seqs,
                complex_dir,
                use_env=True,
                use_filter=True,
                use_templates=False,
                filter=None,
                use_pairing=True,
                pairing_strategy="greedy",
                host_url=host_url,
                user_agent="colabfold/1.5.5",
                email=email,
            )
            paired_paths = [
                os.path.join(complex_dir, str(i), "pairing.a3m")
                for i in range(len(chain_dirs))
            ]
            # Check all sources before replacing any returned paired file.
            for paired, seq in zip(paired_paths, fasta_dict.values()):
                validate_msa_query(paired, seq)
            if set(Path(complex_dir).glob("*/pairing.a3m")) != {
                Path(path) for path in paired_paths
            }:
                raise ValueError(
                    "ColabFold paired MSA query set does not match the request. "
                    "Use a new result directory."
                )
            # The paired search writes complex/<i>/pairing.a3m; the input JSON
            # takes each chain's MSAs from that chain's own result directory.
            for paired, chain_dir in zip(paired_paths, chain_dirs):
                shutil.copyfile(paired, os.path.join(chain_dir, "pairing.a3m"))
        elif os.path.exists(os.path.join(chain_dirs[0], "pairing.a3m")):
            raise ValueError(
                "Cached paired MSA output does not match a single-chain search. "
                "Use a new result directory."
            )
        return chain_dirs
