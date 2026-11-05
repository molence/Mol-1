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
# Upstream file: runner/inference.py
# Changes: Device selection; a run with any failed prediction exits non-zero; C-terminal O/OXT rebuild taken from upstream revision 85767b8; ESM removed.
import json
import logging
import os
import time
import traceback
import urllib.request
from contextlib import nullcontext
from os.path import exists as opexists, join as opjoin
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from biotite.structure import AtomArray

from mol.config.base import configs as configs_base
from mol.config.data import data_configs
from mol.config.inference import inference_configs
from mol.config.model_type import model_configs
from mol.config.parser import parse_configs, parse_sys_args
from mol.data.inference.infer_dataloader import get_inference_dataloader
from mol.model.mol import Mol
from mol.utils.distributed import DIST_WRAPPER
from mol.utils.seed import seed_everything
from mol.utils.torch_utils import to_device
from mol.data.sources import checkpoint_url, missing_reference_data

from mol.runner.dumper import DataDumper

logger = logging.getLogger(__name__)


class InferenceRunner(object):
    """
    Runner class for AlphaFold3 model inference.
    Handles environment setup, model initialization, and running predictions.

    Args:
        configs (Any): Configuration object for inference.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        """
        Initialize the execution environment, including CUDA and distributed setup.
        """
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        requested = getattr(self.configs, "device", "auto")
        self.device = select_inference_device(requested, DIST_WRAPPER.local_rank)
        self.use_cuda = self.device.type == "cuda"
        if self.use_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        logging.info(f"Inference device: {self.device} (requested {requested!r})")

        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "fast_layernorm")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Finished environment initialization.")

    def init_basics(self) -> None:
        """
        Initialize basic directory structures for dumping results and errors.
        """
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """
        Initialize the Mol-1 model and move it to the appropriate device.
        """
        self.model = Mol(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        """
        Load model weights from a checkpoint file.

        Raises:
            FileNotFoundError: If an explicitly requested path does not exist.
        """
        # Same resolution as download_inference_cache, which normally ran
        # first: an explicit path wins, otherwise the default location, and the
        # released weights are fetched only if nothing is there.
        checkpoint_path = resolve_checkpoint(self.configs)

        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        sample_key = list(checkpoint["model"].keys())[0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model parameters: {count_parameters(self.model):.2f}M")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ) -> None:
        """
        Initialize the data dumper for saving predictions.

        Args:
            need_atom_confidence (bool): Whether to dump atom-level confidence.
            sorted_by_ranking_score (bool): Whether to sort results by ranking score.
        """
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    # Adapted from MolTrainer.evaluate in mol/runner/train.py
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Run model prediction on the provided data.

        Args:
            data (Mapping[str, Mapping[str, Any]]): Input data dictionary.

        Returns:
            dict[str, torch.Tensor]: Prediction results.
        """
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                mc_dropout_apply_rate=self.configs.mc_dropout_apply_rate,
            )
        return prediction

    def print(self, msg: str) -> None:
        """
        Print message only on the master rank (rank 0).

        Args:
            msg (str): Message to print.
        """
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        """
        Update the model's configuration.

        Args:
            new_configs (Any): New configuration object.
        """
        self.model.configs = new_configs


def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """Callback for tracking download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    status = f"\r[{bar}] {percent:.1f}%"
    print(status, end="", flush=True)

    if downloaded >= total_size:
        print()


def download_from_url(url: str, cache_path: str, check_weight: bool = False) -> None:
    """Download one file, optionally checking that it loads as a checkpoint.

    Args:
        url (str): Where to fetch from.
        cache_path (str): Where to write it.
        check_weight (bool): Load the result with ``torch.load`` and delete it
            again if that fails. A truncated download otherwise surfaces much
            later as an unintelligible unpickling error.

    Raises:
        RuntimeError: If ``check_weight`` is set and the file does not load.
    """
    urllib.request.urlretrieve(url, cache_path, reporthook=progress_callback)
    if not check_weight:
        return
    try:
        del_me = torch.load(cache_path, map_location="cpu", weights_only=False)
        del del_me
    except Exception as error:
        if opexists(cache_path):
            os.remove(cache_path)
        raise RuntimeError(
            f"The checkpoint downloaded from {url} does not load: {error}. "
            f"The partial file has been removed. Fetch it by hand to "
            f"{cache_path}, or pass --load_checkpoint_path."
        ) from error


def resolve_checkpoint(configs: Any) -> str:
    """Return the checkpoint to load, downloading it only as a last resort.

    An explicitly given ``load_checkpoint_path`` is used as-is and must exist:
    naming a file is a statement about which weights to use, so silently
    fetching different ones would be wrong. Otherwise the default location is
    used, and the released weights are downloaded there from the public
    release if nothing is there.

    Args:
        configs (Any): Inference configuration.

    Returns:
        str: Path to an existing checkpoint file.

    Raises:
        FileNotFoundError: If an explicitly requested path does not exist.
    """
    explicit = (getattr(configs, "load_checkpoint_path", "") or "").strip()
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(
                f"No checkpoint at [{explicit}]. Correct the path, or omit "
                "--load_checkpoint_path to use the released weights."
            )
        return explicit

    default_path = opjoin(configs.load_checkpoint_dir, f"{configs.model_name}.pt")
    if os.path.isfile(default_path):
        return default_path

    os.makedirs(configs.load_checkpoint_dir, exist_ok=True)
    source_url = checkpoint_url(configs.model_name)
    logger.info(
        "No checkpoint at %s; downloading the released weights from\n %s",
        default_path,
        source_url,
    )
    download_from_url(source_url, default_path, check_weight=True)
    return default_path


def download_inference_cache(configs: Any) -> None:
    """
    Check local model weights and the reference data inference reads.

    The weights are downloaded if they are missing. Reference data is not:
    every missing file is reported at once, with how to fetch the archives,
    before the weights are fetched.

    Args:
        configs (Any): Configuration object containing paths and model names.

    Raises:
        ReferenceDataMissing: If a reference file is not on disk.
    """

    required = [
        configs["data"][name]
        for name in (
            "ccd_components_file",
            "ccd_components_rdkit_mol_file",
            "pdb_cluster_file",
            "obsolete_release_data_csv",
        )
    ]
    if configs.use_template:
        required += [
            configs["data"]["template"][name]
            for name in ("obsolete_pdbs_path", "release_dates_path")
        ]
    missing = [path for path in required if not opexists(path)]
    if missing:
        raise missing_reference_data(missing)

    resolve_checkpoint(configs)


def update_inference_configs(configs: Any, n_token: int) -> Any:
    """
    Adjust inference configurations based on the number of tokens to avoid OOM.

    Args:
        configs (Any): Original configurations.
        n_token (int): Number of tokens in the sample.

    Returns:
        Any: Updated configurations.
    """
    # Adjust configurations based on sequence length to manage memory usage
    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    if getattr(configs, "bf16_confidence_head", False):
        configs.skip_amp.confidence_head = False
    if getattr(configs, "bf16_sample_diffusion", False):
        configs.skip_amp.sample_diffusion = False

    return configs


def select_inference_device(requested: str, local_rank: int = 0) -> torch.device:
    """Resolve a ``--device`` request to a concrete device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicitly requested
    device that is unavailable is an error rather than a silent downgrade: a
    run that quietly lands on CPU looks like a hang, not a misconfiguration.
    """
    requested = (requested or "auto").lower()
    cuda_ready = torch.cuda.device_count() > 0
    mps_ready = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )

    if requested == "auto":
        if cuda_ready:
            return torch.device(f"cuda:{local_rank}")
        if mps_ready:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not mps_ready:
            raise RuntimeError(
                "device=mps requested but this PyTorch build cannot use the Metal "
                f"backend (is_built={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built()}, "
                f"is_available={mps_ready}). Use --device cpu or --device auto."
            )
        return torch.device("mps")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_ready:
            raise RuntimeError(
                "device=cuda requested but torch.cuda reports no devices. "
                "Check the driver and the installed PyTorch build, or use "
                "--device cpu. `mol doctor` reports both."
            )
        return torch.device(requested if ":" in requested else f"cuda:{local_rank}")
    raise ValueError(
        f"device={requested!r} is unsupported; choose auto, cuda, cuda:N, mps or cpu."
    )


def fix_cterminal_carboxyl_oxygens(
    pred_coordinate: torch.Tensor,
    atom_array: AtomArray,
) -> torch.Tensor:
    """Rebuild the C-terminal carboxyl oxygens (O / OXT) of every polymer chain.

    Motivation: due to the O<->OXT substructure permutation symmetry, the model
    may leave one of the two carboxylate oxygens under-constrained (it can drift
    far away). For each polymer chain's last residue we regularize the two
    oxygens by an idealized, symmetric carboxylate geometry.

    Per C-terminal residue (identified by having both an ``O`` and an ``OXT``
    atom), using:
        A = C atom, C_ca = CA atom.
    We pick B = the oxygen (``O`` or ``OXT``) nearest to A, keep it and name it
    ``O``. The other oxygen D (``OXT``) is placed as the reflection of B across
    the A->C_ca axis, so that:
        * angle(C_ca, A, B) == angle(C_ca, A, D)   (equal angles at A)
        * |AB| == |AD|                             (equal bond lengths)
        * B, D and the A->C_ca axis are coplanar   (plane ABC)
    Reflection of vector v about unit axis u: v' = 2*(v.u)*u - v.

    The correction is fully vectorized over samples and over all terminal
    residues. Only the ``O`` and ``OXT`` coordinate slots are modified; all
    other atoms (and atom names) are left untouched.

    Args:
        pred_coordinate: predicted coordinates, shape ``[N_sample, N_atom, 3]``.
        atom_array: the AtomArray whose atom order matches ``pred_coordinate``.

    Returns:
        The corrected coordinate tensor (same shape/dtype/device as input).
    """
    n_atom = pred_coordinate.shape[-2]
    if len(atom_array) != n_atom:
        return pred_coordinate

    chain_ids = np.asarray(atom_array.chain_id)
    res_ids = np.asarray(atom_array.res_id)
    atom_names = np.asarray(atom_array.atom_name)

    # Locate the last residue of each chain: a chain block ends where chain_id
    # changes (chains are built as contiguous blocks, residues in order).
    is_chain_end = np.empty(n_atom, dtype=bool)
    is_chain_end[-1] = True
    is_chain_end[:-1] = chain_ids[1:] != chain_ids[:-1]
    chain_end_idx = np.nonzero(is_chain_end)[0]

    # For each chain's C-terminal residue, gather the C / CA / O / OXT indices.
    idx_c, idx_ca, idx_o, idx_oxt = [], [], [], []
    for end_i in chain_end_idx:
        term_mask = (chain_ids == chain_ids[end_i]) & (res_ids == res_ids[end_i])
        names = atom_names[term_mask]
        # only protein C-termini carry an OXT; require the full quartet.
        if not ({"C", "CA", "O", "OXT"} <= set(names.tolist())):
            continue
        term_pos = np.nonzero(term_mask)[0]
        name_to_pos = {atom_names[p]: p for p in term_pos}
        idx_c.append(name_to_pos["C"])
        idx_ca.append(name_to_pos["CA"])
        idx_o.append(name_to_pos["O"])
        idx_oxt.append(name_to_pos["OXT"])

    if not idx_c:
        return pred_coordinate

    device = pred_coordinate.device
    t_c = torch.as_tensor(idx_c, dtype=torch.long, device=device)
    t_ca = torch.as_tensor(idx_ca, dtype=torch.long, device=device)
    t_o = torch.as_tensor(idx_o, dtype=torch.long, device=device)
    t_oxt = torch.as_tensor(idx_oxt, dtype=torch.long, device=device)

    coords = pred_coordinate.clone()
    # Compute in float32 for numerical stability regardless of input dtype.
    work = coords.to(torch.float32)

    A = work[:, t_c, :]  # [S, T, 3]  (C atom, plane point A)
    CA = work[:, t_ca, :]  # [S, T, 3] (CA atom, plane point C)
    O = work[:, t_o, :]  # [S, T, 3]
    OXT = work[:, t_oxt, :]  # [S, T, 3]

    # B = nearest oxygen (O or OXT) to A; this stays and is named O.
    dist_o = torch.linalg.norm(O - A, dim=-1)  # [S, T]
    dist_oxt = torch.linalg.norm(OXT - A, dim=-1)  # [S, T]
    nearest_is_o = (dist_o <= dist_oxt).unsqueeze(-1)  # [S, T, 1]
    B = torch.where(nearest_is_o, O, OXT)  # [S, T, 3]

    # Reflect B across the A->CA axis to place D (OXT).
    axis = CA - A
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # unit u
    vB = B - A
    proj = (vB * axis).sum(dim=-1, keepdim=True)  # vB . u
    D = A + 2.0 * proj * axis - vB  # reflection: 2(v.u)u - v

    # Write back: O slot gets B (kept oxygen), OXT slot gets D (reconstructed).
    B = B.to(coords.dtype)
    D = D.to(coords.dtype)
    n_sample = coords.shape[0]
    sample_ax = torch.arange(n_sample, device=device).unsqueeze(-1)  # [S, 1]
    coords[sample_ax, t_o.unsqueeze(0), :] = B
    coords[sample_ax, t_oxt.unsqueeze(0), :] = D
    return coords


def _peak_memory_mib(device: torch.device) -> float:
    """Peak allocated memory on ``device`` in MiB, or 0 where unavailable."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def _failure_hint(error: BaseException) -> str:
    """Actionable next step for the failures users actually hit."""
    text = str(error).lower()
    if "out of memory" in text or isinstance(error, torch.cuda.OutOfMemoryError):
        return (
            "Hint: this system did not fit in device memory. Peak memory grows "
            "roughly with the square of the token count. Lower "
            "--infer_setting.chunk_size and --model.msa_module.msa_chunk_size, "
            "or use fewer samples "
            "(--sample in mol pred, --sample_diffusion.N_sample otherwise). "
            "Above about 1,500 tokens, also run the triangle multiplication on "
            "PyTorch (--trimul_kernel torch in mol pred, "
            "--triangle_multiplicative torch otherwise), whose chunked in-place "
            "update the other kernels lack, and set "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
            "docs/inference_memory.md lists settings that fit 2,500 tokens on "
            "a 95 GiB GPU.\n"
        )
    if "no kernel image is available" in text:
        return (
            "Hint: the installed PyTorch build has no kernels for this GPU "
            "architecture. Check `mol doctor` for the build's arch list.\n"
        )
    return ""


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """
    Run the full inference process for the given runner and configurations.
    Processes all samples in the dataloader for each specified seed.

    Args:
        runner (InferenceRunner): The initialized runner instance.
        configs (Any): Inference configurations.
    """
    # Data loading
    logger.info(f"Loading data from {configs.input_json_path}")
    with open(configs.input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list) or len(json_data) == 0:
        raise ValueError(
            f"Input JSON must be a non-empty top-level list, got {type(json_data).__name__} "
            f"from {configs.input_json_path}"
        )

    seed_in_json = json_data[0].get("modelSeeds")
    if seed_in_json and configs.use_seeds_in_json:
        seeds = [int(i) for i in seed_in_json]
        logger.info(f"Using seeds from JSON: {seeds}")
    else:
        seeds = configs.seeds

    # An earlier call on the same runner removes the directory when it is empty.
    os.makedirs(runner.error_dir, exist_ok=True)
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = (
            f"Dataloader initialization failed: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a", encoding="utf-8") as f:
            f.write(error_message)
        raise RuntimeError(
            f"Dataloader initialization failed: {e}. "
            f"The report is in {opjoin(runner.error_dir, 'error.txt')}."
        ) from e

    num_data = len(dataloader.dataset)
    t0_start = time.time()
    succeeded = 0
    failures: list[str] = []

    for seed in seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        t1_start = time.time()
        for batch in dataloader:
            sample_name = "unknown"
            try:
                t2_start = time.time()
                data, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                if len(data_error_message) > 0:
                    failures.append(f"{sample_name} [seed:{seed}]")
                    logger.error(f"Data error for {sample_name}: {data_error_message}")
                    with open(
                        opjoin(runner.error_dir, f"{sample_name}.txt"),
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] "
                    f"{sample_name} [seed:{seed}]: "
                    f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                    f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)
                if runner.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(runner.device)
                prediction = runner.predict(data)

                # Regularize the C-terminal carboxyl oxygens (O / OXT) of each
                # polymer chain: the O<->OXT substructure-permutation symmetry can
                # leave one oxygen under-constrained and drift far away. Rebuild
                # them with an idealized symmetric carboxylate geometry. The
                # atom_array order matches prediction["coordinate"] 1:1.
                prediction["coordinate"] = fix_cterminal_carboxyl_oxygens(
                    prediction["coordinate"], atom_array
                )

                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type={
                        k: v
                        for k, v in data["entity_poly_type"].items()
                        if v != "non-polymer"
                    },
                )
                t2_end = time.time()
                # Report tokens against peak memory on every system: peak grows
                # roughly with the square of the token count, but the
                # coefficient is device-specific, so the only way to size a GPU
                # is to fit the curve on the card you will use. Three systems of
                # different sizes are enough. See docs/inference_memory.md.
                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {sample_name} [seed:{seed}] succeeded. "
                    f"Model forward time: {t2_end - t2_start:.2f}s. "
                    f"N_token {data['N_token'].item()}, "
                    f"peak device memory {_peak_memory_mib(runner.device):.0f} MiB. "
                    f"Results saved to {configs.dump_dir}"
                )
                succeeded += 1
                torch.cuda.empty_cache()
            except Exception as e:
                failures.append(f"{sample_name} [seed:{seed}]")
                error_message = (
                    f"[Rank {DIST_WRAPPER.rank}] {sample_name} failed: {e}\n"
                    f"{_failure_hint(e)}"
                    f"{traceback.format_exc()}"
                )
                logger.error(error_message)
                with open(
                    opjoin(runner.error_dir, f"{sample_name}.txt"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(error_message)
                torch.cuda.empty_cache()
        t1_end = time.time()
        logger.info(
            f"[Rank {DIST_WRAPPER.rank}] Seed {seed} completed in {t1_end - t1_start:.2f}s."
        )
    # Remove the error directory if it's empty
    if opexists(runner.error_dir):
        try:
            if not os.listdir(runner.error_dir):
                os.rmdir(runner.error_dir)
        except Exception:
            pass

    t0_end = time.time()
    logger.info(
        f"[Rank {DIST_WRAPPER.rank}] Job completed in {t0_end - t0_start:.2f}s. "
        f"{succeeded} prediction(s) succeeded, {len(failures)} failed."
    )
    if failures:
        # An exit code of zero on a run that produced fewer structures than it
        # was asked for is the worst failure mode for a caller: a shortfall
        # caused by OOM or a bad input looks exactly like success. Fail loudly
        # and name the per-sample reports.
        raise RuntimeError(
            f"{len(failures)} of {succeeded + len(failures)} prediction(s) failed: "
            f"{', '.join(failures[:10])}"
            f"{' ...' if len(failures) > 10 else ''}. "
            f"Per-sample reports are in {runner.error_dir}."
        )


def main(configs: Any) -> None:
    """
    Inference entry point.

    Args:
        configs (Any): Inference configurations.
    """
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def update_gpu_compatible_configs(configs: Any) -> Any:
    """
    Update configurations to ensure compatibility with specific GPU architectures (e.g., V100).

    Args:
        configs (Any): Original configurations.

    Returns:
        Any: Updated configurations.
    """

    def is_gpu_capability_between_7_and_8() -> bool:
        # Check if 7.0 <= device_capability < 8.0
        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        cc = major + minor / 10.0
        return 7.0 <= cc < 8.0

    if is_gpu_capability_between_7_and_8():
        # V100 and similar architectures don't support some kernels or BF16 effectively
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info(
            "Enforcing FP32 and torch kernels for compatibility with detected "
            "GPU (Compute Capability 7.x)."
        )
    return configs


def run() -> None:
    """
    Initialize and execute the inference pipeline.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs=configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model {model_name!r}; available models: {', '.join(model_configs)}")

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logger.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    configs = update_gpu_compatible_configs(configs)
    logger.info(f"Inference by Mol-1: model={model_name}, dtype={configs.dtype}")
    logger.info(
        f"Triangle kernels: multiplicative={configs.triangle_multiplicative}, "
        f"attention={configs.triangle_attention}"
    )
    logger.info(
        f"Optimization: diffusion_cache={configs.diffusion_cache_mode}, "
        f"efficient_fusion={configs.enable_efficient_fusion}, tf32={configs.enable_tf32}"
    )
    download_inference_cache(configs)
    main(configs)


if __name__ == "__main__":
    run()
