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
# Upstream file: runner/batch_inference.py
# Changes: mol doctor, --device and local checkpoint paths; MSA search via ColabFold only; non-zero exit when an input fails.

import difflib
import json
import logging
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

import click
import tqdm
from Bio import SeqIO

from mol.config.base import configs as configs_base
from mol.config.data import data_configs
from mol.config.inference import inference_configs
from mol.config.model_type import model_configs
from ml_collections.config_dict import ConfigDict
from mol.config.parser import parse_configs
from mol.data.inference.json_maker import cif_to_input_json
from mol.data.tools.kalign import find_kalign
from mol.data.utils import pdb_to_cif
from mol.utils.logger import get_logger
from mol.version import __version__

from mol.runner.inference import (
    download_inference_cache,
    infer_predict,
    InferenceRunner,
    update_gpu_compatible_configs,
)
from mol.runner.msa_search import msa_search, update_infer_json
from mol.runner.rna_msa_search import update_rna_msa_info
from mol.runner.template_search import update_template_info

logger = get_logger(__name__)


def init_logging() -> None:
    """Initialize logging configuration."""
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


def preprocess_input(
    input_json: str,
    out_dir: str,
    use_msa: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
) -> str:
    """
    Preprocess the input JSON file by performing MSA, template, and RNA MSA searches as needed.

    Args:
        input_json (str): Path to the input JSON file.
        out_dir (str): Directory to save search results.
        use_msa (bool): Whether to use protein MSA.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.

    Returns:
        str: Path to the updated JSON file.
    """
    # 1. Protein MSA search
    msa_updated_json, _ = update_infer_json(
        input_json, out_dir, use_msa=use_msa
    )

    # Read the data (either original or updated)
    with open(msa_updated_json, "r") as f:
        json_data = json.load(f)

    actual_updated = False

    # 2. Template search
    if use_template:
        template_updated = update_template_info(
            json_data,
            hmmsearch_binary_path=hmmsearch_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            seqres_database_path=seqres_database_path,
        )
        actual_updated = actual_updated or template_updated

    # 3. RNA MSA search
    if use_rna_msa:
        rna_updated = update_rna_msa_info(
            json_data,
            out_dir=out_dir,
            nhmmer_binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_rna_binary_path or hmmbuild_binary_path,
            ntrna_database_path=ntrna_database_path,
            rfam_database_path=rfam_database_path,
            rna_central_database_path=rna_central_database_path,
            nhmmer_n_cpu=nhmmer_n_cpu,
        )
        actual_updated = actual_updated or rna_updated

    if actual_updated:
        base, ext = os.path.splitext(os.path.basename(msa_updated_json))
        if "-update-msa" in base:
            output_json_name = base.replace("-update-msa", "-final-updated") + ext
        else:
            output_json_name = f"{base}-final-updated{ext}"

        output_json = os.path.join(
            os.path.dirname(os.path.abspath(msa_updated_json)), output_json_name
        )

        with open(output_json, "w") as f:
            json.dump(json_data, f, indent=4)
        logger.info(f"Input preprocessing completed, results saved to {output_json}")
        return output_json
    else:
        return msa_updated_json


# Inference memory controls mol pred accepts, under the same dotted names as
# mol/runner/inference.py; docs/inference_memory.md describes each one.
MEMORY_CHUNK_SETTINGS = {
    "infer_setting.chunk_size": "Attention chunk.",
    "model.msa_module.msa_chunk_size": "MSA row chunk.",
    "infer_setting.msa_pair_transition_chunk_size": (
        "Pair-transition chunk in the MSA module."
    ),
    "infer_setting.trunk_pairmixer_transition_chunk_size": (
        "Pair-transition chunk in the PairMixer blocks."
    ),
    "infer_setting.trunk_pairformer_transition_chunk_size": (
        "Pair-transition chunk in the Pairformer blocks."
    ),
    "infer_setting.confidence_pair_transition_chunk_size": (
        "Pair-transition chunk in the confidence head."
    ),
    "infer_setting.sample_diffusion_chunk_size": "Diffusion samples run together.",
    "infer_setting.diffusion_conditioning_row_chunk_size": (
        "Row chunk of the diffusion pair conditioning."
    ),
}


def inference_default(path: str) -> Any:
    """Return the default of a dotted config path as inference sees it."""
    node: Any = {**configs_base, **inference_configs}
    for part in path.split("."):
        node = node[part]
    return getattr(node, "value", node)


def parse_chunk_size(path: str, value: str) -> Optional[int]:
    """Parse a chunk-size option: a positive integer, or none/null for no chunking."""
    if value.strip().lower() in ("none", "null"):
        return None
    try:
        size = int(value)
    except ValueError:
        size = 0
    if size <= 0:
        raise click.BadParameter(
            f"{value!r} is not a positive integer or none", param_hint=f"--{path}"
        )
    return size


def apply_memory_settings(configs: ConfigDict, settings: Mapping[str, Any]) -> None:
    """Set dotted config paths such as ``infer_setting.chunk_size`` in place."""
    for path, value in settings.items():
        *parents, key = path.split(".")
        node = configs
        for part in parents:
            node = node[part]
        node[key] = value


def memory_options(command: Any) -> Any:
    """Add the inference memory controls to a click command.

    Each option is named after its config path and reaches the command as a
    keyword argument with the dots replaced by double underscores.
    """
    command = click.option(
        "--diffusion_cache_mode",
        "diffusion_cache_mode",
        type=click.Choice(["full", "shared", "off"]),
        default=None,
        help=(
            "What the diffusion module reuses across denoising steps. "
            f"Default: {inference_default('diffusion_cache_mode')}."
        ),
    )(command)
    command = click.option(
        "--infer_setting.dynamic_chunk_size",
        "infer_setting__dynamic_chunk_size",
        type=bool,
        default=None,
        help=(
            "Pick the attention chunk from the token count instead. "
            f"Default: {inference_default('infer_setting.dynamic_chunk_size')}."
        ),
    )(command)
    for path, text in reversed(MEMORY_CHUNK_SETTINGS.items()):
        command = click.option(
            f"--{path}",
            path.replace(".", "__"),
            type=str,
            default=None,
            help=f"{text} 'none' for no chunking. Default: {inference_default(path)}.",
        )(command)
    return command


def memory_settings_from_options(options: Mapping[str, Any]) -> dict:
    """Turn the options memory_options added into dotted config settings.

    Options left unset are omitted, so the config default applies.
    """
    settings = {}
    for path in MEMORY_CHUNK_SETTINGS:
        value = options[path.replace(".", "__")]
        if value is not None:
            settings[path] = parse_chunk_size(path, value)
    for path in ("infer_setting.dynamic_chunk_size", "diffusion_cache_mode"):
        value = options[path.replace(".", "__")]
        if value is not None:
            settings[path] = value
    return settings


def get_default_runner(
    seeds: Optional[list] = None,
    n_cycle: int = 20,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: str = "bf16",
    device: str = "auto",
    model_name: str = "mol-1-base",
    use_msa: bool = True,
    trimul_kernel="auto",
    triatt_kernel="auto",
    pairformer_backend: str = "torch",
    enable_fusion=True,
    enable_tf32=False,
    use_template: bool = False,
    use_rna_msa: bool = False,
    use_seeds_in_json: bool = False,
    need_atom_confidence: bool = False,
    load_checkpoint_path: Optional[str] = None,
    memory_settings: Optional[Mapping[str, Any]] = None,
) -> InferenceRunner:
    """
    Get a default InferenceRunner with the specified configurations.

    Args:
        seeds (Optional[list]): List of inference seeds.
        n_cycle (int): Number of Pairformer cycles.
        n_step (int): Number of diffusion steps.
        n_sample (int): Number of samples.
        dtype (str): Inference data type (e.g., 'bf16').
        model_name (str): Registered Mol-1 configuration.
        use_msa (bool): Whether to use MSA.
        trimul_kernel (str): Kernel for triangle multiplicative update.
        triatt_kernel (str): Kernel for triangle attention.
        pairformer_backend (str): Pairformer stack backend (torch or mlx).
        enable_fusion (bool): Whether to enable diffusion transformer fusion.
        enable_tf32 (bool): Whether to enable TF32.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        use_seeds_in_json (bool): Whether to use seeds defined in the JSON file.
        load_checkpoint_path (Optional[str]): Override the local checkpoint path.
        memory_settings (Optional[Mapping[str, Any]]): Dotted config paths,
            such as ``infer_setting.chunk_size``, and the values to set them to.

    Returns:
        InferenceRunner: An instance of InferenceRunner.
    """
    configs = {**configs_base, **{"data": data_configs}, **inference_configs, "model_name": model_name}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    if seeds is not None:
        configs.seeds = seeds
    model_name = configs.model_name
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model {model_name!r}; available models: {', '.join(model_configs)}")
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs.update(model_specfics_configs)
    # the user input configs has the highest priority
    configs.model.N_cycle = n_cycle
    configs.sample_diffusion.N_sample = n_sample
    configs.sample_diffusion.N_step = n_step
    configs.dtype = dtype
    configs.device = device
    if load_checkpoint_path is not None:
        configs.load_checkpoint_path = load_checkpoint_path
    configs.use_msa = use_msa
    configs.triangle_multiplicative = trimul_kernel
    configs.triangle_attention = triatt_kernel
    configs.pairformer_backend = pairformer_backend
    configs.enable_efficient_fusion = enable_fusion
    configs.enable_tf32 = enable_tf32
    configs.use_template = use_template
    configs.use_rna_msa = use_rna_msa
    configs.use_seeds_in_json = use_seeds_in_json
    configs.need_atom_confidence = need_atom_confidence
    apply_memory_settings(configs, memory_settings or {})
    if use_template:
        # Fail before the model loads, not on the first template hit.
        find_kalign()

    configs = update_gpu_compatible_configs(configs)
    logger.info(f"Inference by Mol-1: model={model_name}, dtype={configs.dtype}")
    logger.info(
        f"Triangle_multiplicative kernel: {configs.triangle_multiplicative}, "
        f"Triangle_attention kernel: {configs.triangle_attention}"
    )
    logger.info(f"Pairformer backend: {configs.pairformer_backend}")
    logger.info(
        f"diffusion_cache: {configs.diffusion_cache_mode}, "
        f"enable_efficient_fusion: {configs.enable_efficient_fusion}, "
        f"enable_tf32: {configs.enable_tf32}"
    )
    download_inference_cache(configs)
    return InferenceRunner(configs)


def inference_jsons(
    json_file: str,
    out_dir: str = "./output",
    use_msa: bool = True,
    seeds: list = [101],
    n_cycle: int = 20,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: str = "bf16",
    device: str = "auto",
    model_name: str = "mol-1-base",
    trimul_kernel: str = "auto",
    triatt_kernel: str = "auto",
    pairformer_backend: str = "torch",
    enable_fusion: bool = True,
    enable_tf32: bool = False,
    use_template: bool = False,
    use_rna_msa: bool = False,
    use_seeds_in_json: bool = False,
    need_atom_confidence: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    load_checkpoint_path: Optional[str] = None,
    memory_settings: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Run inference on a single JSON file or a directory of JSON files.

    Args:
        json_file (str): Path to a JSON file or directory containing JSON files.
        out_dir (str): Directory to save inference results.
        use_msa (bool): Whether to use MSA.
        seeds (list): List of inference seeds.
        n_cycle (int): Number of cycles.
        n_step (int): Number of diffusion steps.
        n_sample (int): Number of samples.
        dtype (str): Data type.
        model_name (str): Registered Mol-1 configuration.
        trimul_kernel (str): Kernel for triangle multiplicative.
        triatt_kernel (str): Kernel for triangle attention.
        pairformer_backend (str): Pairformer stack backend (torch or mlx).
        enable_fusion (bool): Enable efficient fusion.
        enable_tf32 (bool): Enable TF32.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        use_seeds_in_json (bool): Whether to use seeds from JSON.
        load_checkpoint_path (Optional[str]): Override the local checkpoint path.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        memory_settings (Optional[Mapping[str, Any]]): Inference memory
            controls by dotted config path; see get_default_runner.
    """
    infer_jsons = []
    if os.path.isdir(json_file):
        infer_jsons = [
            str(file) for file in Path(json_file).rglob("*") if file.is_file()
        ]
    elif os.path.isfile(json_file):
        infer_jsons = [json_file]
    else:
        raise RuntimeError(f"Can not read a special file: {json_file}")
    infer_jsons = [file for file in infer_jsons if file.endswith(".json")]
    if len(infer_jsons) == 0:
        raise RuntimeError(f"Can not read a valid json file in {json_file}")
    logger.info(f"Will infer with {len(infer_jsons)} jsons")

    failed_jsons = []
    inference_configs["dump_dir"] = out_dir
    runner = get_default_runner(
        seeds=seeds,
        n_cycle=n_cycle,
        n_step=n_step,
        n_sample=n_sample,
        dtype=dtype,
        device=device,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint_path,
        use_msa=use_msa,
        trimul_kernel=trimul_kernel,
        triatt_kernel=triatt_kernel,
        pairformer_backend=pairformer_backend,
        enable_fusion=enable_fusion,
        enable_tf32=enable_tf32,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        use_seeds_in_json=use_seeds_in_json,
        need_atom_confidence=need_atom_confidence,
        memory_settings=memory_settings,
    )
    configs = runner.configs
    for infer_json in tqdm.tqdm(infer_jsons):
        try:
            configs["input_json_path"] = preprocess_input(
                infer_json,
                out_dir=out_dir,
                use_msa=use_msa,
                use_template=use_template,
                use_rna_msa=use_rna_msa,
                hmmsearch_binary_path=hmmsearch_binary_path,
                hmmbuild_binary_path=hmmbuild_binary_path,
                seqres_database_path=seqres_database_path,
                nhmmer_binary_path=nhmmer_binary_path,
                hmmalign_binary_path=hmmalign_binary_path,
                hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
                ntrna_database_path=ntrna_database_path,
                rfam_database_path=rfam_database_path,
                rna_central_database_path=rna_central_database_path,
                nhmmer_n_cpu=nhmmer_n_cpu,
            )
        except Exception as exc:
            # The MSA, template or RNA MSA search failed before the model saw
            # the input; report it where infer_predict reports failed samples.
            error_message = (
                f"Preparing {infer_json} failed: {exc}\n{traceback.format_exc()}"
            )
            logger.error(error_message)
            os.makedirs(runner.error_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(infer_json))[0]
            with open(
                os.path.join(runner.error_dir, f"{stem}.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(error_message)
            failed_jsons.append(infer_json)
            continue
        try:
            infer_predict(runner, configs)
        except Exception as exc:
            # infer_predict has already logged and reported each failed sample.
            logger.error(f"Predicting {infer_json} failed: {exc}")
            failed_jsons.append(infer_json)
    if failed_jsons:
        # Keep going through every input, then fail the run: an exit code of
        # zero must mean every structure that was asked for was written.
        raise click.ClickException(
            f"{len(failed_jsons)} of {len(infer_jsons)} input(s) failed: "
            f"{', '.join(failed_jsons[:10])}"
            f"{' ...' if len(failed_jsons) > 10 else ''}. "
            f"Reports are in {runner.error_dir}."
        )


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


class SuggestGroup(click.Group):
    """A Click group that suggests similar commands on error."""

    def resolve_command(self, ctx, args):
        """Try to resolve the command, and suggest matches if it fails."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            if len(args) > 0:
                cmd_name = args[0]
                all_commands = self.list_commands(ctx)
                matches = difflib.get_close_matches(cmd_name, all_commands)
                if matches:
                    e.message += (
                        f"\n\nDid you mean one of these?\n    {', '.join(matches)}"
                    )
            raise e


@click.group(cls=SuggestGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def mol_cli() -> None:
    """
    Mol-1: A trainable reproduction of AlphaFold 3.

    This CLI provides tools for structure prediction, data conversion,
    and MSA/template searching.
    """
    pass


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i", "--input", type=str, required=True, help="Input JSON file or directory."
)
@click.option("-o", "--out_dir", default="./output", type=str, help="Output directory.")
@click.option("-s", "--seeds", type=str, default="101", help="Seeds (comma-separated).")
@click.option("-c", "--cycle", type=int, default=20, help="Pairformer cycle number.")
@click.option("-p", "--step", type=int, default=200, help="Diffusion steps.")
@click.option("-e", "--sample", type=int, default=5, help="Number of samples.")
@click.option("-d", "--dtype", type=str, default="bf16", help="Inference dtype.")
@click.option(
    "--device",
    type=str,
    default="auto",
    help=(
        "Compute device: 'auto' (CUDA, then Apple MPS, then CPU), 'cuda', "
        "'cuda:N', 'mps' or 'cpu'. An explicitly requested device that is "
        "unavailable is an error rather than a silent downgrade."
    ),
)
@click.option(
    "-n",
    "--model_name",
    type=click.Choice(list(model_configs)),
    default="mol-1-base",
    help="Registered Mol-1 configuration.",
)
@click.option(
    "--load_checkpoint_path", type=click.Path(dir_okay=False), default=None,
    help=(
        "Mol-1 checkpoint to load. Defaults to "
        "$MOL_ROOT_DIR/checkpoint/mol-1-base.pt, downloaded there if missing."
    ),
)
@click.option(
    "--use_msa",
    type=bool,
    default=True,
    help="Whether to use MSA for inference.",
)
@click.option(
    "--use_default_params",
    type=bool,
    default=False,
    help="Use recommended default parameters for the selected model.",
)
@click.option(
    "--trimul_kernel",
    type=str,
    default="auto",
    help=(
        "Triangle multiplicative update kernel ('auto', 'cuequivariance', "
        "'torch', 'flashpairformer'). 'auto' picks cuequivariance on CUDA when "
        "it is importable and torch otherwise; it never picks flashpairformer."
    ),
)
@click.option(
    "--triatt_kernel",
    type=str,
    default="auto",
    help=(
        "Triangle attention kernel ('auto', 'triattention', 'cuequivariance', "
        "'metal' or 'torch'). 'auto' picks cuequivariance on CUDA when it is "
        "importable and torch otherwise."
    ),
)
@click.option(
    "--pairformer_backend",
    type=click.Choice(["torch", "mlx"]),
    default="torch",
    help=(
        "Pairformer stack backend. 'mlx' runs the stack with MLX on Apple "
        "silicon (MPS inference only; needs the 'mlx' extra)."
    ),
)
@click.option(
    "--enable_fusion",
    type=bool,
    default=True,
    help="Enable efficient kernel fusion in the diffusion transformer.",
)
@click.option(
    "--enable_tf32",
    type=bool,
    default=False,
    help=(
        "Run FP32 matrix multiplications on TF32 tensor cores: 4-8% faster, "
        "with slightly different results."
    ),
)
@click.option(
    "--use_template",
    type=bool,
    default=False,
    help="Use templates: each protein chain's templatesPath, searched for chains without one.",
)
@click.option(
    "--use_rna_msa",
    type=bool,
    default=False,
    help="Use RNA MSAs: each RNA chain's unpairedMsaPath, searched for chains without one (needs the RNA databases).",
)
@click.option(
    "--use_seeds_in_json",
    type=bool,
    default=False,
    help="Priority to seeds defined in input JSON.",
)
@click.option(
    "--need_atom_confidence",
    type=bool,
    default=False,
    help="Whether to compute atom-level confidence scores.",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch (searches in PATH if not provided).",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild (searches in PATH if not provided).",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "--nhmmer_binary_path",
    type=str,
    default=None,
    help="Path to nhmmer for RNA MSA search.",
)
@click.option(
    "--hmmalign_binary_path",
    type=str,
    default=None,
    help="Path to hmmalign for RNA MSA search.",
)
@click.option(
    "--hmmbuild_rna_binary_path",
    type=str,
    default=None,
    help="Path to RNA-specific hmmbuild.",
)
@click.option(
    "--ntrna_database_path",
    type=str,
    default=None,
    help="Path to the NT-RNA database.",
)
@click.option(
    "--rfam_database_path",
    type=str,
    default=None,
    help="Path to the Rfam database.",
)
@click.option(
    "--rna_central_database_path",
    type=str,
    default=None,
    help="Path to the RNAcentral database.",
)
@click.option(
    "--nhmmer_n_cpu",
    type=int,
    default=None,
    help="Number of CPUs for nhmmer.",
)
@memory_options
def predict(
    input: str,
    out_dir: str,
    seeds: str,
    cycle: int,
    step: int,
    sample: int,
    dtype: str,
    device: str,
    model_name: str,
    use_msa: bool,
    use_default_params: bool,
    trimul_kernel: str,
    triatt_kernel: str,
    pairformer_backend: str,
    enable_fusion: bool,
    enable_tf32: bool,
    use_template: bool,
    use_rna_msa: bool,
    use_seeds_in_json: bool,
    need_atom_confidence: bool,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    load_checkpoint_path: Optional[str] = None,
    **memory_args: Any,
) -> None:
    """
    Run predictions with Mol-1 using various input formats.

    Args:
        input (str): Input JSON file or directory.
        out_dir (str): Output directory for results.
        seeds (str): Comma-separated seeds.
        cycle (int): Number of cycles.
        step (int): Number of diffusion steps.
        sample (int): Number of samples.
        dtype (str): Data type.
        model_name (str): Registered Mol-1 configuration.
        use_msa (bool): Use MSA.
        use_default_params (bool): Use default parameters for the model.
        trimul_kernel (str): Kernel for triangle multiplicative.
        triatt_kernel (str): Kernel for triangle attention.
        pairformer_backend (str): Pairformer stack backend (torch or mlx).
        enable_fusion (bool): Enable efficient fusion.
        enable_tf32 (bool): Enable TF32.
        use_template (bool): Use templates.
        use_rna_msa (bool): Use RNA MSA.
        use_seeds_in_json (bool): Use seeds from JSON.
        need_atom_confidence (bool): Compute atom-level confidence scores.
        load_checkpoint_path (Optional[str]): Override the local checkpoint path.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        **memory_args: The inference memory controls added by memory_options.
    """
    memory_settings = memory_settings_from_options(memory_args)
    init_logging()
    logger.info(f"Run infer with input={input}, out_dir={out_dir}, sample={sample}")
    if use_default_params:
        cycle = configs_base["model"]["N_cycle"]
        step = configs_base["sample_diffusion"]["N_step"]
    logger.info(
        f"Using default params for model {model_name}: "
        f"cycle={cycle}, step={step}, use_msa={use_msa}"
    )
    assert trimul_kernel in [
        "auto", "cuequivariance", "torch", "flashpairformer",
    ], ("Invalid trimul_kernel. Options: 'auto', 'cuequivariance', 'torch', "
        "'flashpairformer'.")
    assert triatt_kernel in [
        "auto", "triattention", "cuequivariance", "metal", "torch",
    ], ("Invalid triatt_kernel. Options: 'auto', 'triattention', "
        "'cuequivariance', 'metal', 'torch'.")
    seeds = list(map(int, seeds.split(",")))

    if use_template:
        logger.info("=" * 50)
        logger.info(
            "Using templates for inference. Template files should have "
            ".hhr, .a3m or .json extensions and be specified in the JSON file.\n"
            "Example: /path/to/template.hhr or /path/to/template.a3m\n"
            "Note: Inference will proceed with automatic template search "
            "if none are provided and use_template is True."
        )
        logger.info("=" * 50)

    if use_rna_msa:
        logger.info("=" * 50)
        logger.info(
            "Using RNA MSA for inference. RNA MSA files should have .a3m "
            "extension and be specified in the JSON file.\n"
            "Example: /path/to/rna_msa.a3m\n"
            "Note: Inference will proceed with automatic RNA MSA search "
            "if none are provided and use_rna_msa is True."
        )
        logger.info("=" * 50)

    if use_seeds_in_json:
        logger.info("=" * 50)
        logger.info(
            "Using seeds defined in JSON file for inference.\n"
            "Note: This will override any seeds passed via command line, "
            "using seeds from modelSeeds defined in the JSON."
        )
        logger.info("=" * 50)
    inference_jsons(
        input,
        out_dir,
        use_msa,
        seeds=seeds,
        n_cycle=cycle,
        n_step=step,
        n_sample=sample,
        dtype=dtype,
        device=device,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint_path,
        trimul_kernel=trimul_kernel,
        triatt_kernel=triatt_kernel,
        pairformer_backend=pairformer_backend,
        enable_fusion=enable_fusion,
        enable_tf32=enable_tf32,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        use_seeds_in_json=use_seeds_in_json,
        need_atom_confidence=need_atom_confidence,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        nhmmer_n_cpu=nhmmer_n_cpu,
        memory_settings=memory_settings,
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="PDB/CIF files or directory to generate inference JSONs.",
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@click.option(
    "--altloc",
    default="first",
    type=str,
    help=(
        "Select the first altloc conformation of each residue, "
        "or specify the altloc letter ('A', 'B', etc.)."
    ),
)
@click.option(
    "--assembly_id",
    default=None,
    type=str,
    help="Assembly ID for structure extension (default: no extension).",
)
@click.option(
    "--include_discont_poly_poly_bonds",
    default=False,
    is_flag=True,
    help="Whether to include discontinuous polymer-polymer bonds.",
)
def tojson(
    input: str,
    out_dir: str = "./output",
    altloc: str = "first",
    assembly_id: Optional[str] = None,
    include_discont_poly_poly_bonds: bool = False,
) -> List[str]:
    """
    Convert PDB or CIF files to JSON files for Mol-1 inference.

    Args:
        input (str): Input PDB/CIF file or directory.
        out_dir (str): Output directory for JSON files.
        altloc (str): Alternate location conformation selection.
        assembly_id (Optional[str]): Assembly ID for structure extension.
        include_discont_poly_poly_bonds (bool): Whether to include discontinuous polymer-polymer bonds.

    Returns:
        List[str]: List of generated JSON file paths.
    """
    init_logging()
    logger.info(
        f"Run tojson with input={input}, out_dir={out_dir}, "
        f"altloc={altloc}, assembly_id={assembly_id}"
        f", include_discont_poly_poly_bonds={include_discont_poly_poly_bonds}"
    )
    input_files = []
    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} not exists.")
    if os.path.isdir(input):
        input_files.extend(
            [str(file) for file in Path(input).rglob("*") if file.is_file()]
        )
    elif os.path.isfile(input):
        input_files.append(input)
    else:
        raise RuntimeError(f"can not read a special file: {input}")

    input_files = [
        file for file in input_files if file.endswith(".pdb") or file.endswith(".cif")
    ]
    if len(input_files) == 0:
        raise RuntimeError(f"can not read a valid `pdb` or `cif` file from {input}")
    logger.info(
        f"will tojson jsons for {len(input_files)} input files with `pdb` or `cif` format."
    )
    output_jsons = []
    os.makedirs(out_dir, exist_ok=True)
    for input_file in input_files:
        stem, _ = os.path.splitext(os.path.basename(input_file))
        pdb_name = stem[:20]
        output_json = os.path.join(out_dir, f"{pdb_name}.json")
        if input_file.endswith(".pdb"):
            with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
                tmp_cif_file = tmp.name
                pdb_to_cif(input_file, tmp_cif_file)
                cif_to_input_json(
                    tmp_cif_file,
                    assembly_id=assembly_id,
                    altloc=altloc,
                    sample_name=pdb_name,
                    output_json=output_json,
                    include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
                )
        elif input_file.endswith(".cif"):
            cif_to_input_json(
                input_file,
                assembly_id=assembly_id,
                altloc=altloc,
                output_json=output_json,
                include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
            )
        else:
            raise RuntimeError(f"can not read a special ligand_file: {input_file}")
        output_jsons.append(output_json)
    logger.info(f"{len(output_jsons)} generated jsons have been saved to {out_dir}.")
    return output_jsons


MSA_INDEX_NAME = "msa_index.json"


def msa_dir_names(record_ids: List[str]) -> List[str]:
    """Return a directory name for each FASTA record: its id, made safe and unique.

    Characters other than letters, digits, ``.``, ``_`` and ``-`` become ``_``;
    a repeated name gets ``_2``, ``_3`` and so on.
    """
    names, taken = [], {MSA_INDEX_NAME}
    for i, record_id in enumerate(record_ids):
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", record_id).strip("._") or f"seq_{i}"
        name, n = base, 2
        while name in taken:
            name, n = f"{base}_{n}", n + 1
        taken.add(name)
        names.append(name)
    return names


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON or FASTA file for MSA search.",
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
def msa(input: str, out_dir: str) -> Union[str, list]:
    """
    Perform MSA search using MMseqs2.
    If input is a FASTA file, it should contain protein sequences.

    For a FASTA file, each record's MSAs go to ``<out_dir>/<record name>/``, and
    ``<out_dir>/msa_index.json`` lists the records in file order with their id,
    sequence and directory.

    Args:
        input (str): Path to a JSON or FASTA file.
        out_dir (str): Directory to save MSA results.

    Returns:
        Union[str, list]: Updated JSON path, or the index entries for a FASTA
        file.
    """
    init_logging()
    logger.info(f"Run msa with input={input}, out_dir={out_dir}")
    if input.endswith(".json"):
        msa_input_json, _ = update_infer_json(
            input, out_dir, use_msa=True
        )
        logger.info(f"msa results have been update to {msa_input_json}")
        return msa_input_json
    elif input.endswith(".fasta"):
        records = list(SeqIO.parse(input, "fasta"))
        if not records:
            raise RuntimeError(f"no sequences in {input}")
        names = msa_dir_names([record.id for record in records])
        os.makedirs(out_dir, exist_ok=True)
        index = []
        # The service leaves numbered working directories and downloads; keep
        # only each record's MSAs, under its own name.
        with tempfile.TemporaryDirectory(dir=out_dir) as search_dir:
            result_dirs = msa_search([str(r.seq) for r in records], search_dir)
            assert len(result_dirs) == len(records), "msa search failed"
            for record, name, result_dir in zip(records, names, result_dirs):
                target = os.path.join(out_dir, name)
                os.makedirs(target, exist_ok=True)
                for filename in ("non_pairing.a3m", "pairing.a3m"):
                    if os.path.exists(os.path.join(result_dir, filename)):
                        shutil.copyfile(
                            os.path.join(result_dir, filename),
                            os.path.join(target, filename),
                        )
                index.append(
                    {"id": record.id, "sequence": str(record.seq),
                     "dir": os.path.abspath(target)}
                )
        index_path = os.path.join(out_dir, MSA_INDEX_NAME)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        for entry in index:
            logger.info(f"MSA for {entry['id']}: {entry['dir']}")
        logger.info(f"Record to directory index: {index_path}")
        return index
    else:
        raise RuntimeError(f"only support `json` or `fasta` format, but got : {input}")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Input JSON; the MSAs and templates it lacks are searched.",
)
@click.option(
    "-o",
    "--out_dir",
    type=str,
    default="./output",
    help="Output directory.",
)
@click.option(
    "--use_rna_msa",
    type=bool,
    default=False,
    help="Search RNA MSAs for RNA chains that have none; needs the RNA databases.",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch.",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild.",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "--nhmmer_binary_path",
    type=str,
    default=None,
    help="Path to nhmmer for RNA MSA search.",
)
@click.option(
    "--hmmalign_binary_path",
    type=str,
    default=None,
    help="Path to hmmalign for RNA MSA search.",
)
@click.option(
    "--hmmbuild_rna_binary_path",
    type=str,
    default=None,
    help="Path to RNA-specific hmmbuild.",
)
@click.option(
    "--ntrna_database_path",
    type=str,
    default=None,
    help="Path to the NT-RNA database.",
)
@click.option(
    "--rfam_database_path",
    type=str,
    default=None,
    help="Path to the Rfam database.",
)
@click.option(
    "--rna_central_database_path",
    type=str,
    default=None,
    help="Path to the RNAcentral database.",
)
@click.option(
    "--nhmmer_n_cpu",
    type=int,
    default=None,
    help="Number of CPUs for nhmmer.",
)
def inputprep(
    input: str,
    out_dir: str,
    use_rna_msa: bool,
    hmmsearch_binary_path: Optional[str],
    hmmbuild_binary_path: Optional[str],
    seqres_database_path: Optional[str],
    nhmmer_binary_path: Optional[str],
    hmmalign_binary_path: Optional[str],
    hmmbuild_rna_binary_path: Optional[str],
    ntrna_database_path: Optional[str],
    rfam_database_path: Optional[str],
    rna_central_database_path: Optional[str],
    nhmmer_n_cpu: Optional[int],
) -> str:
    """
    Perform MSA search, template search, and RNA MSA search sequentially.

    Args:
        input (str): Path to the input JSON file.
        out_dir (str): Directory to save all search results.
        use_rna_msa (bool): Search RNA MSAs for RNA chains that have none.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): Path to NT-RNA database.
        rfam_database_path (Optional[str]): Path to Rfam database.
        rna_central_database_path (Optional[str]): Path to RNAcentral database.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.

    Returns:
        str: Final updated JSON file path with all search information.
    """
    logger.info(f"Run inputprep with input={input}, out_dir={out_dir}")

    if not input.endswith(".json"):
        raise RuntimeError(f"inputprep only supports `json` format, but got: {input}")

    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} does not exist")

    return preprocess_input(
        input_json=input,
        out_dir=out_dir,
        use_msa=True,
        use_template=True,
        use_rna_msa=use_rna_msa,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        nhmmer_n_cpu=nhmmer_n_cpu,
    )


@click.command(context_settings=CONTEXT_SETTINGS)
def doctor() -> None:
    """Print environment diagnostics: devices, kernels, binaries and caches."""
    # Imported lazily so `doctor` still reports a broken PyTorch install
    # instead of failing to start.
    from mol.utils.environment import format_doctor_report

    click.echo(format_doctor_report())


mol_cli.add_command(doctor, name="doctor")
mol_cli.add_command(predict, name="pred")
mol_cli.add_command(tojson, name="json")
mol_cli.add_command(msa, name="msa")
mol_cli.add_command(inputprep, name="prep")

if __name__ == "__main__":
    predict()
