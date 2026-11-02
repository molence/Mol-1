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
# Upstream file: protenix/model/protenix.py
# Changes: Mol-1 trunk: PairMixer blocks ahead of the Pairformer stack; triangle kernels selected per call, including FlashPairformer; ESM input removed.

import copy
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mol.model import sample_confidence
from mol.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
    sample_diffusion_training,
    TrainingNoiseSampler,
)
from mol.model.modules.confidence import ConfidenceHead
from mol.model.modules.diffusion import DiffusionModule
from mol.model.modules.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from mol.model.modules.head import DistogramHead
from mol.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from mol.model.modules.primitives import LinearNoBias
from mol.model.modules.transformer import ProjectedPairBiases
from mol.model.triangular.layers import LayerNorm, resolve_auto_kernel
from mol.model.utils import simple_merge_dict_list
from mol.utils.logger import get_logger
from mol.utils.permutation.permutation import SymmetricPermutation
from mol.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)

def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from mol.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class Mol(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs: Any) -> None:
        super(Mol, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def _triangle_multiplicative_backend(self) -> str:
        """Use the shared backend; training recycling requires a trainable kernel."""
        backend = self.configs.triangle_multiplicative
        allowed = ("auto", "torch", "cuequivariance")
        if not self.training:
            allowed += ("flashpairformer",)
        if backend not in allowed:
            entry = "training" if self.training else "inference"
            raise ValueError(
                f"triangle_multiplicative={backend!r} is unsupported for {entry}; "
                f"choose from {allowed}. FlashPairformer TMK3 supports inference only."
            )
        if backend == "auto":
            return resolve_auto_kernel("triangle_multiplicative", self._device())
        return backend

    def _triangle_attention_backend(self) -> str:
        """Use the same triangle-attention setting in training and inference."""
        backend = self.configs.triangle_attention
        allowed = ("auto", "torch", "cuequivariance", "triattention", "metal")
        if backend not in allowed:
            raise ValueError(f"triangle_attention={backend!r} is unsupported; choose from {allowed}")
        if backend == "auto":
            return resolve_auto_kernel("triangle_attention", self._device())
        return backend

    def _pairformer_backend(self) -> str:
        backend = self.configs.pairformer_backend
        allowed = ("torch", "mlx")
        if backend not in allowed:
            raise ValueError(f"pairformer_backend={backend!r} is unsupported; choose from {allowed}")
        if backend == "mlx" and (self.training or self._device().type != "mps"):
            raise RuntimeError("pairformer_backend='mlx' requires MPS inference")
        return backend

    def _device(self) -> torch.device:
        """Device the model's parameters live on."""
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover - a model always has parameters
            return torch.device("cpu")

    def _pair_transition_chunk_size(
        self,
        config_name: str,
        fallback: Optional[int],
    ) -> Optional[int]:
        """Resolve a stage-specific transition chunk with pair-chunk fallback."""
        configured = getattr(self.configs.infer_setting, config_name, None)
        if configured is None:
            return fallback
        return configured

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        msa_pair_transition_chunk_size = self._pair_transition_chunk_size(
            "msa_pair_transition_chunk_size",
            chunk_size,
        )
        pairmixer_transition_chunk_size = self._pair_transition_chunk_size(
            "trunk_pairmixer_transition_chunk_size",
            chunk_size,
        )
        pairformer_transition_chunk_size = self._pair_transition_chunk_size(
            "trunk_pairformer_transition_chunk_size",
            chunk_size,
        )

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                if mc_dropout:
                    z = z_init + F.dropout(
                        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                        p=self.configs.mc_dropout_rate,
                    )
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self._triangle_multiplicative_backend(),
                            triangle_attention=self._triangle_attention_backend(),
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self._triangle_multiplicative_backend(),
                        triangle_attention=self._triangle_attention_backend(),
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        pair_transition_chunk_size=msa_pair_transition_chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self._triangle_multiplicative_backend(),
                            triangle_attention=self._triangle_attention_backend(),
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self._triangle_multiplicative_backend(),
                        triangle_attention=self._triangle_attention_backend(),
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        pair_transition_chunk_size=msa_pair_transition_chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                if self._pairformer_backend() == "mlx":
                    from mol.model.triangular.mlx_pairformer import (
                        run_mlx_pairformer_stack,
                    )

                    s, z = run_mlx_pairformer_stack(self.pairformer_stack, s, z)
                else:
                    s, z = self.pairformer_stack(
                        s,
                        z,
                        pair_mask=None,
                        triangle_multiplicative=self._triangle_multiplicative_backend(),
                        triangle_attention=self._triangle_attention_backend(),
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        pairmixer_transition_chunk_size=(
                            pairmixer_transition_chunk_size
                        ),
                        pairformer_transition_chunk_size=(
                            pairformer_transition_chunk_size
                        ),
                    )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size
                    if not self.training
                    else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def _inference_diffusion_cache_mode(self) -> str:
        """Resolve the inference-only mode; training keeps its shared-cache switch."""
        cache_mode = getattr(self.configs, "diffusion_cache_mode", "full")
        if cache_mode not in {"full", "shared", "off"}:
            raise ValueError(f"Invalid inference diffusion cache mode: {cache_mode}")
        return cache_mode

    def _prepare_projected_pair_biases(
        self,
        pair_z: Optional[torch.Tensor],
    ) -> Optional[ProjectedPairBiases]:
        """Precompute request-local diffusion pair biases when caching is safe."""
        cache_mode = self._inference_diffusion_cache_mode()
        if (
            cache_mode != "full"
            or self.diffusion_module.training
            or torch.is_grad_enabled()
            or pair_z is None
        ):
            return None
        return autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion
        )(self.diffusion_module.prepare_projected_pair_biases)(
            pair_z=pair_z,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Mol-1 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode in ["inference"]:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                mc_dropout=random.random() < mc_dropout_apply_rate,
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Mol-1 model.
        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        cache = dict()
        cache_mode = self._inference_diffusion_cache_mode()
        conditioning_row_chunk = getattr(
            self.configs.infer_setting,
            "diffusion_conditioning_row_chunk_size",
            None,
        )
        if cache_mode in {"full", "shared"}:
            # line 1-5 of algorithm 21 calculate z in diffusion conditioning
            if conditioning_row_chunk is None:
                cache["pair_z"] = autocasting_disable_decorator(
                    self.configs.skip_amp.sample_diffusion
                )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                    input_feature_dict["relp"], z, False
                )
            else:
                cache["pair_z"] = (
                    self.diffusion_module.diffusion_conditioning.prepare_cache(
                        input_feature_dict["relp"],
                        z,
                        False,
                        row_chunk_size=int(conditioning_row_chunk),
                    )
                )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        cache["projected_pair_biases"] = self._prepare_projected_pair_biases(
            cache["pair_z"]
        )
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
            projected_pair_biases=cache["projected_pair_biases"],
        )
        # These tensors are useful only inside the denoising loop. Drop the
        # request-owned reference before confidence/distogram heads.
        del cache["projected_pair_biases"]
        del cache["pair_z"]
        del cache["p_lm/c_l"]

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self._triangle_multiplicative_backend(),
            triangle_attention=self._triangle_attention_backend(),
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            pair_transition_chunk_size=self._pair_transition_chunk_size(
                "confidence_pair_transition_chunk_size",
                chunk_size,
            ),
        )

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            (
                pred_dict,
                permutation_log_dict,
            ) = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            log_dict.update(permutation_log_dict)
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Mol-1 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        log_dict = {}
        pred_dict = {}

        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            self.diffusion_module.eval()  # use eval mode for mini-rollout
            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(
                    None
                    if cache["p_lm/c_l"][0] is None
                    else cache["p_lm/c_l"][0].detach()
                ),
                c_l=(
                    None
                    if cache["p_lm/c_l"][1] is None
                    else cache["p_lm/c_l"][1].detach()
                ),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
            self.diffusion_module.train()
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            (
                label_dict,
                perm_log_dict,
            ) = symmetric_permutation.permute_label_to_match_mini_rollout(
                coordinate_mini,
                input_feature_dict,
                label_dict,
                label_full_dict,
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self._triangle_multiplicative_backend(),
            triangle_attention=self._triangle_attention_backend(),
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            pair_transition_chunk_size=self._pair_transition_chunk_size(
                "confidence_pair_transition_chunk_size", chunk_size
            ),
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)
        log_dict.update({"noise_level": x_noise_level})

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Mol-1 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "eval", "inference"]
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = N_cycle
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
