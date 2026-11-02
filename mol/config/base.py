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
# Upstream file: configs/configs_base.py

# pylint: disable=C0114,C0301
from mol.config.types import (
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

basic_configs = {
    "project": RequiredValue(str),
    "run_name": RequiredValue(str),
    "base_dir": RequiredValue(str),
    # training
    "eval_interval": RequiredValue(int),
    "log_interval": RequiredValue(int),
    "checkpoint_interval": -1,
    "eval_first": False,  # run evaluate() before training steps
    "iters_to_accumulate": 1,
    "finetune_params_with_substring": [
        ""
    ],  # params with substring will be finetuned with different learning rate: finetune_optim_configs["lr"]
    "eval_only": False,
    # Empty falls back to load_checkpoint_dir/<model_name>.pt, i.e.
    # $MOL_ROOT_DIR/checkpoint/mol-1-base.pt. Override with --load_checkpoint_path.
    "load_checkpoint_path": "",
    "load_ema_checkpoint_path": "",
    "load_strict": True,
    "load_params_only": True,
    "skip_load_step": False,
    "skip_load_optimizer": False,
    "skip_load_scheduler": False,
    "load_step_for_scheduler": False,
    "train_confidence_only": False,
    "use_wandb": True,
    "wandb_id": "",
    "seed": 42,
    "deterministic": False,
    "deterministic_seed": False,
    "ema_decay": -1.0,
    "eval_ema_only": False,  # whether wandb only tracking ema checkpoint metrics
    "ema_mutable_param_keywords": [""],
    "model_name": "mol-1-base",  # train model name
}
data_configs = {
    # Data
    "train_crop_size": 256,
    "test_max_n_token": -1,
    "train_lig_atom_rename": False,
    "train_shuffle_mols": False,
    "train_shuffle_sym_ids": False,
    "test_lig_atom_rename": False,
    "test_shuffle_mols": False,
    "test_shuffle_sym_ids": False,
}
optim_configs = {
    # Optim
    "lr": 0.0018,
    "lr_scheduler": "af3",
    "warmup_steps": 10,
    "max_steps": RequiredValue(int),
    "min_lr_ratio": 0.1,
    "decay_every_n_steps": 50000,
    "grad_clip_norm": 10,
    # Optim - Adam
    "adam": {
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 1e-8,
        "lr": GlobalConfigValue("lr"),
        "use_adamw": False,
    },
    # Optim - LRScheduler
    "af3_lr_scheduler": {
        "warmup_steps": GlobalConfigValue("warmup_steps"),
        "decay_every_n_steps": GlobalConfigValue("decay_every_n_steps"),
        "decay_factor": 0.95,
        "lr": GlobalConfigValue("lr"),
    },
}
# Fine-tuned optimizer settings.
finetune_optim_configs = {
    # Optim
    "lr": 0.0018,
    "lr_scheduler": "cosine_annealing",
    "warmup_steps": 1000,
    "max_steps": 20000,
    "min_lr_ratio": 0.1,
    "decay_every_n_steps": 50000,
}
model_configs = {
    "mc_dropout_apply_rate": 0.4,
    "mc_dropout_rate": 0.4,
    # Model
    "c_s": 384,
    "c_z": 448,
    "c_z_pairmixer": 448,
    "c_z_pairformer": 320,
    "c_s_inputs": 449,  # c_s_inputs == c_token + 32 + 32 + 1
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    "n_blocks": 48,
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_data": 16.0,
    "diffusion_batch_size": 48,
    "diffusion_chunk_size": ValueMaybeNone(4),  # chunksize of diffusion_batch_size
    "blocks_per_ckpt": ValueMaybeNone(
        1
    ),  # NOTE: Number of blocks in each activation checkpoint, if None, no checkpointing is performed.
    "hidden_scale_up": True,  # whether to scale up hidden dim in pairformer and confidence head
    # Number of leading PairMixer blocks in the trunk. For m PairMixer blocks
    # followed by n complete Pairformer blocks, set this to m and n_blocks to m + n.
    "n_pairmixer_blocks": 8,
    # switch of kernels
    "triangle_multiplicative": "cuequivariance",  # training: cuequivariance/torch; inference also supports flashpairformer
    "triangle_attention": "cuequivariance",  # triattention, cuequivariance, torch
    "pairformer_backend": "torch",  # torch or mlx (MPS inference only)
    "enable_diffusion_shared_vars_cache": False,
    "enable_efficient_fusion": False,
    "enable_tf32": False,
    "find_unused_parameters": False,
    "dtype": "bf16",  # default training dtype: bf16
    "loss_metrics_sparse_enable": True,  # the swicth for both sparse lddt metrics and sparse bond/smooth lddt loss
    "skip_amp": {
        "sample_diffusion": True,
        # If confidence_head (below) set to True and triangle_attention set to cuequivariance,
        # RuntimeError: ERROR: Full precision FP32 backward pass for triangle attention is not
        # implemented yet! Please set torch.backends.cuda.matmul.allow_tf32=True.
        "confidence_head": False,
        "sample_diffusion_training": True,
        "loss": True,
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(
            1024
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "dynamic_chunk_size": False,
        "chunk_size_thresholds": {
            "1024": -1,  # -1 means no chunking (equivalent to None)
            "1536": 512,
            "2048": 256,
            "2560": 128,
        },
        # Pair-transition row chunks are independent of triangle-attention
        # chunking.  ``None`` inherits the effective stage pair chunk for
        # backward-compatible automatic memory reduction; explicit values
        # allow widths/live-memory regimes to be tuned independently.
        "msa_pair_transition_chunk_size": ValueMaybeNone(1024),
        "trunk_pairmixer_transition_chunk_size": ValueMaybeNone(128),
        "trunk_pairformer_transition_chunk_size": ValueMaybeNone(128),
        "confidence_pair_transition_chunk_size": ValueMaybeNone(128),
        "sample_diffusion_chunk_size": ValueMaybeNone(
            5
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        # Inference-only, semantics-preserving row chunking for the FP32
        # diffusion pair-conditioning projection.  ``None`` preserves the
        # historical whole-tensor implementation exactly.
        "diffusion_conditioning_row_chunk_size": ValueMaybeNone(768),
        "lddt_metrics_sparse_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "lddt_metrics_chunk_size": ValueMaybeNone(
            1
        ),  # only works if loss_metrics_sparse_enable, can set as default 1
    },
    "train_noise_sampler": {
        "p_mean": -1.2,
        "p_std": 1.5,
        "sigma_data": 16.0,  # NOTE: in EDM, this is 1.0
    },
    "inference_noise_scheduler": {
        "s_max": 160.0,
        "s_min": 4e-4,
        "rho": 7,
        "sigma_data": 16.0,  # NOTE: in EDM, this is 1.0
    },
    "sample_diffusion": {
        "gamma0": 0.8,
        "gamma_min": 1.0,
        "noise_scale_lambda": 1.003,
        "step_scale_eta": 1.5,
        "N_step": 200,
        "N_sample": 5,
        "N_step_mini_rollout": 20,
        "N_sample_mini_rollout": 1,
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "N_cycle": 20,
        "condition_embedding_drop_rate": 0.0,
        "confidence_embedding_drop_rate": 0.0,
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
        },
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
        "template_embedder": {
            "c": 64,
            "c_z": GlobalConfigValue("c_z"),
            "n_blocks": 2,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
        },
        "msa_module": {
            "c_m": 192,
            "c_z": GlobalConfigValue("c_z"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "msa_dropout": 0.15,
            "pair_dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "msa_chunk_size": ValueMaybeNone(2048),
            "msa_max_size": 16384,
        },
        "pairformer": {
            "n_blocks": GlobalConfigValue("n_blocks"),
            "c_z": GlobalConfigValue("c_z"),
            "c_z_pairmixer": GlobalConfigValue("c_z_pairmixer"),
            "c_z_pairformer": GlobalConfigValue("c_z_pairformer"),
            "c_s": GlobalConfigValue("c_s"),
            "n_heads": 16,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "n_pairmixer_blocks": GlobalConfigValue("n_pairmixer_blocks"),
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": True,
            "sigma_data": GlobalConfigValue("sigma_data"),
            "c_token": 768,
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "atom_encoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "transformer": {
                "n_blocks": 24,
                "n_heads": 16,
            },
            "atom_decoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "confidence_head": {
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "max_atoms_per_token": GlobalConfigValue("max_atoms_per_token"),
            "pairformer_dropout": 0.0,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "distance_bin_start": 3.25,
            "distance_bin_end": 52.0,
            "distance_bin_step": 1.25,
            "stop_gradient": True,
        },
        "distogram_head": {
            "c_z": GlobalConfigValue("c_z"),
            "no_bins": GlobalConfigValue("no_bins"),
        },
    },
}
perm_configs = {
    # Chain and Atom Permutation
    "chain_permutation": {
        "train": {
            "mini_rollout": True,
            "diffusion_sample": False,
        },
        "test": {
            "diffusion_sample": True,
        },
        "permute_by_pocket": True,
        "configs": {
            "use_center_rmsd": False,
            "find_gt_anchor_first": False,
            "accept_it_as_it_is": False,
            "enumerate_all_anchor_pairs": False,
            "selection_metric": "aligned_rmsd",
        },
    },
    "atom_permutation": {
        "train": {
            "mini_rollout": True,
            "diffusion_sample": False,
        },
        "test": {
            "diffusion_sample": True,
        },
        "permute_by_pocket": True,
        "global_align_wo_symmetric_atom": False,
    },
}
loss_configs = {
    "loss": {
        "diffusion_lddt_chunk_size": ValueMaybeNone(1),
        "diffusion_bond_chunk_size": ValueMaybeNone(1),
        "diffusion_chunk_size_outer": ValueMaybeNone(1),
        "diffusion_sparse_loss_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "diffusion_lddt_loss_dense": True,  # only set true in initial training for training speed
        "resolution": {"min": 0.1, "max": 4.0},
        "weight": {
            "alpha_confidence": 1e-4,
            "alpha_pae": 0.0,  # or 1 in finetuning stage 3
            "alpha_except_pae": 1.0,
            "alpha_diffusion": 4.0,
            "alpha_distogram": 3e-2,
            "alpha_bond": 0.0,  # or 1 in finetuning stages
            "smooth_lddt": 1.0,  # or 0 in finetuning stages
        },
        "plddt": {
            "min_bin": 0,
            "max_bin": 1.0,
            "no_bins": 50,
            "normalize": True,
            "eps": 1e-6,
        },
        "pde": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "resolved": {
            "eps": 1e-6,
        },
        "pae": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "diffusion": {
            "mse": {
                "weight_mse": 1 / 3,
                "weight_dna": 5.0,
                "weight_rna": 5.0,
                "weight_ligand": 10.0,
                "eps": 1e-6,
            },
            "bond": {
                "eps": 1e-6,
            },
            "smooth_lddt": {
                "eps": 1e-6,
            },
        },
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": 1e-6,
        },
    },
    "metrics": {
        "lddt": {
            "eps": 1e-6,
        },
        "complex_ranker_keys": ListValue(["plddt", "gpde", "ranking_score"]),
        "chain_ranker_keys": ListValue(["chain_ptm", "chain_plddt"]),
        "interface_ranker_keys": ListValue(
            ["chain_pair_iptm", "chain_pair_iptm_global", "chain_pair_plddt"]
        ),
        "clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75},
    },
}

configs = {
    **basic_configs,
    **data_configs,
    **optim_configs,
    **model_configs,
    **perm_configs,
    **loss_configs,
}
configs["finetune"] = finetune_optim_configs
