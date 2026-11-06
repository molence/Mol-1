"""Inference benchmark for Mol-1: load once, run the same input several times.

Times each model stage with CUDA synchronization (trunk, diffusion conditioning
caches, diffusion sampling, distogram + confidence, summary), reports the median
over the timed repeats, and saves the last repeat's outputs for correctness
comparison with ``bench/compare.py``.

Usage (inside the container, from the repo root):
    python bench/mol_bench.py --tag base --repeats 3 [-- runner options...]
Everything after ``--`` is passed to the runner config parser, e.g.
    --enable_tf32 true --infer_setting.sample_diffusion_chunk_size 5
Environment knobs read here: MOL_BENCH_PROFILE=<path> writes a torch profiler
trace of the last repeat; MOL_BENCH_NVTX=1 adds NVTX ranges per stage and per
module and limits CUDA profiling (nsys --capture-range=cudaProfilerApi) to the
last repeat.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mol.config.base import configs as configs_base  # noqa: E402
from mol.config.data import data_configs  # noqa: E402
from mol.config.inference import inference_configs  # noqa: E402
from mol.config.model_type import model_configs  # noqa: E402
from mol.config.parser import parse_configs  # noqa: E402
from mol.model.mol import Mol  # noqa: E402
from mol.runner import inference as inf  # noqa: E402
from mol.utils.seed import seed_everything  # noqa: E402

TIMES = defaultdict(float)


NVTX = os.environ.get("MOL_BENCH_NVTX") == "1"


@contextmanager
def stage(name):
    torch.cuda.synchronize()
    t = time.perf_counter()
    if NVTX:
        torch.cuda.nvtx.range_push(name)
    yield
    if NVTX:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    TIMES[name] += time.perf_counter() - t


def wrap(obj, attr, name):
    fn = getattr(obj, attr)

    def inner(*a, **k):
        with stage(name):
            return fn(*a, **k)

    setattr(obj, attr, inner)


MOD_TIMES = defaultdict(float)
MOD_CALLS = defaultdict(int)


def install_module_timers(model):
    """Synchronous per-module timing (MOL_BENCH_MODTIME=1), keyed by path with block indices dropped."""
    import re

    leaf_types = {
        "TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming", "TriangleAttention",
        "Transition", "AttentionPairBias", "OuterProductMean", "MSAPairWeightedAveraging",
        "PairformerWidthAdapter", "AtomAttentionEncoder", "AtomAttentionDecoder",
        "DiffusionTransformer", "DiffusionConditioning", "InputFeatureEmbedder", "TemplateEmbedder",
    }
    for name, mod in model.named_modules():
        if type(mod).__name__ not in leaf_types:
            continue
        key = re.sub(r"\.\d+\.", ".*.", name)

        def pre(m, a, key=key):
            torch.cuda.synchronize()
            m._t0 = time.perf_counter()

        def post(m, a, o, key=key):
            torch.cuda.synchronize()
            MOD_TIMES[key] += time.perf_counter() - m._t0
            MOD_CALLS[key] += 1

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)


def install_nvtx_ranges(model):
    """NVTX range per module of interest, named by path with block indices dropped (no syncs)."""
    import re

    types = {
        "TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming", "TriangleAttention",
        "Transition", "AttentionPairBias", "OuterProductMean", "MSAPairWeightedAveraging",
        "PairformerWidthAdapter", "AtomAttentionEncoder", "AtomAttentionDecoder",
        "DiffusionTransformer", "DiffusionConditioning", "InputFeatureEmbedder", "TemplateEmbedder",
        "MSAModule", "PairformerStack", "PairformerBlock", "ConfidenceHead",
    }
    for name, mod in model.named_modules():
        if type(mod).__name__ not in types:
            continue
        key = re.sub(r"\.\d+\.", ".*.", name)
        key = re.sub(r"\.\d+$", ".*", key)
        if type(mod).__name__ == "TriangleAttention":
            # The fused inference path calls this method instead of forward().
            fn = mod.fused_residual_update

            def ranged(*a, key=key, fn=fn, **k):
                torch.cuda.nvtx.range_push(key)
                try:
                    return fn(*a, **k)
                finally:
                    torch.cuda.nvtx.range_pop()

            mod.fused_residual_update = ranged
        # Hooks must return None (a pre-hook's return value replaces the inputs).
        mod.register_forward_pre_hook(lambda m, a, key=key: None if torch.cuda.nvtx.range_push(key) else None)
        mod.register_forward_hook(lambda m, a, o: None if torch.cuda.nvtx.range_pop() else None)


def build_configs(extra):
    arg_str = " ".join(extra)
    base = {**configs_base, "data": data_configs, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v

    deep_update(base, model_configs["mol-1-base"])
    configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
    return inf.update_gpu_compatible_configs(configs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--input", default=os.path.join(REPO, "examples/examples_with_template/example_9fm7.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "bench/results"))
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--bf16-diffusion", action="store_true", help="run sample_diffusion under BF16 autocast")
    ap.add_argument("--bf16-confidence", action="store_true", help="run the confidence head under BF16 autocast")
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]

    logging.basicConfig(level=logging.WARNING)
    root = os.environ.get("MOL_ROOT_DIR", os.path.join(REPO, "data_cache"))
    defaults = [
        "--input_json_path", args.input,
        "--dump_dir", os.path.join(args.out, args.tag, "dump"),
        "--load_checkpoint_path", os.path.join(root, "checkpoint/mol-1-base.pt"),
        "--use_template", "false",
        "--triangle_attention", "cuequivariance",
        "--triangle_multiplicative", "cuequivariance",
    ]
    configs = build_configs(defaults + extra)

    runner = inf.InferenceRunner(configs)
    model = runner.model
    wrap(model, "get_pairformer_output", "trunk")
    wrap(model, "sample_diffusion", "diffusion")
    wrap(model, "run_confidence_head", "confidence")
    wrap(model.diffusion_module.diffusion_conditioning, "prepare_cache", "diff_cache_pair")
    wrap(model.diffusion_module.atom_attention_encoder, "prepare_cache", "diff_cache_atom")

    if os.environ.get("MOL_BENCH_MODTIME") == "1":
        install_module_timers(model)
    if NVTX:
        install_nvtx_ranges(model)

    dl = inf.get_inference_dataloader(configs=configs)
    data, atom_array, err = next(iter(dl))[0]
    assert not err, err
    n_tok = data["N_token"].item()
    runner.update_model_configs(inf.update_inference_configs(configs, n_tok))
    if args.bf16_diffusion:
        runner.model.configs.skip_amp.sample_diffusion = False
    if args.bf16_confidence:
        runner.model.configs.skip_amp.confidence_head = False

    results = []
    pred = None
    prof_path = os.environ.get("MOL_BENCH_PROFILE")
    for i in range(args.warmup + args.repeats):
        TIMES.clear()
        MOD_TIMES.clear()
        MOD_CALLS.clear()
        seed_everything(seed=args.seed, deterministic=False)
        torch.cuda.reset_peak_memory_stats()
        last = i == args.warmup + args.repeats - 1
        prof = None
        if last and prof_path:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            )
            prof.__enter__()
        torch.cuda.synchronize()
        if NVTX and last:
            torch.cuda.profiler.start()
        t0 = time.perf_counter()
        pred = runner.predict({**data, "input_feature_dict": dict(data["input_feature_dict"])})
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        if NVTX and last:
            torch.cuda.profiler.stop()
        if prof is not None:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(prof_path)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        rec = {"total": total, **dict(TIMES), "peak_mib": torch.cuda.max_memory_allocated() / 2**20}
        rec["other"] = total - sum(TIMES[k] for k in ("trunk", "diffusion", "confidence", "diff_cache_pair", "diff_cache_atom"))
        print(f"[bench {args.tag}] iter {i}{' (warmup)' if i < args.warmup else ''}: "
              + " ".join(f"{k}={v:.2f}" for k, v in rec.items()), flush=True)
        if i >= args.warmup:
            results.append(rec)

    if MOD_TIMES:
        # Nested modules double count; print sorted, outer ones included.
        for k, v in sorted(MOD_TIMES.items(), key=lambda kv: -kv[1]):
            print(f"[modtime] {v:8.3f}s {MOD_CALLS[k]:6d} calls  {k}")
    keys = results[0].keys()
    med = {k: statistics.median(r[k] for r in results) for k in keys}
    print(f"[bench {args.tag}] MEDIAN: " + " ".join(f"{k}={v:.2f}" for k, v in med.items()), flush=True)

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    keep = {k: pred[k].detach().float().cpu() for k in ("coordinate", "plddt", "pae", "pde", "contact_probs") if k in pred}
    keep["summary"] = [
        {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in s.items()} for s in pred["summary_confidence"]
    ]
    torch.save(keep, os.path.join(out_dir, "pred.pt"))
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump({"median": med, "runs": results, "extra": extra, "n_token": n_tok}, f, indent=1)


if __name__ == "__main__":
    main()
