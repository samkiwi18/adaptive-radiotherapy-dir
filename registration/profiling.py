"""VRAM profiling: per-block activation memory, inference and TTO peaks, weights/gradients/Adam breakdown."""
import gc
import time
from dataclasses import replace

import pandas as pd
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from .model import predict_dvf
from .tto import TTOConfig, run_tto
from .utils import autocast_bf16, load_weights

MIB = 1024 ** 2
TTO_PHASES = ("network_forward", "losses", "network_backward", "optimizer", "other")


def _require_cuda(device):
    if device.type != "cuda":
        raise RuntimeError("VRAM profiling needs a CUDA device (it reads the torch.cuda memory statistics)")


def _tensor_bytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(t) for t in x)
    if isinstance(x, dict):
        return sum(_tensor_bytes(t) for t in x.values())
    return 0


def _param_bytes(params):
    return sum(p.numel() * p.element_size() for p in params)


def profiled_blocks(model, integrator=None):
    """Top-level blocks of the network (SwinUNETR stages and velocity head), plus the integrator if given."""
    base = model.module if hasattr(model, "module") else model
    base = getattr(base, "_orig_mod", base)
    blocks = {f"backbone.{n}": m for n, m in base.backbone.named_children()}
    blocks["reg_head"] = base.reg_head
    if integrator is not None:
        blocks["integrator"] = integrator
    return blocks


class _BlockMemoryRecorder:
    """Hooks measuring memory before/after and peak of each (non-nested) block; overall peak in self.peak."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.records = {}
        self.peak = 0
        self._start = {}
        self._handles = []

    def __enter__(self):
        for name, module in self.blocks.items():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._post_hook(name)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        torch.cuda.synchronize()
        self.peak = max(self.peak, torch.cuda.max_memory_allocated())
        return False

    def _pre_hook(self, name):
        def hook(module, args):
            torch.cuda.synchronize()
            self.peak = max(self.peak, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            self._start[name] = (torch.cuda.memory_allocated(), time.perf_counter())
        return hook

    def _post_hook(self, name):
        def hook(module, args, output):
            torch.cuda.synchronize()
            allocated, t0 = self._start.pop(name)
            peak = torch.cuda.max_memory_allocated()
            self.peak = max(self.peak, peak)
            r = self.records.setdefault(name, dict(order=len(self.records), calls=0, output=0, retained=0,
                                                   peak_increase=0, time=0.0))
            r["calls"] += 1
            r["output"] += _tensor_bytes(output)
            r["retained"] += torch.cuda.memory_allocated() - allocated
            r["peak_increase"] = max(r["peak_increase"], peak - allocated)
            r["time"] += time.perf_counter() - t0
        return hook


class _PhaseMemoryRecorder:
    """Peak allocated memory of each TTO phase: forward, losses, backward, optimizer step, other."""

    def __init__(self, model, integrator):
        self.model = model
        self.integrator = integrator
        self.peaks = dict.fromkeys(TTO_PHASES, 0)
        self.phase = "other"
        self._handles = []

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._handles = [self.model.register_forward_pre_hook(lambda module, args: self._switch("network_forward")),
                         self.integrator.register_forward_hook(self._integrator_hook),
                         register_optimizer_step_pre_hook(lambda opt, args, kwargs: self._switch("optimizer")),
                         register_optimizer_step_post_hook(lambda opt, args, kwargs: self._switch("other"))]
        return self

    def __exit__(self, *exc):
        self._switch("other")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def _switch(self, phase):
        torch.cuda.synchronize()
        self.peaks[self.phase] = max(self.peaks[self.phase], torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        self.phase = phase

    def _integrator_hook(self, module, args, output):
        self._switch("losses")
        if output.requires_grad:
            output.register_hook(lambda grad: self._switch("network_backward"))


def profile_blocks(model, warp_layer, integrator, batch, device, training=True):
    """
    bf16 forward with hooks on the top-level blocks; training=True adds a backward, so 'retained' is what
    each block keeps for it. Returns (DataFrame per block, summary). Profile an uncompiled model.
    """
    _require_cuda(device)
    moving, fixed = batch["moving"].to(device), batch["fixed"].to(device)
    blocks = profiled_blocks(model, integrator)
    was_training = model.training
    model.train(training)
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    with _BlockMemoryRecorder(blocks) as recorder, torch.set_grad_enabled(training), autocast_bf16(device):
        dvf, velocity = predict_dvf(model, integrator, warp_layer, moving, fixed)
    torch.cuda.synchronize()
    forward_time = time.perf_counter() - t0
    retained = torch.cuda.memory_allocated() - baseline
    summary = {"baseline_MiB": baseline / MIB, "forward_retained_MiB": retained / MIB,
               "forward_peak_MiB": recorder.peak / MIB, "forward_s": forward_time}

    if training:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        dvf.float().square().mean().backward()
        torch.cuda.synchronize()
        summary["backward_peak_MiB"] = torch.cuda.max_memory_allocated() / MIB
        summary["backward_s"] = time.perf_counter() - t0
        summary["gradients_MiB"] = _param_bytes(p.grad for p in model.parameters() if p.grad is not None) / MIB
    summary["peak_MiB"] = max(summary["forward_peak_MiB"], summary.get("backward_peak_MiB", 0.0))

    del dvf, velocity
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    rows = [{"block": name, "calls": r["calls"], "weights_MiB": _param_bytes(blocks[name].parameters()) / MIB,
             "output_MiB": r["output"] / MIB, "retained_MiB": r["retained"] / MIB,
             "peak_increase_MiB": r["peak_increase"] / MIB, "time_ms": r["time"] * 1e3}
            for name, r in sorted(recorder.records.items(), key=lambda item: item[1]["order"])]
    inside = sum(r["retained"] for r in recorder.records.values())
    rows.append({"block": "other", "calls": 0, "weights_MiB": 0.0, "output_MiB": 0.0,
                 "retained_MiB": (retained - inside) / MIB, "peak_increase_MiB": 0.0, "time_ms": 0.0})
    return pd.DataFrame(rows), summary


def profile_tto(model, warp_layer, integrator, batch, device, weights_path, cfg=None, n_steps=2, **tto_kwargs):
    """
    Peak VRAM of a short run_tto, per phase and split into weights, gradients, Adam states and
    activations. tto_kwargs go to run_tto; the model keeps the weights of weights_path. n_steps >= 2.
    """
    _require_cuda(device)
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2 (Adam allocates its states at the first step)")
    cfg = cfg or TTOConfig()
    switch_step = n_steps - 1 if cfg.switch_step < cfg.n_steps else n_steps
    short_cfg = replace(cfg, n_steps=n_steps, switch_step=switch_step, log_every=10 ** 9)
    model.zero_grad(set_to_none=True)
    requires_grad = [p.requires_grad for p in model.parameters()]

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    with _PhaseMemoryRecorder(model, integrator) as recorder:
        run_tto(model, warp_layer, integrator, batch, device, weights_path, short_cfg, **tto_kwargs)
    elapsed = time.perf_counter() - t0
    peak = max(recorder.peaks.values())

    params = list(model.parameters())
    trainable = _param_bytes(p for p in params if p.requires_grad)
    gradients, adam = trainable, 2 * trainable
    model.zero_grad(set_to_none=True)
    for p, flag in zip(params, requires_grad):
        p.requires_grad = flag
    load_weights(model, weights_path, device)
    return {"peak_MiB": peak / MIB, "baseline_MiB": baseline / MIB, "weights_MiB": _param_bytes(params) / MIB,
            "gradients_MiB": gradients / MIB, "adam_MiB": adam / MIB,
            "activations_MiB": (peak - baseline - gradients - adam) / MIB,
            "step_s": elapsed / n_steps, **{f"{p}_peak_MiB": v / MIB for p, v in recorder.peaks.items()}}


def memory_snapshot(path, fn, *args, max_entries=200000, **kwargs):
    """Runs fn recording every CUDA allocation and saves the snapshot to path (pytorch.org/memory_viz)."""
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        return fn(*args, **kwargs)
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


def catch_oom(fn, *args, **kwargs):
    """Calls fn(*args, **kwargs); on CUDA out of memory prints the error, frees the cache and returns None."""
    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {fn.__name__}: {str(e).splitlines()[0]}")
    gc.collect()
    torch.cuda.empty_cache()
    return None
