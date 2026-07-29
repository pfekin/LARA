"""LARA core: attach low-rank residual modules to a frozen model.

A behavior is a file. Training is whatever you already do: HF Trainer, TRL
DPOTrainer/GRPOTrainer, or a hand-written loop. LARA only guarantees that the
base weights stay frozen and that the modules are the only trainable tensors.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn

__all__ = ["LARAConfig", "LARA", "resolve_layers", "decoder_layers"]


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture probing
# ─────────────────────────────────────────────────────────────────────────────
def decoder_layers(model):
    """The list of decoder blocks, across the common HF architectures."""
    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "layers"):
        return m.layers                          # Llama, Qwen, Mistral, Gemma
    t = getattr(model, "transformer", None)
    if t is not None and hasattr(t, "h"):
        return t.h                               # GPT-2, GPT-J
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError(
        "Could not locate decoder layers on this model. Pass an explicit layer "
        "list, or open an issue with the architecture name."
    )


def hidden_size(model):
    for attr in ("hidden_size", "n_embd", "d_model"):
        v = getattr(model.config, attr, None)
        if v:
            return v
    raise ValueError("Could not determine hidden size from model.config")


def resolve_layers(spec, depth):
    """`spec` is either a count or an explicit list.

    A count spreads that many modules evenly over the depth, avoiding the first
    and last blocks: position i sits at round(depth * (i+1) / (n+1)). On a
    28-layer model this gives [4, 8, 12, 16, 20, 24] for 6, and [14] for 1,
    which are the placements used in the paper.
    """
    if isinstance(spec, bool):
        raise TypeError("layers must be an int or a list of ints")
    if isinstance(spec, int):
        n = spec
        if not 1 <= n <= depth:
            raise ValueError(f"layers={n} out of range for a {depth}-layer model")
        out = sorted({int(depth * (i + 1) / (n + 1) + 0.5) for i in range(n)})
        out = [min(max(l, 0), depth - 1) for l in out]
        if len(out) != n:                        # collapsed on a very shallow model
            out = sorted(set(out))
        return out
    layers = sorted({int(l) for l in spec})
    for l in layers:
        if not 0 <= l < depth:
            raise ValueError(f"layer {l} out of range for a {depth}-layer model")
    return layers


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LARAConfig:
    layers: object = 6           # int (evenly spaced) or explicit list
    rank: int = 128
    alpha: float = 128.0
    base_model_id: str | None = None
    method: str | None = None    # free-text provenance: "ce", "dpo", "grpo", ...
    resolved_layers: list | None = None

    def to_json(self):
        d = asdict(self)
        d.pop("resolved_layers", None)
        d["layers"] = self.resolved_layers if self.resolved_layers else self.layers
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  The module: h <- h + (alpha/r) * W_up W_down LN(h), scaled by gamma
# ─────────────────────────────────────────────────────────────────────────────
class LARAModule(nn.Module):
    """Linear low-rank read of the residual stream, added back.

    W_up is zero-initialized, so at the start of training the module is a no-op
    and the model is exactly the base.
    """

    def __init__(self, d, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gamma = 1.0                      # inference-time scale; 1.0 while training

    def delta(self, h):
        """The vector this module contributes, before gamma."""
        out = self.up(self.down(self.norm(h.float())))
        return out.to(h.dtype) * self.scaling

    def forward(self, h):
        return h + self.delta(h) * self.gamma


# ─────────────────────────────────────────────────────────────────────────────
#  LARA — attach modules to a frozen model
# ─────────────────────────────────────────────────────────────────────────────
class LARA:
    """Attach LARA modules to a frozen causal LM.

        lara = LARA(model, layers=6, rank=128)
        trainer = Trainer(model=lara.model, ...)     # any trainer
        trainer.train()
        lara.save("behaviors/code", route_samples=texts)

    The modules are registered on the model itself (as `model.lara_modules`), so
    `model.parameters()` reaches them and standard trainers pick them up without
    any special handling.
    """

    ATTR = "lara_modules"

    def __init__(self, model, layers=6, rank=128, alpha=128.0, *,
                 freeze_base=True, base_model_id=None, method=None):
        if hasattr(model, self.ATTR):
            raise RuntimeError(
                "This model already carries LARA modules. Use a fresh model, or "
                "call .detach() on the previous LARA first."
            )
        self.model = model
        self.blocks = decoder_layers(model)
        depth = len(self.blocks)
        d = hidden_size(model)

        self.config = LARAConfig(
            layers=layers, rank=rank, alpha=alpha,
            base_model_id=base_model_id or getattr(model.config, "_name_or_path", None),
            method=method,
        )
        self.layer_ids = resolve_layers(layers, depth)
        self.config.resolved_layers = list(self.layer_ids)

        device = next(model.parameters()).device
        self.modules_ = nn.ModuleDict({
            str(l): LARAModule(d, rank, alpha).to(device) for l in self.layer_ids
        })
        setattr(model, self.ATTR, self.modules_)   # so model.parameters() sees them

        if freeze_base:
            for n_, p in model.named_parameters():
                p.requires_grad = n_.startswith(self.ATTR + ".")

        self._enabled = True
        self._handles = [self.blocks[l].register_forward_hook(self._hook(l))
                         for l in self.layer_ids]

    # --- mechanics -----------------------------------------------------------
    def _hook(self, l):
        mod = self.modules_[str(l)]

        def hook(module, inputs, output):
            if not self._enabled:
                return output
            if isinstance(output, tuple):
                return (mod(output[0]),) + tuple(output[1:])
            return mod(output)

        return hook

    def detach(self):
        """Remove hooks and modules. The model is byte-for-byte the base again."""
        for h in self._handles:
            h.remove()
        self._handles = []
        if hasattr(self.model, self.ATTR):
            delattr(self.model, self.ATTR)

    class _Disabled:
        def __init__(self, owner): self.owner = owner
        def __enter__(self): self.prev = self.owner._enabled; self.owner._enabled = False
        def __exit__(self, *a): self.owner._enabled = self.prev

    def disabled(self):
        """Context manager: run the frozen base. This is the free DPO reference."""
        return self._Disabled(self)

    # --- knobs ---------------------------------------------------------------
    @property
    def gamma(self):
        return next(iter(self.modules_.values())).gamma

    @gamma.setter
    def gamma(self, g):
        for m in self.modules_.values():
            m.gamma = float(g)

    def trainable_parameters(self):
        return [p for p in self.modules_.parameters() if p.requires_grad]

    def num_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    # --- persistence ---------------------------------------------------------
    def save(self, path, route_samples=None, method=None):
        """Write the behavior: weights, config, and optional router samples.

        `route_samples` is a list of short texts representative of this behavior.
        They are what lets a Bank fit its router later without needing the
        original training data, so a behavior can be shared and still routed.
        """
        os.makedirs(path, exist_ok=True)
        if method:
            self.config.method = method
        state = {k: v.detach().cpu().contiguous()
                 for k, v in self.modules_.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(state, os.path.join(path, "adapter.safetensors"))
        except ImportError:                       # torch fallback
            torch.save(state, os.path.join(path, "adapter.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_json(), f, indent=2)
        if route_samples:
            with open(os.path.join(path, "route_samples.jsonl"), "w") as f:
                for t in route_samples:
                    f.write(json.dumps({"text": t}) + "\n")
        return path

    @staticmethod
    def load_state(path, device="cpu"):
        """Read a behavior directory: (state_dict, config_dict, samples)."""
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        st_path = os.path.join(path, "adapter.safetensors")
        if os.path.exists(st_path):
            from safetensors.torch import load_file
            state = load_file(st_path, device=str(device))
        else:
            state = torch.load(os.path.join(path, "adapter.pt"), map_location=device)
        samples = []
        sp = os.path.join(path, "route_samples.jsonl")
        if os.path.exists(sp):
            with open(sp) as f:
                samples = [json.loads(line)["text"] for line in f if line.strip()]
        return state, cfg, samples

    @classmethod
    def from_pretrained(cls, model, path, **kw):
        """Attach a saved behavior to a frozen model."""
        state, cfg, _ = cls.load_state(path)
        check_base(cfg, model, strict=kw.pop("strict", True))
        obj = cls(model, layers=cfg["layers"], rank=cfg["rank"],
                  alpha=cfg["alpha"], method=cfg.get("method"), **kw)
        obj.modules_.load_state_dict(state)
        return obj

    def __repr__(self):
        return (f"LARA(layers={self.layer_ids}, rank={self.config.rank}, "
                f"trainable={self.num_trainable():,})")


def check_base(cfg, model, strict=True):
    """A behavior is tied to the base it was trained on. Fail loudly, not silently."""
    want = cfg.get("base_model_id")
    have = getattr(model.config, "_name_or_path", None)
    if want and have and want != have:
        msg = (f"behavior was trained on '{want}' but the model is '{have}'. "
               f"Adapters do not transfer between bases.")
        if strict:
            raise ValueError(msg)
        print("warning:", msg)
