"""Many behaviors on one frozen base, routed per token.

A Bank holds several behaviors (each a directory of LARA modules) over a single
frozen model, plus a small router that reads the frozen base hidden state and
decides, per token, how to weight them.

The router reads the *frozen* stream, before any module has written to it. That
is what makes routing independent of which behaviors happen to be loaded, and
therefore what makes `add()` cheap: adding a behavior does not invalidate
anything the router already learned about the others.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import LARAModule, decoder_layers, hidden_size, check_base, LARA

__all__ = ["Bank", "Router"]


class Router(nn.Module):
    """Per-token distribution over behaviors, read off the frozen base."""

    def __init__(self, d, n_behaviors, hidden=0):
        super().__init__()
        self.n = n_behaviors
        self.net = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_behaviors))
                    if hidden > 0 else nn.Linear(d, n_behaviors))

    def forward(self, h):                      # [B,T,d] -> [B,T,N]
        return self.net(h.float())

    def grow(self, n_new=1):
        """Append rows for new behaviors, keeping the existing ones intact."""
        if isinstance(self.net, nn.Sequential):
            old = self.net[-1]
        else:
            old = self.net
        new = nn.Linear(old.in_features, old.out_features + n_new,
                        bias=old.bias is not None).to(old.weight.device)
        with torch.no_grad():
            new.weight[:old.out_features] = old.weight
            if old.bias is not None:
                new.bias[:old.out_features] = old.bias
                new.bias[old.out_features:] = 0.0
        if isinstance(self.net, nn.Sequential):
            self.net[-1] = new
        else:
            self.net = new
        self.n += n_new
        return self


class _Pin:
    def __init__(self, bank, spec):
        self.bank, self.spec = bank, spec

    def __enter__(self):
        self.prev = self.bank._pinned
        self.bank._pinned = self.bank._pin_vector(self.spec)
        return self.bank

    def __exit__(self, *a):
        self.bank._pinned = self.prev


class Bank:
    """Several behaviors resident on one frozen model.

        bank = Bank(model, tokenizer)
        bank.add("code",   "behaviors/code")
        bank.add("polite", "behaviors/polite")     # trained with DPO, same artifact
        bank.fit_router()
        bank.top_k = 2
        out = model.generate(**inputs)             # routing happens inside

        with bank.pin("code"):                     # ignore the router
            out = model.generate(**inputs)
    """

    ATTR = "lara_bank"

    def __init__(self, model, tokenizer=None, *, route_layer=None, router_hidden=0,
                 top_k=None, device=None):
        if hasattr(model, self.ATTR):
            raise RuntimeError("This model already carries a Bank. Call .detach() first.")
        self.model = model
        self.tok = tokenizer
        self.blocks = decoder_layers(model)
        self.d = hidden_size(model)
        self.device = device or next(model.parameters()).device

        self.names: list[str] = []
        self.sets: list[nn.ModuleDict] = []
        self.samples: dict[str, list[str]] = {}
        self.layer_ids: list[int] | None = None
        self._route_layer = route_layer
        self.router_hidden = router_hidden
        self.router: Router | None = None

        self.top_k = top_k          # None = full soft; 1 = hard; k = blend top k
        self._enabled = True
        self._pinned = None
        self._w = None              # last routing weights [B,T,N]
        self._active = None         # which behaviors carry weight this forward
        self._logits = None
        self._handles: list = []
        self._store = nn.ModuleList()
        setattr(model, self.ATTR, self._store)

    # ── behaviors ────────────────────────────────────────────────────────────
    def add(self, name, behavior, *, samples=None, strict=True):
        """Add a behavior from a directory (or an in-memory LARA)."""
        if name in self.names:
            raise ValueError(f"behavior '{name}' is already in the bank")

        if isinstance(behavior, LARA):
            state = {k: v.detach().clone() for k, v in behavior.modules_.state_dict().items()}
            cfg = behavior.config.to_json()
            samp = samples or []
        else:
            state, cfg, samp = LARA.load_state(behavior, device=self.device)
            check_base(cfg, self.model, strict=strict)
            samp = samples or samp

        layers = list(cfg["layers"])
        if self.layer_ids is None:
            self.layer_ids = layers
        elif layers != self.layer_ids:
            raise ValueError(
                f"behavior '{name}' sits at layers {layers} but the bank uses "
                f"{self.layer_ids}. Behaviors in one bank must share a placement."
            )

        mods = nn.ModuleDict({
            str(l): LARAModule(self.d, cfg["rank"], cfg["alpha"]).to(self.device)
            for l in layers
        })
        mods.load_state_dict(state)
        mods.eval()
        for p in mods.parameters():
            p.requires_grad = False

        self.names.append(name)
        self.sets.append(mods)
        self._store.append(mods)
        self.samples[name] = list(samp)

        if self.router is None:
            self.router = Router(self.d, len(self.names), self.router_hidden).to(self.device)
        elif self.router.n < len(self.names):
            self.router.grow(len(self.names) - self.router.n)

        self._reattach()
        return self

    def remove(self, name):
        i = self._index(name)
        for coll in (self.names, self.sets):
            coll.pop(i)
        self.samples.pop(name, None)
        self._store = nn.ModuleList(self.sets)
        setattr(self.model, self.ATTR, self._store)
        self.router = None                     # dimensions changed; refit needed
        self._reattach()
        return self

    def _index(self, name):
        if name not in self.names:
            raise KeyError(f"no behavior '{name}' in bank {self.names}")
        return self.names.index(name)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        return self.sets[self._index(name)]

    # ── hooks ────────────────────────────────────────────────────────────────
    @property
    def route_layer(self):
        if self._route_layer is not None:
            return self._route_layer
        return self.layer_ids[0] if self.layer_ids else 0

    def _reattach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if not self.sets:
            return
        # the route hook is registered first, so on a shared layer it reads the
        # frozen output before any module writes to it
        self._handles.append(self.blocks[self.route_layer].register_forward_hook(self._route_hook()))
        for l in self.layer_ids:
            self._handles.append(self.blocks[l].register_forward_hook(self._inject_hook(l)))

    def _route_hook(self):
        def hook(module, inputs, output):
            if not self._enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if self._pinned is not None:
                w = self._pinned.to(hs.dtype).to(hs.device)
                self._w = w.expand(hs.shape[0], hs.shape[1], -1)
                self._logits = None
            else:
                logits = self.router(hs)
                self._logits = logits
                self._w = self._weights(logits).to(hs.dtype)
            with torch.no_grad():               # one sync per forward, not per layer
                self._active = (self._w.detach().abs().amax(dim=(0, 1)) > 0).tolist()
            return output
        return hook

    def _weights(self, logits):
        k = self.top_k
        n = logits.shape[-1]
        if k is None or k >= n:
            return F.softmax(logits, dim=-1)
        if k == 1:
            return F.one_hot(logits.argmax(-1), n).to(logits.dtype)
        vals, idx = logits.topk(k, dim=-1)
        w = torch.zeros_like(logits).scatter_(-1, idx, F.softmax(vals, dim=-1))
        return w

    def _inject_hook(self, l):
        def hook(module, inputs, output):
            if not self._enabled or self._w is None or not self.sets:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            w = self._w
            active = self._active or [True] * len(self.sets)
            acc = None
            for k, mods in enumerate(self.sets):
                if not active[k]:               # zero routing weight: skip the work
                    continue
                m = mods[str(l)]
                d = m.delta(hs) * m.gamma
                d = d * w[..., k].unsqueeze(-1).to(d.dtype)
                acc = d if acc is None else acc + d
            if acc is None:
                return output
            new = hs + acc
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if hasattr(self.model, self.ATTR):
            delattr(self.model, self.ATTR)

    # ── control ──────────────────────────────────────────────────────────────
    def _pin_vector(self, spec):
        v = torch.zeros(len(self.names))
        if isinstance(spec, str):
            v[self._index(spec)] = 1.0
        elif isinstance(spec, dict):
            for k, val in spec.items():
                v[self._index(k)] = float(val)
        else:
            raise TypeError("pin() takes a behavior name or a {name: weight} dict")
        return v.view(1, 1, -1)

    def pin(self, spec):
        """Ignore the router: `with bank.pin('code'):` or pin a manual blend."""
        return _Pin(self, spec)

    class _Disabled:
        def __init__(self, b): self.b = b
        def __enter__(self): self.prev = self.b._enabled; self.b._enabled = False
        def __exit__(self, *a): self.b._enabled = self.prev

    def disabled(self):
        """Run the frozen base."""
        return self._Disabled(self)

    @property
    def gamma(self):
        return {n: self.sets[i][str(self.layer_ids[0])].gamma for i, n in enumerate(self.names)}

    @gamma.setter
    def gamma(self, g):
        for mods in self.sets:
            for m in mods.values():
                m.gamma = float(g)

    def set_gamma(self, name, g):
        for m in self[name].values():
            m.gamma = float(g)

    # ── routing readouts ─────────────────────────────────────────────────────
    @torch.no_grad()
    def route_weights(self, input_ids):
        """Mean routing weight per behavior over the given tokens."""
        prev, self._w = self._w, None
        self.model(input_ids=input_ids.to(self.device), use_cache=False)
        w = self._w.float().mean(dim=(0, 1)).cpu()
        self._w = prev
        return {n: float(w[i]) for i, n in enumerate(self.names)}

    # ── router fitting ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _features(self, texts, max_len=256):
        """Cache the frozen-base hidden state the router reads. Modules off."""
        if self.tok is None:
            raise ValueError("Bank needs a tokenizer to fit the router")
        grab = {}
        h = self.blocks[self.route_layer].register_forward_hook(
            lambda m, i, o: grab.__setitem__(
                "x", (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
        feats = []
        with self.disabled():
            for t in texts:
                ids = self.tok(t, return_tensors="pt", truncation=True,
                               max_length=max_len).input_ids.to(self.device)
                if ids.numel() == 0:
                    continue
                self.model(input_ids=ids, use_cache=False)
                feats.append(grab["x"][0])                       # [T,d]
        h.remove()
        return feats

    def fit_router(self, samples=None, *, mode="refit", steps=300, lr=1e-3,
                   batch=512, max_len=256, verbose=False):
        """Train the router on each behavior's route samples.

        mode="refit"  retrain all rows (default; the router is tiny, this is seconds)
        mode="append" freeze existing rows and train only the new ones, for when
                      the earlier behaviors' samples are no longer available
        """
        samples = samples or self.samples
        missing = [n for n in self.names if not samples.get(n)]
        if missing:
            raise ValueError(
                f"no route samples for {missing}. Pass samples={{name: [texts]}}, "
                f"or save behaviors with route_samples=... so they carry their own."
            )

        X, y = [], []
        for i, n in enumerate(self.names):
            for f in self._features(samples[n], max_len=max_len):
                X.append(f)
                y.append(torch.full((f.shape[0],), i, dtype=torch.long))
        X = torch.cat(X).to(self.device)
        y = torch.cat(y).to(self.device)

        if self.router is None or self.router.n != len(self.names):
            self.router = Router(self.d, len(self.names), self.router_hidden).to(self.device)

        frozen_rows = 0
        if mode == "append":
            frozen_rows = getattr(self, "_fitted_rows", 0)

        params = [p for p in self.router.parameters()]
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr)

        head = self.router.net[-1] if isinstance(self.router.net, nn.Sequential) else self.router.net
        n_tok = X.shape[0]
        for step in range(steps):
            idx = torch.randint(0, n_tok, (min(batch, n_tok),), device=self.device)
            logits = self.router(X[idx].unsqueeze(0)).squeeze(0)
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if frozen_rows:                     # keep the earlier rows exactly as they were
                if head.weight.grad is not None:
                    head.weight.grad[:frozen_rows] = 0
                if head.bias is not None and head.bias.grad is not None:
                    head.bias.grad[:frozen_rows] = 0
            opt.step()
            if verbose and step % max(1, steps // 5) == 0:
                acc = (logits.argmax(-1) == y[idx]).float().mean().item()
                print(f"  router {step:4d}/{steps}  loss={loss.item():.3f}  acc={acc:.2f}")

        for p in params:
            p.requires_grad = False
        self._fitted_rows = len(self.names)
        self.router.eval()
        with torch.no_grad():                   # training-set accuracy, as a sanity signal
            pred = self.router(X.unsqueeze(0)).squeeze(0).argmax(-1)
            self.router_train_acc = float((pred == y).float().mean())
        if verbose:
            print(f"  router fitted: train acc {self.router_train_acc:.2f} "
                  f"over {len(self.names)} behaviors ({X.shape[0]:,} tokens)")
        return self

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        meta = {
            "names": self.names,
            "layers": self.layer_ids,
            "route_layer": self.route_layer,
            "router_hidden": self.router_hidden,
            "top_k": self.top_k,
            "base_model_id": getattr(self.model.config, "_name_or_path", None),
        }
        with open(os.path.join(path, "bank.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if self.router is not None:
            torch.save(self.router.state_dict(), os.path.join(path, "router.pt"))
        for i, n in enumerate(self.names):
            sub = os.path.join(path, "behaviors", n)
            os.makedirs(sub, exist_ok=True)
            state = {k: v.detach().cpu().contiguous() for k, v in self.sets[i].state_dict().items()}
            try:
                from safetensors.torch import save_file
                save_file(state, os.path.join(sub, "adapter.safetensors"))
            except ImportError:
                torch.save(state, os.path.join(sub, "adapter.pt"))
            m0 = self.sets[i][str(self.layer_ids[0])]
            with open(os.path.join(sub, "config.json"), "w") as f:
                json.dump({"layers": self.layer_ids,
                           "rank": m0.down.out_features,
                           "alpha": m0.scaling * m0.down.out_features,
                           "base_model_id": meta["base_model_id"]}, f, indent=2)
            if self.samples.get(n):
                with open(os.path.join(sub, "route_samples.jsonl"), "w") as f:
                    for t in self.samples[n]:
                        f.write(json.dumps({"text": t}) + "\n")
        return path

    @classmethod
    def load(cls, path, model, tokenizer=None, **kw):
        with open(os.path.join(path, "bank.json")) as f:
            meta = json.load(f)
        bank = cls(model, tokenizer, route_layer=meta.get("route_layer"),
                   router_hidden=meta.get("router_hidden", 0),
                   top_k=meta.get("top_k"), **kw)
        for n in meta["names"]:
            bank.add(n, os.path.join(path, "behaviors", n))
        rp = os.path.join(path, "router.pt")
        if os.path.exists(rp):
            bank.router.load_state_dict(torch.load(rp, map_location=bank.device))
            bank.router.eval()
            bank._fitted_rows = len(bank.names)
        return bank

    def __repr__(self):
        k = "soft" if self.top_k is None else f"top_k={self.top_k}"
        return f"Bank({len(self.names)} behaviors: {self.names}, {k})"
