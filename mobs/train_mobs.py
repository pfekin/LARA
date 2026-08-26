#!/usr/bin/env python3
"""Train a bank of MoBs behaviors.
    
    !pip -q install transformers accelerate datasets safetensors kernels
    !pip -q install git+https://github.com/pfekin/LARA.git
    
    python train_mobs.py --base Qwen/Qwen3-4B --out behaviors
    python train_mobs.py --only code,math --steps 400
    python train_mobs.py --push pfekin/mobs-qwen3-4b

Colab: paste the whole file into one cell, then call

    main()                                    # defaults
    main(["--base", "Qwen/Qwen3.5-4B", "--out", "behaviors",
          "--push", "you/mobs-qwen3.5-4b"])

after installing:

    !pip -q install transformers accelerate datasets safetensors
    !pip -q install git+https://github.com/pfekin/LARA.git

Adding a dataset or a method means adding one decorated function; nothing else
in the file needs to change. Behaviors train independently against the frozen
base and only meet at fit_router(), which is what lets a behavior trained
elsewhere drop into the bank later with just its route samples.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from dataclasses import dataclass, field

import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
#  Behavior definitions — edit here
# ─────────────────────────────────────────────────────────────────────────────
#  dataset:  "hf:repo[:split]"  |  "file:path.jsonl"  |  a name registered below
#  method:   any key in METHODS
#  Each dataset loader returns SFT dicts {"prompt","completion"} or
#  DPO dicts {"prompt","chosen","rejected"}.

BEHAVIORS = [
    dict(name="code",    method="sft", dataset="hf:sahil2801/CodeAlpaca-20k",
         n=3000, layers=6, rank=128, alpha=128),
    dict(name="math",    method="sft", dataset="hf:openai/gsm8k:main",
         n=3000, layers=6, rank=128, alpha=128),
    dict(name="medical", method="sft", dataset="hf:lavita/ChatDoctor-HealthCareMagic-100k",
         n=3000, layers=6, rank=128, alpha=128),
    dict(name="summary", method="sft", dataset="hf:EdinburghNLP/xsum",
         n=3000, layers=6, rank=128, alpha=128),
    dict(name="polite",  method="dpo", dataset="hf:Anthropic/hh-rlhf",
         n=2000, layers=1, rank=128, alpha=128, beta=0.1,
         micro_batch=2, max_len=256),
]

DEFAULTS = dict(
    base="Qwen/Qwen3.5-4B",
    out="behaviors",
    max_steps=600, batch=2, accum=8, lr=1e-4, max_len=320,
    eval_every=50, patience=2, min_delta=5e-3, eval_frac=0.1,
    route_samples=200, seed=0,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Registries
# ─────────────────────────────────────────────────────────────────────────────
DATASETS: dict = {}
METHODS: dict = {}


def dataset(name):
    def deco(fn):
        DATASETS[name] = fn
        return fn
    return deco


def method(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


# ── dataset loaders ──────────────────────────────────────────────────────────
def _hf(spec, n, seed):
    """hf:repo[:config_or_split][:split]"""
    from datasets import load_dataset
    parts = spec.split(":")[1:]
    repo = parts[0]
    cfg = parts[1] if len(parts) > 1 else None
    split = parts[2] if len(parts) > 2 else "train"
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    ds = ds.shuffle(seed=seed)
    return [ds[i] for i in range(min(n, len(ds)))]


def _file(spec, n, seed):
    """file:path.jsonl — one json object per line, already in the right shape."""
    path = spec.split(":", 1)[1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


# Field mapping: raw HF rows -> the shape the trainers want. Add a case here
# rather than reshaping the dataset on disk.
FIELD_MAPS = {
    "CodeAlpaca":   lambda r: {"prompt": (r["instruction"] + ("\n" + r["input"] if r.get("input") else "")),
                               "completion": r["output"]},
    "gsm8k":        lambda r: {"prompt": r["question"], "completion": r["answer"]},
    "ChatDoctor":   lambda r: {"prompt": r["input"], "completion": r["output"]},
    "xsum":         lambda r: {"prompt": "Summarise:\n" + r["document"][:3000],
                               "completion": r["summary"]},
    "hh-rlhf":      lambda r: {"prompt": "", "chosen": r["chosen"], "rejected": r["rejected"]},
}


def normalise(rows, spec):
    """Apply the first field map whose key appears in the dataset spec."""
    for key, fn in FIELD_MAPS.items():
        if key.lower() in spec.lower():
            return [fn(r) for r in rows]
    need = {"prompt", "completion"} , {"prompt", "chosen", "rejected"}
    if rows and (need[0] <= rows[0].keys() or need[1] <= rows[0].keys()):
        return rows
    raise ValueError(
        f"no field map for '{spec}' and rows are not already in shape. "
        f"Add an entry to FIELD_MAPS, or pre-convert to jsonl with "
        f"prompt/completion or prompt/chosen/rejected keys. Saw: {list(rows[0].keys())}")


def load_rows(spec, n, seed):
    if spec in DATASETS:
        return DATASETS[spec](n, seed)
    if spec.startswith("hf:"):
        return normalise(_hf(spec, n, seed), spec)
    if spec.startswith("file:"):
        return normalise(_file(spec, n, seed), spec)
    raise ValueError(f"unknown dataset spec: {spec}")


# ── prompt formatting ────────────────────────────────────────────────────────
_TEMPLATE_KW = None


def chatml(tok, prompt, completion=None):
    """enable_thinking is a Qwen3 template argument; other bases reject it.

    Decided once on first call and cached, so this costs nothing per example.
    """
    global _TEMPLATE_KW
    msg = [{"role": "user", "content": prompt}]
    if _TEMPLATE_KW is None:
        for kw in ({"enable_thinking": False}, {}):
            try:
                tok.apply_chat_template(msg, tokenize=False,
                                        add_generation_prompt=True, **kw)
                _TEMPLATE_KW = kw
                if not kw:
                    print("  note: template rejected enable_thinking; omitting it")
                break
            except Exception:
                continue
        if _TEMPLATE_KW is None:
            _TEMPLATE_KW = {}
    text = tok.apply_chat_template(msg, tokenize=False,
                                   add_generation_prompt=True, **_TEMPLATE_KW)
    return text + completion if completion is not None else text


def sft_batch(tok, rows, max_len, device):
    """Tokenise, and mask the prompt so loss falls only on the completion."""
    input_ids, labels = [], []
    for r in rows:
        p = tok(chatml(tok, r["prompt"]), add_special_tokens=False).input_ids
        c = tok(r["completion"] + tok.eos_token, add_special_tokens=False).input_ids
        ids = (p + c)[:max_len]
        lab = ([-100] * len(p) + c)[:max_len]
        input_ids.append(ids); labels.append(lab)
    m = max(len(x) for x in input_ids)
    pad = tok.pad_token_id or tok.eos_token_id
    att = [[1] * len(x) + [0] * (m - len(x)) for x in input_ids]
    input_ids = [x + [pad] * (m - len(x)) for x in input_ids]
    labels = [x + [-100] * (m - len(x)) for x in labels]
    t = lambda v, d=torch.long: torch.tensor(v, dtype=d, device=device)
    return t(input_ids), t(att), t(labels)





def lara_target(model, verbose=True):
    """The object to hand LARA and Bank.

    lara.decoder_layers() looks for `model.model.layers`. A multimodal wrapper
    puts the stack one level deeper -- Qwen3.5 keeps it at
    `model.model.language_model.layers` -- so pass the inner text model instead.
    Hooks land on the same block objects either way, and the outer model's
    forward still runs through them.
    """
    from lara import decoder_layers
    try:
        decoder_layers(model)
        return model
    except Exception:
        pass
    for name, sub in model.named_modules():
        if isinstance(getattr(sub, "layers", None), torch.nn.ModuleList) \
                and len(sub.layers) > 2:
            try:
                decoder_layers(sub)
            except Exception:
                continue
            if verbose:
                print(f"  note: attaching to model.{name} "
                      f"({len(sub.layers)} blocks) — the wrapper hides them")
            return sub
    raise RuntimeError(
        "could not find a decoder stack lara can attach to; run check_base.py "
        "and look at the candidate block lists it prints")


def freeze_all(model):
    """LARA freezes what it attaches to; on a wrapper that leaves the vision
    tower and head trainable, and autograd would keep their activations."""
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += 1
    return n


def load_base(name, dtype, attn, trust=True):
    """AutoModelForCausalLM first; fall back for wrappers that are not that.

    Newer bases are sometimes natively multimodal, in which case the causal-LM
    class either does not exist or wraps a nested language model.
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    tries = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    try:
        from transformers import AutoModelForImageTextToText
        tries.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    except ImportError:
        pass
    last = None
    for label, cls in tries:
        for a in ([attn] if attn else []) + ["sdpa", "eager", None]:
            kw = dict(dtype=dtype, trust_remote_code=trust)
            if a:
                kw["attn_implementation"] = a
            try:
                m = cls.from_pretrained(name, **kw)
                if label != "AutoModelForCausalLM" or a != attn:
                    print(f"  note: loaded via {label}"
                          + (f" with attn={a}" if a else " with default attention"))
                return m
            except Exception as e:
                last = e
    raise RuntimeError(f"could not load {name}: {type(last).__name__}: {last}")


# ── loss helpers ─────────────────────────────────────────────────────────────
# Qwen3's vocabulary is 151,936. Running the LM head over every position and
# upcasting to fp32 costs ~2.5 GB at batch 4 x 512 -- more than the model.
# Only the completion tokens carry loss, so only they need logits.

def _decoder(model):
    """The module that owns the decoder blocks.

    get_decoder() is right for plain causal LMs. Multimodal wrappers can return
    something one level too high, so fall back to searching for the module that
    actually holds the layer list.
    """
    dec = None
    try:
        dec = model.get_decoder()
    except Exception:
        pass
    if dec is not None and any(hasattr(dec, a) for a in ("layers", "blocks", "h")):
        return dec
    for name, mod in model.named_modules():
        if hasattr(mod, "layers") and isinstance(
                getattr(mod, "layers"), torch.nn.ModuleList):
            if dec is None:
                print(f"  note: using {name or 'model'} as the decoder")
            return mod
    if dec is None:
        raise RuntimeError("could not locate the decoder stack on this model")
    return dec


def _hidden(model, ids, att):
    out = _decoder(model)(input_ids=ids, attention_mask=att, use_cache=False)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def masked_ce(model, ids, att, lab):
    """Cross-entropy over the unmasked positions only."""
    h = _hidden(model, ids, att)[:, :-1]
    tgt = lab[:, 1:]
    keep = tgt != -100
    if not keep.any():
        return h.sum() * 0.0
    head = model.get_output_embeddings()
    logits = head(h[keep]).float()
    return F.cross_entropy(logits, tgt[keep])


def masked_logprob_sum(model, ids, att, lab):
    """Per-sequence sum of log p(target) over the unmasked positions."""
    h = _hidden(model, ids, att)[:, :-1]
    tgt = lab[:, 1:]
    keep = tgt != -100
    head = model.get_output_embeddings()
    B = ids.shape[0]
    out = h.new_zeros(B, dtype=torch.float32)
    for b in range(B):                       # per row: keeps the head input small
        k = keep[b]
        if not k.any():
            continue
        lp = torch.log_softmax(head(h[b][k]).float(), dim=-1)
        out = out + torch.zeros_like(out).index_put(
            (torch.tensor([b], device=out.device),),
            lp.gather(-1, tgt[b][k].unsqueeze(-1)).squeeze(-1).sum().unsqueeze(0))
    return out



class _Scalar:
    """Stands in for a tensor where only .item() is used."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


# ── early stopping ───────────────────────────────────────────────────────────
class Stopper:
    """Stop when held-out loss stops improving.

    Training loss alone cannot tell "converged" from "memorised", and a 4M
    parameter module on 3k examples can do the latter. `patience` is counted in
    evaluations, not steps, so it means the same thing at any --eval-every.
    """

    def __init__(self, patience=3, min_delta=1e-3, max_steps=4000):
        self.patience, self.min_delta, self.max_steps = patience, min_delta, max_steps
        self.best, self.bad, self.best_step = float("inf"), 0, 0
        self.best_state = None

    # LARA is not an nn.Module, so snapshot the trainable tensors directly.
    # Ordering from trainable_parameters() is stable across calls; the length
    # check below turns any future change into a loud failure rather than a
    # silently mismatched restore.
    @staticmethod
    def _snap(module):
        return [p.detach().clone() for p in module.trainable_parameters()]

    def update(self, step, loss, module):
        if loss < self.best - self.min_delta:
            self.best, self.bad, self.best_step = loss, 0, step
            self.best_state = self._snap(module)
            return False, "best"
        self.bad += 1
        return self.bad >= self.patience, f"{self.bad}/{self.patience}"

    def restore(self, module):
        """Roll back to the best checkpoint, not whatever the last step left."""
        if self.best_state is None:
            return False
        params = list(module.trainable_parameters())
        if len(params) != len(self.best_state):
            print(f"  WARNING: parameter list changed shape "
                  f"({len(params)} vs {len(self.best_state)}); not restoring")
            return False
        with torch.no_grad():
            for p, q in zip(params, self.best_state):
                p.copy_(q)
        return True


def split_rows(rows, frac, seed, cap=96):
    """Hold out a slice, capped.

    The held-out loss is a stopping signal, not a headline number. Beyond ~100
    examples the extra precision costs seconds per evaluation and changes no
    decision.
    """
    r = list(rows)
    random.Random(seed).shuffle(r)
    n_eval = min(cap, max(16, int(len(r) * frac))) if len(r) > 64 else 0
    return (r[n_eval:], r[:n_eval]) if n_eval else (r, [])


# ── methods ──────────────────────────────────────────────────────────────────
@method("sft")
def train_sft(lr_mod, model, tok, rows, cfg, log):
    train, held = split_rows(rows, cfg.eval_frac, cfg.seed, cfg.eval_cap)
    opt = torch.optim.AdamW(lr_mod.trainable_parameters(), lr=cfg.lr)
    dev = next(model.parameters()).device
    stop = Stopper(cfg.patience, cfg.min_delta, cfg.max_steps)

    def evaluate():
        if not held:
            return None
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(held), cfg.batch):
                chunk = held[i:i + cfg.batch]
                ids, att, lab = sft_batch(tok, chunk, cfg.max_len, dev)
                tot += masked_ce(model, ids, att, lab).item() * len(chunk)
                n += len(chunk)
        return tot / max(n, 1)

    step = 0
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.accum):
            batch = random.sample(train, min(cfg.batch, len(train)))
            ids, att, lab = sft_batch(tok, batch, cfg.max_len, dev)
            loss = masked_ce(model, ids, att, lab)
            (loss / cfg.accum).backward()
            total += loss.item() / cfg.accum
        torch.nn.utils.clip_grad_norm_(lr_mod.trainable_parameters(), 1.0)
        opt.step()
        step += 1

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            ev = evaluate()
            done, tag = stop.update(step, ev if ev is not None else total, lr_mod)
            log(step, total, extra=(f"eval={ev:.4f} [{tag}]" if ev is not None
                                    else f"[{tag}]"))
            if done:
                print(f"  stopped at {step}: no improvement for "
                      f"{cfg.patience} evals (best {stop.best:.4f} @ {stop.best_step})")
                break
        elif step % max(1, cfg.eval_every // 2) == 0:
            log(step, total)

    if stop.restore(lr_mod):
        print(f"  restored the checkpoint from step {stop.best_step}")


@method("dpo")
def train_dpo(lr_mod, model, tok, rows, cfg, log):
    """The reference model is the base with the modules off — no second copy."""
    beta = cfg.extra.get("beta", 0.1)
    train, held = split_rows(rows, cfg.eval_frac, cfg.seed, cfg.eval_cap)
    opt = torch.optim.AdamW(lr_mod.trainable_parameters(), lr=cfg.lr)
    dev = next(model.parameters()).device
    stop = Stopper(cfg.patience, cfg.min_delta, cfg.max_steps)

    def logps(rows_, key):
        pairs = [{"prompt": r["prompt"], "completion": r[key]} for r in rows_]
        ids, att, lab = sft_batch(tok, pairs, cfg.max_len, dev)
        return masked_logprob_sum(model, ids, att, lab)

    # The reference is the frozen base with the modules off, so its logprobs
    # for a given row never change. Recomputing them every step was half the
    # work in the loop. Cache on first sight; by the end most rows are hits.
    ref_cache = {}

    def reference(batch):
        miss = [r for r in batch if id(r) not in ref_cache]
        if miss:
            with torch.no_grad(), lr_mod.disabled():
                c = logps(miss, "chosen")
                r_ = logps(miss, "rejected")
            for k, row in enumerate(miss):
                ref_cache[id(row)] = (c[k].item(), r_[k].item())
        vals = [ref_cache[id(r)] for r in batch]
        t = lambda i: torch.tensor([v[i] for v in vals], dtype=torch.float32,
                                   device=dev)
        return t(0), t(1)

    def pair_loss(batch):
        ref_c, ref_r = reference(batch)
        pol_c, pol_r = logps(batch, "chosen"), logps(batch, "rejected")
        margin = (pol_c - ref_c) - (pol_r - ref_r)
        return -F.logsigmoid(beta * margin).mean(), margin.mean()

    def evaluate():
        if not held:
            return None, None
        tot, marg, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(held), micro):
                chunk = held[i:i + micro]
                l, m = pair_loss(chunk)
                tot += l.item() * len(chunk); marg += m.item() * len(chunk)
                n += len(chunk)
                del l, m
        return tot / max(n, 1), marg / max(n, 1)

    # DPO keeps two graphs alive per example -- chosen and rejected -- where SFT
    # keeps one. Micro-batching them and backward-ing each keeps peak memory at
    # one pair rather than the whole batch.
    micro = cfg.extra.get("micro_batch", 2)
    step = 0
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        batch = random.sample(train, min(cfg.batch * cfg.accum, len(train)))
        tot_loss, tot_margin, n_micro = 0.0, 0.0, 0
        for i in range(0, len(batch), micro):
            chunk = batch[i:i + micro]
            l, m = pair_loss(chunk)
            (l / max(1, len(batch) // micro)).backward()
            tot_loss += l.item(); tot_margin += m.item(); n_micro += 1
            del l, m
        loss = _Scalar(tot_loss / n_micro)
        margin = _Scalar(tot_margin / n_micro)
        torch.nn.utils.clip_grad_norm_(lr_mod.trainable_parameters(), 1.0)
        opt.step()
        step += 1

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            ev, ev_m = evaluate()
            done, tag = stop.update(step, ev if ev is not None else loss.item(), lr_mod)
            log(step, loss.item(),
                extra=(f"margin={margin.item():+.3f} eval={ev:.4f} "
                       f"eval_margin={ev_m:+.3f} [{tag}]" if ev is not None
                       else f"margin={margin.item():+.3f} [{tag}]"))
            if done:
                print(f"  stopped at {step}: no improvement for "
                      f"{cfg.patience} evals (best {stop.best:.4f} @ {stop.best_step})")
                break
        elif step % max(1, cfg.eval_every // 2) == 0:
            log(step, loss.item(), extra=f"margin={margin.item():+.3f}")

    if stop.restore(lr_mod):
        print(f"  restored the checkpoint from step {stop.best_step}")


# ─────────────────────────────────────────────────────────────────────────────
def preflight():
    """transformers imports break outright when `kernels` is a mismatched version.

    hub_kernels.py builds LayerRepository(...) at import time; if the installed
    kernels expects a `revision` or `version` argument and this transformers
    does not pass one, every model import fails with a traceback that looks
    like a model problem and is not.
    """
    try:
        import transformers.activations  # noqa: F401
        return
    except Exception as e:
        if "revision or a version" not in str(e) and "LayerRepository" not in repr(e):
            raise
    import importlib.metadata as md
    ver = lambda p: (md.version(p) if p else "?")
    try:
        k, t = ver("kernels"), ver("transformers")
    except Exception:
        k = t = "?"
    raise SystemExit(
        f"transformers ({t}) and kernels ({k}) are incompatible — every model "
        f"import fails, not just this one.\n"
        f"  CPU box : pip uninstall -y kernels      (they are CUDA kernels; "
        f"transformers works fine without them)\n"
        f"  GPU box : pip install -U transformers   (or pin kernels to a version "
        f"this transformers expects)")


def gpu_note(dev):
    """Free / total VRAM, so creeping allocation between behaviors is visible."""
    if dev != "cuda":
        return ""
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return (f"   [vram {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB used, "
            f"peak {peak:.1f} GB]")


def hf_token():
    """A write token, from wherever this is running.

    Colab secrets are not environment variables -- they need userdata.get(),
    and the secret must have notebook access granted (key icon, left sidebar).
    Falls back to the environment, then to a cached `huggingface-cli login`, so
    the same call works in a notebook and a shell.
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok, "environment"
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok      # so libraries downstream see it
            return tok, "Colab secret"
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder
        tok = HfFolder.get_token()
        if tok:
            return tok, "cached CLI login"
    except Exception:
        pass
    return None, None


def push_folder(folder, repo, path_in_repo=None):
    from huggingface_hub import HfApi
    tok, where = hf_token()
    if not tok:
        raise SystemExit(
            "--push needs a write token and none was found.\n"
            "  Colab : add HF_TOKEN under the key icon and grant notebook access\n"
            "  shell : export HF_TOKEN=hf_...   (or run huggingface-cli login)")
    dest = f"{repo}/{path_in_repo}" if path_in_repo else repo
    print(f"  pushing {folder} -> {dest}  (token from {where})")
    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model",
                      path_in_repo=path_in_repo)
    if not path_in_repo:
        print(f"  https://huggingface.co/{repo}")


@dataclass
class Cfg:
    batch: int; accum: int; lr: float; max_len: int
    max_steps: int; eval_every: int; patience: int; min_delta: float
    eval_frac: float; eval_cap: int; seed: int
    extra: dict = field(default_factory=dict)


def main(argv=None):
    # In a notebook sys.argv carries the kernel's own "-f .../kernel.json",
    # which argparse rejects. Treat a bare main() there as "use the defaults".
    if argv is None and ("ipykernel" in sys.modules or "google.colab" in sys.modules):
        argv = []
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--only", default=None, help="comma-separated behavior names")
    ap.add_argument("--steps", type=int, default=None,
                    help="fixed step count; disables early stopping")
    ap.add_argument("--push", default=None, help="HF repo id to upload the bank to")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--no-trust", dest="trust", action="store_false", default=True)
    ap.add_argument("--redo", action="store_true",
                    help="retrain behaviors that are already on disk")
    ap.add_argument("--eval_cap", type=int, default=96,
                    help="max held-out examples; more costs time, changes nothing")
    ap.add_argument("--layers", type=int, default=None,
                    help="override the per-behavior layer count for every behavior")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    a = ap.parse_args(argv)

    preflight()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lara import LARA, Bank

    want = set(a.only.split(",")) if a.only else None
    specs = [b for b in BEHAVIORS if not want or b["name"] in want]
    if not specs:
        raise SystemExit(f"no behaviors matched {a.only}")

    if a.steps:                       # explicit --steps means "no early stopping"
        a.max_steps, a.patience = a.steps, 10 ** 9

    if a.push and not hf_token()[0]:
        raise SystemExit(
            f"--push {a.push} was given but no write token was found. Checking "
            f"stops now rather than after training.\n"
            "  Colab : add HF_TOKEN under the key icon and grant this notebook access\n"
            "  shell : export HF_TOKEN=hf_...   (or run huggingface-cli login)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = getattr(torch, a.dtype)
    print(f"base {a.base} on {dev} ({a.dtype}, {a.attn}, "
          f"grad_ckpt={a.grad_ckpt}, batch {a.batch}x{a.accum} @ {a.max_len})")
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=a.trust)
    # sdpa, not eager: the hooks are on decoder layers, so the attention
    # implementation is free to be the memory-efficient one. Eager materialises
    # [B, heads, T, T] per layer, which alone was 2.4 GB at batch 4 x 512.
    model = load_base(a.base, dt, a.attn, a.trust).to(dev)
    model.config.use_cache = False
    frozen = freeze_all(model)
    target = lara_target(model)
    if target is not model:
        print(f"  froze {frozen} parameter tensors on the wrapper")
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()   # needed with a fully frozen base
    model.eval()

    os.makedirs(a.out, exist_ok=True)
    for spec in specs:
        name = spec["name"]
        done = os.path.join(a.out, name, "adapter.safetensors")
        if os.path.exists(done) and not a.redo:
            print(f"\n=== {name}: already trained, skipping (--redo to force) ===")
            continue
        print(f"\n=== {name}  ({spec['method']}, {spec['dataset']}) ==={gpu_note(dev)}")
        rows = load_rows(spec["dataset"], spec.get("n", 2000), a.seed)
        print(f"  {len(rows)} examples")

        mod = LARA(target, layers=a.layers or spec.get("layers", 6),
                   rank=spec.get("rank", 128),
                   alpha=spec.get("alpha", 128.0), base_model_id=a.base,
                   method=spec["method"])
        print(f"  {mod.num_trainable():,} trainable parameters")

        cfg = Cfg(a.batch, a.accum, a.lr, spec.get("max_len", a.max_len),
                  spec.get("max_steps", a.max_steps), a.eval_every,
                  a.patience, a.min_delta, a.eval_frac, a.eval_cap, a.seed,
                  {k: v for k, v in spec.items()
                   if k not in ("name", "method", "dataset", "n", "layers",
                                "rank", "alpha", "max_steps", "max_len")})
        t0 = time.time()

        def log(step, loss, extra=""):
            el = time.time() - t0
            eta = ""
            if step >= 25:
                rate = el / step
                left = (cfg.max_steps - step) * rate
                eta = f" eta<={left / 60:.0f}m @ {rate:.1f}s/step"
            print(f"  {step:5d}/{cfg.max_steps}  loss={loss:.4f}  {extra}  "
                  f"[{el:.0f}s{eta}]")

        METHODS[spec["method"]](mod, model, tok, rows, cfg, log)

        # route samples: representative *inputs*, so a bank can route this
        # behavior later without ever seeing its training data
        samples = [r.get("prompt") or r.get("chosen", "")[:400]
                   for r in rows[:a.route_samples]]
        samples = [s for s in samples if s]
        path = os.path.join(a.out, name)
        mod.save(path, route_samples=samples, method=spec["method"])
        if a.push:
            # Colab sessions die, and the filesystem dies with them. Uploading
            # each behavior as it completes means a timeout costs the one in
            # flight, not the four already trained.
            try:
                push_folder(path, a.push, path_in_repo=name)
            except Exception as e:
                print(f"  upload failed ({type(e).__name__}: {e}); "
                      f"{path} is still on local disk")
        mod.detach()
        del mod
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print(f"  saved {path}{gpu_note(dev)}")

    # ── fit the router over whatever is now on disk ──────────────────────────
    names = sorted(d for d in os.listdir(a.out) if os.path.isdir(os.path.join(a.out, d)))
    print(f"\n=== router over {names} ===")
    model.config.use_cache = False
    bank = Bank(target, tok, top_k=None)
    for n in names:
        bank.add(n, os.path.join(a.out, n))
    bank.fit_router(verbose=True)
    bank.save(os.path.join(a.out, "_bank"))
    json.dump({"base": a.base, "behaviors": names,
               "layers": bank.layer_ids, "route_layer": bank.route_layer,
               "router_train_acc": bank.router_train_acc},
              open(os.path.join(a.out, "bank.json"), "w"), indent=2)
    print(f"  train acc {bank.router_train_acc:.3f}, layers {bank.layer_ids}")

    if a.push:
        push_folder(a.out, a.push)


if __name__ == "__main__":
    main(["--base", "Qwen/Qwen3.5-4B", "--out", "behaviors",
          "--push", "pfekin/mobs-qwen3.5-4b"])
