#!/usr/bin/env python3
"""
=============================================================================
 benchmark.py — LARA: a base-preserving, residual-stream adapter
=============================================================================
THE METHOD — one frozen base, low-rank operators on the residual stream:
    at each bridge layer, read the layer's hidden state h, and add back
        h ← h + scaling · proj(LN(h))
    Only the operators (proj + LayerNorm) train; the base weights are never
    touched. Adaptation lives BESIDE the base transformation, not inside the
    weights — LoRA replaces W with W+BA; RC preserves W and adds a residual.
    Single forward pass; bridges off == the frozen base (used as a free
    DPO reference). Injection is recorded/scalable/maskable at inference for
    analysis (see rc_analysis.py: logit-lens, γ-steering, position-masking).

ARMS:  rc (the residual-stream adapter) · lora (weight-space baseline, matched params)
TASKS: "ft"  -> CE on python_600, PPL on in_dist / Dolly / wikitext
       "dpo" -> length-normalized + NLL-anchored DPO on ultrafeedback,
                reference = bridges off (free), preference-accuracy eval.

Tuning axes (config, not commented code):
    proj_type     linear | gelu | swiglu
    bridge_layers placement/density — vary this; it is the one knob with real effect
    rank, alpha   capacity (match LoRA's trainable count for a fair comparison)
=============================================================================
"""
import math, json, gc, random, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _HAVE_PEFT = True
except Exception as _e:
    _HAVE_PEFT = False
    print(f"  (peft not importable: {_e}; 'lora' arm skipped)")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — one dict drives everything
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "model":        "Qwen/Qwen2.5-1.5B-Instruct",
    "quantization": "8bit",
    "seed":         42,

    "task":         "ft",                       # "ft" | "dpo"
    "arms":         ["rc", "lora"],             # rc = residual-stream low-rank adapter; lora = weight-space baseline

    # --- bridge operators: low-rank read of the residual stream, added back (weights untouched) ---
    "bridge_layers": [4, 8, 12, 16, 20, 24],    # placement/density is a tuning axis — vary this
    #"bridge_layers": [14], 
    "proj_type":     "linear",                  # linear | gelu | swiglu
    "rank":          128,
    "alpha":         128.0,

    # --- LoRA baseline (match trainable count to RC; both are printed) ---
    "lora_rank":     16,
    "lora_target":   ["q_proj", "v_proj"],
    "lora_alpha":    None,                       # None => 2*rank
    "lora_dropout":  0.05,

    # ===== FINE-TUNING (task="ft") =====
    "ft": {
        "train_url":   "https://raw.githubusercontent.com/pfekin/LARA/refs/heads/main/research/data/python_600.json",
        "system_prompt": "You are an expert programmer. Provide precise code solutions and systems architectural reasoning.",
        "max_seq_len":   512,
        "indist_holdout":0.15,
        "eval_n":        50,
        "ood_sources": [
            {"name": "gen-instruct", "hf": "databricks/databricks-dolly-15k",
             "config": None, "split": "train", "kind": "instruct"},
            {"name": "raw-text", "hf": "Salesforce/wikitext",
             "config": "wikitext-103-raw-v1", "split": "train", "kind": "text", "field": "text"},
        ],
        "steps":      600,
        "lr":         2e-4,
        "grad_accum": 4, #16,
    },

    # ===== DPO (task="dpo") =====
    "dpo": {
        "hf_dataset":  "HuggingFaceH4/ultrafeedback_binarized",
        "train_split": "train_prefs",
        "eval_split":  "test_prefs",
        "train_n":     512, #1024,
        "eval_n":      512, #256, #512,
        "max_prompt_len": 128, #256,
        "max_resp_len":   128, #256,
        "system_prompt":  None,
        "length_norm": True,
        "nll_lambda":  0.1, #0.5,
        "beta":        5.0, #2.0,         # length-normalized (SimPO/CPO) regime — NOT 0.1-0.25
        "steps":       60, #300,
        "lr":          2e-4, #5e-5,
        "grad_accum":  16, #8,
        "base_gap_n":  32,          # diagnostic: base logprob gap (chosen−rejected) over N pairs; 0 to skip
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Model loading (inline; frozen base, no kbit-prep — only bridges train)
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_id, quantization=None, device=DEVICE):
    print(f"  ➔ Loading base model: {model_id} (quant={quantization})")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "right"
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": {"": device} if device == "cuda" else None, "torch_dtype": DTYPE}
    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE) if quantization == "4bit"
            else BitsAndBytesConfig(load_in_8bit=True))
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if quantization not in ("4bit", "8bit") and device != "cuda":
        model = model.to(device)
    model.requires_grad_(False)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
    model.eval()
    return model, tok


def _decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers            # Llama/Qwen
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h           # GPT-2
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    raise ValueError("Unknown architecture: can't locate decoder layers")


# ─────────────────────────────────────────────────────────────────────────────
#  Projections (zero-init up => identity at init).  proj: d -> d through rank r.
# ─────────────────────────────────────────────────────────────────────────────
class LinearProj(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up   = nn.Linear(r, d, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(self.down(x))

class GELUProj(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(self.act(self.down(x)))

class SwiGLUProj(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.down_gate  = nn.Linear(d, r, bias=False)
        self.down_value = nn.Linear(d, r, bias=False)
        self.act = nn.SiLU()
        self.up  = nn.Linear(r, d, bias=True)
        nn.init.orthogonal_(self.down_gate.weight)
        nn.init.orthogonal_(self.down_value.weight)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(self.act(self.down_gate(x)) * self.down_value(x))

def make_proj(kind, d, r):
    return {"linear": LinearProj, "gelu": GELUProj, "swiglu": SwiGLUProj}[kind](d, r)


# ─────────────────────────────────────────────────────────────────────────────
#  Bridge — low-rank read of the residual stream, added back. Weights untouched.
#  Injection = scaling · proj(LN(h)) · gamma. No clean reference, no gate (those
#  belonged to the read-clean variant, which is no longer part of the method).
# ─────────────────────────────────────────────────────────────────────────────
class Bridge(nn.Module):
    def __init__(self, d, r, alpha, proj_type):
        super().__init__()
        self.scaling = alpha / r
        self.norm = nn.LayerNorm(d)
        self.proj = make_proj(proj_type, d, r)
        # inference-time controls — inert during training (gamma=1, no mask, no record)
        self.gamma       = 1.0      # scale the injection at inference (γ-steering)
        self._pos_mask   = None     # optional [B,T,1] mask: inject only where nonzero
        self._record     = False    # capture the injection for logit-lens decoding
        self._last_inj   = None     # vector actually added to the stream
        self._last_delta = None     # same as _last_inj at gamma=1, mask=None

    def steer(self, h):
        """The vector added to the residual stream: scaling · proj(LN(h)) · gamma, optionally masked.
        Exposed separately so a merged model can sum injections from several adapters."""
        delta = self.proj(self.norm(h.float())).to(h.dtype) * self.scaling
        inj = delta * self.gamma
        if self._pos_mask is not None:
            inj = inj * self._pos_mask.to(inj.dtype)
        if self._record:
            self._last_delta = delta.detach()
            self._last_inj = inj.detach()
        return inj

    def forward(self, h):
        return h + self.steer(h)


class _SameTok:
    def __init__(self, vocab): self.vocab_S = vocab; self._same = True
    def adapt_text(self, ids): return ids


# ─────────────────────────────────────────────────────────────────────────────
#  SteeringModel — frozen base + hooked low-rank bridges (single forward pass).
# ─────────────────────────────────────────────────────────────────────────────
class SteeringModel(nn.Module):
    def __init__(self, base, tok, bridge_layers, *, proj_type, rank, alpha, device=DEVICE):
        super().__init__()
        self.base, self.tok, self.device = base, tok, device
        d = base.config.hidden_size
        self.layers = _decoder_layers(base)
        self.bridge_ids = [l for l in bridge_layers if l < len(self.layers)]
        self.bridges = nn.ModuleDict({
            str(l): Bridge(d, rank, alpha, proj_type).to(device) for l in self.bridge_ids})
        self._on = False
        self._handles = [self.layers[l].register_forward_hook(self._make_hook(l))
                         for l in self.bridge_ids]
        self.preferred_head = "anchor"
        self.vocab_A = self.vocab_S = base.config.vocab_size
        self.tok_bridge = _SameTok(self.vocab_S)

    def _make_hook(self, l):
        bridge = self.bridges[str(l)]
        def hook(module, inputs, output):
            if not self._on:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            new_hs = bridge(hs)                              # read this layer's output, add the residual
            return (new_hs,) + tuple(output[1:]) if isinstance(output, tuple) else new_hs
        return hook

    def trainable_params(self):
        return [p for p in self.bridges.parameters() if p.requires_grad]

    # --- inference-time analysis controls ---
    def set_gamma(self, g):
        for b in self.bridges.values(): b.gamma = g
    def set_record(self, on=True):
        for b in self.bridges.values(): b._record = on   # keeps last capture readable after on=False
    def set_pos_mask(self, mask):
        for b in self.bridges.values(): b._pos_mask = mask
    def injections(self):
        return {int(l): b._last_inj for l, b in self.bridges.items() if b._last_inj is not None}
    def deltas(self):
        return {int(l): b._last_delta for l, b in self.bridges.items() if b._last_delta is not None}

    def train(self, mode=True):
        self.bridges.train(mode); self.base.eval(); return self
    def eval(self):
        return self.train(False)

    def remove_hooks(self):
        for h in self._handles: h.remove()
        self._handles = []

    def forward(self, input_ids_A, input_ids_S=None, use_bridges=True, target_head=None, **kw):
        ids = input_ids_A.to(self.device)
        self._on = bool(use_bridges)                        # bridges off == frozen base (the free DPO reference)
        logits = self.base(input_ids=ids, use_cache=False).logits.float()
        self._on = False
        return logits, logits, logits


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA baseline — PEFT LoRA wrapped to the same forward contract.
# ─────────────────────────────────────────────────────────────────────────────
class LoRAEngine(nn.Module):
    def __init__(self, base, rank=16, device=DEVICE, kbit=True):
        super().__init__()
        if kbit:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        cfg = LoraConfig(r=rank, lora_alpha=(CFG["lora_alpha"] or 2 * rank),
                         target_modules=CFG["lora_target"], lora_dropout=CFG["lora_dropout"],
                         bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)
        self.device = device; self.preferred_head = "anchor"
        self.vocab_S = getattr(base.config, "vocab_size", None)
        self.tok_bridge = _SameTok(self.vocab_S)
        print(f"  🪛 LoRA r={rank} on {CFG['lora_target']} | trainable "
              f"{sum(p.numel() for p in self.trainable_params()):,}")

    def trainable_params(self): return [p for p in self.model.parameters() if p.requires_grad]
    def train(self, mode=True): self.model.train(mode); return self
    def eval(self): self.model.eval(); return self
    def restore_base(self): return self.model.unload()

    def forward(self, input_ids_A, input_ids_S=None, use_bridges=True, target_head=None, **kw):
        ids = input_ids_A.to(self.device)
        if use_bridges:
            return (self.model(input_ids=ids, use_cache=False).logits.float(),) * 3
        with self.model.disable_adapter():
            return (self.model(input_ids=ids, use_cache=False).logits.float(),) * 3


def build_arm(arm, base, tok):
    if arm == "lora":
        return LoRAEngine(base, rank=CFG["lora_rank"], device=DEVICE,
                          kbit=(CFG["quantization"] is not None))
    return SteeringModel(base, tok, CFG["bridge_layers"], proj_type=CFG["proj_type"],
                         rank=CFG["rank"], alpha=CFG["alpha"], device=DEVICE)


def n_trainable(eng): return sum(p.numel() for p in eng.trainable_params())


# ─────────────────────────────────────────────────────────────────────────────
#  Shared logprob core
# ─────────────────────────────────────────────────────────────────────────────
def _resp_logprob(logits, ids, prompt_len):
    """(sum_logprob, n_response_tokens) over RESPONSE positions only."""
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    resp = tok_lp[:, max(prompt_len - 1, 0):]
    return resp.sum(dim=-1).squeeze(0), max(resp.shape[1], 1)

def _reduced(logits, ids, prompt_len, length_norm):
    s, n = _resp_logprob(logits, ids, prompt_len)
    return s / n if length_norm else s

def _ce_sum(logits, ids):
    T = min(logits.shape[1], ids.shape[1]) - 1
    if T <= 0: return 0.0, 0
    lp  = logits[:, :T, :].reshape(-1, logits.shape[-1])
    tgt = ids[:, 1:T + 1].reshape(-1)
    return F.cross_entropy(lp, tgt, reduction="sum").item(), tgt.numel()


# ═════════════════════════════════════════════════════════════════════════════
#  FINE-TUNING  (CE on python_600; PPL on in_dist / Dolly-instruct / wikitext)
# ═════════════════════════════════════════════════════════════════════════════
def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def fmt_chatml(system, user, assistant=""):
    s = f"<|im_start|>system\n{system.strip()}<|im_end|>\n" if system else ""
    s += f"<|im_start|>user\n{user.strip()[:300]}<|im_end|>\n<|im_start|>assistant\n"
    if assistant: s += f"{assistant.strip()[:600]}<|im_end|>"
    return s

def _format_corpus(rows, system, seed):
    texts = []
    for it in rows:
        if isinstance(it, dict):
            instr = it.get("instruction", it.get("query", it.get("prompt", "")))
            out   = it.get("output", it.get("answer", it.get("response", "")))
            if instr and out: texts.append(fmt_chatml(system, instr, out))
        elif isinstance(it, str):
            texts.append(it)
    random.seed(seed); random.shuffle(texts)
    return texts

def _to_ids(tok, texts, max_len, cap):
    out = []
    for t in texts[:cap]:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
        if ids.shape[1] >= 2: out.append(ids)
    return out

def _text_windows(tok, rows, field, n_windows, win):
    buf, budget = [], win * n_windows * 20
    for r in rows:
        t = (r.get(field, "") if isinstance(r, dict) else str(r)) or ""
        if t.strip(): buf.append(t.strip())
        if sum(len(x) for x in buf) > budget: break
    if not buf: return []
    ids = tok("\n\n".join(buf), return_tensors="pt").input_ids[0]
    out = []
    for i in range(0, ids.shape[0] - 1, win):
        chunk = ids[i:i + win]
        if chunk.shape[0] >= 2: out.append(chunk.unsqueeze(0).to(DEVICE))
        if len(out) >= n_windows: break
    return out

def _stream_rows(hf, config, split, n):
    from datasets import load_dataset
    ds = (load_dataset(hf, config, split=split, streaming=True) if config
          else load_dataset(hf, split=split, streaming=True))
    rows = []
    for r in ds:
        rows.append(r)
        if len(rows) >= n: break
    return rows

def load_ft_data(tok):
    ft = CFG["ft"]; sysp = ft["system_prompt"]; seed = CFG["seed"]
    raw = _fetch_json(ft["train_url"])
    random.seed(seed); random.shuffle(raw)
    n_hold = max(8, int(ft["indist_holdout"] * len(raw)))
    hold, train = raw[:n_hold], raw[n_hold:]
    train_texts = _format_corpus(train, sysp, seed)
    splits = [("in_dist", _to_ids(tok, _format_corpus(hold, sysp, seed), ft["max_seq_len"], ft["eval_n"]))]
    for src in ft["ood_sources"]:
        try:
            need = ft["eval_n"] * (40 if src["kind"] == "text" else 2)
            rows = _stream_rows(src["hf"], src.get("config"), src["split"], need)
            if src["kind"] == "text":
                ids = _text_windows(tok, rows, src.get("field", "text"), ft["eval_n"], ft["max_seq_len"])
            else:
                ids = _to_ids(tok, _format_corpus(rows, sysp, seed), ft["max_seq_len"], ft["eval_n"])
            if ids:
                splits.append((src["name"], ids)); print(f"   ✓ split '{src['name']}': {len(ids)} ex")
        except Exception as e:
            print(f"    split '{src['name']}' unavailable ({type(e).__name__}); skipping")
    return train_texts, splits

def _text_stream(tok, texts, max_len):
    while True:
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
            if ids.shape[1] >= 2: yield ids

def _to_ids_all(tok, texts, max_len):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
        if ids.shape[1] >= 2: out.append(ids)
    return out

def train_ce(model, enc_train, steps, lr, grad_accum):
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    idx = list(range(len(enc_train))); random.shuffle(idx); ptr = 0
    print(f"   CE: {steps} steps · lr={lr} · accum={grad_accum} · {n_trainable(model):,} params")
    for step in range(steps):
        opt.zero_grad(); running = 0.0
        for _ in range(grad_accum):
            if ptr >= len(idx): random.shuffle(idx); ptr = 0
            i = idx[ptr]; ptr += 1
            ids = enc_train[i]
            logits = model(input_ids_A=ids, use_bridges=True, target_head="anchor")[0]
            T = min(logits.shape[1], ids.shape[1]) - 1
            loss = F.cross_entropy(logits[:, :T].reshape(-1, logits.shape[-1]),
                                   ids[:, 1:T + 1].reshape(-1)) / grad_accum
            loss.backward(); running += loss.item() * grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"     step {step:>4}/{steps}  ce={running/grad_accum:.4f}")

@torch.no_grad()
def eval_ppl(model, eval_ids, use_bridges):
    ce, n = 0.0, 0
    for ids in eval_ids:
        sel = model(input_ids_A=ids, use_bridges=use_bridges, target_head="anchor")[0]
        c, k = _ce_sum(sel, ids); ce += c; n += k
    return math.exp(ce / n) if n else float("nan")

def run_ft():
    set_seed(CFG["seed"])
    base, tok = load_model(CFG["model"], CFG["quantization"])
    print("\n Loading fine-tuning data (python_600 + OOD)...")
    train_texts, splits = load_ft_data(tok)
    enc_train = _to_ids_all(tok, train_texts, CFG["ft"]["max_seq_len"])
    print(f"   train sequences: {len(enc_train)}")

    print("\n FROZEN BASELINES — PPL (floor):")
    base_ppl = {}
    probe = build_arm("rc", base, tok)                  # any arm gives the adapter-off base
    for name, ids in splits:
        base_ppl[name] = eval_ppl(probe, ids, use_bridges=False)
        print(f"   {name:<14} | {base_ppl[name]:.2f}")
    probe.remove_hooks(); del probe; gc.collect(); torch.cuda.empty_cache()

    arms = [a for a in CFG["arms"] if a != "lora" or _HAVE_PEFT]
    arms = [a for a in arms if a != "lora"] + (["lora"] if "lora" in arms else [])
    rows = []
    for arm in arms:
        print(f"\n╭─ arm: {arm} " + "─" * 56)
        set_seed(CFG["seed"])
        ft = CFG["ft"]
        if arm == "lora":
            for L in _decoder_layers(base): L._forward_hooks.clear(); L._forward_pre_hooks.clear()
        eng = build_arm(arm, base, tok)
        print(f"   trainable params: {n_trainable(eng):,}")
        train_ce(eng, enc_train, ft["steps"], ft["lr"], ft["grad_accum"])
        res = {name: eval_ppl(eng, ids, use_bridges=True) for name, ids in splits}
        rows.append((arm, n_trainable(eng), res))
        if hasattr(eng, "remove_hooks"): eng.remove_hooks()
        if arm == "lora": base = eng.model.unload()
        del eng; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═" * 78)
    print(f" RC FINE-TUNING — PPL, lower=better   (model={CFG['model'].split('/')[-1]}, "
          f"proj={CFG['proj_type']}, bridges@{CFG['bridge_layers']})")
    print("═" * 78)
    names = [n for n, _ in splits]
    header = f"   {'arm':<10}| {'params':>10} | " + " | ".join(f"{n:>13}" for n in names)
    print(header); print("   " + "-" * (len(header) - 3))
    print(f"   {'(base)':<10}| {'—':>10} | " + " | ".join(f"{base_ppl[n]:>13.2f}" for n in names))
    for arm, p, res in rows:
        cells = " | ".join(f"{res[n]:>10.2f} {'OK' if res[n] < base_ppl[n] else 'X'}" for n in names)
        print(f"   {arm:<10}| {p:>10,} | {cells}")
    print("═" * 78)
    print("   READ: in_dist/instruct gains = format adaptation; raw-text ≈ base = non-destructive.")
    print("   in_dist/instruct gains vs flat raw-text = format adaptation. rc vs lora at MATCHED params.")


# ═════════════════════════════════════════════════════════════════════════════
#  DPO  (ultrafeedback; length-normalized + NLL-anchored; free toggled reference)
# ═════════════════════════════════════════════════════════════════════════════
def load_pairs(split, n):
    from datasets import load_dataset
    ds = load_dataset(CFG["dpo"]["hf_dataset"], split=split, streaming=True)
    out = []
    for r in ds:
        prompt, ch, rj = r.get("prompt"), r.get("chosen"), r.get("rejected")
        cw = ch[-1]["content"] if isinstance(ch, list) and ch else (ch if isinstance(ch, str) else None)
        cl = rj[-1]["content"] if isinstance(rj, list) and rj else (rj if isinstance(rj, str) else None)
        if not prompt or not cw or not cl or cw.strip() == cl.strip(): continue
        out.append((prompt, cw, cl))
        if len(out) >= n: break
    return out

def _chat_prompt(tok, prompt):
    sysp = CFG["dpo"]["system_prompt"]
    msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        s = (sysp + "\n") if sysp else ""
        return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def _encode(tok, prompt, response):
    d = CFG["dpo"]
    p_ids = tok(_chat_prompt(tok, prompt), add_special_tokens=False).input_ids[:d["max_prompt_len"]]
    r_ids = tok(response, add_special_tokens=False).input_ids[:d["max_resp_len"]]
    if tok.eos_token_id is not None: r_ids = r_ids + [tok.eos_token_id]
    return torch.tensor([p_ids + r_ids], device=DEVICE), len(p_ids)

def encode_pairs(tok, pairs):
    enc = []
    for prompt, cw, cl in pairs:
        fw, pw = _encode(tok, prompt, cw); fl, pl = _encode(tok, prompt, cl)
        if fw.shape[1] - pw < 1 or fl.shape[1] - pl < 1: continue
        enc.append(dict(full_w=fw, plen_w=pw, full_l=fl, plen_l=pl))
    return enc

@torch.no_grad()
def reference_logprobs(model, enc, length_norm):
    model.eval()
    rw = torch.empty(len(enc), device=DEVICE); rl = torch.empty(len(enc), device=DEVICE)
    for i, e in enumerate(enc):
        rw[i] = _reduced(model(input_ids_A=e["full_w"], use_bridges=False)[0], e["full_w"], e["plen_w"], length_norm)
        rl[i] = _reduced(model(input_ids_A=e["full_l"], use_bridges=False)[0], e["full_l"], e["plen_l"], length_norm)
    return rw, rl

def train_dpo(model, enc, ref_w, ref_l, beta, steps, lr, grad_accum, nll_lambda, length_norm):
    if length_norm and beta < 1.0:
        print(f"   β={beta} low for length-normalized DPO (SimPO/CPO use β≈2)")
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=lr); model.train()
    idx = list(range(len(enc))); random.shuffle(idx); ptr = 0
    print(f"   DPO{'(len-norm)' if length_norm else '(sum)'}: {steps} steps · β={beta} · "
          f"λ_nll={nll_lambda} · lr={lr} · accum={grad_accum} · {len(enc)} pairs · {n_trainable(model):,} params")
    for step in range(steps):
        opt.zero_grad(); rd = rn = 0.0; acc = 0
        for _ in range(grad_accum):
            if ptr >= len(idx): random.shuffle(idx); ptr = 0
            i = idx[ptr]; ptr += 1; e = enc[i]
            lp_w = _reduced(model(input_ids_A=e["full_w"], use_bridges=True)[0], e["full_w"], e["plen_w"], length_norm)
            lp_l = _reduced(model(input_ids_A=e["full_l"], use_bridges=True)[0], e["full_l"], e["plen_l"], length_norm)
            dpo_term = -F.logsigmoid(beta * ((lp_w - lp_l) - (ref_w[i] - ref_l[i])))
            nll_term = -lp_w
            ((dpo_term + nll_lambda * nll_term) / grad_accum).backward()
            rd += dpo_term.item(); rn += nll_term.item()
            acc += int((lp_w - lp_l) > (ref_w[i] - ref_l[i]))
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"     step {step:>4}/{steps}  dpo={rd/grad_accum:.4f}  nll={rn/grad_accum:.3f}  "
                  f"batch_acc={acc/grad_accum:.2f}")

@torch.no_grad()
def eval_dpo(model, enc, ref_w, ref_l, beta, length_norm):
    model.eval(); n = len(enc)
    racc = marg = dch = drj = dpo = bacc = 0.0
    for i, e in enumerate(enc):
        pw = _reduced(model(input_ids_A=e["full_w"], use_bridges=True)[0], e["full_w"], e["plen_w"], length_norm)
        pl = _reduced(model(input_ids_A=e["full_l"], use_bridges=True)[0], e["full_l"], e["plen_l"], length_norm)
        rw_, rl_ = beta * (pw - ref_w[i]), beta * (pl - ref_l[i])
        racc += float(rw_ > rl_); marg += float(rw_ - rl_)
        dch += float(pw - ref_w[i]); drj += float(pl - ref_l[i])
        dpo += float(-F.logsigmoid(beta * ((pw - pl) - (ref_w[i] - ref_l[i]))))
        bacc += float(ref_w[i] > ref_l[i])
    return dict(reward_acc=racc/n, reward_margin=marg/n, d_chosen=dch/n,
                d_rejected=drj/n, dpo_loss=dpo/n, base_pref_acc=bacc/n)

@torch.no_grad()
def base_preference_gap(base, enc, n=32):
    """Does the FROZEN BASE already assign higher logprob to chosen than rejected?
    If yes, DPO has a gradient to ride and a flat loss is a TUNING problem (β↑, λ↓, lr).
    If chosen≈rejected (frac>0 ≈ 0.5), the preference is below what this model can resolve
    and no hyperparameter will open a margin — change the model or the dataset."""
    base.eval()
    sg, mg = [], []
    for e in enc[:n]:
        lw, nw = _resp_logprob(base(input_ids=e["full_w"].to(DEVICE), use_cache=False).logits.float(),
                               e["full_w"], e["plen_w"])
        ll, nl = _resp_logprob(base(input_ids=e["full_l"].to(DEVICE), use_cache=False).logits.float(),
                               e["full_l"], e["plen_l"])
        sg.append((lw - ll).item())                      # summed gap
        mg.append((lw / nw - ll / nl).item())            # per-token gap (the reduction the loss uses)
    k = len(mg)
    mean_m = sum(mg) / k
    std_m = (sum((x - mean_m) ** 2 for x in mg) / k) ** 0.5
    fpos_m = sum(x > 0 for x in mg) / k
    fpos_s = sum(x > 0 for x in sg) / k
    print(f"\n🔬 BASE PREFERENCE GAP — chosen − rejected, frozen base, {k} pairs:")
    print(f"   per-token (loss reduction): mean={mean_m:+.4f}  std={std_m:.4f}  "
          f"min={min(mg):+.3f}  max={max(mg):+.3f}  frac>0={fpos_m:.2f}")
    print(f"   summed:                     mean={sum(sg)/k:+.2f}  frac>0={fpos_s:.2f}")
    if fpos_m >= 0.62 and mean_m > 0:
        print(f"   → SIGNAL PRESENT (frac>0={fpos_m:.2f}). The base separates the pairs; a flat loss is")
        print(f"     a TUNING problem — raise β (preference gain), lower nll_lambda (anchor), tune lr.")
    elif fpos_m <= 0.57:
        print(f"   → SIGNAL ~ABSENT (frac>0={fpos_m:.2f} ≈ chance). The base barely separates chosen from")
        print(f"     rejected — ultrafeedback's fine-grained preferences are likely beyond a "
              f"{CFG['model'].split('/')[-1]}.")
        print(f"     No β/lr opens a margin that isn't in the data. Move to a larger base (3B) or a")
        print(f"     blunter dataset (e.g. Anthropic HH helpful/harmless) before more tuning.")
    else:
        print(f"   → BORDERLINE (frac>0={fpos_m:.2f}). Marginal signal; tuning may help but the ceiling is low.")
    print()


def run_dpo():
    d = CFG["dpo"]; set_seed(CFG["seed"])
    base, tok = load_model(CFG["model"], CFG["quantization"])
    print("\n Loading ultrafeedback pairs...")
    enc_tr = encode_pairs(tok, load_pairs(d["train_split"], d["train_n"]))
    enc_ev = encode_pairs(tok, load_pairs(d["eval_split"], d["eval_n"]))
    print(f"   train pairs: {len(enc_tr)} | eval pairs: {len(enc_ev)}")

    if d.get("base_gap_n", 0):
        base_preference_gap(base, enc_ev, n=d["base_gap_n"])

    arms = [a for a in CFG["arms"] if a != "lora" or _HAVE_PEFT]
    arms = [a for a in arms if a != "lora"] + (["lora"] if "lora" in arms else [])
    rows = []
    for arm in arms:
        print(f"\n╭─ arm: {arm} " + "─" * 56)
        set_seed(CFG["seed"])
        if arm == "lora":
            for L in _decoder_layers(base): L._forward_hooks.clear(); L._forward_pre_hooks.clear()
        eng = build_arm(arm, base, tok)
        print(f"   trainable params: {n_trainable(eng):,}")
        ref_w, ref_l = reference_logprobs(eng, enc_tr, d["length_norm"])
        ev_w, ev_l = reference_logprobs(eng, enc_ev, d["length_norm"])
        train_dpo(eng, enc_tr, ref_w, ref_l, d["beta"], d["steps"], d["lr"],
                  d["grad_accum"], d["nll_lambda"], d["length_norm"])
        rows.append((arm, n_trainable(eng), eval_dpo(eng, enc_ev, ev_w, ev_l, d["beta"], d["length_norm"])))
        if hasattr(eng, "remove_hooks"): eng.remove_hooks()
        if arm == "lora": base = eng.model.unload()
        del eng; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═" * 100)
    print(f" RC DPO — RC operators vs weight-LoRA   (β={d['beta']}, λ_nll={d['nll_lambda']}, "
          f"len_norm={d['length_norm']}, {d['hf_dataset'].split('/')[-1]})")
    print("═" * 100)
    print(f"   {'arm':<10}| {'params':>10} | {'reward_acc':>10} | {'margin':>8} | "
          f"{'Δlp_chos':>9} | {'Δlp_rej':>9} | {'dpo_loss':>8} | {'base_acc':>8}")
    print("   " + "-" * 97)
    for arm, p, m in rows:
        print(f"   {arm:<10}| {p:>10,} | {m['reward_acc']:>10.3f} | {m['reward_margin']:>+8.3f} | "
              f"{m['d_chosen']:>+9.3f} | {m['d_rejected']:>+9.3f} | {m['dpo_loss']:>8.4f} | {m['base_pref_acc']:>8.3f}")
    print("═" * 100)
    print("   READ: reward_acc>base_acc AND Δlp_chosen≥0 = real preference learning (not collapse).")
    print("   rc vs lora at MATCHED params; weights stay frozen (base-preserving).")


def run():
    print("*" * 60)
    print(f"* RC harness — task={CFG['task']} · arms={CFG['arms']} · {CFG['model'].split('/')[-1]}")
    print("*" * 60)
    (run_dpo if CFG["task"] == "dpo" else run_ft)()


# ═════════════════════════════════════════════════════════════════════════════
#  SELF-TEST — mechanism checks for the residual-stream adapter (tiny model, CPU)
# ═════════════════════════════════════════════════════════════════════════════
def selftest():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    print("=== rc.py selftest (tiny Qwen2, CPU) ===")
    set_seed(0); dev = "cpu"; V = 64
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True)
    base = Qwen2ForCausalLM(cfg).to(dev).eval(); base.requires_grad_(False)

    class T: vocab_size = V; pad_token = "<p>"; eos_token = "<e>"; eos_token_id = V - 1; padding_side = "r"
    blayers = [1, 2, 3]
    ids = torch.randint(0, V, (1, 12), device=dev)

    # ---- bridges ON change the output; OFF == frozen base ----
    set_seed(1)
    m = SteeringModel(base, T(), blayers, proj_type="linear", rank=8, alpha=8.0, device=dev)
    for b in m.bridges.values():                            # perturb so injections are non-zero
        nn.init.normal_(b.proj.up.weight, std=0.05); nn.init.normal_(b.proj.up.bias, std=0.05)
    raw = base(input_ids=ids, use_cache=False).logits.float()
    on  = m(input_ids_A=ids, use_bridges=True)[0]
    off = m(input_ids_A=ids, use_bridges=False)[0]
    assert torch.allclose(off, raw, atol=1e-5), "bridges off must equal the frozen base"
    assert not torch.allclose(on, raw, atol=1e-4), "bridges on must change the output"
    print("[1] bridges off == frozen base (free DPO reference); bridges on change the output")

    # ---- one injection recorded per bridge (read residual, add low-rank delta) ----
    m.set_record(True); m(input_ids_A=ids, use_bridges=True); m.set_record(False)
    injs = m.injections()
    assert set(injs) == set(blayers) and all(injs[l].shape[-1] == 64 for l in blayers)
    print(f"[2] one injection recorded per bridge {sorted(injs)} (weights untouched, base preserved)")

    # ---- gamma scales the injection: γ=0 == base, γ=1 ≠ γ=2 ----
    m.set_gamma(0.0); g0 = m(input_ids_A=ids, use_bridges=True)[0]
    assert torch.allclose(g0, raw, atol=1e-5), "γ=0 must equal base"
    m.set_gamma(1.0); g1 = m(input_ids_A=ids, use_bridges=True)[0]
    m.set_gamma(2.0); g2 = m(input_ids_A=ids, use_bridges=True)[0]; m.set_gamma(1.0)
    assert not torch.allclose(g1, g2, atol=1e-4)
    print("[3] γ-steering: γ=0 == base (bit-identical), γ=1 ≠ γ=2 (injection scales)")

    # ---- position-mask: response-only injection leaves prompt logits == base ----
    plen = 5; mask = torch.zeros(1, ids.shape[1], 1, device=dev); mask[:, plen:, :] = 1
    m.set_pos_mask(mask); masked = m(input_ids_A=ids, use_bridges=True)[0]; m.set_pos_mask(None)
    assert torch.allclose(masked[:, :plen], raw[:, :plen], atol=1e-5), "prompt positions must be untouched"
    assert not torch.allclose(masked[:, plen:], raw[:, plen:], atol=1e-4), "response positions must be steered"
    print("[4] position-masking: response-only injection leaves prompt logits == base")

    # ---- projection menu all runs finite ----
    for pt in ["linear", "gelu", "swiglu"]:
        mm = SteeringModel(base, T(), blayers, proj_type=pt, rank=8, alpha=8.0, device=dev)
        o = mm(input_ids_A=ids, use_bridges=True)[0]
        assert torch.isfinite(o).all(), f"{pt} produced non-finite logits"
        mm.remove_hooks()
    print("[5] linear / gelu / swiglu projections all run and produce finite logits")

    # ---- gradient flows into bridges (CE) ----
    g = SteeringModel(base, T(), blayers, proj_type="linear", rank=8, alpha=8.0, device=dev)
    logits = g(input_ids_A=ids, use_bridges=True)[0]
    F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.trainable_params()), "no CE gradient to bridges"
    g.remove_hooks()
    print("[6] CE loss sends gradient into the bridges (through the frozen base)")

    # ---- DPO trains; NLL anchor keeps Δlp_chosen ≥ 0 ----
    set_seed(2)
    dm = SteeringModel(base, T(), blayers, proj_type="linear", rank=8, alpha=8.0, device=dev)
    enc = []; gen = torch.Generator().manual_seed(7)
    for _ in range(16):
        pl = int(torch.randint(4, 7, (1,), generator=gen).item())
        pr = torch.randint(10, V - 2, (pl,), generator=gen).tolist()
        enc.append(dict(full_w=torch.tensor([pr + [5]*4 + [V-1]], device=dev), plen_w=pl,
                        full_l=torch.tensor([pr + [7]*4 + [V-1]], device=dev), plen_l=pl))
    rw, rl = reference_logprobs(dm, enc, True)
    train_dpo(dm, enc, rw, rl, beta=2.0, steps=60, lr=1e-2, grad_accum=4, nll_lambda=0.5, length_norm=True)
    res = eval_dpo(dm, enc, rw, rl, beta=2.0, length_norm=True)
    assert res["d_chosen"] >= -1e-3, f"NLL anchor failed: Δlp_chosen={res['d_chosen']}"
    assert res["reward_acc"] >= res["base_pref_acc"], "DPO did not improve preference"
    print(f"[7] DPO trains bridges; anchor holds Δlp_chosen={res['d_chosen']:+.3f}≥0, "
          f"reward_acc {res['reward_acc']:.2f}≥base {res['base_pref_acc']:.2f}")

    print("\nALL SELFTESTS PASSED — base-preserving residual-stream adapter, weights untouched.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        run()

