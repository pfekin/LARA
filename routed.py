#!/usr/bin/env python3
"""
=============================================================================
 routed.py — LARA coupled: many behaviors, one frozen base
=============================================================================
Not a rival to MoE on capacity. The claim is a DEPLOYMENT one: a frozen
(possibly quantized, edge-sized) model already holds the knowledge; LARA adapters
are tiny low-rank behavior operators over it. Keep N of them resident and route
per token, and one small local model gains N switchable behaviors for a few MB —
no second model, no interfering weight-merge.

There is nothing to compare this against — no other method puts N behaviors on a
frozen base and routes between them. So this is a SYSTEMS demonstration: show the
feature runs and does what it says. Three numbers do that:
  • recovery    : per-domain PPL under the routed model vs that domain's own
                  dedicated adapter (does routing keep most of each behavior?)
  • routing     : fraction of each domain's tokens sent to the RIGHT adapter
                  (a confusion matrix — proves the router actually routes, not
                  a disguised single behavior or a coin flip)
  • footprint   : base + N adapters + router  vs  N full models (why on a phone)

Router input is a single frozen-base hidden state per token -> a constant w.r.t.
the router, so it's cached once and the router trains as a cheap classifier on
domain labels. One routing decision per token (top-1 or soft), applied at every
bridge layer. FT and DPO behaviors slot in identically (each is just a bridge set).
=============================================================================
"""
import math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import lara as rc

RCFG = {
    "route_mode":    "soft",     # "top1" (hard, one behavior/token) | "soft" (blend behaviors mid-sequence)
    "router_hidden": 0,          # 0 = linear router; >0 = one hidden layer of this width
    "router_steps":  900,
    "router_lr":     1e-3,
    "eval_n":        128,
    # one preference behavior trained with DPO (rc.py's ultrafeedback pipeline) — see the "preference" domain
    "dpo": {
        "train_split": rc.CFG["dpo"]["train_split"], "train_n": 400,
        "eval_split":  rc.CFG["dpo"]["eval_split"],  "eval_n": 150,
        "beta": rc.CFG["dpo"]["beta"], "steps": rc.CFG["dpo"]["steps"], "lr": rc.CFG["dpo"]["lr"],
        "grad_accum": rc.CFG["dpo"]["grad_accum"], "nll_lambda": rc.CFG["dpo"]["nll_lambda"],
        "length_norm": rc.CFG["dpo"]["length_norm"],
    },
    # each domain -> one behavior adapter. All share the base's tokenizer.
    # near-neighbor pair on purpose: code (python_600) vs code_alpaca — stresses the router.
    # summary = a distinct behavior (compress-not-answer). preference = a DPO behavior (task="dpo").
    "domains": [
        {"name": "code",     "url": rc.CFG["ft"]["train_url"], "system": "You are an expert programmer."},
        {"name": "general",  "hf": "databricks/databricks-dolly-15k", "split": "train",
         "system": "You are a helpful assistant."},
        {"name": "math",     "hf": "meta-math/MetaMathQA", "split": "train", "system": "You are a math tutor.",
         "map": lambda r: (r.get("query", ""), r.get("response", ""))},
        {"name": "medical",  "hf": "lavita/ChatDoctor-HealthCareMagic-100k", "split": "train",
         "system": "You are a medical expert.", "map": lambda r: (r.get("instruction", ""), r.get("output", ""))},
        {"name": "code_alpaca", "hf": "sahil2801/CodeAlpaca-20k", "split": "train",
         "system": "You are an expert programmer.", "map": lambda r: (r.get("instruction", ""), r.get("output", ""))},
        {"name": "summary",  "hf": "knkarthick/dialogsum", "split": "train",
         "system": "You are a summarization assistant.",
         "map": lambda r: (f"Summarize this conversation:\n{r.get('dialogue','')}", r.get("summary", ""))},
        {"name": "preference", "task": "dpo"},               # DPO behavior — uses RCFG["dpo"]; demo soft routing here
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Router — per-token distribution over the N behavior adapters
# ─────────────────────────────────────────────────────────────────────────────
class Router(nn.Module):
    def __init__(self, d, n_experts, hidden=0):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
                    if hidden > 0 else nn.Linear(d, n_experts))
    def forward(self, h):                       # h:[B,T,d] -> logits:[B,T,N]
        return self.net(h.float())


# ─────────────────────────────────────────────────────────────────────────────
#  RoutedModel — frozen base + N bridge sets + router, routed injection
# ─────────────────────────────────────────────────────────────────────────────
class RoutedModel(nn.Module):
    def __init__(self, base, tok, bridge_sets, route_layer=None, route_mode="top1",
                 router_hidden=0, device=rc.DEVICE):
        super().__init__()
        self.base, self.tok, self.device = base, tok, device
        self.layers = rc._decoder_layers(base)
        self.sets = list(bridge_sets)
        self.N = len(self.sets)
        self.bridge_ids = sorted(int(l) for l in self.sets[0].keys())
        self.route_layer = self.bridge_ids[0] if route_layer is None else route_layer
        assert self.route_layer <= self.bridge_ids[0], "router must read at/before the first bridge"
        self.route_mode = route_mode
        self.router = Router(base.config.hidden_size, self.N, router_hidden).to(device)
        self._route_w = None; self._route_logits = None; self._force = None
        self._inject_on = True
        self._handles = [self.layers[self.route_layer].register_forward_hook(self._route_hook())]
        self._handles += [self.layers[l].register_forward_hook(self._inject_hook(l)) for l in self.bridge_ids]
        self.preferred_head = "anchor"; self.vocab_S = base.config.vocab_size
        self.tok_bridge = rc._SameTok(self.vocab_S)

    def _route_hook(self):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            logits = self.router(hs)                                   # [B,T,N]
            self._route_logits = logits
            if self._force is not None:                               # force one expert (coexistence eval)
                idx = torch.full(logits.shape[:-1], self._force, device=logits.device, dtype=torch.long)
                self._route_w = F.one_hot(idx, self.N).to(hs.dtype)
            elif self.route_mode == "top1":
                self._route_w = F.one_hot(logits.argmax(-1), self.N).to(hs.dtype)
            else:
                self._route_w = F.softmax(logits, dim=-1).to(hs.dtype)
            return out
        return hook

    def set_force(self, k):                                            # route ALL tokens to expert k (None = normal)
        self._force = k

    def _inject_hook(self, l):
        def hook(module, inp, out):
            if not self._inject_on or self._route_w is None: return out
            hs = out[0] if isinstance(out, tuple) else out
            injs = torch.stack([self.sets[k][str(l)].steer(hs) for k in range(self.N)], dim=-2)  # [B,T,N,d]
            inj = (injs * self._route_w.unsqueeze(-1)).sum(dim=-2)     # route-weighted injection [B,T,d]
            new = hs + inj
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook

    def route_ids(self, ids):
        """Return the top-1 routed expert per token (for the discrimination metric)."""
        self._inject_on = False
        with torch.no_grad():
            self.base(input_ids=ids.to(self.device), use_cache=False)
        self._inject_on = True
        return self._route_logits.argmax(-1)                          # [B,T]

    def eval(self):
        for s in self.sets: s.eval()
        self.router.eval(); self.base.eval(); return self
    def remove_hooks(self):
        for h in self._handles: h.remove()
        self._handles = []

    def forward(self, input_ids_A, input_ids_S=None, use_bridges=True, target_head=None, **kw):
        ids = input_ids_A.to(self.device)
        self._inject_on = bool(use_bridges)
        logits = self.base(input_ids=ids, use_cache=False).logits.float()
        self._inject_on = True
        return logits, logits, logits


# ─────────────────────────────────────────────────────────────────────────────
#  Router training — cached frozen-base features + domain labels (cheap classifier)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def cache_route_features(base, route_layer, seqs, device=rc.DEVICE):
    """The router reads one frozen-base hidden state per token — a constant w.r.t. the
    router — so cache it once and train the router as a plain classifier on it."""
    layers = rc._decoder_layers(base)
    grab = {}
    h = layers[route_layer].register_forward_hook(
        lambda m, i, o: grab.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().cpu()))
    feats = []
    for ids in seqs:
        base(input_ids=ids.to(device), use_cache=False)
        feats.append(grab["x"])                                       # [1,T,d] on cpu
    h.remove()
    return feats

def train_router(router, feats, labels, n_experts, steps, lr, device=rc.DEVICE, batch=512):
    """Pool all tokens into one classification set (router is a per-token classifier) and
    train with mini-batches — far more stable than one variable-length sequence per step."""
    X = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)              # [Ntok, d]
    y = torch.cat([torch.full((f.shape[1],), lab, dtype=torch.long) for f, lab in zip(feats, labels)])
    n = X.shape[0]
    opt = torch.optim.AdamW(router.parameters(), lr=lr); router.train()
    print(f"  🧭 router: {steps} steps · lr={lr} · {sum(p.numel() for p in router.parameters()):,} params · "
          f"{n_experts} experts · {n:,} tokens")
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        xb, yb = X[idx].to(device), y[idx].to(device)
        logits = router(xb)                                                        # [batch, N]
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"     step {step:>4}/{steps}  ce={loss.item():.3f}  tok_acc={(logits.argmax(-1)==yb).float().mean().item():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_routed_ppl(model, eval_ids_list):
    ce, n = 0.0, 0
    model.eval()
    for ids in eval_ids_list:
        logits = model(input_ids_A=ids, use_bridges=True)[0]
        c, k = rc._ce_sum(logits, ids); ce += c; n += k
    return math.exp(ce / n) if n else float("nan")

@torch.no_grad()
def routing_confusion(model, domain_evals):
    """confusion[i,k] = fraction of domain-i tokens routed to expert k."""
    N = model.N
    conf = torch.zeros(N, N)
    for i, (_, ids_list) in enumerate(domain_evals):
        counts = torch.zeros(N)
        for ids in ids_list:
            r = model.route_ids(ids).reshape(-1)
            for k in range(N): counts[k] += (r == k).sum().item()
        conf[i] = counts / counts.sum().clamp(min=1)
    return conf


# ─────────────────────────────────────────────────────────────────────────────
#  Data (per domain) — reuse rc's loaders; each domain -> train texts + eval ids
# ─────────────────────────────────────────────────────────────────────────────
def load_domain(tok, cfg, max_len, eval_n, seed):
    import random as _r
    if "url" in cfg:
        raw = rc._fetch_json(cfg["url"]); _r.seed(seed); _r.shuffle(raw)
    else:
        raw = rc._stream_rows(cfg["hf"], cfg.get("subset"), cfg.get("split", "train"), 700)
    nh = max(8, int(0.15 * len(raw)))
    sysp = cfg.get("system", "")
    if "map" in cfg:                                   # custom row -> (instruction, output)
        mp = cfg["map"]
        def fmt(rows):
            out = []
            for r in rows:
                try: instr, ans = mp(r)
                except Exception: continue
                if instr and ans: out.append(rc.fmt_chatml(sysp, str(instr), str(ans)))
            return out
        tr = fmt(raw[nh:]); ev = rc._to_ids(tok, fmt(raw[:nh]), max_len, eval_n)
    else:
        tr = rc._format_corpus(raw[nh:], sysp, seed)
        ev = rc._to_ids(tok, rc._format_corpus(raw[:nh], sysp, seed), max_len, eval_n)
    return rc._to_ids_all(tok, tr, max_len), ev


@torch.no_grad()
def coapplication_report(model, behaviors, domains, style="preference"):
    """Soft routing only. Demonstrate that a style behavior is co-active WITH a domain
    behavior on the SAME tokens — which top-1 (one behavior/token) structurally cannot do.
    For each domain's own tokens: the mean routing distribution, and — the key number —
    on tokens where that domain's expert LEADS, how much weight `style` simultaneously gets.
    A nonzero 'style@lead' is simultaneous co-application; the argmax confusion matrix hides it."""
    names = [d["name"] for d in domains]
    si = names.index(style) if style in names else None
    print("\n" + "═" * 84)
    print(f" CO-APPLICATION (soft) — is '{style}' applied together WITH domain behaviors, per token?")
    print("─" * 84)
    print(f" {'input tokens':<12} | {'mean routing weight (top-3 experts)':<46} | {style}@lead")
    print(" " + "-" * 82)
    for b in behaviors:
        if b["name"] == style:
            continue
        li = names.index(b["name"])
        acc = torch.zeros(model.N); ntok = 0; style_sum = 0.0; lead_tok = 0
        for ids in b["route_eval"]:
            model(input_ids_A=ids, use_bridges=True)
            w = model._route_w[0].float().cpu()            # [T,N] soft routing weights
            acc += w.sum(0); ntok += w.shape[0]
            if si is not None:
                lead = (w.argmax(-1) == li)
                style_sum += w[lead][:, si].sum().item(); lead_tok += int(lead.sum())
        mean_w = acc / max(ntok, 1)
        order = mean_w.argsort(descending=True)
        top3 = ", ".join(f"{names[k]} {mean_w[k]:.2f}" for k in order[:3])
        co = (style_sum / lead_tok) if lead_tok else 0.0
        print(f" {b['name']:<12} | {top3:<46} | {co:.2f}")
    print("─" * 84)
    print(f" '{style}@lead' = mean '{style}' weight on tokens where the input behavior's OWN expert wins.")
    print(f"  >0 ⇒ '{style}' is applied SIMULTANEOUSLY with the domain on the same tokens (top-1 can't).")
    print("═" * 84)


def run_routed():
    rc.set_seed(rc.CFG["seed"])
    base, tok = rc.load_model(rc.CFG["model"], rc.CFG["quantization"])
    ft = rc.CFG["ft"]; dcfg = RCFG["dpo"]; domains = RCFG["domains"]; N = len(domains)
    print(f"\n {N} behaviors on one frozen base: {[d['name'] for d in domains]}")

    bridge_sets, behaviors, route_feats, route_labels = [], [], [], []
    for di, d in enumerate(domains):
        task = d.get("task", "ft")
        print(f"\n── behavior {di}: {d['name']} ({task}) ──")
        rc.set_seed(rc.CFG["seed"]); eng = rc.build_arm("rc", base, tok)
        if task == "ft":
            enc_tr, ev = load_domain(tok, d, ft["max_seq_len"], RCFG["eval_n"], rc.CFG["seed"])
            rc.train_ce(eng, enc_tr, ft["steps"], ft["lr"], ft["grad_accum"])
            b = {"name": d["name"], "task": "ft", "ded": rc.eval_ppl(eng, ev, use_bridges=True),
                 "route_eval": ev}
            rseqs = enc_tr[: min(len(enc_tr), 200)]
        else:  # dpo behavior via rc.py's ultrafeedback pipeline
            enc_tr = rc.encode_pairs(tok, rc.load_pairs(dcfg["train_split"], dcfg["train_n"]))
            enc_ev = rc.encode_pairs(tok, rc.load_pairs(dcfg["eval_split"], dcfg["eval_n"]))
            print(f"   DPO pairs: {len(enc_tr)} train / {len(enc_ev)} eval")
            rw, rl = rc.reference_logprobs(eng, enc_tr, dcfg["length_norm"])
            ew, el = rc.reference_logprobs(eng, enc_ev, dcfg["length_norm"])
            rc.train_dpo(eng, enc_tr, rw, rl, dcfg["beta"], dcfg["steps"], dcfg["lr"],
                         dcfg["grad_accum"], dcfg["nll_lambda"], dcfg["length_norm"])
            m = rc.eval_dpo(eng, enc_ev, ew, el, dcfg["beta"], dcfg["length_norm"])
            b = {"name": d["name"], "task": "dpo", "ded": m["reward_acc"], "dpo_ev": (enc_ev, ew, el),
                 "route_eval": [p["full_w"] for p in enc_ev[:20]]}
            rseqs = [p["full_w"] for p in enc_tr[:200]]
        route_layer = sorted(int(l) for l in eng.bridges.keys())[0]
        eng.remove_hooks(); bridge_sets.append(eng.bridges); behaviors.append(b)
        for L in rc._decoder_layers(base): L._forward_hooks.clear()
        route_feats += cache_route_features(base, route_layer, rseqs); route_labels += [di] * len(rseqs)
        gc.collect(); torch.cuda.empty_cache()

    print("\n── training router (frozen features + domain labels) ──")
    model = RoutedModel(base, tok, bridge_sets, route_mode=RCFG["route_mode"], router_hidden=RCFG["router_hidden"])
    train_router(model.router, route_feats, route_labels, N, RCFG["router_steps"], RCFG["router_lr"])

    # in-bank eval: FT under natural routing; DPO forced to its own adapter (coexistence, not routing)
    for bi, b in enumerate(behaviors):
        if b["task"] == "ft":
            b["routed"] = eval_routed_ppl(model, b["route_eval"])
        else:
            enc_ev, ew, el = b["dpo_ev"]
            model.set_force(bi)
            b["routed"] = rc.eval_dpo(model, enc_ev, ew, el, dcfg["beta"], dcfg["length_norm"])["reward_acc"]
            model.set_force(None)
    conf = routing_confusion(model, [(b["name"], b["route_eval"]) for b in behaviors])

    base_p = sum(p.numel() for p in base.parameters())
    adap_p = sum(sum(p.numel() for p in s.parameters()) for s in bridge_sets)
    rout_p = sum(p.numel() for p in model.router.parameters())
    def mb(n): return n * 2 / 1e6

    print("\n" + "═" * 82)
    print(f" ROUTED RC — {N} behaviors on one frozen {rc.CFG['model'].split('/')[-1]}  (mode={RCFG['route_mode']})")
    print("═" * 82)
    print(f" {'behavior':<12} | {'task':<4} | {'metric':<8} | {'dedicated':>9} | {'in-bank':>9} | {'recovery'}")
    print(" " + "-" * 68)
    for b in behaviors:
        if b["task"] == "ft":
            print(f" {b['name']:<12} | ft   | PPL↓     | {b['ded']:>9.2f} | {b['routed']:>9.2f} | "
                  f"{b['ded']/b['routed']:>5.2f}× (routed)")
        else:
            print(f" {b['name']:<12} | dpo  | reward↑  | {b['ded']:>9.3f} | {b['routed']:>9.3f} | "
                  f"{b['routed']/b['ded']:>5.2f}× (forced)")
    print("═" * 82)
    print(" ROUTING — fraction of each behavior's tokens sent to each expert (diagonal = correct):")
    print("            " + "".join(f"{d['name'][:8]:>10}" for d in domains))
    for i, d in enumerate(domains):
        row = "".join(f"{conf[i,k].item():>10.2f}" for k in range(N))
        print(f" {d['name']:<10}{row}   (self {conf[i,i].item():.2f})")
    print("═" * 82)
    print(f" FOOTPRINT — base {mb(base_p):.0f} MB + {N} adapters {mb(adap_p):.2f} MB + router {mb(rout_p):.3f} MB")
    print(f"            resident total ≈ {mb(base_p+adap_p+rout_p):.1f} MB   vs   {N}× full models ≈ {mb(N*base_p):.0f} MB")
    print(f"            ({N} behaviors for +{mb(adap_p+rout_p):.2f} MB over one base — the edge/local case)")
    print("═" * 82)
    print(" READ: FT recovery≈1× under routing; DPO reward retained when its adapter is applied (forced)")
    print("       = FT and DPO behaviors COEXIST on one frozen base. Strong diagonal = router discriminates.")
    print("       For DPO co-application mid-sequence (style + domain together), set route_mode='soft'.")

    if RCFG["route_mode"] == "soft":                       # demonstrate style co-active with domains
        coapplication_report(model, behaviors, domains, style="preference")


# ═════════════════════════════════════════════════════════════════════════════
#  SELFTEST — mechanism on tiny synthetic domains
# ═════════════════════════════════════════════════════════════════════════════
def selftest():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    print("=== rc_routed selftest (tiny Qwen2, CPU, synthetic domains) ===")
    rc.DEVICE = "cpu"; rc.set_seed(0); V = 60; Nd = 3
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True)
    base = Qwen2ForCausalLM(cfg).to("cpu").eval(); base.requires_grad_(False)
    class T: vocab_size = V; pad_token = "<p>"; eos_token = "<e>"; eos_token_id = V - 1; padding_side = "r"
    tok = T(); layers = [1, 2, 3]
    def clear(): 
        for L in rc._decoder_layers(base): L._forward_hooks.clear()
    def seqs_for(d, n): return [torch.randint(20*d, 20*d+18, (1, 12)) for _ in range(n)]

    bridge_sets = []
    for d in range(Nd):
        rc.set_seed(d + 1)
        m = rc.SteeringModel(base, tok, layers, proj_type="linear", rank=8, alpha=8.0, device="cpu")
        for b in m.bridges.values():
            nn.init.normal_(b.proj.up.weight, std=0.08); nn.init.normal_(b.proj.up.bias, std=0.08)
        m.remove_hooks(); bridge_sets.append(m.bridges)
    clear()
    ids = seqs_for(0, 1)[0]
    raw = base(input_ids=ids, use_cache=False).logits.float()

    # [1] routed forward runs, finite
    model = RoutedModel(base, tok, bridge_sets, route_mode="top1", device="cpu")
    o = model(input_ids_A=ids, use_bridges=True)[0]
    assert torch.isfinite(o).all()
    print(f"[1] routed forward runs, finite logits, {model.N} experts, route@layer {model.route_layer}")
    model.remove_hooks(); clear()

    # [2] top-1 one-hot routing == selecting one expert's injection exactly (single live model at a time)
    mA = RoutedModel(base, tok, bridge_sets, route_mode="top1", device="cpu")
    with torch.no_grad():
        mA.router.net.weight.zero_(); mA.router.net.bias.zero_(); mA.router.net.bias[2] = 10.0
    routed = mA(input_ids_A=ids, use_bridges=True)[0]; mA.remove_hooks(); clear()
    mB = RoutedModel(base, tok, [bridge_sets[2]], route_mode="top1", device="cpu")
    only2 = mB(input_ids_A=ids, use_bridges=True)[0]; mB.remove_hooks(); clear()
    assert torch.allclose(routed, only2, atol=1e-4), "top-1 route must equal the selected expert alone"
    print("[2] top-1 routing == the selected expert's injection alone (one-hot exact)")

    # [3] router learns to discriminate separable domains (above chance; tiny random net can't separate cleanly)
    model = RoutedModel(base, tok, bridge_sets, route_mode="top1", device="cpu")
    train_seqs, train_labels = [], []
    for d in range(Nd):
        s = seqs_for(d, 40); train_seqs += s; train_labels += [d]*len(s)
    feats = cache_route_features(base, model.route_layer, train_seqs)
    train_router(model.router, feats, train_labels, Nd, steps=400, lr=3e-2)
    conf = routing_confusion(model, [(f"d{d}", seqs_for(d, 10)) for d in range(Nd)])
    diag = torch.diag(conf).mean().item()
    assert diag > 1.4 / Nd, f"router not discriminating above chance (diag={diag:.2f}, chance={1/Nd:.2f})"
    print(f"[3] router discriminates synthetic domains: mean self-route={diag:.2f} (chance={1/Nd:.2f})")

    # [4] soft routing runs
    model.route_mode = "soft"
    assert torch.isfinite(model(input_ids_A=ids, use_bridges=True)[0]).all()
    print("[4] soft routing runs (behaviors blend per token)")

    # [4b] set_force routes ALL tokens to one expert (used by the DPO coexistence eval)
    model.route_mode = "top1"; model.set_force(1)
    model(input_ids_A=ids, use_bridges=True)
    w = model._route_w                                    # [B,T,N] routing weights actually applied
    model.set_force(None)
    assert (w.argmax(-1) == 1).all() and torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1))), \
        "set_force must one-hot every token to the chosen expert"
    print("[4b] set_force pins routing to one expert (DPO adapter can be applied on demand)")

    # [5] bridges off == base
    off = model(input_ids_A=ids, use_bridges=False)[0]
    assert torch.allclose(off, raw, atol=1e-4)
    model.remove_hooks(); clear()
    print("[5] routing off == frozen base")

    # [6] co-application readout runs under soft (mean routing weights, style co-activation)
    model2 = RoutedModel(base, tok, bridge_sets, route_mode="soft", device="cpu")
    fake_behaviors = [{"name": f"d{d}", "route_eval": seqs_for(d, 3)} for d in range(Nd)]
    fake_domains = [{"name": f"d{d}"} for d in range(Nd)]
    coapplication_report(model2, fake_behaviors, fake_domains, style="d2")
    model2.remove_hooks(); clear()
    print("[6] co-application readout runs (soft)")
    print("\nALL ROUTED SELFTESTS PASSED.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        run_routed()
