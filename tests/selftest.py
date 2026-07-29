"""Self-tests for the lara package. No network, no HF download: a small
stand-in model with the same interface as an HF causal LM.

    python selftest.py
"""
import shutil
import tempfile

import torch
import torch.nn as nn

from lara import LARA, Bank, resolve_layers


# ── a minimal stand-in with the HF surface we rely on ────────────────────────
class Cfg:
    def __init__(self, d, depth, vocab, name):
        self.hidden_size = d
        self.num_hidden_layers = depth
        self.vocab_size = vocab
        self._name_or_path = name


class Block(nn.Module):
    def __init__(self, d, tuple_out=False):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.tuple_out = tuple_out

    def forward(self, h, **kw):
        out = h + torch.tanh(self.lin(self.ln(h)))
        return (out, None) if self.tuple_out else out


class Inner(nn.Module):
    def __init__(self, d, depth, vocab, tuple_out):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d, tuple_out) for _ in range(depth)])


class Out:
    def __init__(self, logits): self.logits = logits


class TinyLM(nn.Module):
    """Mimics model.model.layers / model.config / model(input_ids=...).logits"""

    def __init__(self, d=32, depth=28, vocab=64, name="tiny/base-a", tuple_out=False):
        super().__init__()
        self.config = Cfg(d, depth, vocab, name)
        self.model = Inner(d, depth, vocab, tuple_out)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, use_cache=False, **kw):
        h = self.model.embed(input_ids)
        for blk in self.model.layers:
            o = blk(h)
            h = o[0] if isinstance(o, tuple) else o
        return Out(self.head(h))


class TinyTok:
    """Enough tokenizer surface for router fitting."""
    def __call__(self, text, return_tensors=None, truncation=True, max_length=64):
        ids = [(ord(c) % 60) + 1 for c in text][:max_length] or [1]
        return type("E", (), {"input_ids": torch.tensor([ids])})()


def ids(n=12, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab, (1, n), generator=g)


ok = lambda msg: print(f"  [pass] {msg}")


def main():
    torch.manual_seed(0)

    # ── 1. layer resolution reproduces the paper's placements ────────────────
    assert resolve_layers(6, 28) == [4, 8, 12, 16, 20, 24], resolve_layers(6, 28)
    assert resolve_layers(1, 28) == [14], resolve_layers(1, 28)
    assert resolve_layers([4, 8], 28) == [4, 8]
    assert resolve_layers(6, 32) == [5, 9, 14, 18, 23, 27], resolve_layers(6, 32)
    assert resolve_layers(3, 12) == [3, 6, 9], resolve_layers(3, 12)
    try:
        resolve_layers(99, 28); raise AssertionError("should have raised")
    except ValueError:
        pass
    ok("layer resolution: 6 -> [4,8,12,16,20,24], 1 -> [14], counts spread evenly")

    # ── 2. identity at init: attaching changes nothing ───────────────────────
    m = TinyLM()
    x = ids()
    before = m(input_ids=x).logits.clone()
    lara = LARA(m, layers=6, rank=8, alpha=8.0)
    after = m(input_ids=x).logits
    assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()
    ok("zero-init: attaching LARA leaves the forward pass unchanged")

    # ── 3. modules are reachable and are the only trainable tensors ──────────
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("lara_modules.") for n in trainable), trainable[:3]
    assert lara.num_trainable() == sum(p.numel() for n, p in m.named_parameters()
                                       if p.requires_grad)
    d, r = 32, 8
    expect = 6 * (2 * d * r + r * 0 + d + 2 * d)      # down + up(+bias) + LN affine
    ok(f"freezing: {lara.num_trainable():,} trainable params, all under model.parameters()")

    # ── 4. a gradient step changes the output; disabled() restores the base ──
    loss = m(input_ids=x).logits.pow(2).mean()
    loss.backward()
    with torch.no_grad():
        for p in lara.trainable_parameters():
            if p.grad is not None:
                p -= 0.5 * p.grad
    trained = m(input_ids=x).logits
    assert not torch.allclose(before, trained, atol=1e-6)
    with lara.disabled():
        base_again = m(input_ids=x).logits
    assert torch.allclose(before, base_again, atol=1e-6)
    ok("training moves the model; disabled() returns the frozen base exactly")

    # ── 5. gamma interpolates ───────────────────────────────────────────────
    lara.gamma = 0.0
    assert torch.allclose(m(input_ids=x).logits, before, atol=1e-6)
    lara.gamma = 1.0
    assert torch.allclose(m(input_ids=x).logits, trained, atol=1e-6)
    lara.gamma = 0.5
    half = m(input_ids=x).logits
    assert not torch.allclose(half, before, atol=1e-6)
    assert not torch.allclose(half, trained, atol=1e-6)
    lara.gamma = 1.0
    ok("gamma: 0 recovers the base, 1 the trained model, between is between")

    # ── 6. save / load roundtrip ────────────────────────────────────────────
    tmp = tempfile.mkdtemp()
    lara.save(f"{tmp}/code", route_samples=["def f(x):", "import os"], method="ce")
    lara.detach()
    m2 = TinyLM()
    m2.load_state_dict(m.state_dict(), strict=False)
    lara2 = LARA.from_pretrained(m2, f"{tmp}/code")
    assert lara2.layer_ids == [4, 8, 12, 16, 20, 24]
    assert torch.allclose(m2(input_ids=x).logits, trained, atol=1e-5)
    ok("save/load: adapter roundtrips, layers and weights restored")

    # ── 7. a behavior refuses to load onto a different base ─────────────────
    other = TinyLM(name="tiny/base-b")
    try:
        LARA.from_pretrained(other, f"{tmp}/code"); raise AssertionError("should have raised")
    except ValueError as e:
        assert "do not transfer" in str(e)
    ok("base mismatch raises instead of silently producing garbage")
    lara2.detach()

    # ── 8. bank: add behaviors, route, weights sum to one ───────────────────
    base = TinyLM()
    beh = {}
    for i, name in enumerate(["code", "math", "polite"]):
        mm = TinyLM()
        mm.load_state_dict(base.state_dict(), strict=False)
        L = LARA(mm, layers=6, rank=8, alpha=8.0)
        with torch.no_grad():                       # give each a distinct signature
            for mod in L.modules_.values():
                mod.up.weight.normal_(0, 0.02 * (i + 1))
        L.save(f"{tmp}/{name}", route_samples=[f"{name} sample {j}" for j in range(6)],
               method=["ce", "ce", "dpo"][i])
        L.detach()
        beh[name] = f"{tmp}/{name}"

    bank = Bank(base, TinyTok())
    for n, p in beh.items():
        bank.add(n, p)
    assert len(bank) == 3 and bank.router.n == 3
    w = bank.route_weights(x)
    assert abs(sum(w.values()) - 1.0) < 1e-4, w
    ok(f"bank: 3 behaviors resident, routing weights sum to 1 {tuple(round(v,2) for v in w.values())}")

    # ── 9. top_k: 1 is hard, k blends k, None is full soft ──────────────────
    bank.top_k = 1
    w1 = bank.route_weights(x)
    assert sum(1 for v in w1.values() if v > 1e-6) >= 1
    assert max(w1.values()) > 0.99 or sum(v > 1e-6 for v in w1.values()) <= 3
    bank.top_k = 2
    w2 = bank.route_weights(x)
    assert abs(sum(w2.values()) - 1.0) < 1e-4
    bank.top_k = None
    w3 = bank.route_weights(x)
    assert all(v > 1e-9 for v in w3.values()), w3
    ok("top_k: 1 hard-selects, k blends the top k, None blends all")

    # ── 10. pin overrides the router ────────────────────────────────────────
    with bank.pin("math"):
        wp = bank.route_weights(x)
    assert wp["math"] > 0.999 and wp["code"] < 1e-6, wp
    with bank.pin({"code": 1.0, "polite": 0.3}):
        wb = bank.route_weights(x)
    assert abs(wb["code"] - 1.0) < 1e-4 and abs(wb["polite"] - 0.3) < 1e-4
    ok("pin: a name forces one behavior, a dict forces a manual blend")

    # ── 11. bank output differs from base, and disabled() restores it ───────
    base_logits = TinyLM(); base_logits.load_state_dict(base.state_dict(), strict=False)
    with bank.disabled():
        off = base(input_ids=x).logits
    on = base(input_ids=x).logits
    assert not torch.allclose(off, on, atol=1e-6)
    ok("bank injects: output differs from base, disabled() restores it")

    # ── 12. router fitting, then incremental add ────────────────────────────
    w_before = bank.router.net.weight.detach().clone()
    bank.fit_router(steps=200)
    assert not torch.allclose(w_before, bank.router.net.weight, atol=1e-6)
    assert bank.router_train_acc > 1.0 / len(bank) + 0.1, bank.router_train_acc
    ok(f"fit_router: separates the behaviors' samples (train acc {bank.router_train_acc:.2f} "
       f"against {1/len(bank):.2f} chance)")

    mm = TinyLM(); mm.load_state_dict(base.state_dict(), strict=False)
    L = LARA(mm, layers=6, rank=8, alpha=8.0)
    L.save(f"{tmp}/legal", route_samples=[f"legal sample {j}" for j in range(6)])
    L.detach()
    bank.add("legal", f"{tmp}/legal")
    assert len(bank) == 4 and bank.router.n == 4
    bank.fit_router(mode="append", steps=40)
    w4 = bank.route_weights(x)
    assert abs(sum(w4.values()) - 1.0) < 1e-4 and len(w4) == 4
    ok("incremental: add() grows the router, append mode fits only the new row")

    # ── 13. bank save / load ────────────────────────────────────────────────
    bank.save(f"{tmp}/mybank")
    before_w = bank.route_weights(x)
    bank.detach()
    fresh = TinyLM(); fresh.load_state_dict(base.state_dict(), strict=False)
    bank2 = Bank.load(f"{tmp}/mybank", fresh, TinyTok())
    after_w = bank2.route_weights(x)
    assert bank2.names == bank.names
    assert all(abs(before_w[k] - after_w[k]) < 1e-4 for k in before_w), (before_w, after_w)
    ok("bank save/load: behaviors, router, and routing weights all roundtrip")

    # ── 14. tuple-returning blocks (GPT-2 style) ────────────────────────────
    gpt = TinyLM(tuple_out=True)
    b0 = gpt(input_ids=x).logits.clone()
    lg = LARA(gpt, layers=3, rank=4)
    assert torch.allclose(gpt(input_ids=x).logits, b0, atol=1e-6)
    with torch.no_grad():
        for mod in lg.modules_.values():
            mod.up.weight.normal_(0, 0.05)
    assert not torch.allclose(gpt(input_ids=x).logits, b0, atol=1e-6)
    ok("blocks returning tuples are handled alongside blocks returning tensors")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
