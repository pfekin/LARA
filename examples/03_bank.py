"""Put several behaviors on one frozen model and route between them per token.

No training here. This loads the behaviors produced by the two previous
examples, fits a small router over them, and generates. Run 01 and 02 first.

Note what does not happen: no behavior is merged into the weights, nothing is
loaded or unloaded between requests, and adding a behavior later does not touch
the others.

    python 03_bank.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lara import Bank

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")

# behaviors go on the shared frozen base
bank = Bank(model, tok)
bank.add("code", "behaviors/code")        # trained with cross-entropy (01)
bank.add("polite", "behaviors/polite")    # trained with DPO (02)
print(bank)

# The router reads the frozen stream, so it needs each behavior's route samples,
# which the behaviors carry with them. Seconds on any GPU.
bank.fit_router(steps=300, verbose=True)

mb = sum(p.numel() * p.element_size() for s in bank.sets for p in s.parameters()) / 1e6
print(f"\n{len(bank)} behaviors resident for {mb:.1f} MB over one base")

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "My script throws a KeyError. Can you help?"}],
    tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(model.device)


def show(label):
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, do_sample=False)
    print(f"\n─── {label} " + "─" * 40)
    print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))


# routed: the router decides per token, and can apply both at once
bank.top_k = None                 # blend all behaviors by weight
show("routed (soft)")
print("mean routing weight:", {k: round(v, 3) for k, v in bank.route_weights(ids.input_ids).items()})

# top_k caps the cost: only the k highest-weighted behaviors are applied
bank.top_k = 1                    # hard selection, one behavior per token
show("routed (top_k=1)")

# pin overrides the router entirely, for when you know what you want
with bank.pin("code"):
    show("pinned to code")

with bank.pin({"code": 1.0, "polite": 0.4}):
    show("code, with the aligned style at 0.4")

# the base is still in there, untouched 
with bank.disabled():
    show("frozen base")
