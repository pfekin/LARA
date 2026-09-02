"""Build mobs_playground.ipynb.

The notebook is a driver, not a copy. It clones the repo and runs the real
serve_mobs.py and train_mobs.py, so a fix to either reaches notebook users on
their next run and there is only ever one copy of the code.
"""
import json

C = []


def lines(s):
    out = s.split("\n")
    return [l + "\n" for l in out[:-1]] + out[-1:]


def md(s):
    C.append({"cell_type": "markdown", "metadata": {}, "source": lines(s.strip())})


def code(s):
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": lines(s.strip("\n"))})


md("""
# MoBs: many behaviors on one frozen base

A bank of small adapters over a language model whose weights are never touched.
Each behavior is a few megabytes. A router picks a mixture for every token, and a
scale per behavior lets you turn each one up or down while the model is running.
At zero the forward pass is the original model, bit for bit.

Run the cells and a mixer appears at the bottom: a prompt box, one fader per
behavior, and the output underneath.

**Runtime > Change runtime type > GPU** first. A T4 is enough.

Training your own behavior is at the end, switched off by default.
""")

md("## Setup")

code("""
!pip -q install transformers accelerate safetensors kernels bitsandbytes
!pip -q install fastapi uvicorn "huggingface_hub>=0.34"
!pip -q install git+https://github.com/pfekin/LARA.git
# A second run in the same session must pick up pushes rather than silently
# reusing whatever is already on disk: git clone fails when the directory
# exists and leaves the old checkout in place.
!rm -rf /content/LARA
!git clone -q https://github.com/pfekin/LARA.git /content/LARA
!cd /content/LARA && git log -1 --format="repo at %h, %ar"

import os, sys, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
sys.path.insert(0, "/content/LARA/mobs")
os.chdir("/content/LARA/mobs")

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "none. Runtime > Change runtime type > GPU")

# serve_mobs may already be imported from an earlier run in this session, in
# which case a fresh clone changes nothing until the module is reloaded.
import importlib
for m in ("serve_mobs", "train_mobs"):
    if m in sys.modules:
        importlib.reload(sys.modules[m])
        print(f"reloaded {m} from the new checkout")
""")

md("""
## Configuration

`BANK` and `QUANT` are read from `serve_mobs.py`, so the notebook and the command
line agree by construction. The bank is a Hugging Face repository holding one
folder per behavior, named `<subject>_<method>`. A behavior only loads onto the
base it was trained against, so the base is taken from the artifacts rather than
set here.
""")

code('''
# Defaults come from serve_mobs.py so there is one place to change them.
import serve_mobs as S
_d = S.defaults()

BANK = _d["bank"]                  # public, no token needed
QUANT = _d["quant"]
TRAIN = False                      # the last section; wants an L4

import json
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download

print(f"bank {BANK}, {QUANT}")

root = snapshot_download(BANK, repo_type="model")
meta = [f for f in list_repo_files(BANK) if f.endswith("/config.json")]
BASE = json.load(open(hf_hub_download(BANK, meta[0])))["base_model_id"]

names = sorted(d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))
               and os.path.exists(os.path.join(root, d, "config.json")))
for n in names:
    mb = sum(os.path.getsize(os.path.join(root, n, f))
             for f in os.listdir(os.path.join(root, n))) / 1e6
    print(f"  {n:<24}{mb:>7.1f} MB")
print(f"\\nbase {BASE}, frozen")
''')

md("""
## Start the server

`serve_mobs.py` loads the base once, attaches every behavior, fits the router,
and serves the mixer page. It runs in a background thread so the notebook stays
usable.

First run downloads the base model, which takes a few minutes.
""")

code('''
import threading, time, socket, urllib.request

def free_port(start=8000):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)):
                return p
    raise RuntimeError("no free port")

PORT = free_port()

class Args:
    base, bank, quant = BASE, root, QUANT
    dtype = "float16" if torch.cuda.get_device_capability(0)[0] < 8 else _d["dtype"]
    host, port, max_new = "127.0.0.1", PORT, _d["max_new"]

model, tok, bank, names = S.build(Args.base, Args.bank, Args.quant, Args.dtype)
app = S.make_app(model, tok, bank, names, Args)

import uvicorn
cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()

for _ in range(60):                       # wait for it to answer
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/meta", timeout=1)
        print(f"serving on port {PORT}")
        break
    except Exception:
        time.sleep(1)
else:
    print("server did not start; check the output above")
''')

md("""
## The mixer

Type a prompt and press Generate. Each fader is one behavior.

1.0 is the strength a behavior was trained at, which is often too weak to see.
What reaches the model is the total across everything switched on, so one
behavior near 2, or two at 1.5 each, is usually where the effect arrives. Pull
them all to zero and you have the untouched base to compare against.
""")

code('''
from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(PORT, height=760)
''')

md("""
## Without the interface

The same thing from Python, if you would rather script it. Behaviors are set by
name, and the bank is a normal object.
""")

code('''
import textwrap

def gen(prompt, gammas=None, max_new=300, base=False, temperature=0.8):
    """gammas is {name: strength}. Anything omitted is silenced."""
    for n in names:
        bank.set_gamma(n, (gammas or {}).get(n, 0.0))
    text = S._chatml(tok, prompt)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    # only these two: a tokenizer that adds token_type_ids breaks generate, and
    # carrying anything else between calls confuses the hybrid cache
    enc = {k: v.to(model.device) for k, v in enc.items()
           if k in ("input_ids", "attention_mask")}
    ctx = bank.disabled() if base else S._null()
    with ctx, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             past_key_values=None, use_cache=True,
                             temperature=temperature, top_p=0.95,
                             repetition_penalty=1.05,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


def show(title, text):
    print()
    print(title)
    print(chr(9472) * 80)
    for para in text.split(chr(10)):
        print(textwrap.fill(para, 80) if para.strip() else "")


# Pick a prompt that the behaviors in this bank actually have something to say
# about. The default bank is code, math, medical, summary and polite, so a
# question that touches several of them shows more than a literary one would.
PROMPT = ("A patient weighs 68 kg. Work out the maximum daily paracetamol dose "
          "at 60 mg per kg, and write a short Python function that checks a "
          "proposed dose against it.")

show("frozen base", gen(PROMPT, base=True))
for n in names:
    show(f"{n} at 2.0", gen(PROMPT, {n: 2.0}))
''')

md("""
## What the router is doing

The router reads the stream at one early layer and produces a weight per behavior
for every token. It separates behaviors that occupy different kinds of input, so
a code request and a medical request route differently.

It cannot separate behaviors that answer the same kind of request. Several
writing styles all look like "write a paragraph about ...", so the weights come
back close to uniform and the useful control is the faders rather than the
router.

The bank loaded here is domain-shaped rather than style-shaped, so the weights
should not be uniform. A question about dosage and a request for a function
should pull on different behaviors.
""")

code('''
ids = tok(S._chatml(tok, PROMPT), return_tensors="pt",
          add_special_tokens=False).input_ids
w = bank.route_weights(ids)
for k, v in sorted(w.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<24}{v:.3f}  {'#' * int(v * 60)}")
''')

md("""
## Train your own

Set `TRAIN = True` in the configuration cell to run this.

`train_mobs.py` takes a list of behaviors, each with a method and a dataset, and
writes one artifact per behavior. Datasets are jsonl: `prompt` and `completion`
for supervised fine-tuning, `prompt`, `chosen` and `rejected` for DPO. Point it
at a Hub dataset or a local file.

This wants an L4. Loading and running the result does not.
""")

code('''
if not TRAIN:
    print("TRAIN is off. Set it in the configuration cell to train.")
else:
    import train_mobs as T

    T.BEHAVIORS = [
        dict(name="mystyle", method="sft", dataset="file:data/mystyle.jsonl",
             n=2000, layers=6, rank=128, alpha=128),
    ]

    T.main(["--base", BASE, "--out", "behaviors",
            "--push", "you/your-bank"])       # needs HF_TOKEN for the push
''')

md("""
## What this is

The base model was never modified. Its weights are identical to the ones on the
Hub, and with every fader at zero the forward pass reproduces it exactly. Each
behavior is a file of a few megabytes that reads the hidden state at six layers,
computes a low-rank correction, and adds it back to the residual stream.

Because nothing is merged into the weights, behaviors do not have to be chosen
before the model is loaded. They can be added, removed and scaled while it runs,
and a base with a bank on it costs one model in memory rather than one per
adaptation.

Code and method: [github.com/pfekin/LARA](https://github.com/pfekin/LARA)
""")

nb = {"cells": C,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("mobs_playground.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {len(C)} cells")
