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

The bank is a Hugging Face repository holding one folder per behavior, named
`<subject>_<method>`. A behavior only loads onto the base it was trained against,
so the base model is read from the artifacts rather than set here.
""")

code('''
BANK = "pfekin/mobs-qwen3.5-4b"    # public, no token needed
QUANT = "4bit"                     # 4bit, 8bit or none
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

import serve_mobs as S

class Args:
    base, bank, quant = BASE, root, QUANT
    dtype = "float16" if torch.cuda.get_device_capability(0)[0] < 8 else "bfloat16"
    host, port, max_new = "127.0.0.1", PORT, 512

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

1.0 is the strength a behavior was trained at. Depending on the bank that may be
plenty or it may be barely visible, since what reaches the model is the total
across everything switched on rather than any single value. Pull them all to zero
and you have the untouched base to compare against.
""")

code('''
from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(PORT, height=760)
''')

md("""
## Things to try in the mixer

Three prompts worth pasting into the box above, with the fader settings that
make the difference visible. The point of each is left for you to see rather
than described here.

**Write a Python function that reverses a dictionary.**
Run it once with every fader at zero, then again with every fader at one. Watch
what happens to the length.

**A train leaves at 14:20 travelling at 90 km/h. Another leaves the same station
at 14:50 at 120 km/h. When does the second catch the first?**
Every fader at zero, then every fader at one. Here the difference is not the
presentation. Check the arithmetic yourself before deciding which one to trust.

**What is the difference between TCP and UDP?**
Every fader at zero, then every fader at zero except summary, pinned, at 1.0 or
above. The change between 0.75 and 1.0 is worth stepping through slowly: the
dial is not always smooth.

Not every behavior earns its place. A medical behavior adds little to a model
this size, which already knows the material, and a politeness behavior adds
almost nothing to a model that instruction tuning has already made polite. A
behavior is worth having when the base has a habit worth overriding.
""")

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
