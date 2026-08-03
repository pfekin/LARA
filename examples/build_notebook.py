#!/usr/bin/env python3
"""Assemble examples/quickstart.ipynb from the three example scripts.

Each code cell loads the base model itself and frees it at the end, so cells can
be run in any order on a single T4 without three copies piling up in memory.
"""
import json
import re

SRC = "examples"


def body(path, drop_header_until=None):
    """Script text with the module docstring and imports removed."""
    txt = open(f"{SRC}/{path}").read()
    txt = re.sub(r'^""".*?"""\n', "", txt, flags=re.S)      # module docstring
    return txt.strip()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/pfekin/LARA/blob/main/"
         "examples/quickstart.ipynb)")

CLEANUP = """
# free the base before the next cell loads its own copy
lara.detach()
del model, trainer
gc.collect(); torch.cuda.empty_cache()
"""

CLEANUP_BANK = """
# free the base
bank.detach()
del model
gc.collect(); torch.cuda.empty_cache()
"""

cells = []

cells.append(md(f"""# LARA quickstart

{BADGE}

**Lightweight Additive Residual Adaptation**: residual-stream adapters for frozen
language models, routed per token.

This notebook trains two behaviors with two different objectives, then puts both
on one frozen model and routes between them. The base model is never modified.

**Runtime:** pick a GPU runtime (Runtime, Change runtime type, T4).

**Time:** about 20 minutes for the first cell, 5 for the second, 1 for the third,
plus a 3 GB model download on first use.

Run the cells in order. Cell 3 uses what cells 1 and 2 write to disk.
"""))

cells.append(md("## Setup"))
cells.append(code("""!pip install -q git+https://github.com/pfekin/LARA.git
!pip install -q transformers datasets accelerate trl

import gc
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from trl import DPOConfig, DPOTrainer

from lara import LARA, Bank

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU: pick a GPU runtime")"""))

cells.append(md("""## 1. Train a behavior with cross-entropy

The only LARA-specific lines are the three marked below. Everything else is
ordinary Hugging Face training: LARA does not wrap, replace, or subclass the
trainer. It attaches modules to the model and freezes everything else, so
`model.parameters()` reaches the modules and any trainer picks them up."""))

cells.append(code(body("01_finetune.py") + "\n" + CLEANUP))

cells.append(md("""## 2. Align a behavior with DPO

A different objective, the same three LARA lines. The artifact this produces is
indistinguishable from the one above: a directory of projections that a bank can
route alongside any other behavior.

This is not a reproduction of the paper. The paper's DPO numbers come from the
harness in `research/`, which uses its own loss with an NLL anchor term that TRL
does not apply. Read the numbers below as a check that the modules trained, not
as a result."""))

cells.append(code(body("02_dpo.py") + "\n" + CLEANUP))

cells.append(md("""## 3. Both behaviors on one frozen model

No training here. This loads what the two cells above wrote, fits a small router
over them, and generates.

Note what does not happen: no behavior is merged into the weights, nothing is
loaded or unloaded between prompts, and adding a behavior later would not touch
the others."""))

cells.append(code(body("03_bank.py") + "\n" + CLEANUP_BANK))

cells.append(md("""## What to look at

**The scale is a runtime value.** Cell 1 prints the top token and entropy across
gamma. The greedy path quantizes what you see, but the distribution moves
continuously underneath.

**The objective does not matter to the artifact.** Cell 2 trains with DPO and
writes the same kind of directory as cell 1.

**Routing has limits, and pinning is the answer.** In cell 3 the router reads the
prompt, and both behaviors here take English prompts, so it cannot separate them
from the prompt alone. Behaviors that differ by input domain (math against
medical, say) route cleanly. Behaviors that differ by output style are better
selected explicitly with `bank.pin(...)`, which the cell demonstrates.

## Next

- Repository: https://github.com/pfekin/LARA
- The experiments behind the paper: `research.md` in the repository"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = f"{SRC}/quickstart.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
