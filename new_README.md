# LARA

**Lightweight Additive Residual Adaptation**: post-training for frozen language models, with small adaptations that can be combined as a **Mixture of Behaviors (MoBs)**.

<div align="left">

[![arXiv](https://img.shields.io/badge/arXiv-2607.28669-v1.svg)](https://doi.org/10.48550/arXiv.2607.28669)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfekin/LARA/blob/main/examples/quickstart.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/LARA_readme_hero_dark.png">
  <img alt="LARA reads the hidden state between layers, computes a low-rank correction, and adds it back to the residual stream" src="media/LARA_readme_hero.png">
</picture>

[Paper](https://doi.org/10.48550/arXiv.2607.28669) · [Slides](media/LARA_slides.pdf) · [Video walkthrough](https://github.com/user-attachments/assets/9591acc0-d7f4-4c71-895a-26798a0b03e5)

Foundation models made a different approach to AI development possible: build a general model first, then tailor it for particular purposes afterwards. LARA is a post-training method for that second step. It adapts a frozen language model without modifying its weights, producing a small behavior artifact that is kept separate from the base model.

At a small set of layers, LARA reads the hidden state, computes a low-rank correction, and adds it back to the residual stream. The base model is loaded once and stays frozen, so each behavior is a separate file of a few megabytes, and several can sit on one model at the same time.

What it adds over a weight-space adapter is a scale you can turn at inference, and the ability to hold many behaviors resident on one base and pick between them per token.

The larger system built around LARA is **Mixture of Behaviors (MoBs)**. A MoB is a collection of independently trained behaviors sharing one frozen base. Behaviors can be trained with SFT, DPO, GRPO or other post-training methods. At inference they can be selected, scaled, pinned or composed, with hard or soft routing.

The distinction is useful:

```text
capability        = base model
behavior          = learned adaptation
selection         = router
strength          = runtime scaling
combination       = composition
```

This turns post-training into a modular layer around a foundation model.

## Why LARA

LoRA showed that useful adaptation does not require updating every parameter of a language model. LARA follows the same motivation but places the correction in the residual stream rather than in the model's weight matrices.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/figure1_lara_vs_lora_dark.svg">
  <img alt="LoRA changes the weight matrices; LARA reads the residual stream and adds a low-rank correction back to it, leaving the block frozen." src="media/figure1_lara_vs_lora.svg">
</picture>

The base model is loaded once and stays frozen. Each behavior is a separate file, typically a few megabytes. That makes it practical to keep several behaviors on the same model rather than producing a full model for every specialization.

The point is not smaller fine-tuning. It is adaptation that is **independent, composable and runtime-selectable**.

## How it works

At each of its layers, an adapter does this:

```python
h = h + gamma * (alpha / rank) * up(down(layer_norm(h)))
```

`down` projects to rank `r`, `up` projects back. `up` starts at zero, so an untrained adapter contributes nothing and the model is exactly the base. Nothing else is touched: with the adapters removed the forward pass is identical, bit for bit.

At rank 128 over six layers of a 1.5B model that is about 2.4M trainable parameters, a few megabytes on disk.

Because no weights are modified, behaviors compose. A small router reads the frozen hidden state and produces a per-token distribution over the behaviors in the bank, and their corrections are blended by weight. There is no limit to how many you install.

## Install

```bash
pip install git+https://github.com/pfekin/LARA.git
```

Or from a clone:

```bash
git clone https://github.com/pfekin/LARA.git
cd LARA
pip install -e .
```

Python 3.9 or later, torch 2.0 or later. The examples also need `transformers`, `datasets`, `accelerate` and `trl`:

```bash
pip install -e ".[examples]"
```

## Train a behavior

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from lara import LARA

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

lara = LARA(model, layers=6, rank=128)     # base frozen, 2.4M trainable
print(lara.num_trainable())

trainer = Trainer(model=model, args=TrainingArguments(...), train_dataset=ds)
trainer.train()

lara.save("behaviors/code", route_samples=prompts[:200])
```

`layers=6` spreads six adapters evenly over the depth. On a 28-layer model that resolves to `[4, 8, 12, 16, 20, 24]`. Pass a list instead for exact control.

LARA does not replace or wrap the trainer. It attaches the adapters to the model and freezes everything else, so `model.parameters()` reaches the adapters and any trainer picks them up: the HF `Trainer`, TRL's `DPOTrainer` or `GRPOTrainer`, or a loop you wrote yourself. The objective makes no difference to the artifact. A behavior trained with cross-entropy, with DPO, or with a policy gradient loads and routes exactly the same way, and enters the same bank.

Fine-tuning benefits from several insertion points. Preference optimization and reinforcement learning reach the same quality with a single adapter in the middle of the network, so `layers=1` is often enough for both. See [research.md](research.md).

## Turn the adaptation up or down

The correction is additive over an unchanged base, so its strength is a runtime value rather than something fixed at training time:

```python
lara.gamma = 0.0     # the frozen base
lara.gamma = 0.5     # halfway
lara.gamma = 1.0     # the trained behavior
```

Useful range is roughly 0 to 1.5. Past that the correction is amplified beyond what it was trained for and quality falls away.

## Mixture of Behaviors

A **MoB** is the bank of behaviors and the mechanism that selects or combines them over one frozen base.

For example:

```text
one base model

    + code behavior
    + math behavior
    + medical behavior
    + summary behavior
    + writing-style behavior
    + tutor behavior
```

Each item is an independent artifact. The MoB is the collection and the runtime mixture; a single behavior is not itself a MoB.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/figure2_mobs_dark.svg">
  <img alt="One frozen base with a bank of behaviors. Corrections are added to the residual stream between layers, and a small router weights them per token." src="media/figure2_mobs.svg">
</picture>

<sub>Five behaviors on Qwen3-1.7B. Sizes vary with rank and layer count.</sub>

Load several behaviors:

```python
from lara import Bank

bank = Bank(model, tokenizer)
bank.add("code",   "behaviors/code")      # trained with cross-entropy
bank.add("math",   "behaviors/math")
bank.add("polite", "behaviors/polite")    # trained with DPO
bank.fit_router()

out = model.generate(**inputs)            # routing happens inside the forward pass
```

`fit_router` trains a linear classifier, about 11k parameters, on the route samples stored inside each behavior. It takes seconds.

Adding one later does not disturb the others:

```python
bank.add("legal", "behaviors/legal")
bank.fit_router(mode="append")            # keeps the existing rows, fits the new one
```

Save and reload the whole bank:

```python
bank.save("mybank/")
bank = Bank.load("mybank/", model, tokenizer)
```

## Routing

`top_k` controls how many behaviors are applied to each token:

```python
bank.top_k = None    # blend all of them by weight (default)
bank.top_k = 2       # blend the two highest, ignore the rest
bank.top_k = 1       # hard selection, one behavior per token
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/figure3_per_token_routing_dark.gif">
  <img alt="Routing weight across a sentence: the mix shifts from one behavior to another as the text is generated" src="media/figure3_per_token_routing.gif">
</picture>

Behaviors with no weight are skipped, so cost tracks `top_k` rather than the size of the bank. Blending is worth having when behaviors overlap, since a token the router is unsure about gets a mixture rather than a single wrong choice.

Weight-space adapters do not blend as readily. A LoRA update costs nothing at inference once it is merged into the weight matrix, but merging commits the model to one adapter, the same for every token. Mixing several per token means leaving them all unmerged and computing each one's contribution at every matrix it targets, so the work grows with the number of adapters times the number of target matrices. A LARA behavior contributes one vector per layer it sits on, whatever the base does at that layer, so mixing is a weighted sum of vectors.

To override the router:

```python
with bank.pin("code"):                        # one behavior, router ignored
    out = model.generate(**inputs)

with bank.pin({"code": 1.0, "polite": 0.4}):  # a blend you chose
    out = model.generate(**inputs)

with bank.disabled():                         # the frozen base
    out = model.generate(**inputs)
```

Per-behavior scales work too:

```python
bank.set_gamma("polite", 0.6)
```

To see what the router is doing:

```python
bank.route_weights(input_ids)
# {'code': 0.71, 'math': 0.04, 'polite': 0.25}
```

## Why MoBs matter

With conventional fine-tuning, specialization tends to produce another model:

```text
base model
   │
   ├── fine-tune → model A
   ├── fine-tune → model B
   └── fine-tune → model C
```

With LARA and MoBs the base capability is shared and the learned behaviors stay independent. This becomes more useful as the number of adaptations grows: a bank of behaviors occupies megabytes of adapter storage rather than one multi-gigabyte model per specialization.

Behaviors also need not be mutually exclusive. A system can combine them at inference: a mathematical behavior together with a tutoring behavior and a preferred writing style, without training a full model for that combination.

That suggests a general way to think about post-training: **capability in the foundation model, behavior in small learned modules**.

## Modular cognition

The MoB architecture can be read as a form of modular cognition. Different learned behaviors encode different ways of using the same underlying model:

```text
base capability
      │
      ├── planner
      ├── verifier
      ├── critic
      ├── domain specialist
      ├── tutor
      └── personal style
```

A router can select among them or combine them.

The analogy with mixture-of-experts is useful, but the target is different. A conventional MoE routes among experts that are part of one model, mainly to increase capacity. MoBs route among lightweight adaptations over a shared base, to make post-training modular. That difference makes behaviors closer to software components than to further copies of a model.

## Applications

The same architecture applies wherever one model needs several modes of operation. A local AI tutor could combine mathematics, tutoring and a student's preferred explanation style. A coding assistant could combine language-specific coding, debugging, testing and security review. An enterprise model could share one base across departments while keeping separate legal, finance, support and engineering adaptations. A personal AI could maintain a small bank of writing, task and preference behaviors.

These applications do not depend on local inference. The modularity applies to server-side systems as well.

## Why this matters for small local models

Local AI changes the constraints around customization. A small model running on a phone or PC has less room for long system prompts, extended histories and large RAG contexts than a frontier model in the cloud. Those techniques remain useful, but they become an expensive way of telling a small model how to behave.

LARA offers another way to express customization: learn the behavior once and keep it in a small residual-stream adaptation rather than spelling it out in a long prompt at every interaction. A model can acquire a user's preferred writing style, a tutor's teaching method, a company's house style, or a domain-specific way of reasoning without reconstructing all of that from the context window each time.

This does not replace RAG. RAG provides information the model does not have; a behavior changes how the model uses information it already has. The two are complementary. The point is that behavior need not consume context.

The small size of the artifacts is what makes this practical at the edge. A device can carry one base model and a bank of specialized behaviors, and a new specialization is distributed as a small adapter rather than another copy of the base. That matters where storage, memory and network access are constrained, and it allows personal adaptations to stay on the device rather than being sent to a server.

## Small model, specialized behavior

A small base model can become substantially more useful through domain adaptation. An informal test with **Qwen3-1.7B** illustrates the idea. The question was:

> Name a common over-the-counter pain reliever that reduces fever but does NOT increase bleeding risk.

The question is deliberately discriminative. Acetaminophen and ibuprofen both reduce pain and fever, but ibuprofen is an NSAID and carries a bleeding warning, whereas acetaminophen is not an NSAID. The intended answer was **acetaminophen (Tylenol)**.

With all behaviors disabled, the 1.7B model answered:

```text
A common over-the-counter (OTC) pain reliever that reduces and does not
significantly increase bleeding risk is ibuprofen.
```

With the medical behavior enabled, the same model answered:

```text
Acetaminophen (Tylenol) is a common over-the-counter pain reliever that
reduces fever but does NOT increase bleeding risk.
```

One example is an illustration and not a measurement. It shows the narrow point: a lightweight learned behavior can change the model's answer on a domain-specific distinction without changing the base model. A 2.4M parameter correction cannot store medical knowledge, so it did not teach the model that fact. It made a fact the base already held reachable.

That suggests a local-AI pattern:

```text
small general model
        +
learned behavior
        ↓
specialized local model
```

The same pattern applies to problem solving, coding, tutoring, writing style and other kinds of specialization. Measurements across five behaviors and four base models are in [research.md](research.md).

## What a behavior looks like on disk

```text
behaviors/code/
  adapter.safetensors     the projections, a few MB
  config.json             layers, rank, alpha, base model id
  route_samples.jsonl     short texts typical of this behavior
```

The route samples are what let a behavior be routed by someone who does not have the data it was trained on. Without them, `fit_router` has nothing to separate.

A behavior records the base it was trained against and refuses to load onto a different one, rather than loading and producing noise.

## API

`LARA(model, layers=6, rank=128, alpha=128)` attaches a behavior to a frozen model. `layers` is a count or a list.

- `lara.gamma` scale applied at inference
- `lara.num_trainable()` parameter count
- `lara.save(path, route_samples=..., method=...)` write the behavior
- `LARA.from_pretrained(model, path)` attach a saved behavior
- `lara.disabled()` context manager that runs the frozen base
- `lara.detach()` remove the adapters and hooks

`Bank(model, tokenizer, top_k=None)` holds several behaviors over one frozen model.

- `bank.add(name, path)` add a behavior
- `bank.remove(name)` drop one
- `bank.fit_router(mode="refit"|"append", steps=300)` train the router
- `bank.top_k` how many behaviors apply per token
- `bank.pin(name_or_dict)` context manager that overrides the router
- `bank.disabled()` context manager that runs the frozen base
- `bank.gamma`, `bank.set_gamma(name, g)` scales
- `bank.route_weights(input_ids)` mean routing weight per behavior
- `bank.save(path)`, `Bank.load(path, model, tokenizer)`

## Examples

Runnable end to end on one GPU:

- `examples/01_finetune.py` trains a behavior with the HF `Trainer`
- `examples/02_dpo.py` aligns one with TRL's `DPOTrainer`, producing the same kind of artifact
- `examples/03_bank.py` loads both onto one model, fits a router, and generates under soft routing, hard routing and pinning

Run 01 and 02 first, since 03 uses what they write.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfekin/LARA/blob/main/examples/quickstart.ipynb)

## Tests

```bash
python tests/test_lara.py
```

Covers layer resolution, the no-op at initialization, freezing, save and load, routing, `top_k`, pinning, and incremental addition. No model download, so it runs anywhere.

## Results

LARA matches LoRA at equal parameter counts on fine-tuning, preference optimization and reinforcement learning, and the behaviors carry across base models including one whose weights are binary.

Measurements, the gamma sweeps and the routing tables are in [research.md](research.md), alongside the [benchmark code](https://github.com/pfekin/LARA/tree/main/research) that produced the numbers in the [preprint](https://doi.org/10.48550/arXiv.2607.28669) and the instructions to rerun it.

## Citation

```bibtex
@article{ekin2026laralightweightadaptersresidual,
      title={LARA: Lightweight Adapters in the Residual Stream for Composable Adaptation and Alignment},
      author={Pascal Ekin and Hyosun Choi and Wei Jie},
      year={2026},
      eprint={2607.28669},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.28669},
}
@software{ekin2026lara,
  title   = {Lightweight Additive Residual Adaptation (LARA): residual-stream adapters for frozen LLMs. Runs many behaviors per token.},
  author  = {Pascal Ekin},
  year    = {2026},
  url     = {https://github.com/pfekin/LARA}
}
```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
