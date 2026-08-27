# LARA

**Lightweight Additive Residual Adaptation**: post-training for frozen language models, with lightweight adaptations that can be combined into a **Mixture of Behaviors (MoBs)**.

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

[Slides](media/LARA_slides.pdf) · [Video walkthrough](https://github.com/user-attachments/assets/9591acc0-d7f4-4c71-895a-26798a0b03e5)

Foundation models made a different approach to AI development possible: build a general model first, then tailor it for particular purposes afterwards. LARA is a post-training method for that second step. It adapts a frozen language model without modifying its weights, producing a small behavior artifact that can be kept separate from the base model.

The larger system built around LARA is **Mixture of Behaviors (MoBs)**. An MoB is a collection of independently trained behaviors sharing one frozen base. Behaviors can be trained with SFT, DPO, GRPO or other post-training methods. At inference time they can be selected, scaled, pinned or composed, with hard or soft routing.

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

The base model is loaded once and stays frozen. Each LARA behavior is a separate file, typically only a few megabytes. That makes it practical to keep several behaviors on the same model rather than creating a separate full model for every specialization.

The point is not simply smaller fine-tuning. It is to make adaptation **independent, composable and runtime-selectable**.

## How it works

At each selected layer, an adapter does this:

```python
h = h + gamma * (alpha / rank) * up(down(layer_norm(h)))
```

`down` projects to rank `r`, `up` projects back. `up` starts at zero, so an untrained adapter contributes nothing and the model is exactly the base. Nothing else is touched.

At rank 128 over six layers of a 1.5B model that is about 2.4M trainable parameters, or roughly 33 MB for seven behaviors.

Because no base weights are modified, the behaviors remain separate. A small router can read the frozen hidden state and produce a per-token distribution over the behaviors in the bank, and the corresponding corrections can be blended by weight.

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

LARA attaches to the model and freezes everything else, so the same adapter mechanism can be used with the HF `Trainer`, TRL's `DPOTrainer` or `GRPOTrainer`, or a training loop of your own.

The training objective does not change the artifact. A behavior trained with supervised fine-tuning, DPO or reinforcement learning can enter the same MoB bank.

## Turn a behavior up or down

The correction is additive over the unchanged base, so its strength remains a runtime value:

```python
lara.gamma = 0.0     # the frozen base
lara.gamma = 0.5     # partial adaptation
lara.gamma = 1.0     # the trained behavior
```

This provides a simple way to control how strongly a learned behavior is applied at inference time.

## Mixture of Behaviors

A **MoB (Mixture of Behaviors)** is the bank of behaviors and the mechanism that selects or combines them over one frozen base model.

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

Each item is an independent behavior artifact. The MoB is the collection and runtime mixture; a single behavior is not itself a MoB.

Load several behaviors:

```python
from lara import Bank

bank = Bank(model, tokenizer)

bank.add("code", "behaviors/code")
bank.add("math", "behaviors/math")
bank.add("polite", "behaviors/polite")

bank.fit_router()

out = model.generate(**inputs)
```

`fit_router` trains a small classifier from the route samples stored with each behavior. A behavior can be added later and the router can be updated without retraining the existing behaviors.

## Routing

`top_k` controls how many behaviors contribute to each token:

```python
bank.top_k = None    # blend all of them by weight
bank.top_k = 2       # blend the two highest
bank.top_k = 1       # hard selection
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/LARA_readme_routing_dark.png">
  <img alt="Routing weight across a sentence: the mix shifts from one behavior to an other as the text is generated" src="media/figure3_per_token_routing.gif">
</picture>

Soft routing allows overlapping behaviors to contribute together. Hard routing selects one behavior per token.

Behaviors can also be explicitly pinned:

```python
with bank.pin("code"):
    out = model.generate(**inputs)

with bank.pin({"code": 1.0, "polite": 0.4}):
    out = model.generate(**inputs)

with bank.disabled():
    out = model.generate(**inputs)
```

A per-behavior scale can also be changed:

```python
bank.set_gamma("polite", 0.6)
```

The result is a system in which behavior selection and strength remain separate from the base model itself.

## Why MoBs matter

With conventional fine-tuning, specialization tends to produce another model:

```text
base model
   │
   ├── fine-tune → model A
   ├── fine-tune → model B
   └── fine-tune → model C
```

With LARA and MoBs:

```text
                  frozen base
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
         behavior   behavior   behavior
            │         │         │
            └─────────┼─────────┘
                      ▼
                   routing
```

The base capability is shared. The learned behaviors remain independent.

This becomes increasingly useful as the number of adaptations grows. Seven behaviors in the 1.5B example occupy about 33 MB of adapter storage rather than requiring seven copies of a multi-gigabyte model.

More importantly, behaviors need not be mutually exclusive. A system can combine them at inference time. A mathematical behavior could be used together with a tutoring behavior and a preferred writing style without training a new full model for that combination.

This is potentially a more general way to think about post-training: **capability in the foundation model, behavior in small learned modules**.

## Modular cognition

The MoB architecture can be viewed as a form of modular cognition.

Different learned behaviors can encode different ways of using the same underlying model:

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

The analogy with mixture-of-experts is useful, but the target is different. A conventional MoE routes among experts that are part of one model, primarily to increase model capacity. MoBs route among lightweight adaptations over a shared base, with the aim of making post-training modular.

That difference makes behaviors closer to software components than to additional copies of a model.

## Applications

The same architecture can be used wherever one model needs several modes of operation.

A local AI tutor could combine mathematics, tutoring and a student's preferred explanation style. A coding assistant could combine language-specific coding, debugging, testing and security-review behaviors. An enterprise model could share one base model across departments while keeping separate legal, finance, support and engineering adaptations. A personal AI could maintain a small bank of writing, task and preference behaviors.

These applications do not depend on local inference. The modularity also applies to server-side systems.

## Why this matters for small local models

Local AI changes the constraints around customization. A small language model running on a phone or PC has less room for long system prompts, extended conversation histories and large RAG contexts than a frontier model running in the cloud. Those techniques remain useful, but they can become an expensive way of telling a small model how it should behave.

LARA provides another way to express customization: learn the behavior once and keep it in a small residual-stream adaptation rather than spelling it out in a long prompt at every interaction. This is useful for both application-specific behavior and personalization. A model can acquire a user's preferred writing style, a tutor's teaching method, a company's house style, or a domain-specific way of reasoning without requiring all of that information to be reconstructed from the context window each time.

This does not replace RAG. RAG provides information the model does not have, while a behavior changes how the model uses the information it already has. The two can be complementary. The point is that behavior does not have to consume context.

## Small model, specialized behavior

A small base model can also become substantially more useful through domain adaptation. An informal test with **Qwen3-1.7B** illustrates the idea. The question was:

> Name a common over-the-counter pain reliever that reduces fever but does NOT increase bleeding risk.

The question is deliberately discriminative. Acetaminophen and ibuprofen both reduce pain and fever, but ibuprofen is an NSAID and carries a bleeding warning, whereas acetaminophen is not an NSAID. The intended answer was therefore **acetaminophen (Tylenol)**.

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

In this test, the adapted 1.7B model gave the same answer as a larger model used as a comparison. This is one example, not evidence that a 1.7B model generally performs like a larger model. It demonstrates the narrower point that a lightweight learned behavior can change the model's decision on a domain-specific distinction without changing the base model.

That suggests a useful local-AI pattern:

```text
small general model
        +
learned behavior
        ↓
specialized local model
```

The same pattern can apply to problem solving, coding, tutoring, writing style and other kinds of specialization.

## Local and edge AI

The small size of the behavior artifacts makes LARA particularly suitable for local and edge inference.

A device can carry one base model and a bank of specialized behaviors:

```text
base model
    +
math
medical
code
tutor
house style
personal behavior
```

A new specialization can then be distributed as a small adapter rather than another copy of the base model.

This is especially useful for devices where storage, memory and network access matter. It also opens the possibility of keeping personal adaptations on the device rather than sending them to a server.

The local AI use case is therefore one consequence of the architecture, rather than its definition.

## What a behavior looks like on disk

```text
behaviors/code/
  adapter.safetensors     the projections, a few MB
  config.json             layers, rank, alpha, base model id
  route_samples.jsonl     short texts typical of this behavior
```

The route samples allow a behavior to be routed without access to the data used for training.

A behavior records the base model it was trained against and will not load onto a different one.

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

Python 3.9 or later and PyTorch 2.0 or later are required. The examples also use `transformers`, `datasets`, `accelerate` and `trl`:

```bash
pip install -e ".[examples]"
```

## API

`LARA(model, layers=6, rank=128, alpha=128)` attaches one behavior to a frozen model.

- `lara.gamma` scales the behavior at inference time
- `lara.num_trainable()` reports its parameter count
- `lara.save(path, route_samples=..., method=...)` writes the behavior
- `LARA.from_pretrained(model, path)` attaches a saved behavior
- `lara.disabled()` temporarily runs the frozen base
- `lara.detach()` removes the adapters

`Bank(model, tokenizer, top_k=None)` holds several behaviors over one base.

- `bank.add(name, path)` adds a behavior
- `bank.remove(name)` drops one
- `bank.fit_router(mode="refit"|"append", steps=300)` trains the router
- `bank.top_k` controls how many behaviors apply per token
- `bank.pin(name_or_dict)` overrides the router
- `bank.disabled()` runs the frozen base
- `bank.set_gamma(name, g)` scales a behavior
- `bank.route_weights(input_ids)` reports routing weights
- `bank.save(path)`, `Bank.load(path, model, tokenizer)` save and reload a bank

## Examples

Runnable end to end on one GPU:

- `examples/01_finetune.py` trains a behavior with the HF `Trainer`
- `examples/02_dpo.py` trains a behavior with TRL's `DPOTrainer`
- `examples/03_bank.py` loads several behaviors onto one model and generates under soft routing, hard routing and pinning

Run 01 and 02 first, since 03 uses the artifacts they create.

## Tests

```bash
python tests/test_lara.py
```

The tests cover layer resolution, initialization, freezing, save/load, routing, `top_k`, pinning and incremental addition.

## Paper and experiments

The benchmark [code](https://github.com/pfekin/LARA/tree/main/research) produced the numbers in the [preprint](https://doi.org/10.48550/arXiv.2607.28669), with the [configuration and instructions](research.md) to rerun the experiments.

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
