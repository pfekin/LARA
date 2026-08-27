# LARA: Lightweight Additive Residual Adaptation

**LARA** is a post-training method for adapting a frozen language model with small low-rank corrections in the residual stream.

The work began from a familiar foundation-model idea: train a general model once, then tailor it afterwards. LARA is intended to make that adaptation layer lighter and more modular, so that specialization does not have to produce another copy of the model.

The broader system is **Mixture of Behaviors (MoBs)**: a collection of independently trained behaviors that share one frozen base and can be selected or combined at inference time.

![LoRA vs LARA](media/figure1_lora_vs_lara.svg)

*LoRA adapts in weight space. LARA adapts in the residual stream. The base block stays frozen while a low-rank correction is read from the stream and added back.*

## 1. The adaptation problem

Foundation models are useful because the same learned representation can support many applications. Post-training then provides a way to specialize that general capability.

LoRA made this economical by learning low-rank changes to a frozen model. A LoRA adapter, however, naturally becomes a distinct adaptation of the base. Once several adaptations are required, the deployment problem becomes one of maintaining and selecting among multiple model variants.

LARA separates the adaptation from the base in a different way. Instead of modifying weight matrices, it adds a low-rank correction to the residual stream:

```python
h = h + gamma * (alpha / rank) * up(down(layer_norm(h)))
```

The base weights remain frozen. A new adapter starts as a no-op because its `up` projection is initialized to zero.

## 2. LARA as an adaptation mechanism

At rank 128 over six layers of a 1.5B model, the reported configuration has about 2.4M trainable parameters.

The important property is not only parameter count. The resulting adapter remains a separate artifact:

```text
base model
    +
small behavior artifact
```

The runtime strength `gamma` can then change how strongly the correction is applied:

```text
gamma = 0       frozen base
gamma = 0.5     partial adaptation
gamma = 1.0     trained behavior
```

The experiments show that this produces a continuous control parameter rather than fixing one adaptation strength permanently at training time.

## 3. MoBs: Mixture of Behaviors

The central extension is from one adaptation to a bank of adaptations.

A **behavior** is one learned LARA module.

A **MoB** is the mixture of those behaviors and the mechanism used to select or combine them.

For example:

```text
                  frozen base
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
       code          math        medical
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                   MoB/router
```

An MoB can contain behaviors trained with different objectives. A code behavior can come from supervised fine-tuning, another from DPO, another from GRPO. Their training method does not need to determine how they are deployed.

This creates a useful separation:

```text
capability        = base model
behavior          = learned adaptation
selection         = router
strength          = runtime scale
combination       = composition
```

The model becomes a shared substrate and the behaviors become modular post-training components.

## 4. Routing and composition

The `Bank` runtime can route behavior at token level.

With soft routing, several behaviors contribute according to the router weights. With hard routing, one behavior is selected. Pinning bypasses routing and explicitly chooses a behavior or a chosen mixture.

```text
soft:
code       0.60
math       0.10
polite     0.30

hard:
code       1.00

pinned:
code       1.00
polite     0.40
```

The same base model can therefore carry several behaviors without merging them into the weights.

This also permits combinations that were not separately trained as full models. A mathematical behavior can be combined with a tutoring behavior, or a coding behavior with a house style, without producing a new copy of the base.

## 5. Training with SFT, DPO and RL

LARA does not depend on one post-training objective.

The repository contains experiments using:

- supervised fine-tuning;
- preference optimization with DPO;
- reinforcement learning with GRPO.

The behavior artifact has the same form after training and can enter the same MoB bank.

### Matched-parameter fine-tuning and DPO

On Qwen2.5-1.5B-Instruct, the reported matched-parameter experiments use roughly 2.4M trainable parameters for LARA against 2.2M for LoRA. The reported fine-tuning perplexity and DPO reward accuracy are comparable between the two methods.

### Reinforcement learning

A GRPO experiment on Qwen3-1.7B compared a 530k-parameter LARA behavior with an 8.7M-parameter LoRA. The LARA result was within noise of the LoRA result on the reported rule-checking evaluation, while using far fewer trainable parameters.

The evaluation set was small, so this result does not establish superiority. It does show that a small residual-stream behavior can learn a reinforcement-learning objective at a much lower parameter count.

## 6. Seven behaviors on one base

The routed experiments place seven behaviors on one frozen 3 GB base for about 33 MB of adapter storage.

This changes the scaling problem:

```text
seven separate adapted models
≈ 7 × base model

one base + seven behaviors
≈ base model + tens of MB
```

The benefit grows with the number of adaptations because the base is shared.

More importantly, the seven behaviors are not necessarily seven mutually exclusive model choices. The router can combine them at inference time.

## 7. Small models, context and learned behavior

Local AI changes the constraints around customization. A small language model running on a phone or PC has less room for long system prompts, extended conversation histories and large RAG contexts than a frontier model running in the cloud. RAG remains useful when a system needs external or current information, but prompting can also become a significant part of the machinery used to make a small model behave in a particular way.

LARA provides another route. A behavior can be learned once and represented as a small residual-stream correction rather than described again in a long prompt at every interaction. This makes behavior and knowledge separate concerns: RAG can supply information, while a learned behavior can change how the model uses that information.

This matters for customization and personalization. A local model could carry behaviors for a particular domain, application, user or organization without requiring all of those preferences to consume context on every request.

## 8. Small model, specialized behavior

An informal medical test illustrates why a small local model plus a learned behavior can be interesting. The base was **Qwen3-1.7B**, a 1.7-billion-parameter dense language model.

The prompt was:

> Name a common over-the-counter pain reliever that reduces fever but does NOT increase bleeding risk.

The distinction is deliberate. Acetaminophen and ibuprofen both reduce pain and fever, but ibuprofen is an NSAID and carries a bleeding warning, whereas acetaminophen is not an NSAID. The intended answer was **acetaminophen (Tylenol)**.

With all behaviors disabled, the model produced:

```text
A common over-the-counter (OTC) pain reliever that reduces and does
not significantly increase bleeding risk is ibuprofen...
```

With the medical behavior enabled, the same 1.7B model produced:

```text
Acetaminophen (Tylenol) is a common over-the-counter pain reliever
that reduces fever but does NOT increase bleeding risk.
```

In this test, the adapted 1.7B model gave the same answer as a larger model used as a comparison. The result should not be read as evidence that LARA makes a 1.7B model generally equivalent to a larger model. It demonstrates a more specific property: a small domain behavior can change which of several plausible answers the model selects on a domain-specific question.

The base model already contains general language and world knowledge. The behavior changes how that capability is expressed in a particular domain. That is the relevant proposition for small local models:

```text
small general model
        +
learned behavior
        ↓
specialized local model
```

The same idea can apply to problem solving, coding, tutoring, writing style and other forms of specialization.

## 9. Modular cognition

A mixture of behaviors suggests a different interpretation of post-training.

A foundation model supplies broad capability. Post-training can then encode different policies for using that capability:

```text
                 base capability
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      planner        critic        verifier
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                    response
```

A coding system could use separate behaviors for implementation, debugging, testing and security review. An educational system could combine mathematical problem solving with Socratic tutoring and a student's preferred explanation style. A personal assistant could retain writing and task preferences independently of the base model.

This resembles modular cognition more than conventional fine-tuning because the learned modes remain separable and can be assembled at inference time.

The comparison with mixture-of-experts is useful but limited. MoE routes among experts that are part of one model. MoBs route among lightweight adaptations over one shared base, with the purpose of modular post-training rather than increasing the base model's parameter capacity.

## 10. Local and edge AI

The same modularity has a practical consequence for local inference.

Running a capable model on a phone, laptop or edge device makes model memory important. If every specialization requires another full model, the storage cost quickly becomes dominant.

With LARA:

```text
one frozen base
      +
many small behaviors
      +
router / composition
```

Adding a new specialization can therefore mean adding a few megabytes rather than another copy of the base.

This supports a local deployment model in which a device can keep a bank of behaviors for different tasks and update those behaviors independently.

An offline tutor is one example. A phone could use a local vision pipeline to read a photographed problem, a local language model for reasoning and explanation, and an MoB to control tutoring behavior, subject specialization or the student's preferred style.

Local AI is therefore an important application of the architecture, but the architecture also applies to conventional server inference.

## 11. A possible software architecture for model behavior

The repository suggests treating learned behaviors as deployable components:

```text
base model
    │
    ├── install behavior
    ├── remove behavior
    ├── update behavior
    ├── scale behavior
    ├── compose behaviors
    └── route behaviors
```

A behavior artifact contains the learned projections, its configuration and route samples:

```text
behaviors/code/
  adapter.safetensors
  config.json
  route_samples.jsonl
```

That makes a behavior closer to a software component than to another model checkpoint.

It also suggests a possible distribution model in which the base model becomes the platform and behaviors become small installable modules.

## 12. Experimental limits

The reported experiments establish several properties of the current implementation:

- LARA reaches comparable results to matched-parameter LoRA on the reported Qwen2.5-1.5B fine-tuning and DPO tasks.
- A GRPO-trained LARA behavior reached a similar reported score to a much larger LoRA configuration, although the evaluation was too small for a statistical distinction.
- Multiple behaviors can share one frozen base and be routed using soft or hard selection.
- Runtime scaling changes the strength of the learned behavior.

Other ideas remain broader hypotheses. In particular, robust claims about large behavior banks, extensive composition, cross-objective transfer, or the quality of arbitrary routing strategies require larger controlled experiments.

## 13. Current research direction

The central research question is whether post-training can become modular.

A useful system should allow behaviors to be learned independently, remain separate from the base model, coexist with other behaviors, and be selected or combined at inference time.

That leads to questions about:

- interference as the behavior bank grows;
- the conditions under which soft routing beats hard routing;
- composition between behaviors trained with different objectives;
- behavior transfer across tasks and domains;
- the relationship between adapter size and behavioral specificity;
- efficient deployment of behavior banks on local and edge devices.

The underlying premise is simple:

> Keep the foundation model general. Put adaptation into small learned behaviors.

LARA provides the adaptation mechanism. MoBs provide the architecture for using many such adaptations around one shared model.

## Contents

`benchmark.py` trains a single behavior with LARA or LoRA and compares them at matched parameters for fine-tuning or preference optimization. It also sweeps the inference scale `gamma`.

`benchmark_routed.py` places several behaviors on one frozen base and routes among them per token, hard or soft. It reports recovery, routing weights and co-application.

## Requirements

Python 3.10 or later and a CUDA GPU. The experiments run on a single T4 in 8-bit.

```bash
pip install torch transformers peft datasets bitsandbytes accelerate
```

## Usage

Both scripts are driven by configuration dictionaries at the top of the file.

```bash
# benchmark.py: set task="ft" or task="dpo"
python benchmark.py

# benchmark_routed.py: set route_mode="soft" or "top1"
python benchmark_routed.py
```

Self-tests are available:

```bash
python benchmark.py selftest
python benchmark_routed.py selftest
```

## Configuration

Main fields in `benchmark.py`:

- `task`: `"ft"` or `"dpo"`
- `arms`: methods to run, `["rc", "lora"]` where `rc` is LARA
- `bridge_layers`: layers at which LARA inserts adapters
- `rank`, `alpha`: LARA rank and scale
- `lora_rank`, `lora_target`: the LoRA baseline, set to match LARA's parameter count

Main fields in `benchmark_routed.py`:

- `route_mode`: `"top1"` or `"soft"`
- behavior list: each entry names a dataset and its task

## Data

The scripts pull public datasets from the Hugging Face Hub: a code corpus, Databricks Dolly, WikiText, UltraFeedback, and the domain sets used for routing. The code corpus is included under `data/`.

## Results

At a matched budget of roughly 2.4M trainable parameters against 2.2M for LoRA, LARA reaches comparable fine-tuning perplexity and comparable DPO reward accuracy on Qwen2.5-1.5B-Instruct.

A separate GRPO experiment on Qwen3-1.7B used 530k trainable LARA parameters against 8.7M LoRA parameters. The reported constraint-satisfaction score was within noise of the LoRA result.

The inference scale `gamma` interpolates between the base and adapted model.

Seven behaviors can sit on one frozen 3 GB base for about 33 MB of adapters and route per token.

## Observations outside the benchmarks

### Reinforcement learning

A behavior was trained with GRPO against a rule checker, with reward equal to the proportion of formatting constraints satisfied. On Qwen3-1.7B, a single rank-128 LARA adapter reached 28% of prompts with every constraint satisfied, compared with 29% for LoRA using rank 8 across seven target modules. LARA used 530k trainable parameters and 2.2 MB on disk, versus 8.7M parameters and 34.9 MB for LoRA. The base model reached 2%.

LARA also exposes a runtime strength parameter. In this experiment, increasing the strength improved performance on the trained constraints up to 2.0, while held-out constraints peaked at lower strength. This provides direct control over how strongly the learned behavior is applied at inference time.

Notebook: [Qwen](https://github.com/pfekin/LARA/blob/main/examples/reinforcement_learning/Qwen3-1.7B/grpo.ipynb)

### Thai

A behavior trained for Thai works well in interactive use. This was an informal trial rather than a benchmark run, and the configuration was not recorded.

Perplexity does not capture it well. A proper claim would need blind pairwise preference testing by native speakers.

Thai is already partly represented in the base, so the correction has something to act on. A low-rank correction is not expected to install a language that the base does not represent.

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
```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
