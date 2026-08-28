# Reproducing the LARA experiments

This document contains the information needed to reproduce the experiments reported in the paper. The project overview, the LARA architecture, the MoB concept, local-AI motivation, and application examples are in [README.md](README.md).

## 1. Experiment scripts

The repository contains two main experiment scripts.

`lara.py` trains one adaptation with either LARA or LoRA and compares them at a matched parameter budget. It supports:

- supervised fine-tuning (`task="ft"`)
- preference optimization (`task="dpo"`)
- an inference-time `gamma` sweep

`routed.py` places several trained behaviors on one frozen base and evaluates hard and soft routing. It reports recovery, routing weights, and co-application.

## 2. Environment

The experiments run on a single NVIDIA T4 GPU with the model loaded in 8-bit.

Python 3.10+ is required.

```bash
pip install torch transformers peft datasets bitsandbytes accelerate
```

## 3. Base model and parameter matching

The matched-parameter fine-tuning and DPO experiments use `Qwen2.5-1.5B-Instruct`.

The LARA configuration used in the reported comparison has approximately 2.4M trainable parameters. The LoRA baseline is configured at approximately 2.2M trainable parameters.

In `lara.py`, the relevant configuration fields are:

```text
task           "ft" or "dpo"
arms           ["rc", "lora"]   # rc is the LARA arm
bridge_layers  layers receiving LARA modules
rank           LARA rank
alpha          LARA scale
lora_rank      LoRA rank
lora_target    LoRA target modules
```

The LoRA settings should remain matched to the LARA trainable-parameter count when reproducing the paper comparison.

## 4. Fine-tuning experiment

Set:

```python
task = "ft"
```

in `lara.py`, then run:

```bash
python lara.py
```

The script trains the configured LARA and LoRA arms and reports the evaluation perplexity together with the trainable parameter counts.

The reported result uses a matched budget of roughly 2.4M trainable parameters for LARA and 2.2M for LoRA on Qwen2.5-1.5B-Instruct. The paper reports comparable fine-tuning perplexity at that budget.

## 5. DPO experiment

Set:

```python
task = "dpo"
```

in `lara.py`, then run:

```bash
python lara.py
```

The DPO experiment uses the configured preference data and the same matched-parameter LARA/LoRA comparison.

The paper reports comparable DPO reward accuracy for LARA and LoRA at approximately matched trainable-parameter counts.

## 6. Inference-strength sweep

`lara.py` also evaluates the trained LARA behavior at different values of `gamma`.

The interpretation is:

```text
gamma = 0       base model
gamma = 1       normal trained behavior
other values    scaled behavior
```

The reported sweep shows smooth interpolation between the base and adapted model as the behavior strength changes.

## 7. Routed MoB experiment

`routed.py` loads several separately trained behaviors onto one frozen base. The reported routed setup contains seven behaviors.

Each behavior has its own adapter artifact and route samples. The router is fitted from those route samples.

The important configuration field is:

```python
route_mode = "soft"     # blend behaviors
# or
route_mode = "top1"     # one behavior per token
```

`top_k` controls how many behaviors can contribute to a token:

```text
top_k = 1       hard routing
top_k > 1       top-k mixture
None            all behaviors participate according to weight
```

Run:

```bash
python routed.py
```

The script reports recovery, routing weights and co-application. The seven behaviors share one frozen base model; together their adapters occupy about 33 MB, compared with roughly 21 GB for seven separate 3 GB models.

## 8. Reinforcement-learning experiment

A separate GRPO experiment uses `Qwen3-1.7B`.

The reported LARA configuration has:

```text
rank                     128
trainable parameters     530k
adapter size             2.2 MB
```

The LoRA comparison has:

```text
trainable parameters     8.7M
adapter size             34.9 MB
```

The reward checks formatting constraints. The reported result is:

```text
all constraints satisfied:
LARA    28%
LoRA    29%
base     2%
```

The evaluation set is small, so the result does not establish a statistically meaningful difference between LARA and LoRA. It demonstrates that the LARA representation can also be trained with a reinforcement-learning objective at substantially lower trainable parameter count.

Notebook: [Qwen3-1.7B GRPO](examples/reinforcement_learning/Qwen3-1.7B/grpo.ipynb)

## 9. Data

The experiments pull public datasets from the Hugging Face Hub. The repository also contains the code corpus under `data/`.

The datasets used by the scripts include:

- a code corpus
- Databricks Dolly
- WikiText
- UltraFeedback
- the domain datasets used for routed behaviors

The exact dataset names, task definitions and sampling settings are controlled in the configuration at the top of the scripts.

## 10. Self-tests

The scripts provide basic self-tests:

```bash
python lara.py selftest
python routed.py selftest
```

These check the implementation without running the full experiments.

## 11. Paper results

The main reported findings are:

- LARA and LoRA reach comparable fine-tuning perplexity at approximately matched trainable-parameter counts on Qwen2.5-1.5B-Instruct.
- LARA and LoRA reach comparable DPO reward accuracy at approximately matched trainable-parameter counts on Qwen2.5-1.5B-Instruct.
- The LARA `gamma` parameter provides continuous inference-time control over behavior strength.
- Seven behaviors can share one frozen 3 GB base for about 33 MB of adapter storage and can be routed per token with hard or soft routing.
- In the reported Qwen3-1.7B GRPO experiment, a 530k-parameter LARA behavior reached a similar constraint-satisfaction result to an 8.7M-parameter LoRA behavior, while using substantially less trainable and stored parameter capacity.

### Memory footprint

Sections 4 and 5 hold LARA and LoRA at the same parameter budget by construction, so those comparisons say nothing about size.

The size difference shows up in two other places.

The first is per behavior. The number of insertion points depends on the training objective.

| objective | insertion points | LARA | LoRA |
|---|---|---|---|
| fine-tuning | six layers | 2.4M parameters | 2.2M, matched |
| DPO, GRPO | one (middle) layer | 530k parameters, 2.2 MB | 8.7M parameters, 34.9 MB |

The second is per bank. The base model is shared rather than copied, so a behavior does not carry the model it adapts. Seven behaviors occupy about 33 MB over one frozen 3 GB base, against roughly 21 GB for seven separately fine-tuned models.

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
