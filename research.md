# LARA

**Lightweight Additive Residual Adaptation**: a residual-stream adapter for frozen language models.

LARA adapts a frozen model by reading the hidden state at a small set of layers and adding a low-rank correction back to the residual stream. The base weights are never changed. On a code fine-tuning task and on preference optimization (DPO), LARA matches LoRA at equal parameter counts. Because each adapted behavior is a small module over a shared frozen base, many behaviors can be held resident at once and routed per token.

This repository contains the code for the experiments in the accompanying paper (link to follow).

## Why this matters for local AI

Running a capable model on a phone, a laptop, or an edge device hits a wall made of memory. One fine-tuned model is already large, and most real products need more than one behavior: a coding helper, a summarizer, a medical assistant, each with its own tone or safety rules. Shipping a separate fine-tuned model for every behavior multiplies the footprint until it stops fitting on the device.

LARA takes a different route by reusing weights. The base model is loaded once, frozen, and shared across every behavior. Each behavior is a thin module that reads the shared residual stream and adds a small correction, and a small router decides token by token which behavior to apply. Another behavior costs a few megabytes, not another copy of the model. Seven behaviors fit on one 3 GB base for about 33 MB of adapters, against roughly 21 GB for seven separate models.

Because the router works per token, behaviors are not a switch the user flips. The device blends them on the fly: a coding answer written in your product's voice, a medical reply that follows your safety rules, a summary in a house style. A fine-tuned skill and a preference-tuned behavior live on the same frozen base and can apply together on the same token.

### A use case

Picture an assistant that runs entirely on the device, offline and private, with no round trip to a server. It ships with one frozen base and a small bank of adapters for code, math, medical, summarization, and a house style. It answers each query with the right specialist, chosen per token, and stays within a few gigabytes. To add a skill, adjust the tone, or fix a behavior, you ship a small adapter rather than a new model. The base never moves, so an update is megabytes over the wire instead of a full model download.

### A different angle on mixture-of-experts

Mixture-of-experts models showed that routing each token to a few specialized modules works well. LARA borrows the routing idea and aims it at a different problem. An MoE routes among experts inside one large model to add capacity, mostly in the datacenter. LARA routes among lightweight adapters over one small frozen base to make many behaviors cheaply available at the edge. Its experts are megabyte-scale corrections that share the same frozen weights, so one more behavior stays cheap and the base is never duplicated. This does not make a model larger or more capable. It makes one small model wear many hats, on hardware where a stack of full models or a large MoE would never fit.

## Contents

`lara.py` trains a single behavior with LARA or LoRA and compares them at matched parameters, for fine-tuning (`task="ft"`) or preference optimization (`task="dpo"`). It also sweeps the scale gamma applied at inference.

`routed.py` places several behaviors on one frozen base and routes among them per token, hard or soft. It reports recovery, routing weights, and the co-application readout.

## Requirements

Python 3.10 or later and a CUDA GPU. The experiments run on a single T4 in 8-bit.

```
pip install torch transformers peft datasets bitsandbytes accelerate
```

## Usage

Both scripts are driven by a configuration dictionary at the top of the file. Edit it, then run the file directly.

```
# lara.py: set task="ft" for fine-tuning, task="dpo" for preference optimization
python lara.py

# routed.py: set route_mode="soft" (blend) or "top1" (one behavior per token)
python routed.py
```

Each script prints the trainable parameter counts for both methods and the result tables. Self-tests are available:

```
python lara.py selftest
python routed.py selftest
```

## Configuration

Main fields in `lara.py`:

- `task`: `"ft"` or `"dpo"`
- `arms`: methods to run, `["rc", "lora"]` (the `rc` arm is LARA)
- `bridge_layers`: the layers at which LARA inserts modules
- `rank`, `alpha`: LARA rank and scale
- `lora_rank`, `lora_target`: the LoRA baseline, set to match LARA's parameter count

Main fields in `routed.py`:

- `route_mode`: `"top1"` or `"soft"`
- the behavior list: each entry names a dataset and its task (fine-tuning or DPO)

## Data

The scripts pull public datasets from the Hugging Face Hub: a code corpus, Databricks Dolly, WikiText, UltraFeedback, and the domain sets used for routing. The code corpus is included under `data/`.

## Results

At a matched budget of roughly 2.4M trainable parameters against 2.2M for LoRA, LARA reaches comparable fine-tuning perplexity and comparable DPO reward accuracy on Qwen2.5-1.5B-Instruct. The scale gamma, applied at inference, interpolates smoothly between the base and the adapted model. Seven behaviors sit on one frozen 3 GB base for about 33 MB of adapters and route per token. See the paper for the full tables.

## Citation

```
@misc{lara2026,
  title        = {LARA: Lightweight Adapters in the Residual Stream for
                  Composable Adaptation and Alignment on Frozen Models},
  author       = {Ekin, Pascal},
  year         = {2026},
  note         = {Preprint}
}
```

## License

Add a license of your choice (for example, MIT).
