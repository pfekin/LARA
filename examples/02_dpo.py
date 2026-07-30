"""Align a behavior with DPO, using TRL's DPOTrainer.

Same three LARA lines as the fine-tuning example, a different objective. The
artifact this produces is indistinguishable from a fine-tuned one: a directory
of projections that a Bank can route alongside any other behavior. That is what
this example is for.

It is not a reproduction of the paper. The paper's DPO numbers come from the
harness in research/, which uses its own loss with an NLL anchor term that TRL
does not apply, and settings tuned for it. Run at these settings on 512 pairs,
the training loss falls a long way while held out accuracy barely moves, which
is what overfitting a small preference set looks like. Treat the numbers below
as a check that the modules trained, not as a result.

DPO needs a reference model. Because the modules are zero-initialized, an
untrained LARA model is exactly the base, so the reference log probabilities can
be precomputed before the first step instead of loading a second copy of the
model. See the comment on DPOConfig below for when that shortcut does not hold.

Written against trl 1.9. TRL renames trainer arguments fairly often, so if this
fails on an argument name, check the DPOConfig fields for your version.

    pip install torch transformers trl datasets accelerate
    python 02_dpo.py
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from lara import LARA

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "behaviors/polite"

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")

# ── LARA (1/3): attach ───────────────────────────────────────────────────────
# Preference optimization is less demanding about placement than fine-tuning:
# a single middle module reaches parity with six. `layers=1` resolves to [14]
# on a 28-layer model.
lara = LARA(model, layers=1, rank=128, alpha=128)
print(lara, f"\nbase frozen, {lara.num_trainable():,} trainable parameters")

ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:512]")


def to_pairs(ex):
    return {"prompt": ex["prompt"],
            "chosen": ex["chosen"][-1]["content"],
            "rejected": ex["rejected"][-1]["content"]}


ds = ds.map(to_pairs, remove_columns=ds.column_names)

# The reference model is the frozen base. Rather than load a second copy of it,
# precompute the reference log probabilities before training starts: at that
# point the modules are still zero-initialized, so the model *is* the base.
# (That holds when training a fresh behavior, as here. If you continue training
# an existing one, the modules are no longer zero and this shortcut is wrong.)
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=DPOConfig(
        output_dir="runs/polite", max_steps=60, learning_rate=2e-4,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        beta=5.0, max_length=256, precompute_ref_log_probs=True,
        logging_steps=10, save_strategy="no", bf16=True, report_to=[],
    ),
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()

# ── LARA (2/3): save, same artifact shape as any other behavior ──────────────
lara.save(OUT, route_samples=[ex["prompt"] for ex in ds.select(range(200))], method="dpo")
print(f"wrote {OUT}")

# DPOConfig turns gradient checkpointing on and the KV cache off, and the trainer
# leaves the model in training mode. Generating in that state produces garbage,
# so put the model back into an inference state first.
model.gradient_checkpointing_disable()
model.config.use_cache = True
model.eval()

# ── LARA (3/3): what DPO actually moved ──────────────────────────────────────
# The margin between chosen and rejected is what DPO optimizes, and it is
# continuous. Reward accuracy thresholds it at zero, so a real shift in the
# margin can leave accuracy untouched if no pair crosses over.
import torch.nn.functional as F

eval_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                       split="test_prefs[:128]").map(to_pairs, remove_columns=None)


@torch.no_grad()
def seq_logprob(prompt, completion, normalize=False):
    p = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    full = tok(prompt + completion, return_tensors="pt").input_ids.to(model.device)
    logits = model(full).logits[:, :-1]
    tgt = full[:, 1:]
    lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    lp = lp[:, p.shape[1] - 1:]                     # completion tokens only
    return (lp.mean() if normalize else lp.sum()).item()


# TRL's DPO sums log probabilities over the completion. The paper's harness
# normalizes by length instead, which divides the same shift by a hundred or so
# tokens, so report both and compare like with like.
for g in (0.0, 1.0):
    lara.gamma = g
    summed = [seq_logprob(ex["prompt"], ex["chosen"])
              - seq_logprob(ex["prompt"], ex["rejected"]) for ex in eval_ds]
    normed = [seq_logprob(ex["prompt"], ex["chosen"], normalize=True)
              - seq_logprob(ex["prompt"], ex["rejected"], normalize=True) for ex in eval_ds]
    acc = sum(m > 0 for m in summed) / len(summed)
    print(f"gamma={g}  margin (summed) {sum(summed) / len(summed):+.3f}  "
          f"(per token) {sum(normed) / len(normed):+.4f}  accuracy {acc:.3f}")

# a sample generation, for a qualitative look
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "My code crashed again. What now?"}],
    tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(model.device)

for g in (0.0, 1.0):
    lara.gamma = g
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    print(f"\n─── gamma={g} " + "─" * 40)
    print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
