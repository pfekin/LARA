"""Align a behavior with DPO, using TRL's DPOTrainer.

Same three LARA lines as the fine-tuning example, a different objective. The
artifact this produces is indistinguishable from a fine-tuned one: a directory
of projections that a Bank can route alongside any other behavior.

One perk of a frozen base: DPO needs a reference model, and the reference here
is just the base with the modules switched off. No second copy in memory.

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

# ref_model=None makes TRL disable the adapters for the reference pass, which is
# exactly the frozen base, so the reference costs nothing.
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=DPOConfig(
        output_dir="runs/polite", max_steps=60, learning_rate=2e-4,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        beta=5.0, max_prompt_length=128, max_length=256,
        logging_steps=10, save_strategy="no", bf16=True, report_to=[],
    ),
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()

# ── LARA (2/3): save, same artifact shape as any other behavior ──────────────
lara.save(OUT, route_samples=[ex["prompt"] for ex in ds.select(range(200))], method="dpo")
print(f"wrote {OUT}")

# ── LARA (3/3): dial the alignment strength at inference ─────────────────────
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
