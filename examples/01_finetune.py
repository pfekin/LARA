"""Train a behavior with cross-entropy, using the stock HF Trainer.

The only LARA-specific lines are the three marked below. Everything else is
ordinary HF training: LARA does not wrap, replace, or subclass the trainer.

    pip install torch transformers datasets accelerate
    python examples/01_finetune.py
"""
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

from lara import LARA

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "behaviors/code"
MAX_LEN = 512

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")

# ── LARA (1/3): attach adapters to the frozen base
# `layers=6` spreads six modules evenly over the depth. On a 28-layer model that
# is [4, 8, 12, 16, 20, 24]. Pass a list if you want exact control.
lara = LARA(model, layers=6, rank=128, alpha=128)
print(lara, f"\nbase frozen, {lara.num_trainable():,} trainable parameters")

raw = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")


def to_text(ex):
    msgs = [{"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}


def tokenize(ex):
    return tok(ex["text"], truncation=True, max_length=MAX_LEN)


ds = raw.map(to_text).map(tokenize, batched=True, remove_columns=raw.column_names)

# ── any trainer works: LARA only requires that the base stays frozen
trainer = Trainer(
    model=model,                      # the model itself; LARA rides along inside it
    args=TrainingArguments(
        output_dir="runs/code", max_steps=600, learning_rate=2e-4,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        logging_steps=50, save_strategy="no", bf16=True, report_to=[],
    ),
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
trainer.train()

# ── LARA (2/3): save the behavior
# `route_samples` are short texts typical of this behavior. A Bank uses them to
# fit its router later, so the behavior can be shared and still routed without
# anyone needing this training set again.
#
# Use the outputs, not the instructions. The instructions in this corpus are
# English questions about code, so a router fitted on them would be separating
# two sets of English prose and could latch onto something incidental. The
# outputs are the code itself, which is what distinguishes this behavior.
samples = [ex["output"] for ex in raw.select(range(200))]
lara.save(OUT, route_samples=samples, method="ce")
print(f"wrote {OUT}")

# put the model back into an inference state before generating
model.gradient_checkpointing_disable()
model.config.use_cache = True
model.eval()

# ── LARA (3/3): the scale is a runtime knob, no retraining
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a function that reverses a linked list."}],
    tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(model.device)

# gamma is continuous, but greedy decoding takes an argmax at every step, so
# nearby values often decode to identical text. The two ends show the shift.
# To see the middle, read the logits (below) or sample instead of decoding greedily.
for g in (0.0, 1.0):
    lara.gamma = g                    # 0.0 is the untouched base
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    print(f"\n─── gamma={g} " + "─" * 40)
    print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))

# the knob is continuous underneath: next-token entropy moves with gamma even
# where the greedy path does not
print()
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    lara.gamma = g
    with torch.no_grad():
        lp = torch.log_softmax(model(**ids).logits[0, -1], dim=-1)
    print(f"gamma={g:<5} top={tok.decode(lp.argmax())!r:<12} "
          f"logprob={lp.max():.3f}  entropy={-(lp.exp() * lp).sum():.3f}")
