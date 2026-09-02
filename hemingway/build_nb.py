"""Build hemingway_style.ipynb."""
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
# Teaching a model to write like Hemingway

Without changing the model.

This notebook trains two **behaviors** on a frozen Qwen3.5-4B: one with supervised
fine-tuning, one with DPO. Each is a few megabytes. The base weights are never
touched, so with the behaviors switched off the model is bit-for-bit what it was
before, and a scale called gamma turns the style up and down at inference.

At the end there is a small command line to try it: same prompt, different
settings, side by side.

**Runtime.** Training wants an L4 or better. Inference alone runs on a T4.

**Just want to try it?** Run every cell. `TRAIN` is off by default, so the data
and training sections skip themselves and section 5 downloads behaviors that are
already trained. No account, no token, nothing to configure.

**Want to train your own?** Set `TRAIN = True` in the configuration cell. That
wants an L4 and, if you also want to upload the result, a write token.

**Texts.** Downloaded from Project Gutenberg. Anything Gutenberg hosts is public
domain in the United States. Copyright terms differ elsewhere: in the UK and the
EU the term is life plus seventy, so Hemingway is protected until 2032 while
Gertrude Stein and F. Scott Fitzgerald are clear. Run whichever you like locally;
if you intend to publish or distribute a trained behavior, use an author who is
public domain where you are.
""")

code("""
!pip -q install transformers accelerate datasets safetensors kernels
!pip -q install git+https://github.com/pfekin/LARA.git

import importlib, os, sys, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for mod in ("transformers", "lara", "datasets", "kernels"):
    importlib.import_module(mod)

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    BF16 = cap[0] >= 8                      # Ampere and later; a T4 is 7.5
    print(f"{name}, compute {cap[0]}.{cap[1]}, {total:.0f} GB, bf16 {BF16}")
else:
    BF16 = False
    print("No GPU. Runtime > Change runtime type > GPU.")


def delta_rule_ms(base, layers=8, seq=512):
    # Time it rather than inspect it. The kernels wrapper replaces the fallback
    # function in place, keeping its name, so checking whether
    # torch_chunk_gated_delta_rule was called cannot tell the two apart.
    # Under ~150 ms for 8 layers at 512 tokens is the CUDA kernel; the python
    # loop is several hundred.
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
        from transformers import AutoConfig
    except Exception:
        return None                      # not a hybrid model, nothing to time

    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    t = getattr(cfg, "text_config", cfg)
    if isinstance(getattr(t, "layer_types", None), (list, tuple)):
        t.layer_types = t.layer_types[:layers]
    t.num_hidden_layers = layers
    if hasattr(cfg, "vision_config"):
        cfg.vision_config.num_hidden_layers = 1

    net = m.Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).cuda().eval()
    ids = torch.randint(1, 1000, (1, seq)).cuda()
    with torch.no_grad():
        net(input_ids=ids)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            net(input_ids=ids)
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 5 * 1000
    del net, ids
    torch.cuda.empty_cache()
    return ms


# huggingface_hub asks Colab for an HF_TOKEN secret even when only reading
# public repositories. Loading needs no token; silence the notice.
import warnings
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
""")

md("""
## Configuration

Works are given by Gutenberg id rather than title, since title search is
unreliable. The id is the number in the ebook URL. Swap the block for another
author and the rest of the notebook is unchanged.
""")

code('''
# Off by default: load behaviors that are already trained and try them out.
# Turn it on to build the dataset and train your own, which wants an L4 and
# a write token.
TRAIN = False

AUTHOR = "Hemingway"

# Gutenberg ids. The number in the ebook URL: gutenberg.org/ebooks/67138 -> 67138.
MANUAL_IDS = {
    "In Our Time":                 61085,
    "The Sun Also Rises":          67138,
    "Men Without Women":           69683,
    "A Farewell to Arms":          75201,
    "Three Stories and Ten Poems": 59603,
}

# Public-domain alternatives, clear in the UK and the EU. Look the ids up on
# gutenberg.org and replace the block above.
#   Stein: Three Lives, Tender Buttons, Geography and Plays
#   Fitzgerald: This Side of Paradise, The Beautiful and Damned, Tales of the Jazz Age

BASE = "Qwen/Qwen3.5-4B"
FALLBACK_BASE = "Qwen/Qwen3-4B"    # if the linear-attention kernel is missing

SLUG = AUTHOR.lower()

N_PASSAGES = 800          # training examples; a few hundred is already enough
PASSAGE_TOKENS = (120, 320)
MAX_LEN = 448             # instruction plus passage
MICRO, ACCUM = 4, 4       # effective batch 16; peak memory follows MICRO

SFT_LAYERS, SFT_STEPS = 6, 200      # skills want several insertion points
DPO_LAYERS, DPO_STEPS = 1, 200      # preference reaches the same from one
RANK = 128

HF_REPO = "pfekin/mobs-hemingway"   # public: loading needs no token.
                                    # It holds every author, not just one; the
                                    # folders inside are named <author>_<method>.
PUSH = TRAIN                        # uploading does

import os, re, json, random, time, gc, textwrap
import torch, torch.nn.functional as F
random.seed(0); torch.manual_seed(0)
DT = torch.bfloat16 if BF16 else torch.float16
os.makedirs("data", exist_ok=True); os.makedirs("behaviors", exist_ok=True)
print(f"{AUTHOR}, base {BASE}, {DT}")
print("TRAIN is on: sections 1 to 4 will build a dataset and train"
      if TRAIN else
      "TRAIN is off: run the setup cell, then go to section 5")
''')

md("""
## Setup

Every function the notebook uses, in one place, and the tokenizer.

Sections 1 to 4 below build a dataset and train the behaviors. With `TRAIN` off
they print a line and do nothing, so running the whole notebook top to bottom
goes straight to section 5.
""")

code('''
from transformers import AutoModelForCausalLM, AutoTokenizer

# A behavior records the base it was trained against and will not load onto a
# different one. When loading rather than training, the artifact decides which
# base to use, not the configuration above.
if not TRAIN:
    from huggingface_hub import list_repo_files, hf_hub_download
    try:
        cfgs = [f for f in list_repo_files(HF_REPO) if f.endswith("/config.json")]
        if cfgs:
            meta = json.load(open(hf_hub_download(HF_REPO, cfgs[0])))
            recorded = meta.get("base_model_id") or meta.get("base_model")
            if recorded and recorded != BASE:
                print(f"behaviors in {HF_REPO} were trained on {recorded}")
                BASE = recorded
    except Exception as e:
        print(f"could not read the base from {HF_REPO} ({type(e).__name__}); "
              f"using {BASE}")
elif torch.cuda.is_available() and "3.5" in BASE:
    # Qwen3.5's linear attention needs a CUDA kernel. Without it the model runs
    # a python loop: seven times slower, and enough extra memory through
    # autograd to exhaust a 24 GB card. Qwen3-4B needs no kernel.
    ms = delta_rule_ms(BASE)
    if ms is None:
        pass
    elif ms < 150:
        print(f"linear attention: {ms:.0f} ms for 8 layers, fast path in use")
    else:
        print(f"linear attention: {ms:.0f} ms for 8 layers, the python fallback.")
        print("  pip install flash-linear-attention, then Runtime > Restart session.")
        print(f"  Training on {FALLBACK_BASE} instead, which needs no kernel.")
        BASE = FALLBACK_BASE

print(f"base model: {BASE}")
if TRAIN and torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total < 20:
        print(f"  {total:.0f} GB is enough to run this model but not to train it.")
        print("  Runtime > Change runtime type > L4, or leave TRAIN off.")

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)


def free_gpu(keep=()):
    """Drop any model left on the card by a previous cell.

    A training cell that is interrupted never reaches its own cleanup, and Colab
    keeps the globals alive, so the next load has nowhere to go. Call this by
    hand after stopping a cell.
    """
    for name in ("model", "mod", "net", "bank", "opt", "best"):
        if name in keep:
            continue
        if name in globals():
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"  {free/1e9:.1f} of {total/1e9:.1f} GB free")


def load_base(dtype=DT):
    free_gpu()
    try:
        m = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=dtype, attn_implementation="sdpa", trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(
            BASE, dtype=dtype, attn_implementation="sdpa", trust_remote_code=True)
    return m.cuda().eval()

_TKW = None
def chat(prompt):
    """enable_thinking is Qwen-specific; decide once and cache."""
    global _TKW
    msg = [{"role": "user", "content": prompt}]
    if _TKW is None:
        for kw in ({"enable_thinking": False}, {}):
            try:
                tok.apply_chat_template(msg, tokenize=False,
                                        add_generation_prompt=True, **kw)
                _TKW = kw; break
            except Exception:
                continue
        _TKW = _TKW or {}
    return tok.apply_chat_template(msg, tokenize=False,
                                   add_generation_prompt=True, **_TKW)


@torch.no_grad()
def batch_generate(model, prompts, max_new=90, batch=8, temperature=0.0):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch):
        chunk = [chat(p) for p in prompts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=temperature > 0,
                             temperature=temperature or None,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j][enc.input_ids.shape[1]:],
                                  skip_special_tokens=True).strip())
        if i % (batch * 10) == 0:
            print(f"  {i + len(chunk)}/{len(prompts)}", end="\\r")
    print()
    return out


class Best:
    """Keep the parameters from the best held-out score, and stop when it stops
    improving. Without this, a run that overfits ships the overfit weights, and
    interrupting it by hand loses everything."""

    def __init__(self, module, patience=2, min_delta=5e-3, higher_is_better=False):
        self.m, self.patience, self.min_delta = module, patience, min_delta
        self.sign = -1.0 if higher_is_better else 1.0
        self.best, self.bad, self.at, self.snap = float("inf"), 0, 0, None

    def update(self, step, score):
        v = self.sign * score
        if v < self.best - self.min_delta:
            self.best, self.bad, self.at = v, 0, step
            self.snap = [p.detach().clone() for p in self.m.trainable_parameters()]
            return False, "best"
        self.bad += 1
        return self.bad >= self.patience, f"{self.bad}/{self.patience}"

    def restore(self):
        if self.snap is None:
            return False
        params = list(self.m.trainable_parameters())
        if len(params) != len(self.snap):
            return False
        with torch.no_grad():
            for p, q in zip(params, self.snap):
                p.copy_(q)
        print(f"  restored the checkpoint from step {self.at}")
        return True


from huggingface_hub import HfApi

def hf_token():
    """Colab secrets are not environment variables; they need userdata.get(),
    and the secret needs notebook access granted under the key icon."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t
    try:
        from google.colab import userdata
        t = userdata.get("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
            return t
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def push(folder, name):
    """Upload one behavior as soon as it is trained, so a lost session costs
    the one in flight rather than everything before it."""
    if not PUSH:
        return
    t = hf_token()
    if not t:
        print("  no write token, skipping upload "
              "(add HF_TOKEN under the key icon, with notebook access)")
        return
    api = HfApi(token=t)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=HF_REPO, repo_type="model",
                      path_in_repo=name)
    print(f"  pushed to https://huggingface.co/{HF_REPO}/tree/main/{name}")


from lara import LARA, Bank, decoder_layers

def lara_target(model):
    """The object to hand LARA: a wrapper hides the blocks one level down."""
    try:
        decoder_layers(model); return model
    except Exception:
        pass
    for name, sub in model.named_modules():
        if isinstance(getattr(sub, "layers", None), torch.nn.ModuleList) \\
                and len(sub.layers) > 2:
            try:
                decoder_layers(sub)
            except Exception:
                continue
            print(f"  attaching to model.{name} ({len(sub.layers)} blocks)")
            return sub
    raise RuntimeError("no decoder stack found")


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _decoder(model):
    d = getattr(model, "get_decoder", lambda: None)()
    if d is not None and hasattr(d, "layers"):
        return d
    for _, m in model.named_modules():
        if isinstance(getattr(m, "layers", None), torch.nn.ModuleList):
            return m
    raise RuntimeError("no decoder")


def hidden(model, ids, att):
    o = _decoder(model)(input_ids=ids, attention_mask=att, use_cache=False)
    return o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]


def batch_of(rows, key, max_len=MAX_LEN):
    """Tokenize, and mask the prompt so loss falls only on the completion."""
    ids_, lab_ = [], []
    for r in rows:
        p = tok(chat(r["prompt"]), add_special_tokens=False).input_ids
        c = tok(r[key] + tok.eos_token, add_special_tokens=False).input_ids
        ids_.append((p + c)[:max_len])
        lab_.append(([-100] * len(p) + c)[:max_len])
    m = max(len(x) for x in ids_)
    pad = tok.pad_token_id or tok.eos_token_id
    att = [[1] * len(x) + [0] * (m - len(x)) for x in ids_]
    ids_ = [x + [pad] * (m - len(x)) for x in ids_]
    lab_ = [x + [-100] * (m - len(x)) for x in lab_]
    t = lambda v: torch.tensor(v, dtype=torch.long, device="cuda")
    return t(ids_), t(att), t(lab_)


def masked_ce(model, ids, att, lab):
    h = hidden(model, ids, att)[:, :-1]
    tgt = lab[:, 1:]
    keep = tgt != -100
    if not keep.any():
        return h.sum() * 0.0
    return F.cross_entropy(model.get_output_embeddings()(h[keep]).float(), tgt[keep])


def masked_logprob_sum(model, ids, att, lab):
    h = hidden(model, ids, att)[:, :-1]
    tgt = lab[:, 1:]
    keep = tgt != -100
    head = model.get_output_embeddings()
    out = []
    for b in range(ids.shape[0]):
        k = keep[b]
        if not k.any():
            out.append(h.new_zeros((), dtype=torch.float32)); continue
        lp = torch.log_softmax(head(h[b][k]).float(), -1)
        out.append(lp.gather(-1, tgt[b][k].unsqueeze(-1)).squeeze(-1).sum())
    return torch.stack(out)
''')

md("""
## 1. Fetch the texts

Gutendex resolves titles to Gutenberg IDs, so there is nothing to hardcode and
nothing to parse. Gutenberg serves plain text; the only cleaning needed is
removing the licence header and footer.
""")

code('''
if not TRAIN:
    print("TRAIN is off, skipping the download. Go to section 5.")
else:
    import urllib.request, time

    # Gutenberg rejects the default Python-urllib user agent with a 403.
    UA = {"User-Agent": "Mozilla/5.0 (compatible; LARA-style-notebook/1.0)"}

    def fetch(url, tries=3, timeout=45):
        last = None
        for i in range(tries):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.read()
            except Exception as e:
                last = e
                if getattr(e, "code", None) in (403, 429, 500, 502, 503, None):
                    time.sleep(2 * (i + 1)); continue
                break
        raise RuntimeError(f"gave up on {url}: {last}")


    def text_urls(gid):
        """Three layouts are in use; try the current one first."""
        return [f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
                f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8"]


    START = re.compile(r"\\*\\*\\*\\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\\*\\*\\*",
                       re.I | re.S)
    END = re.compile(r"\\*\\*\\*\\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)

    def strip_boilerplate(t):
        m = START.search(t)
        if m:
            t = t[m.end():]
        m = END.search(t)
        if m:
            t = t[:m.start()]
        t = re.sub(r"\\r\\n", "\\n", t)
        t = re.sub(r"\\n{3,}", "\\n\\n", t)
        # transcriber notes and contents sit at the top; drop the first short lines
        paras = t.split("\\n\\n")
        while paras and len(paras[0]) < 200:
            paras.pop(0)
        return "\\n\\n".join(paras).strip()


    texts = {}
    for title, gid in MANUAL_IDS.items():
        raw = None
        for u in text_urls(gid):
            try:
                raw = fetch(u).decode("utf-8", errors="replace"); break
            except Exception:
                continue
        if raw is None:
            print(f"  no text file for #{gid}: {title}")
            continue
        body = strip_boilerplate(raw)
        texts[title] = body
        print(f"  {title:<32} #{gid:<6} {len(body.split()):>7,} words")

    if not texts:
        raise SystemExit("Nothing downloaded. Check the ids against gutenberg.org.")
    print(f"\\ntotal {sum(len(t.split()) for t in texts.values()):,} words")
''')

md("""
## 2. Cut it into passages

Split on paragraph boundaries and keep passages in a usable length band. Very
short paragraphs are dialogue fragments with no style signal on their own, so
they are joined to their neighbours rather than dropped.
""")

code('''
if not TRAIN:
    print("TRAIN is off, skipping. Go to section 5.")
else:
    lo, hi = PASSAGE_TOKENS
    passages = []
    for title, body in texts.items():
        buf = []
        for para in body.split("\\n\\n"):
            para = " ".join(para.split())
            if not para:
                continue
            buf.append(para)
            n = len(tok(" ".join(buf), add_special_tokens=False).input_ids)
            if n >= lo:
                if n <= hi:
                    passages.append(" ".join(buf))
                buf = []
    random.shuffle(passages)
    passages = passages[:N_PASSAGES]

    lens = [len(tok(p, add_special_tokens=False).input_ids) for p in passages]
    print(f"{len(passages)} passages, {sum(lens)/len(lens):.0f} tokens on average")
    print(f"\\nfirst 200 characters of one:\\n")
    print(textwrap.fill(passages[0][:200], 78))
''')

md("""
## 3. Turn passages into instructions

Continuation training teaches a model to continue Gutenberg text, which is not
the same as writing in a style on request. What we want is an instruction paired
with a passage in the target voice.

The instructions are produced by running the reverse task on the base model: given
a passage, what request would produce it. One short generation per passage.

The same instructions then give DPO pairs for nothing. Chosen is the author's
passage, rejected is the base model's own answer to that instruction. On-policy,
no judge, and the preference is exactly the thing being trained: this voice rather
than yours.
""")

code('''
if not TRAIN:
    print("TRAIN is off, skipping. Go to section 5.")
else:
    ASK = ("Read the passage below. Write the one-sentence instruction a writer "
           "could have been given that would produce it. Describe the subject and "
           "the situation only. Do not mention style, the author, or the passage.\\n\\n"
           "Passage:\\n{p}\\n\\nInstruction:")

    model = load_base()
    print("writing instructions")
    instructions = batch_generate(model, [ASK.format(p=p[:1200]) for p in passages],
                                  max_new=48)
    instructions = [re.sub(r'^["\\'*\\s]+|["\\'*\\s]+$', "", i).split("\\n")[0]
                    for i in instructions]

    print("writing the base model's own answers, for the DPO pairs")
    rejected = batch_generate(model, instructions, max_new=220, temperature=0.8)

    sft = [{"prompt": i, "completion": p}
           for i, p in zip(instructions, passages) if len(i) > 15]
    dpo = [{"prompt": i, "chosen": p, "rejected": r}
           for i, p, r in zip(instructions, passages, rejected)
           if len(i) > 15 and len(r) > 40]

    for name, rows in (("sft", sft), ("dpo", dpo)):
        with open(f"data/{SLUG}_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\\n")
        print(f"data/{SLUG}_{name}.jsonl  {len(rows)} rows")

    print(f"\\nexample instruction:\\n  {sft[0]['prompt']}")
    del model; gc.collect(); torch.cuda.empty_cache()
''')

md("""
## 4. Training

Two pieces of plumbing, both worth knowing about.

Qwen3.5 is a multimodal wrapper, so its decoder blocks sit one level deeper than
`lara.decoder_layers` looks. `lara_target` finds them.

And the vocabulary is 248,320 entries. Running the language-model head over every
position and upcasting to fp32 costs more memory than the model does. Only the
completion tokens carry loss, so only they need logits.
""")

md("""### 4a. The supervised behavior

If you stop a training cell by hand, run `free_gpu()` before the next one. The
cell never reaches its own cleanup, and the model stays on the card.""")

code('''
%%time
if not TRAIN:
    print("TRAIN is off, skipping. Go to section 5.")
else:
    model = load_base()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()          # transformers gates checkpointing on self.training, so
                           # a model left in eval() keeps every layer's activations
    freeze_all(model)
    assert _decoder(model).gradient_checkpointing and model.training, \
        "gradient checkpointing is not active; the forward pass will not fit"

    rows = [json.loads(l) for l in open(f"data/{SLUG}_sft.jsonl")]
    train, held = rows[:-64], rows[-64:]

    mod = LARA(lara_target(model), layers=SFT_LAYERS, rank=RANK, alpha=RANK,
               base_model_id=BASE, method="sft")
    print(f"{mod.num_trainable():,} trainable parameters")

    opt = torch.optim.AdamW(mod.trainable_parameters(), lr=1e-4)
    best = Best(mod)
    t0 = time.time()
    for step in range(1, SFT_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(ACCUM):
            ids, att, lab = batch_of(random.sample(train, MICRO), "completion")
            loss = masked_ce(model, ids, att, lab)
            (loss / ACCUM).backward()
            tot += loss.item() / ACCUM
            del loss, ids, att, lab
        torch.nn.utils.clip_grad_norm_(mod.trainable_parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                ev = sum(masked_ce(model, *batch_of(held[i:i+2], "completion")).item()
                         for i in range(0, len(held), 2)) / (len(held) // 2)
            el = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9
            stop, tag = best.update(step, ev)
            print(f"  {step:4d}/{SFT_STEPS}  train {tot:.4f}  held-out {ev:.4f} [{tag}]  "
                  f"[{el/step:.1f}s/step, peak {vram:.1f} GB]")
            if stop:
                print(f"  held-out stopped improving; best was {best.best:.4f} "
                      f"at step {best.at}")
                break

    best.restore()
    mod.save(f"behaviors/{SLUG}_sft", route_samples=[r["prompt"] for r in rows[:200]],
             method="sft")
    mod.detach(); del mod, model
    gc.collect(); torch.cuda.empty_cache()
    print(f"saved behaviors/{SLUG}_sft")
    push(f"behaviors/{SLUG}_sft", f"{SLUG}_sft")
''')

md("""### 4b. The preference behavior

One adapter, in the middle of the network. Preference objectives reach the same
result from a single insertion point, so this artifact is a quarter the size of
the one above.

DPO runs four forward passes per pair. Two of them are the reference, which is the
frozen base with the adapter off, so a given row's reference log-probabilities
never change. They are cached the first time each row is seen.""")

code('''
%%time
if not TRAIN:
    print("TRAIN is off, skipping. Go to section 5.")
else:
    model = load_base()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()          # transformers gates checkpointing on self.training, so
                           # a model left in eval() keeps every layer's activations
    freeze_all(model)
    assert _decoder(model).gradient_checkpointing and model.training, \
        "gradient checkpointing is not active; the forward pass will not fit"

    rows = [json.loads(l) for l in open(f"data/{SLUG}_dpo.jsonl")]
    train, held = rows[:-48], rows[-48:]

    mod = LARA(lara_target(model), layers=DPO_LAYERS, rank=RANK, alpha=RANK,
               base_model_id=BASE, method="dpo")
    print(f"{mod.num_trainable():,} trainable parameters")

    BETA = 0.1
    NLL = 0.2   # holds log p(chosen) up. DPO only optimises the gap between
                # chosen and rejected, so it can widen the gap by pushing both
                # down, and degenerate loops are what falls out of that.
    opt = torch.optim.AdamW(mod.trainable_parameters(), lr=1e-4)
    best = Best(mod, higher_is_better=True)
    ref_cache = {}

    def logps(batch, key, lengths=False):
        ids, att, lab = batch_of(batch, key)
        total = masked_logprob_sum(model, ids, att, lab)
        if not lengths:
            return total
        return total, (lab[:, 1:] != -100).sum(-1).clamp(min=1)

    def reference(batch):
        miss = [r for r in batch if id(r) not in ref_cache]
        if miss:
            with torch.no_grad(), mod.disabled():
                c, r_ = logps(miss, "chosen"), logps(miss, "rejected")
            for k, row in enumerate(miss):
                ref_cache[id(row)] = (c[k].item(), r_[k].item())
        v = [ref_cache[id(r)] for r in batch]
        f = lambda i: torch.tensor([x[i] for x in v], dtype=torch.float32, device="cuda")
        return f(0), f(1)

    def pair_loss(batch):
        rc, rr = reference(batch)
        pc, n = logps(batch, "chosen", lengths=True)
        pr = logps(batch, "rejected")
        margin = (pc - rc) - (pr - rr)
        pref = -F.logsigmoid(BETA * margin).mean()
        nll = -(pc / n).mean()                    # per token, so lengths compare
        return pref + NLL * nll, margin.mean(), (pc - rc).mean(), (pr - rr).mean()

    t0 = time.time()
    for step in range(1, DPO_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        batch = random.sample(train, 8)
        tl = tm = 0.0
        dc = dr = 0.0
        k = 8 // MICRO
        for i in range(0, 8, MICRO):
            l, m, c, r = pair_loss(batch[i:i + MICRO])
            (l / k).backward()
            tl += l.item() / k; tm += m.item() / k
            dc += c.item() / k; dr += r.item() / k
            del l, m, c, r
        torch.nn.utils.clip_grad_norm_(mod.trainable_parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                em = sum(pair_loss(held[i:i+2])[1].item()
                         for i in range(0, len(held), 2)) / (len(held) // 2)
            el = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9
            stop, tag = best.update(step, em)
            print(f"  {step:4d}/{DPO_STEPS}  loss {tl:.4f}  margin {tm:+.3f}  "
                  f"chosen {dc:+.2f}  rejected {dr:+.2f}  "
                  f"held-out {em:+.3f} [{tag}]  [{el/step:.1f}s/step]")
            if dc < -1.0:
                print("     log p(chosen) is falling; raise NLL or BETA")
            if stop:
                print(f"  held-out margin stopped improving at step {best.at}")
                break

    best.restore()
    mod.save(f"behaviors/{SLUG}_dpo", route_samples=[r["prompt"] for r in rows[:200]],
             method="dpo")
    mod.detach(); del mod, model
    gc.collect(); torch.cuda.empty_cache()
    print(f"saved behaviors/{SLUG}_dpo")
    push(f"behaviors/{SLUG}_dpo", f"{SLUG}_dpo")
''')

md("""
## 5. Load the bank

Both behaviors on one frozen base.

If you trained them above they are read from disk. Otherwise they come from the
Hub, which is a public repository, so this needs no account and no token.
""")

code('''
# Everything in the repo, plus anything trained in this session, in one folder.
# Behaviors are named <author>_<method>, so retraining an author overwrites its
# folder and training a new one adds another. Locally trained copies win.
import shutil
from huggingface_hub import snapshot_download

root = "bank"
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root, exist_ok=True)


def is_behavior(d):
    return os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))


def collect(src, label):
    n = 0
    for d in sorted(os.listdir(src)):
        full = os.path.join(src, d)
        if not is_behavior(full):
            continue
        dst = os.path.join(root, d)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(full, dst)
        n += 1
    if n:
        print(f"  {n} from {label}")
    return n


try:
    collect(snapshot_download(HF_REPO, repo_type="model"), HF_REPO)
except Exception as e:
    print(f"  nothing from {HF_REPO} ({type(e).__name__})")

if os.path.isdir("behaviors"):
    collect("behaviors", "this session")

# A behavior only loads onto the base it was trained against, so a bank mixing
# bases cannot be assembled. Keep the ones that match and say what was dropped.
names, dropped = [], []
for d in sorted(os.listdir(root)):
    meta = json.load(open(os.path.join(root, d, "config.json")))
    rec = meta.get("base_model_id") or meta.get("base_model")
    (names if rec in (None, BASE) else dropped).append((d, rec))
names = [d for d, _ in names]

if dropped:
    print(f"  skipped, trained on another base: "
          + ", ".join(f"{d} ({r})" for d, r in dropped))
if not names:
    raise SystemExit(f"No behaviors for {BASE}. Train some, or set BASE to one "
                     f"the artifacts were trained on.")

model = load_base()
model.config.use_cache = True

bank = Bank(lara_target(model), tok, top_k=None)
for n in names:
    bank.add(n, os.path.join(root, n))    # strict: refuses a mismatched base
bank.fit_router()

for n in names:
    mb = sum(os.path.getsize(os.path.join(root, n, f))
             for f in os.listdir(os.path.join(root, n))) / 1e6
    print(f"  {n:<20} {mb:>6.1f} MB")
print(f"\\nbase {BASE}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters, frozen")
''')

md("""
## 6. Compare

The operation that matters here is comparison: the same prompt under different
settings, read side by side. Three functions do that.

`contrast(prompt)` is the one to start with: the frozen base, then every author
in the bank, each behavior alone and then its objectives blended. One subject,
every voice, side by side.

`compare(prompt, author="stein")` is the same for a single author, for when you
are iterating rather than reading.

`sweep(prompt)` scales whatever is currently switched on, keeping the ratios, so
it scales the mix rather than replacing it. It runs 0, 1, 2, 3.

One thing worth knowing before you read the output: **1.0 is not where these
behaviors are visible.** It is the strength they were trained at, but on this
base a single behavior at 1.0 barely moves the writing. Around 2.0 it arrives.
Two behaviors at 1.5 each arrive as well, since what matters is the total
correction reaching the stream, not any one value. Keep going and it turns into
a parody of itself, which is worth seeing at least once.

Four prompts are provided. `PROMPTS["object"]` separates the three authors most
sharply, because a static subject gives style nowhere to hide, while a situation
lets any model fall back on plot.

`panel()` puts the same controls behind sliders if you would rather drag than
retype.
""")

code('''
class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


@torch.no_grad()
def gen(prompt, max_new=180, base=False, pin=None, temperature=0.8, seed=0,
        repetition_penalty=1.1):
    torch.manual_seed(seed)                    # same sample across settings
    enc = tok(chat(prompt), return_tensors="pt", add_special_tokens=False).to("cuda")
    ctx = bank.disabled() if base else (bank.pin(pin) if pin else _null())
    with ctx:
        out = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=temperature > 0, temperature=temperature,
                             top_p=0.95, repetition_penalty=repetition_penalty,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def block(title, text, width=88):
    bar = chr(9472) * width
    print(chr(10) + chr(27) + '[1m' + title + chr(27) + '[0m')
    print(bar)
    for para in text.split(chr(10)):
        print(textwrap.fill(para, width) if para.strip() else '')


def with_gammas(g):
    """Set every behavior at once. A dict silences the ones it omits, so a
    single author can be isolated while others stay loaded."""
    was = dict(bank.gamma)
    for n in names:
        bank.set_gamma(n, g if isinstance(g, float) else g.get(n, 0.0))
    return was


def authors():
    """Behaviors are named <author>_<method>, so the prefix groups them."""
    out = {}
    for n in names:
        out.setdefault(n.rsplit("_", 1)[0], []).append(n)
    return out


def compare(prompt, author=None, max_new=180, seed=0):
    """The frozen base, then one author's behaviors alone, then both together.

    With several authors in the bank, showing all of them buries the comparison
    that matters. One author at a time, picked at random unless you name one.
    """
    by = authors()
    author = author or random.choice(sorted(by))
    mine = by[author]

    was = dict(bank.gamma)
    with_gammas(0.0)                       # silence the rest of the bank too
    block("frozen base", gen(prompt, max_new, base=True, seed=seed))
    for n in mine:
        with_gammas({n: 1.0})
        block(n, gen(prompt, max_new, seed=seed))
    if len(mine) > 1:
        with_gammas({n: 1.0 for n in mine})
        block(f"{author}, both routed", gen(prompt, max_new, seed=seed))
    for n, v in was.items():
        bank.set_gamma(n, v)
    print()
    print(f"({author}; pass author= to choose. "
          f"in the bank: {', '.join(sorted(by))})")


def contrast(prompt, max_new=180, seed=0):
    """One prompt across the whole bank: the frozen base, then every author.

    For three authors trained with two objectives each that is ten passages,
    which is the figure worth putting in front of a reader. Everything not being
    shown is silenced, so each block is that behavior alone.
    """
    by = authors()
    was = dict(bank.gamma)
    with_gammas(0.0)
    block("frozen base", gen(prompt, max_new, base=True, seed=seed))
    for a in sorted(by):
        for n in by[a]:
            with_gammas({n: 1.0})
            block(n, gen(prompt, max_new, seed=seed))
        if len(by[a]) > 1:
            with_gammas({n: 1.0 for n in by[a]})
            block(f"{a}, both routed", gen(prompt, max_new, seed=seed))
    for n, v in was.items():
        bank.set_gamma(n, v)


def sweep(prompt, scale=(0.0, 1.0, 2.0, 3.0), max_new=180, seed=0):
    """Scale whatever is currently switched on, from nothing to past trained.

    Ratios between behaviors are kept, so this scales the mix you set with the
    sliders rather than replacing it.

    At 0 the model is the untouched base. At 1 it is the strength the behavior
    was trained at, which on this base is usually too weak to see. What matters
    is the total across everything switched on: one behavior near 2, or two at
    1.5 each, is where these styles arrive. Push further to find where the
    correction is amplified past anything it saw and the writing turns into a
    parody of itself.
    """
    was = dict(bank.gamma)
    on = {n: v for n, v in was.items() if v > 0}
    if not on:
        print("nothing is switched on. Set a gamma, or use contrast().")
        return
    print("scaling " + ", ".join(f"{n} at {v:g}" for n, v in sorted(on.items())))
    for x in scale:
        with_gammas({n: v * float(x) for n, v in on.items()})
        block(f"x {x}", gen(prompt, max_new, seed=seed))
    for n, v in was.items():
        bank.set_gamma(n, v)


# Situations let a model fall back on plot, where every style converges. A
# static subject gives style nowhere to hide, which is what makes the first of
# these the one to use when comparing authors.
PROMPTS = {
    "object":  "Describe a glass of water on a windowsill in the afternoon.",
    "waiting": "Write a paragraph about a man waiting alone in a cafe before a storm.",
    "party":   "Write a paragraph about a young man standing at the edge of a "
               "party he was not invited to.",
    "leaving": "Write a short exchange between two people who disagree about leaving.",
}
PROMPT = PROMPTS["object"]

contrast(PROMPT)
''')

md("""
### Sliders

Same thing with controls, if you prefer to drag. One slider per behavior, and the
text box takes any prompt.

The sliders run to 3.0 rather than stopping at the trained strength, because on
this base a single behavior at 1.0 is barely visible. Start around 1.5 and raise
it until the style arrives.
""")

code('''
try:
    import ipywidgets as W
    from IPython.display import display, clear_output

    box = W.Textarea(value=PROMPT, layout=W.Layout(width="100%", height="70px"))
    sliders = {n: W.FloatSlider(value=1.5, min=0.0, max=3.0, step=0.1,
                                description=n, continuous_update=False,
                                readout_format=".1f",
                                layout=W.Layout(width="460px"))
               for n in names}
    note = W.HTML(
        "<div style='font-size:90%;color:#666;margin:6px 0 10px'>"
        "1.0 is the strength each behavior was trained at, and on this base that "
        "is usually too weak to see. What counts is the total across everything "
        "switched on: one behavior near 2, or two at 1.5 each. Above that the "
        "correction is amplified past anything it saw."
        "</div>")
    length = W.IntSlider(value=180, min=60, max=400, step=20, description="tokens",
                         layout=W.Layout(width="420px"))
    run = W.Button(description="Generate", button_style="primary")
    base_btn = W.Button(description="Frozen base")
    sweep_btn = W.Button(description="Sweep gamma")
    out = W.Output()

    def _run(_):
        with out:
            clear_output()
            for n, s in sliders.items():
                bank.set_gamma(n, s.value)
            block("output", gen(box.value, length.value))

    def _base(_):
        with out:
            clear_output()
            block("frozen base", gen(box.value, length.value, base=True))

    def _sweep(_):
        with out:
            clear_output()
            for n, sl in sliders.items():        # sweep scales what is set here
                bank.set_gamma(n, sl.value)
            sweep(box.value, max_new=length.value)

    run.on_click(_run); base_btn.on_click(_base); sweep_btn.on_click(_sweep)
    display(W.VBox([box, *sliders.values(), note, length,
                    W.HBox([run, base_btn, sweep_btn]), out]))
except Exception as e:
    print(f"widgets unavailable ({type(e).__name__}); use compare() and sweep()")
''')

md("""
### Where it breaks

Everything above stays inside the useful range. The dial does not stop there, and
the far end is worth seeing once: the correction is amplified past anything the
behavior was trained on, and the writing turns into a parody of the style rather
than an example of it.
""")

code('''
by = authors()
who = sorted(by)[0]
one = by[who][0]

was = dict(bank.gamma)
for g in (1.0, 2.0, 3.0, 4.0, 5.0):
    with_gammas({one: g})
    block(f"{one} at {g}", gen(PROMPTS["waiting"], max_new=140))
for n, v in was.items():
    bank.set_gamma(n, v)
''')

md("""
## What happened

The base model was never modified. Its weights after training are identical to
its weights before, bit for bit, and with every gamma at zero the forward pass
reproduces the original exactly. The style lives in two files of a few megabytes,
and the slider that turns it on is a floating-point number read at inference.

That is the part worth taking away. Fine-tuning produces a new model. This
produces a file that sits next to one.

Two behaviors on the same base also compose, which is why they were trained
separately rather than on one mixed objective. Add a third for another author,
or a fourth for a genre, and the router chooses between them per token without
any of them being retrained.

### On the dial

The strength a behavior is trained at is not the strength it is useful at. Every
behavior here was trained at 1.0, and at 1.0 on this base the writing barely
moves. Around 2.0 the style arrives. Two behaviors at 1.5 each arrive as well,
because what reaches the residual stream is the total correction rather than any
one behavior's setting.

That number is not a property of the method. Measured across several other base
models it sits nearer 0.5 or 1.0, so it has to be found per base rather than
assumed. It costs one sweep, and getting it wrong looks exactly like a behavior
that did not train.

### Things worth trying

Train a second author and blend them. Stein and Fitzgerald pull in opposite
directions and both are public domain, so a blend of the two is a real test of
whether composition works rather than one behavior winning.

Compare the two objectives. The supervised behavior and the preference behavior
were trained on the same passages toward the same target. They do not learn the
same thing, and the difference is visible in the output.

Measure it rather than judging by eye. Mean sentence length, adjective rate and
type-token ratio are three numbers you can compute on the base, on the corpus,
and on each setting.

### Note on other corpora

Screenplays are an appealing target, since format and voice are both strong. Sites
that host them carry material that is almost entirely in copyright, which makes
that a private experiment rather than something to publish or distribute.
""")

nb = {"cells": C,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("hemingway_style.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {len(C)} cells")
