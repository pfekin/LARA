#!/usr/bin/env python3
"""MoBs inference server. One process, one page, no cloud.

    pip install fastapi uvicorn transformers accelerate bitsandbytes
    python serve_mobs.py --base Qwen/Qwen3-4B --bank behaviors
    python serve_mobs.py --bank pfekin/mobs-qwen3-4b --quant 4bit

Then open http://localhost:8000

Coefficients arrive with each request rather than living in server state, so
two clients can hold different mixes without racing. The bank itself is global
mutable state, so generation is serialised behind a lock -- fine for a handful
of users on one GPU, which is the case this is for.
"""
from __future__ import annotations

import argparse, asyncio, json, os, sys, threading, time
from pathlib import Path

import torch

HERE = Path(__file__).parent
STATIC = HERE / "static"

# Shared with train_mobs so the prompt the server builds is byte-identical to
# the one the behaviors were trained on. Falls back to a local copy if this
# file is deployed on its own.
try:
    sys.path.insert(0, str(HERE))
    sys.modules.setdefault("ipykernel", type(sys)("ipykernel"))
    from train_mobs import chatml as _chatml
except Exception:
    _TEMPLATE_KW = None

    def _chatml(tok, prompt, completion=None):
        global _TEMPLATE_KW
        msg = [{"role": "user", "content": prompt}]
        if _TEMPLATE_KW is None:
            for kw in ({"enable_thinking": False}, {}):
                try:
                    tok.apply_chat_template(msg, tokenize=False,
                                            add_generation_prompt=True, **kw)
                    _TEMPLATE_KW = kw
                    break
                except Exception:
                    continue
            _TEMPLATE_KW = _TEMPLATE_KW or {}
        text = tok.apply_chat_template(msg, tokenize=False,
                                       add_generation_prompt=True, **_TEMPLATE_KW)
        return text + completion if completion is not None else text


# ─────────────────────────────────────────────────────────────────────────────
def preflight():
    """transformers imports break outright when `kernels` is a mismatched version.

    hub_kernels.py builds LayerRepository(...) at import time; if the installed
    kernels expects a `revision` or `version` argument and this transformers
    does not pass one, every model import fails with a traceback that looks
    like a model problem and is not.
    """
    try:
        import transformers.activations  # noqa: F401
        return
    except Exception as e:
        if "revision or a version" not in str(e) and "LayerRepository" not in repr(e):
            raise
    import importlib.metadata as md
    ver = lambda p: (md.version(p) if p else "?")
    try:
        k, t = ver("kernels"), ver("transformers")
    except Exception:
        k = t = "?"
    raise SystemExit(
        f"transformers ({t}) and kernels ({k}) are incompatible — every model "
        f"import fails, not just this one.\n"
        f"  CPU box : pip uninstall -y kernels      (they are CUDA kernels; "
        f"transformers works fine without them)\n"
        f"  GPU box : pip install -U transformers   (or pin kernels to a version "
        f"this transformers expects)")


def load_bank_dir(spec):
    """A local directory, or a HF repo id to pull down."""
    if os.path.isdir(spec):
        return spec
    if "/" not in spec:
        raise SystemExit(
            f"--bank {spec!r} is neither a directory here nor a Hub repo id "
            f"(those look like 'owner/name'). Either cd to where {spec}/ lives, "
            f"or pass the repo you pushed to.")
    from huggingface_hub import snapshot_download
    print(f"fetching bank {spec} from the Hub")
    try:
        return snapshot_download(spec, repo_type="model")
    except Exception as e:
        raise SystemExit(
            f"could not fetch {spec}: {type(e).__name__}. If the repo is "
            f"private, export HF_TOKEN first.")


def build(base, bank_dir, quant, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lara import Bank

    cuda = torch.cuda.is_available()
    kw = dict(dtype=getattr(torch, dtype), attn_implementation="eager")

    if quant == "4bit" and cuda:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    elif quant == "8bit" and cuda:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw["device_map"] = {"": 0}
    elif cuda:
        kw["device_map"] = {"": 0}

    if quant != "none" and not cuda:
        print(f"note: {quant} needs CUDA; loading unquantised on CPU instead")

    print(f"loading {base} ({quant if cuda else 'cpu/' + dtype})")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    kw["trust_remote_code"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **kw)
    except Exception as e:
        # some newer bases are natively multimodal and are not causal-LM class
        from transformers import AutoModelForImageTextToText
        print(f"  AutoModelForCausalLM failed ({type(e).__name__}); "
              f"trying AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(base, **kw)
    if not cuda:
        model = model.to("cpu")
    model.eval()
    model.config.use_cache = True

    names = sorted(d.name for d in Path(bank_dir).iterdir()
                   if d.is_dir() and not d.name.startswith("_")
                   and (d / "config.json").exists())
    if not names:
        raise SystemExit(f"no behavior directories in {bank_dir}")

    from train_mobs import lara_target
    target = lara_target(model)
    bank = Bank(target, tok, top_k=None)
    for n in names:
        bank.add(n, str(Path(bank_dir) / n), strict=False)

    router_pt = Path(bank_dir) / "_bank"
    if router_pt.exists():
        try:
            from lara import Bank as _B
            state = torch.load(router_pt / "router.pt", map_location=bank.device,
                               weights_only=False)
            bank.router.load_state_dict(state["state"] if "state" in state else state)
            bank.router.eval()
            bank._fitted_rows = len(names)
            print("router loaded from disk")
        except Exception as e:
            print(f"could not load saved router ({e}); refitting")
            bank.fit_router(verbose=False)
    else:
        print("no saved router; fitting now")
        bank.fit_router(verbose=False)

    dev = next(model.parameters()).device
    print(f"ready: {len(names)} behaviors {names} on {dev}")
    return model, tok, bank, names


# ─────────────────────────────────────────────────────────────────────────────
def make_app(model, tok, bank, names, args):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from transformers import TextIteratorStreamer

    app = FastAPI(title="MoBs")
    lock = asyncio.Lock()

    @app.get("/api/meta")
    def meta():
        return {
            "base": args.base,
            "behaviors": names,
            "layers": bank.layer_ids,
            "route_layer": bank.route_layer,
            "device": str(next(model.parameters()).device),
            "quant": args.quant if torch.cuda.is_available() else "none",
            "max_new": args.max_new,
        }

    @app.post("/api/generate")
    async def generate(body: dict):
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse({"error": "empty prompt"}, status_code=400)
        gammas = body.get("gammas") or {}
        pinned = bool(body.get("pin"))
        max_new = int(body.get("max_new") or args.max_new)
        temp = float(body.get("temperature", 0.0))

        text = _chatml(tok, prompt)     # handles bases that reject enable_thinking
        enc = tok(text, return_tensors="pt")
        # some tokenizers add token_type_ids, which generate() rejects
        dev = next(model.parameters()).device
        ids = {k: v.to(dev) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}

        async def stream():
            async with lock:
                for n in names:                       # per-request, applied here
                    bank.set_gamma(n, float(gammas.get(n, 1.0)))
                streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                                skip_special_tokens=True)
                kw = dict(**ids, max_new_tokens=max_new, streamer=streamer,
                          do_sample=temp > 0, pad_token_id=tok.eos_token_id)
                if temp > 0:
                    kw.update(temperature=temp, top_p=0.95)

                ctx = bank.pin({n: float(gammas.get(n, 1.0)) for n in names}) \
                    if pinned else _null()
                t0, n_tok = time.time(), 0

                err = {}

                def run():
                    # A crash in here used to leave the streamer never ending,
                    # so the request hung instead of reporting the failure.
                    try:
                        with ctx, torch.no_grad():
                            model.generate(**kw)
                    except BaseException as e:
                        err["e"] = f"{type(e).__name__}: {e}"
                    finally:
                        streamer.end()

                th = threading.Thread(target=run, daemon=True)
                th.start()
                try:
                    for piece in streamer:
                        if piece:
                            n_tok += 1
                            yield f"data: {json.dumps({'t': piece})}\n\n"
                            await asyncio.sleep(0)
                finally:
                    th.join(timeout=5)
                if err:
                    yield "data: " + json.dumps({"error": err["e"]}) + "\n\n"
                secs = max(time.time() - t0, 1e-6)
                yield ("data: " + json.dumps(
                    {"done": True, "tokens": n_tok,
                     "tok_per_s": round(n_tok / secs, 2)}) + "\n\n")

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/route")
    async def route(body: dict):
        """What the router makes of a prompt, without generating anything."""
        text = (body.get("prompt") or "").strip()
        if not text:
            return {"weights": {}}
        ids = tok(text, return_tensors="pt").input_ids
        async with lock:
            try:
                return {"weights": bank.route_weights(ids)}
            except Exception as e:
                # The route hook returns early when the bank is disabled, so the
                # weights are simply unavailable. This is a display hint; it must
                # never take the server down.
                return {"weights": {}, "note": type(e).__name__}

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        def index():
            return FileResponse(STATIC / "index.html")

    return app


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


DEFAULTS = dict(
    bank="pfekin/mobs-qwen3.5-4b",
    quant="4bit",
    dtype="bfloat16",
    max_new=512,
)


def defaults():
    """So a notebook or another caller reads the same values as the CLI."""
    return dict(DEFAULTS)


def main(argv=None):
    if argv is None and ("ipykernel" in sys.modules or "google.colab" in sys.modules):
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--bank", default=DEFAULTS["bank"],
                    help="directory or HF repo id")
    ap.add_argument("--quant", default=DEFAULTS["quant"],
                    choices=["4bit", "8bit", "none"])
    ap.add_argument("--dtype", default=DEFAULTS["dtype"],
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-new", type=int, default=DEFAULTS["max_new"])
    args = ap.parse_args(argv)

    preflight()
    bank_dir = load_bank_dir(args.bank)
    meta = Path(bank_dir) / "bank.json"
    if meta.exists() and args.base == ap.get_default("base"):
        args.base = json.load(open(meta)).get("base", args.base)
        print(f"base taken from bank.json: {args.base}")

    model, tok, bank, names = build(args.base, bank_dir, args.quant, args.dtype)
    app = make_app(model, tok, bank, names, args)

    import uvicorn
    print(f"\n  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
