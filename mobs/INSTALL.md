# Running MoBs locally

A frozen base model with a bank of behaviors on top, served to a browser page
with one fader per behavior. Nothing is written to the model, so pulling every
fader to zero gives you the original back exactly.

If you would rather not install anything, the
[Colab notebook](../mobs/mobs_playground.ipynb) does the same thing in a
browser tab.


<a href="https://colab.research.google.com/github/pfekin/LARA/blob/main/mobs/mobs_playground.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>


## What you need

An NVIDIA card with 8 GB or more (a GTX 1660 or RTX 2060 is enough), Python 3.10
to 3.12, and about 12 GB of disk for the model. It runs on CPU as well, slowly;
see the notes at the end.

Python 3.13 is worth avoiding for now, since some of the dependencies have no
prebuilt wheel for it and fall back to compiling from source.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate safetensors huggingface_hub
pip install fastapi uvicorn bitsandbytes kernels
pip install git+https://github.com/pfekin/LARA.git
```

The first line is the one to check rather than copy. Run `nvidia-smi`, read the
CUDA version in the top right, and take the matching command from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

Then confirm the card is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True` and the name of your card means you are set. `False` means the CUDA build
does not match the driver, so go back to the first line.

`kernels` supplies the optimized CUDA kernels the base model uses. With an NVIDIA
card it is worth roughly a threefold speedup. **On a machine with no GPU, do not
install it**: at a mismatched version it breaks `import transformers` outright.

## Get the files

```bash
git clone https://github.com/pfekin/LARA.git
cd LARA/mobs
```

The folder holds `serve_mobs.py`, `train_mobs.py` and `static/index.html`. The
page has to stay in `static/`, or the server starts and the browser shows
nothing.

## Run it

```bash
python serve_mobs.py --bank pfekin/mobs-qwen3.5-4b --quant 4bit
```

The first run downloads the base model, about 9 GB. Later runs read it from the
cache and start in under a minute.

When it is ready:

```
ready: 5 behaviors [...] on cuda:0

  http://127.0.0.1:8000
```

Open that address. Leave the terminal open while you use it; `Ctrl+C` stops it.

## Using it

Each fader is one behavior. 1.0 is the strength it was trained at. What reaches
the model is the total across everything switched on, so two behaviors at 1.0
carry further than one does.

Three prompts worth trying, with the settings that make the difference visible:

**Write a Python function that reverses a dictionary.**
Every fader at zero, then every fader at one. Watch the length.

**Solve for x in 3x² + 11x − 4 = 0. What is the larger of the two solutions?**
Every fader at zero, then `math` alone with *pin* on so the router cannot dilute
it.

**What is the difference between TCP and UDP?**
Every fader at zero, then `summary` alone, pinned, at 1.0. Step slowly from 0.75
to 1.0: the dial is not always smooth.

Then try your own. Ask for something you would normally have to explain in a
paragraph of instructions, and see how far a fader gets you instead.

## Options

| | |
|---|---|
| `--bank` | a Hugging Face repo id, or a local directory of behaviors |
| `--quant` | `4bit` (default), `8bit`, or `none` |
| `--max-new` | tokens per answer, default 512 |
| `--port` | default 8000 |
| `--host` | default `127.0.0.1` |

`--bank` accepts a local path, so a bank you trained yourself with
`train_mobs.py` works the same way as one from the Hub.

## If something goes wrong

**`CUDA out of memory`** — close anything else using the card, then retry. If it
persists, `--max-new 256` shortens the answers, and `--quant 4bit` is already the
smallest setting.

**`Either a revision or a version must be specified`** — `kernels` and
`transformers` disagree about a version. `pip install -U transformers` usually
settles it. If not, `pip uninstall -y kernels`: the server then runs without the
optimized kernels, slower but working.

**The page is blank** — check the terminal is still showing the address with no
error after it. If it is, `static/index.html` is missing or in the wrong place.

**`can't open file 'serve_mobs.py'`** — wrong directory. `cd` into `LARA/mobs`
and run `ls` to confirm the file is there.

## Without a GPU

It runs, at a few tokens per second, which is enough to see what a behavior does
but not enough to use for anything.

```bash
pip uninstall -y kernels bitsandbytes
python serve_mobs.py --bank pfekin/mobs-qwen3.5-4b --quant none
```

Both of those packages are CUDA-only, and `kernels` at a mismatched version stops
`transformers` importing at all, so removing them is the fix rather than a
workaround.
