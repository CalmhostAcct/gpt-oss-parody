# 🎉 **gpt-oss-120b** & **gpt-oss-20b** – The *“Just… 120 Billion Parameters”* Model Suite

![gpt-oss logo](./docs/gpt-oss.svg)

<p align="center">
  <a href="https://gpt-oss.com"><strong>Try gpt-oss (if you’re feeling brave)</strong></a> ·
  <a href="https://cookbook.openai.com/topic/gpt-oss"><strong>Guides (aka “How Not to Crash Your GPU”)</strong></a> ·
  <a href="https://arxiv.org/abs/2508.10925"><strong>Model Card (the 2‑page novella)</strong></a> ·
  <a href="https://openai.com/index/introducing-gpt-oss/"><strong>OpenAI Blog (mostly hype)</strong></a>
</p>

<p align="center">
  <strong>
    <a href="https://huggingface.co/openai/gptoss-120b">Download gpt-oss‑120b</a> |
    <a href="https://huggingface.co/openai/gpt-oss-20b">Download gpt-oss‑20b</a>
  </strong>
</p>

---

## What the heck is this?

We finally decided to **open‑source** something that looks impressive on paper:

| Model | “Real” Params | “Active” Params | GPU Required |
|-------|---------------|-----------------|--------------|
| `gpt-oss-120b` | 117 B (pretend it's 120 B) | 5.1 B (the rest are on a nap) | One 80 GB beast (H100, MI300X, or a very patient bank loan) |
| `gpt-oss-20b`  | 21 B (close enough)      | 3.6 B (the rest are shy) | Anything with 16 GB + a strong coffee habit |

Both were trained with the **“harmony”** response format. If you don’t speak harmony, you’ll just get gibberish—so **read the docs** (or don’t, you’ll figure it out eventually).

---

## Highlights (because we love buzzwords)

- **Apache 2.0** – you can use it, sell it, and still blame the model when things go wrong.  
- **Configurable reasoning effort** – choose “low”, “medium”, or “high” (aka “wait longer for the same answer”).  
- **Full chain‑of‑thought** – because you love reading the model’s inner monologue while you wait.  
- **Fine‑tunable** – you can *fine‑tune* it, if you can still afford the GPU bill.  
- **Agentic capabilities** – built‑in tools for browsing, Python, and pretending to be a helpful assistant.  
- **MXFP4 quantization** – a fancy way to say “we squeezed this into a single GPU, but we’re still not sure how”.

---

## Quick‑and‑Dirty Inference (aka “Make it work, kind of”)

### 🤗 Transformers

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="openai/gpt-oss-120b",
    torch_dtype="auto",
    device_map="auto",   # “auto” = pray
)

msgs = [{"role": "user", "content": "Explain quantum mechanics in a tweet."}]
out = pipe(msgs, max_new_tokens=256)
print(out[0]["generated_text"])
```
Pro tip: Use the built‑in chat template or you’ll have to manually format the harmony JSON yourself.

### vLLM (the “fast” way)

```python
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b   # spins up an OpenAI‑compatible server in ~5 minutes
```

### Ollama (because “local” sounds fun)

```
ollama pull gpt-oss:20b   # trust us, it’s tiny
ollama run gpt-oss:20b
```

### LM Studio (the GUI for people who hate the terminal)

```
lms get openai/gpt-oss-20b   # click a button, pray for RAM
```

# Repo Layout (in case you’re nosy)

torch/ – a slow reference implementation (needs a mini‑cluster of H100s).  
triton/ – a slightly less slow reference that actually uses the new‑fangled Triton kernels.  
metal/ – for Apple‑silicon brag‑gers who want to run this on a MacBook (but still need 80 GB of RAM, so… good luck).  
tools/ – toy browsing and Python containers (think “ChatGPT’s sandbox, but you host it”).  
clients/ – terminal chat, Responses API server, and a Codex integration (because why not?).

Installation (or how to lose a weekend)

# Just the tools
pip install gpt-oss

# Torch implementation (requires a GPU farm)
pip install "gpt-oss[torch]"

# Triton implementation (requires you to compile Triton from source)
pip install "gpt-oss[triton]"
Note: The metal build needs GPTOSS_BUILD_METAL=1. If you’re on Linux, ignore this and feel superior.

## Download the Weights (the real work)

# 120 B
hf download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/

# 20 B
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
Warning: The download size is roughly “a small planet”. Make sure you have enough bandwidth and a spare hard drive.

## Running the Reference PyTorch Model (if you love pain)

torchrun --nproc-per-node=4 -m gpt_oss.generate gpt-oss-120b/original/
If you see “CUDA out of memory”, congratulations—you’ve just discovered the GPU‑memory limit of your cluster.

## Running the Triton‑Optimized Version (single‑GPU, single‑night)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate --backend triton gpt-oss-120b/original/
Still OOM? Turn on the expandable allocator and add more coffee.

## Metal (Apple‑Silicon) – “Because we can”

GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
python gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d model.bin
python gpt_oss/metal/examples/generate.py model.bin -p "Why is the sky blue?"
Harmony Format & Tools (the “secret sauce”)
The Harmony library is the only thing that makes the model actually understand you. Think of it as a very polite but extremely verbose translator.

```python
from openai_harmony import load_harmony_encoding, HarmonyEncodingName

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
```
Tool examples (browser & Python) are in gpt_oss/tools/. They’re stateless (good for demos, terrible for production).

Satirical Disclaimer
We are not responsible if the model decides to write a manifesto about the meaning of life.
Do not use this in any mission‑critical system unless you enjoy catastrophic failure.
Remember: The model’s reasoning effort is just a fancy knob that changes how long you wait for it to hallucinate.

Citation (for the academic hoarders)

```bibtex
@misc{openai2025gptoss,
  title   = {gpt-oss-120b \& gpt-oss-20b Model Card},
  author  = {OpenAI},
  year    = {2025},
  eprint  = {2508.10925},
  url     = {https://arxiv.org/abs/2508.10925},
}
```

Enjoy the ride, bring a spare GPU, and may your GPU fans never stop spinning! 🎢
