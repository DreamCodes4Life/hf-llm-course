# Hugging Face LLM Course — my notebooks

This repo is my working record of the free [Hugging Face LLM Course](https://huggingface.co/learn/llm-course): the code I ran, the experiments I tried and short notes on what each chapter taught me. It is the first course in a longer self-study plan towards AI engineering (LLMs, fine-tuning, inference and agents), so the aim is to understand each idea well enough to use it later, not just to get the notebooks to run.

Everything runs locally in Jupyter on one NVIDIA GPU (RTX 5090, 32 GB) with PyTorch and 🤗 Transformers. The steps below work on any Linux or macOS machine, with or without a GPU.

## Contents

- [Why local Jupyter instead of Colab](#why-local-jupyter-instead-of-colab)
- [Prerequisites](#prerequisites)
- [Setup, step by step](#setup-step-by-step)
- [Check that it works](#check-that-it-works)
- [Troubleshooting](#troubleshooting)
- [What's inside](#whats-inside)
- [Course map](#course-map)

## Why local Jupyter instead of Colab

The course suggests [Google Colab](https://colab.research.google.com/): a notebook in the browser with a free GPU for small jobs and nothing to install. That is still the quickest start, and the right choice on Windows, which the course doesn't cover.

This repo runs the same notebooks locally instead:

| | Google Colab | Local Jupyter (this repo) |
|---|---|---|
| Setup | None | About 15 minutes, once |
| Hardware | Shared GPU, limited free time | Your own GPU, or CPU |
| Sessions | Time out; files are lost unless saved to Drive | No limits; notebooks and models stay on your disk |
| Installing packages | `!pip install …` in every new session | Once, in a virtual environment |

## Prerequisites

- **Linux or macOS.** On Windows, the course recommends Colab.
- **Python 3.11 or newer.** Transformers 5 and PyTorch 2.15 require it. Check with `python3 --version`.
- **Git**, to clone this repo.
- **About 10 GB of free disk space.** PyTorch with CUDA takes about 6 GB, plus the models you download.
- **An NVIDIA GPU (optional).** A CPU is fine for the first chapters.
- **A free [Hugging Face account](https://huggingface.co/join).** Most of the course relies on it.
- **A notebook app:** JupyterLab (installed in step 4) and/or [VS Code](https://code.visualstudio.com/) with the Python and Jupyter extensions.

## Setup, step by step

### 1. Get the code

```bash
git clone https://github.com/DreamCodes4Life/hf-llm-course.git
cd hf-llm-course
```

The course creates an empty project folder (`mkdir ~/transformers-course`). Here the cloned repo is the project folder.

### 2. Create a virtual environment

A virtual environment is a separate Python with its own packages, so the course can't break your other projects.

```bash
python3 -m venv .env
source .env/bin/activate      # your prompt now starts with (.env)
which python                  # …/hf-llm-course/.env/bin/python
```

`activate` only changes the current terminal; `deactivate` leaves the environment. `.env/` is gitignored.

> **Tip:** if the repo lives on a slow disk, create the environment on a faster one and link it: `python3 -m venv ~/.venvs/hf-llm-course && ln -s ~/.venvs/hf-llm-course .env`. That is how this repo is set up.

### 3. Install PyTorch

```bash
pip install torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Linux, the default build (torch 2.15) ships with CUDA 13.2, which supports recent cards including the RTX 50 series. If the last line prints `False` on a machine with an NVIDIA GPU, get the right command for your system from the [PyTorch install selector](https://pytorch.org/get-started/locally/). On a CPU-only machine, `False` is expected.

### 4. Install the course libraries and Jupyter

```bash
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) contains:

| Package | Why |
|---|---|
| `torch` | Already installed in step 3; listed so the file is complete |
| `transformers[sentencepiece]` | The course's main library. The `[sentencepiece]` extra adds the packages some tokenizers need. Plain `transformers` does not install PyTorch |
| `datasets`, `evaluate`, `accelerate` | Loading data, measuring results and training on any hardware; used from Chapter 3 on |
| `jupyterlab` | The notebook app that replaces Colab. It also installs `ipykernel`, which VS Code uses to run notebooks |
| `ipywidgets` | Proper progress bars in notebooks |
| `pillow` | Loads images for the image pipelines (Chapter 1.3) |

### 5. Log in to Hugging Face

```bash
hf auth login
hf auth whoami
```

Recent versions of `huggingface_hub` ask how you want to log in: **with your browser** (you get a link and a code to approve on huggingface.co) or **by pasting an access token** from [your settings](https://huggingface.co/settings/tokens). The token is stored in `~/.cache/huggingface/`, outside the repo. Older guides call this command `huggingface-cli login`.

### 6. Open the notebooks

**JupyterLab:** with the environment activated, run

```bash
jupyter lab
```

It opens in your browser at `http://localhost:8888/lab`. Because it was started from `.env`, its **Python 3 (ipykernel)** kernel is the environment's Python.

**VS Code:** install the **Python** and **Jupyter** extensions, open the `hf-llm-course` folder, open a notebook, then **Select Kernel → Python Environments → .env**. Choosing the kernel is the notebook's equivalent of activating the environment.

## Check that it works

Open [00-setup/setup.ipynb](00-setup/setup.ipynb) and run all the cells. You should see:

- the Transformers and PyTorch versions, and `GPU available: True` if you have an NVIDIA GPU;
- the course's `transformers-course` project folder created inside the repo;
- a sentiment-analysis pipeline labelling *"I've been waiting for a HuggingFace course my whole life."* as `POSITIVE` (0.96) and *"I hate this so much!"* as `NEGATIVE` (0.999).

The first run downloads a small model (about 270 MB) to `~/.cache/huggingface/`.

Tested in November 2026 with:

| | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.15.0 (CUDA 13.2) |
| Transformers | 5.22.2 |
| Datasets | 5.1.0 |
| JupyterLab | 4.6.6 |
| huggingface_hub | 2.4.0 |

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Reinstall PyTorch with the command from [pytorch.org](https://pytorch.org/get-started/locally/) for your OS and CUDA version |
| A package installs in a notebook but won't import | `!pip` uses whichever `pip` is first on `PATH`. Use `%pip install …`, which installs into the notebook's own kernel |
| `!cd folder` has no effect | Each `!` line runs in its own short-lived shell. Use `%cd folder` to move the notebook, or a `%%bash` cell to run several shell lines together |
| `[Errno 2] No such file or directory` from `%cd transformers-course` | The kernel is already inside that folder from an earlier run. Restart the kernel and run the notebook from the top |
| `huggingface-cli: command not found` | The CLI is now called `hf`: `hf auth login` |
| `TqdmWarning: IProgress not found` | Install `ipywidgets` (it is in `requirements.txt`) and restart the kernel |

Never commit a Hugging Face token. `hf auth login` keeps it outside this folder, and `.gitignore` blocks common secret files.

## What's inside

One folder per chapter, numbered like the course:

```
00-setup/                  environment check and a first pipeline
  setup.ipynb
01-transformer-models/     what Transformers can do and how they work
  1.3-transformers-what-can-they-do.ipynb
02-using-transformers/     (added as I go)
...
12-build-reasoning-models/
```

Each chapter folder has the notebook(s) I worked in and a `notes.md` with the key takeaways and anything that tripped me up. Model weights, datasets and training outputs are never committed; they go in each chapter's `outputs/` folder (gitignored) and can be rebuilt from the notebooks.

## Course map

| Ch. | Topic | What it covers | Folder |
|---|---|---|---|
| 0 | [Setup](https://huggingface.co/learn/llm-course/chapter0/1) | Python environment, installing Transformers, Jupyter | [00-setup](00-setup/) |
| 1 | [Transformer models](https://huggingface.co/learn/llm-course/chapter1/1) | What Transformers and LLMs are, architectures, inference, limits | |
| 2 | [Using Transformers](https://huggingface.co/learn/llm-course/chapter2/1) | Pipelines, models, tokenizers, batching | |
| 3 | [Fine-tuning a pretrained model](https://huggingface.co/learn/llm-course/chapter3/1) | Trainer API, a full training loop, learning curves | |
| 4 | [Sharing models and tokenizers](https://huggingface.co/learn/llm-course/chapter4/1) | The Hub, model cards | |
| 5 | [The Datasets library](https://huggingface.co/learn/llm-course/chapter5/1) | Loading, slicing, big data, semantic search with FAISS | |
| 6 | [The Tokenizers library](https://huggingface.co/learn/llm-course/chapter6/1) | Training tokenizers, BPE, WordPiece, Unigram | |
| 7 | [Classical NLP tasks](https://huggingface.co/learn/llm-course/chapter7/1) | NER, masked LM, translation, summarization, causal LM, QA | |
| 8 | [How to ask for help](https://huggingface.co/learn/llm-course/chapter8/1) | Debugging and good issues | |
| 9 | [Building and sharing demos](https://huggingface.co/learn/llm-course/chapter9/1) | Gradio | |
| 10 | [Curate high-quality datasets](https://huggingface.co/learn/llm-course/chapter10/1) | Argilla | |
| 11 | [Fine-tune large language models](https://huggingface.co/learn/llm-course/chapter11/1) | Chat templates, SFT, LoRA, evaluation | |
| 12 | [Build reasoning models](https://huggingface.co/learn/llm-course/chapter12/1) | RL for LLMs, GRPO, DeepSeek R1 | |

## Notes

- [`CLAUDE.md`](CLAUDE.md) tells Claude Code how to help in this repo. I use it as a study partner while working through the chapters.
- Course content © Hugging Face, Apache 2.0. My notebooks and notes here are my own work.
