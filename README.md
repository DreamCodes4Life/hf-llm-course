# Hugging Face LLM Course — my notebooks

This repo is my working record of the free [Hugging Face LLM Course](https://huggingface.co/learn/llm-course): the code I ran, the experiments I tried and short notes on what each chapter taught me. It is the first course in a longer self-study plan towards AI engineering (LLMs, fine-tuning, inference and agents), so the aim is to understand each idea well enough to use it later, not just to get the notebooks to run.

Everything runs locally on one NVIDIA GPU (RTX 5090, 32 GB) with PyTorch and 🤗 Transformers. A [Colab](https://colab.research.google.com/) notebook works for the early chapters too.

## What's inside

One folder per chapter, numbered like the course:

```
00-setup/                  environment check and a first pipeline
01-transformer-models/     (added as I go)
...
12-build-reasoning-models/
```

Each chapter folder has the notebook(s) I worked in and a `notes.md` with the key takeaways and anything that tripped me up. Model weights, datasets and training outputs are never committed; they can be rebuilt from the notebooks.

## Course map

| Ch. | Topic | What it covers | Folder |
|---|---|---|---|
| 0 | [Setup](https://huggingface.co/learn/llm-course/chapter0/1) | Python environment, installing Transformers | [00-setup](00-setup/) |
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

## Running it

```bash
python3 -m venv .env            # or point .env at an existing virtual environment
source .env/bin/activate
pip install -r requirements.txt
python -c "import torch, transformers; print(transformers.__version__, torch.cuda.is_available())"
```

Open the folder in VS Code with the Python and Jupyter extensions and pick `.env/bin/python` as the kernel, or run `jupyter lab`.

On RTX 50-series cards, if `torch.cuda.is_available()` prints `False`, install the CUDA 12.8 build of PyTorch:
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`

## Notes

- Never commit a Hugging Face token. `huggingface-cli login` stores it outside this folder.
- [`CLAUDE.md`](CLAUDE.md) tells Claude Code how to help in this repo. I use it as a study partner while working through the chapters.
- Course content © Hugging Face, Apache 2.0. My notebooks and notes here are my own work.
