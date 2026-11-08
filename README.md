# Hugging Face LLM Course — my notebooks

My notes and code from working through the free [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), one notebook per chapter, run locally on an NVIDIA GPU.

## Setup

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python -c "import torch, transformers; print(transformers.__version__, torch.cuda.is_available())"
```

Then open the folder in VS Code (Python + Jupyter extensions) and pick `.env/bin/python` as the kernel, or run `jupyter lab`.

On RTX 50-series cards, if `torch.cuda.is_available()` is `False`, install the CUDA 12.8 build of PyTorch:
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`

## Progress

| Ch. | Topic | Notebook |
|---|---|---|
| 0 | [Setup](https://huggingface.co/learn/llm-course/chapter0/1) | [00-setup.ipynb](00-setup.ipynb) |
| 1 | [Transformer models](https://huggingface.co/learn/llm-course/chapter1/1) | |
| 2 | [Using Transformers](https://huggingface.co/learn/llm-course/chapter2/1) | |
| 3 | [Fine-tuning a pretrained model](https://huggingface.co/learn/llm-course/chapter3/1) | |
| 4 | [Sharing models and tokenizers](https://huggingface.co/learn/llm-course/chapter4/1) | |
| 5 | [The Datasets library](https://huggingface.co/learn/llm-course/chapter5/1) | |
| 6 | [The Tokenizers library](https://huggingface.co/learn/llm-course/chapter6/1) | |
| 7 | [Classical NLP tasks](https://huggingface.co/learn/llm-course/chapter7/1) | |
| 8 | [How to ask for help](https://huggingface.co/learn/llm-course/chapter8/1) | |
| 9 | [Building and sharing demos](https://huggingface.co/learn/llm-course/chapter9/1) | |
| 10 | [Curate high-quality datasets](https://huggingface.co/learn/llm-course/chapter10/1) | |
| 11 | [Fine-tune large language models](https://huggingface.co/learn/llm-course/chapter11/1) | |
| 12 | [Build reasoning models](https://huggingface.co/learn/llm-course/chapter12/1) | |

## Notes

- Model weights, datasets and training outputs are not committed (see `.gitignore`).
- Never commit a Hugging Face token. Log in with `huggingface-cli login`, which stores it outside this folder.
