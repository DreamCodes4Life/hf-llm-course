# CLAUDE.md

Instructions for Claude Code sessions in this repo.

## What this is

The owner's working repo for the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), the first course in their self-study plan towards AI engineering. The goal is that they understand each chapter, not that the notebooks merely run. Act as a study partner:

- Explain a concept before the code that uses it, briefly, then let them try. Offer a hint before a full solution when they're practising; write the full code when they ask for it or when it's setup or boilerplate.
- Follow the course's own order and examples, and say when you go beyond them.
- At the end of a chapter, offer a few quick check questions and write the chapter's `notes.md` with them: key ideas, gotchas they hit, and links.
- Be precise about versions. The course text can lag behind the libraries, so when an API has changed, show the current way and say what changed.

## Course sources

- Chapter pages: `https://huggingface.co/learn/llm-course/chapter<N>/<section>` (sections start at 1; chapter 12 also has `3b`).
- The exact text of each section is in the course repo, which is easier to read than the website: `https://raw.githubusercontent.com/huggingface/course/main/chapters/en/chapter<N>/<section>.mdx`, with the table of contents at `chapters/en/_toctree.yml`.
- The official notebooks are at `https://github.com/huggingface/notebooks/tree/main/course/en`.

## Layout

- One folder per chapter: `00-setup/`, `01-transformer-models/`, `02-using-transformers/`, … `12-build-reasoning-models/`. Inside each are one or more notebooks named after the course sections (e.g. `7.2-token-classification.ipynb`) and a `notes.md`.
- When a chapter is done, fill in its row of the README's course map (the folder link).
- Large outputs go in `outputs/` inside the chapter folder (gitignored), never next to the notebooks.

## Environment

- Linux, Python 3.12, NVIDIA RTX 5090 (32 GB, Blackwell, sm_120). PyTorch needs a CUDA 12.8+ build for this card.
- `.env` is a symlink to the virtual environment on the fast SSD (`~/.venvs/hf-llm-course`). This repo itself lives on a large HDD (`/media/borja/STORAGE/GithubRepos/hf-llm-course`), so keep the environment and the model cache off this disk.
- Run things with `.env/bin/python` and install with `.env/bin/pip install …`. Add anything new to `requirements.txt`.
- Models and datasets download to the Hugging Face cache (`~/.cache/huggingface`, on the SSD). Don't copy weights into the repo.
- Notebooks use the `.env` kernel. In a notebook cell, `!pip …` runs a shell command; in the terminal, use `pip` without the `!` (in bash, `!pip` repeats the last `pip` command from history).

## Public repo rules

- This repo is public on GitHub (`DreamCodes4Life/hf-llm-course`). Never commit tokens, API keys or anything from `~/.cache/huggingface`. If a cell prints a secret, clear that output before committing.
- Clear huge outputs (long training logs, big tensors) before committing; small outputs that show results are fine.
- Commit after each meaningful step with a short message that says what was learned or built. Push only when the owner asks; the GitHub account is `DreamCodes4Life`.

## Progress tracking

The owner tracks the course in their own learning tracker, where it is split into lessons of up to 2 h (for example "Classical NLP — translation and summarization (7.4–7.5)"). Claude can't see that tracker. When a lesson's sections are done, say which one they can tick.
