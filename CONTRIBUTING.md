# Contributing to Agentic RAG System

Thanks for looking. This is a Python retrieval-augmented generation system with three entry points — an AutoGen multi-agent CLI (`main.py`), an advanced single-system mode (`advanced_rag.py`), and a Gradio web UI (`app.py`) — over a shared LangChain/Chroma pipeline in `src/`. Contributions of code, docs, and bug reports are all welcome.

## Run it locally

Prerequisites: Python 3.8+ and an OpenAI API key (the system is OpenAI-only today).

```bash
git clone https://github.com/vilaswasnik/agentic-rag-system.git
cd agentic-rag-system
pip install -r requirements.txt
pip install gradio        # needed for the web UI; not yet in requirements.txt
cp .env.example .env      # add your OPENAI_API_KEY
python ingest_documents.py
python main.py            # or: python app.py for the web UI
```

Ground rules:

- **The API key lives in `.env` only** — never in code, config, or a commit. `config.yaml` is for non-secret settings.
- **Keep the three entry points consistent.** If you change models, chunking, or store settings, check all three (`main.py`/`config.yaml`, `advanced_rag.py`, `app.py`) — they have drifted apart before.
- **No absolute paths from your machine.** Paths must be relative to the repo (a leftover `/workspaces/...` path in `app.py`'s data directory is a known bug, not a pattern to copy).
- New runtime imports go in `requirements.txt` in the same PR that introduces them.

## Filing an issue

Use GitHub Issues. A useful bug report includes: which entry point (CLI / advanced / UI), what documents you ingested (format and rough size), your question, what the system answered, and any traceback. For retrieval-quality issues, include the sources the UI/advanced mode returned.

## PR process

1. Fork, branch from `main`, keep the change focused.
2. Run `python test_system.py` and exercise the entry point you touched with a small document set.
3. Say in the PR description what you ran and what the system answered.
4. One approval from the maintainer merges.

## Good first issues

Concrete, evidence-based starting points (each cites where the gap is):

1. **Fix the dependency manifest.** `app.py` imports `gradio` and `main.py` imports `yaml`, but neither is declared in `requirements.txt` (`gradio` doesn't arrive transitively at all). Add both and verify a clean-venv install can start every entry point.
2. **Replace the hardcoded UI data path.** `app.py` sets its data directory to an absolute path from another workspace (`/workspaces/sample.ai/data/documents`). Point it at the repo's `data/documents/` like the rest of the system.
3. **Unify the model defaults.** `.env.example` (gpt-4 / text-embedding-ada-002), `config.yaml` (gpt-4o-mini / text-embedding-3-small), and `app.py` (gpt-3.5-turbo) disagree. Make one source of truth and have the others defer to it.
4. **Add a retrieval smoke test.** Extend `test_system.py` to ingest one tiny document and assert that a question about it returns a source-backed answer, so pipeline regressions are catchable without manual runs.
