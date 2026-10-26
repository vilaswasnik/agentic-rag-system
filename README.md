# Agentic RAG System

**Ask questions of your own documents — and let a team of AI agents find, read, and reason over the answers.** This is a retrieval-augmented generation (RAG) system in Python: ingest PDF, DOCX, and TXT files into a Chroma vector store, then query them through a multi-agent AutoGen pipeline, a single-system advanced mode, or a Gradio web UI with streaming answers and document upload.

Built with Microsoft AutoGen, LangChain, ChromaDB, and OpenAI models.

## What it does

- **Multi-agent answers** — three AutoGen agents collaborate on each question: an **Orchestrator** coordinates, a **DocumentRetriever** searches the vector store, and a **ContentAnalyzer** extracts and composes the answer (`src/agents.py`, configured in `config.yaml`).
- **Document ingestion** — drop files in `data/documents/` and run the ingestion script; documents are chunked (1000 tokens, 200 overlap) and embedded with OpenAI `text-embedding-3-small` into a persistent Chroma store (`src/document_processor.py`, `src/vectorstore.py`).
- **Gradio web UI** — chat with your documents in the browser with word-by-word streaming responses, source display, and an upload section that embeds new documents into the store immediately (`app.py`).
- **Advanced mode** — `advanced_rag.py` offers a single-system path with query, query-with-sources, and whole-collection summarisation.
- **YAML configuration** — model, temperature, embedding settings, vector-store location, and each agent's system message live in `config.yaml`, not in code.

## Quickstart

Prerequisites: Python 3.8+ and an OpenAI API key.

```bash
git clone https://github.com/vilaswasnik/agentic-rag-system.git
cd agentic-rag-system
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

Add documents and ingest them:

```bash
# put PDF / DOCX / TXT files in data/documents/, then:
python ingest_documents.py
```

Then pick an interface:

```bash
python main.py          # multi-agent AutoGen system (interactive CLI)
python advanced_rag.py  # advanced single-system mode (interactive)
python app.py           # Gradio web UI (chat + document upload)
./run.sh                # menu launcher for all modes, incl. the smoke test
```

> **Note:** the Gradio UI needs `gradio` installed (`pip install gradio`) — it is not yet listed in `requirements.txt`. The multi-agent system defaults (in `config.yaml`) are GPT-4o-mini for chat and `text-embedding-3-small` for embeddings; the Gradio UI currently instantiates its backend with GPT-3.5-turbo.

## How it works

1. **Ingest** — `DocumentProcessor` loads and chunks your documents; embeddings are written to Chroma under `./vectorstore` (collection `agentic_rag_collection`).
2. **Retrieve** — a question is embedded and matched against the store for the most relevant chunks.
3. **Reason** — in multi-agent mode, the Retriever agent fetches context and the Analyzer agent drafts the answer under the Orchestrator's coordination; in the UI and advanced modes, `AdvancedRAGSystem` performs retrieval and generation directly, returning sources alongside the answer.

Deeper detail lives in `ARCHITECTURE.md` and `RAG_IMPLEMENTATION.md`.

## Project layout

```text
main.py               Multi-agent (AutoGen) interactive CLI
advanced_rag.py       AdvancedRAGSystem: query, sources, summarisation
app.py                Gradio web UI (streaming chat + document upload)
ingest_documents.py   Build/refresh the vector store from data/documents/
query.py              Small direct-query example
example_rag_chain.py  Minimal LangChain RAG example
run.sh                Interactive launcher for all modes
config.yaml           LLM, embedding, vector store, and agent configuration
src/
  agents.py           AutoGen agent definitions
  document_processor.py  Loading and chunking
  vectorstore.py      Chroma store manager
data/documents/       Where your source documents go
```

## Configuration

- `.env` — `OPENAI_API_KEY` (required). See `.env.example`.
- `config.yaml` — LLM model/temperature, embedding model and chunking, vector-store directory and collection, and the three agents' names and system messages.

## Current limitations (stated up front)

- OpenAI-only: models and embeddings are OpenAI's; there is no local-model path yet.
- Answers are only as good as the ingested documents — with an empty store, there is nothing to retrieve.
- The three entry points evolved at different times and use slightly different model defaults (see the note in Quickstart); `config.yaml` is authoritative for the multi-agent CLI.

## Contributing

Contributions are welcome — see `CONTRIBUTING.md`. Please don't commit API keys; the key belongs in your local `.env` only.

## License

MIT — see `LICENSE`.
