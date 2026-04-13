# Project Memory — codebase-rag

## Overview

Built a full RAG system for codebases using ChromaDB, Langchain, and tree-sitter.
Repo: `https://github.com/jackdafox/codebase-multiagent.git`

## Key Decisions Made

- **Per-language ChromaDB collections** — each language gets its own collection (`codebase_python_chunks`, etc.)
- **OpenAI `text-embedding-3-small`** for embeddings
- **tree-sitter** for AST parsing and chunk extraction (functions, classes, methods, types)
- **CLI + Python library** interface
- **`.env` loading**: Uses `Path.cwd() / ".env"` in the `cli()` callback — `__file__`-based paths break when running as an installed `.exe`
- **Max chunk size**: `MAX_CHUNK_CHARS = 24_000` — oversized AST nodes are split by children to stay under OpenAI's 8192-token limit
- **Langchain integration**: `CodebaseRAG` is compatible but not yet a full Langchain vectorstore wrapper

## Important Bugs Fixed

1. **chunks_by_file vs chunks_by_lang** — `index()` passed `{path: chunks}` to `_upsert_chunks` which expected `{lang: chunks}`. Caused ChromaDB collection names to be file paths → validation error.
2. **embeddings not passed to ChromaDB** — `_embedding_factory.embed_texts()` was called but result was discarded. Fixed to pass embeddings to `collection.upsert()`.
3. **import ordering in cli/main.py** — `.env` loading was in `main()` which Click bypasses. Moved into `cli()` callback.
4. **Path.cwd() for .env** — `__file__`-based path breaks in installed exe. Fixed to use `Path.cwd() / ".env"`.
5. **trufflehog pip package** — only supports git URL scanning, not filesystem. Switched to `gitleaks-action`.
6. **pip-audit --fail** — flag doesn't exist. pip-audit exits non-zero automatically on vulnerabilities.

## Current State

- All 29 unit tests passing
- `ruff`, `mypy --ignore-missing-imports`, `bandit` — all clean
- `pip-audit` — no vulnerabilities
- `gitleaks` — no secrets detected
- 2 active branches: `master` (pushed to origin), clean working tree

## User Context

- Runs on Windows (`.venv/Scripts/rag`)
- Has OpenAI API key in `.env`
- Obsidian vault at `C:\Users\serve\obsidian-vault\research\Research`
