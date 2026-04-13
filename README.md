# codebase-rag

A Retrieval Augmented Generation (RAG) system for codebases. Index source code with tree-sitter AST parsing, store chunks in ChromaDB, and query with vector similarity — or use the multi-agent pipeline to generate code from your codebase's context.

## Features

- **AST-based chunking** — tree-sitter extracts functions, classes, methods, types as semantic units (not fixed-size splits)
- **Per-language ChromaDB collections** — targeted retrieval per language
- **Multi-agent code generation** — LangGraph pipeline: Architect -> Engineer -> Validator loop with retry
- **CLI + library** — `rag index`, `rag query`, `rag dev` commands, or use as a Python package

## Quick Start

```bash
pip install -e .

# Index a codebase
rag index ./myproject -l python,go

# Query the index
rag query "how is authentication implemented" -l python

# Generate code with multi-agent pipeline
rag dev "add a health check endpoint" -l python
```

## Library Usage

```python
from codebase_rag import CodebaseIndexer, CodebaseRAG

# Index
indexer = CodebaseIndexer(persist_directory="./chroma_db")
result = indexer.index("/path/to/project")
print(f"Indexed {result.files_indexed} files, {result.chunks_created} chunks")

# Query
rag = CodebaseRAG(persist_directory="./chroma_db")
result = rag.query("how is the cache implemented?", language="python", n_results=5)

# Multi-agent code generation
from codebase_rag.agents.coordinator import MultiAgentCoordinator
coord = MultiAgentCoordinator(persist_directory="./chroma_db")
state = coord.run("add a health check endpoint", language="python")
print(state["code"])
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB data directory |
| `ARCHITECT_MODEL` | `gpt-4o` | Architect agent model |
| `ENGINEER_MODEL` | `gpt-4o` | Engineer agent model |
| `VALIDATOR_MODEL` | `gpt-4o` | Validator agent model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `AGENT_MAX_ITERATIONS` | `3` | Max Engineer <-> Validator loops |
| `AGENT_TIMEOUT_SECONDS` | `60` | Timeout per agent LLM call |

## CLI Commands

### `rag index <path>`
Index a codebase. Options: `-l/--language` (comma-separated), `-b/--batch-size`, `--no-progress`, `--glob` (JSON mapping).

### `rag query "<question>"`
Query the index. Options: `-l/--language`, `-n/--n-results`, `-f/--file-path`, `-o/--format` (text|json|markdown), `--context-only`.

### `rag dev "<request>"`
Multi-agent code generation (Architect -> Engineer -> Validator loop). Options: `-l/--language`, `-n/--max-iterations`, `-o/--format` (code|json|verbose).

## Architecture

```
CLI / Library
    |
    +-- CodebaseIndexer --> ChromaDB (per-language collections)
    |       |
    |       +-- CodeChunker (tree-sitter AST -> CodeChunk)
    |
    +-- CodebaseRAG -----> ChromaDB + Langchain wrapper
    |       |
    |       +-- OpenAI embeddings (text-embedding-3-small)
    |
    +-- MultiAgentCoordinator (LangGraph StateGraph)
            |
            +-- Architect --> Engineer --> Validator
                                      ^-- (retry loop) --+
```

## Supported Languages

Python, JavaScript, TypeScript, TSX, JSX, Go, Rust, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, Scala, Haskell, Lua, Bash, SQL, HTML, CSS, JSON, YAML, TOML, Markdown, Dockerfile, and more.
