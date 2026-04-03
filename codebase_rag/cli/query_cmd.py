"""The `rag query` CLI command."""

from __future__ import annotations

import json
import sys

import click

from codebase_rag.retrieval.rag import CodebaseRAG


@click.command("query")
@click.argument("question", default=None, required=False)
@click.option(
    "--question",
    "-q",
    "question_opt",
    help="Query string (alternative to positional arg).",
    default=None,
)
@click.option(
    "--language",
    "-l",
    help="Restrict search to a specific language.",
    default=None,
)
@click.option(
    "--file",
    "-f",
    "files",
    multiple=True,
    help="Restrict search to specific file(s).",
)
@click.option(
    "--n-results",
    "-n",
    type=int,
    default=5,
    help="Number of results to retrieve.",
)
@click.option(
    "--format",
    "-o",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--context-only",
    is_flag=True,
    help="Output only the context string (for piping to LLMs).",
)
@click.pass_context
def query(
    ctx: click.Context,
    question: str | None,
    question_opt: str | None,
    language: str | None,
    files: tuple[str, ...],
    n_results: int,
    format: str,
    context_only: bool,
) -> None:
    """Query the indexed codebase."""
    persist_dir: str = ctx.obj["persist_dir"]

    # Resolve question
    q = question or question_opt
    if not q:
        # Interactive mode
        q = click.prompt("Enter your question")

    rag = CodebaseRAG(persist_directory=persist_dir)

    # Check if there are any indexed languages
    indexed_langs = rag.get_all_indexed_languages()
    if not indexed_langs:
        click.echo("Error: No indexed codebases found. Run `rag index` first.", err=True)
        sys.exit(1)

    file_paths = list(files) if files else None

    if context_only:
        context = rag.get_context(
            question=q,
            language=language,
            file_paths=file_paths,
            n_results=n_results,
        )
        click.echo(context)
        return

    result = rag.query(
        question=q,
        language=language,
        file_paths=file_paths,
        n_results=n_results,
        include_chunks=True,
        include_scores=True,
    )

    if format == "json":
        output = []
        for chunk in result.chunks:
            output.append({
                "id": chunk.id,
                "text": chunk.text,
                "language": chunk.language,
                "node_type": chunk.node_type,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "fully_qualified_name": chunk.fully_qualified_name,
                "score": chunk.score,
            })
        click.echo(json.dumps(output, indent=2))
        return

    if not result.chunks:
        click.echo("No results found.")
        return

    for i, chunk in enumerate(result.chunks, 1):
        header = f"[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        if chunk.fully_qualified_name:
            header += f" ({chunk.fully_qualified_name})"
        header += f" [{chunk.node_type}]"
        if chunk.score is not None:
            header += f" score={chunk.score:.4f}"

        if format == "markdown":
            click.echo(f"\n## {header}\n")
        else:
            click.echo(f"\n{header}\n")

        click.echo(chunk.text)

        if format == "markdown":
            click.echo("\n```")
