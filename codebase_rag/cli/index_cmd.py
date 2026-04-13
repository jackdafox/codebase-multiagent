"""The `rag index` CLI command."""

from __future__ import annotations

import json

import click

from codebase_rag.indexing.indexer import CodebaseIndexer


@click.command("index")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--languages",
    "-l",
    default=None,
    help="Comma-separated languages to index (default: all supported).",
)
@click.option(
    "--glob",
    "-g",
    default=None,
    help="JSON dict of language->glob pattern overrides.",
)
@click.option(
    "--batch-size",
    "-b",
    default=100,
    type=int,
    help="Files per embedding batch.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bars.",
)
@click.pass_context
def index(
    ctx: click.Context,
    path: str,
    languages: str | None,
    glob: str | None,
    batch_size: int,
    no_progress: bool,
) -> None:
    """Index a codebase directory into ChromaDB."""
    persist_dir: str = ctx.obj["persist_dir"]

    # Parse languages
    lang_list: list[str] | None = None
    if languages:
        lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    # Parse glob overrides
    glob_patterns: dict[str, str] | None = None
    if glob:
        try:
            glob_patterns = json.loads(glob)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON for --glob: {e}")

    indexer = CodebaseIndexer(
        persist_directory=persist_dir,
        batch_size=batch_size,
    )

    click.echo(f"Indexing {path}...")
    result = indexer.index(
        root_path=path,
        languages=lang_list,
        glob_patterns=glob_patterns,
        show_progress=not no_progress,
    )

    click.echo("\nIndexing complete:")
    click.echo(f"  Files indexed:  {result.files_indexed}")
    click.echo(f"  Chunks created: {result.chunks_created}")
    click.echo(f"  Languages:     {', '.join(sorted(result.languages_found)) or 'none'}")
    click.echo(f"  Duration:       {result.duration_seconds:.2f}s")

    if result.errors:
        click.echo(f"\nErrors ({len(result.errors)}):")
        for err in result.errors[:10]:
            click.echo(f"  {err['file']}: {err['error']}")
        if len(result.errors) > 10:
            click.echo(f"  ... and {len(result.errors) - 10} more errors")
