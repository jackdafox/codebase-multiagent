"""CLI entry point for codebase_rag."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from codebase_rag import __version__
from codebase_rag.cli.index_cmd import index
from codebase_rag.cli.query_cmd import query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--persist-dir",
    default="./chroma_db",
    help="ChromaDB persistence directory.",
    envvar="CHROMA_PERSIST_DIR",
)
@click.pass_context
def cli(ctx: click.Context, persist_dir: str) -> None:
    """codebase-rag — RAG for codebases using ChromaDB, Langchain, and tree-sitter."""
    ctx.ensure_object(dict)
    ctx.obj["persist_dir"] = persist_dir


cli.add_command(index)
cli.add_command(query)


def main() -> None:
    """Entry point for the rag CLI command."""
    # Load .env file from project root
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
    try:
        cli()
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
