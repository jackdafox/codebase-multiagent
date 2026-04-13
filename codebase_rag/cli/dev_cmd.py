"""The `rag dev` CLI command -- multi-agent code generation."""

from __future__ import annotations

import json
import logging
import sys

import click

from codebase_rag.agents.coordinator import MultiAgentCoordinator

logger = logging.getLogger(__name__)


@click.command("dev")
@click.argument("query", default=None, required=False)
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
    default="python",
    help="Target programming language.",
)
@click.option(
    "--max-iterations",
    "-n",
    type=int,
    default=3,
    help="Max Engineer<->Validator loops.",
)
@click.option(
    "--format",
    "-o",
    type=click.Choice(["code", "json", "verbose"]),
    default="code",
    help="Output format.",
)
@click.pass_context
def dev(
    ctx: click.Context,
    query: str | None,
    question_opt: str | None,
    language: str,
    max_iterations: int,
    format: str,
) -> None:
    """Multi-agent code generation: Architect -> Engineer -> Validator loop.

    Query is required -- either as positional argument or via --question / -q.
    """
    persist_dir: str = ctx.obj["persist_dir"]

    q = query or question_opt
    if not q:
        click.echo("Error: query is required. Use positional arg or --question.", err=True)
        sys.exit(1)

    click.echo(f"Running multi-agent pipeline for: {q}\n")

    coordinator = MultiAgentCoordinator(
        persist_directory=persist_dir,
        max_iterations=max_iterations,
    )

    try:
        state = coordinator.run(query=q, language=language)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        click.echo(f"Pipeline error: {exc}", err=True)
        sys.exit(1)

    if format == "json":
        click.echo(json.dumps(state.to_dict(), indent=2))
        return

    if format == "verbose":
        click.echo(f"Status: {state.status.value}")
        click.echo(f"Iterations: {state.iteration}/{state.max_iterations}")
        if state.plan:
            click.echo(f"\n--- Architect Plan ---\n{state.plan}")
        if state.validation.errors:
            click.echo(f"\n--- Validation Errors ({len(state.validation.errors)}) ---")
            for err in state.validation.errors:
                click.echo(f"  • {err}")
        if state.validation.warnings:
            click.echo(f"\n--- Warnings ({len(state.validation.warnings)}) ---")
            for w in state.validation.warnings:
                click.echo(f"  • {w}")
        if state.validation.suggestions:
            click.echo(f"\n--- Suggestions ({len(state.validation.suggestions)}) ---")
            for s in state.validation.suggestions:
                click.echo(f"  • {s}")

    if state.code:
        click.echo("\n--- Generated Code ---\n")
        click.echo(state.code)
    elif state.error:
        click.echo(f"Error: {state.error}", err=True)
        sys.exit(1)

    if not state.validation.is_valid:
        click.echo(
            f"\n⚠ Pipeline did not converge after {state.iteration} iterations.",
            err=True,
        )
        if state.validation.errors:
            click.echo("Validation errors:", err=True)
            for err in state.validation.errors[:5]:
                click.echo(f"  • {err}", err=True)
