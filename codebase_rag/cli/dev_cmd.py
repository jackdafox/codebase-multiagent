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
        click.echo(json.dumps(state, indent=2))
        return

    status = state.get("status", "unknown")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    validation = state.get("validation", {})
    code = state.get("code", "")
    error = state.get("error")
    plan = state.get("plan", "")

    if format == "verbose":
        click.echo(f"Status: {status}")
        click.echo(f"Iterations: {iteration}/{max_iter}")
        if plan:
            click.echo(f"\n--- Architect Plan ---\n{plan}")
        if validation.get("errors"):
            click.echo(f"\n--- Validation Errors ({len(validation['errors'])}) ---")
            for err in validation["errors"]:
                click.echo(f"  - {err}")
        if validation.get("warnings"):
            click.echo(f"\n--- Warnings ({len(validation['warnings'])}) ---")
            for w in validation["warnings"]:
                click.echo(f"  - {w}")
        if validation.get("suggestions"):
            click.echo(f"\n--- Suggestions ({len(validation['suggestions'])}) ---")
            for s in validation["suggestions"]:
                click.echo(f"  - {s}")

    if code:
        click.echo("\n--- Generated Code ---\n")
        click.echo(code)
    elif error:
        click.echo(f"Error: {error}", err=True)
        sys.exit(1)

    if not validation.get("is_valid", False):
        click.echo(
            f"\nWarning: Pipeline did not converge after {iteration} iterations.",
            err=True,
        )
        if validation.get("errors"):
            click.echo("Validation errors:", err=True)
            for err in validation["errors"][:5]:
                click.echo(f"  - {err}", err=True)
