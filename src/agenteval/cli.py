"""Command-line interface for agenteval."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, ParamSpec, TypeVar

import typer

from agenteval import __version__
from agenteval.github_client import GitHubClient, GitHubClientError
from agenteval.scenario import ScenarioError, load_scenario
from agenteval.seeder import reset as reset_repo
from agenteval.seeder import seed as seed_scenario
from agenteval.seeder import verify_deterministic

app = typer.Typer(
    help="Seed reproducible GitHub test environments for evaluating AI coding agents."
)

P = ParamSpec("P")
R = TypeVar("R")


def _configure_logging() -> None:
    """Send progress to stdout; errors go to stderr via typer.secho."""
    root = logging.getLogger("agenteval")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def handle_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Turn expected failures into a one-line message and a non-zero exit."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (ScenarioError, GitHubClientError) as exc:
            first_line = str(exc).strip().splitlines()[0]
            detail = str(exc).strip()
            typer.secho(f"error: {first_line}", err=True, fg=typer.colors.RED)
            for extra in detail.splitlines()[1:]:
                typer.secho(extra, err=True, fg=typer.colors.RED)
            raise typer.Exit(1) from None

    return wrapper


@app.callback()
def main() -> None:
    """Seed reproducible GitHub test environments for evaluating AI coding agents."""
    _configure_logging()


@app.command()
def version() -> None:
    """Print the agenteval version."""
    typer.echo(__version__)


@app.command()
@handle_errors
def validate(
    scenario_path: Annotated[Path, typer.Argument(help="Path to a scenario YAML file.")],
) -> None:
    """Validate a scenario without touching the network."""
    scenario = load_scenario(scenario_path)
    shas = verify_deterministic(scenario)

    typer.secho(f"OK  {scenario_path}", fg=typer.colors.GREEN)
    typer.echo(f"  name       : {scenario.name}")
    typer.echo(f"  description: {scenario.description}")
    typer.echo(f"  rubric     : {scenario.rubric}")
    typer.echo(f"  files      : {len(scenario.files)}")
    typer.echo(f"  commits    : {len(scenario.commits)}")
    typer.echo("")
    typer.echo("  expected commits (computed offline):")
    for index, (commit, sha) in enumerate(zip(scenario.commits, shas, strict=True)):
        touched = len(commit.files)
        total = len(scenario.content_at(index))
        typer.echo(
            f"    {sha[:10]}  {commit.author_date.date()}  "
            f"{touched} changed / {total} total  {commit.message}"
        )


@app.command()
@handle_errors
def seed(
    scenario_path: Annotated[Path, typer.Argument(help="Path to a scenario YAML file.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Log what would happen; make no writes.")
    ] = False,
) -> None:
    """Seed a scenario into a fresh GitHub repository."""
    scenario = load_scenario(scenario_path)
    client = GitHubClient(dry_run=dry_run)

    if dry_run:
        typer.secho("dry run: no writes will be made", fg=typer.colors.YELLOW)

    result = seed_scenario(scenario, client)

    typer.echo("")
    typer.secho(f"seeded {result.repo_name}", fg=typer.colors.GREEN)
    typer.echo(f"  url: {result.repo_url}")
    typer.echo("  commits:")
    for sha, commit in zip(result.commit_shas, scenario.commits, strict=True):
        typer.echo(f"    {sha}  {commit.message}")


@app.command()
@handle_errors
def reset(
    name: Annotated[str, typer.Argument(help="Name of the seeded repo to delete.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete a seeded repo. Refuses repos agenteval did not create."""
    if not yes:
        typer.confirm(f"Delete repo {name!r}? This cannot be undone.", abort=True)

    reset_repo(name, GitHubClient())
