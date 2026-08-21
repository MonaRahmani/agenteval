import typer

from agenteval import __version__

app = typer.Typer(
    help="Seed reproducible GitHub test environments for evaluating AI coding agents."
)


@app.callback()
def main() -> None:
    """Seed reproducible GitHub test environments for evaluating AI coding agents."""


@app.command()
def version() -> None:
    """Print the agenteval version."""
    typer.echo(__version__)
