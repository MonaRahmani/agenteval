def test_package_version() -> None:
    import agenteval

    assert isinstance(agenteval.__version__, str)
    assert agenteval.__version__


def test_cli_app_exists() -> None:
    import typer

    from agenteval.cli import app

    assert app is not None
    assert isinstance(app, typer.Typer)


def test_cli_version_command() -> None:
    from typer.testing import CliRunner

    from agenteval import __version__
    from agenteval.cli import app

    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0, result.output
    assert __version__ in result.stdout
