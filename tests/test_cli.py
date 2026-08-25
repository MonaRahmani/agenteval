import contextlib
import re
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
import yaml
from github import UnknownObjectException
from typer.testing import CliRunner

from agenteval import github_client as gc
from agenteval.cli import app
from agenteval.github_client import MANAGED_TOPIC, TOKEN_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SCENARIO = REPO_ROOT / "scenarios" / "failing-import.yaml"

runner = CliRunner()


def output(result: object) -> str:
    """Everything the command printed, stdout and stderr alike."""
    text = getattr(result, "output", "") or ""
    with contextlib.suppress(ValueError):  # stderr may not be captured separately
        text += getattr(result, "stderr", "") or ""
    return text


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)


@pytest.fixture
def fake_github(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock PyGithub so no CLI test can reach the network."""
    monkeypatch.setenv(TOKEN_ENV_VAR, "t0ken")
    github_cls = MagicMock(name="Github")
    monkeypatch.setattr(gc, "Github", github_cls)

    instance = cast(MagicMock, github_cls.return_value)
    user = instance.get_user.return_value
    user.login = "octocat"
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})
    return instance


def write_scenario(tmp_path: Path, data: object, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- version ---------------------------------------------------------------


def test_version_still_works() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "validate", "seed", "reset"):
        assert command in result.output


# --- validate --------------------------------------------------------------


def test_validate_shipped_scenario_exits_zero() -> None:
    result = runner.invoke(app, ["validate", str(SHIPPED_SCENARIO)])
    assert result.exit_code == 0, output(result)


def test_validate_prints_a_summary() -> None:
    result = runner.invoke(app, ["validate", str(SHIPPED_SCENARIO)])

    assert "failing-import" in result.output
    assert "files" in result.output
    assert "commits" in result.output
    assert "rubrics/failing-import.yaml" in result.output
    for message in ("Add inventory package", "Reuse report.format_currency"):
        assert message in result.output


def test_validate_prints_expected_shas() -> None:
    from agenteval.scenario import load_scenario
    from agenteval.seeder import verify_deterministic

    shas = verify_deterministic(load_scenario(SHIPPED_SCENARIO))
    result = runner.invoke(app, ["validate", str(SHIPPED_SCENARIO)])

    for sha in shas:
        assert sha[:10] in result.output


def test_validate_needs_no_token_or_network() -> None:
    """No GITHUB_TOKEN is set by the autouse fixture, and PyGithub is untouched."""
    result = runner.invoke(app, ["validate", str(SHIPPED_SCENARIO)])
    assert result.exit_code == 0, output(result)


def test_validate_malformed_yaml_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\nfiles: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code != 0
    assert "Malformed YAML" in output(result)


def test_validate_invalid_scenario_exits_nonzero(tmp_path: Path) -> None:
    path = write_scenario(
        tmp_path,
        {
            "name": "bad name",
            "description": "d",
            "seeded_bug": "b",
            "rubric": "r.yaml",
            "files": [{"path": "a.py", "description": "x"}],
            "commits": [
                {
                    "message": "m",
                    "author_date": "2026-01-05T09:00:00Z",
                    "files": [{"path": "a.py", "content": "x\n"}],
                }
            ],
        },
    )

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code != 0
    assert "invalid repo name" in output(result)


def test_validate_error_shows_no_traceback(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\nfiles: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path)])

    text = output(result)
    assert "Traceback" not in text
    assert "pydantic" not in text.lower()
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_validate_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])

    assert result.exit_code != 0
    assert "not found" in output(result)
    assert "Traceback" not in output(result)


# --- seed ------------------------------------------------------------------


def test_seed_dry_run_performs_no_write_calls(fake_github: MagicMock) -> None:
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    assert result.exit_code == 0, output(result)

    user = fake_github.get_user.return_value
    user.create_repo.assert_not_called()
    assert not user.create_repo.return_value.method_calls


def test_seed_dry_run_says_it_is_a_dry_run(fake_github: MagicMock) -> None:
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    assert "dry run" in result.output.lower()


def test_seed_dry_run_prints_one_sha_per_commit(fake_github: MagicMock) -> None:
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    for message in ("Add inventory package", "Reuse report.format_currency"):
        assert message in result.output


def test_seed_dry_run_does_not_print_a_live_url(fake_github: MagicMock) -> None:
    """Nothing was created, so no URL should be presented as if it existed."""
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    assert "https://github.com/octocat/failing-import" not in output(result)
    assert "url:" not in output(result)


def test_seed_dry_run_needs_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Previewing a seed must not require credentials."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    text = output(result)
    assert result.exit_code == 0, text
    assert "error:" not in text
    assert "No GitHub token available" not in text
    # The unauthenticated notice is informational, not a failure.
    assert "unauthenticated" in text.lower()


def test_seed_dry_run_without_token_prints_planned_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    text = output(result)

    assert result.exit_code == 0, text
    for message in (
        "Add inventory package with Item model and Store",
        "Add tests covering Store totals and duplicate SKUs",
        "Add report module for formatted stock listings",
        "Reuse report.format_currency in Item.summary",
    ):
        assert message in text, f"missing planned commit: {message}"


def test_seed_dry_run_without_token_never_constructs_a_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    github_cls = MagicMock(name="Github")
    monkeypatch.setattr(gc, "Github", github_cls)

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    assert result.exit_code == 0, output(result)
    github_cls.assert_not_called()


def test_seed_without_dry_run_and_no_token_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real path still demands credentials."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO)])

    assert result.exit_code != 0
    assert TOKEN_ENV_VAR in output(result)
    assert "Traceback" not in output(result)


def test_seed_on_malformed_scenario_never_builds_a_client(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\nfiles: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["seed", str(path)])

    assert result.exit_code != 0
    assert "Malformed YAML" in output(result)


# --- reset -----------------------------------------------------------------


def test_reset_without_yes_prompts(fake_github: MagicMock) -> None:
    result = runner.invoke(app, ["reset", "failing-import"], input="n\n")

    assert "Delete repo" in output(result)
    assert result.exit_code != 0  # aborted
    fake_github.get_user.return_value.get_repo.assert_not_called()


def test_reset_prompt_accepted_proceeds(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    repo = MagicMock()
    repo.get_topics.return_value = [MANAGED_TOPIC]
    user.get_repo.side_effect = None
    user.get_repo.return_value = repo

    result = runner.invoke(app, ["reset", "failing-import"], input="y\n")

    assert result.exit_code == 0, output(result)
    repo.delete.assert_called_once_with()


def test_reset_with_yes_does_not_prompt(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    repo = MagicMock()
    repo.get_topics.return_value = [MANAGED_TOPIC]
    user.get_repo.side_effect = None
    user.get_repo.return_value = repo

    result = runner.invoke(app, ["reset", "failing-import", "--yes"])

    assert result.exit_code == 0, output(result)
    assert "Delete repo" not in output(result)
    repo.delete.assert_called_once_with()


def test_reset_missing_repo_exits_zero(fake_github: MagicMock) -> None:
    result = runner.invoke(app, ["reset", "never-existed", "--yes"])

    assert result.exit_code == 0, output(result)
    assert "does not exist" in output(result)


def test_reset_refuses_unmarked_repo(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    repo = MagicMock()
    repo.get_topics.return_value = ["production"]
    user.get_repo.side_effect = None
    user.get_repo.return_value = repo

    result = runner.invoke(app, ["reset", "real-work", "--yes"])

    assert result.exit_code != 0
    assert "Refusing to delete" in output(result)
    assert "Traceback" not in output(result)
    repo.delete.assert_not_called()


# --- dry-run and validate must agree on SHAs -------------------------------


def shas_from_seed_output(text: str) -> list[str]:
    """Full SHAs in first-seen order, as `seed` prints them."""
    return list(dict.fromkeys(re.findall(r"\b[0-9a-f]{40}\b", text)))


def prefixes_from_validate_output(text: str) -> list[str]:
    return re.findall(r"^ {4}([0-9a-f]{10})  ", text, re.M)


def test_dry_run_seed_and_validate_print_identical_shas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preview that disagrees with the prediction would undermine the whole tool."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    seeded = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    validated = runner.invoke(app, ["validate", str(SHIPPED_SCENARIO)])

    assert seeded.exit_code == 0, output(seeded)
    assert validated.exit_code == 0, output(validated)

    seed_shas = shas_from_seed_output(output(seeded))
    validate_prefixes = prefixes_from_validate_output(output(validated))

    assert len(validate_prefixes) == 4
    assert [sha[:10] for sha in seed_shas] == validate_prefixes


def test_dry_run_seed_shas_match_verify_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agenteval.scenario import load_scenario
    from agenteval.seeder import verify_deterministic

    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    expected = verify_deterministic(load_scenario(SHIPPED_SCENARIO))

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    assert shas_from_seed_output(output(result)) == expected


def test_dry_run_output_contains_no_placeholder_shas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every 40-hex string printed must be a real predicted SHA."""
    from agenteval.scenario import load_scenario
    from agenteval.seeder import verify_deterministic

    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    predicted = set(verify_deterministic(load_scenario(SHIPPED_SCENARIO)))

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    printed = set(re.findall(r"\b[0-9a-f]{40}\b", output(result)))

    assert printed, "no SHAs printed at all"
    assert printed - predicted == set(), "placeholder SHAs leaked into dry-run output"


def test_dry_run_is_stable_across_invocations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    first = shas_from_seed_output(
        output(runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"]))
    )
    second = shas_from_seed_output(
        output(runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"]))
    )

    assert first == second


# --- an existing repo must not block `seed --dry-run` ----------------------


@pytest.fixture
def existing_repo_github(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """PyGithub mocked so the target repo already exists on the remote."""
    monkeypatch.setenv(TOKEN_ENV_VAR, "t0ken")
    github_cls = MagicMock(name="Github")
    monkeypatch.setattr(gc, "Github", github_cls)

    instance = cast(MagicMock, github_cls.return_value)
    user = instance.get_user.return_value
    user.login = "MonaRahmani"
    user.get_repo.return_value = MagicMock(name="ExistingRepo")  # lookup succeeds
    return instance


def test_dry_run_against_an_existing_repo_exits_zero(existing_repo_github: MagicMock) -> None:
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    assert result.exit_code == 0, output(result)
    assert "error:" not in output(result)


def test_dry_run_against_an_existing_repo_prints_all_predicted_shas(
    existing_repo_github: MagicMock,
) -> None:
    from agenteval.scenario import load_scenario
    from agenteval.seeder import verify_deterministic

    expected = verify_deterministic(load_scenario(SHIPPED_SCENARIO))

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    text = output(result)

    assert result.exit_code == 0, text
    assert shas_from_seed_output(text) == expected
    for sha in expected:
        assert sha in text, f"predicted SHA missing from output: {sha}"


def test_dry_run_against_an_existing_repo_says_a_live_run_would_refuse(
    existing_repo_github: MagicMock,
) -> None:
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    text = output(result)

    assert "already exists" in text
    assert "would refuse" in text
    assert "agenteval reset failing-import" in text


def test_dry_run_against_an_existing_repo_makes_no_writes(
    existing_repo_github: MagicMock,
) -> None:
    runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    existing_repo_github.get_user.return_value.create_repo.assert_not_called()


def test_live_seed_against_an_existing_repo_still_exits_nonzero(
    existing_repo_github: MagicMock,
) -> None:
    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO)])
    text = output(result)

    assert result.exit_code != 0
    assert "already exists" in text
    assert "agenteval reset failing-import" in text
    assert "Traceback" not in text
    existing_repo_github.get_user.return_value.create_repo.assert_not_called()


def test_logging_survives_repeated_invocations(fake_github: MagicMock) -> None:
    """A cached StreamHandler would write to the first invocation's stdout."""
    first = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])
    second = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"])

    for result in (first, second):
        text = output(result)
        assert result.exit_code == 0, text
        assert "Logging error" not in text
        assert "would create repo" in text, "log output lost on a repeat invocation"

    assert shas_from_seed_output(output(first)) == shas_from_seed_output(output(second))


# --- dry-run output hygiene ------------------------------------------------


def test_dry_run_summary_does_not_claim_anything_was_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    text = output(runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"]))

    assert "seeded" not in text.lower()
    assert "predicted history" in text
    assert "nothing was written" in text


def test_live_run_summary_still_says_seeded(fake_github: MagicMock) -> None:
    """The live wording is unchanged; only the dry-run branch differs."""
    user = fake_github.get_user.return_value
    created = user.create_repo.return_value
    created.html_url = "https://github.com/octocat/failing-import"
    created.default_branch = "main"
    created.create_git_blob.return_value = MagicMock(sha="blob")
    created.create_git_tree.return_value = MagicMock(sha="tree")
    created.create_git_commit.return_value = MagicMock(sha="c" * 40)

    result = runner.invoke(app, ["seed", str(SHIPPED_SCENARIO)])

    text = output(result)
    assert result.exit_code == 0, text
    assert "seeded failing-import" in text
    assert "url: https://github.com/octocat/failing-import" in text


def test_commit_messages_have_no_trailing_whitespace_in_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Messages carry a trailing newline for git; it must not reach the user."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    text = output(runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"]))

    assert "\\n" not in text, "a literal escaped newline leaked into output"
    for line in text.splitlines():
        assert line == line.rstrip(), f"trailing whitespace in: {line!r}"


def test_dry_run_reports_each_commit_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's line and the seeder's progress line used to duplicate."""
    from agenteval.scenario import load_scenario

    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    scenario = load_scenario(SHIPPED_SCENARIO)

    text = output(runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"]))

    assert "would commit" in text
    assert not re.search(r"^commit \d+/\d+ ", text, re.M), "seeder progress line duplicates"

    for commit in scenario.commits:
        message = commit.message.strip()
        # once in the "would commit" log, once in the final SHA summary
        assert text.count(message) == 2, f"{message!r} reported {text.count(message)} times"


def test_dry_run_still_prints_the_full_sha_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenteval.scenario import load_scenario
    from agenteval.seeder import verify_deterministic

    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    expected = verify_deterministic(load_scenario(SHIPPED_SCENARIO))

    text = output(runner.invoke(app, ["seed", str(SHIPPED_SCENARIO), "--dry-run"]))

    assert "commits a live run would create:" in text
    for sha in expected:
        assert f"    {sha}  " in text
