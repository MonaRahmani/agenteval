import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agenteval.github_client import (
    DEFAULT_AUTHOR_EMAIL,
    DEFAULT_AUTHOR_NAME,
    GitHubClientError,
)
from agenteval.scenario import Scenario, load_scenario
from agenteval.seeder import (
    SeedResult,
    commit_sha,
    normalize_message,
    reset,
    seed,
    tree_sha,
    verify_deterministic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SCENARIO = REPO_ROOT / "scenarios" / "failing-import.yaml"

HAS_GIT = shutil.which("git") is not None


def build_scenario() -> Scenario:
    """Three commits: two files at commit 0, one rewrite, then one new file."""
    return Scenario.model_validate(
        {
            "name": "sample-scenario",
            "description": "A three-commit project.",
            "seeded_bug": "calc.total averages instead of summing.",
            "rubric": "rubrics/sample.yaml",
            "files": [
                {"path": "pkg/__init__.py", "description": "Public surface."},
                {"path": "pkg/calc.py", "description": "The total() helper."},
                {"path": "README.md", "description": "Docs."},
            ],
            "commits": [
                {
                    "message": "Add package skeleton",
                    "author_date": "2026-01-05T09:00:00Z",
                    "files": [
                        {"path": "pkg/__init__.py", "content": "from pkg.calc import total\n"},
                        {"path": "pkg/calc.py", "content": "def total(v):\n    return sum(v)\n"},
                    ],
                },
                {
                    "message": "Break total()",
                    "author_date": "2026-01-06T09:00:00Z",
                    "files": [
                        {
                            "path": "pkg/calc.py",
                            "content": "def total(v):\n    return sum(v) / len(v)\n",
                        }
                    ],
                },
                {
                    "message": "Document the package",
                    "author_date": "2026-01-07T09:00:00Z",
                    "files": [{"path": "README.md", "content": "# pkg\n"}],
                },
            ],
        }
    )


@pytest.fixture
def client() -> MagicMock:
    """A GitHubClient mock whose create_commit returns predictable SHAs."""
    mock = MagicMock(name="GitHubClient")
    mock.create_repo.return_value = MagicMock(html_url="https://github.com/octocat/sample-scenario")
    counter = iter(f"sha{i}" for i in range(100))
    mock.create_commit.side_effect = lambda *a, **k: next(counter)
    mock.repo_exists.return_value = True
    return mock


def commit_calls(client: MagicMock) -> list[Any]:
    return list(client.create_commit.call_args_list)


# --- seed: ordering --------------------------------------------------------


def test_seed_creates_the_repo_first(client: MagicMock) -> None:
    scenario = build_scenario()
    seed(scenario, client)

    client.create_repo.assert_called_once_with(scenario.name, scenario.description)


def test_seed_walks_commits_in_scenario_order(client: MagicMock) -> None:
    scenario = build_scenario()
    seed(scenario, client)

    messages = [call.args[1] for call in commit_calls(client)]
    assert messages == [normalize_message(c.message) for c in scenario.commits]


def test_seed_passes_each_commits_author_date(client: MagicMock) -> None:
    scenario = build_scenario()
    seed(scenario, client)

    dates = [call.args[4] for call in commit_calls(client)]
    assert dates == [c.author_date for c in scenario.commits]


def test_seed_returns_result_with_url_and_shas(client: MagicMock) -> None:
    result = seed(build_scenario(), client)

    assert isinstance(result, SeedResult)
    assert result.repo_name == "sample-scenario"
    assert result.repo_url == "https://github.com/octocat/sample-scenario"
    assert result.commit_shas == ["sha0", "sha1", "sha2"]
    assert result.head == "sha2"


# --- seed: cumulative trees (the easy thing to get wrong) ------------------


def test_untouched_file_survives_into_later_trees(client: MagicMock) -> None:
    """__init__.py is written at commit 0 and never again; it must persist."""
    seed(build_scenario(), client)

    trees = [call.args[2] for call in commit_calls(client)]

    assert set(trees[0]) == {"pkg/__init__.py", "pkg/calc.py"}
    assert set(trees[1]) == {"pkg/__init__.py", "pkg/calc.py"}
    assert set(trees[2]) == {"pkg/__init__.py", "pkg/calc.py", "README.md"}

    # Untouched at commits 1 and 2, byte-identical throughout.
    assert trees[0]["pkg/__init__.py"] == "from pkg.calc import total\n"
    assert trees[1]["pkg/__init__.py"] == trees[0]["pkg/__init__.py"]
    assert trees[2]["pkg/__init__.py"] == trees[0]["pkg/__init__.py"]


def test_rewritten_file_advances_in_the_tree(client: MagicMock) -> None:
    seed(build_scenario(), client)
    trees = [call.args[2] for call in commit_calls(client)]

    assert trees[0]["pkg/calc.py"] == "def total(v):\n    return sum(v)\n"
    assert trees[1]["pkg/calc.py"] == "def total(v):\n    return sum(v) / len(v)\n"
    assert trees[2]["pkg/calc.py"] == trees[1]["pkg/calc.py"]


def test_tree_never_shrinks(client: MagicMock) -> None:
    seed(build_scenario(), client)
    sizes = [len(call.args[2]) for call in commit_calls(client)]
    assert sizes == sorted(sizes)


def test_seed_does_not_send_only_the_touched_files(client: MagicMock) -> None:
    """The failure mode this guards: commit 1 touches one file, tree has two."""
    scenario = build_scenario()
    seed(scenario, client)

    touched_at_1 = {entry.path for entry in scenario.commits[1].files}
    sent_at_1 = set(commit_calls(client)[1].args[2])

    assert touched_at_1 == {"pkg/calc.py"}
    assert sent_at_1 > touched_at_1


# --- seed: parent chaining -------------------------------------------------


def test_first_commit_has_no_parent(client: MagicMock) -> None:
    seed(build_scenario(), client)
    assert commit_calls(client)[0].args[3] is None


def test_each_commit_parents_the_previous_sha(client: MagicMock) -> None:
    seed(build_scenario(), client)
    parents = [call.args[3] for call in commit_calls(client)]
    assert parents == [None, "sha0", "sha1"]


# --- seed: the shipped scenario -------------------------------------------


def test_seed_shipped_scenario_produces_four_commits(client: MagicMock) -> None:
    scenario = load_scenario(SHIPPED_SCENARIO)
    result = seed(scenario, client)

    assert len(result.commit_shas) == 4
    assert client.create_commit.call_count == 4

    final_tree = commit_calls(client)[-1].args[2]
    assert set(final_tree) == {f.path for f in scenario.files}


def test_shipped_scenario_final_tree_contains_the_bug(client: MagicMock) -> None:
    scenario = load_scenario(SHIPPED_SCENARIO)
    seed(scenario, client)

    final_tree = commit_calls(client)[-1].args[2]
    assert "from inventory.report import" in final_tree["inventory/models.py"]
    assert "from inventory.models import" in final_tree["inventory/report.py"]


# --- seed: failure reporting ----------------------------------------------


def test_failure_midway_names_the_failing_commit(client: MagicMock) -> None:
    calls = {"n": 0}

    def explode(*args: Any, **kwargs: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise GitHubClientError("GitHub API error while creating tree: 500 kaboom")
        return f"sha{calls['n'] - 1}"

    client.create_commit.side_effect = explode

    with pytest.raises(GitHubClientError, match=r"failed at commit 1 of 3"):
        seed(build_scenario(), client)


def test_failure_midway_includes_the_commit_message_and_cause(client: MagicMock) -> None:
    client.create_commit.side_effect = GitHubClientError("boom")

    with pytest.raises(GitHubClientError, match="Add package skeleton") as excinfo:
        seed(build_scenario(), client)

    assert isinstance(excinfo.value.__cause__, GitHubClientError)


def test_repo_creation_failure_propagates(client: MagicMock) -> None:
    client.create_repo.side_effect = GitHubClientError("already exists; run reset")

    with pytest.raises(GitHubClientError, match="already exists"):
        seed(build_scenario(), client)

    client.create_commit.assert_not_called()


# --- verify_deterministic --------------------------------------------------


def test_verify_deterministic_is_stable_across_calls() -> None:
    scenario = load_scenario(SHIPPED_SCENARIO)
    assert verify_deterministic(scenario) == verify_deterministic(scenario)


def test_verify_deterministic_is_stable_across_reloads() -> None:
    first = verify_deterministic(load_scenario(SHIPPED_SCENARIO))
    second = verify_deterministic(load_scenario(SHIPPED_SCENARIO))
    assert first == second


def test_verify_deterministic_returns_one_sha_per_commit() -> None:
    scenario = load_scenario(SHIPPED_SCENARIO)
    shas = verify_deterministic(scenario)

    assert len(shas) == len(scenario.commits)
    assert len(set(shas)) == len(shas)
    assert all(len(sha) == 40 for sha in shas)
    assert all(c in "0123456789abcdef" for sha in shas for c in sha)


def test_verify_deterministic_touches_no_client() -> None:
    """It must work with no token and no network at all."""
    scenario = load_scenario(SHIPPED_SCENARIO)
    assert verify_deterministic(scenario)  # would raise if it tried to authenticate


def test_changing_content_changes_the_sha() -> None:
    base = build_scenario()
    baseline = verify_deterministic(base)

    mutated = base.model_copy(deep=True)
    mutated.commits[0].files[0].content = "# different\n"

    assert verify_deterministic(mutated)[0] != baseline[0]


def test_changing_author_date_changes_the_sha() -> None:
    base = build_scenario()
    baseline = verify_deterministic(base)

    mutated = base.model_copy(deep=True)
    mutated.commits[0].author_date = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)

    assert verify_deterministic(mutated)[0] != baseline[0]


def test_verify_deterministic_chains_parents() -> None:
    """Changing commit 0 must cascade into every later SHA."""
    base = build_scenario()
    baseline = verify_deterministic(base)

    mutated = base.model_copy(deep=True)
    mutated.commits[0].files[0].content = "# different\n"
    changed = verify_deterministic(mutated)

    assert all(a != b for a, b in zip(baseline, changed, strict=True))


# --- git object hashing, checked against real git -------------------------


def test_blob_and_tree_hashing_matches_git_hash_object(tmp_path: Path) -> None:
    if not HAS_GIT:
        pytest.skip("git not available")

    content = "def total(v):\n    return sum(v)\n"
    (tmp_path / "f.py").write_text(content)
    expected = subprocess.run(
        ["git", "hash-object", str(tmp_path / "f.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    from agenteval.seeder import _blob_sha

    assert _blob_sha(content) == expected


def test_verify_deterministic_matches_real_git(tmp_path: Path) -> None:
    """Build the same history with git itself and compare every SHA."""
    if not HAS_GIT:
        pytest.skip("git not available")

    scenario = build_scenario()
    work = tmp_path / "repo"
    work.mkdir()

    def git(*args: str, **env: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=work,
            capture_output=True,
            text=True,
            check=True,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(tmp_path),
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                **env,
            },
        )
        return result.stdout.strip()

    git("init", "-q", "-b", "main")

    expected: list[str] = []
    for index, commit in enumerate(scenario.commits):
        for path, content in scenario.content_at(index).items():
            target = work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        git("add", "-A")
        stamp = commit.author_date.strftime("%Y-%m-%dT%H:%M:%S+0000")
        git(
            "commit",
            "-q",
            "-m",
            commit.message,
            GIT_AUTHOR_NAME=DEFAULT_AUTHOR_NAME,
            GIT_AUTHOR_EMAIL=DEFAULT_AUTHOR_EMAIL,
            GIT_COMMITTER_NAME=DEFAULT_AUTHOR_NAME,
            GIT_COMMITTER_EMAIL=DEFAULT_AUTHOR_EMAIL,
            GIT_AUTHOR_DATE=stamp,
            GIT_COMMITTER_DATE=stamp,
        )
        expected.append(git("rev-parse", "HEAD"))

    assert verify_deterministic(scenario) == expected


def test_tree_sha_matches_git_for_nested_paths(tmp_path: Path) -> None:
    if not HAS_GIT:
        pytest.skip("git not available")

    tree = {
        "a.py": "a\n",
        "pkg/__init__.py": "i\n",
        "pkg/sub/deep.py": "d\n",
        "README.md": "r\n",
    }
    work = tmp_path / "repo"
    work.mkdir()
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    subprocess.run(["git", "init", "-q"], cwd=work, check=True, env=env)
    for path, content in tree.items():
        target = work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True, env=env)
    expected = subprocess.run(
        ["git", "write-tree"], cwd=work, capture_output=True, text=True, check=True, env=env
    ).stdout.strip()

    assert tree_sha(tree) == expected


def test_normalize_message_adds_exactly_one_newline() -> None:
    assert normalize_message("hello") == "hello\n"
    assert normalize_message("hello\n") == "hello\n"
    assert normalize_message("hello\n\n\n") == "hello\n"


def test_commit_sha_is_pure() -> None:
    args: tuple[str, list[str], str, datetime] = (
        "4b825dc642cb6eb9a060e54bf8d69288fbee4904",
        [],
        "m",
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert commit_sha(*args) == commit_sha(*args)


# --- reset -----------------------------------------------------------------


def test_reset_deletes_an_existing_repo(client: MagicMock) -> None:
    client.repo_exists.return_value = True

    reset("failing-import", client)

    client.delete_repo.assert_called_once_with("failing-import")


def test_reset_on_missing_repo_exits_cleanly(client: MagicMock) -> None:
    client.repo_exists.return_value = False

    reset("never-existed", client)  # must not raise

    client.delete_repo.assert_not_called()


def test_reset_on_missing_repo_says_so(client: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    client.repo_exists.return_value = False

    with caplog.at_level("INFO", logger="agenteval.seeder"):
        reset("never-existed", client)

    assert "does not exist" in caplog.text


def test_reset_propagates_a_refused_deletion(client: MagicMock) -> None:
    client.repo_exists.return_value = True
    client.delete_repo.side_effect = GitHubClientError("Refusing to delete repo 'real-work'")

    with pytest.raises(GitHubClientError, match="Refusing to delete"):
        reset("real-work", client)


# --- auto-init commit must not survive into the history --------------------


def test_seed_force_updates_the_ref_to_the_final_commit(client: MagicMock) -> None:
    scenario = build_scenario()

    result = seed(scenario, client)

    client.update_ref.assert_called_once()
    repo_arg, sha_arg = client.update_ref.call_args.args
    assert repo_arg is client.create_repo.return_value
    assert sha_arg == result.commit_shas[-1] == "sha2"


def test_ref_is_updated_after_every_commit(client: MagicMock) -> None:
    """Moving the branch early would leave the later commits unreferenced."""
    order: list[str] = []
    counter = iter(f"sha{i}" for i in range(100))

    def record_commit(*args: Any, **kwargs: Any) -> str:
        order.append("commit")
        return next(counter)

    def record_ref(*args: Any, **kwargs: Any) -> None:
        order.append("ref")

    client.create_commit.side_effect = record_commit
    client.update_ref.side_effect = record_ref

    seed(build_scenario(), client)

    assert order == ["commit", "commit", "commit", "ref"]


def test_seed_never_parents_the_first_commit_on_auto_init(client: MagicMock) -> None:
    """The scenario's first commit is a root commit; auto-init is not its parent."""
    seed(build_scenario(), client)

    assert commit_calls(client)[0].args[3] is None


def test_ref_update_failure_is_reported_clearly(client: MagicMock) -> None:
    client.update_ref.side_effect = GitHubClientError("422 cannot force")

    with pytest.raises(GitHubClientError, match="could not move the default branch"):
        seed(build_scenario(), client)


def test_seeded_history_excludes_the_auto_init_commit(tmp_path: Path) -> None:
    """Replay the real GitHub sequence with git itself and inspect `git log`.

    Mirrors production exactly: an auto-init commit exists first, the scenario's
    commits are written as objects with the first having no parent, then the
    branch is force-moved onto the last one.
    """
    if not HAS_GIT:
        pytest.skip("git not available")

    scenario = build_scenario()
    work = tmp_path / "repo"
    work.mkdir()
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": DEFAULT_AUTHOR_NAME,
        "GIT_AUTHOR_EMAIL": DEFAULT_AUTHOR_EMAIL,
        "GIT_COMMITTER_NAME": DEFAULT_AUTHOR_NAME,
        "GIT_COMMITTER_EMAIL": DEFAULT_AUTHOR_EMAIL,
    }

    def git(*args: str, **extra: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=work,
            capture_output=True,
            text=True,
            check=True,
            env={**env, **extra},
        ).stdout.strip()

    # 1. What auto_init=True produces: a repo with exactly one commit.
    git("init", "-q", "-b", "main")
    (work / "README.md").write_text("# auto-init\n")
    git("add", "README.md")
    git(
        "commit",
        "-q",
        "-m",
        "Initial commit",
        GIT_AUTHOR_DATE="2020-01-01T00:00:00+0000",
        GIT_COMMITTER_DATE="2020-01-01T00:00:00+0000",
    )
    auto_init_sha = git("rev-parse", "HEAD")
    assert git("log", "--format=%H") == auto_init_sha
    (work / "README.md").unlink()  # the scenario does not contain the auto-init file

    # 2. Write the scenario's commits as objects. The first has NO parent.
    parent: str | None = None
    shas: list[str] = []
    for index, commit in enumerate(scenario.commits):
        for path, content in scenario.content_at(index).items():
            target = work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        git("read-tree", "--empty")  # scenario trees, not auto-init's
        git("add", "-A")
        tree = git("write-tree")
        stamp = commit.author_date.strftime("%Y-%m-%dT%H:%M:%S+0000")
        parent_args = ["-p", parent] if parent else []
        sha = git(
            "commit-tree",
            tree,
            *parent_args,
            "-m",
            commit.message,
            GIT_AUTHOR_DATE=stamp,
            GIT_COMMITTER_DATE=stamp,
        )
        shas.append(sha)
        parent = sha

    # 3. Force the branch onto the last scenario commit, orphaning auto-init.
    git("update-ref", "refs/heads/main", shas[-1])

    log = git("log", "--format=%H %s").splitlines()

    assert [line.split(" ", 1)[1] for line in log] == [
        c.message for c in reversed(scenario.commits)
    ]
    assert auto_init_sha not in git("log", "--format=%H")
    assert "Initial commit" not in git("log", "--format=%s")
    assert len(log) == len(scenario.commits)

    # The root commit really is parentless.
    assert git("rev-list", "--max-parents=0", "HEAD") == shas[0]

    # And the resulting SHAs are exactly what verify_deterministic predicted.
    assert shas == verify_deterministic(scenario)
