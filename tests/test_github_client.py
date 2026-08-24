from datetime import UTC, datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock

import pytest
from github import GithubException, UnknownObjectException

from agenteval import github_client as gc
from agenteval.github_client import (
    DESCRIPTION_PREFIX,
    MANAGED_TOPIC,
    TOKEN_ENV_VAR,
    UNAUTHENTICATED_LOGIN,
    GitHubClient,
    GitHubClientError,
)

AUTHOR_DATE = datetime(2026, 1, 5, 9, 14, 0, tzinfo=UTC)
EXPECTED_TIMESTAMP = "2026-01-05T09:14:00Z"


class FakeAuthor:
    """Stand-in for github.InputGitAuthor that records what it was built with."""

    def __init__(self, name: str, email: str, date: str) -> None:
        self.name = name
        self.email = email
        self.date = date


class FakeTreeElement:
    """Stand-in for github.InputGitTreeElement."""

    def __init__(self, path: str, mode: str, type: str, sha: str) -> None:  # noqa: A002
        self.path = path
        self.mode = mode
        self.type = type
        self.sha = sha


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real developer token leak into these tests."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)


@pytest.fixture
def fake_github(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace PyGithub's entry points; no test may touch the network."""
    github_cls = MagicMock(name="Github")
    monkeypatch.setattr(gc, "Github", github_cls)
    monkeypatch.setattr(gc, "InputGitAuthor", FakeAuthor)
    monkeypatch.setattr(gc, "InputGitTreeElement", FakeTreeElement)

    instance = cast(MagicMock, github_cls.return_value)
    instance.get_user.return_value.login = "octocat"
    return instance


def make_client(fake_github: MagicMock, dry_run: bool = False) -> GitHubClient:
    return GitHubClient(token="t0ken", dry_run=dry_run)


def make_repo() -> MagicMock:
    """A mock Repository whose Git Data calls return objects with .sha."""
    repo = MagicMock(name="Repository")
    repo.full_name = "octocat/failing-import"

    blobs = iter(f"blob{i}" for i in range(100))
    repo.create_git_blob.side_effect = lambda content, encoding: MagicMock(sha=next(blobs))
    repo.create_git_tree.return_value = MagicMock(sha="tree1")
    repo.create_git_commit.return_value = MagicMock(sha="commit1")
    repo.get_git_commit.return_value = MagicMock(sha="parent1", tree=MagicMock(sha="parenttree"))
    return repo


def top_level_calls(mock: MagicMock) -> list[str]:
    """Names of methods called directly on `mock`, in call order."""
    return [name.split(".")[0].removesuffix("()") for name, _, _ in mock.mock_calls if name]


# --- construction / auth ---------------------------------------------------


def test_missing_token_raises_naming_the_env_var() -> None:
    with pytest.raises(GitHubClientError, match=TOKEN_ENV_VAR):
        GitHubClient()


def test_missing_token_error_is_actionable() -> None:
    with pytest.raises(GitHubClientError, match="personal access token"):
        GitHubClient()


def test_empty_token_env_var_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "")
    with pytest.raises(GitHubClientError, match=TOKEN_ENV_VAR):
        GitHubClient()


def test_token_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, fake_github: MagicMock
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "env-token")
    client = GitHubClient()
    assert client.login == "octocat"


def test_explicit_token_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch, fake_github: MagicMock
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "env-token")
    assert GitHubClient(token="explicit").login == "octocat"


# --- repo_exists / create_repo ---------------------------------------------


def test_repo_exists_true_when_lookup_succeeds(fake_github: MagicMock) -> None:
    assert make_client(fake_github).repo_exists("thing") is True


def test_repo_exists_false_on_unknown_object(fake_github: MagicMock) -> None:
    fake_github.get_user.return_value.get_repo.side_effect = UnknownObjectException(404, "nope", {})
    assert make_client(fake_github).repo_exists("thing") is False


def test_repo_exists_wraps_other_api_errors(fake_github: MagicMock) -> None:
    fake_github.get_user.return_value.get_repo.side_effect = GithubException(500, "boom", {})
    with pytest.raises(GitHubClientError, match="GitHub API error"):
        make_client(fake_github).repo_exists("thing")


def test_create_repo_on_existing_name_raises_mentioning_reset(fake_github: MagicMock) -> None:
    with pytest.raises(GitHubClientError, match="reset"):
        make_client(fake_github).create_repo("failing-import", "desc")


def test_create_repo_on_existing_name_names_the_repo(fake_github: MagicMock) -> None:
    with pytest.raises(GitHubClientError, match="already exists"):
        make_client(fake_github).create_repo("failing-import", "desc")


def test_create_repo_marks_the_repo(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})
    created = user.create_repo.return_value

    repo = make_client(fake_github).create_repo("failing-import", "A seeded scenario.")

    assert repo is created
    kwargs = user.create_repo.call_args.kwargs
    assert kwargs["name"] == "failing-import"
    assert kwargs["description"] == f"{DESCRIPTION_PREFIX} A seeded scenario."
    cast(MagicMock, created).replace_topics.assert_called_once_with([MANAGED_TOPIC])


def test_create_repo_wraps_api_failure(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})
    user.create_repo.side_effect = GithubException(422, "name taken", {})

    with pytest.raises(GitHubClientError, match="GitHub API error while creating repo"):
        make_client(fake_github).create_repo("failing-import", "desc")


# --- create_commit: call sequence ------------------------------------------


def test_create_commit_uses_git_data_api_in_order(fake_github: MagicMock) -> None:
    repo = make_repo()

    sha = make_client(fake_github).create_commit(
        repo,
        "Add package",
        {"a.py": "x = 1\n", "b.py": "y = 2\n"},
        parent_sha="parent1",
        author_date=AUTHOR_DATE,
    )

    assert sha == "commit1"
    assert top_level_calls(repo) == [
        "create_git_blob",
        "create_git_blob",
        "get_git_commit",
        "create_git_tree",
        "create_git_commit",
    ]


def test_create_commit_never_uses_the_contents_api(fake_github: MagicMock) -> None:
    repo = make_repo()
    make_client(fake_github).create_commit(
        repo, "Add package", {"a.py": "x = 1\n"}, None, AUTHOR_DATE
    )

    for forbidden in ("create_file", "update_file", "delete_file", "get_contents"):
        assert not getattr(repo, forbidden).called, f"{forbidden} is the Contents API"


def test_create_commit_builds_blobs_and_tree_elements(fake_github: MagicMock) -> None:
    repo = make_repo()
    make_client(fake_github).create_commit(
        repo, "Add package", {"a.py": "x = 1\n"}, None, AUTHOR_DATE
    )

    repo.create_git_blob.assert_called_once_with("x = 1\n", "utf-8")
    elements = repo.create_git_tree.call_args.args[0]
    assert len(elements) == 1
    assert elements[0].path == "a.py"
    assert elements[0].mode == "100644"
    assert elements[0].type == "blob"
    assert elements[0].sha == "blob0"


# --- create_commit: determinism -------------------------------------------


def test_create_commit_sets_author_and_committer_to_the_same_date(
    fake_github: MagicMock,
) -> None:
    """The determinism guarantee: a defaulted committer date changes the SHA."""
    repo = make_repo()
    make_client(fake_github).create_commit(
        repo, "Add package", {"a.py": "x = 1\n"}, None, AUTHOR_DATE
    )

    kwargs = repo.create_git_commit.call_args.kwargs
    author, committer = kwargs["author"], kwargs["committer"]

    assert author.date == EXPECTED_TIMESTAMP
    assert committer.date == EXPECTED_TIMESTAMP
    assert author.date == committer.date
    assert committer.date is not None


def test_create_commit_is_byte_identical_across_runs(fake_github: MagicMock) -> None:
    """Same inputs must produce the same outbound API payload every time."""
    payloads = []
    for _ in range(2):
        repo = make_repo()
        make_client(fake_github).create_commit(
            repo, "Add package", {"a.py": "x = 1\n"}, None, AUTHOR_DATE
        )
        kwargs = repo.create_git_commit.call_args.kwargs
        payloads.append(
            (
                kwargs["message"],
                kwargs["author"].name,
                kwargs["author"].email,
                kwargs["author"].date,
                kwargs["committer"].date,
            )
        )
    assert payloads[0] == payloads[1]


def test_create_commit_normalizes_naive_dates_to_utc(fake_github: MagicMock) -> None:
    repo = make_repo()
    make_client(fake_github).create_commit(
        repo, "m", {"a.py": "x\n"}, None, datetime(2026, 1, 5, 9, 14, 0)
    )
    assert repo.create_git_commit.call_args.kwargs["author"].date == EXPECTED_TIMESTAMP


def test_create_commit_converts_offset_dates_to_utc(fake_github: MagicMock) -> None:
    repo = make_repo()
    offset_date = datetime(2026, 1, 5, 4, 14, 0, tzinfo=timezone(timedelta(hours=-5)))
    make_client(fake_github).create_commit(repo, "m", {"a.py": "x\n"}, None, offset_date)
    assert repo.create_git_commit.call_args.kwargs["author"].date == EXPECTED_TIMESTAMP


# --- create_commit: initial vs subsequent ----------------------------------


def test_initial_commit_has_no_parents(fake_github: MagicMock) -> None:
    """The first scenario commit is a root commit, not a child of auto-init."""
    repo = make_repo()

    make_client(fake_github).create_commit(
        repo, "Initial", {"a.py": "x = 1\n"}, parent_sha=None, author_date=AUTHOR_DATE
    )

    assert repo.create_git_commit.call_args.kwargs["parents"] == []
    repo.get_git_commit.assert_not_called()


def test_create_commit_touches_no_refs(fake_github: MagicMock) -> None:
    """Refs move once, at the end, via update_ref — never per commit."""
    repo = make_repo()

    make_client(fake_github).create_commit(
        repo, "Initial", {"a.py": "x = 1\n"}, parent_sha=None, author_date=AUTHOR_DATE
    )

    repo.create_git_ref.assert_not_called()
    repo.get_git_ref.assert_not_called()


def test_initial_commit_tree_has_no_base_tree(fake_github: MagicMock) -> None:
    repo = make_repo()
    make_client(fake_github).create_commit(repo, "Initial", {"a.py": "x\n"}, None, AUTHOR_DATE)
    assert len(repo.create_git_tree.call_args.args) == 1


def test_subsequent_commit_uses_the_parent(fake_github: MagicMock) -> None:
    repo = make_repo()

    make_client(fake_github).create_commit(
        repo, "Second", {"a.py": "x = 2\n"}, parent_sha="parent1", author_date=AUTHOR_DATE
    )

    repo.get_git_commit.assert_called_once_with("parent1")
    parent = repo.get_git_commit.return_value
    assert repo.create_git_commit.call_args.kwargs["parents"] == [parent]
    assert repo.create_git_tree.call_args.args[1] is parent.tree
    repo.get_git_ref.assert_not_called()
    repo.create_git_ref.assert_not_called()


def test_update_ref_targets_a_custom_branch(fake_github: MagicMock) -> None:
    repo = make_repo()
    make_client(fake_github).update_ref(repo, "commit1", branch="trunk")
    repo.get_git_ref.assert_called_once_with("heads/trunk")


def test_empty_file_map_is_refused(fake_github: MagicMock) -> None:
    repo = make_repo()
    with pytest.raises(GitHubClientError, match="empty commit"):
        make_client(fake_github).create_commit(repo, "Nothing", {}, None, AUTHOR_DATE)
    assert top_level_calls(repo) == []


# --- error propagation -----------------------------------------------------


@pytest.mark.parametrize(
    ("method", "phrase"),
    [
        ("create_git_blob", "creating blob"),
        ("create_git_tree", "creating tree"),
        ("create_git_commit", "creating commit"),
    ],
)
def test_api_failure_mid_sequence_surfaces_as_client_error(
    fake_github: MagicMock, method: str, phrase: str
) -> None:
    repo = make_repo()
    getattr(repo, method).side_effect = GithubException(500, "kaboom", {})

    with pytest.raises(GitHubClientError, match=phrase) as excinfo:
        make_client(fake_github).create_commit(
            repo, "Add package", {"a.py": "x = 1\n"}, None, AUTHOR_DATE
        )

    assert isinstance(excinfo.value.__cause__, GithubException)


def test_raw_github_exception_never_escapes(fake_github: MagicMock) -> None:
    repo = make_repo()
    repo.create_git_tree.side_effect = GithubException(422, "bad tree", {})

    with pytest.raises(GitHubClientError):
        make_client(fake_github).create_commit(repo, "m", {"a.py": "x\n"}, None, AUTHOR_DATE)


# --- delete_repo safety guard ----------------------------------------------


def test_delete_repo_refuses_without_the_marker_topic(fake_github: MagicMock) -> None:
    repo = fake_github.get_user.return_value.get_repo.return_value
    repo.get_topics.return_value = ["python", "production"]

    with pytest.raises(GitHubClientError, match="Refusing to delete"):
        make_client(fake_github).delete_repo("important-work")

    repo.delete.assert_not_called()


def test_delete_refusal_names_the_required_topic(fake_github: MagicMock) -> None:
    repo = fake_github.get_user.return_value.get_repo.return_value
    repo.get_topics.return_value = []

    with pytest.raises(GitHubClientError, match=MANAGED_TOPIC):
        make_client(fake_github).delete_repo("important-work")

    repo.delete.assert_not_called()


def test_delete_repo_proceeds_when_the_marker_is_present(fake_github: MagicMock) -> None:
    repo = fake_github.get_user.return_value.get_repo.return_value
    repo.get_topics.return_value = [MANAGED_TOPIC, "python"]

    make_client(fake_github).delete_repo("failing-import")

    repo.delete.assert_called_once_with()


def test_delete_repo_on_missing_repo_raises(fake_github: MagicMock) -> None:
    fake_github.get_user.return_value.get_repo.side_effect = UnknownObjectException(404, "x", {})

    with pytest.raises(GitHubClientError, match="not found"):
        make_client(fake_github).delete_repo("gone")


# --- dry run ---------------------------------------------------------------


def test_dry_run_create_repo_performs_no_writes(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})

    repo = make_client(fake_github, dry_run=True).create_repo("failing-import", "A scenario.")

    user.create_repo.assert_not_called()
    assert repo.name == "failing-import"
    assert repo.full_name == "octocat/failing-import"
    assert repo.description == f"{DESCRIPTION_PREFIX} A scenario."
    assert MANAGED_TOPIC in repo.get_topics()


def test_dry_run_create_commit_performs_no_writes(fake_github: MagicMock) -> None:
    repo = make_repo()

    sha = make_client(fake_github, dry_run=True).create_commit(
        repo, "Add package", {"a.py": "x = 1\n"}, None, AUTHOR_DATE
    )

    assert top_level_calls(repo) == []
    for method in (
        "create_git_blob",
        "create_git_tree",
        "create_git_commit",
        "create_git_ref",
        "get_git_ref",
    ):
        getattr(repo, method).assert_not_called()

    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_dry_run_commit_sha_is_stable_and_input_sensitive(fake_github: MagicMock) -> None:
    client = make_client(fake_github, dry_run=True)
    repo = make_repo()

    first = client.create_commit(repo, "m", {"a.py": "x\n"}, None, AUTHOR_DATE)
    again = client.create_commit(repo, "m", {"a.py": "x\n"}, None, AUTHOR_DATE)
    different = client.create_commit(repo, "m", {"a.py": "y\n"}, None, AUTHOR_DATE)

    assert first == again
    assert first != different


def test_dry_run_delete_still_enforces_the_marker(fake_github: MagicMock) -> None:
    repo = fake_github.get_user.return_value.get_repo.return_value
    repo.get_topics.return_value = []

    with pytest.raises(GitHubClientError, match="Refusing to delete"):
        make_client(fake_github, dry_run=True).delete_repo("important-work")

    repo.delete.assert_not_called()


def test_dry_run_delete_performs_no_write(fake_github: MagicMock) -> None:
    repo = fake_github.get_user.return_value.get_repo.return_value
    repo.get_topics.return_value = [MANAGED_TOPIC]

    make_client(fake_github, dry_run=True).delete_repo("failing-import")

    repo.delete.assert_not_called()


def test_no_test_here_touches_the_network(fake_github: MagicMock) -> None:
    """The Github class itself is a mock, so no transport is ever constructed."""
    make_client(fake_github)
    assert isinstance(getattr(gc, "Github"), MagicMock)  # noqa: B009


# --- unauthenticated dry run -----------------------------------------------


def test_dry_run_client_constructs_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Previewing a seed must not require credentials."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    client = GitHubClient(dry_run=True)

    assert client.dry_run is True
    assert client.authenticated is False


def test_dry_run_without_token_never_constructs_a_github(
    monkeypatch: pytest.MonkeyPatch, fake_github: MagicMock
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    GitHubClient(dry_run=True)

    cast(MagicMock, getattr(gc, "Github")).assert_not_called()  # noqa: B009


def test_dry_run_without_token_synthesizes_a_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    assert GitHubClient(dry_run=True).login == UNAUTHENTICATED_LOGIN


def test_dry_run_without_token_assumes_repo_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    assert GitHubClient(dry_run=True).repo_exists("anything") is False


def test_dry_run_without_token_can_create_a_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    repo = GitHubClient(dry_run=True).create_repo("failing-import", "A scenario.")

    assert repo.name == "failing-import"
    assert repo.full_name == f"{UNAUTHENTICATED_LOGIN}/failing-import"
    assert repo.description == f"{DESCRIPTION_PREFIX} A scenario."


def test_dry_run_without_token_can_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    client = GitHubClient(dry_run=True)
    repo = client.create_repo("failing-import", "A scenario.")

    sha = client.create_commit(repo, "Add files", {"a.py": "x\n"}, None, AUTHOR_DATE)

    assert len(sha) == 40


def test_dry_run_without_token_can_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    GitHubClient(dry_run=True).delete_repo("failing-import")  # must not raise


def test_dry_run_without_token_warns_it_cannot_verify_the_marker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    with caplog.at_level("WARNING", logger="agenteval.github_client"):
        GitHubClient(dry_run=True).delete_repo("failing-import")

    assert "could not verify" in caplog.text
    assert MANAGED_TOPIC in caplog.text


def test_non_dry_run_still_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unauthenticated path must not leak into real runs."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    with pytest.raises(GitHubClientError, match=TOKEN_ENV_VAR):
        GitHubClient(dry_run=False)

    with pytest.raises(GitHubClientError, match=TOKEN_ENV_VAR):
        GitHubClient()


def test_authenticated_dry_run_still_checks_the_real_marker(fake_github: MagicMock) -> None:
    """With a token, dry-run reads are real — the guard is not weakened."""
    repo = fake_github.get_user.return_value.get_repo.return_value
    repo.get_topics.return_value = ["production"]

    client = GitHubClient(token="t0ken", dry_run=True)
    assert client.authenticated is True

    with pytest.raises(GitHubClientError, match="Refusing to delete"):
        client.delete_repo("real-work")


def test_authenticated_client_reports_authenticated(fake_github: MagicMock) -> None:
    assert GitHubClient(token="t0ken").authenticated is True


# --- empty-repo 409 and the auto_init fix ----------------------------------


class EmptyRepoError(GithubException):
    """The real 409 GitHub returns for Git Data writes into a commitless repo."""

    def __init__(self) -> None:
        super().__init__(409, {"message": "Git Repository is empty."}, None)


def stateful_fake_repo(has_commits: bool) -> MagicMock:
    """A repo that enforces GitHub's real rule: no blobs before a first commit."""
    repo = MagicMock(name="StatefulRepo")
    repo.has_commits = has_commits
    repo.default_branch = "main"
    blobs = iter(f"blob{i}" for i in range(100))

    def blob(content: str, encoding: str) -> MagicMock:
        if not repo.has_commits:
            raise EmptyRepoError()
        return MagicMock(sha=next(blobs))

    repo.create_git_blob.side_effect = blob
    repo.create_git_tree.return_value = MagicMock(sha="tree1")
    repo.create_git_commit.return_value = MagicMock(sha="commit1")
    return repo


def wire_repo_creation(fake_github: MagicMock) -> MagicMock:
    """Make create_repo honour auto_init the way GitHub does."""
    user = fake_github.get_user.return_value
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})

    def create_repo(**kwargs: object) -> MagicMock:
        return stateful_fake_repo(has_commits=bool(kwargs.get("auto_init")))

    user.create_repo.side_effect = create_repo
    return cast(MagicMock, user)


def test_create_repo_passes_auto_init_true(fake_github: MagicMock) -> None:
    """Without auto_init the Git Data API rejects the very first blob."""
    user = wire_repo_creation(fake_github)

    make_client(fake_github).create_repo("failing-import", "desc")

    assert user.create_repo.call_args.kwargs["auto_init"] is True


def test_the_fake_reproduces_the_409_when_auto_init_is_false(fake_github: MagicMock) -> None:
    """Guard on the guard: the fixture really does model the reported bug."""
    repo = stateful_fake_repo(has_commits=False)

    with pytest.raises(GitHubClientError, match="Git Repository is empty"):
        make_client(fake_github).create_commit(repo, "m", {"a.py": "x\n"}, None, AUTHOR_DATE)


def test_happy_path_no_longer_hits_the_empty_repo_error(fake_github: MagicMock) -> None:
    """End to end through create_repo: the 409 can no longer occur."""
    wire_repo_creation(fake_github)
    client = make_client(fake_github)

    repo = client.create_repo("failing-import", "desc")
    sha = client.create_commit(repo, "Initial", {"a.py": "x\n"}, None, AUTHOR_DATE)

    assert sha == "commit1"
    assert cast(MagicMock, repo).has_commits is True


# --- update_ref ------------------------------------------------------------


def test_update_ref_forces_by_default(fake_github: MagicMock) -> None:
    repo = make_repo()
    repo.default_branch = "main"

    make_client(fake_github).update_ref(repo, "finalsha")

    repo.get_git_ref.assert_called_once_with("heads/main")
    repo.get_git_ref.return_value.edit.assert_called_once_with("finalsha", force=True)


def test_update_ref_uses_the_repos_default_branch(fake_github: MagicMock) -> None:
    repo = make_repo()
    repo.default_branch = "master"

    make_client(fake_github).update_ref(repo, "finalsha")

    repo.get_git_ref.assert_called_once_with("heads/master")


def test_update_ref_wraps_api_errors(fake_github: MagicMock) -> None:
    repo = make_repo()
    repo.default_branch = "main"
    repo.get_git_ref.side_effect = GithubException(422, "cannot force", {})

    with pytest.raises(GitHubClientError, match="force-updating ref"):
        make_client(fake_github).update_ref(repo, "finalsha")


def test_dry_run_update_ref_performs_no_write(fake_github: MagicMock) -> None:
    repo = make_repo()
    repo.default_branch = "main"

    make_client(fake_github, dry_run=True).update_ref(repo, "finalsha")

    repo.get_git_ref.assert_not_called()


# --- 403 fine-grained token hint -------------------------------------------


def test_403_on_repo_creation_explains_fine_grained_tokens(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})
    user.create_repo.side_effect = GithubException(
        403, {"message": "Resource not accessible by personal access token"}, None
    )

    with pytest.raises(GitHubClientError) as excinfo:
        make_client(fake_github).create_repo("failing-import", "desc")

    message = str(excinfo.value)
    assert "Fine-grained" in message
    assert "classic" in message
    assert "delete_repo" in message
    assert TOKEN_ENV_VAR in message


def test_non_403_repo_creation_error_keeps_the_generic_message(fake_github: MagicMock) -> None:
    user = fake_github.get_user.return_value
    user.get_repo.side_effect = UnknownObjectException(404, "nope", {})
    user.create_repo.side_effect = GithubException(500, "server exploded", {})

    with pytest.raises(GitHubClientError, match="GitHub API error while creating repo") as excinfo:
        make_client(fake_github).create_repo("failing-import", "desc")

    assert "Fine-grained" not in str(excinfo.value)
