"""Thin, deterministic wrapper around PyGithub.

Two properties matter here and are enforced rather than documented:

* **Determinism.** Commits are built through the Git Data API with both the
  author *and* committer date pinned to a caller-supplied timestamp. A commit's
  SHA hashes both dates, so letting the committer date default to "now" would
  make every seeding run produce different SHAs.
* **Safety.** Every repo this client creates is marked with a topic. Deletion
  refuses to touch anything lacking that marker, so a name collision with a
  real repository can never destroy it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from github import (
    Auth,
    Github,
    GithubException,
    InputGitAuthor,
    InputGitTreeElement,
    UnknownObjectException,
)
from github.GitCommit import GitCommit
from github.Repository import Repository

logger = logging.getLogger(__name__)

TOKEN_ENV_VAR = "GITHUB_TOKEN"
MANAGED_TOPIC = "agenteval-managed"
DESCRIPTION_PREFIX = "[agenteval]"
DEFAULT_BRANCH = "main"
DEFAULT_AUTHOR_NAME = "agenteval"
DEFAULT_AUTHOR_EMAIL = "agenteval@users.noreply.github.com"

_BLOB_MODE = "100644"


class GitHubClientError(Exception):
    """Raised for any GitHub interaction that fails or is refused."""


@contextmanager
def _api(description: str) -> Iterator[None]:
    """Translate PyGithub failures into GitHubClientError with context."""
    try:
        yield
    except GithubException as exc:
        raise GitHubClientError(f"GitHub API error while {description}: {exc}") from exc


def _git_timestamp(value: datetime) -> str:
    """Format a datetime the way the Git Data API expects, pinned to UTC."""
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return aware.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _DryRunRepository:
    """Stand-in returned by create_repo when dry_run is on.

    Carries the attributes a caller would read off a real Repository so dry-run
    code paths exercise the same shape as live ones.
    """

    name: str
    full_name: str
    description: str
    html_url: str
    default_branch: str = DEFAULT_BRANCH
    topics: list[str] = field(default_factory=lambda: [MANAGED_TOPIC])

    def get_topics(self) -> list[str]:
        return list(self.topics)


class GitHubClient:
    """Authenticated GitHub client scoped to the operations the seeder needs."""

    def __init__(self, token: str | None = None, dry_run: bool = False) -> None:
        resolved = token if token is not None else os.environ.get(TOKEN_ENV_VAR)
        if not resolved:
            raise GitHubClientError(
                f"No GitHub token available. Pass token=... or set the {TOKEN_ENV_VAR} "
                f"environment variable to a personal access token with 'repo' scope."
            )

        self.dry_run = dry_run
        self._token = resolved
        self._github = Github(auth=Auth.Token(resolved))
        self._login: str | None = None

    # -- identity ----------------------------------------------------------

    @property
    def login(self) -> str:
        """Login of the authenticated user, fetched once and cached."""
        if self._login is None:
            with _api("resolving the authenticated user"):
                self._login = self._github.get_user().login
        return self._login

    # -- reads -------------------------------------------------------------

    def repo_exists(self, name: str) -> bool:
        """Whether the authenticated user already owns a repo called `name`."""
        try:
            self._github.get_user().get_repo(name)
        except UnknownObjectException:
            return False
        except GithubException as exc:
            raise GitHubClientError(
                f"GitHub API error while checking for repo {name!r}: {exc}"
            ) from exc
        return True

    def get_repo(self, name: str) -> Repository:
        """Fetch a repo owned by the authenticated user."""
        try:
            return self._github.get_user().get_repo(name)
        except UnknownObjectException as exc:
            raise GitHubClientError(f"Repo {name!r} not found under {self.login!r}.") from exc
        except GithubException as exc:
            raise GitHubClientError(
                f"GitHub API error while fetching repo {name!r}: {exc}"
            ) from exc

    # -- writes ------------------------------------------------------------

    def create_repo(self, name: str, description: str) -> Repository:
        """Create a marked repo under the authenticated user.

        Raises:
            GitHubClientError: a repo of that name already exists. The caller
                should reset rather than seed on top of unknown content.
        """
        if self.repo_exists(name):
            raise GitHubClientError(
                f"Repo {name!r} already exists under {self.login!r}. Run "
                f"`agenteval reset {name}` to delete it first — refusing to seed "
                f"into an existing repository."
            )

        marked_description = f"{DESCRIPTION_PREFIX} {description}"

        if self.dry_run:
            logger.info("[dry-run] would create repo %r with topic %r", name, MANAGED_TOPIC)
            return cast(
                Repository,
                _DryRunRepository(
                    name=name,
                    full_name=f"{self.login}/{name}",
                    description=marked_description,
                    html_url=f"https://github.com/{self.login}/{name}",
                ),
            )

        with _api(f"creating repo {name!r}"):
            repo = self._github.get_user().create_repo(
                name=name,
                description=marked_description,
                private=True,
                auto_init=False,
            )

        with _api(f"tagging repo {name!r} as managed"):
            repo.replace_topics([MANAGED_TOPIC])

        logger.info("created repo %s", repo.full_name)
        return repo

    def create_commit(
        self,
        repo: Repository,
        message: str,
        files: Mapping[str, str],
        parent_sha: str | None,
        author_date: datetime,
        branch: str = DEFAULT_BRANCH,
        author_name: str = DEFAULT_AUTHOR_NAME,
        author_email: str = DEFAULT_AUTHOR_EMAIL,
    ) -> str:
        """Commit `files` via the Git Data API and move `branch` to it.

        Both the author and committer date are set to `author_date`. Commit SHAs
        hash both, so pinning only the author date would still drift per run.

        Args:
            parent_sha: SHA to build on, or None for an initial commit (the ref
                is created rather than updated).

        Returns:
            The SHA of the new commit.
        """
        if not files:
            raise GitHubClientError(f"Refusing to create empty commit {message!r}: no files given")

        timestamp = _git_timestamp(author_date)
        identity = InputGitAuthor(author_name, author_email, timestamp)

        if self.dry_run:
            sha = self._stub_sha(message, files, parent_sha, timestamp)
            logger.info(
                "[dry-run] would commit %r (%d file(s)) onto %s as %s",
                message,
                len(files),
                parent_sha or "<no parent>",
                sha,
            )
            return sha

        elements: list[InputGitTreeElement] = []
        for path, content in files.items():
            with _api(f"creating blob for {path!r}"):
                blob = repo.create_git_blob(content, "utf-8")
            elements.append(
                InputGitTreeElement(path=path, mode=_BLOB_MODE, type="blob", sha=blob.sha)
            )

        parents: list[GitCommit] = []
        if parent_sha is None:
            with _api("creating tree"):
                tree = repo.create_git_tree(elements)
        else:
            with _api(f"fetching parent commit {parent_sha}"):
                parent = repo.get_git_commit(parent_sha)
            parents = [parent]
            with _api("creating tree"):
                tree = repo.create_git_tree(elements, parent.tree)

        with _api(f"creating commit {message!r}"):
            commit = repo.create_git_commit(
                message=message,
                tree=tree,
                parents=parents,
                author=identity,
                committer=identity,
            )

        ref_name = f"refs/heads/{branch}"
        if parent_sha is None:
            with _api(f"creating ref {ref_name}"):
                repo.create_git_ref(ref_name, commit.sha)
        else:
            with _api(f"updating ref {ref_name}"):
                repo.get_git_ref(f"heads/{branch}").edit(commit.sha)

        logger.info("committed %s (%s)", commit.sha, message)
        return commit.sha

    def delete_repo(self, name: str) -> None:
        """Delete a repo, but only if this tool created it.

        Raises:
            GitHubClientError: the repo is missing, or lacks the managed marker.
                An unmarked repo is assumed to be someone's real work.
        """
        repo = self.get_repo(name)

        with _api(f"reading topics for {name!r}"):
            topics = repo.get_topics()

        if MANAGED_TOPIC not in topics:
            raise GitHubClientError(
                f"Refusing to delete repo {name!r}: it is missing the {MANAGED_TOPIC!r} "
                f"topic, so it was not created by agenteval. Found topics: "
                f"{sorted(topics) or 'none'}. Delete it by hand if that is really what you want."
            )

        if self.dry_run:
            logger.info("[dry-run] would delete repo %r", name)
            return

        with _api(f"deleting repo {name!r}"):
            repo.delete()

        logger.info("deleted repo %s", name)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _stub_sha(
        message: str, files: Mapping[str, str], parent_sha: str | None, timestamp: str
    ) -> str:
        """A stable, realistic-looking SHA so dry runs are reproducible too."""
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(message.encode("utf-8"))
        digest.update((parent_sha or "").encode("utf-8"))
        digest.update(timestamp.encode("utf-8"))
        for path in sorted(files):
            digest.update(path.encode("utf-8"))
            digest.update(files[path].encode("utf-8"))
        return digest.hexdigest()
