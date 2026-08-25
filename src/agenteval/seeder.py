"""Seeding: turn a validated Scenario into a real GitHub repository.

The two invariants worth stating up front:

* **Cumulative trees.** Each commit is built from the repo's *entire* state at
  that point, not just the paths the commit touches. A tree written from only
  the touched paths would delete every untouched file.
* **Reproducible SHAs.** `verify_deterministic` recomputes the expected commit
  SHAs locally using git's own object hashing, so the determinism property can
  be asserted in CI without a token or a network.
* **No auto-init commit.** Repos are created with auto_init=True because the Git
  Data API rejects blobs in an empty repo, but the seeded history must not
  contain that commit. The first scenario commit is a root commit (no parent),
  and the branch is force-moved onto the last scenario commit at the end, which
  orphans the auto-init commit.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from agenteval.github_client import (
    DEFAULT_AUTHOR_EMAIL,
    DEFAULT_AUTHOR_NAME,
    GitHubClient,
    GitHubClientError,
)
from agenteval.scenario import Scenario

logger = logging.getLogger(__name__)

_BLOB_MODE = "100644"
_TREE_MODE = "40000"


@dataclass(frozen=True)
class SeedResult:
    """What a successful seeding run produced."""

    repo_name: str
    repo_url: str
    commit_shas: list[str]

    @property
    def head(self) -> str:
        return self.commit_shas[-1]


# --- message normalization -------------------------------------------------


def normalize_message(message: str) -> str:
    """Give a commit message exactly one trailing newline, as git does.

    Applied identically when sending to the API and when hashing locally, so the
    two agree on the bytes that go into the commit object.
    """
    return message.rstrip("\n") + "\n"


# --- git object hashing (offline) ------------------------------------------


def _hash_object(obj_type: str, body: bytes) -> str:
    header = f"{obj_type} {len(body)}\0".encode()
    return hashlib.sha1(header + body, usedforsecurity=False).hexdigest()


def _blob_sha(content: str) -> str:
    return _hash_object("blob", content.encode("utf-8"))


def _nest(tree: Mapping[str, str]) -> dict[str, object]:
    """Turn flat 'a/b/c.py' -> content into nested dicts, mirroring git trees."""
    root: dict[str, object] = {}
    for path, content in tree.items():
        node = root
        parts = path.split("/")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"path {path!r} conflicts with a file at {part!r}")
            node = child
        node[parts[-1]] = content
    return root


def _write_tree(node: Mapping[str, object]) -> str:
    """Hash one tree level. Git sorts entries by name, directories as 'name/'."""
    entries: list[tuple[str, str, str, bool]] = []
    for name, value in node.items():
        if isinstance(value, dict):
            entries.append((_TREE_MODE, name, _write_tree(value), True))
        else:
            entries.append((_BLOB_MODE, name, _blob_sha(str(value)), False))

    entries.sort(key=lambda entry: entry[1] + "/" if entry[3] else entry[1])

    body = b"".join(
        f"{mode} {name}\0".encode() + bytes.fromhex(sha) for mode, name, sha, _ in entries
    )
    return _hash_object("tree", body)


def tree_sha(tree: Mapping[str, str]) -> str:
    """The git SHA of a tree given a flat path -> content mapping."""
    return _write_tree(_nest(tree))


def _git_time(value: datetime) -> str:
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return f"{int(aware.timestamp())} +0000"


def commit_sha(
    tree: str,
    parents: list[str],
    message: str,
    author_date: datetime,
    author_name: str = DEFAULT_AUTHOR_NAME,
    author_email: str = DEFAULT_AUTHOR_EMAIL,
) -> str:
    """The git SHA of a commit object, computed exactly as git would."""
    identity = f"{author_name} <{author_email}> {_git_time(author_date)}"
    lines = [f"tree {tree}"]
    lines.extend(f"parent {parent}" for parent in parents)
    lines.append(f"author {identity}")
    lines.append(f"committer {identity}")
    lines.append("")
    lines.append(normalize_message(message))
    return _hash_object("commit", "\n".join(lines).encode("utf-8"))


def verify_deterministic(scenario: Scenario) -> list[str]:
    """Expected commit SHAs for `scenario`, computed offline.

    Calls no API and needs no token. Two calls with the same scenario always
    return the same list; that is the determinism guarantee, testable in CI.
    """
    shas: list[str] = []
    parent: str | None = None
    for index, commit in enumerate(scenario.commits):
        sha = commit_sha(
            tree_sha(scenario.content_at(index)),
            [parent] if parent is not None else [],
            commit.message,
            commit.author_date,
        )
        shas.append(sha)
        parent = sha
    return shas


# --- seeding ---------------------------------------------------------------


def seed(scenario: Scenario, client: GitHubClient) -> SeedResult:
    """Create the repo and replay the scenario's commits onto it.

    In dry-run mode nothing is written and the reported SHAs come from
    `verify_deterministic`, so a preview shows exactly what a live seed produces.

    Raises:
        GitHubClientError: repo creation or any commit failed. The message names
            the commit that failed.
    """
    repo = client.create_repo(scenario.name, scenario.description)

    # A dry run must preview the SHAs a live run would produce, not placeholders.
    # These are computed offline from git's own hashing rules.
    predicted: list[str] | None = verify_deterministic(scenario) if client.dry_run else None

    shas: list[str] = []
    parent_sha: str | None = None

    for index, commit in enumerate(scenario.commits):
        # The whole tree as of this commit, not just what the commit touches.
        cumulative = scenario.content_at(index)
        try:
            sha = client.create_commit(
                repo,
                normalize_message(commit.message),
                cumulative,
                parent_sha,
                commit.author_date,
                expected_sha=predicted[index] if predicted else None,
            )
        except GitHubClientError as exc:
            raise GitHubClientError(
                f"Seeding {scenario.name!r} failed at commit {index} of "
                f"{len(scenario.commits)} ({commit.message!r}): {exc}"
            ) from exc

        if not client.dry_run:
            # In dry-run the client already logs a richer line for each commit
            # (parent SHA and file count); a second one is just noise.
            logger.info(
                "commit %d/%d %s %s",
                index + 1,
                len(scenario.commits),
                sha[:7],
                commit.message.strip(),
            )
        shas.append(sha)
        parent_sha = sha

    # The repo was created with auto_init=True so the Git Data API would accept
    # blobs. That auto-init commit is NOT part of the scenario, and the first
    # scenario commit is a root commit rather than its child. Forcing the branch
    # onto the last scenario commit orphans it, leaving exactly this history.
    try:
        client.update_ref(repo, shas[-1])
    except GitHubClientError as exc:
        raise GitHubClientError(
            f"Seeding {scenario.name!r} committed every scenario commit but could not "
            f"move the default branch to {shas[-1]}: {exc}"
        ) from exc

    return SeedResult(
        repo_name=scenario.name,
        repo_url=str(getattr(repo, "html_url", "")),
        commit_shas=shas,
    )


def reset(name: str, client: GitHubClient) -> None:
    """Delete a seeded repo. A missing repo is not an error.

    Raises:
        GitHubClientError: the repo exists but lacks the managed marker, or the
            deletion itself failed.
    """
    if not client.repo_exists(name):
        logger.info("Repo %r does not exist; nothing to reset.", name)
        return

    client.delete_repo(name)
    logger.info("Deleted repo %r.", name)
