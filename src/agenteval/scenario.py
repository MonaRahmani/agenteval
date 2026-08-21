"""Scenario schema — the config contract the seeder consumes.

A scenario describes a synthetic repository as a *progressive* history: the
top-level `files` list is a manifest of every path the repo may ever contain,
and each commit carries the full content of the paths it touches as of that
commit. A file may be rewritten by later commits — that is how a scenario
introduces a bug into previously working code.

Validation here is deliberately strict: a scenario that parses must be seedable
and must seed identically every time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_REPO_NAME_CHARS = "letters, digits, hyphens, and underscores"


class ScenarioError(Exception):
    """Raised when a scenario file cannot be read, parsed, or validated."""


def _check_relative_path(value: str, label: str) -> str:
    """Reject anything that is not a plain relative path inside the repo root."""
    if not value.strip():
        raise ValueError(f"{label} must not be empty")
    if "\\" in value:
        raise ValueError(f"{label} must use '/' separators, got {value!r}")
    if value.startswith("/") or PurePosixPath(value).is_absolute():
        raise ValueError(f"{label} must be relative, not absolute: {value!r}")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"{label} must not contain '..': {value!r}")
    return value


def _as_utc(value: datetime) -> datetime:
    """Normalize to an aware datetime so naive and aware dates stay comparable."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


class ScenarioFile(BaseModel):
    """A manifest entry: a path the repo may contain, and what it is for.

    Carries no content — content lives on the commits that write the path.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    description: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _check_relative_path(value, "file path")

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("file description must not be empty")
        return value


class ScenarioCommitFile(BaseModel):
    """The full content of one path as of one commit."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _check_relative_path(value, "commit file path")


class ScenarioCommit(BaseModel):
    """One commit in the synthetic history."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    files: list[ScenarioCommitFile] = Field(min_length=1)
    author_date: datetime

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("commit message must not be empty")
        return value

    @property
    def paths(self) -> list[str]:
        """Paths this commit touches, in declaration order."""
        return [entry.path for entry in self.files]


class Scenario(BaseModel):
    """A complete, seedable scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    files: list[ScenarioFile] = Field(min_length=1)
    commits: list[ScenarioCommit] = Field(min_length=1)
    seeded_bug: str
    rubric: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("invalid repo name: must not be empty")
        if not all(c.isascii() and (c.isalnum() or c in "-_") for c in value):
            raise ValueError(f"invalid repo name {value!r}: must contain only {_REPO_NAME_CHARS}")
        return value

    @field_validator("rubric")
    @classmethod
    def _validate_rubric(cls, value: str) -> str:
        return _check_relative_path(value, "rubric path")

    @model_validator(mode="after")
    def _validate_history(self) -> Self:
        declared = self._declared_paths()
        self._check_commits_against_manifest(declared)
        self._check_every_declared_path_is_touched()
        self._check_no_op_rewrites()
        self._check_ascending_dates()
        return self

    def _declared_paths(self) -> set[str]:
        declared: set[str] = set()
        for scenario_file in self.files:
            if scenario_file.path in declared:
                raise ValueError(f"duplicate file path {scenario_file.path!r} in 'files'")
            declared.add(scenario_file.path)
        return declared

    def _check_commits_against_manifest(self, declared: set[str]) -> None:
        for index, commit in enumerate(self.commits):
            seen: set[str] = set()
            for entry in commit.files:
                if entry.path not in declared:
                    raise ValueError(
                        f"commit {index} ({commit.message!r}) references undeclared file "
                        f"{entry.path!r}; declare it in 'files' or remove it from the commit"
                    )
                if entry.path in seen:
                    raise ValueError(
                        f"commit {index} ({commit.message!r}) writes file {entry.path!r} "
                        f"more than once; a commit may write each path only once"
                    )
                seen.add(entry.path)

    def _check_every_declared_path_is_touched(self) -> None:
        touched = {entry.path for commit in self.commits for entry in commit.files}
        for scenario_file in self.files:
            if scenario_file.path not in touched:
                raise ValueError(
                    f"file {scenario_file.path!r} is never touched by any commit, so it "
                    f"would never reach the seeded repo"
                )

    def _check_no_op_rewrites(self) -> None:
        """A rewrite must change the file, or the commit produces an empty diff."""
        latest: dict[str, tuple[int, str]] = {}
        for index, commit in enumerate(self.commits):
            for entry in commit.files:
                previous = latest.get(entry.path)
                if previous is not None and previous[1] == entry.content:
                    prior_index = previous[0]
                    raise ValueError(
                        f"commit {index} ({commit.message!r}) writes file {entry.path!r} with "
                        f"content identical to commit {prior_index} "
                        f"({self.commits[prior_index].message!r}); that is a no-op commit"
                    )
                latest[entry.path] = (index, entry.content)

    def _check_ascending_dates(self) -> None:
        for index in range(1, len(self.commits)):
            previous, current = self.commits[index - 1], self.commits[index]
            if _as_utc(current.author_date) <= _as_utc(previous.author_date):
                raise ValueError(
                    f"commit author_dates must be strictly ascending: commit {index} "
                    f"({current.message!r}) has author_date {current.author_date.isoformat()}, "
                    f"which is not after commit {index - 1} "
                    f"({previous.author_date.isoformat()})"
                )

    def content_at(self, commit_index: int) -> dict[str, str]:
        """The full tree state — path to content — after `commit_index` is applied."""
        if not -len(self.commits) <= commit_index < len(self.commits):
            raise IndexError(f"commit index {commit_index} out of range")
        if commit_index < 0:
            commit_index += len(self.commits)
        tree: dict[str, str] = {}
        for commit in self.commits[: commit_index + 1]:
            for entry in commit.files:
                tree[entry.path] = entry.content
        return tree

    def versions_of(self, path: str) -> list[tuple[int, str]]:
        """Every version of `path`, as (commit index, content), in commit order."""
        return [
            (index, entry.content)
            for index, commit in enumerate(self.commits)
            for entry in commit.files
            if entry.path == path
        ]


def _format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        message = error["msg"]
        for prefix in ("Value error, ", "Assertion failed, "):
            if message.startswith(prefix):
                message = message[len(prefix) :]
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  - {location}: {message}" if location else f"  - {message}")
    return "\n".join(lines)


def load_scenario(path: Path) -> Scenario:
    """Load and validate a scenario YAML file.

    Raises:
        ScenarioError: the file is missing, unreadable, not valid YAML, or does
            not satisfy the scenario schema. Pydantic errors never escape.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScenarioError(f"Scenario file not found: {path}") from exc
    except OSError as exc:
        raise ScenarioError(f"Could not read scenario file {path}: {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"Malformed YAML in scenario file {path}: {exc}") from exc

    if data is None:
        raise ScenarioError(f"Malformed YAML in scenario file {path}: file is empty")
    if not isinstance(data, dict):
        raise ScenarioError(
            f"Malformed YAML in scenario file {path}: expected a mapping at the top "
            f"level, got {type(data).__name__}"
        )

    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(
            f"Invalid scenario file {path}:\n{_format_validation_error(exc)}"
        ) from exc
