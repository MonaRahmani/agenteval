import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from agenteval.scenario import Scenario, ScenarioError, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SCENARIO = REPO_ROOT / "scenarios" / "failing-import.yaml"

INIT_V1 = "from pkg.calc import total\n"
CALC_V1 = "def total(values):\n    return sum(values)\n"
CALC_V2 = "def total(values):\n    return sum(values) / len(values)\n"


def valid_scenario() -> dict[str, Any]:
    """A minimal scenario that satisfies every rule; tests mutate copies of it."""
    return {
        "name": "sample-scenario",
        "description": "A two-file project with a two-commit history.",
        "seeded_bug": "total() averages instead of summing.",
        "rubric": "rubrics/sample.yaml",
        "files": [
            {"path": "pkg/__init__.py", "description": "Public surface."},
            {"path": "pkg/calc.py", "description": "The total() helper. Broken by commit 1."},
        ],
        "commits": [
            {
                "message": "Add package skeleton",
                "author_date": "2026-01-05T09:00:00Z",
                "files": [
                    {"path": "pkg/__init__.py", "content": INIT_V1},
                    {"path": "pkg/calc.py", "content": CALC_V1},
                ],
            },
            {
                "message": "Break total()",
                "author_date": "2026-01-06T09:00:00Z",
                "files": [{"path": "pkg/calc.py", "content": CALC_V2}],
            },
        ],
    }


def write_scenario(tmp_path: Path, data: Any, name: str = "scenario.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def load_mutated(tmp_path: Path, mutate: Any) -> Scenario:
    data = valid_scenario()
    mutate(data)
    return load_scenario(write_scenario(tmp_path, data))


# --- happy path -----------------------------------------------------------


def test_valid_scenario_loads_with_correct_fields(tmp_path: Path) -> None:
    scenario = load_scenario(write_scenario(tmp_path, valid_scenario()))

    assert scenario.name == "sample-scenario"
    assert scenario.description == "A two-file project with a two-commit history."
    assert scenario.seeded_bug == "total() averages instead of summing."
    assert scenario.rubric == "rubrics/sample.yaml"

    assert [f.path for f in scenario.files] == ["pkg/__init__.py", "pkg/calc.py"]
    assert scenario.files[0].description == "Public surface."

    assert [c.message for c in scenario.commits] == ["Add package skeleton", "Break total()"]
    assert scenario.commits[0].paths == ["pkg/__init__.py", "pkg/calc.py"]
    assert scenario.commits[0].files[1].content == CALC_V1
    assert scenario.commits[0].author_date.year == 2026
    assert scenario.commits[0].author_date < scenario.commits[1].author_date


def test_manifest_entries_carry_no_content(tmp_path: Path) -> None:
    """Content belongs on commits; a manifest entry with content is a schema error."""

    def mutate(data: dict[str, Any]) -> None:
        data["files"][0]["content"] = "oops\n"

    with pytest.raises(ScenarioError, match=r"[Ee]xtra inputs are not permitted"):
        load_mutated(tmp_path, mutate)


# --- per-commit content ---------------------------------------------------


def test_same_file_across_commits_with_differing_content_is_valid(tmp_path: Path) -> None:
    scenario = load_scenario(write_scenario(tmp_path, valid_scenario()))

    versions = scenario.versions_of("pkg/calc.py")
    assert [index for index, _ in versions] == [0, 1]
    assert [content for _, content in versions] == [CALC_V1, CALC_V2]


def test_same_file_across_commits_with_identical_content_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["files"][0]["content"] = CALC_V1

    expected = re.escape("writes file 'pkg/calc.py' with content identical to commit 0")
    with pytest.raises(ScenarioError, match=expected):
        load_mutated(tmp_path, mutate)


def test_no_op_rewrite_error_says_it_is_a_no_op(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["files"][0]["content"] = CALC_V1

    with pytest.raises(ScenarioError, match="no-op commit"):
        load_mutated(tmp_path, mutate)


def test_reverting_to_an_older_content_is_valid(tmp_path: Path) -> None:
    """A -> B -> A is a real diff at each step, unlike A -> A."""

    def mutate(data: dict[str, Any]) -> None:
        data["commits"].append(
            {
                "message": "Revert the break",
                "author_date": "2026-01-07T09:00:00Z",
                "files": [{"path": "pkg/calc.py", "content": CALC_V1}],
            }
        )

    scenario = load_mutated(tmp_path, mutate)
    assert [content for _, content in scenario.versions_of("pkg/calc.py")] == [
        CALC_V1,
        CALC_V2,
        CALC_V1,
    ]


def test_content_at_each_commit_is_retrievable_in_order(tmp_path: Path) -> None:
    scenario = load_scenario(write_scenario(tmp_path, valid_scenario()))

    at_zero = scenario.content_at(0)
    assert at_zero == {"pkg/__init__.py": INIT_V1, "pkg/calc.py": CALC_V1}

    at_one = scenario.content_at(1)
    assert at_one == {"pkg/__init__.py": INIT_V1, "pkg/calc.py": CALC_V2}

    # Untouched files carry forward; touched files advance.
    assert at_one["pkg/__init__.py"] == at_zero["pkg/__init__.py"]
    assert at_one["pkg/calc.py"] != at_zero["pkg/calc.py"]

    assert scenario.content_at(-1) == at_one


def test_content_at_rejects_out_of_range_index(tmp_path: Path) -> None:
    scenario = load_scenario(write_scenario(tmp_path, valid_scenario()))

    with pytest.raises(IndexError, match="out of range"):
        scenario.content_at(2)


def test_versions_of_unknown_path_is_empty(tmp_path: Path) -> None:
    scenario = load_scenario(write_scenario(tmp_path, valid_scenario()))
    assert scenario.versions_of("pkg/nope.py") == []


def test_writing_the_same_path_twice_in_one_commit_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["files"].append({"path": "pkg/calc.py", "content": "# again\n"})

    with pytest.raises(ScenarioError, match="more than once"):
        load_mutated(tmp_path, mutate)


# --- manifest / commit agreement ------------------------------------------


def test_commit_referencing_undeclared_file_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["files"].append({"path": "pkg/missing.py", "content": "x = 1\n"})

    expected = re.escape("references undeclared file 'pkg/missing.py'")
    with pytest.raises(ScenarioError, match=expected):
        load_mutated(tmp_path, mutate)


def test_undeclared_file_error_names_the_commit(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["files"].append({"path": "pkg/missing.py", "content": "x = 1\n"})

    with pytest.raises(ScenarioError, match=re.escape("commit 1 ('Break total()')")):
        load_mutated(tmp_path, mutate)


def test_file_never_touched_by_a_commit_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"].append({"path": "pkg/orphan.py", "description": "Never written."})

    expected = re.escape("file 'pkg/orphan.py' is never touched by any commit")
    with pytest.raises(ScenarioError, match=expected):
        load_mutated(tmp_path, mutate)


def test_duplicate_file_paths_in_manifest_are_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"].append({"path": "pkg/calc.py", "description": "Same path again."})

    with pytest.raises(ScenarioError, match=re.escape("duplicate file path 'pkg/calc.py'")):
        load_mutated(tmp_path, mutate)


# --- ordering --------------------------------------------------------------


def test_out_of_order_commit_dates_are_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["author_date"] = "2026-01-04T09:00:00Z"

    with pytest.raises(ScenarioError, match="author_dates must be strictly ascending"):
        load_mutated(tmp_path, mutate)


def test_equal_commit_dates_are_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["author_date"] = data["commits"][0]["author_date"]

    with pytest.raises(ScenarioError, match="author_dates must be strictly ascending"):
        load_mutated(tmp_path, mutate)


def test_naive_and_aware_dates_are_compared_not_crashed(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][0]["author_date"] = "2026-01-05T09:00:00Z"
        data["commits"][1]["author_date"] = "2026-01-04 09:00:00"

    with pytest.raises(ScenarioError, match="author_dates must be strictly ascending"):
        load_mutated(tmp_path, mutate)


# --- path safety -----------------------------------------------------------


def test_path_traversal_in_manifest_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"][0]["path"] = "../etc/passwd"

    with pytest.raises(ScenarioError, match=re.escape("must not contain '..'")):
        load_mutated(tmp_path, mutate)


def test_nested_path_traversal_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"][0]["path"] = "pkg/../../etc/passwd"

    with pytest.raises(ScenarioError, match=re.escape("must not contain '..'")):
        load_mutated(tmp_path, mutate)


def test_absolute_path_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"][0]["path"] = "/etc/passwd"

    with pytest.raises(ScenarioError, match="must be relative, not absolute"):
        load_mutated(tmp_path, mutate)


def test_traversal_in_commit_file_path_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][0]["files"][0]["path"] = "../outside.py"

    with pytest.raises(ScenarioError, match=re.escape("must not contain '..'")):
        load_mutated(tmp_path, mutate)


# --- field-level rules -----------------------------------------------------


@pytest.mark.parametrize("name", ["my repo", "my/repo", "owner/name", "repo!", "café"])
def test_invalid_repo_names_are_rejected(tmp_path: Path, name: str) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["name"] = name

    with pytest.raises(ScenarioError, match="invalid repo name"):
        load_mutated(tmp_path, mutate)


@pytest.mark.parametrize("name", ["repo", "my-repo", "my_repo", "Repo123", "a"])
def test_valid_repo_names_are_accepted(tmp_path: Path, name: str) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["name"] = name

    assert load_mutated(tmp_path, mutate).name == name


def test_empty_files_list_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"] = []

    with pytest.raises(ScenarioError, match="at least 1 item"):
        load_mutated(tmp_path, mutate)


def test_empty_commits_list_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"] = []

    with pytest.raises(ScenarioError, match="at least 1 item"):
        load_mutated(tmp_path, mutate)


def test_commit_with_no_files_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][1]["files"] = []

    with pytest.raises(ScenarioError, match="at least 1 item"):
        load_mutated(tmp_path, mutate)


def test_empty_commit_message_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["commits"][0]["message"] = "   "

    with pytest.raises(ScenarioError, match="commit message must not be empty"):
        load_mutated(tmp_path, mutate)


def test_empty_manifest_description_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["files"][0]["description"] = "  "

    with pytest.raises(ScenarioError, match="file description must not be empty"):
        load_mutated(tmp_path, mutate)


def test_empty_file_content_is_allowed(tmp_path: Path) -> None:
    """An empty file is legitimate (e.g. __init__.py); empty content is not an error."""

    def mutate(data: dict[str, Any]) -> None:
        data["commits"][0]["files"][0]["content"] = ""

    assert load_mutated(tmp_path, mutate).content_at(0)["pkg/__init__.py"] == ""


def test_missing_author_date_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        del data["commits"][0]["author_date"]

    with pytest.raises(ScenarioError, match="author_date"):
        load_mutated(tmp_path, mutate)


# --- loader error handling -------------------------------------------------


def test_malformed_yaml_raises_scenario_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: failing-import\nfiles: [unclosed\n", encoding="utf-8")

    with pytest.raises(ScenarioError, match="Malformed YAML"):
        load_scenario(path)


def test_non_mapping_yaml_raises_scenario_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ScenarioError, match="expected a mapping at the top level"):
        load_scenario(path)


def test_empty_yaml_raises_scenario_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ScenarioError, match="file is empty"):
        load_scenario(path)


def test_missing_file_raises_scenario_error(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="Scenario file not found"):
        load_scenario(tmp_path / "nope.yaml")


def test_error_message_names_the_scenario_file(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["name"] = "bad name"

    with pytest.raises(ScenarioError) as excinfo:
        load_mutated(tmp_path, mutate)

    assert "scenario.yaml" in str(excinfo.value)


# --- the shipped example ---------------------------------------------------


def test_shipped_failing_import_scenario_loads() -> None:
    scenario = load_scenario(SHIPPED_SCENARIO)

    assert scenario.name == "failing-import"
    assert scenario.rubric.endswith(".yaml")
    assert len(scenario.commits) >= 3


def test_shipped_scenario_introduces_the_bug_by_rewriting_an_existing_file() -> None:
    """The bug must arrive as a change to working code, not in the first commit."""
    scenario = load_scenario(SHIPPED_SCENARIO)

    versions = scenario.versions_of("inventory/models.py")
    assert len(versions) >= 2, "models.py must be rewritten, not written once"

    first_index, first_content = versions[0]
    last_index, last_content = versions[-1]
    assert first_index == 0
    assert last_index == len(scenario.commits) - 1

    assert "inventory.report" not in first_content, "the first version must not import report"
    assert "from inventory.report import" in last_content, "the last version must import report"

    # report.py imports models.py, which is what closes the cycle.
    report = scenario.content_at(last_index)["inventory/report.py"]
    assert "from inventory.models import" in report


def test_shipped_scenario_tree_grows_monotonically() -> None:
    scenario = load_scenario(SHIPPED_SCENARIO)

    sizes = [len(scenario.content_at(i)) for i in range(len(scenario.commits))]
    assert sizes == sorted(sizes), "a commit should never shrink the tree"
    assert sizes[0] < sizes[-1]
