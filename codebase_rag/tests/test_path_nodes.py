from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from tree_sitter import Tree

from codebase_rag.language_config import get_language_config_by_name
from codebase_rag.parsers.definition_processor import DefinitionProcessor
from codebase_rag.parsers.import_processor import ImportProcessor


class RecordingIngestor:
    """Records node and relationship ingestions for assertions."""

    def __init__(self) -> None:
        self.node_calls: list[tuple[str, dict[str, Any]]] = []
        self.rel_calls: list[
            tuple[
                tuple[str, str, Any], str, tuple[str, str, Any], dict[str, Any] | None
            ]
        ] = []

    def ensure_node_batch(self, label: str, properties: dict[str, Any]) -> None:
        """Record node creation calls."""
        self.node_calls.append((label, properties))

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, Any],
        rel_type: str,
        to_spec: tuple[str, str, Any],
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Record relationship creation calls."""
        self.rel_calls.append((from_spec, rel_type, to_spec, properties))

    def flush_all(self) -> None:
        """No-op for tests."""
        return None


class DummyImportProcessor(ImportProcessor):
    """No-op import processor that satisfies the ImportProcessor type hint."""

    def __init__(self) -> None:
        super().__init__(
            repo_path_getter=lambda: Path("."), project_name_getter=lambda: "dummy"
        )
        self.import_mapping: dict[str, dict[str, str]] = {}

    def parse_imports(
        self, root_node: Any, module_qn: str, language: str, queries: dict[str, Any]
    ) -> None:
        """Override the parent method with a no-op for tests."""
        pass


class FakeParser:
    """Minimal parser stub that returns a valid, but empty, tree-sitter Tree."""

    def parse(self, source_bytes: bytes) -> Tree:
        # Create a mock tree that has a valid, albeit minimal, root_node.
        # This prevents the "is not an acceptable base type" error.
        # We can't directly instantiate Tree or Node from Python, so we need a real parser.
        # The simplest way is to parse an empty string with a real parser.
        from tree_sitter import Language, Parser
        from tree_sitter_python import language as python_language

        parser = Parser(Language(python_language()))
        return parser.parse(b"")


class DummyQueryCursor:
    """QueryCursor stub that returns empty captures, avoiding real tree-sitter usage."""

    def __init__(self, query: Any) -> None:
        self._query = query

    def captures(self, node: Any) -> dict[str, list[Any]]:
        """Return an empty capture set so downstream loops do nothing."""
        return defaultdict(list)


def _make_definition_processor(
    repo_path: Path, project_name: str
) -> tuple[DefinitionProcessor, RecordingIngestor]:
    """Construct a DefinitionProcessor with a recording ingestor and no-op import processor."""
    ingestor = RecordingIngestor()
    dp = DefinitionProcessor(
        ingestor=ingestor,
        repo_path=repo_path,
        project_name=project_name,
        function_registry={},
        simple_name_lookup=defaultdict(set),
        import_processor=DummyImportProcessor(),
        module_qn_to_file_path={},
    )
    return dp, ingestor


def _make_queries_for(language: str) -> dict[str, Any]:
    """Create a minimal queries map for the given language."""
    config = get_language_config_by_name(language)
    if not config:
        raise ValueError(f"No config found for language: {language}")
    return {
        language: {
            "parser": FakeParser(),
            "config": config,
            "functions": object(),
            "classes": object(),
            "calls": object(),
            "imports": object(),
            "language": object(),
        }
    }


def _find_node_call(ingestor: RecordingIngestor, label: str) -> dict[str, Any] | None:
    """Find the first node call with the given label."""
    for lbl, props in ingestor.node_calls:
        if lbl == label:
            return props
    return None


def _has_relationship(
    ingestor: RecordingIngestor,
    from_spec: tuple[str, str, Any],
    rel_type: str,
    to_spec: tuple[str, str, Any],
) -> bool:
    """Check if a relationship call matching the given parameters exists."""
    for f, r, t, _ in ingestor.rel_calls:
        if f == from_spec and r == rel_type and t == to_spec:
            return True
    return False


def test_process_file_emits_filepath_node_and_relationships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensures Path node, Module-[:AT_PATH]->Path, and Path-[:BELONGS_TO]->Project are emitted for a regular file."""
    import codebase_rag.parsers.definition_processor as dp_mod

    monkeypatch.setattr(dp_mod, "QueryCursor", DummyQueryCursor)

    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    project_name = "testproj"

    file_path = repo_path / "pkg" / "mod.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text('print("hello")\n', encoding="utf-8")

    dp, ingestor = _make_definition_processor(repo_path, project_name)
    queries = _make_queries_for("python")

    # project node created in GraphUpdater, we can just mock that here
    ingestor.ensure_node_batch("Project", {"name": project_name})

    dp.process_file(
        file_path=file_path,
        language="python",
        queries=queries,
        structural_elements={},
    )

    relative_path = file_path.relative_to(repo_path)
    expected_module_qn = ".".join(
        [project_name] + list(relative_path.with_suffix("").parts)
    )
    expected_filepath_qualified_name = f"{project_name}.{relative_path}"

    fp_node = _find_node_call(ingestor, "Path")
    assert fp_node is not None, "Expected a Path node to be created"
    assert fp_node.get("qualified_name") == expected_filepath_qualified_name
    assert fp_node.get("path") == str(relative_path.as_posix())

    assert _has_relationship(
        ingestor,
        ("Module", "qualified_name", expected_module_qn),
        "AT_PATH",
        ("Path", "qualified_name", expected_filepath_qualified_name),
    ), "Expected Module -[:AT_PATH]-> Path relationship"

    assert _has_relationship(
        ingestor,
        ("Path", "qualified_name", expected_filepath_qualified_name),
        "BELONGS_TO",
        ("Project", "name", project_name),
    ), "Expected Path -[:BELONGS_TO]-> Project relationship"


def test_process_file_handles_init_py_module_qn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates that __init__.py maps the module QN to the parent package and still emits Path nodes and relationships."""
    import codebase_rag.parsers.definition_processor as dp_mod

    monkeypatch.setattr(dp_mod, "QueryCursor", DummyQueryCursor)

    repo_path = tmp_path / "repo2"
    repo_path.mkdir(parents=True, exist_ok=True)
    project_name = "testproj2"

    file_path = repo_path / "pkg" / "__init__.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("# package init\n", encoding="utf-8")

    dp, ingestor = _make_definition_processor(repo_path, project_name)
    queries = _make_queries_for("python")

    dp.process_file(
        file_path=file_path,
        language="python",
        queries=queries,
        structural_elements={},
    )

    relative_path = file_path.relative_to(repo_path)
    expected_module_qn = ".".join([project_name] + list(relative_path.parent.parts))
    expected_filepath_qn = f"{project_name}.{relative_path}"

    fp_node = _find_node_call(ingestor, "Path")
    assert fp_node is not None
    assert fp_node.get("qualified_name") == expected_filepath_qn
    assert fp_node.get("path") == str(relative_path)

    assert _has_relationship(
        ingestor,
        ("Module", "qualified_name", expected_module_qn),
        "AT_PATH",
        ("Path", "qualified_name", expected_filepath_qn),
    )

    assert _has_relationship(
        ingestor,
        ("Path", "qualified_name", expected_filepath_qn),
        "BELONGS_TO",
        ("Project", "name", project_name),
    )
