from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from codebase_rag.language_config import get_language_config_by_name
from codebase_rag.parsers.definition_processor import DefinitionProcessor
from codebase_rag.parsers.import_processor import ImportProcessor
from codebase_rag.parsers.utils import generate_range_id


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
        return None


class FakeNode:
    """Minimal tree-sitter-like Node for testing start/end points, names, and bodies."""

    def __init__(
        self,
        node_type: str,
        start_point: tuple[int, int] = (0, 0),
        end_point: tuple[int, int] = (0, 0),
        text: bytes | None = None,
        fields: dict[str, FakeNode] | None = None,
    ) -> None:
        self.type = node_type
        self.start_point = start_point
        self.end_point = end_point
        self.text = text
        self.parent: FakeNode | None = None
        self.children: list[FakeNode] = []
        self._fields = fields or {}

    def child_by_field_name(self, name: str) -> FakeNode | None:
        return self._fields.get(name)


class DummyQueryCursor:
    """Configurable cursor that returns pre-registered captures by (query_id, node_id)."""

    registry: dict[tuple[int, int], dict[str, list[FakeNode]]] = {}

    def __init__(self, query: Any) -> None:
        self._query = query

    def captures(self, node: Any) -> dict[str, list[FakeNode]]:
        return DummyQueryCursor.registry.get((id(self._query), id(node)), {})


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


def _queries_for(
    language: str,
    functions_query: Any,
    classes_query: Any,
    language_obj: Any | None = None,
) -> dict[str, Any]:
    """Create a minimal queries map for the given language."""
    config = get_language_config_by_name(language)
    if not config:
        raise ValueError(f"No config found for language: {language}")
    return {
        language: {
            "parser": object(),
            "config": config,
            "functions": functions_query,
            "classes": classes_query,
            "calls": object(),
            "imports": object(),
            "language": language_obj or object(),
        }
    }


def _find_node(
    ingestor: RecordingIngestor, label: str, name: str
) -> dict[str, Any] | None:
    """Find the first node call with the given label and name."""
    for lbl, props in ingestor.node_calls:
        if lbl == label and props.get("name") == name:
            return props
    return None


def test_python_function_class_and_method_range_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure Python Function, Class, and Method nodes have correct range_id computed from file path and exact ranges."""
    import codebase_rag.parsers.definition_processor as dp_mod

    # Monkeypatch QueryCursor to our dummy
    monkeypatch.setattr(dp_mod, "QueryCursor", DummyQueryCursor)
    monkeypatch.setattr(dp_mod, "Node", FakeNode)

    project_name = "proj"
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    rel_path = "pkg/mod.py"
    file_path = repo_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("# stub\n", encoding="utf-8")

    dp, ingestor = _make_definition_processor(repo_path, project_name)

    # Build fake AST nodes
    root_module = FakeNode("module")

    func_name_node = FakeNode("identifier", text=b"f")
    func_node = FakeNode(
        "function_definition",
        start_point=(9, 5),
        end_point=(9, 15),
        fields={"name": func_name_node},
    )
    func_node.parent = root_module

    class_name_node = FakeNode("identifier", text=b"C")
    class_body_node = FakeNode("block")
    class_node = FakeNode(
        "class_definition",
        start_point=(19, 0),
        end_point=(29, 2),
        fields={"name": class_name_node, "body": class_body_node},
    )
    class_node.parent = root_module

    method_name_node = FakeNode("identifier", text=b"m")
    method_node = FakeNode(
        "function_definition",
        start_point=(20, 4),
        end_point=(28, 1),
        fields={"name": method_name_node},
    )
    method_node.parent = class_body_node

    # Create query sentinels
    functions_query = object()
    classes_query = object()
    queries = _queries_for("python", functions_query, classes_query)

    # Register captures for functions/classes and class body methods
    DummyQueryCursor.registry[(id(functions_query), id(root_module))] = {
        "function": [func_node]
    }
    DummyQueryCursor.registry[(id(classes_query), id(root_module))] = {
        "class": [class_node]
    }
    DummyQueryCursor.registry[(id(functions_query), id(class_body_node))] = {
        "function": [method_node]
    }

    # Ingest functions, classes, and methods
    module_qn = ".".join([project_name] + list(Path(rel_path).with_suffix("").parts))
    dp._ingest_all_functions(root_module, module_qn, "python", queries, rel_path)
    dp._ingest_classes_and_methods(root_module, module_qn, "python", queries, rel_path)

    # Function assertions
    f_props = _find_node(ingestor, "Function", "f")
    assert f_props is not None
    assert f_props["start_line"] == func_node.start_point[0] + 1
    assert f_props["end_line"] == func_node.end_point[0] + 1
    expected_func_id = generate_range_id(
        project_name,
        rel_path,
        func_node.start_point[0] + 1,
        func_node.start_point[1],
        func_node.end_point[0] + 1,
        func_node.end_point[1],
    )
    assert f_props["range_id"] == expected_func_id

    # Class assertions
    c_props = _find_node(ingestor, "Class", "C")
    assert c_props is not None
    assert c_props["start_line"] == class_node.start_point[0] + 1
    assert c_props["end_line"] == class_node.end_point[0] + 1
    expected_class_id = generate_range_id(
        project_name,
        rel_path,
        class_node.start_point[0] + 1,
        class_node.start_point[1],
        class_node.end_point[0] + 1,
        class_node.end_point[1],
    )
    assert c_props["range_id"] == expected_class_id

    # Method assertions
    m_props = _find_node(ingestor, "Method", "m")
    assert m_props is not None
    assert m_props["start_line"] == method_node.start_point[0] + 1
    assert m_props["end_line"] == method_node.end_point[0] + 1
    expected_method_id = generate_range_id(
        project_name,
        rel_path,
        method_node.start_point[0] + 1,
        method_node.start_point[1],
        method_node.end_point[0] + 1,
        method_node.end_point[1],
    )
    assert m_props["range_id"] == expected_method_id


def test_es6_export_function_range_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure ES6 export function nodes get correct range_id derived from file path and exact ranges."""
    import codebase_rag.parsers.definition_processor as dp_mod

    # State-independent dummy Query and QueryCursor for ES6 export ingestion
    class ES6DummyQuery:
        def __init__(self, _lang: Any, _text: str) -> None:
            self._lang = _lang
            self._text = _text

    class ES6DummyCursor:
        def __init__(self, _query: Any) -> None:
            self._query = _query

        def captures(self, _node: Any) -> dict[str, list[FakeNode]]:
            # Return a single export: export const foo = function() {}
            export_name = FakeNode("identifier", text=b"foo")
            export_fn = FakeNode(
                "function_expression", start_point=(4, 2), end_point=(7, 5)
            )
            return {"export_name": [export_name], "export_function": [export_fn]}

    # Monkeypatch Query and QueryCursor used by ES6 export ingestion
    monkeypatch.setattr(dp_mod, "Query", ES6DummyQuery)
    monkeypatch.setattr(dp_mod, "QueryCursor", ES6DummyCursor)

    project_name = "webproj"
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    rel_path = "lib/mod.js"

    dp, ingestor = _make_definition_processor(repo_path, project_name)

    # Minimal queries for JS; language object is arbitrary since we stub Query
    queries = _queries_for(
        "javascript",
        functions_query=object(),
        classes_query=object(),
        language_obj=object(),
    )

    module_qn = ".".join([project_name] + list(Path(rel_path).with_suffix("").parts))
    root_node = FakeNode("program")

    # Call ES6 export ingestion directly with path
    dp._ingest_es6_exports(root_node, module_qn, "javascript", queries, rel_path)

    # Assert a Function node for 'foo' exists with correct range_id
    foo_props = _find_node(ingestor, "Function", "foo")
    assert foo_props is not None
    expected_id = generate_range_id(
        project_name, rel_path, 5, 2, 8, 5
    )  # start/end lines are +1 from points
    assert foo_props["range_id"] == expected_id
