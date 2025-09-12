import pytest
import xxhash

from codebase_rag.parsers.utils import generate_range_id


def test_build_range_id() -> None:
    """Test that the range ID is correctly generated matching output in
    our other codebase"""
    project_name = "projId"
    file_path = "a/b/c.py"
    start_line = 20
    start_char = 1
    end_line = 22
    end_char = 8
    range_id = generate_range_id(
        project_name, file_path, start_line, start_char, end_line, end_char
    )
    assert range_id == "2e17c17c947236aa"


def test_hashes_single_line_range_deterministically() -> None:
    """Single-line ranges hash deterministically to xxhash64(raw)."""
    proj_id = "projId"
    file_path = "main.go"
    sl, sc, el, ec = 10, 5, 10, 15

    raw = f"{proj_id}|{file_path}|{sl}:{sc}-{el}:{ec}"
    expected = xxhash.xxh64_hexdigest(raw, seed=0)

    got = generate_range_id(proj_id, file_path, sl, sc, el, ec)
    assert got == expected


def test_handles_zero_values_correctly() -> None:
    """Zero values are valid and should hash deterministically."""
    proj_id = "projId"
    file_path = "file.txt"
    sl, sc, el, ec = 0, 0, 0, 0

    raw = f"{proj_id}|{file_path}|{sl}:{sc}-{el}:{ec}"
    expected = xxhash.xxh64_hexdigest(raw, seed=0)

    got = generate_range_id(proj_id, file_path, sl, sc, el, ec)
    assert got == expected


def test_returns_empty_string_when_file_path_is_empty() -> None:
    """Empty file path should yield an empty ID."""
    proj_id = "projId"
    file_path = ""
    sl, sc, el, ec = 10, 5, 10, 15

    got = generate_range_id(proj_id, file_path, sl, sc, el, ec)
    assert got == ""


@pytest.mark.parametrize(
    "sl,sc,el,ec",
    [
        (-1, 0, 1, 0),
        (10, 5, -1, 15),
        (10, 5, 10, -1),
        (-1, -1, -1, -1),
    ],
)
def test_returns_empty_string_when_any_range_value_is_negative(
    sl: int, sc: int, el: int, ec: int
) -> None:
    """Any negative range value should yield an empty ID."""
    proj_id = "projId"
    file_path = "file.txt"

    got = generate_range_id(proj_id, file_path, sl, sc, el, ec)
    assert got == ""


def test_is_reproducible_for_the_same_inputs() -> None:
    """Same inputs produce the same ID."""
    id1 = generate_range_id("projId", "main.go", 10, 5, 10, 15)
    id2 = generate_range_id("projId", "main.go", 10, 5, 10, 15)
    assert id1 == id2


def test_differs_across_different_project_ids_for_same_range() -> None:
    """Different project IDs produce different IDs for the same range."""
    id1 = generate_range_id("projA", "main.go", 10, 5, 10, 15)
    id2 = generate_range_id("projB", "main.go", 10, 5, 10, 15)
    assert id1 != id2
