"""Documentation style checks for the publication Python sources."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCES = sorted((ROOT / "src/geopose").rglob("*.py")) + [
    ROOT / "preregister.py",
    ROOT / "train.py",
    ROOT / "test.py",
]
DOCUMENTED_NODES = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def test_every_python_module_has_a_concise_header():
    for path in PYTHON_SOURCES:
        module = ast.parse(path.read_text())
        docstring = ast.get_docstring(module, clean=False)
        assert docstring, f"Missing module docstring: {path.relative_to(ROOT)}"
        assert "\n" not in docstring, f"Module header must be one line: {path.relative_to(ROOT)}"
        assert docstring.endswith("."), f"Module header must be a sentence: {path.relative_to(ROOT)}"


def test_docstrings_are_single_purpose_summaries():
    for path in PYTHON_SOURCES:
        module = ast.parse(path.read_text())
        for node in ast.walk(module):
            if not isinstance(node, DOCUMENTED_NODES):
                continue
            docstring = ast.get_docstring(node, clean=False)
            assert docstring is None or "\n" not in docstring, (
                f"Multiline docstring in {path.relative_to(ROOT)}:"
                f"{getattr(node, 'name', '<module>')}"
            )
