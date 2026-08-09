import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/geopose"


def test_publication_package_has_stage_oriented_hierarchy():
    expected = {
        "cli": {"preregister.py", "train.py", "test.py"},
        "init": {"data.py", "loss.py", "model.py"},
        "refine": {"backbone.py", "data.py", "loss.py", "model.py", "network.py"},
        "registration": {
            "geometry.py",
            "images.py",
            "initialization.py",
            "optimization.py",
            "pipeline.py",
            "preregistration.py",
            "views.py",
        },
        "data": {"augmentations.py", "fiducials.py", "preparation.py", "splits.py"},
        "shared": {
            "blocks.py",
            "contracts.py",
            "losses.py",
            "metrics.py",
            "pose.py",
            "visualization.py",
        },
    }
    for package, files in expected.items():
        present = {path.name for path in (PACKAGE / package).glob("*.py")}
        assert files <= present, package


def test_obsolete_flat_modules_are_not_source_files():
    obsolete = [
        PACKAGE / "inference.py",
        PACKAGE / "training.py",
        PACKAGE / "preregistration.py",
        PACKAGE / "data_preparation.py",
        PACKAGE / "models/__init__.py",
    ]
    assert not any(path.exists() for path in obsolete)


def test_internal_imports_do_not_reference_obsolete_packages():
    forbidden = (
        "geopose.models",
        "geopose.inference",
        "geopose.training",
        "geopose.preregistration",
        "geopose.data_preparation",
    )
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), path
            if module is not None:
                assert not module.startswith(forbidden), path


def test_root_scripts_only_forward_to_cli_modules():
    expected = {
        "preregister.py": "geopose.cli.preregister",
        "train.py": "geopose.cli.train",
        "test.py": "geopose.cli.test",
    }
    for filename, module in expected.items():
        tree = ast.parse((ROOT / filename).read_text())
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert module in imported
