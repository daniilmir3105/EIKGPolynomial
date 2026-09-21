import importlib
import importlib.metadata

import eikg


def test_public_version_matches_distribution_metadata() -> None:
    assert "__version__" in eikg.__all__
    assert eikg.__version__ == importlib.metadata.version("eikgpolynomial")


def test_public_version_falls_back_for_uninstalled_source_tree(monkeypatch) -> None:
    def missing_distribution(distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    with monkeypatch.context() as context:
        context.setattr(importlib.metadata, "version", missing_distribution)
        reloaded = importlib.reload(eikg)
        assert reloaded.__version__ == "0.2.0"

    # Restore the metadata-derived value for tests that import the module later.
    importlib.reload(eikg)
