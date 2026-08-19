from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_gaussian_module_is_present() -> None:
    assert (REPOSITORY_ROOT / "src/scmolt/trackc/gaussian.py").is_file()


def test_internal_runner_tree_is_not_vendored() -> None:
    assert not (REPOSITORY_ROOT / "scripts/a_new").exists()
    assert not (REPOSITORY_ROOT / "architecture").exists()
