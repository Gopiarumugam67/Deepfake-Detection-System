"""
Tests for src/preprocessing/dataset_builder.py.

Uses monkeypatched CFG paths so the test never touches the real
data/processed or data/splits directories.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_dummy_image  # noqa: E402
from src.preprocessing import dataset_builder  # noqa: E402


def _make_fake_cfg(tmp_dirs: dict[str, Path]):
    """Build a stand-in CFG object with paths pointed at tmp_project_dirs."""
    return SimpleNamespace(
        paths=SimpleNamespace(
            data_processed=tmp_dirs["processed"],
            data_splits=tmp_dirs["splits"],
        ),
        split=SimpleNamespace(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            random_seed=42, stratify=True,
        ),
        class_names=("real", "fake"),
    )


def _populate_staging(staging_dir: Path, n_real: int = 10, n_fake: int = 10) -> None:
    for i in range(n_real):
        make_dummy_image(staging_dir / "real" / f"real_{i}.jpg", color=(0, 200, 0))
    for i in range(n_fake):
        make_dummy_image(staging_dir / "fake" / f"fake_{i}.jpg", color=(200, 0, 0))


class TestCollectLabeledPaths:
    def test_collects_all_images_with_correct_labels(self, tmp_project_dirs):
        _populate_staging(tmp_project_dirs["staging"], n_real=3, n_fake=5)

        df = dataset_builder._collect_labeled_paths(tmp_project_dirs["staging"])

        assert len(df) == 8
        assert set(df["label"].unique()) == {"real", "fake"}
        assert (df["label"] == "real").sum() == 3
        assert (df["label"] == "fake").sum() == 5

    def test_raises_if_no_images_found(self, tmp_project_dirs):
        empty_staging = tmp_project_dirs["root"] / "empty_staging"
        (empty_staging / "real").mkdir(parents=True)
        (empty_staging / "fake").mkdir(parents=True)

        with pytest.raises(RuntimeError, match="No labeled images found"):
            dataset_builder._collect_labeled_paths(empty_staging)

    def test_warns_but_continues_on_missing_class_dir(self, tmp_project_dirs, caplog):
        # Only populate "real" — "fake" dir intentionally absent.
        for i in range(3):
            make_dummy_image(tmp_project_dirs["staging"] / "real" / f"real_{i}.jpg")
        (tmp_project_dirs["staging"] / "fake").rmdir()

        df = dataset_builder._collect_labeled_paths(tmp_project_dirs["staging"])
        assert len(df) == 3
        assert set(df["label"].unique()) == {"real"}


class TestBuildSplits:
    def test_split_ratios_and_stratification(self, tmp_project_dirs, monkeypatch):
        monkeypatch.setattr(dataset_builder, "CFG", _make_fake_cfg(tmp_project_dirs))
        _populate_staging(tmp_project_dirs["staging"], n_real=20, n_fake=20)

        splits = dataset_builder.build_splits(staging_dir=tmp_project_dirs["staging"])

        total = sum(len(df) for df in splits.values())
        assert total == 40
        # Both classes should be represented in every split (stratified).
        for split_name, df in splits.items():
            assert set(df["label"].unique()) == {"real", "fake"}, f"{split_name} missing a class"

        # Roughly the configured ratios (60/20/20), allowing for rounding.
        assert 20 <= len(splits["train"]) <= 26
        assert 5 <= len(splits["val"]) <= 11
        assert 5 <= len(splits["test"]) <= 11

    def test_manifests_written_to_disk(self, tmp_project_dirs, monkeypatch):
        monkeypatch.setattr(dataset_builder, "CFG", _make_fake_cfg(tmp_project_dirs))
        _populate_staging(tmp_project_dirs["staging"], n_real=10, n_fake=10)

        dataset_builder.build_splits(staging_dir=tmp_project_dirs["staging"])

        for split_name in ("train", "val", "test"):
            manifest = tmp_project_dirs["splits"] / f"{split_name}.csv"
            assert manifest.exists(), f"missing manifest for {split_name}"

    def test_files_physically_copied_into_processed_dirs(self, tmp_project_dirs, monkeypatch):
        monkeypatch.setattr(dataset_builder, "CFG", _make_fake_cfg(tmp_project_dirs))
        _populate_staging(tmp_project_dirs["staging"], n_real=10, n_fake=10)

        splits = dataset_builder.build_splits(staging_dir=tmp_project_dirs["staging"])

        for split_name, df in splits.items():
            for filepath in df["filepath"]:
                assert Path(filepath).exists()
                assert str(tmp_project_dirs["processed"]) in filepath
