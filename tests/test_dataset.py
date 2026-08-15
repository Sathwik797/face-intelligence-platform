import os
import shutil
import tempfile
import json
import pytest
import pandas as pd
import numpy as np
from PIL import Image

from scripts.prepare_dataset import prepare_identification_splits, calculate_sha256, parse_lfw_pairs
from scripts.validate_dataset import run_dataset_validation

@pytest.fixture
def temp_dataset_dir():
    """Creates a temporary synthetic dataset for deterministic testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create 3 synthetic identities with varying numbers of sample images
        for idx, (name, count) in enumerate([("Person_A", 25), ("Person_B", 22), ("Person_C", 5)]):
            person_dir = os.path.join(temp_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            for i in range(count):
                img = Image.new("RGB", (100, 100), color=(idx * 40, i * 10 % 255, 100))
                img.save(os.path.join(person_dir, f"{name}_{i:04d}.jpg"))
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_deterministic_split_generation(temp_dataset_dir):
    """Test that split generation with same random seed is 100% deterministic."""
    out_dir_1 = os.path.join(temp_dataset_dir, "eval_1")
    out_dir_2 = os.path.join(temp_dataset_dir, "eval_2")

    df_id_1, df_splits_1 = prepare_identification_splits(
        lfw_dir=temp_dataset_dir,
        output_eval_dir=out_dir_1,
        min_faces=20,
        enrollment_samples=2,
        val_ratio=0.5,
        seed=42
    )

    df_id_2, df_splits_2 = prepare_identification_splits(
        lfw_dir=temp_dataset_dir,
        output_eval_dir=out_dir_2,
        min_faces=20,
        enrollment_samples=2,
        val_ratio=0.5,
        seed=42
    )

    # Identical identities selected (Person_A and Person_B, Person_C filtered due to <20 images)
    assert list(df_id_1["identity"]) == ["Person_A", "Person_B"]
    assert list(df_id_1["identity"]) == list(df_id_2["identity"])

    # Identical split assignments and image order
    assert list(df_splits_1["split"]) == list(df_splits_2["split"])
    assert list(df_splits_1["image_name"]) == list(df_splits_2["image_name"])
    assert list(df_splits_1["file_hash"]) == list(df_splits_2["file_hash"])


def test_zero_cross_split_leakage(temp_dataset_dir):
    """Verify that there is zero overlap between enrollment, validation, and test splits."""
    out_dir = os.path.join(temp_dataset_dir, "eval")
    _, df_splits = prepare_identification_splits(
        lfw_dir=temp_dataset_dir,
        output_eval_dir=out_dir,
        min_faces=20,
        enrollment_samples=2,
        val_ratio=0.5,
        seed=42
    )

    enroll_hashes = set(df_splits[df_splits["split"] == "enrollment"]["file_hash"])
    val_hashes = set(df_splits[df_splits["split"] == "validation"]["file_hash"])
    test_hashes = set(df_splits[df_splits["split"] == "test"]["file_hash"])

    assert len(enroll_hashes.intersection(val_hashes)) == 0
    assert len(enroll_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0


def test_insufficient_samples_filtered(temp_dataset_dir):
    """Person_C with only 5 images should be excluded when min_faces=20."""
    out_dir = os.path.join(temp_dataset_dir, "eval")
    df_ids, _ = prepare_identification_splits(
        lfw_dir=temp_dataset_dir,
        output_eval_dir=out_dir,
        min_faces=20,
        seed=42
    )
    assert "Person_C" not in list(df_ids["identity"])
    assert len(df_ids) == 2


def test_metadata_files_exist():
    """Verify that metadata CSVs and JSON summary exist and have valid entries."""
    meta_dir = "data/metadata"
    identities_csv = os.path.join(meta_dir, "identities.csv")
    splits_csv = os.path.join(meta_dir, "splits.csv")
    pairs_csv = os.path.join(meta_dir, "verification_pairs.csv")
    summary_json = os.path.join(meta_dir, "dataset_summary.json")

    assert os.path.exists(identities_csv)
    assert os.path.exists(splits_csv)
    assert os.path.exists(summary_json)

    df_splits = pd.read_csv(splits_csv)
    assert len(df_splits) > 0
    assert set(df_splits["split"].unique()) == {"enrollment", "validation", "test"}

    with open(summary_json, "r") as f:
        summary = json.load(f)
    assert summary["status"] == "PASSED"
    assert summary["cross_split_leakage_count"] == 0

    if os.path.exists(pairs_csv):
        df_pairs = pd.read_csv(pairs_csv)
        assert len(df_pairs) == 6000
        assert set(df_pairs["fold"].unique()) == set(range(1, 11))
        assert set(df_pairs["is_same"].unique()) == {0, 1}
