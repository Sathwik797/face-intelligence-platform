import os
import tempfile
import pytest
import numpy as np

from ml.gallery import IdentityGallery

def test_gallery_initialization():
    gallery = IdentityGallery()
    assert gallery.total_templates == 0
    assert gallery.unique_identities == []
    assert gallery.embedding_dim == 512


def test_gallery_add_single_and_multiple_templates():
    gallery = IdentityGallery()
    emb_alice_1 = np.zeros((1, 512), dtype=np.float32)
    emb_alice_1[0, 0] = 1.0

    emb_alice_2 = np.zeros((1, 512), dtype=np.float32)
    emb_alice_2[0, 0] = 0.9
    emb_alice_2[0, 1] = 0.1

    emb_bob = np.zeros((1, 512), dtype=np.float32)
    emb_bob[0, 50] = 1.0

    gallery.add_templates("Alice", emb_alice_1)
    gallery.add_templates("Alice", emb_alice_2)
    gallery.add_templates("Bob", emb_bob)

    assert gallery.total_templates == 3
    assert gallery.unique_identities == ["Alice", "Bob"]
    assert gallery.identities == ["Alice", "Alice", "Bob"]


def test_gallery_search_known_identity():
    gallery = IdentityGallery()
    emb_alice = np.zeros((1, 512), dtype=np.float32)
    emb_alice[0, 0] = 1.0

    emb_bob = np.zeros((1, 512), dtype=np.float32)
    emb_bob[0, 1] = 1.0

    gallery.add_templates("Alice", emb_alice)
    gallery.add_templates("Bob", emb_bob)

    # Query Alice
    query_alice = np.zeros(512, dtype=np.float32)
    query_alice[0] = 0.95
    query_alice[2] = 0.1

    rec_id, best_cand, sim, is_rec = gallery.search(query_alice, threshold=0.45)
    assert rec_id == "Alice"
    assert best_cand == "Alice"
    assert is_rec is True
    assert sim > 0.9


def test_gallery_search_unknown_open_set_rejection():
    gallery = IdentityGallery()
    emb_alice = np.zeros((1, 512), dtype=np.float32)
    emb_alice[0, 0] = 1.0
    gallery.add_templates("Alice", emb_alice)

    # Query orthogonal to Alice (similarity 0.0 < 0.45 threshold)
    query_charlie = np.zeros(512, dtype=np.float32)
    query_charlie[200] = 1.0

    rec_id, best_cand, sim, is_rec = gallery.search(query_charlie, threshold=0.45)
    assert rec_id is None
    assert best_cand == "Alice"
    assert is_rec is False
    assert sim < 0.45


def test_gallery_dimension_mismatch_error():
    gallery = IdentityGallery()
    emb_valid = np.zeros((1, 512), dtype=np.float32)
    emb_valid[0, 0] = 1.0
    gallery.add_templates("Alice", emb_valid)

    # 128D instead of 512D
    emb_invalid = np.zeros((1, 128), dtype=np.float32)
    with pytest.raises(ValueError):
        gallery.add_templates("Bob", emb_invalid)


def test_gallery_save_and_load():
    gallery = IdentityGallery()
    emb_alice = np.zeros((1, 512), dtype=np.float32)
    emb_alice[0, 0] = 1.0
    gallery.add_templates("Alice", emb_alice)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_gallery.npz")
        gallery.save(save_path)
        assert os.path.exists(save_path)

        loaded_gallery = IdentityGallery.load(save_path)
        assert loaded_gallery.total_templates == 1
        assert loaded_gallery.unique_identities == ["Alice"]
        assert loaded_gallery.embeddings.shape == (1, 512)
