import numpy as np


def test_arcface_preprocess_resizes_non_112_crop():
    from ml.embedder import ArcFaceEmbedder
    embedder = object.__new__(ArcFaceEmbedder)
    crop = np.zeros((96, 96, 3), dtype=np.uint8)
    blob = embedder._preprocess_crop(crop)
    assert blob.shape == (3, 112, 112)


def test_gallery_round_trip_does_not_require_pickle(tmp_path):
    from ml.gallery import IdentityGallery
    gallery = IdentityGallery()
    gallery.add_templates("Alice", np.eye(512, dtype=np.float32)[:1])
    path = tmp_path / "gallery.npz"
    gallery.save(str(path))
    loaded = IdentityGallery.load(str(path))
    assert loaded.identities == ["Alice"]
    assert loaded.embeddings.shape == (1, 512)
