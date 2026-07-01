"""Regression tests for the code-review fixes (cache interval, candidate
stratification/exclusion, idempotent writes, normalization, atomic labels)."""

import numpy as np
import pytest

from peaks.cache import EmbeddingCache
from peaks.config import ScoringConfig
from peaks.embedding import FakeEmbedder
from peaks.labels import LabelStore
from peaks.models import Scene
from peaks.pipeline import embed_library, gather_candidates, score_library
from peaks.scoring import Segment, extract_segments, normalize_scores
from peaks.scoring import make_similarity_scorer


# --- cache interval validation ----------------------------------------------


def test_cache_has_validates_interval(tmp_path):
    cache = EmbeddingCache(tmp_path)
    t = np.array([0.0, 2.0], dtype=np.float32)
    v = np.zeros((2, 4), dtype=np.float32)
    cache.save("k", "fake", t, v, meta={"interval": 2.0})
    assert cache.has("k", "fake")  # existence only
    assert cache.has("k", "fake", interval=2.0)
    assert not cache.has("k", "fake", interval=4.0)  # density changed -> re-embed


def test_cache_has_interval_missing_meta_is_stale(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.save("k", "fake", np.zeros(1, dtype=np.float32), np.zeros((1, 4), dtype=np.float32))
    assert not cache.has("k", "fake", interval=2.0)


class _StubImage:
    def __init__(self, payload: bytes):
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _StubSampler:
    interval = 2.0

    def __init__(self, frames):
        self._frames = frames

    def iter_frames(self, path):
        yield from self._frames


def _scene(id_, key):
    return Scene.from_dict(
        {
            "id": id_,
            "title": "",
            "files": [{"path": f"/m/{id_}.mp4", "fingerprints": [
                {"type": "oshash", "value": key}]}],
            "scene_markers": [],
        }
    )


def test_embed_reembeds_when_interval_changes(tmp_path):
    emb = FakeEmbedder(dim=8)
    cache = EmbeddingCache(tmp_path)
    frames = [(i * 2.0, _StubImage(f"f{i}".encode())) for i in range(3)]
    s1 = _StubSampler(frames)
    stats = embed_library([_scene("1", "k1")], s1, emb, cache, log=lambda *_: None)
    assert stats["embedded"] == 1

    s2 = _StubSampler(frames)
    s2.interval = 4.0  # coarser sampling requested
    stats2 = embed_library([_scene("1", "k1")], s2, emb, cache, log=lambda *_: None)
    assert stats2["embedded"] == 1 and stats2["skipped"] == 0  # not a stale hit


# --- candidate stratification + exclusion ------------------------------------


def _seeded_cache(tmp_path, n=20, dim=8):
    cache = EmbeddingCache(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    times = np.arange(n, dtype=np.float32) * 2.0
    cache.save("k1", "fake", times, vecs, meta={"scene_id": "1", "path": "/m/1.mp4"})
    return cache


def test_limit_preserves_random_negatives(tmp_path):
    cache = _seeded_cache(tmp_path)

    # scorer: frame index order (later frames score higher)
    def score_frames(vecs):
        return np.arange(vecs.shape[0], dtype=np.float32)

    cands = gather_candidates(
        cache, "fake", score_frames,
        top_per_scene=3, random_per_scene=3, seed=42, limit=4,
    )
    assert len(cands) == 4
    # the random pool must survive the limit: not all four can be the
    # top-3 scorers (indices 17,18,19 -> scores >= 17)
    assert any(c.score < 17.0 for c in cands)


def test_exclude_already_labeled(tmp_path):
    cache = _seeded_cache(tmp_path)

    def score_frames(vecs):
        return np.arange(vecs.shape[0], dtype=np.float32)

    first = gather_candidates(
        cache, "fake", score_frames, top_per_scene=2, random_per_scene=0, seed=0
    )
    labeled = {(c.key, round(c.time, 2)) for c in first}
    second = gather_candidates(
        cache, "fake", score_frames,
        top_per_scene=2, random_per_scene=0, seed=0, exclude=labeled,
    )
    assert not {(c.key, round(c.time, 2)) for c in second} & labeled


def test_labeled_ids_feeds_exclusion(tmp_path):
    store = LabelStore(tmp_path / "labels.json")
    store.add("k1", 38.0, 1, "apex")
    assert ("k1", 38.0) in store.labeled_ids("apex")
    assert store.labeled_ids("other") == set()


# --- idempotent marker writes -------------------------------------------------


class _CapturingClient:
    def __init__(self):
        self.markers = []

    def find_or_create_tag(self, name):
        return type("Tag", (), {"id": "t1", "name": name})()

    def create_scene_marker(self, *, scene_id, seconds, primary_tag_id, title, end_seconds):
        self.markers.append((scene_id, seconds, end_seconds))


def _scene_with_marker(id_, key, start, end, tag="apex"):
    return Scene.from_dict(
        {
            "id": id_,
            "title": "",
            "files": [{"path": f"/m/{id_}.mp4", "fingerprints": [
                {"type": "oshash", "value": key}]}],
            "scene_markers": [
                {"id": "m1", "seconds": start, "end_seconds": end,
                 "title": tag, "primary_tag": {"id": "t1", "name": tag}}
            ],
        }
    )


def _loved_setup(tmp_path):
    emb = FakeEmbedder(dim=24)
    cache = EmbeddingCache(tmp_path)
    loved = _StubImage(b"loved")
    frames = [_StubImage(f"noise{i}".encode()) for i in range(20)]
    for i in range(8, 12):
        frames[i] = loved
    vecs = emb.embed_images(frames)
    times = np.arange(20, dtype=np.float32) * 2.0
    cache.save("k1", emb.name, times, vecs)
    refs = emb.embed_images([loved])
    scoring = ScoringConfig(
        high=0.9, low=0.7, min_duration=2.0, merge_gap=2.0, smooth_window=1, pad=0.0
    )
    return emb, cache, refs, scoring


def test_rewrite_skips_overlapping_existing_marker(tmp_path):
    emb, cache, refs, scoring = _loved_setup(tmp_path)
    client = _CapturingClient()
    scorer = make_similarity_scorer(refs, "max")
    # the found segment is 16..22s; scene already carries an apex marker there
    scene = _scene_with_marker("1", "k1", 15.0, 23.0)
    stats = score_library(
        [scene], cache, emb.name, scorer, scoring,
        client=client, tag_name="apex", write=True, log=lambda *_: None,
    )
    assert client.markers == []  # nothing duplicated
    assert stats["existing"] == 1


def test_rewrite_ignores_other_tags_markers(tmp_path):
    emb, cache, refs, scoring = _loved_setup(tmp_path)
    client = _CapturingClient()
    scorer = make_similarity_scorer(refs, "max")
    scene = _scene_with_marker("1", "k1", 15.0, 23.0, tag="somebody-elses-tag")
    score_library(
        [scene], cache, emb.name, scorer, scoring,
        client=client, tag_name="apex", write=True, log=lambda *_: None,
    )
    assert len(client.markers) == 1  # our tag wasn't there -> marker created


# --- score normalization ------------------------------------------------------


def test_normalize_none_passthrough():
    s = np.array([0.1, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(normalize_scores(s, "none"), s)


def test_normalize_scene_z():
    s = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    z = normalize_scores(s, "scene-z")
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std()) - 1.0) < 1e-5


def test_normalize_scene_z_constant_scores():
    z = normalize_scores(np.full(5, 0.7, dtype=np.float32), "scene-z")
    np.testing.assert_array_equal(z, np.zeros(5, dtype=np.float32))


def test_normalize_unknown_mode():
    with pytest.raises(ValueError):
        normalize_scores(np.zeros(2), "bogus")


# --- segment merge weighting + post-pad overlap -------------------------------


def test_merge_weights_means_by_duration():
    # long low segment (0-8, mean~1) + short high one (9-10, mean~3), gap 1
    scores = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3], dtype=np.float32)
    times = np.arange(12, dtype=np.float32)
    segs = extract_segments(scores, times, high=0.5, min_duration=1.0, merge_gap=2.0)
    assert len(segs) == 1
    # duration-weighted mean must sit near the long segment's mean, not 2.0
    assert segs[0].mean_score < 1.5


def test_pad_remerges_created_overlaps():
    scores = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.float32)
    times = np.arange(8, dtype=np.float32)
    segs = extract_segments(
        scores, times, high=0.5, min_duration=1.0, merge_gap=0.5, pad=1.5
    )
    # padding pushes 1-2 and 5-6 into overlap -> must come back merged
    assert len(segs) == 1
    for a, b in zip(segs, segs[1:]):
        assert b.start > a.end


# --- atomic label saves --------------------------------------------------------


def test_label_save_is_atomic(tmp_path):
    path = tmp_path / "labels.json"
    store = LabelStore(path)
    store.add("k1", 1.0, 1, "apex")
    store.save()
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()  # temp cleaned up
    assert len(LabelStore(path)) == 1
