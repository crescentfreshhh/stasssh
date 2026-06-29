"""Orchestration: the two passes that tie the pieces together.

  embed_library  — Stash scenes → sampled frames → embeddings → on-disk cache.
                   The GPU-heavy, resumable, one-time pass.

  score_library  — cached embeddings → similarity vs your references → segments
                   → Stash `apex` markers (or a dry-run preview).

These need ffmpeg + the `[ml]` extra + a live Stash to actually run; the
building blocks they call (sampler, embedder, cache, scorer) are unit-tested
offline. Kept deliberately thin so the logic lives in the tested modules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .cache import EmbeddingCache, path_key
from .embedding import Embedder
from .models import Scene
from .sampling import FrameSampler
from .scoring import Segment, extract_segments, smooth

Logger = Callable[[str], None]

# A frame scorer maps (n, dim) embeddings -> (n,) per-frame scores. Both the
# Tier-1 similarity closure and a trained classifier's predict_proba fit this.
ScoreFn = Callable[[np.ndarray], np.ndarray]


def scene_key(scene: Scene) -> str:
    """Cache key for a scene: prefer the file fingerprint, else hash the path."""
    return scene.fingerprint or path_key(scene.path or scene.id)


def _embed_in_batches(
    embedder: Embedder, images: list, batch_size: int
) -> np.ndarray:
    if not images:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    chunks = []
    for i in range(0, len(images), batch_size):
        chunks.append(embedder.embed_images(images[i : i + batch_size]))
    return np.concatenate(chunks, axis=0)


def embed_library(
    scenes: Iterable[Scene],
    sampler: FrameSampler,
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    batch_size: int = 64,
    log: Logger = print,
) -> dict:
    """Embed every scene not already cached. Resumable + idempotent."""
    stats = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
    for scene in scenes:
        key = scene_key(scene)
        if cache.has(key, embedder.name):
            stats["skipped"] += 1
            continue
        if not scene.path:
            log(f"  ! scene {scene.id} has no file; skipping")
            stats["failed"] += 1
            continue
        try:
            times, frames = [], []
            for ts, img in sampler.iter_frames(scene.path):
                times.append(ts)
                frames.append(img.copy())  # detach from the temp-file context
            vecs = _embed_in_batches(embedder, frames, batch_size)
            cache.save(
                key,
                embedder.name,
                np.asarray(times, dtype=np.float32),
                vecs,
                meta={
                    "scene_id": scene.id,
                    "path": scene.path,
                    "interval": sampler.interval,
                    "model": embedder.name,
                    "dim": embedder.dim,
                    "n_frames": len(times),
                },
            )
            stats["embedded"] += 1
            stats["frames"] += len(times)
            log(f"  + scene {scene.id}: {len(times)} frames -> cache")
        except Exception as exc:  # keep the batch going; log the casualty
            log(f"  ! scene {scene.id} failed: {exc}")
            stats["failed"] += 1
    return stats


def load_references(embedder: Embedder, references_dir: str | Path) -> np.ndarray:
    """Embed every image in a directory into reference vectors (m, dim)."""
    from PIL import Image as PILImage  # lazy

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(
        p for p in Path(references_dir).glob("**/*") if p.suffix.lower() in exts
    )
    if not files:
        raise FileNotFoundError(f"no reference images found in {references_dir}")
    images = [PILImage.open(p).convert("RGB") for p in files]
    return embedder.embed_images(images)


def score_scene(
    times: np.ndarray,
    vecs: np.ndarray,
    score_frames: ScoreFn,
    scoring,
) -> list[Segment]:
    """Pure scoring for one scene's cached embeddings (no I/O).

    `score_frames` is the Tier-agnostic scorer: similarity closure or a trained
    classifier's predict_proba.
    """
    scores = smooth(score_frames(vecs), scoring.smooth_window)
    return extract_segments(
        scores,
        times,
        high=scoring.high,
        low=scoring.low,
        min_duration=scoring.min_duration,
        merge_gap=scoring.merge_gap,
        max_duration=scoring.max_duration or None,
        pad=scoring.pad,
    )


def score_library(
    scenes: Iterable[Scene],
    cache: EmbeddingCache,
    embedder_name: str,
    score_frames: ScoreFn,
    scoring,
    *,
    client=None,
    tag_name: str = "apex",
    write: bool = False,
    log: Logger = print,
) -> dict:
    """Score cached scenes into segments; optionally write Stash markers.

    write=False is a dry run: it logs the segments it *would* create, perfect
    for tuning thresholds before touching Stash.
    """
    stats = {"scenes": 0, "segments": 0, "skipped": 0}
    tag = None
    if write:
        if client is None:
            raise ValueError("write=True requires a client")
        tag = client.find_or_create_tag(tag_name)

    for scene in scenes:
        key = scene_key(scene)
        if not cache.has(key, embedder_name):
            stats["skipped"] += 1
            continue
        times, vecs, _ = cache.load(key, embedder_name)
        segs = score_scene(times, vecs, score_frames, scoring)
        stats["scenes"] += 1
        stats["segments"] += len(segs)
        for s in segs:
            if write:
                client.create_scene_marker(
                    scene_id=scene.id,
                    seconds=s.start,
                    primary_tag_id=tag.id,
                    title=tag_name,
                    end_seconds=s.end,
                )
            else:
                log(
                    f"  ~ scene {scene.id}: {s.start:7.1f}-{s.end:7.1f}s "
                    f"peak={s.peak_score:.3f}"
                )
    return stats


# --- Tier 2: training-set assembly + candidate gathering --------------------


@dataclass
class Candidate:
    """A frame to be labeled: where it is + the current model's score for it."""

    key: str
    scene_id: str | None
    path: str | None
    time: float
    score: float


def build_training_set(
    label_store, cache: EmbeddingCache, model_name: str, profile: str
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (X, y) from labels: each label's nearest cached frame vector.

    Loads each scene's cache once. Skips labels whose scene isn't cached.
    """
    by_key: dict[str, list] = defaultdict(list)
    for lab in label_store.for_profile(profile):
        by_key[lab.key].append(lab)

    rows, ys = [], []
    for key, labs in by_key.items():
        if not cache.has(key, model_name):
            continue
        times, vecs, _ = cache.load(key, model_name)
        if len(times) == 0:
            continue
        for lab in labs:
            idx = int(np.argmin(np.abs(times - lab.time)))
            rows.append(vecs[idx])
            ys.append(lab.label)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.asarray(rows, dtype=np.float32), np.asarray(ys, dtype=int)


def gather_candidates(
    cache: EmbeddingCache,
    model_name: str,
    score_frames: ScoreFn,
    *,
    top_per_scene: int = 3,
    random_per_scene: int = 1,
    seed: int = 0,
    limit: int | None = None,
) -> list[Candidate]:
    """Propose frames to label: each scene's highest-scoring frames (active
    learning) plus a few random ones (for diverse negatives)."""
    rng = np.random.default_rng(seed)
    cands: list[Candidate] = []
    for key in cache.keys(model_name):
        times, vecs, meta = cache.load(key, model_name)
        n = len(times)
        if n == 0:
            continue
        scores = score_frames(vecs)
        top = np.argsort(scores)[::-1][:top_per_scene]
        rand = rng.choice(n, size=min(random_per_scene, n), replace=False)
        for idx in sorted(set(top.tolist()) | set(rand.tolist())):
            cands.append(
                Candidate(
                    key=key,
                    scene_id=meta.get("scene_id"),
                    path=meta.get("path"),
                    time=float(times[idx]),
                    score=float(scores[idx]),
                )
            )
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:limit] if limit else cands


def train_profile(
    label_store, cache: EmbeddingCache, model_name: str, profile: str, kind: str = "logreg"
):
    """Build the training set and fit a TasteClassifier for `profile`."""
    from .classifier import TasteClassifier

    X, y = build_training_set(label_store, cache, model_name, profile)
    clf = TasteClassifier(kind=kind, model_name=model_name, profile=profile)
    clf.train(X, y)
    return clf, {"samples": int(X.shape[0]), "positives": int((y == 1).sum())}
