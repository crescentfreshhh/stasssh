"""Frame sampling via ffmpeg.

The actual decode shells out to ffmpeg/ffprobe at runtime (not bundled in this
repo). The *planning* logic — which timestamps to sample — is a pure function
(`plan_timestamps`) and is unit-tested offline.

Typical use (on a box with ffmpeg + the `[ml]` extra installed):

    sampler = FrameSampler(interval_seconds=2.0)
    for ts, img in sampler.iter_frames("/path/to/video.mp4"):
        ...  # img is a PIL.Image
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


def plan_timestamps(
    duration: float, interval: float, *, offset: float = 0.0
) -> list[float]:
    """Expected sample timestamps for a clip: `offset, offset+interval, ...`.

    Matches how ffmpeg's `fps=1/interval` filter emits frames: the first output
    frame comes from the very start of the clip, then one per interval. Frame i
    therefore represents source time ~`i * interval` — timestamps must NOT be
    shifted, or every downstream marker/label lands offset from its frame.
    """
    if duration <= 0 or interval <= 0:
        return []
    times: list[float] = []
    t = offset
    while t < duration:
        times.append(round(t, 3))
        t += interval
    return times


class SamplerError(RuntimeError):
    pass


class FrameSampler:
    def __init__(
        self,
        interval_seconds: float = 2.0,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        frame_size: int = 288,
    ):
        """`frame_size`: short-side pixel size frames are downscaled to during
        extraction (0 = keep original). Embedders resize to ~224px anyway, so
        decoding small keeps temp-disk and RAM usage ~40x lower on HD sources
        while staying above the model input size."""
        self.interval = interval_seconds
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.frame_size = frame_size

    def probe_duration(self, path: str) -> float:
        """Return duration in seconds via ffprobe."""
        cmd = [
            self.ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffprobe} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise SamplerError(f"ffprobe failed for {path}: {exc.stderr[:300]}") from exc
        try:
            return float(json.loads(out.stdout)["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SamplerError(f"could not parse duration for {path}") from exc

    def grab_frame(self, path: str, time: float) -> "Image":
        """Decode a single frame at `time` seconds (used by the labeler)."""
        from io import BytesIO

        from PIL import Image as PILImage  # lazy

        cmd = [
            self.ffmpeg,
            "-v", "error",
            "-ss", f"{time:g}",
            "-i", path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise SamplerError(
                f"ffmpeg frame grab failed for {path}@{time}s: {exc.stderr[:200]}"
            ) from exc
        if not out.stdout:
            raise SamplerError(f"no frame decoded for {path}@{time}s")
        return PILImage.open(BytesIO(out.stdout)).convert("RGB")

    def _vf(self) -> str:
        """The ffmpeg filtergraph: sample rate + optional downscale."""
        vf = f"fps=1/{self.interval:g}"
        if self.frame_size:
            # short side -> frame_size, other side scales up to keep aspect
            vf += (
                f",scale=w={self.frame_size}:h={self.frame_size}"
                ":force_original_aspect_ratio=increase:force_divisible_by=2"
            )
        return vf

    def iter_frames(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Yield (timestamp_seconds, PIL.Image) sampled every `interval` seconds.

        Extracts with a single ffmpeg pass (`fps=1/interval`) into a temp dir,
        loads frames, and cleans up. The fps filter emits frame i from source
        time ~`i * interval`, so timestamps come from the actual frame index —
        never from a precomputed plan that could drift or truncate silently.
        """
        from PIL import Image as PILImage  # lazy

        duration = self.probe_duration(path)
        if duration <= 0 or self.interval <= 0:
            return

        tmpdir = Path(tempfile.mkdtemp(prefix="peaks-frames-"))
        try:
            cmd = [
                self.ffmpeg,
                "-v", "error",
                "-i", path,
                "-vf", self._vf(),
                "-q:v", "3",
                str(tmpdir / "f-%06d.jpg"),
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as exc:
                raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
            except subprocess.CalledProcessError as exc:
                raise SamplerError(
                    f"ffmpeg failed for {path}: {exc.stderr[:300]}"
                ) from exc

            for i, fp in enumerate(sorted(tmpdir.glob("f-*.jpg"))):
                with PILImage.open(fp) as im:
                    yield round(i * self.interval, 3), im.convert("RGB")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
