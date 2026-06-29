import json

from peaks.playlist import build_playlist, write_playlist


class _MarkerClient:
    """Minimal client stub: serves markers and builds stream URLs."""

    def __init__(self, markers):
        self._markers = markers

    def iter_markers_by_tag(self, tag_name):
        yield from self._markers

    def stream_url(self, scene_id, start=None):
        return f"http://stash.test/scene/{scene_id}/stream?start={start:g}&apikey=k"


def _marker(scene_id, seconds, end_seconds, title="apex"):
    return {
        "marker_id": f"m{scene_id}",
        "scene_id": scene_id,
        "seconds": seconds,
        "end_seconds": end_seconds,
        "title": title,
    }


def test_build_playlist_basic():
    client = _MarkerClient([_marker("1", 10.0, 25.0), _marker("2", 5.0, 12.0)])
    pl = build_playlist(client, "apex")
    assert pl["tag"] == "apex" and pl["count"] == 2
    a = pl["apexes"][0]
    assert a["scene_id"] == "1"
    assert a["start"] == 10.0 and a["end"] == 25.0 and a["duration"] == 15.0
    assert "start=10" in a["url"] and "apikey=k" in a["url"]


def test_point_marker_gets_default_clip_length():
    client = _MarkerClient([_marker("1", 30.0, None)])
    pl = build_playlist(client, "apex", default_clip_seconds=20.0)
    a = pl["apexes"][0]
    assert a["start"] == 30.0 and a["end"] == 50.0 and a["duration"] == 20.0


def test_zero_length_marker_uses_default():
    client = _MarkerClient([_marker("1", 30.0, 30.0)])  # end <= start
    a = build_playlist(client, "apex", default_clip_seconds=10.0)["apexes"][0]
    assert a["duration"] == 10.0


def test_marker_without_scene_skipped():
    bad = _marker("x", 1.0, 2.0)
    bad["scene_id"] = None
    client = _MarkerClient([bad, _marker("2", 1.0, 2.0)])
    pl = build_playlist(client, "apex")
    assert pl["count"] == 1 and pl["apexes"][0]["scene_id"] == "2"


def test_limit_caps_apexes():
    client = _MarkerClient([_marker(str(i), 1.0, 2.0) for i in range(10)])
    assert build_playlist(client, "apex", limit=3)["count"] == 3


def test_write_playlist_roundtrip(tmp_path):
    client = _MarkerClient([_marker("1", 10.0, 25.0)])
    pl = build_playlist(client, "apex")
    out = write_playlist(pl, tmp_path / "sub" / "playlist.json")
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["count"] == 1 and loaded["apexes"][0]["scene_id"] == "1"
