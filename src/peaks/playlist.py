"""Build the megaboard playlist from Stash markers.

Queries Stash for every marker tagged with your apex tag and emits a JSON file
the static megaboard webapp loads. Stream URLs are baked in here (Python is the
only place that handles the API key), so the browser never needs to talk to the
Stash GraphQL API — it just plays `<video>` tiles.

The output (`playlist.json`) contains your API key inside the stream URLs and is
gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CLIP_SECONDS = 20.0  # used when a marker has no end_seconds (point marker)


def build_playlist(
    client,
    tag_name: str,
    *,
    default_clip_seconds: float = DEFAULT_CLIP_SECONDS,
    limit: int | None = None,
) -> dict:
    """Return a playlist dict: {tag, count, apexes:[{scene_id,start,end,
    duration,title,url}]}."""
    apexes = []
    for m in client.iter_markers_by_tag(tag_name):
        if not m["scene_id"]:
            continue
        start = m["seconds"]
        end = m["end_seconds"]
        if end is None or end <= start:
            end = start + default_clip_seconds
        apexes.append(
            {
                "scene_id": m["scene_id"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "title": m["title"] or tag_name,
                "url": client.stream_url(m["scene_id"], start=start),
            }
        )
        if limit and len(apexes) >= limit:
            break
    return {"tag": tag_name, "count": len(apexes), "apexes": apexes}


def write_playlist(playlist: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(playlist, indent=2))
    return out
