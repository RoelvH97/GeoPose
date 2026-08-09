"""Published DSA channel conventions and signed view-role labels."""

from __future__ import annotations


VIEW_CHANNELS = {
    "pre": {"lat": "a", "pa": "b"},
    "post": {"lat": "c", "pa": "d"},
}
CHANNEL_PAIR = {"a": "c", "c": "a", "b": "d", "d": "b"}
MAP_META_CHANNEL_OVERRIDES = {
    ("sub-stroke0016", "a"): "b",
    ("sub-stroke0016", "b"): "a",
    ("sub-stroke0020", "a"): "c",
    ("sub-stroke0020", "d"): "b",
}
DSA_PATH_CHANNEL_OVERRIDES = {
    ("sub-stroke0016", "pre"): {
        "dsa": {"lat": "b", "pa": "a"},
        "mask": {"lat": "a", "pa": "b"},
    },
    ("sub-stroke0020", "pre"): {"meta": {"lat": "c", "pa": "b"}},
    ("sub-stroke0020", "post"): {"meta": {"lat": "c", "pa": "b"}},
}
VIEW_THRESHOLD_DEGREES = 45.0


def view_label_from_alpha(alpha_degrees: float) -> int:
    signed_rotation = -float(alpha_degrees)
    if signed_rotation > VIEW_THRESHOLD_DEGREES:
        return 2
    if signed_rotation < -VIEW_THRESHOLD_DEGREES:
        return 0
    return 1
