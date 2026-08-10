"""Published DSA channel conventions and signed view-role labels."""

from __future__ import annotations


# DSA acquisitions are stored one channel per series: 'a'/'b' are the lateral
# and PA runs of the pre-treatment study, 'c'/'d' the same for post-treatment.
VIEW_CHANNELS = {
    "pre": {"lat": "a", "pa": "b"},
    "post": {"lat": "c", "pa": "d"},
}
# The same view across the two timestamps, used to fall back to the other
# study's acquisition metadata when a channel's own JSON is absent.
CHANNEL_PAIR = {"a": "c", "c": "a", "b": "d", "d": "b"}

# Acquisition errata for two subjects, applied so the published pipeline reads
# the same files the reported results were produced from. These are properties
# of the source archive, not of the method; both were confirmed against the
# stored C-arm angles rather than inferred.
#
#   sub-stroke0016 (pre): the lateral and PA series were exported under swapped
#       channel letters. The DSA volumes need the swap; the MAP support masks
#       were generated after the fact and are already in the nominal order.
#   sub-stroke0020: channels 'a' and 'd' carry no usable metadata JSON, so the
#       geometry is read from the paired channel of the other timestamp.
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

# |alpha| beyond this is a lateral acquisition; inside it, a PA acquisition.
VIEW_THRESHOLD_DEGREES = 45.0


def view_label_from_alpha(alpha_degrees: float) -> int:
    signed_rotation = -float(alpha_degrees)
    if signed_rotation > VIEW_THRESHOLD_DEGREES:
        return 2
    if signed_rotation < -VIEW_THRESHOLD_DEGREES:
        return 0
    return 1
