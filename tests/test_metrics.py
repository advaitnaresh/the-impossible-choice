from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import MetricConfig, analyze_play, normalize_play_direction, time_to_cover
from visualization import build_conflict_timeline, build_play_animation


def test_left_direction_normalization_rotates_coordinates_and_angles() -> None:
    raw = pd.DataFrame({
        "playDirection": ["left", "right"],
        "x": [20.0, 20.0], "y": [10.0, 10.0], "dir": [45.0, 45.0], "o": [45.0, 45.0],
    })
    normalized = normalize_play_direction(raw)
    assert normalized.loc[0, "x_norm"] == 100.0
    assert normalized.loc[0, "y_norm"] == 43.3
    assert normalized.loc[0, "dir_norm"] == 225.0
    assert normalized.loc[1, "x_norm"] == 20.0
    assert normalized.loc[1, "y_norm"] == 10.0


def test_time_to_cover_increases_with_distance() -> None:
    config = MetricConfig()
    times = time_to_cover(np.array([3.0, 8.0]), np.array([2.0, 2.0]), config)
    assert times[1] > times[0] > 0


def test_analyze_play_flags_designed_unhelped_divergent_pair() -> None:
    rows = []
    for frame_id in range(10, 21):
        event = "ball_snap" if frame_id == 10 else "pass_forward" if frame_id == 20 else "None"
        # Two route runners travel to distinct, diverging projected areas.  The
        # defender is geometrically relevant to both, while his teammate is too far to help.
        rows.extend([
            {"gameId": 1, "playId": 1, "nflId": 101, "frameId": frame_id, "team": "OFF", "playDirection": "right", "x": 46.0, "y": 20.0, "s": 4.0, "a": 0.0, "o": 0.0, "dir": 0.0, "event": event},
            {"gameId": 1, "playId": 1, "nflId": 102, "frameId": frame_id, "team": "OFF", "playDirection": "right", "x": 46.0, "y": 33.0, "s": 4.0, "a": 0.0, "o": 0.0, "dir": 180.0, "event": event},
            {"gameId": 1, "playId": 1, "nflId": 201, "frameId": frame_id, "team": "DEF", "playDirection": "right", "x": 43.6, "y": 26.5, "s": 2.0, "a": 0.0, "o": 0.0, "dir": 0.0, "event": event},
            {"gameId": 1, "playId": 1, "nflId": 202, "frameId": frame_id, "team": "DEF", "playDirection": "right", "x": 15.0, "y": 5.0, "s": 1.0, "a": 0.0, "o": 0.0, "dir": 0.0, "event": event},
        ])
    test_config = MetricConfig(conflict_threshold=0.10)
    result, segments, frames = analyze_play(pd.DataFrame(rows), 1, 1, [101, 102], [201, 202], config=test_config)
    assert result is not None
    assert result["prethrowFrames"] == 10
    assert result["ddtFrames"] > 0
    assert result["ddt"] == result["ddtFrames"] * 0.1
    assert result["topTripleDdt"] == result["topPairFrames"] * 0.1
    assert result["ddtRate"] <= 1
    assert segments
    assert frames


def test_modeled_overlay_is_limited_to_dtt_active_frame() -> None:
    rows = []
    for frame_id in [10, 11, 12]:
        rows.extend([
            {"gameId": 1, "playId": 2, "nflId": 101, "frameId": frame_id, "jerseyNumber": 1, "team": "OFF", "playDirection": "right", "x": 45.0, "y": 20.0, "s": 4.0, "a": 0.0, "o": 0.0, "dir": 0.0, "event": "None"},
            {"gameId": 1, "playId": 2, "nflId": 102, "frameId": frame_id, "jerseyNumber": 2, "team": "OFF", "playDirection": "right", "x": 45.0, "y": 33.0, "s": 4.0, "a": 0.0, "o": 0.0, "dir": 180.0, "event": "None"},
            {"gameId": 1, "playId": 2, "nflId": 201, "frameId": frame_id, "jerseyNumber": 3, "team": "DEF", "playDirection": "right", "x": 43.0, "y": 26.5, "s": 2.0, "a": 0.0, "o": 0.0, "dir": 0.0, "event": "None"},
        ])
    tracking = pd.DataFrame(rows)
    conflict = pd.DataFrame([{
        "gameId": 1, "playId": 2, "frameId": 11, "receiverAId": 101, "receiverBId": 102,
        "defenderId": 201, "severity": 0.5, "helperA": 1.0, "helperB": 1.0,
    }])
    replay = build_play_animation(
        tracking, "OFF", {101: "Receiver A", 102: "Receiver B", 201: "Defender"},
        [101, 102], 201, 10, 12, conflict, 50.0, 60.0,
    )
    assert len(replay.frames) == 3
    # Trace 7 is the modeled shaded reachability region. It exists only at frame 11.
    assert len(replay.frames[0].data[7].x) == 0
    assert len(replay.frames[1].data[7].x) == 4
    assert len(replay.frames[2].data[7].x) == 0
    timeline = build_conflict_timeline(conflict, 10, 12)
    assert len(timeline.data[1].x) == 1
