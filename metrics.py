"""Deterministic geometric coverage-stress metrics for The Impossible Choice.

DDT intentionally measures *potential* coverage stress, not a defender's real
assignment. All geometry is restricted to frames from the snap through, but not
including, the pass-forward frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import acos
from typing import Iterable

import numpy as np
import pandas as pd


FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
PASS_EVENTS = {"pass_forward", "autoevent_passforward"}


@dataclass(frozen=True)
class MetricConfig:
    """Pre-registered, interpretable assumptions for geometric DDT."""

    frame_seconds: float = 0.1
    projection_seconds: float = 0.4
    min_receiver_speed: float = 1.5
    min_pair_distance: float = 3.0
    max_pair_distance: float = 18.0
    min_divergence_degrees: float = 50.0
    defender_cushion_yards: float = 2.0
    defender_acceleration_yps2: float = 4.0
    max_reaction_time: float = 1.35
    minimum_reaction_time: float = 0.15
    help_time_seconds: float = 0.75
    conflict_threshold: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_play_direction(tracking: pd.DataFrame) -> pd.DataFrame:
    """Return positions/angles with the offense moving left-to-right.

    A left-moving play is rotated 180 degrees, so both x and y are reflected and
    movement/facing angles are rotated by 180 degrees. Original fields are kept.
    """

    result = tracking.copy()
    left = result["playDirection"].eq("left")
    result["x_norm"] = np.where(left, FIELD_LENGTH - result["x"], result["x"])
    result["y_norm"] = np.where(left, FIELD_WIDTH - result["y"], result["y"])
    result["dir_norm"] = np.where(left, (result["dir"] + 180.0) % 360.0, result["dir"])
    result["o_norm"] = np.where(left, (result["o"] + 180.0) % 360.0, result["o"])
    return result


def find_window_frames(tracking: pd.DataFrame) -> tuple[int | None, int | None]:
    """Find the first snap and first subsequent forward-pass frame."""

    event = tracking["event"].fillna("")
    snap_frames = tracking.loc[event.isin(SNAP_EVENTS), "frameId"]
    if snap_frames.empty:
        return None, None
    snap_frame = int(snap_frames.min())
    pass_frames = tracking.loc[
        event.isin(PASS_EVENTS) & (tracking["frameId"] > snap_frame), "frameId"
    ]
    if pass_frames.empty:
        return snap_frame, None
    return snap_frame, int(pass_frames.min())


def time_to_cover(
    distances: np.ndarray,
    speeds: np.ndarray,
    config: MetricConfig,
) -> np.ndarray:
    """Estimate time for a defender to close to the coverage cushion.

    The model assumes a capped-free, constant acceleration of 4 yd/s² after the
    observed speed. It is deliberately a transparent reachability approximation,
    not a biomechanics model.
    """

    remaining = np.maximum(np.asarray(distances, dtype=float) - config.defender_cushion_yards, 0.0)
    initial_speed = np.maximum(np.asarray(speeds, dtype=float), 0.5)
    acceleration = config.defender_acceleration_yps2
    return (-initial_speed + np.sqrt(initial_speed**2 + 2 * acceleration * remaining)) / acceleration


def _velocity_components(speed: np.ndarray, direction_degrees: np.ndarray) -> np.ndarray:
    """Convert NFL x-axis movement direction to x/y velocity components."""

    radians = np.deg2rad(np.asarray(direction_degrees, dtype=float))
    speed = np.asarray(speed, dtype=float)
    return np.column_stack((speed * np.cos(radians), speed * np.sin(radians)))


def _id_indices(ids: np.ndarray, requested_ids: Iterable[int]) -> list[int]:
    index = {int(nfl_id): idx for idx, nfl_id in enumerate(ids)}
    return [index[int(nfl_id)] for nfl_id in requested_ids if int(nfl_id) in index]


def _pair_key(receiver_a: int, receiver_b: int) -> tuple[int, int]:
    return tuple(sorted((int(receiver_a), int(receiver_b))))


def _summarize_segment(segment: dict) -> dict:
    """Convert internal segment state to a compact serializable record."""

    frames = segment["end_frame"] - segment["start_frame"] + 1
    return {
        "gameId": segment["gameId"],
        "playId": segment["playId"],
        "receiverAId": segment["receiverAId"],
        "receiverBId": segment["receiverBId"],
        "defenderId": segment["defenderId"],
        "startFrame": segment["start_frame"],
        "endFrame": segment["end_frame"],
        "startSecondsAfterSnap": round((segment["start_frame"] - segment["snap_frame"]) * segment["frame_seconds"], 2),
        "endSecondsAfterSnap": round((segment["end_frame"] - segment["snap_frame"] + 1) * segment["frame_seconds"], 2),
        "durationSeconds": round(frames * segment["frame_seconds"], 2),
        "meanSeverity": round(segment["severity_sum"] / frames, 4),
        "meanDivergenceDegrees": round(segment["angle_sum"] / frames, 2),
        "meanDefenderTTIA": round(segment["tti_a_sum"] / frames, 3),
        "meanDefenderTTIB": round(segment["tti_b_sum"] / frames, 3),
        "minHelperTTIA": round(segment["helper_a_min"], 3),
        "minHelperTTIB": round(segment["helper_b_min"], 3),
    }


def analyze_play(
    tracking: pd.DataFrame,
    game_id: int,
    play_id: int,
    route_ids: Iterable[int],
    coverage_ids: Iterable[int],
    target_nfl_id: int | None = None,
    config: MetricConfig = MetricConfig(),
) -> tuple[dict | None, list[dict], list[dict]]:
    """Calculate one play's DDT, baseline geometry, and conflict segments.

    Returns `(play_record, segments, frame_records)`. `frame_records` contains
    the winning triple per qualifying frame and drives the replay timeline. No
    information after the pass-forward frame participates in the calculation.
    """

    snap_frame, throw_frame = find_window_frames(tracking)
    if snap_frame is None or throw_frame is None or throw_frame <= snap_frame:
        return None, [], []

    route_ids = sorted({int(value) for value in route_ids})
    coverage_ids = sorted({int(value) for value in coverage_ids})
    if len(route_ids) < 2 or not coverage_ids:
        return None, [], []

    play = normalize_play_direction(tracking)
    play = play.loc[(play["frameId"] >= snap_frame) & (play["frameId"] < throw_frame)].copy()
    play = play.loc[play["nflId"].notna()].copy()
    if play.empty:
        return None, [], []

    frame_records: list[dict] = []
    qualifying_frames = 0
    severity_values: list[float] = []
    all_min_separations: list[float] = []
    all_route_speeds: list[float] = []
    nearby_defender_counts: list[float] = []
    local_density_counts: list[float] = []
    target_separations: list[float] = []
    target_speeds: list[float] = []
    segments: list[dict] = []
    active_segment: dict | None = None
    pair_frame_counts: dict[tuple[int, int], int] = {}
    pair_severity_sums: dict[tuple[int, int], float] = {}
    triple_frame_counts: dict[tuple[int, int, int], int] = {}
    defender_frame_counts: dict[int, int] = {}

    for frame_id, frame in play.groupby("frameId", sort=True):
        ids = frame["nflId"].astype(int).to_numpy()
        route_idx = _id_indices(ids, route_ids)
        defender_idx = _id_indices(ids, coverage_ids)
        if len(route_idx) < 2 or not defender_idx:
            continue

        positions = frame[["x_norm", "y_norm"]].to_numpy(dtype=float)
        speeds = frame["s"].fillna(0.0).to_numpy(dtype=float)
        directions = frame["dir_norm"].fillna(0.0).to_numpy(dtype=float)
        route_positions = positions[route_idx]
        defender_positions = positions[defender_idx]
        route_speeds = speeds[route_idx]
        defender_speeds = speeds[defender_idx]
        route_identifiers = ids[route_idx]
        defender_identifiers = ids[defender_idx]

        distance_matrix = np.linalg.norm(
            route_positions[:, None, :] - defender_positions[None, :, :], axis=2
        )
        route_min_separation = distance_matrix.min(axis=1)
        all_min_separations.append(float(route_min_separation.mean()))
        all_route_speeds.append(float(route_speeds.mean()))
        nearby_defender_counts.append(float((distance_matrix <= 5.0).sum(axis=1).mean()))
        route_pair_distances = np.linalg.norm(
            route_positions[:, None, :] - route_positions[None, :, :], axis=2
        )
        local_density_counts.append(float(((route_pair_distances <= 7.0).sum(axis=1) - 1).mean()))

        if target_nfl_id is not None and int(target_nfl_id) in route_identifiers:
            target_index = int(np.where(route_identifiers == int(target_nfl_id))[0][0])
            target_separations.append(float(route_min_separation[target_index]))
            target_speeds.append(float(route_speeds[target_index]))

        velocities = _velocity_components(route_speeds, directions[route_idx])
        projected_routes = route_positions + velocities * config.projection_seconds
        projected_routes[:, 0] = np.clip(projected_routes[:, 0], 0.0, FIELD_LENGTH)
        projected_routes[:, 1] = np.clip(projected_routes[:, 1], 0.0, FIELD_WIDTH)

        best: dict | None = None
        for left_idx, right_idx in combinations(range(len(route_idx)), 2):
            if (
                route_speeds[left_idx] < config.min_receiver_speed
                or route_speeds[right_idx] < config.min_receiver_speed
            ):
                continue
            receiver_a = int(route_identifiers[left_idx])
            receiver_b = int(route_identifiers[right_idx])
            target_a = projected_routes[left_idx]
            target_b = projected_routes[right_idx]
            pair_distance = float(np.linalg.norm(target_a - target_b))
            if not (config.min_pair_distance <= pair_distance <= config.max_pair_distance):
                continue

            vectors_a = target_a - defender_positions
            vectors_b = target_b - defender_positions
            distances_a = np.linalg.norm(vectors_a, axis=1)
            distances_b = np.linalg.norm(vectors_b, axis=1)
            tti_a = time_to_cover(distances_a, defender_speeds, config)
            tti_b = time_to_cover(distances_b, defender_speeds, config)

            for defender_local_idx, defender_id in enumerate(defender_identifiers):
                t_a = float(tti_a[defender_local_idx])
                t_b = float(tti_b[defender_local_idx])
                if not (
                    config.minimum_reaction_time < t_a <= config.max_reaction_time
                    and config.minimum_reaction_time < t_b <= config.max_reaction_time
                ):
                    continue

                vector_a = vectors_a[defender_local_idx]
                vector_b = vectors_b[defender_local_idx]
                denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
                if denominator == 0.0:
                    continue
                cosine = float(np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0))
                angle = float(np.degrees(acos(cosine)))
                if angle < config.min_divergence_degrees:
                    continue

                other_indices = [idx for idx in range(len(defender_idx)) if idx != defender_local_idx]
                if other_indices:
                    helper_a = float(np.min(tti_a[other_indices]))
                    helper_b = float(np.min(tti_b[other_indices]))
                else:
                    helper_a = float("inf")
                    helper_b = float("inf")

                # DDT rises with divergent demands and combined reach time, while
                # nearby alternate coverage help suppresses the score.
                divergence = (angle - config.min_divergence_degrees) / (180.0 - config.min_divergence_degrees)
                time_demand = np.clip((t_a + t_b - 0.55) / 1.45, 0.0, 1.0)
                help_a_factor = np.clip(helper_a / config.help_time_seconds, 0.0, 1.0)
                help_b_factor = np.clip(helper_b / config.help_time_seconds, 0.0, 1.0)
                help_factor = float(np.sqrt(help_a_factor * help_b_factor))
                severity = float(divergence * time_demand * help_factor)
                if severity < config.conflict_threshold:
                    continue

                candidate = {
                    "frameId": int(frame_id),
                    "receiverAId": receiver_a,
                    "receiverBId": receiver_b,
                    "defenderId": int(defender_id),
                    "severity": severity,
                    "angle": angle,
                    "ttiA": t_a,
                    "ttiB": t_b,
                    "helperA": helper_a,
                    "helperB": helper_b,
                    "pairDistance": pair_distance,
                }
                if best is None or candidate["severity"] > best["severity"]:
                    best = candidate

        if best is None:
            if active_segment is not None:
                segments.append(_summarize_segment(active_segment))
                active_segment = None
            continue

        qualifying_frames += 1
        severity_values.append(best["severity"])
        best["secondsAfterSnap"] = round((int(frame_id) - snap_frame) * config.frame_seconds, 2)
        frame_records.append({"gameId": game_id, "playId": play_id, **best})
        pair = _pair_key(best["receiverAId"], best["receiverBId"])
        pair_frame_counts[pair] = pair_frame_counts.get(pair, 0) + 1
        pair_severity_sums[pair] = pair_severity_sums.get(pair, 0.0) + best["severity"]
        triple = (pair[0], pair[1], int(best["defenderId"]))
        triple_frame_counts[triple] = triple_frame_counts.get(triple, 0) + 1
        defender_frame_counts[best["defenderId"]] = defender_frame_counts.get(best["defenderId"], 0) + 1

        same_segment = (
            active_segment is not None
            and active_segment["receiverAId"] == best["receiverAId"]
            and active_segment["receiverBId"] == best["receiverBId"]
            and active_segment["defenderId"] == best["defenderId"]
            and active_segment["end_frame"] == int(frame_id) - 1
        )
        if not same_segment:
            if active_segment is not None:
                segments.append(_summarize_segment(active_segment))
            active_segment = {
                "gameId": game_id,
                "playId": play_id,
                "receiverAId": best["receiverAId"],
                "receiverBId": best["receiverBId"],
                "defenderId": best["defenderId"],
                "start_frame": int(frame_id),
                "end_frame": int(frame_id),
                "snap_frame": snap_frame,
                "frame_seconds": config.frame_seconds,
                "severity_sum": best["severity"],
                "angle_sum": best["angle"],
                "tti_a_sum": best["ttiA"],
                "tti_b_sum": best["ttiB"],
                "helper_a_min": best["helperA"],
                "helper_b_min": best["helperB"],
            }
        else:
            active_segment["end_frame"] = int(frame_id)
            active_segment["severity_sum"] += best["severity"]
            active_segment["angle_sum"] += best["angle"]
            active_segment["tti_a_sum"] += best["ttiA"]
            active_segment["tti_b_sum"] += best["ttiB"]
            active_segment["helper_a_min"] = min(active_segment["helper_a_min"], best["helperA"])
            active_segment["helper_b_min"] = min(active_segment["helper_b_min"], best["helperB"])

    if active_segment is not None:
        segments.append(_summarize_segment(active_segment))

    prethrow_frames = throw_frame - snap_frame
    if prethrow_frames <= 0:
        return None, [], []
    top_triple = max(triple_frame_counts, key=triple_frame_counts.get) if triple_frame_counts else (None, None, None)
    top_pair = (top_triple[0], top_triple[1]) if triple_frame_counts else (None, None)
    top_defender = top_triple[2] if triple_frame_counts else None
    play_record = {
        "gameId": game_id,
        "playId": play_id,
        "snapFrame": snap_frame,
        "passForwardFrame": throw_frame,
        "prethrowFrames": prethrow_frames,
        "prethrowDurationSeconds": round(prethrow_frames * config.frame_seconds, 2),
        "routeRunnerCount": len(route_ids),
        "coverageDefenderCount": len(coverage_ids),
        "ddtFrames": qualifying_frames,
        "ddt": round(qualifying_frames * config.frame_seconds, 2),
        "ddtRate": round(qualifying_frames / prethrow_frames, 4),
        "dss": round(float(np.sum(severity_values) / prethrow_frames), 4) if severity_values else 0.0,
        "meanConflictSeverity": round(float(np.mean(severity_values)), 4) if severity_values else 0.0,
        "segmentCount": len(segments),
        "topReceiverAId": top_pair[0],
        "topReceiverBId": top_pair[1],
        "topDefenderId": top_defender,
        "topPairFrames": triple_frame_counts.get(top_triple, 0) if triple_frame_counts else 0,
        "topTripleDdt": round(triple_frame_counts.get(top_triple, 0) * config.frame_seconds, 2) if triple_frame_counts else 0.0,
        "meanRouteNearestDefenderSeparation": round(float(np.mean(all_min_separations)), 3) if all_min_separations else np.nan,
        "meanRouteSpeed": round(float(np.mean(all_route_speeds)), 3) if all_route_speeds else np.nan,
        "meanNearbyDefenders": round(float(np.mean(nearby_defender_counts)), 3) if nearby_defender_counts else np.nan,
        "meanLocalReceiverDensity": round(float(np.mean(local_density_counts)), 3) if local_density_counts else np.nan,
        "targetSeparation": round(float(target_separations[-1]), 3) if target_separations else np.nan,
        "targetSpeed": round(float(target_speeds[-1]), 3) if target_speeds else np.nan,
    }
    return play_record, segments, frame_records
