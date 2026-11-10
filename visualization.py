"""Premium, data-honest Plotly coaching views for The Impossible Choice.

Observed tracking is always drawn as solid player points and solid trails. Amber
open markers, dashed links, and shaded regions are explicitly modeled geometric
reachability aids and appear only on calculated DDT frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from metrics import FIELD_LENGTH, FIELD_WIDTH, normalize_play_direction


NAVY = "#081421"
PANEL = "#0D2033"
FIELD = "#103D3A"
END_ZONE = "#0A2A31"
NYU_PURPLE = "#8B5CF6"
OFFENSE_COLOR = "#3DA5D9"
RECEIVER_COLOR = "#2DD4BF"
DEFENSE_COLOR = "#FF6B45"
BALL_COLOR = "#F6C85F"
AMBER = "#FFC857"
WHITE = "#F7FAFC"
MUTED = "#9FB3C8"


def _field_shapes(line_of_scrimmage: float | None, first_down_marker: float | None) -> list[dict]:
    """Build a proportional 120 × 53.3-yard NFL field with context markers."""

    shapes: list[dict] = [
        {"type": "rect", "x0": 0, "x1": FIELD_LENGTH, "y0": 0, "y1": FIELD_WIDTH, "line": {"color": "rgba(247,250,252,0.8)", "width": 1}, "fillcolor": FIELD, "layer": "below"},
        {"type": "rect", "x0": 0, "x1": 10, "y0": 0, "y1": FIELD_WIDTH, "line": {"width": 0}, "fillcolor": END_ZONE, "layer": "below"},
        {"type": "rect", "x0": 110, "x1": 120, "y0": 0, "y1": FIELD_WIDTH, "line": {"width": 0}, "fillcolor": END_ZONE, "layer": "below"},
    ]
    for yardline in range(10, 111, 5):
        major = yardline % 10 == 0
        shapes.append({
            "type": "line", "x0": yardline, "x1": yardline, "y0": 0, "y1": FIELD_WIDTH,
            "line": {"color": "rgba(247,250,252,0.34)" if major else "rgba(247,250,252,0.16)", "width": 1},
            "layer": "below",
        })
    if line_of_scrimmage is not None:
        shapes.append({"type": "line", "x0": line_of_scrimmage, "x1": line_of_scrimmage, "y0": 0, "y1": FIELD_WIDTH, "line": {"color": NYU_PURPLE, "width": 3}, "layer": "above"})
    if first_down_marker is not None and 0 <= first_down_marker <= FIELD_LENGTH:
        shapes.append({"type": "line", "x0": first_down_marker, "x1": first_down_marker, "y0": 0, "y1": FIELD_WIDTH, "line": {"color": AMBER, "width": 2, "dash": "dot"}, "layer": "above"})
    return shapes


def _field_annotations(line_of_scrimmage: float | None, first_down_marker: float | None) -> list[dict]:
    annotations: list[dict] = []
    if line_of_scrimmage is not None:
        annotations.append({"x": line_of_scrimmage, "y": FIELD_WIDTH + 1.2, "text": "LOS", "showarrow": False, "font": {"color": NYU_PURPLE, "size": 11, "family": "Arial Black"}})
    if first_down_marker is not None and 0 <= first_down_marker <= FIELD_LENGTH:
        annotations.append({"x": first_down_marker, "y": FIELD_WIDTH + 1.2, "text": "FIRST DOWN", "showarrow": False, "font": {"color": AMBER, "size": 10, "family": "Arial Black"}})
    return annotations


def _empty_trace(name: str, mode: str = "markers") -> go.Scatter:
    return go.Scatter(x=[], y=[], mode=mode, name=name, showlegend=False, hoverinfo="skip")


def _player_trace(
    frame: pd.DataFrame,
    ids: set[int],
    name_map: dict[int, str],
    color: str,
    size: int,
    symbol: str,
    name: str,
    show_text: bool = False,
) -> go.Scatter:
    if not ids:
        return _empty_trace(name, "markers+text")
    players = frame.loc[frame["nflId"].notna()].copy()
    subset = players.loc[players["nflId"].astype(int).isin(ids)].copy()
    if subset.empty:
        return _empty_trace(name, "markers+text")
    labels = [name_map.get(int(value), str(int(value))) for value in subset["nflId"]]
    return go.Scatter(
        x=subset["x_norm"],
        y=subset["y_norm"],
        mode="markers+text" if show_text else "markers",
        text=[label.split(" ")[-1] for label in labels] if show_text else None,
        textposition="top center",
        textfont={"color": WHITE, "size": 9},
        hovertext=[f"{name_map.get(int(value), value)}<br>Jersey #{number}<br>Observed tracking" for value, number in zip(subset["nflId"], subset["jerseyNumber"])],
        hoverinfo="text",
        marker={"size": size, "color": color, "symbol": symbol, "line": {"color": NAVY, "width": 1.5}},
        name=name,
        showlegend=True,
    )


def _ball_trace(frame: pd.DataFrame) -> go.Scatter:
    ball = frame.loc[frame["team"].eq("football")]
    if ball.empty:
        return _empty_trace("Football")
    return go.Scatter(
        x=ball["x_norm"], y=ball["y_norm"], mode="markers", name="Football",
        marker={"size": 9, "color": BALL_COLOR, "line": {"color": "#765322", "width": 1.5}},
        hovertext=["Observed football tracking" for _ in range(len(ball))], hoverinfo="text", showlegend=True,
    )


def _trail_trace(normalized: pd.DataFrame, upto_frame: int, ids: set[int], color: str, name: str) -> go.Scatter:
    """Draw observed historical paths only, with gaps separating players."""

    if not ids:
        return _empty_trace(name, "lines")
    history = normalized.loc[(normalized["frameId"] <= upto_frame) & normalized["nflId"].notna()].copy()
    history = history.loc[history["nflId"].astype(int).isin(ids)].sort_values(["nflId", "frameId"])
    if history.empty:
        return _empty_trace(name, "lines")
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for _, path in history.groupby("nflId", sort=False):
        x_values.extend(path["x_norm"].tolist() + [None])
        y_values.extend(path["y_norm"].tolist() + [None])
    return go.Scatter(
        x=x_values, y=y_values, mode="lines", name=name,
        line={"color": color, "width": 2.5}, opacity=0.85,
        hoverinfo="skip", showlegend=True,
    )


def _positions_for_ids(frame: pd.DataFrame, ids: Iterable[int]) -> dict[int, pd.Series]:
    players = frame.loc[frame["nflId"].notna()].copy()
    players["nflIdInt"] = players["nflId"].astype(int)
    return {int(identifier): row for identifier, row in players.loc[players["nflIdInt"].isin({int(value) for value in ids})].set_index("nflIdInt").iterrows()}


def _modeled_overlay(frame: pd.DataFrame, conflict: dict | None) -> tuple[go.Scatter, go.Scatter, go.Scatter, list[dict]]:
    """Return explicit model-only overlays for a genuine calculated DDT frame."""

    if conflict is None:
        return (
            _empty_trace("Modeled shared-reachability region", "lines"),
            _empty_trace("Modeled 0.4 s receiver projection", "lines+markers"),
            _empty_trace("Modeled conflict links", "lines"),
            [],
        )
    receiver_a = int(conflict["receiverAId"])
    receiver_b = int(conflict["receiverBId"])
    defender = int(conflict["defenderId"])
    positions = _positions_for_ids(frame, [receiver_a, receiver_b, defender])
    if not {receiver_a, receiver_b, defender}.issubset(positions):
        return (
            _empty_trace("Modeled shared-reachability region", "lines"),
            _empty_trace("Modeled 0.4 s receiver projection", "lines+markers"),
            _empty_trace("Modeled conflict links", "lines"),
            [],
        )

    def project(row: pd.Series) -> tuple[float, float]:
        radians = np.deg2rad(float(row["dir_norm"]))
        x = float(np.clip(row["x_norm"] + float(row["s"]) * np.cos(radians) * 0.4, 0, FIELD_LENGTH))
        y = float(np.clip(row["y_norm"] + float(row["s"]) * np.sin(radians) * 0.4, 0, FIELD_WIDTH))
        return x, y

    receiver_a_row, receiver_b_row, defender_row = positions[receiver_a], positions[receiver_b], positions[defender]
    a_projected, b_projected = project(receiver_a_row), project(receiver_b_row)
    defender_point = (float(defender_row["x_norm"]), float(defender_row["y_norm"]))
    region = go.Scatter(
        x=[defender_point[0], a_projected[0], b_projected[0], defender_point[0]],
        y=[defender_point[1], a_projected[1], b_projected[1], defender_point[1]],
        mode="lines", fill="toself", fillcolor="rgba(255, 200, 87, 0.15)",
        line={"color": "rgba(255, 200, 87, 0.55)", "width": 1, "dash": "dot"},
        name="Modeled shared-reachability region", hoverinfo="skip", showlegend=True,
    )
    projections = go.Scatter(
        x=[float(receiver_a_row["x_norm"]), a_projected[0], None, float(receiver_b_row["x_norm"]), b_projected[0]],
        y=[float(receiver_a_row["y_norm"]), a_projected[1], None, float(receiver_b_row["y_norm"]), b_projected[1]],
        mode="lines+markers", line={"color": AMBER, "width": 2, "dash": "dash"},
        marker={"size": [0, 10, 0, 0, 10], "symbol": "circle-open", "color": AMBER, "line": {"color": AMBER, "width": 2}},
        name="Modeled 0.4 s receiver projection", hovertext=["Observed receiver", "Modeled 0.4 s projection", "", "Observed receiver", "Modeled 0.4 s projection"], hoverinfo="text", showlegend=True,
    )
    links = go.Scatter(
        x=[defender_point[0], a_projected[0], None, defender_point[0], b_projected[0]],
        y=[defender_point[1], a_projected[1], None, defender_point[1], b_projected[1]],
        mode="lines", line={"color": AMBER, "width": 1.5, "dash": "dot"},
        name="Modeled conflict links", hoverinfo="skip", showlegend=True,
    )
    annotation = [{
        "x": defender_point[0], "y": defender_point[1] - 4.0,
        "text": "DDT ACTIVE<br><span style='font-size:10px'>modeled reachability</span>",
        "showarrow": False, "align": "center", "bgcolor": "rgba(8,20,33,0.85)",
        "bordercolor": AMBER, "borderwidth": 1, "font": {"color": AMBER, "size": 11, "family": "Arial Black"},
    }]
    return region, projections, links, annotation


def build_play_animation(
    tracking: pd.DataFrame,
    possession_team: str,
    name_map: dict[int, str],
    highlight_receivers: Iterable[int] = (),
    highlight_defender: int | None = None,
    snap_frame: int | None = None,
    throw_frame: int | None = None,
    conflict_frames: pd.DataFrame | None = None,
    line_of_scrimmage: float | None = None,
    first_down_marker: float | None = None,
    title: str = "Actual pre-throw player movement",
) -> go.Figure:
    """Create an animated, coordinate-normalized replay using actual tracking.

    The animated field slider is the authoritative frame scrubber. It is labeled
    in seconds after snap so it aligns with the DDT timeline shown below it.
    """

    normalized = normalize_play_direction(tracking)
    if snap_frame is not None and throw_frame is not None:
        normalized = normalized.loc[(normalized["frameId"] >= snap_frame) & (normalized["frameId"] <= throw_frame)].copy()
    normalized = normalized.sort_values("frameId")
    if normalized.empty:
        return go.Figure()

    selected_receivers = {int(value) for value in highlight_receivers if pd.notna(value)}
    selected_defender = {int(highlight_defender)} if highlight_defender is not None and pd.notna(highlight_defender) else set()
    offense = set(normalized.loc[normalized["team"].eq(possession_team) & normalized["nflId"].notna(), "nflId"].astype(int))
    defense = set(normalized.loc[(normalized["team"] != possession_team) & (normalized["team"] != "football") & normalized["nflId"].notna(), "nflId"].astype(int))
    regular_offense = offense - selected_receivers
    regular_defense = defense - selected_defender
    frame_ids = normalized["frameId"].drop_duplicates().astype(int).tolist()
    conflicts: dict[int, dict] = {}
    if conflict_frames is not None and not conflict_frames.empty:
        conflicts = {int(row.frameId): row._asdict() for row in conflict_frames.sort_values("severity", ascending=False).drop_duplicates("frameId").itertuples(index=False)}

    base_annotations = _field_annotations(line_of_scrimmage, first_down_marker)

    def traces_at(frame_id: int) -> tuple[list[go.Scatter], list[dict]]:
        frame = normalized.loc[normalized["frameId"].eq(frame_id)]
        conflict = conflicts.get(int(frame_id))
        active_pair = {int(conflict["receiverAId"]), int(conflict["receiverBId"])} if conflict else selected_receivers
        active_defender = {int(conflict["defenderId"])} if conflict else selected_defender
        region, projections, links, model_annotations = _modeled_overlay(frame, conflict)
        traces = [
            _player_trace(frame, regular_offense, name_map, OFFENSE_COLOR, 12, "circle", "Offense — observed"),
            _player_trace(frame, regular_defense, name_map, DEFENSE_COLOR, 12, "diamond", "Defense — observed"),
            _trail_trace(normalized, frame_id, active_pair, RECEIVER_COLOR, "Selected receiver paths — observed"),
            _trail_trace(normalized, frame_id, active_defender, DEFENSE_COLOR, "Selected defender path — observed"),
            _player_trace(frame, active_pair, name_map, RECEIVER_COLOR, 18, "circle", "Receiver pair — observed", show_text=True),
            _player_trace(frame, active_defender, name_map, DEFENSE_COLOR, 19, "x", "Potentially stressed defender — observed", show_text=True),
            _ball_trace(frame),
            region,
            projections,
            links,
        ]
        elapsed = (frame_id - int(snap_frame or frame_ids[0])) * 0.1
        state = "DDT ACTIVE" if conflict else "NO DDT FLAG"
        event_label = "BALL SNAP" if frame_id == snap_frame else "PASS RELEASE" if frame_id == throw_frame else state
        frame_annotation = {
            "x": 60, "y": -3.4,
            "text": f"{elapsed:.1f}s after snap  •  {event_label}", "showarrow": False,
            "font": {"color": AMBER if conflict else MUTED, "size": 13, "family": "Arial Black"},
        }
        return traces, base_annotations + model_annotations + [frame_annotation]

    initial_frame = frame_ids[0]
    initial_traces, initial_annotations = traces_at(initial_frame)
    fig = go.Figure(data=initial_traces)
    animation_frames: list[go.Frame] = []
    for frame_id in frame_ids:
        traces, annotations = traces_at(frame_id)
        animation_frames.append(go.Frame(data=traces, name=str(frame_id), layout={"annotations": annotations}))
    fig.frames = animation_frames

    first_name = str(frame_ids[0])
    fig.update_layout(
        title={"text": title, "font": {"color": WHITE, "size": 19, "family": "Arial Black"}},
        template="plotly_dark", height=650, paper_bgcolor=NAVY, plot_bgcolor=NAVY,
        margin={"l": 12, "r": 12, "t": 70, "b": 50},
        xaxis={"range": [-2, 122], "showgrid": False, "zeroline": False, "showticklabels": False, "scaleanchor": "y"},
        yaxis={"range": [-5, 56], "showgrid": False, "zeroline": False, "showticklabels": False},
        shapes=_field_shapes(line_of_scrimmage, first_down_marker), annotations=initial_annotations,
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 10, "color": MUTED}, "bgcolor": "rgba(0,0,0,0)"},
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.01, "y": -0.05,
            "buttons": [
                {"label": "▶ 1×", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "▶ 0.5×", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "▶ 2×", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                {"label": "↺ Restart", "method": "animate", "args": [[first_name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.37, "len": 0.60, "y": -0.055,
            "currentvalue": {"prefix": "Frame: ", "font": {"color": MUTED}},
            "steps": [{
                "label": f"{(frame_id - int(snap_frame or frame_ids[0])) * 0.1:.1f}s", "method": "animate",
                "args": [[str(frame_id)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            } for frame_id in frame_ids],
        }],
    )
    return fig


def build_conflict_timeline(
    frame_records: pd.DataFrame,
    snap_frame: int,
    pass_forward_frame: int,
    frame_seconds: float = 0.1,
) -> go.Figure:
    """Create the full pre-throw timeline with snap, conflict intervals, release."""

    prethrow_duration = (pass_forward_frame - snap_frame) * frame_seconds
    timeline = frame_records.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, prethrow_duration], y=[0.04, 0.04], mode="lines", line={"color": "#4B637A", "width": 4},
        name="Eligible pre-throw window", hoverinfo="skip",
    ))
    if not timeline.empty:
        timeline["secondsAfterSnap"] = (timeline["frameId"] - snap_frame) * frame_seconds
        fig.add_trace(go.Bar(
            x=timeline["secondsAfterSnap"], y=timeline["severity"], width=frame_seconds * 0.9,
            marker={"color": AMBER, "line": {"color": "#FFE7A1", "width": 0.5}},
            hovertext=[
                f"{seconds:.1f}s after snap<br>DDT severity: {severity:.3f}<br>Pair {int(a)} / {int(b)}<br>Defender {int(d)}"
                for seconds, severity, a, b, d in zip(timeline["secondsAfterSnap"], timeline["severity"], timeline["receiverAId"], timeline["receiverBId"], timeline["defenderId"])
            ],
            hoverinfo="text", name="DDT-active frame",
        ))
        conflict_start = float(timeline["secondsAfterSnap"].min())
        conflict_end = float(timeline["secondsAfterSnap"].max() + frame_seconds)
        fig.add_vline(x=conflict_start, line={"color": AMBER, "width": 2, "dash": "dash"})
        fig.add_annotation(x=conflict_start, y=0.96, text="CONFLICT START", showarrow=False, xanchor="left", font={"color": AMBER, "size": 10, "family": "Arial Black"})
        fig.add_vline(x=conflict_end, line={"color": AMBER, "width": 2, "dash": "dash"})
        fig.add_annotation(x=conflict_end, y=0.84, text="LAST DDT END", showarrow=False, xanchor="right", font={"color": AMBER, "size": 10, "family": "Arial Black"})
    else:
        fig.add_annotation(text="No qualifying DDT frames in this play.", x=prethrow_duration / 2, y=0.55, showarrow=False, font={"color": MUTED, "size": 14})
    fig.add_vline(x=0, line={"color": NYU_PURPLE, "width": 3})
    fig.add_annotation(x=0, y=1.07, text="SNAP", showarrow=False, font={"color": NYU_PURPLE, "size": 11, "family": "Arial Black"})
    fig.add_vline(x=prethrow_duration, line={"color": DEFENSE_COLOR, "width": 3})
    fig.add_annotation(x=prethrow_duration, y=1.07, text="PASS RELEASE", showarrow=False, font={"color": DEFENSE_COLOR, "size": 11, "family": "Arial Black"})
    fig.update_layout(
        title={"text": "Defensive Dilemma Timeline — same seconds as replay slider", "font": {"color": WHITE, "size": 16, "family": "Arial Black"}},
        template="plotly_dark", height=285, paper_bgcolor=NAVY, plot_bgcolor=PANEL,
        margin={"l": 48, "r": 25, "t": 60, "b": 48}, barmode="overlay",
        xaxis={"title": "Seconds after snap", "range": [-0.1, prethrow_duration + 0.1], "gridcolor": "rgba(159,179,200,0.15)", "color": MUTED},
        yaxis={"title": "Geometric severity", "range": [0, 1.15], "gridcolor": "rgba(159,179,200,0.15)", "color": MUTED},
        legend={"orientation": "h", "y": -0.28, "x": 0, "font": {"color": MUTED, "size": 10}},
    )
    return fig


def write_featured_demo(
    output_dir: str | Path,
    tracking: pd.DataFrame,
    possession_team: str,
    name_map: dict[int, str],
    featured_record: dict,
    conflict_frames: pd.DataFrame,
    line_of_scrimmage: float | None,
    first_down_marker: float | None,
) -> Path:
    """Write a self-contained Plotly HTML replay based on actual selected data."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay = build_play_animation(
        tracking=tracking,
        possession_team=possession_team,
        name_map=name_map,
        highlight_receivers=[featured_record.get("topReceiverAId"), featured_record.get("topReceiverBId")],
        highlight_defender=featured_record.get("topDefenderId"),
        snap_frame=int(featured_record["snapFrame"]),
        throw_frame=int(featured_record["passForwardFrame"]),
        conflict_frames=conflict_frames,
        line_of_scrimmage=line_of_scrimmage,
        first_down_marker=first_down_marker,
        title=f"THE IMPOSSIBLE CHOICE  •  {int(featured_record['gameId'])} / play {int(featured_record['playId'])}",
    )
    replay_path = output / "featured_play_replay.html"
    replay.write_html(replay_path, include_plotlyjs=True, full_html=True)
    return replay_path
