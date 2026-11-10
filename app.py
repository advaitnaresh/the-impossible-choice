"""THE IMPOSSIBLE CHOICE — Defensive Dilemma Intelligence.

Run after analysis.py has produced outputs:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import load_static_data, load_tracking_game, resolve_data_dir
from visualization import build_conflict_timeline, build_play_animation


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
NAVY = "#081421"
PANEL = "#0D2033"
NYU_PURPLE = "#8B5CF6"
TEAL = "#2DD4BF"
AMBER = "#FFC857"
ORANGE = "#FF6B45"

st.set_page_config(page_title="The Impossible Choice", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[int, str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(OUTPUT_DIR / "ddt_play_results.csv")
    segments_path = OUTPUT_DIR / "ddt_conflict_segments.csv"
    segments = pd.read_csv(segments_path) if segments_path.exists() else pd.DataFrame()
    featured = json.loads((OUTPUT_DIR / "featured_plays.json").read_text(encoding="utf-8"))
    static = load_static_data(resolve_data_dir())
    names = dict(zip(static["players"]["nflId"].astype(int), static["players"]["displayName"]))
    baseline_path = OUTPUT_DIR / "baseline_comparison.csv"
    pair_path = OUTPUT_DIR / "receiver_pair_rankings.csv"
    team_path = OUTPUT_DIR / "team_stress_summary.csv"
    baselines = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    pairs = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    teams = pd.read_csv(team_path) if team_path.exists() else pd.DataFrame()
    return results, segments, featured, names, baselines, pairs, teams


@st.cache_data(show_spinner="Loading actual tracking positions for one game…")
def load_play_tracking(game_id: int, play_id: int) -> pd.DataFrame:
    tracking = load_tracking_game(resolve_data_dir(), game_id)
    return tracking.loc[tracking["playId"].eq(play_id)].copy()


@st.cache_data(show_spinner=False)
def load_conflict_frames() -> pd.DataFrame:
    path = OUTPUT_DIR / "ddt_conflict_frames.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def player_name(nfl_id: object, names: dict[int, str]) -> str:
    if pd.isna(nfl_id):
        return "No qualifying player"
    return names.get(int(nfl_id), f"NFL ID {int(nfl_id)}")


def selected_record(results: pd.DataFrame, game_id: int, play_id: int) -> pd.Series:
    selection = results.loc[(results["gameId"] == game_id) & (results["playId"] == play_id)]
    if selection.empty:
        raise ValueError("The requested play was not in the processed output.")
    return selection.iloc[0]


def field_markers(record: pd.Series, tracking: pd.DataFrame) -> tuple[float | None, float | None]:
    if tracking.empty or pd.isna(record.absoluteYardlineNumber):
        return None, None
    direction = str(tracking["playDirection"].dropna().iloc[0])
    line_of_scrimmage = 120.0 - float(record.absoluteYardlineNumber) if direction == "left" else float(record.absoluteYardlineNumber)
    first_down = line_of_scrimmage + float(record.yardsToGo) if pd.notna(record.yardsToGo) else None
    return line_of_scrimmage, first_down


def clean_choices(results: pd.DataFrame) -> pd.DataFrame:
    choices = results.copy()
    clean_columns = {"play_type", "pass_attempt", "penalty"}
    if clean_columns.issubset(choices.columns):
        clean = choices.loc[
            choices["play_type"].eq("pass")
            & choices["pass_attempt"].eq(1)
            & choices["penalty"].eq(0)
        ]
        if not clean.empty:
            choices = clean
    choices = choices.sort_values(["ddtRate", "dss"], ascending=False)[["gameId", "playId", "ddtRate", "pff_passCoverageType"]].copy()
    choices["label"] = choices.apply(
        lambda row: f"{int(row.gameId)} · Play {int(row.playId)} · DDT Rate {row.ddtRate:.0%} · {row.pff_passCoverageType}", axis=1
    )
    return choices


def render_replay(
    record: pd.Series,
    tracking: pd.DataFrame,
    frames: pd.DataFrame,
    names: dict[int, str],
    title: str,
    height: int | None = None,
) -> None:
    pair = [record["topReceiverAId"], record["topReceiverBId"]]
    line_of_scrimmage, first_down = field_markers(record, tracking)
    figure = build_play_animation(
        tracking=tracking,
        possession_team=str(record.possessionTeam),
        name_map=names,
        highlight_receivers=pair,
        highlight_defender=int(record.topDefenderId) if pd.notna(record.topDefenderId) else None,
        snap_frame=int(record.snapFrame),
        throw_frame=int(record.passForwardFrame),
        conflict_frames=frames,
        line_of_scrimmage=line_of_scrimmage,
        first_down_marker=first_down,
        title=title,
    )
    if height is not None:
        figure.update_layout(height=height)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": False})


def metric_support_label(play_frames: pd.DataFrame) -> tuple[str, str]:
    if play_frames.empty or not {"helperA", "helperB"}.issubset(play_frames.columns):
        return "—", "No DDT frame, so there is no selected modeled-support value."
    support = min(float(play_frames["helperA"].min()), float(play_frames["helperB"].min()))
    return f"{support:.2f} s", "Shortest modeled time for another coverage defender to reach either selected receiver across DDT-active frames."


def main() -> None:
    st.markdown(
        f"""<style>
        .stApp {{ background: {NAVY}; color: #F7FAFC; }}
        [data-testid='stSidebar'] {{ background: #0B1B2C; border-right: 1px solid rgba(139,92,246,0.45); }}
        [data-testid='stMetric'] {{ background: {PANEL}; border: 1px solid rgba(139,92,246,0.45); border-radius: 10px; padding: 12px; }}
        [data-testid='stMetricLabel'] {{ color: #9FB3C8; }}
        [data-testid='stMetricValue'] {{ color: #F7FAFC; }}
        .eyebrow {{ color: {TEAL}; font-weight: 700; letter-spacing: 0.14em; font-size: 0.76rem; }}
        .legend-note {{ color: #BFD1E5; border-left: 3px solid {AMBER}; padding-left: 12px; }}
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown("<p class='eyebrow'>2021 NFL TRACKING • PRE-THROW GEOMETRIC COVERAGE STRESS</p>", unsafe_allow_html=True)
    st.title("🏈 THE IMPOSSIBLE CHOICE")
    st.caption("Defensive Dilemma Intelligence — real player tracking, actual DDT calculations, and explicit uncertainty about defensive assignments.")

    required = [OUTPUT_DIR / "ddt_play_results.csv", OUTPUT_DIR / "featured_plays.json", OUTPUT_DIR / "ddt_conflict_frames.csv"]
    if not all(path.exists() for path in required):
        st.error("Processed results are missing. Run `python analysis.py --download-pbp` first.")
        st.stop()

    results, segments, featured, names, baselines, pairs, teams = load_app_data()
    all_frames = load_conflict_frames()
    finding_path = OUTPUT_DIR / "finding.md"
    if finding_path.exists():
        with st.expander("The evidence in one minute", expanded=True):
            st.markdown(finding_path.read_text(encoding="utf-8"))

    st.sidebar.header("Replay controls")
    default = featured["highDdt"]
    choices = clean_choices(results)
    mode = st.sidebar.radio(
        "Judge showcase scene",
        ["1. High observed DDT", "2. Matched low-DDT control", "3. Dense zero-DDT counterexample", "Browse clean pass attempts"],
    )
    selected = default
    if mode == "1. High observed DDT":
        selected = featured["highDdt"]
    elif mode == "2. Matched low-DDT control":
        selected = featured["matchedLowDdt"]
    elif mode == "3. Dense zero-DDT counterexample":
        selected = featured["counterexample"]
    else:
        default_index = int(choices.index[choices["gameId"].eq(int(default["gameId"])) & choices["playId"].eq(int(default["playId"]))][0])
        selected_index = st.sidebar.selectbox("Processed play", choices.index.tolist(), index=choices.index.tolist().index(default_index), format_func=lambda idx: choices.loc[idx, "label"])
        selected = {"gameId": int(choices.loc[selected_index, "gameId"]), "playId": int(choices.loc[selected_index, "playId"])}

    record = selected_record(results, int(selected["gameId"]), int(selected["playId"]))
    tracking = load_play_tracking(int(record.gameId), int(record.playId))
    play_frames = all_frames.loc[(all_frames["gameId"] == int(record.gameId)) & (all_frames["playId"] == int(record.playId))].copy()
    play_segments = segments.loc[(segments["gameId"] == int(record.gameId)) & (segments["playId"] == int(record.playId))].copy() if not segments.empty else pd.DataFrame()

    if record.ddtFrames > 0:
        pair_names = " + ".join([player_name(record.topReceiverAId, names), player_name(record.topReceiverBId, names)])
        defender_name = player_name(record.topDefenderId, names)
        st.subheader(f"{pair_names}  ↔  {defender_name}")
    else:
        pair_names, defender_name = "No qualifying DDT pair", "No selected defender"
        st.subheader("Dense geometric control — no qualifying DDT triple")
    st.write(record.playDescription)
    st.caption(f"Week {int(record.week)} · {record.possessionTeam} offense vs. {record.defensiveTeam} · {record.pff_passCoverage} ({record.pff_passCoverageType}) · {record.offenseFormation}")

    support_value, support_help = metric_support_label(play_frames)
    first_row = st.columns(4)
    first_row[0].metric("DDT · whole play", f"{record.ddt:.1f} s", help="Union of all DDT-active pre-throw frames across qualifying triples × 0.1 seconds.")
    first_row[1].metric("DDT Rate · whole play", f"{record.ddtRate:.0%}", help="Whole-play DDT divided by the eligible snap-to-release duration.")
    first_row[2].metric("Top triple DDT", f"{record.topTripleDdt:.1f} s", help="Duration occupied by the displayed receiver A / receiver B / defender triple. It can be lower than whole-play DDT because other triples may qualify.")
    first_row[3].metric("DSS · whole play", f"{record.dss:.3f}", help="Pre-throw duration-weighted geometric severity, documented in METRIC.md.")
    second_row = st.columns(3)
    target_sep = f"{record.targetSeparation:.1f} yd" if pd.notna(record.targetSeparation) else "Not linked"
    second_row[0].metric("Target separation", target_sep, help="Observed final pre-throw distance from the nflverse-identified target to the nearest PFF coverage defender; not used to calculate DDT.")
    second_row[1].metric("Closest alternate help", support_value, help=support_help)
    second_row[2].metric("Snap → release", f"{record.prethrowDurationSeconds:.1f} s", help="Eligible pre-throw window in actual tracking frames.")

    st.markdown("#### Scene 1–5 · Watch the actual play develop")
    st.markdown("<p class='legend-note'><b>Solid paths and player markers are observed tracking.</b> Amber open markers, dashed links, and the shaded triangle are a modeled 0.4-second reachability aid and appear only when a calculated DDT frame is active. The replay slider and the timeline both use seconds since snap.</p>", unsafe_allow_html=True)
    render_replay(record, tracking, play_frames, names, "Actual tracking replay — conflict overlay only on DDT frames")

    timeline = build_conflict_timeline(play_frames, int(record.snapFrame), int(record.passForwardFrame))
    st.plotly_chart(timeline, width="stretch", config={"displaylogo": False, "scrollZoom": False})
    if not play_segments.empty:
        st.caption("Contiguous potential-stress windows. Player names identify the highest-severity geometric triple, not verified coverage assignments.")
        st.dataframe(
            play_segments[["receiverAName", "receiverBName", "defenderName", "startSecondsAfterSnap", "endSecondsAfterSnap", "durationSeconds", "meanSeverity"]],
            width="stretch", hide_index=True,
        )

    st.markdown("### Coach's read")
    if record.ddtFrames > 0:
        st.write(
            f"The displayed pair-defender triple was DDT-active for **{record.topTripleDdt:.1f} seconds** across {int(record.topPairFrames)} frames, with limited alternate geometric help ({support_value}). "
            f"Whole-play DDT is **{record.ddt:.1f} seconds** because other qualifying triples can contribute to the union of DDT-active frames. "
            "This is a route-combination scouting prompt: review whether the defensive call actually assigned this defender to both threats or supplied an expected handoff."
        )
    else:
        st.write(
            "This is an important counterexample: receivers can be spatially dense or closely covered without satisfying the model's divergent, shared-reachability, and support conditions. "
            "DDT is not a generic crowding score."
        )

    st.markdown("### Scene 6 · Matched comparison")
    st.caption("High DDT and matched zero-DDT plays are both real clean pass attempts. Their animated sliders remain independently controllable, so use the identical seconds-since-snap labels to inspect corresponding phases.")
    compare = st.toggle("Show high-DDT vs. matched zero-DDT replay", value=False)
    if compare:
        high = selected_record(results, int(featured["highDdt"]["gameId"]), int(featured["highDdt"]["playId"]))
        low = selected_record(results, int(featured["matchedLowDdt"]["gameId"]), int(featured["matchedLowDdt"]["playId"]))
        high_tracking = load_play_tracking(int(high.gameId), int(high.playId))
        low_tracking = load_play_tracking(int(low.gameId), int(low.playId))
        high_frames = all_frames.loc[(all_frames["gameId"] == int(high.gameId)) & (all_frames["playId"] == int(high.playId))]
        low_frames = all_frames.loc[(all_frames["gameId"] == int(low.gameId)) & (all_frames["playId"] == int(low.playId))]
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Play A · High DDT: {high.ddt:.1f}s ({high.ddtRate:.0%})**")
            render_replay(high, high_tracking, high_frames, names, "Play A — high DDT", height=500)
        with right:
            st.markdown(f"**Play B · Matched control: {low.ddt:.1f}s ({low.ddtRate:.0%})**")
            render_replay(low, low_tracking, low_frames, names, "Play B — zero DDT", height=500)
        comparison = pd.DataFrame([
            {"play": "A · high DDT", "coverage": high.pff_passCoverage, "route runners": high.routeRunnerCount, "whole-play DDT": high.ddt, "top triple DDT": high.topTripleDdt, "DDT Rate": high.ddtRate, "mean route-nearest-defender separation": high.meanRouteNearestDefenderSeparation},
            {"play": "B · matched zero DDT", "coverage": low.pff_passCoverage, "route runners": low.routeRunnerCount, "whole-play DDT": low.ddt, "top triple DDT": low.topTripleDdt, "DDT Rate": low.ddtRate, "mean route-nearest-defender separation": low.meanRouteNearestDefenderSeparation},
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)

    with st.expander("Descriptive DDT rankings — sample-qualified, not player grades"):
        st.caption("These are exploratory tables, not validated player evaluations. Receiver pairs require at least three DDT-qualified plays; teams require at least 100 DDT-eligible plays.")
        if not pairs.empty:
            st.markdown("**Receiver pairs with the highest median DDT Rate**")
            st.dataframe(pairs[["receiverAName", "receiverBName", "qualifyingPlays", "medianDdtSeconds", "medianDdtRate", "meanDss"]].head(10), width="stretch", hide_index=True)
        if not teams.empty:
            st.markdown("**Offenses by median DDT Rate**")
            st.dataframe(teams[["possessionTeam", "qualifyingPlays", "activeDdtPlayRate", "medianDdtSeconds", "medianDdtRate", "meanDss"]].head(10), width="stretch", hide_index=True)

    with st.expander("Baseline result and limits"):
        if not baselines.empty and "model" in baselines.columns:
            st.dataframe(baselines, width="stretch", hide_index=True)
            st.caption("The B2 confidence interval crosses zero on this fixed game-level holdout. DDT does not yet demonstrate reliable incremental completion-prediction value beyond conventional geometry.")
        st.warning(
            "The 2021 data labels coverage participation and whole-play coverage family, not individual defender-to-receiver responsibility or zone handoffs. "
            "DDT is a transparent potential-stress measure through pass release, not proof of a coverage bust, target read, or causal offensive effect."
        )


if __name__ == "__main__":
    main()
