"""Run the reproducible The Impossible Choice analysis.

Example:
    python analysis.py --max-games 4 --download-pbp --force
    python analysis.py --download-pbp --force

The full run reads a single tracking-game file at a time and writes compact,
play-level caches beneath outputs/cache/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import (
    build_role_lookup,
    load_nflverse_enrichment,
    load_static_data,
    load_tracking_game,
    resolve_data_dir,
    select_games,
)
from metrics import MetricConfig, analyze_play
from visualization import build_conflict_timeline, write_featured_demo


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
METRIC_VERSION = "1.0.2-geometric-candidate"


def _target_lookup(enrichment: pd.DataFrame | None) -> dict[tuple[int, int], int]:
    if enrichment is None:
        return {}
    valid = enrichment.dropna(subset=["targetNflId"])
    return {
        (int(row.gameId), int(row.playId)): int(row.targetNflId)
        for row in valid[["gameId", "playId", "targetNflId"]].itertuples(index=False)
    }


def _cache_paths(cache_dir: Path, game_id: int) -> dict[str, Path]:
    return {
        "plays": cache_dir / f"play_results_{game_id}.csv",
        "segments": cache_dir / f"segments_{game_id}.csv",
        "frames": cache_dir / f"frames_{game_id}.csv",
    }


def _read_cached_game(paths: dict[str, Path]) -> tuple[list[dict], list[dict], list[dict]] | None:
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        records = pd.read_csv(paths["plays"]).to_dict("records")
        segments = pd.read_csv(paths["segments"]).to_dict("records")
        frames = pd.read_csv(paths["frames"]).to_dict("records")
        return records, segments, frames
    except pd.errors.EmptyDataError:
        return None


def _write_cached_game(paths: dict[str, Path], records: list[dict], segments: list[dict], frames: list[dict]) -> None:
    pd.DataFrame(records).to_csv(paths["plays"], index=False)
    pd.DataFrame(segments).to_csv(paths["segments"], index=False)
    pd.DataFrame(frames).to_csv(paths["frames"], index=False)


def process_game(
    data_dir: Path,
    game_id: int,
    route_lookup: dict[tuple[int, int], list[int]],
    coverage_lookup: dict[tuple[int, int], list[int]],
    target_lookup: dict[tuple[int, int], int],
    config: MetricConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Process one tracking file and retain only compact DDT outputs."""

    tracking = load_tracking_game(data_dir, game_id)
    records: list[dict] = []
    segments: list[dict] = []
    frames: list[dict] = []
    for (_, play_id), play_tracking in tracking.groupby(["gameId", "playId"], sort=False):
        key = (int(game_id), int(play_id))
        route_ids = route_lookup.get(key, [])
        coverage_ids = coverage_lookup.get(key, [])
        if len(route_ids) < 2 or not coverage_ids:
            continue
        result, play_segments, play_frames = analyze_play(
            tracking=play_tracking,
            game_id=key[0],
            play_id=key[1],
            route_ids=route_ids,
            coverage_ids=coverage_ids,
            target_nfl_id=target_lookup.get(key),
            config=config,
        )
        if result is not None:
            records.append(result)
            segments.extend(play_segments)
            frames.extend(play_frames)
    return records, segments, frames


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _row_to_json(row: pd.Series) -> dict[str, Any]:
    return {str(key): _safe_json_value(value) for key, value in row.items()}


def build_featured_plays(results: pd.DataFrame) -> dict[str, Any]:
    """Select real high, low, and counterexample plays after results are known."""

    showcase_pool = results.copy()
    clean_columns = {"play_type", "pass_attempt", "penalty"}
    if clean_columns.issubset(showcase_pool.columns):
        clean = showcase_pool.loc[
            showcase_pool["play_type"].eq("pass")
            & showcase_pool["pass_attempt"].eq(1)
            & showcase_pool["penalty"].eq(0)
        ]
        if not clean.empty:
            showcase_pool = clean
    active = showcase_pool.loc[showcase_pool["ddtFrames"] > 0].sort_values(
        ["ddtRate", "dss", "ddt"], ascending=False
    )
    if active.empty:
        raise RuntimeError("No qualifying DDT frames were found; review metric assumptions before making a claim.")
    high = active.iloc[0]
    matched_low = showcase_pool.loc[
        (showcase_pool["ddtFrames"] == 0)
        & (showcase_pool["pff_passCoverageType"] == high["pff_passCoverageType"])
        & (showcase_pool["routeRunnerCount"] == high["routeRunnerCount"])
    ]
    if matched_low.empty:
        matched_low = showcase_pool.loc[showcase_pool["ddtFrames"] == 0]
    low = matched_low.sort_values("meanRouteNearestDefenderSeparation", ascending=False).iloc[0]

    zero = showcase_pool.loc[showcase_pool["ddtFrames"] == 0]
    if not zero.empty:
        density_cutoff = zero["meanLocalReceiverDensity"].quantile(0.85)
        counter_pool = zero.loc[zero["meanLocalReceiverDensity"] >= density_cutoff]
        counterexample = counter_pool.sort_values("meanNearbyDefenders", ascending=False).iloc[0] if not counter_pool.empty else zero.iloc[0]
    else:
        counterexample = low

    return {
        "metricVersion": METRIC_VERSION,
        "highDdt": _row_to_json(high),
        "matchedLowDdt": _row_to_json(low),
        "counterexample": _row_to_json(counterexample),
        "selectionRule": "Highest observed DDT Rate; a zero-DDT matched low comparison when available; and a dense zero-DDT counterexample.",
    }


def _mean_with_ci(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    valid = values.dropna().to_numpy(dtype=float)
    if len(valid) < 2:
        return (float(valid.mean()) if len(valid) else None, None, None)
    standard_error = valid.std(ddof=1) / np.sqrt(len(valid))
    return (float(valid.mean()), float(valid.mean() - 1.96 * standard_error), float(valid.mean() + 1.96 * standard_error))


def coverage_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Produce descriptive coverage-family results with transparent uncertainty."""

    rows: list[dict[str, Any]] = []
    for coverage, group in results.groupby("pff_passCoverageType", dropna=False):
        mean_rate, ci_low, ci_high = _mean_with_ci(group["ddtRate"])
        row: dict[str, Any] = {
            "coverageType": coverage if pd.notna(coverage) else "Missing",
            "plays": int(len(group)),
            "activeDdtRate": float((group["ddtFrames"] > 0).mean()),
            "meanDdtSeconds": float(group["ddt"].mean()),
            "meanDdtRate": mean_rate,
            "meanDdtRateCiLow": ci_low,
            "meanDdtRateCiHigh": ci_high,
            "meanDss": float(group["dss"].mean()),
        }
        if "epa" in group:
            row["meanEpa"] = float(group["epa"].mean()) if group["epa"].notna().any() else None
        if "complete_pass" in group:
            row["completionRate"] = float(group["complete_pass"].mean()) if group["complete_pass"].notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("plays", ascending=False)


def _build_pipeline(numeric_features: list[str], categorical_features: list[str]):
    """Build a fixed preprocessing/logistic pipeline only for baseline comparison."""

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ])
    return Pipeline([
        ("transform", transform),
        ("model", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    metric = {
        "logLoss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brierScore": float(brier_score_loss(y_true, probability)),
    }
    metric["auc"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else float("nan")
    return metric


def _bootstrap_logloss_delta(
    test: pd.DataFrame,
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    ddt_probability: np.ndarray,
    iterations: int = 500,
) -> tuple[float, float]:
    """Game-cluster bootstrap CI for B1 minus B2 log loss, positive favors DDT."""

    epsilon = 1e-12
    baseline_loss = -(y_true * np.log(np.clip(baseline_probability, epsilon, 1 - epsilon)) + (1 - y_true) * np.log(np.clip(1 - baseline_probability, epsilon, 1 - epsilon)))
    ddt_loss = -(y_true * np.log(np.clip(ddt_probability, epsilon, 1 - epsilon)) + (1 - y_true) * np.log(np.clip(1 - ddt_probability, epsilon, 1 - epsilon)))
    loss = pd.DataFrame({"gameId": test["gameId"].to_numpy(), "delta": baseline_loss - ddt_loss})
    game_losses = loss.groupby("gameId")["delta"].agg(list)
    games = list(game_losses.index)
    if len(games) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260210)
    means = []
    for _ in range(iterations):
        sampled = rng.choice(games, size=len(games), replace=True)
        observations = np.concatenate([np.asarray(game_losses.loc[game], dtype=float) for game in sampled])
        means.append(float(observations.mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_baselines(results: pd.DataFrame) -> pd.DataFrame:
    """Evaluate fixed B0/B1/B2 completion models on held-out games.

    The target receiver is used only in the conventional separation baseline;
    DDT itself never uses the PBP target identity or post-release outcome.
    """

    required = {"complete_pass", "pass_attempt", "penalty", "play_type", "targetSeparation"}
    if not required.issubset(results.columns):
        return pd.DataFrame([{"status": "Skipped: nflverse PBP enrichment unavailable."}])

    eligible = results.loc[
        results["play_type"].eq("pass")
        & results["pass_attempt"].eq(1)
        & results["penalty"].eq(0)
        & results["complete_pass"].notna()
    ].copy()
    eligible["complete_pass"] = eligible["complete_pass"].astype(int)
    eligible = eligible.loc[eligible["prethrowDurationSeconds"].notna()]
    game_ids = sorted(eligible["gameId"].unique())
    if len(game_ids) < 10 or eligible["complete_pass"].nunique() < 2:
        return pd.DataFrame([{"status": "Skipped: insufficient clean, labeled games for held-out baseline evaluation."}])

    # Every fifth game is a stable, held-out game; no frame/play splitting occurs.
    test_games = set(game_ids[::5])
    test = eligible.loc[eligible["gameId"].isin(test_games)].copy()
    train = eligible.loc[~eligible["gameId"].isin(test_games)].copy()
    if test["complete_pass"].nunique() < 2 or train["complete_pass"].nunique() < 2:
        return pd.DataFrame([{"status": "Skipped: holdout did not contain both completion classes."}])

    situation_numeric = ["down", "yardsToGo", "absoluteYardlineNumber", "prethrowDurationSeconds", "routeRunnerCount"]
    situation_categorical = ["pff_passCoverageType", "offenseFormation", "personnelO", "pff_playAction"]
    geometry_numeric = ["targetSeparation", "targetSpeed", "meanRouteSpeed", "meanNearbyDefenders", "meanLocalReceiverDensity"]
    ddt_numeric = ["ddtRate", "dss"]
    specifications = [
        ("B0 situation", situation_numeric, situation_categorical),
        ("B1 situation + conventional geometry", situation_numeric + geometry_numeric, situation_categorical),
        ("B2 B1 + DDT", situation_numeric + geometry_numeric + ddt_numeric, situation_categorical),
    ]
    prediction_map: dict[str, np.ndarray] = {}
    report_rows: list[dict[str, Any]] = []
    y_train = train["complete_pass"].to_numpy()
    y_test = test["complete_pass"].to_numpy()
    for label, numeric, categorical in specifications:
        pipeline = _build_pipeline(numeric, categorical)
        columns = numeric + categorical
        pipeline.fit(train[columns], y_train)
        prediction = pipeline.predict_proba(test[columns])[:, 1]
        prediction_map[label] = prediction
        metrics = _binary_metrics(y_test, prediction)
        report_rows.append({
            "model": label,
            "trainPlays": int(len(train)),
            "testPlays": int(len(test)),
            "trainGames": int(train["gameId"].nunique()),
            "testGames": int(test["gameId"].nunique()),
            **metrics,
        })

    baseline_label = "B1 situation + conventional geometry"
    ddt_label = "B2 B1 + DDT"
    ci_low, ci_high = _bootstrap_logloss_delta(test, y_test, prediction_map[baseline_label], prediction_map[ddt_label])
    for row in report_rows:
        if row["model"] == ddt_label:
            row["vsB1LogLossImprovement"] = report_rows[1]["logLoss"] - row["logLoss"]
            row["vsB1LogLossImprovementCiLow"] = ci_low
            row["vsB1LogLossImprovementCiHigh"] = ci_high
            row["interpretation"] = "Positive improvement favors B2; a confidence interval crossing zero is inconclusive."
    return pd.DataFrame(report_rows)


def build_finding_markdown(summary: dict[str, Any], coverage: pd.DataFrame, baselines: pd.DataFrame) -> str:
    """Write a data-derived finding without assuming that DDT wins its baseline."""

    lines = [
        "# Observed evidence snapshot",
        "",
        f"The full processed output contains {summary['processedPlayRecords']:,} DDT-eligible snap-to-throw plays from {summary['processedGames']} games. "
        f"Geometric DDT appears on {summary['playsWithDdt']:,} plays ({summary['activeDdtPlayRate']:.1%}), with a mean duration of {summary['meanDdtSeconds']:.2f} seconds.",
        "",
    ]
    substantial = coverage.loc[coverage["plays"] >= 100].copy()
    if len(substantial) >= 2:
        high = substantial.sort_values("meanDdtRate", ascending=False).iloc[0]
        man_reference = substantial.loc[substantial["coverageType"].eq("Man")]
        low = man_reference.iloc[0] if not man_reference.empty and high.coverageType != "Man" else substantial.sort_values("meanDdtRate", ascending=True).iloc[0]
        lines.extend([
            "## Descriptive coverage contrast",
            "",
            f"Among coverage groups with at least 100 processed plays, **{high.coverageType}** had the highest mean DDT Rate ({high.meanDdtRate:.1%}; n={int(high.plays):,}) and **{low.coverageType}** the lowest ({low.meanDdtRate:.1%}; n={int(low.plays):,}). This is descriptive geometry, not evidence that one coverage family is inherently worse or that assignment responsibility was known.",
            "",
        ])
    if "model" in baselines.columns and (baselines["model"] == "B2 B1 + DDT").any():
        b1 = baselines.loc[baselines["model"] == "B1 situation + conventional geometry"].iloc[0]
        b2 = baselines.loc[baselines["model"] == "B2 B1 + DDT"].iloc[0]
        improvement = b2.get("vsB1LogLossImprovement", np.nan)
        lower = b2.get("vsB1LogLossImprovementCiLow", np.nan)
        upper = b2.get("vsB1LogLossImprovementCiHigh", np.nan)
        if pd.notna(improvement) and pd.notna(lower) and pd.notna(upper):
            conclusion = "The interval excludes zero, so this fixed held-out split supports incremental completion-prediction information." if lower > 0 else "The interval crosses zero, so this fixed held-out split does not establish incremental completion-prediction value."
            lines.extend([
                "## Held-out baseline result",
                "",
                f"On {int(b2.testPlays):,} clean pass attempts from {int(b2.testGames)} held-out games, conventional geometry (B1) had log loss {b1.logLoss:.4f}; B1 plus DDT (B2) had {b2.logLoss:.4f}. The B1-minus-B2 log-loss difference was {improvement:.5f} with a game-cluster bootstrap 95% interval of [{lower:.5f}, {upper:.5f}]. {conclusion}",
                "",
            ])
    else:
        lines.extend(["## Held-out baseline result", "", "A held-out completion comparison was not available for this run.", ""])
    lines.extend([
        "## Required interpretation",
        "",
        "DDT identifies potential shared geometric stress before pass release. The data lacks individual coverage responsibility and all-route labels, so it cannot prove a coverage bust, a quarterback read, or a causal route-combination effect.",
    ])
    return "\n".join(lines) + "\n"


def descriptive_rankings(results: pd.DataFrame, static: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample-qualified descriptive, not validated, DDT rankings."""

    names = static["players"][["nflId", "displayName"]].copy()
    active = results.loc[
        results["ddtFrames"].gt(0)
        & results["topReceiverAId"].notna()
        & results["topReceiverBId"].notna()
    ].copy()
    active["receiverAId"] = active["topReceiverAId"].astype(int)
    active["receiverBId"] = active["topReceiverBId"].astype(int)
    pair = active.groupby(["receiverAId", "receiverBId"], as_index=False).agg(
        qualifyingPlays=("playId", "size"),
        medianDdtSeconds=("ddt", "median"),
        medianDdtRate=("ddtRate", "median"),
        meanDss=("dss", "mean"),
    )
    pair = pair.loc[pair["qualifyingPlays"] >= 3].sort_values(
        ["medianDdtRate", "qualifyingPlays"], ascending=[False, False]
    )
    pair = pair.merge(names.rename(columns={"nflId": "receiverAId", "displayName": "receiverAName"}), on="receiverAId", how="left")
    pair = pair.merge(names.rename(columns={"nflId": "receiverBId", "displayName": "receiverBName"}), on="receiverBId", how="left")
    pair["minimumSampleRule"] = "At least 3 DDT-qualified plays; descriptive only."

    team = results.groupby("possessionTeam", as_index=False).agg(
        qualifyingPlays=("playId", "size"),
        activeDdtPlayRate=("ddtFrames", lambda values: float((values > 0).mean())),
        medianDdtSeconds=("ddt", "median"),
        medianDdtRate=("ddtRate", "median"),
        meanDss=("dss", "mean"),
    )
    team = team.loc[team["qualifyingPlays"] >= 100].sort_values("medianDdtRate", ascending=False)
    team["minimumSampleRule"] = "At least 100 DDT-eligible plays; descriptive only."
    return pair.reset_index(drop=True), team.reset_index(drop=True)


def normalized_field_markers(play_context: pd.Series, tracking: pd.DataFrame) -> tuple[float | None, float | None]:
    """Translate source LOS and first-down context to normalized left-to-right x."""

    raw_los = play_context.get("absoluteYardlineNumber")
    if pd.isna(raw_los) or tracking.empty:
        return None, None
    direction = str(tracking["playDirection"].dropna().iloc[0])
    line_of_scrimmage = 120.0 - float(raw_los) if direction == "left" else float(raw_los)
    yards_to_go = play_context.get("yardsToGo")
    first_down = line_of_scrimmage + float(yards_to_go) if pd.notna(yards_to_go) else None
    return line_of_scrimmage, first_down


def write_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    segments: pd.DataFrame,
    frames: pd.DataFrame,
    static: dict[str, pd.DataFrame],
    data_dir: Path,
    config: MetricConfig,
    processed_games: list[int],
    enrichment_loaded: bool,
) -> dict[str, Any]:
    """Write reproducible compact results, actual feature replay, and summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "ddt_play_results.csv", index=False)
    segments.to_csv(output_dir / "ddt_conflict_segments.csv", index=False)
    frames.to_csv(output_dir / "ddt_conflict_frames.csv", index=False)

    coverage = coverage_summary(results)
    coverage.to_csv(output_dir / "coverage_summary.csv", index=False)
    pair_rankings, team_rankings = descriptive_rankings(results, static)
    pair_rankings.to_csv(output_dir / "receiver_pair_rankings.csv", index=False)
    team_rankings.to_csv(output_dir / "team_stress_summary.csv", index=False)
    baselines = evaluate_baselines(results) if enrichment_loaded else pd.DataFrame([{"status": "Skipped: nflverse PBP enrichment unavailable."}])
    baselines.to_csv(output_dir / "baseline_comparison.csv", index=False)

    featured = build_featured_plays(results)
    (output_dir / "featured_plays.json").write_text(json.dumps(featured, indent=2), encoding="utf-8")
    high = featured["highDdt"]
    high_tracking = load_tracking_game(data_dir, int(high["gameId"]))
    high_tracking = high_tracking.loc[high_tracking["playId"].eq(int(high["playId"]))]
    play_context = results.loc[(results["gameId"] == int(high["gameId"])) & (results["playId"] == int(high["playId"]))].iloc[0]
    names = dict(zip(static["players"]["nflId"].astype(int), static["players"]["displayName"]))
    featured_frames = frames.loc[(frames["gameId"] == int(high["gameId"])) & (frames["playId"] == int(high["playId"]))]
    line_of_scrimmage, first_down_marker = normalized_field_markers(play_context, high_tracking)
    replay_path = write_featured_demo(
        output_dir=output_dir,
        tracking=high_tracking,
        possession_team=str(play_context["possessionTeam"]),
        name_map=names,
        featured_record=high,
        conflict_frames=featured_frames,
        line_of_scrimmage=line_of_scrimmage,
        first_down_marker=first_down_marker,
    )
    timeline = build_conflict_timeline(featured_frames, int(high["snapFrame"]), int(high["passForwardFrame"]), config.frame_seconds)
    timeline_path = output_dir / "featured_play_timeline.html"
    timeline.write_html(timeline_path, include_plotlyjs=True, full_html=True)

    active_rate = float((results["ddtFrames"] > 0).mean())
    summary = {
        "metricVersion": METRIC_VERSION,
        "dataScope": "2021 regular season Weeks 1-8 regional tracking data",
        "processedGames": len(processed_games),
        "processedPlayRecords": int(len(results)),
        "processedPopulation": "Plays with snap/pass-forward events, at least two PFF Pass Route WR/TE/RB players, and at least one PFF Coverage defender.",
        "playsWithDdt": int((results["ddtFrames"] > 0).sum()),
        "activeDdtPlayRate": active_rate,
        "meanDdtSeconds": float(results["ddt"].mean()),
        "meanDdtRate": float(results["ddtRate"].mean()),
        "metricAssumptions": config.to_dict(),
        "prethrowOnly": True,
        "primaryLimit": "DDT is a geometric potential-stress measure; the 2021 source has no individual coverage assignments or defender-receiver matchup labels.",
        "featuredReplay": replay_path.name,
        "featuredTimeline": timeline_path.name,
        "baselineFile": "baseline_comparison.csv",
        "coverageSummaryFile": "coverage_summary.csv",
        "receiverPairRankingsFile": "receiver_pair_rankings.csv",
        "teamStressSummaryFile": "team_stress_summary.csv",
        "findingFile": "finding.md",
    }
    (output_dir / "finding.md").write_text(build_finding_markdown(summary, coverage, baselines), encoding="utf-8")
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "cache"
    if args.force and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    static = load_static_data(data_dir)
    route_lookup, coverage_lookup = build_role_lookup(static)
    enrichment = load_nflverse_enrichment(output_dir / "source_cache", download=args.download_pbp)
    target_lookup = _target_lookup(enrichment)
    games = select_games(static["games"], weeks=args.weeks, max_games=args.max_games)
    if not games:
        raise ValueError("No games selected.")

    config = MetricConfig()
    all_records: list[dict] = []
    all_segments: list[dict] = []
    all_frames: list[dict] = []
    for index, game_id in enumerate(games, start=1):
        paths = _cache_paths(cache_dir, game_id)
        cached = None if args.force else _read_cached_game(paths)
        if cached is None:
            records, segments, frames = process_game(data_dir, game_id, route_lookup, coverage_lookup, target_lookup, config)
            _write_cached_game(paths, records, segments, frames)
            state = "processed"
        else:
            records, segments, frames = cached
            state = "cached"
        all_records.extend(records)
        all_segments.extend(segments)
        all_frames.extend(frames)
        print(f"[{index}/{len(games)}] {state} game {game_id}: {len(records)} valid plays, {len(frames)} qualifying frames")

    if not all_records:
        raise RuntimeError("No valid plays were processed. Check source paths and event definitions.")
    results = pd.DataFrame(all_records)
    play_context_columns = [
        "gameId", "playId", "playDescription", "quarter", "down", "yardsToGo", "possessionTeam", "defensiveTeam",
        "absoluteYardlineNumber", "passResult", "playResult", "offenseFormation", "personnelO", "defendersInBox",
        "personnelD", "dropBackType", "pff_playAction", "pff_passCoverage", "pff_passCoverageType",
    ]
    results = results.merge(static["plays"][play_context_columns], on=["gameId", "playId"], how="left", validate="one_to_one")
    results = results.merge(static["games"][["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"]], on="gameId", how="left", validate="many_to_one")
    if enrichment is not None:
        results = results.merge(enrichment, on=["gameId", "playId"], how="left", validate="one_to_one")
    results = results.sort_values(["gameId", "playId"]).reset_index(drop=True)
    segments = pd.DataFrame(all_segments)
    frames = pd.DataFrame(all_frames)
    if segments.empty:
        segments = pd.DataFrame(columns=["gameId", "playId", "receiverAId", "receiverBId", "defenderId"])
    if frames.empty:
        frames = pd.DataFrame(columns=["gameId", "playId", "frameId", "receiverAId", "receiverBId", "defenderId", "severity"])

    player_names = static["players"][["nflId", "displayName"]]
    for identifier, label in [("receiverAId", "receiverAName"), ("receiverBId", "receiverBName"), ("defenderId", "defenderName")]:
        if identifier in segments.columns:
            segments = segments.merge(player_names.rename(columns={"nflId": identifier, "displayName": label}), on=identifier, how="left")
    summary = write_outputs(output_dir, results, segments, frames, static, data_dir, config, games, enrichment is not None)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process The Impossible Choice DDT analysis.")
    parser.add_argument("--data-dir", default=None, help="Path to the regional data directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for compact caches and results.")
    parser.add_argument("--max-games", type=int, default=None, help="Limit games for a fast smoke test.")
    parser.add_argument("--weeks", type=int, nargs="*", default=None, help="Optional week subset, e.g. --weeks 1 2.")
    parser.add_argument("--download-pbp", action="store_true", help="Download compact nflverse 2021 PBP/player files into outputs/source_cache.")
    parser.add_argument("--force", action="store_true", help="Recompute project-owned compact cache files.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
