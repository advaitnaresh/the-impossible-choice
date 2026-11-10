"""Input loading and verified key crosswalks for The Impossible Choice."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve
import os

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR.parent / "nfl-big-data-bowl-regional-event-data" / "data"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz"
NFLVERSE_PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv.gz"
ELIGIBLE_POSITIONS = {"WR", "TE", "RB"}


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve and validate the regional dataset directory."""

    source = data_dir or os.environ.get("BDB_DATA_DIR") or DEFAULT_DATA_DIR
    candidate = Path(source).expanduser().resolve()
    required = [candidate / "games.csv", candidate / "plays.csv", candidate / "players.csv", candidate / "pffScoutingData.csv", candidate / "tracking"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Regional data directory is incomplete: {', '.join(missing)}")
    return candidate


def load_static_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load small static regional tables once; tracking stays on disk."""

    base = resolve_data_dir(data_dir)
    return {
        "games": pd.read_csv(base / "games.csv"),
        "plays": pd.read_csv(base / "plays.csv"),
        "players": pd.read_csv(base / "players.csv"),
        "pff": pd.read_csv(base / "pffScoutingData.csv"),
    }


def build_role_lookup(static: dict[str, pd.DataFrame]) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    """Return actual WR/TE/RB route runners and PFF coverage defenders by play."""

    pff = static["pff"]
    player_positions = static["players"][["nflId", "officialPosition"]]
    tagged = pff.merge(player_positions, on="nflId", how="left", validate="many_to_one")
    route_rows = tagged.loc[
        tagged["pff_role"].eq("Pass Route") & tagged["officialPosition"].isin(ELIGIBLE_POSITIONS),
        ["gameId", "playId", "nflId"],
    ]
    coverage_rows = tagged.loc[
        tagged["pff_role"].eq("Coverage"), ["gameId", "playId", "nflId"]
    ]

    route_lookup = {
        (int(game_id), int(play_id)): group["nflId"].astype(int).tolist()
        for (game_id, play_id), group in route_rows.groupby(["gameId", "playId"])
    }
    coverage_lookup = {
        (int(game_id), int(play_id)): group["nflId"].astype(int).tolist()
        for (game_id, play_id), group in coverage_rows.groupby(["gameId", "playId"])
    }
    return route_lookup, coverage_lookup


def load_tracking_game(data_dir: str | Path, game_id: int) -> pd.DataFrame:
    """Load exactly one game tracking file to control memory use."""

    path = Path(data_dir) / "tracking" / f"tracking_{int(game_id)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No tracking file found for game {game_id}: {path}")
    return pd.read_csv(path, low_memory=False)


def select_games(
    games: pd.DataFrame,
    game_ids: Iterable[int] | None = None,
    weeks: Iterable[int] | None = None,
    max_games: int | None = None,
) -> list[int]:
    """Choose a deterministic game list for smoke tests or a full run."""

    selected = games.copy()
    if game_ids:
        requested = {int(value) for value in game_ids}
        selected = selected.loc[selected["gameId"].isin(requested)]
    if weeks:
        requested_weeks = {int(value) for value in weeks}
        selected = selected.loc[selected["week"].isin(requested_weeks)]
    selected = selected.sort_values(["week", "gameId"])
    if max_games is not None:
        selected = selected.head(int(max_games))
    return selected["gameId"].astype(int).tolist()


def _download_if_missing(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    urlretrieve(url, temporary)
    temporary.replace(path)


def load_nflverse_enrichment(cache_dir: str | Path, download: bool = False) -> pd.DataFrame | None:
    """Load optional PBP outcomes and map its GSIS target IDs to numeric nflId.

    The returned data has one record per `(gameId, playId)` when key uniqueness
    holds. `download=True` fetches only nflverse's compact PBP and player files.
    """

    cache = Path(cache_dir)
    pbp_path = cache / "play_by_play_2021.csv.gz"
    players_path = cache / "nflverse_players.csv.gz"
    if download:
        _download_if_missing(pbp_path, PBP_URL)
        _download_if_missing(players_path, NFLVERSE_PLAYERS_URL)
    if not pbp_path.exists() or not players_path.exists():
        return None

    pbp_columns = [
        "old_game_id", "play_id", "play_type", "pass_attempt", "penalty", "qb_spike",
        "complete_pass", "incomplete_pass", "interception", "first_down", "first_down_pass",
        "passing_yards", "receiving_yards", "air_yards", "yards_after_catch", "pass_location",
        "pass_length", "receiver_player_id", "receiver_player_name", "epa", "success", "wp",
        "wpa", "cpoe", "cp", "sack", "qb_scramble",
    ]
    pbp = pd.read_csv(pbp_path, usecols=pbp_columns, low_memory=False)
    crosswalk = pd.read_csv(players_path, usecols=["gsis_id", "nfl_id"], low_memory=False).dropna()
    crosswalk["nfl_id"] = pd.to_numeric(crosswalk["nfl_id"], errors="coerce")
    crosswalk = crosswalk.dropna(subset=["nfl_id"]).drop_duplicates("gsis_id")

    enriched = pbp.merge(
        crosswalk.rename(columns={"nfl_id": "targetNflId"}),
        left_on="receiver_player_id",
        right_on="gsis_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["gsis_id"])
    enriched = enriched.rename(columns={"old_game_id": "gameId", "play_id": "playId"})
    enriched["gameId"] = enriched["gameId"].astype(int)
    enriched["playId"] = enriched["playId"].astype(int)
    enriched["targetNflId"] = enriched["targetNflId"].astype("Int64")
    if enriched.duplicated(["gameId", "playId"]).any():
        raise ValueError("nflverse PBP did not have unique (gameId, playId) keys")
    return enriched
