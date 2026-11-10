"""Render README snapshots from the project's saved results and real tracking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import load_tracking_game, resolve_data_dir
from metrics import normalize_play_direction


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "snapshots"
BG = "#07131F"
PANEL = "#10263A"
INK = "#F4F8FB"
MUTED = "#A9C0D0"
TEAL = "#36E3CD"
AMBER = "#FFC857"
PURPLE = "#A98AFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    base = "/System/Library/Fonts/Supplemental/Arial"
    try:
        return ImageFont.truetype(base + (" Bold.ttf" if bold else ".ttf"), size)
    except OSError:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def line(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str, size: int, color: str = INK, bold: bool = False) -> None:
    draw.text(xy, content, font=font(size, bold), fill=color)


def field_xy(x: float, y: float) -> tuple[int, int]:
    return int(95 + (x - 20) / 40 * 980), int(270 + y / 53.3 * 580)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    featured = json.loads((ROOT / "outputs/featured_plays.json").read_text())
    high = featured["highDdt"]
    game_id, play_id = int(high["gameId"]), int(high["playId"])
    conflicts = pd.read_csv(ROOT / "outputs/ddt_conflict_frames.csv")
    conflicts = conflicts.loc[(conflicts.gameId == game_id) & (conflicts.playId == play_id)]
    chosen = conflicts.sort_values("severity", ascending=False).iloc[0]
    frame_id = int(chosen.frameId)
    tracking = load_tracking_game(resolve_data_dir(), game_id)
    tracking = normalize_play_direction(tracking.loc[(tracking.playId == play_id) & (tracking.frameId == frame_id)])
    offense = str(high["possessionTeam"])
    receiver_ids = [int(chosen.receiverAId), int(chosen.receiverBId)]
    defender_id = int(chosen.defenderId)

    im = Image.new("RGB", (1800, 1000), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((42, 38, 1758, 962), radius=32, fill=PANEL, outline="#2B4A60", width=2)
    line(d, (84, 72), "THE IMPOSSIBLE CHOICE", 62, INK, True)
    line(d, (86, 154), "When two routes pull one defender in different directions", 30, MUTED)
    d.rounded_rectangle((70, 235, 1110, 884), radius=26, fill="#0B413C", outline="#31877D", width=3)
    for yard in range(20, 61, 5):
        x, _ = field_xy(yard, 0)
        d.line((x, 263, x, 855), fill="#246861", width=2)
        line(d, (x - 16, 806), str(yard - 10), 22, "#76AFA5")
    for row in tracking.itertuples():
        if pd.isna(row.nflId):
            continue
        ident = int(row.nflId)
        x, y = field_xy(float(row.x_norm), float(row.y_norm))
        if ident in receiver_ids:
            color, radius = TEAL, 20
        elif ident == defender_id:
            color, radius = AMBER, 22
        elif row.team == offense:
            color, radius = "#9CE8DE", 10
        else:
            color, radius = "#D2E1EA", 10
        d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=BG, width=3)
        if ident in receiver_ids:
            line(d, (x+23, y-16), "ROUTE " + ("A" if ident == receiver_ids[0] else "B"), 23, TEAL, True)
        if ident == defender_id:
            line(d, (x+27, y-15), "DEFENDER", 23, AMBER, True)
    positions = {int(r.nflId): field_xy(float(r.x_norm), float(r.y_norm)) for r in tracking.itertuples() if pd.notna(r.nflId)}
    for ident in receiver_ids:
        if ident in positions and defender_id in positions:
            d.line((positions[defender_id], positions[ident]), fill=AMBER, width=4)
    d.rounded_rectangle((1136, 235, 1728, 884), radius=26, fill="#0B1D2D", outline="#2B4A60", width=2)
    line(d, (1180, 282), "ONE PLAY, TWO THREATS", 27, TEAL, True)
    line(d, (1178, 334), "2.9 s", 86, INK, True)
    line(d, (1184, 438), "whole-play DDT", 28, MUTED)
    d.line((1180, 506, 1680, 506), fill="#35516B", width=2)
    line(d, (1180, 544), "29 / 34", 54, AMBER, True)
    line(d, (1184, 617), "pre-throw frames flagged", 27, MUTED)
    line(d, (1184, 653), "displayed triple: 1.5 s", 23, MUTED)
    line(d, (1184, 713), "2021  |  Week 4", 28, INK, True)
    line(d, (1184, 755), "IND offense vs MIA", 27, MUTED)
    line(d, (1184, 817), "Observed positions + modeled links", 23, PURPLE)
    line(d, (82, 908), "Actual NFL tracking frame  |  Game 2021100304, play 1243  |  Peak-severity pre-throw frame " + str(frame_id), 23, MUTED)
    im.save(OUT / "defensive_dilemma.png", optimize=True)

    evidence = Image.new("RGB", (1800, 520), BG)
    e = ImageDraw.Draw(evidence)
    e.rounded_rectangle((35, 35, 1765, 485), radius=30, fill=PANEL, outline="#2B4A60", width=2)
    line(e, (82, 68), "THE EVIDENCE, WITHOUT THE HYPE", 43, INK, True)
    cards = [
        ("7,578", "eligible plays", "122 games"),
        ("23.8 / 16.2", "mean DDT Rate (%)", "zone / man; descriptive"),
        ("+0.00032", "held-out log-loss gain", "95% CI crosses zero"),
    ]
    for i, (number, title, note) in enumerate(cards):
        left = 82 + i * 570
        e.rounded_rectangle((left, 160, left + 520, 430), radius=22, fill="#0B1D2D", outline="#35516B", width=2)
        line(e, (left + 28, 202), number, 49 if i == 1 else 64, TEAL if i < 2 else AMBER, True)
        line(e, (left + 30, 300), title, 29, INK, True)
        line(e, (left + 30, 351), note, 25, MUTED)
    evidence.save(OUT / "evidence_snapshot.png", optimize=True)


if __name__ == "__main__":
    main()
