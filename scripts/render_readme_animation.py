"""Build a compact README GIF from actual tracking and saved DDT frame results."""

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
OUT = ROOT / "snapshots" / "play_replay.gif"
NAVY, PANEL, FIELD = "#07131F", "#10263A", "#0B413C"
WHITE, MUTED, TEAL, GOLD = "#F4F8FB", "#A9C0D0", "#36E3CD", "#FFC857"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    base = "/System/Library/Fonts/Supplemental/Arial"
    try:
        return ImageFont.truetype(base + (" Bold.ttf" if bold else ".ttf"), size)
    except OSError:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill: str = WHITE, bold: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill)


def position(x: float, y: float) -> tuple[int, int]:
    return int(74 + (x - 20) / 40 * 744), int(145 + y / 53.3 * 380)


def render_frame(
    frame_id: int,
    tracking: pd.DataFrame,
    snap: int,
    throw: int,
    pair: tuple[int, int],
    defender: int,
    offense: str,
    active_all: set[int],
    active_top: set[int],
) -> Image.Image:
    image = Image.new("RGB", (1200, 660), NAVY)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 20, 1180, 640), radius=22, fill=PANEL, outline="#35516B", width=2)
    text(draw, (48, 42), "THE IMPOSSIBLE CHOICE", 39, WHITE, True)
    text(draw, (50, 92), "Watch two real routes develop around one defender", 20, MUTED)
    draw.rounded_rectangle((48, 125, 844, 548), radius=18, fill=FIELD, outline="#31877D", width=2)

    for yard in range(20, 61, 5):
        x, _ = position(yard, 0)
        draw.line((x, 139, x, 534), fill="#246861", width=2)
        text(draw, (x - 10, 510), str(yard - 10), 16, "#8CBFB7")

    current = tracking.loc[tracking.frameId == frame_id]
    focus_ids = [*pair, defender]
    for ident in focus_ids:
        path = tracking.loc[(tracking.nflId == ident) & (tracking.frameId <= frame_id)].sort_values("frameId")
        points = [position(float(row.x_norm), float(row.y_norm)) for row in path.itertuples()]
        if len(points) > 1:
            draw.line(points, fill="#1DB3A4" if ident in pair else "#B88628", width=4, joint="curve")

    row_by_id = {int(row.nflId): row for row in current.itertuples() if pd.notna(row.nflId)}
    if frame_id in active_top and all(ident in row_by_id for ident in focus_ids):
        d = row_by_id[defender]
        origin = position(float(d.x_norm), float(d.y_norm))
        for ident in pair:
            r = row_by_id[ident]
            draw.line((origin, position(float(r.x_norm), float(r.y_norm))), fill=GOLD, width=4)

    for ident, row in row_by_id.items():
        x, y = position(float(row.x_norm), float(row.y_norm))
        if not (55 < x < 837 and 135 < y < 538):
            continue
        if ident in pair:
            color, radius = TEAL, 13
        elif ident == defender:
            color, radius = GOLD, 15
        elif row.team == offense:
            color, radius = "#A8DBD6", 7
        else:
            color, radius = "#C3CBD9", 7
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=NAVY, width=2)
        if ident in pair:
            text(draw, (x+16, y-14), "A" if ident == pair[0] else "B", 21, TEAL, True)
        elif ident == defender:
            text(draw, (x+18, y-14), "D", 21, GOLD, True)

    draw.rounded_rectangle((865, 125, 1152, 548), radius=18, fill="#0B1D2D", outline="#35516B", width=2)
    text(draw, (888, 148), "HOW TO READ THIS", 20, TEAL, True)
    draw.ellipse((891, 197, 915, 221), fill=TEAL)
    text(draw, (928, 193), "two route runners", 20)
    draw.ellipse((891, 239, 915, 263), fill=GOLD)
    text(draw, (928, 235), "one defender", 20)
    draw.line((890, 300, 919, 300), fill=GOLD, width=4)
    text(draw, (928, 285), "modeled conflict", 19)
    text(draw, (888, 353), "2.9 s", 49, WHITE, True)
    text(draw, (891, 411), "whole-play DDT", 20, MUTED)
    text(draw, (891, 447), "1.5 s shown trio", 18, MUTED)
    badge = "TRIO FLAGGED" if frame_id in active_top else "ROUTES DEVELOPING"
    draw.rounded_rectangle((886, 486, 1133, 527), radius=12, fill="#644E1F" if frame_id in active_top else "#254457")
    text(draw, (902, 495), badge, 18, GOLD if frame_id in active_top else MUTED, True)

    text(draw, (50, 568), f"{(frame_id - snap) / 10:.1f} s after snap", 20, WHITE, True)
    text(draw, (230, 568), "Teal = any DDT frame   Gold = shown trio", 17, MUTED)
    start, width = 52, 1095
    draw.rounded_rectangle((start, 605, start + width, 621), radius=6, fill="#244057")
    for f in range(snap, throw):
        x0 = start + int((f - snap) / (throw - snap) * width)
        x1 = start + int((f - snap + 1) / (throw - snap) * width) - 2
        if f in active_all:
            draw.rectangle((x0, 605, x1, 621), fill=TEAL)
        if f in active_top:
            draw.rectangle((x0, 605, x1, 612), fill=GOLD)
    cursor_x = start + int((frame_id - snap + 0.5) / (throw - snap) * width)
    draw.line((cursor_x, 596, cursor_x, 631), fill=WHITE, width=3)
    return image


def main() -> None:
    featured = json.loads((ROOT / "outputs/featured_plays.json").read_text())
    high = featured["highDdt"]
    game, play = int(high["gameId"]), int(high["playId"])
    snap, throw = int(high["snapFrame"]), int(high["passForwardFrame"])
    pair = (int(high["topReceiverAId"]), int(high["topReceiverBId"]))
    defender = int(high["topDefenderId"])
    tracking = load_tracking_game(resolve_data_dir(), game)
    tracking = normalize_play_direction(tracking.loc[(tracking.playId == play) & (tracking.frameId >= snap) & (tracking.frameId < throw)])
    frame_results = pd.read_csv(ROOT / "outputs/ddt_conflict_frames.csv")
    frame_results = frame_results.loc[(frame_results.gameId == game) & (frame_results.playId == play)]
    active_all = set(frame_results.frameId.astype(int))
    active_top = {
        int(row.frameId) for row in frame_results.itertuples()
        if {int(row.receiverAId), int(row.receiverBId)} == set(pair) and int(row.defenderId) == defender
    }
    assert len(active_all) == int(high["ddtFrames"])
    assert len(active_top) == int(high["topPairFrames"])
    frames = [render_frame(f, tracking, snap, throw, pair, defender, str(high["possessionTeam"]), active_all, active_top) for f in range(snap, throw)]
    # GitHub uses frame zero as the poster, so show a real flagged frame first.
    cover = frames[min(active_top) - snap]
    frames = [cover, *frames]
    durations = [1100] + [135] * (len(frames) - 2) + [1300]
    OUT.parent.mkdir(exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
