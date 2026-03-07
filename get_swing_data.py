from __future__ import annotations
from pybaseball import playerid_lookup, statcast, statcast_pitcher, batting_stats, playerid_reverse_lookup
from pybaseball.cache import cache
from tqdm import tqdm

import pandas as pd

cache.enable()


PLATE_WIDTH = 17 / 12  # inches to feet


def normalise_pitch_coords(df: pd.DataFrame) -> None:
    df["px"] = df["plate_x"] / (PLATE_WIDTH / 2)
    df["pz"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    df.drop(columns=["plate_x", "plate_z", "sz_top", "sz_bot"], inplace=True)

    df["prev_px"] = df["prev_plate_x"] / (PLATE_WIDTH / 2)
    df["prev_pz"] = (df["prev_plate_z"] - df["prev_sz_bot"]) / (df["prev_sz_top"] - df["prev_sz_bot"])
    df.drop(columns=["prev_plate_x", "prev_plate_z", "prev_sz_top", "prev_sz_bot"], inplace=True)


chunks = []
# date_ranges = [("2025-03-10", "2025-03-31")]
date_ranges = []
for year in range(2024, 2026):
    date_ranges.extend([
        (f"{year}-03-01", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-15"),
    ])

cols = [
    "batter",
    "game_year",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "p_throws",
    "stand",
    "inning",
    "inning_topbot",
    "bat_score",
    "fld_score",
    "outs_when_up",
    "on_3b",
    "on_2b",
    "on_1b",
    "balls",
    "strikes",
    "pitch_type",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "type",
    "des",
]
required = [col for col in cols if col not in ["on_3b", "on_2b", "on_1b"]]

for start_date, end_date in tqdm(date_ranges):
    print(f"Fetching {start_date} to {end_date}...")
    chunk = statcast(start_date, end_date)[cols].dropna(subset=required)
    chunks.append(chunk)

all_pitches = pd.concat(chunks)

all_pitches["on_3b"] = (~all_pitches["on_3b"].isna()).astype(bool)
all_pitches["on_2b"] = (~all_pitches["on_2b"].isna()).astype(bool)
all_pitches["on_1b"] = (~all_pitches["on_1b"].isna()).astype(bool)

all_pitches["inning_topbot"] = all_pitches["inning_topbot"].str.lower()
all_pitches["pitch_type"] = all_pitches["pitch_type"].str.lower()

# Sort into pitch sequence order so shift gives the previous pitch within each at-bat
all_pitches = all_pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])

# Compute previous-pitch features (NaN for the first pitch of each at-bat)
grp = all_pitches.groupby(["game_pk", "at_bat_number"])
all_pitches["prev_pitch_type"] = grp["pitch_type"].shift(1)
all_pitches["prev_plate_x"]    = grp["plate_x"].shift(1)
all_pitches["prev_plate_z"]    = grp["plate_z"].shift(1)
all_pitches["prev_sz_top"]     = grp["sz_top"].shift(1)
all_pitches["prev_sz_bot"]     = grp["sz_bot"].shift(1)
all_pitches["prev_type"]       = grp["type"].shift(1)

all_pitches.drop(columns=["type"], inplace=True)

# Fetch previous-year K% and BB% for each batter
prev_stats_frames = []
for year in [2024, 2025]:
    stats = batting_stats(year - 1, qual=0)[["IDfg", "K%", "BB%"]].copy()
    stats = stats.rename(columns={"IDfg": "key_fangraphs"})
    stats["game_year"] = year
    prev_stats_frames.append(stats)
all_prev_stats = pd.concat(prev_stats_frames)

# Map MLBAM batter IDs to FanGraphs IDs, then join K%/BB%
unique_batters = all_pitches["batter"].dropna().unique().astype(int).tolist()
id_map = playerid_reverse_lookup(unique_batters, key_type="mlbam")[["key_mlbam", "key_fangraphs"]]

all_pitches = all_pitches.merge(id_map, left_on="batter", right_on="key_mlbam", how="left")
all_pitches = all_pitches.merge(all_prev_stats, on=["key_fangraphs", "game_year"], how="left")
all_pitches.drop(columns=["batter", "game_year", "key_mlbam", "key_fangraphs", "game_pk", "at_bat_number", "pitch_number"], inplace=True)

all_pitches = all_pitches.dropna(subset=["K%", "BB%"])

def row_is_swing(row: pd.Series) -> bool:
    if row["type"] == "X":
        return True
    if row["type"] == "B":
        return False

    if "swing" in row["des"].lower() or "foul" in row["des"].lower():
        return True
    return False

all_pitches["swing"] = all_pitches.apply(row_is_swing, axis=1).astype(bool)
all_pitches.drop(columns=["des"], inplace=True)

normalise_pitch_coords(all_pitches)

all_pitches.to_csv("swing_data.csv", index=False)

print(f"Total pitches: {len(all_pitches)}")
print(all_pitches.head())
