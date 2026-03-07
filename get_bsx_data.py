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


chunks = []
# date_ranges = [("2025-03-20", "2025-03-31")]
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
    "p_throws",
    "stand",
    "pitch_type",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "des",
    "type",
]

for start_date, end_date in tqdm(date_ranges):
    print(f"Fetching {start_date} to {end_date}...")
    chunk = statcast(start_date, end_date)[cols].dropna()
    chunks.append(chunk)

all_pitches = pd.concat(chunks)

all_pitches["pitch_type"] = all_pitches["pitch_type"].str.lower()

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

all_pitches.to_csv("bsx_data.csv", index=False)

print(f"Total pitches: {len(all_pitches)}")
print(all_pitches.head())
