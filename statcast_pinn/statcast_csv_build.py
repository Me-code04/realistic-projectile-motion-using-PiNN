import pandas as pd
from pybaseball import statcast

# 1) Pull raw Statcast pitches for a date range
df = statcast(start_dt="2025-06-01", end_dt="2025-06-30")  # change dates
# pybaseball returns a big dataframe of pitch-level data. :contentReference[oaicite:6]{index=6}

# 2) Keep only the columns we care about for physics
cols = [
    "game_date", "player_name", "pitch_type",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
    "release_spin_rate", "spin_axis"
]
keep = [c for c in cols if c in df.columns]
df_phys = df.dropna(subset=[c for c in keep if c not in ["spin_axis"]])[keep].copy()

# 3) Save
df_phys.to_csv("statcast_pitches_physics.csv", index=False)
print("Saved:", len(df_phys), "rows to statcast_pitches_physics.csv")
