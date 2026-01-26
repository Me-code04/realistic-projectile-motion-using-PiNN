import pandas as pd

df = pd.read_csv("statcast_pitches_physics.csv")
cols = ["release_pos_x","release_pos_y","release_pos_z","vx0","vy0","vz0","ax","ay","az","release_spin_rate","spin_axis"]
print(df[cols].head(3))
print("\nSummary:")
print(df[cols].describe().loc[["min","mean","max"]])
