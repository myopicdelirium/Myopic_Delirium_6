import os
import numpy as np
import pandas as pd
def replay_frame(run_dir: str, t: int, h: int, w: int, f: int):
    tensor = np.zeros((h, w, f), dtype=np.float32)
    p = os.path.join(run_dir, "grid", "deltas.parquet")
    if not os.path.exists(p):
        return tensor
    df = pd.read_parquet(p)
    df = df[df["tick"] <= t]
    for _, row in df.iterrows():
        tensor[int(row["y"]), int(row["x"]), int(row["field_id"])] += float(row["delta"])
    return tensor
