import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

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

def hydrate_tick(run_dir: str, tick: int) -> np.ndarray:
    scenario_path = os.path.join(run_dir, "scenario.json")
    with open(scenario_path, "r") as f:
        cfg = json.load(f)
    
    h = cfg["world"]["height"]
    w = cfg["world"]["width"]
    num_fields = len(cfg["fields"])
    
    from .engine import build_seed_partitions, assemble_initial_tensor
    from .registry import build_registry
    
    seeds = build_seed_partitions(cfg["randomness"]["seed"], cfg["randomness"]["partitions"])
    registry = build_registry(cfg)
    result = assemble_initial_tensor(cfg, seeds, registry)
    initial_tensor = result["tensor"]
    
    deltas_path = os.path.join(run_dir, "grid", "deltas.parquet")
    if os.path.exists(deltas_path) and tick > 0:
        df = pd.read_parquet(deltas_path)
        df = df[df["tick"] <= tick]
        for _, row in df.iterrows():
            y = int(row["y"])
            x = int(row["x"])
            field_id = int(row["field_id"])
            delta = float(row["delta"])
            initial_tensor[y, x, field_id] += delta
    
    for i in range(num_fields):
        lo, hi = registry["bounds"][i]
        initial_tensor[:, :, i] = np.clip(initial_tensor[:, :, i], lo, hi)
    
    return initial_tensor

def get_field_names(run_dir: str) -> list[str]:
    scenario_path = os.path.join(run_dir, "scenario.json")
    with open(scenario_path, "r") as f:
        cfg = json.load(f)
    return [field["name"] for field in cfg["fields"]]

def get_field_index(run_dir: str, field_name: str) -> int:
    scenario_path = os.path.join(run_dir, "scenario.json")
    with open(scenario_path, "r") as f:
        cfg = json.load(f)
    for i, field in enumerate(cfg["fields"]):
        if field["name"] == field_name:
            return i
    raise ValueError(f"Field '{field_name}' not found")

def get_tick_range(run_dir: str) -> tuple[int, int]:
    deltas_path = os.path.join(run_dir, "grid", "deltas.parquet")
    if not os.path.exists(deltas_path):
        return (0, 0)
    df = pd.read_parquet(deltas_path)
    if len(df) == 0:
        return (0, 0)
    return (0, int(df["tick"].max()))
