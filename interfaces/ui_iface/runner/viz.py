import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import pandas as pd
from typing import Dict, Any, List, Optional
from .hydrator import replay_frame
from .registry import build_registry
def create_colormap(field_name: str) -> mcolors.Colormap:
    if field_name == "temperature":
        return plt.cm.RdYlBu_r
    elif field_name == "hydration":
        return plt.cm.Blues
    elif field_name == "vegetation":
        return plt.cm.Greens
    elif field_name == "movement_cost":
        return plt.cm.Reds
    else:
        return plt.cm.viridis
def plot_field(tensor: np.ndarray, field_idx: int, field_name: str, title: str = None, save_path: str = None):
    plt.figure(figsize=(10, 8))
    field_data = tensor[:, :, field_idx]
    cmap = create_colormap(field_name)
    im = plt.imshow(field_data, cmap=cmap, origin='lower')
    plt.colorbar(im, label=field_name)
    plt.title(title or f"{field_name.title()} Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_hydrology(run_dir: str, save_path: str = None):
    with open(os.path.join(run_dir, "scenario.json"), "r") as f:
        import json
        cfg = json.load(f)
    reg = build_registry(cfg)
    h = cfg["world"]["height"]
    w = cfg["world"]["width"]
    tensor = replay_frame(run_dir, 0, h, w, len(reg["names"]))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fields = ["temperature", "hydration", "vegetation", "movement_cost"]
    for i, field in enumerate(fields):
        if field in reg["indices"]:
            ax = axes[i//2, i%2]
            field_idx = reg["indices"][field]
            field_data = tensor[:, :, field_idx]
            cmap = create_colormap(field)
            im = ax.imshow(field_data, cmap=cmap, origin='lower')
            ax.set_title(f"{field.title()}")
            plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_metrics_timeseries(run_dir: str, save_path: str = None):
    df_field = pd.read_parquet(os.path.join(run_dir, "metrics", "field_stats.parquet"))
    df_hydro = pd.read_parquet(os.path.join(run_dir, "metrics", "hydrology.parquet"))
    df_struct = pd.read_parquet(os.path.join(run_dir, "metrics", "structure.parquet"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for field in df_field["field"].unique():
        field_data = df_field[df_field["field"] == field]
        axes[0, 0].plot(field_data["tick"], field_data["mean"], label=field, marker='o', markersize=2)
    axes[0, 0].set_title("Field Means Over Time")
    axes[0, 0].set_xlabel("Tick")
    axes[0, 0].set_ylabel("Mean Value")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(df_hydro["tick"], df_hydro["river_length"], label="River Length", color='blue')
    axes[0, 1].plot(df_hydro["tick"], df_hydro["lake_area"], label="Lake Area", color='cyan')
    axes[0, 1].set_title("Hydrology Metrics")
    axes[0, 1].set_xlabel("Tick")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    for field in df_struct["field"].unique():
        field_data = df_struct[df_struct["field"] == field]
        axes[1, 0].plot(field_data["tick"], field_data["moran_like"], label=field, marker='s', markersize=2)
    axes[1, 0].set_title("Spatial Coherence (Moran's I-like)")
    axes[1, 0].set_xlabel("Tick")
    axes[1, 0].set_ylabel("Coherence")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    for field in df_field["field"].unique():
        field_data = df_field[df_field["field"] == field]
        axes[1, 1].plot(field_data["tick"], field_data["var"], label=field, marker='^', markersize=2)
    axes[1, 1].set_title("Field Variance Over Time")
    axes[1, 1].set_xlabel("Tick")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def create_animation(run_dir: str, field_name: str, output_path: str = None, max_frames: int = 100):
    with open(os.path.join(run_dir, "scenario.json"), "r") as f:
        import json
        cfg = json.load(f)
    reg = build_registry(cfg)
    h = cfg["world"]["height"]
    w = cfg["world"]["width"]
    f = len(reg["names"])
    if field_name not in reg["indices"]:
        print(f"Field {field_name} not found in registry")
        return
    field_idx = reg["indices"][field_name]
    df_deltas = pd.read_parquet(os.path.join(run_dir, "grid", "deltas.parquet"))
    max_tick = min(df_deltas["tick"].max(), max_frames - 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = create_colormap(field_name)
    tensor = replay_frame(run_dir, 0, h, w, f)
    im = ax.imshow(tensor[:, :, field_idx], cmap=cmap, origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, label=field_name)
    title = ax.set_title(f"{field_name.title()} - Tick 0")
    def animate(frame):
        tensor = replay_frame(run_dir, frame, h, w, f)
        im.set_array(tensor[:, :, field_idx])
        title.set_text(f"{field_name.title()} - Tick {frame}")
        return im, title
    anim = FuncAnimation(fig, animate, frames=max_tick+1, interval=100, blit=False, repeat=True)
    if output_path:
        anim.save(output_path, writer='pillow', fps=10)
    plt.show()
    return anim
