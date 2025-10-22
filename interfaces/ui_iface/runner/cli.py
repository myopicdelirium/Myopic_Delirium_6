import os, json, typer, yaml, hashlib
from jsonschema import validate
from ..schemas.schema import get_schema
app = typer.Typer(add_completion=False)
def scenario_defaults() -> dict:
    return {
        "world": {"type": "grid", "width": 256, "height": 256, "wrap": {"x": True, "y": True}, "ticks_per_day": 1440},
        "randomness": {"seed": 1337, "partitions": {"terrain_elevation": 1, "precipitation": 2, "river_routing": 3, "vegetation_seed": 4, "kernel_noise": 5}},
        "fields": [
            {"name": "temperature", "bounds": [0.0, 1.0], "coeffs": {"diffusion": 0.18, "advection": {"vx": 0.0, "vy": 0.0}, "decay": 0.0, "replenish": 0.0}, "derived": False},
            {"name": "hydration", "bounds": [0.0, 1.0], "coeffs": {"diffusion": 0.12, "advection": {"vx": 0.0, "vy": 0.0}, "decay": 0.0, "replenish": 0.0}, "derived": False},
            {"name": "vegetation", "bounds": [0.0, 1.0], "coeffs": {"diffusion": 0.05, "advection": {"vx": 0.0, "vy": 0.0}, "decay": 0.0, "replenish": 0.0}, "derived": False},
            {"name": "movement_cost", "bounds": [0.0, 1.0], "coeffs": {"diffusion": 0.0, "advection": {"vx": 0.0, "vy": 0.0}, "decay": 0.0, "replenish": 0.0}, "derived": True}
        ],
        "dynamics": {"boundary": "wrap", "passes": {"diffusion": True, "advection": True, "coupling": True, "decay": True, "replenishment": True, "derived": True, "metrics": True}},
        "outputs": {"metrics_cadence": 1, "deltas_cadence": 1, "snapshots_cadence": 0},
        "heat_profile": {"direction": "north_hot", "amplitude": 0.6, "noise_amp": 0.05},
        "water_profile": {"elevation_scale": 96, "octaves": 4, "ridge_strength": 0.4, "precipitation_scale": 128, "lake_fill_threshold": 0.15, "river_percentile": 0.92, "river_incision": 0.02, "river_decay_radius": 6},
        "vegetation_profile": {"k": 0.08, "water_half": 0.35, "heat_optimum": 0.65, "heat_sigma": 0.18, "carrying_capacity": 1.0}
    }
@app.command()
def init():
    d = scenario_defaults()
    out = os.path.join("interfaces", "ui_iface", "scenarios")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "env-b.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(d, f, sort_keys=True)
    typer.echo(p)
@app.command()
def validate_scenario(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    validate(cfg, get_schema())
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
    typer.echo(h)
@app.command()
def run(path: str, ticks: int = 256, out: str = "runs", label: str = None):
    from .engine import load_scenario, run_headless
    cfg = load_scenario(path)
    rd = run_headless(cfg, ticks, out, label)
    typer.echo(os.path.abspath(rd))
@app.command()
def inspect(run_dir: str):
    import pandas as pd
    mp = os.path.join(run_dir, "manifest.json")
    with open(mp, "r") as f:
        m = json.load(f)
    typer.echo(json.dumps({"label": m.get("label"), "ticks": m.get("ticks"), "runtime_s": m.get("runtime_s")}, separators=(",", ":"), sort_keys=True))
    fp = os.path.join(run_dir, "metrics", "field_stats.parquet")
    df = pd.read_parquet(fp)
    tail = df.sort_values("tick").tail(5)
    typer.echo(tail.to_string(index=False))
@app.command()
def visualize(run_dir: str, field: str = "hydration", plot_type: str = "field", tick: int = 0, save: str = None):
    from .viz import plot_field, plot_hydrology, plot_metrics_timeseries, create_animation
    from .hydrator import hydrate_tick, get_field_index, get_field_names
    
    if plot_type == "field":
        try:
            tensor = hydrate_tick(run_dir, tick)
            field_idx = get_field_index(run_dir, field)
            plot_field(tensor, field_idx, field, title=f"{field.title()} at Tick {tick}", save_path=save)
        except ValueError as e:
            field_names = get_field_names(run_dir)
            typer.echo(f"Error: {e}")
            typer.echo(f"Available fields: {field_names}")
    elif plot_type == "hydrology":
        plot_hydrology(run_dir, save_path=save)
    elif plot_type == "metrics":
        plot_metrics_timeseries(run_dir, save_path=save)
    elif plot_type == "animation":
        create_animation(run_dir, field, output_path=save)
    else:
        typer.echo(f"Unknown plot type: {plot_type}. Available: field, hydrology, metrics, animation")
if __name__ == "__main__":
    app()
