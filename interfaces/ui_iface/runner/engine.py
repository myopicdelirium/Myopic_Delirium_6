import os, json, time, hashlib, yaml, numpy as np, pandas as pd
from typing import Any, Dict
from jsonschema import validate
from blake3 import blake3
from .registry import build_registry
from . import initgen
from .kernels import step_kernels
from ..schemas.schema import get_schema
def load_scenario(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    validate(cfg, get_schema())
    cfg = apply_defaults(cfg)
    scenario_hash = stable_hash(cfg)
    cfg["_scenario_hash"] = scenario_hash
    return cfg
def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    w = cfg["world"]
    if "wrap" not in w:
        w["wrap"] = {"x": True, "y": True}
    if "ticks_per_day" not in w:
        w["ticks_per_day"] = 1440
    out = cfg.get("outputs", {})
    out.setdefault("metrics_cadence", 1)
    out.setdefault("deltas_cadence", 1)
    out.setdefault("snapshots_cadence", 0)
    cfg["outputs"] = out
    dyn = cfg.get("dynamics", {})
    passes = dyn.get("passes", {})
    for k in ["diffusion", "advection", "coupling", "decay", "replenishment", "derived", "metrics"]:
        passes.setdefault(k, True)
    dyn["passes"] = passes
    dyn.setdefault("boundary", "wrap")
    cfg["dynamics"] = dyn
    return cfg
def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
def build_seed_partitions(base_seed: int, partitions: Dict[str, int]) -> Dict[str, np.random.Generator]:
    seeds = {}
    for k, off in partitions.items():
        seeds[k] = np.random.default_rng(int(base_seed + off))
    return seeds
def assemble_initial_tensor(cfg: Dict[str, Any], seeds: Dict[str, np.random.Generator], registry: Dict[str, Any]) -> Dict[str, Any]:
    h = int(cfg["world"]["height"])
    w = int(cfg["world"]["width"])
    ep = cfg["water_profile"]
    prp = cfg["water_profile"]
    lp = cfg["water_profile"]
    hp = cfg["heat_profile"]
    vp = cfg["vegetation_profile"]
    e_seed = int(cfg["randomness"]["seed"] + cfg["randomness"]["partitions"]["terrain_elevation"])
    p_seed = int(cfg["randomness"]["seed"] + cfg["randomness"]["partitions"]["precipitation"])
    r_seed = int(cfg["randomness"]["seed"] + cfg["randomness"]["partitions"]["river_routing"])
    v_seed = int(cfg["randomness"]["seed"] + cfg["randomness"]["partitions"]["vegetation_seed"])
    t_seed = int(cfg["randomness"]["seed"] + cfg["randomness"]["partitions"]["kernel_noise"])
    E = initgen.elevation(h, w, ep, e_seed)
    P = initgen.precipitation(h, w, prp, p_seed, E)
    A, closed = initgen.flow_accumulation(E)
    lake_mask, Efill = initgen.lakes(E, A, float(lp.get("lake_fill_threshold", 0.15)))
    H2O = initgen.hydration_from_hydrology(Efill, A, lake_mask, cfg["water_profile"])
    T = initgen.temperature_meridional(h, w, hp, t_seed)
    V0 = initgen.vegetation_init(H2O, T, vp, v_seed)
    f = len(registry["names"])
    tensor = np.zeros((h, w, f), dtype=np.float32)
    for name, idx in registry["indices"].items():
        if name == "temperature":
            tensor[:, :, idx] = T
        elif name == "hydration":
            tensor[:, :, idx] = H2O
        elif name == "vegetation":
            tensor[:, :, idx] = V0
        else:
            tensor[:, :, idx] = 0.0
    for i, (lo, hi) in enumerate(registry["bounds"]):
        tensor[:, :, i] = np.clip(tensor[:, :, i], lo, hi)
    aux = {"E": Efill, "P": P, "A": A, "lake_mask": lake_mask}
    return {"tensor": tensor, "aux": aux}
def write_checksums(run_dir: str, files: list[str]):
    os.makedirs(os.path.join(run_dir, "checksums"), exist_ok=True)
    for fp in files:
        with open(fp, "rb") as f:
            h = blake3()
            while True:
                b = f.read(1048576)
                if not b:
                    break
                h.update(b)
        out = os.path.join(run_dir, "checksums", os.path.basename(fp) + ".blake3")
        with open(out, "w") as o:
            o.write(h.hexdigest())
def metrics_spatial_coherence(arr: np.ndarray) -> float:
    h, w = arr.shape
    m = arr.mean()
    v = arr.var() + 1e-8
    xn = np.roll(arr, 1, axis=1)
    xp = np.roll(arr, -1, axis=1)
    yn = np.roll(arr, 1, axis=0)
    yp = np.roll(arr, -1, axis=0)
    c = ((arr - m) * (xn - m) + (arr - m) * (xp - m) + (arr - m) * (yn - m) + (arr - m) * (yp - m)) / (4.0 * (h * w))
    return float(c.mean() / v)
def run_headless(cfg: Dict[str, Any], ticks: int, out_dir: str, label: str | None = None) -> str:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    reg = build_registry(cfg)
    assembled = assemble_initial_tensor(cfg, {}, reg)
    tensor = assembled["tensor"]
    E = assembled["aux"]["E"]
    A = assembled["aux"]["A"]
    lake_mask = assembled["aux"]["lake_mask"]
    run_label = label or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_dir, f"run-{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "grid"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "streams"), exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "scenario_hash": cfg["_scenario_hash"],
        "seed_partitions": cfg["randomness"]["partitions"],
        "created": int(time.time()),
        "ticks": int(ticks),
        "world": cfg["world"],
        "label": run_label
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"), sort_keys=True)
    with open(os.path.join(run_dir, "scenario.json"), "w") as f:
        json.dump(cfg, f, separators=(",", ":"), sort_keys=True)
    deltas_rows = []
    metrics_field_rows = []
    metrics_hydro_rows = []
    metrics_struct_rows = []
    names = reg["names"]
    derived = reg["derived"]
    wrapx = bool(cfg["world"]["wrap"]["x"])
    wrapy = bool(cfg["world"]["wrap"]["y"])
    mg = np.random.default_rng(int(cfg["randomness"]["seed"] + cfg["randomness"]["partitions"]["kernel_noise"]))
    for t in range(ticks):
        new_tensor = step_kernels(tensor, cfg, reg, wrapx, wrapy, mg)
        delta = new_tensor - tensor
        for i, name in enumerate(names):
            if derived[i]:
                continue
            idxs = np.where(np.abs(delta[:, :, i]) > 1e-8)
            for y, x in zip(idxs[0], idxs[1]):
                deltas_rows.append((t, int(x), int(y), int(i), float(delta[y, x, i])))
        tensor = new_tensor
        if (t + 1) % int(cfg["outputs"]["metrics_cadence"]) == 0:
            for i, name in enumerate(names):
                if derived[i]:
                    continue
                arr = tensor[:, :, i]
                metrics_field_rows.append((t, name, float(arr.mean()), float(arr.var())))
            river_len = int((A >= np.percentile(A, 100.0 * (1.0 - float(cfg["water_profile"]["river_percentile"])))).sum())
            lake_area = int(lake_mask.sum())
            thr = float(cfg["water_profile"]["river_percentile"])
            metrics_hydro_rows.append((t, river_len, lake_area, thr))
            for i, name in enumerate(names):
                if derived[i]:
                    continue
                mcoh = metrics_spatial_coherence(tensor[:, :, i])
                metrics_struct_rows.append((t, name, float(mcoh)))
        with open(os.path.join(run_dir, "streams", "events.ndjson"), "a") as s:
            s.write(json.dumps({"tick": t, "mean": {names[i]: float(tensor[:, :, i].mean()) for i in range(len(names)) if not derived[i]}}) + "\n")
    if len(deltas_rows) > 0:
        df = pd.DataFrame(deltas_rows, columns=["tick", "x", "y", "field_id", "delta"])
        df.to_parquet(os.path.join(run_dir, "grid", "deltas.parquet"), index=False)
    dfm = pd.DataFrame(metrics_field_rows, columns=["tick", "field", "mean", "var"])
    dfm.to_parquet(os.path.join(run_dir, "metrics", "field_stats.parquet"), index=False)
    dfh = pd.DataFrame(metrics_hydro_rows, columns=["tick", "river_length", "lake_area", "flow_thresholds"])
    dfh.to_parquet(os.path.join(run_dir, "metrics", "hydrology.parquet"), index=False)
    dfs = pd.DataFrame(metrics_struct_rows, columns=["tick", "field", "moran_like"])
    dfs.to_parquet(os.path.join(run_dir, "metrics", "structure.parquet"), index=False)
    files = [
        os.path.join(run_dir, "manifest.json"),
        os.path.join(run_dir, "scenario.json"),
        os.path.join(run_dir, "grid", "deltas.parquet"),
        os.path.join(run_dir, "metrics", "field_stats.parquet"),
        os.path.join(run_dir, "metrics", "hydrology.parquet"),
        os.path.join(run_dir, "metrics", "structure.parquet"),
        os.path.join(run_dir, "streams", "events.ndjson"),
    ]
    files = [fp for fp in files if os.path.exists(fp)]
    write_checksums(run_dir, files)
    dt = time.time() - t0
    manifest["runtime_s"] = dt
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"), sort_keys=True)
    return run_dir
