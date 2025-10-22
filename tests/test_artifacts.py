import pytest
import os
import json
import tempfile
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
import pandas as pd

@pytest.fixture
def test_run():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_scenario(scenario_path)
        run_dir = run_headless(cfg, ticks=50, out_dir=tmpdir, label="artifacts")
        yield run_dir

def test_manifest_exists(test_run):
    manifest_path = os.path.join(test_run, "manifest.json")
    assert os.path.exists(manifest_path), "manifest.json must exist"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    assert "label" in manifest
    assert "ticks" in manifest
    assert "runtime_s" in manifest
    assert manifest["ticks"] == 50

def test_scenario_saved(test_run):
    scenario_path = os.path.join(test_run, "scenario.json")
    assert os.path.exists(scenario_path), "scenario.json must exist"
    
    with open(scenario_path, "r") as f:
        cfg = json.load(f)
    
    assert "_scenario_hash" in cfg
    assert "world" in cfg
    assert "randomness" in cfg
    assert "fields" in cfg

def test_deltas_parquet(test_run):
    deltas_path = os.path.join(test_run, "grid", "deltas.parquet")
    assert os.path.exists(deltas_path), "deltas.parquet must exist"
    
    df = pd.read_parquet(deltas_path)
    assert "tick" in df.columns
    assert "y" in df.columns
    assert "x" in df.columns
    assert "field_id" in df.columns
    assert "delta" in df.columns
    assert len(df) > 0
    assert df["tick"].max() <= 50

def test_field_stats(test_run):
    stats_path = os.path.join(test_run, "metrics", "field_stats.parquet")
    assert os.path.exists(stats_path), "field_stats.parquet must exist"
    
    df = pd.read_parquet(stats_path)
    assert "tick" in df.columns
    assert "field" in df.columns
    assert "mean" in df.columns
    assert "var" in df.columns
    assert "min" in df.columns
    assert "max" in df.columns
    assert len(df) > 0

def test_hydrology_metrics(test_run):
    hydro_path = os.path.join(test_run, "metrics", "hydrology.parquet")
    assert os.path.exists(hydro_path), "hydrology.parquet must exist"
    
    df = pd.read_parquet(hydro_path)
    assert "tick" in df.columns
    assert "river_length" in df.columns
    assert "lake_area" in df.columns

def test_structure_metrics(test_run):
    struct_path = os.path.join(test_run, "metrics", "structure.parquet")
    assert os.path.exists(struct_path), "structure.parquet must exist"
    
    df = pd.read_parquet(struct_path)
    assert "tick" in df.columns
    assert "field" in df.columns
    assert "moran_like" in df.columns

def test_events_log(test_run):
    events_path = os.path.join(test_run, "streams", "events.ndjson")
    assert os.path.exists(events_path), "events.ndjson must exist"
    
    with open(events_path, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 2

def test_checksums_directory(test_run):
    checksums_dir = os.path.join(test_run, "checksums")
    assert os.path.exists(checksums_dir), "checksums directory must exist"
    
    checksum_files = os.listdir(checksums_dir)
    assert len(checksum_files) > 0, "Checksums must be generated"
    
    for f in checksum_files:
        assert f.endswith(".blake3"), "Checksum files must have .blake3 extension"

def test_checksum_validity(test_run):
    from blake3 import blake3
    
    scenario_path = os.path.join(test_run, "scenario.json")
    checksum_path = os.path.join(test_run, "checksums", "scenario.json.blake3")
    
    if os.path.exists(checksum_path):
        with open(scenario_path, "rb") as f:
            h = blake3()
            h.update(f.read())
            computed_hash = h.hexdigest()
        
        with open(checksum_path, "r") as f:
            stored_hash = f.read().strip()
        
        assert computed_hash == stored_hash, "Checksum must match file content"

def test_metrics_temporal_coverage(test_run):
    stats_path = os.path.join(test_run, "metrics", "field_stats.parquet")
    df = pd.read_parquet(stats_path)
    
    ticks_present = sorted(df["tick"].unique())
    assert 0 in ticks_present, "Tick 0 must have metrics"
    assert 50 in ticks_present, "Final tick must have metrics"
    assert len(ticks_present) >= 50, "All ticks should have metrics"

