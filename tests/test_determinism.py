import pytest
import numpy as np
import tempfile
import os
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick

def test_deterministic_initialization():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        run1 = run_headless(cfg, ticks=0, out_dir=tmpdir1, label="run1")
        tensor1 = hydrate_tick(run1, 0)
    
    with tempfile.TemporaryDirectory() as tmpdir2:
        run2 = run_headless(cfg, ticks=0, out_dir=tmpdir2, label="run2")
        tensor2 = hydrate_tick(run2, 0)
    
    assert np.array_equal(tensor1, tensor2), "Initial tensors must be identical for same seed"

def test_deterministic_simulation():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        run1 = run_headless(cfg, ticks=50, out_dir=tmpdir1, label="run1")
        tensor1_t0 = hydrate_tick(run1, 0)
        tensor1_t50 = hydrate_tick(run1, 50)
    
    with tempfile.TemporaryDirectory() as tmpdir2:
        run2 = run_headless(cfg, ticks=50, out_dir=tmpdir2, label="run2")
        tensor2_t0 = hydrate_tick(run2, 0)
        tensor2_t50 = hydrate_tick(run2, 50)
    
    assert np.array_equal(tensor1_t0, tensor2_t0), "Initial states must match"
    assert np.array_equal(tensor1_t50, tensor2_t50), "Final states must match"

def test_seed_independence():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    
    cfg1 = load_scenario(scenario_path)
    cfg1["randomness"]["seed"] = 1337
    
    cfg2 = load_scenario(scenario_path)
    cfg2["randomness"]["seed"] = 9999
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        run1 = run_headless(cfg1, ticks=0, out_dir=tmpdir1, label="seed1337")
        tensor1 = hydrate_tick(run1, 0)
    
    with tempfile.TemporaryDirectory() as tmpdir2:
        run2 = run_headless(cfg2, ticks=0, out_dir=tmpdir2, label="seed9999")
        tensor2 = hydrate_tick(run2, 0)
    
    assert not np.array_equal(tensor1, tensor2), "Different seeds must produce different tensors"

def test_scenario_hash_consistency():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    cfg1 = load_scenario(scenario_path)
    hash1 = cfg1["_scenario_hash"]
    
    cfg2 = load_scenario(scenario_path)
    hash2 = cfg2["_scenario_hash"]
    
    assert hash1 == hash2, "Scenario hash must be consistent"

def test_numerical_stability():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=100, out_dir=tmpdir, label="stability")
        
        for tick in [0, 25, 50, 75, 100]:
            tensor = hydrate_tick(run_dir, tick)
            assert not np.any(np.isnan(tensor)), f"NaN values detected at tick {tick}"
            assert not np.any(np.isinf(tensor)), f"Inf values detected at tick {tick}"
            assert np.all(tensor >= 0.0), f"Negative values at tick {tick}"
            assert np.all(tensor <= 1.0), f"Values > 1.0 at tick {tick}"

