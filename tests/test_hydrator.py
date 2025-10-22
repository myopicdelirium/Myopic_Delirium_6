import pytest
import numpy as np
import tempfile
from interfaces.ui_iface.runner.engine import load_scenario, run_headless, assemble_initial_tensor, build_seed_partitions
from interfaces.ui_iface.runner.hydrator import hydrate_tick, get_field_names, get_field_index, get_tick_range
from interfaces.ui_iface.runner.registry import build_registry

@pytest.fixture
def test_run():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_scenario(scenario_path)
        run_dir = run_headless(cfg, ticks=25, out_dir=tmpdir, label="hydrator")
        yield run_dir, cfg

def test_hydrate_tick_zero(test_run):
    run_dir, cfg = test_run
    tensor = hydrate_tick(run_dir, 0)
    
    assert tensor is not None
    assert tensor.shape == (256, 256, 4)
    assert tensor.dtype == np.float32
    assert np.all(tensor >= 0.0)
    assert np.all(tensor <= 1.0)

def test_hydrate_matches_initial(test_run):
    run_dir, cfg = test_run
    
    hydrated = hydrate_tick(run_dir, 0)
    
    seeds = build_seed_partitions(cfg["randomness"]["seed"], cfg["randomness"]["partitions"])
    registry = build_registry(cfg)
    result = assemble_initial_tensor(cfg, seeds, registry)
    initial = result["tensor"]
    
    assert np.allclose(hydrated, initial, atol=1e-6), "Hydrated tick 0 must match initial tensor"

def test_hydrate_temporal_sequence(test_run):
    run_dir, cfg = test_run
    
    t0 = hydrate_tick(run_dir, 0)
    t10 = hydrate_tick(run_dir, 10)
    t25 = hydrate_tick(run_dir, 25)
    
    assert not np.array_equal(t0, t10), "State must change over time"
    assert not np.array_equal(t10, t25), "State must continue changing"

def test_get_field_names(test_run):
    run_dir, cfg = test_run
    names = get_field_names(run_dir)
    
    assert isinstance(names, list)
    assert "temperature" in names
    assert "hydration" in names
    assert "vegetation" in names
    assert "movement_cost" in names
    assert len(names) == 4

def test_get_field_index(test_run):
    run_dir, cfg = test_run
    
    temp_idx = get_field_index(run_dir, "temperature")
    assert isinstance(temp_idx, int)
    assert 0 <= temp_idx < 4
    
    hydration_idx = get_field_index(run_dir, "hydration")
    assert isinstance(hydration_idx, int)
    assert temp_idx != hydration_idx

def test_get_field_index_invalid(test_run):
    run_dir, cfg = test_run
    
    with pytest.raises(ValueError, match="not found"):
        get_field_index(run_dir, "nonexistent_field")

def test_get_tick_range(test_run):
    run_dir, cfg = test_run
    
    min_tick, max_tick = get_tick_range(run_dir)
    assert min_tick == 0
    assert max_tick == 25

def test_hydrator_bounds_preservation(test_run):
    run_dir, cfg = test_run
    
    for tick in [0, 10, 25]:
        tensor = hydrate_tick(run_dir, tick)
        
        for field_name in ["temperature", "hydration", "vegetation", "movement_cost"]:
            field_idx = get_field_index(run_dir, field_name)
            field_data = tensor[:, :, field_idx]
            
            assert np.all(field_data >= 0.0), f"{field_name} at tick {tick} has values < 0"
            assert np.all(field_data <= 1.0), f"{field_name} at tick {tick} has values > 1"

def test_hydrator_no_nan_inf(test_run):
    run_dir, cfg = test_run
    
    for tick in [0, 10, 25]:
        tensor = hydrate_tick(run_dir, tick)
        
        assert not np.any(np.isnan(tensor)), f"NaN detected at tick {tick}"
        assert not np.any(np.isinf(tensor)), f"Inf detected at tick {tick}"

def test_hydrator_deterministic_reload(test_run):
    run_dir, cfg = test_run
    
    tensor1 = hydrate_tick(run_dir, 15)
    tensor2 = hydrate_tick(run_dir, 15)
    
    assert np.array_equal(tensor1, tensor2), "Reloading same tick must give identical result"

def test_hydrator_evolution_bounded(test_run):
    run_dir, cfg = test_run
    
    t0 = hydrate_tick(run_dir, 0)
    t25 = hydrate_tick(run_dir, 25)
    
    for field_idx in range(4):
        diff = np.abs(t25[:, :, field_idx] - t0[:, :, field_idx])
        max_change = diff.max()
        
        assert max_change < 0.5, f"Field {field_idx} changed too much (max: {max_change})"

