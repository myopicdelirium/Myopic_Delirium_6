import pytest
import numpy as np
import tempfile
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick, get_field_index

@pytest.fixture
def test_run():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_scenario(scenario_path)
        run_dir = run_headless(cfg, ticks=0, out_dir=tmpdir, label="quality")
        yield run_dir

def test_temperature_equator_hot(test_run):
    tensor = hydrate_tick(test_run, 0)
    temp_idx = get_field_index(test_run, "temperature")
    temp = tensor[:, :, temp_idx]
    
    north_temp = temp[0:20, :].mean()
    equator_temp = temp[118:138, :].mean()
    south_temp = temp[-20:, :].mean()
    
    assert equator_temp > north_temp, "Equator must be hotter than north pole"
    assert equator_temp > south_temp, "Equator must be hotter than south pole"
    assert abs(north_temp - south_temp) < 0.1, "Poles must have similar temperature (symmetric)"
    assert equator_temp > 0.6, "Equator temperature must be substantial"
    assert north_temp < 0.4, "Pole temperature must be low"

def test_hydration_distribution(test_run):
    tensor = hydrate_tick(test_run, 0)
    hydration_idx = get_field_index(test_run, "hydration")
    hydration = tensor[:, :, hydration_idx]
    
    assert hydration.min() >= 0.0, "Hydration must be non-negative"
    assert hydration.max() <= 1.0, "Hydration must be <= 1.0"
    assert hydration.mean() > 0.5, "Mean hydration should be substantial"
    high_hydration = (hydration > 0.8).sum() / hydration.size
    assert high_hydration > 0.5, "Majority of cells should have high hydration"

def test_vegetation_temperature_correlation(test_run):
    tensor = hydrate_tick(test_run, 0)
    temp_idx = get_field_index(test_run, "temperature")
    veg_idx = get_field_index(test_run, "vegetation")
    
    temp = tensor[:, :, temp_idx].flatten()
    veg = tensor[:, :, veg_idx].flatten()
    
    correlation = np.corrcoef(temp, veg)[0, 1]
    assert correlation > 0.3, "Vegetation must correlate positively with temperature"

def test_vegetation_hydration_correlation(test_run):
    tensor = hydrate_tick(test_run, 0)
    hydration_idx = get_field_index(test_run, "hydration")
    veg_idx = get_field_index(test_run, "vegetation")
    
    hydration = tensor[:, :, hydration_idx].flatten()
    veg = tensor[:, :, veg_idx].flatten()
    
    correlation = np.corrcoef(hydration, veg)[0, 1]
    assert correlation > 0.0, "Vegetation must correlate positively with hydration"

def test_field_bounds(test_run):
    tensor = hydrate_tick(test_run, 0)
    
    for field_name in ["temperature", "hydration", "vegetation", "movement_cost"]:
        field_idx = get_field_index(test_run, field_name)
        field_data = tensor[:, :, field_idx]
        
        assert np.all(field_data >= 0.0), f"{field_name} has values < 0"
        assert np.all(field_data <= 1.0), f"{field_name} has values > 1"

def test_spatial_coherence(test_run):
    tensor = hydrate_tick(test_run, 0)
    
    for field_name in ["temperature", "hydration", "vegetation"]:
        field_idx = get_field_index(test_run, field_name)
        field_data = tensor[:, :, field_idx]
        
        neighbors = (
            np.roll(field_data, 1, axis=0) +
            np.roll(field_data, -1, axis=0) +
            np.roll(field_data, 1, axis=1) +
            np.roll(field_data, -1, axis=1)
        ) / 4.0
        
        diff = np.abs(field_data - neighbors)
        mean_diff = diff.mean()
        
        assert mean_diff < 0.3, f"{field_name} lacks spatial coherence (mean diff: {mean_diff})"

def test_temperature_variance(test_run):
    tensor = hydrate_tick(test_run, 0)
    temp_idx = get_field_index(test_run, "temperature")
    temp = tensor[:, :, temp_idx]
    
    assert temp.std() > 0.15, "Temperature must have sufficient variance"
    assert temp.std() < 0.35, "Temperature variance should not be excessive"

def test_vegetation_range(test_run):
    tensor = hydrate_tick(test_run, 0)
    veg_idx = get_field_index(test_run, "vegetation")
    veg = tensor[:, :, veg_idx]
    
    assert veg.max() > 0.3, "Maximum vegetation should be substantial"
    assert veg.min() < 0.3, "Minimum vegetation should allow for low-vegetation areas"
    range_val = veg.max() - veg.min()
    assert range_val > 0.3, "Vegetation should have sufficient dynamic range"

