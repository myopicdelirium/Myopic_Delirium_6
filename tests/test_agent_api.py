import pytest
import numpy as np
import os
import tempfile
from interfaces.ui_iface.runner.agent_api import EnvironmentGrid, get_agent_grid
from interfaces.ui_iface.runner.engine import load_scenario, run_headless

@pytest.fixture
def test_run():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_scenario(scenario_path)
        run_dir = run_headless(cfg, ticks=10, out_dir=tmpdir, label="test_agent")
        yield run_dir

def test_environment_grid_initialization(test_run):
    env = EnvironmentGrid(test_run)
    assert env.h == 256
    assert env.w == 256
    assert env.f == 4
    assert env.current_tick == 0
    assert env.tensor is None

def test_load_tick(test_run):
    env = EnvironmentGrid(test_run)
    tensor = env.load_tick(0)
    assert tensor is not None
    assert tensor.shape == (256, 256, 4)
    assert env.current_tick == 0
    assert np.all(tensor >= 0.0)
    assert np.all(tensor <= 1.0)

def test_get_field(test_run):
    env = EnvironmentGrid(test_run)
    env.load_tick(0)
    
    temp = env.get_field("temperature")
    assert temp.shape == (256, 256)
    assert 0.0 <= temp.min() <= 1.0
    assert 0.0 <= temp.max() <= 1.0
    
    hydration = env.get_field("hydration")
    assert hydration.shape == (256, 256)
    assert hydration.mean() > 0.5

def test_get_cell(test_run):
    env = EnvironmentGrid(test_run)
    env.load_tick(0)
    
    temp_value = env.get_cell(128, 128, "temperature")
    assert isinstance(temp_value, float)
    assert 0.0 <= temp_value <= 1.0

def test_get_all_fields_at(test_run):
    env = EnvironmentGrid(test_run)
    env.load_tick(0)
    
    fields = env.get_all_fields_at(100, 100)
    assert isinstance(fields, dict)
    assert "temperature" in fields
    assert "hydration" in fields
    assert "vegetation" in fields
    assert "movement_cost" in fields
    assert all(0.0 <= v <= 1.0 for v in fields.values())

def test_get_neighborhood(test_run):
    env = EnvironmentGrid(test_run)
    env.load_tick(0)
    
    neighborhood = env.get_neighborhood(128, 128, radius=2)
    assert isinstance(neighborhood, dict)
    assert "temperature" in neighborhood
    assert neighborhood["temperature"].shape == (5, 5)

def test_get_agent_grid(test_run):
    env = get_agent_grid(test_run, tick=0)
    assert env.tensor is not None
    assert env.current_tick == 0

def test_error_before_load(test_run):
    env = EnvironmentGrid(test_run)
    with pytest.raises(ValueError, match="Call load_tick"):
        env.get_field("temperature")
    with pytest.raises(ValueError, match="Call load_tick"):
        env.get_cell(0, 0, "temperature")

def test_temporal_evolution(test_run):
    env = EnvironmentGrid(test_run)
    env.load_tick(0)
    temp_t0 = env.get_field("temperature").copy()
    
    env.load_tick(10)
    temp_t10 = env.get_field("temperature").copy()
    
    assert not np.array_equal(temp_t0, temp_t10)
    assert np.abs(temp_t0 - temp_t10).mean() < 0.1

def test_field_names_property(test_run):
    env = EnvironmentGrid(test_run)
    names = env.field_names
    assert "temperature" in names
    assert "hydration" in names
    assert "vegetation" in names
    assert "movement_cost" in names

def test_shape_property(test_run):
    env = EnvironmentGrid(test_run)
    assert env.shape == (256, 256, 4)

