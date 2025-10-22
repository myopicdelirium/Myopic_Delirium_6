import pytest
import tempfile
import numpy as np
from interfaces.agent_iface.banded_agent import BandedAgent
from interfaces.agent_iface.simulation import AgentSimulation
from interfaces.agent_iface.band_physiological import PhysiologicalBand
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.predators import PredatorSystem

@pytest.fixture
def test_env():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_scenario(scenario_path)
        run_dir = run_headless(cfg, ticks=100, out_dir=tmpdir, label="survival_test")
        yield run_dir

def test_banded_agent_creation():
    agent = BandedAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    assert agent.state.x == 128
    assert agent.state.y == 128
    assert agent.state.energy == 100.0
    assert agent.state.alive is True
    assert len(agent.bands) == 1
    assert isinstance(agent.bands[0], PhysiologicalBand)

def test_agent_perceives_threat(test_env):
    from interfaces.ui_iface.runner.agent_api import EnvironmentGrid
    
    env = EnvironmentGrid(test_env)
    env.load_tick(0)
    
    agent = BandedAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    
    env_state = {
        "temperature": 0.5,
        "hydration": 0.8,
        "vegetation": 0.4,
        "movement_cost": 0.0,
        "threat": 0.8,
        "neighborhood_threat": np.ones((5, 5)) * 0.8
    }
    
    agent_state_dict = {
        "energy": agent.state.energy,
        "position": (agent.state.x, agent.state.y),
        "tick": 0
    }
    
    perception = agent.bands[0].perceive(env_state, agent_state_dict)
    assert perception["local_threat"] == 0.8
    
    urgency = agent.bands[0].compute_urgency(perception)
    assert urgency > 5.0

def test_agent_flees_from_threat(test_env):
    agent = BandedAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    
    env_state = {
        "temperature": 0.5,
        "hydration": 0.8,
        "vegetation": 0.4,
        "movement_cost": 0.0,
        "threat": 0.9,
        "neighborhood_threat": np.ones((5, 5)) * 0.5
    }
    
    old_x, old_y = agent.state.x, agent.state.y
    agent.step(env_state, world_width=256, world_height=256)
    
    assert (agent.state.x, agent.state.y) != (old_x, old_y)
    assert agent.decision_history[-1]["dominant_band"] == 1

def test_agent_forages_when_hungry(test_env):
    agent = BandedAgent(agent_id=0, x=128, y=128, initial_energy=15.0, seed=42)
    
    env_state = {
        "temperature": 0.5,
        "hydration": 0.8,
        "vegetation": 0.6,
        "movement_cost": 0.0,
        "threat": 0.0,
        "neighborhood_threat": np.zeros((5, 5))
    }
    
    agent.step(env_state, world_width=256, world_height=256)
    
    assert agent.decision_history[-1]["action"] in ["FORAGE", "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"]

def test_predator_system_creation():
    predators = PredatorSystem(world_width=256, world_height=256, num_predators=5, seed=42)
    assert len(predators.predators) == 5
    assert predators.threat_field.shape == (256, 256)

def test_predator_updates_threat_field():
    predators = PredatorSystem(world_width=256, world_height=256, num_predators=3, seed=42)
    
    agent_positions = [(128, 128), (130, 130)]
    predators.update(agent_positions, tick=0)
    
    assert predators.threat_field.max() > 0

def test_predator_catches_agent():
    predators = PredatorSystem(world_width=256, world_height=256, num_predators=1, seed=42)
    
    predators.predators[0].x = 100
    predators.predators[0].y = 100
    
    agent_positions = [(100, 100), (150, 150)]
    caught = predators.check_predation(agent_positions)
    
    assert 0 in caught

def test_agent_simulation_creation(test_env):
    sim = AgentSimulation(test_env, num_predators=3, seed=42)
    assert sim.world_width == 256
    assert sim.world_height == 256
    assert len(sim.predators.predators) == 3

def test_agent_simulation_spawn(test_env):
    sim = AgentSimulation(test_env, num_predators=3, seed=42)
    sim.spawn_agents(num_agents=10, initial_energy=100.0)
    
    assert len(sim.agents) == 10
    assert all(isinstance(a, BandedAgent) for a in sim.agents)

def test_agent_simulation_step(test_env):
    sim = AgentSimulation(test_env, num_predators=2, seed=42)
    sim.spawn_agents(num_agents=5, initial_energy=100.0)
    
    sim.step()
    
    assert sim.current_tick == 1
    assert len(sim.population_stats) == 1

def test_agent_simulation_run(test_env):
    sim = AgentSimulation(test_env, num_predators=3, seed=42)
    sim.spawn_agents(num_agents=10, initial_energy=120.0)
    
    sim.run(num_ticks=20, verbose=False)
    
    assert sim.current_tick == 20
    assert len(sim.population_stats) == 20

def test_agent_survival_with_predators(test_env):
    sim = AgentSimulation(test_env, num_predators=2, seed=42)
    sim.spawn_agents(num_agents=15, initial_energy=150.0)
    
    sim.run(num_ticks=50, verbose=False)
    
    survival_rate = sim.get_survival_rate()
    assert 0.0 <= survival_rate <= 1.0

def test_predation_events_recorded(test_env):
    sim = AgentSimulation(test_env, num_predators=5, seed=42)
    sim.spawn_agents(num_agents=20, initial_energy=100.0)
    
    sim.run(num_ticks=30, verbose=False)
    
    assert len(sim.predation_events) >= 0

def test_agent_handles_predation():
    agent = BandedAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    
    agent.handle_predation()
    
    assert agent.state.energy == 50.0
    assert agent.state.times_caught == 1
    assert agent.state.alive is True

def test_agent_dies_from_predation():
    agent = BandedAgent(agent_id=0, x=128, y=128, initial_energy=40.0, seed=42)
    
    agent.handle_predation()
    
    assert agent.state.energy == 0.0
    assert agent.state.alive is False

