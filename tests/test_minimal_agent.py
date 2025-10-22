import pytest
import tempfile
import os
from interfaces.agent_iface.base_agent import BaseAgent, RandomAgent, GradientAgent, Action, AgentState, Perception
from interfaces.agent_iface.agent_manager import AgentManager
from interfaces.ui_iface.runner.engine import load_scenario, run_headless

@pytest.fixture
def test_env():
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_scenario(scenario_path)
        run_dir = run_headless(cfg, ticks=50, out_dir=tmpdir, label="agent_test")
        yield run_dir

def test_agent_state_creation():
    state = AgentState(agent_id=0, x=100, y=100, energy=100.0, tick=0)
    assert state.agent_id == 0
    assert state.x == 100
    assert state.y == 100
    assert state.energy == 100.0
    assert state.alive is True

def test_agent_state_to_dict():
    state = AgentState(agent_id=1, x=50, y=75, energy=80.0, tick=10)
    state_dict = state.to_dict()
    assert state_dict["agent_id"] == 1
    assert state_dict["x"] == 50
    assert state_dict["y"] == 75
    assert state_dict["energy"] == 80.0

def test_random_agent_creation():
    agent = RandomAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    assert agent.state.x == 128
    assert agent.state.y == 128
    assert agent.state.energy == 100.0
    assert agent.state.alive is True

def test_agent_perception(test_env):
    from interfaces.ui_iface.runner.agent_api import EnvironmentGrid
    
    env = EnvironmentGrid(test_env)
    env.load_tick(0)
    
    agent = RandomAgent(agent_id=0, x=128, y=128, initial_energy=100.0)
    perception = agent.perceive(env)
    
    assert isinstance(perception, Perception)
    assert 0.0 <= perception.local_temperature <= 1.0
    assert 0.0 <= perception.local_hydration <= 1.0
    assert 0.0 <= perception.local_vegetation <= 1.0
    assert perception.position == (128, 128)

def test_agent_movement():
    agent = RandomAgent(agent_id=0, x=100, y=100, initial_energy=100.0)
    
    agent.execute_action(Action.MOVE_NORTH, world_width=256, world_height=256)
    assert agent.state.y == 99
    assert agent.state.x == 100
    
    agent.execute_action(Action.MOVE_EAST, world_width=256, world_height=256)
    assert agent.state.x == 101
    assert agent.state.y == 99

def test_agent_wrapping():
    agent = RandomAgent(agent_id=0, x=0, y=0, initial_energy=100.0)
    
    agent.execute_action(Action.MOVE_WEST, world_width=256, world_height=256)
    assert agent.state.x == 255
    
    agent.execute_action(Action.MOVE_NORTH, world_width=256, world_height=256)
    assert agent.state.y == 255

def test_agent_energy_update():
    agent = RandomAgent(agent_id=0, x=128, y=128, initial_energy=100.0)
    
    agent.update_energy(-10.0)
    assert agent.state.energy == 90.0
    assert agent.state.alive is True
    
    agent.update_energy(-95.0)
    assert agent.state.energy == 0.0
    assert agent.state.alive is False

def test_agent_step(test_env):
    from interfaces.ui_iface.runner.agent_api import EnvironmentGrid
    
    env = EnvironmentGrid(test_env)
    env.load_tick(0)
    
    agent = RandomAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    initial_energy = agent.state.energy
    
    agent.step(env, world_width=256, world_height=256)
    
    assert agent.state.tick == 1
    assert len(agent.perception_history) == 1
    assert len(agent.action_history) == 1
    assert agent.state.energy != initial_energy

def test_agent_manager_creation(test_env):
    manager = AgentManager(test_env, seed=42)
    assert manager.current_tick == 0
    assert len(manager.agents) == 0
    assert manager.world_width == 256
    assert manager.world_height == 256

def test_agent_manager_spawn(test_env):
    manager = AgentManager(test_env, seed=42)
    manager.spawn_agents(RandomAgent, num_agents=10, initial_energy=100.0)
    
    assert len(manager.agents) == 10
    assert all(isinstance(agent, RandomAgent) for agent in manager.agents)
    assert all(agent.state.alive for agent in manager.agents)

def test_agent_manager_step(test_env):
    manager = AgentManager(test_env, seed=42)
    manager.spawn_agents(RandomAgent, num_agents=5, initial_energy=100.0)
    
    manager.step()
    
    assert manager.current_tick == 1
    assert all(agent.state.tick == 1 for agent in manager.agents)

def test_agent_manager_simulation(test_env):
    manager = AgentManager(test_env, seed=42)
    manager.spawn_agents(RandomAgent, num_agents=5, initial_energy=150.0)
    
    manager.run_simulation(num_ticks=10)
    
    assert manager.current_tick == 10
    assert manager.get_alive_count() >= 0

def test_agent_trajectory_recording(test_env):
    from interfaces.ui_iface.runner.agent_api import EnvironmentGrid
    
    env = EnvironmentGrid(test_env)
    env.load_tick(0)
    
    agent = RandomAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    
    for _ in range(5):
        agent.step(env, world_width=256, world_height=256)
        env.load_tick(agent.state.tick)
    
    trajectory = agent.get_trajectory()
    assert len(trajectory) == 5
    assert all("tick" in record for record in trajectory)
    assert all("action" in record for record in trajectory)

def test_gradient_agent(test_env):
    from interfaces.ui_iface.runner.agent_api import EnvironmentGrid
    
    env = EnvironmentGrid(test_env)
    env.load_tick(0)
    
    agent = GradientAgent(agent_id=0, x=128, y=128, initial_energy=100.0, seed=42)
    perception = agent.perceive(env)
    action = agent.decide(perception)
    
    assert isinstance(action, Action)

