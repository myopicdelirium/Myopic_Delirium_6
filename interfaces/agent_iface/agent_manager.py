import numpy as np
import json
import os
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentState
from ..ui_iface.runner.agent_api import EnvironmentGrid

class AgentManager:
    def __init__(self, run_dir: str, seed: int = 42):
        self.run_dir = run_dir
        self.env = EnvironmentGrid(run_dir)
        self.agents: List[BaseAgent] = []
        self.rng = np.random.default_rng(seed)
        self.current_tick = 0
        self.world_width = self.env.w
        self.world_height = self.env.h
        
    def add_agent(self, agent: BaseAgent):
        self.agents.append(agent)
    
    def spawn_agents(self, agent_class, num_agents: int, initial_energy: float = 100.0, agent_seed_base: int = 1000):
        for i in range(num_agents):
            x = self.rng.integers(0, self.world_width)
            y = self.rng.integers(0, self.world_height)
            agent_seed = agent_seed_base + i
            agent = agent_class(
                agent_id=i,
                x=x,
                y=y,
                initial_energy=initial_energy,
                seed=agent_seed
            )
            self.add_agent(agent)
    
    def step(self):
        self.env.load_tick(self.current_tick)
        
        for agent in self.agents:
            if agent.state.alive:
                agent.step(self.env, self.world_width, self.world_height)
        
        self.current_tick += 1
    
    def run_simulation(self, num_ticks: int):
        for _ in range(num_ticks):
            self.step()
    
    def get_alive_count(self) -> int:
        return sum(1 for agent in self.agents if agent.state.alive)
    
    def get_agent_states(self) -> List[Dict[str, Any]]:
        return [agent.state.to_dict() for agent in self.agents]
    
    def get_agent_trajectories(self) -> Dict[int, List[Dict[str, Any]]]:
        return {
            agent.state.agent_id: agent.get_trajectory()
            for agent in self.agents
        }
    
    def save_trajectories(self, output_path: str):
        trajectories = self.get_agent_trajectories()
        
        with open(output_path, 'w') as f:
            json.dump(trajectories, f, indent=2)
    
    def get_population_stats(self) -> Dict[str, Any]:
        alive_agents = [a for a in self.agents if a.state.alive]
        
        if not alive_agents:
            return {
                "tick": self.current_tick,
                "alive_count": 0,
                "mean_energy": 0.0,
                "std_energy": 0.0,
                "mean_x": 0.0,
                "mean_y": 0.0
            }
        
        energies = [a.state.energy for a in alive_agents]
        positions_x = [a.state.x for a in alive_agents]
        positions_y = [a.state.y for a in alive_agents]
        
        return {
            "tick": self.current_tick,
            "alive_count": len(alive_agents),
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "mean_x": float(np.mean(positions_x)),
            "mean_y": float(np.mean(positions_y)),
            "min_energy": float(np.min(energies)),
            "max_energy": float(np.max(energies))
        }
    
    def save_population_stats(self, output_path: str):
        stats_over_time = []
        
        for agent in self.agents:
            for tick in range(len(agent.action_history)):
                stats_over_time.append({
                    "tick": tick,
                    "agent_id": agent.state.agent_id,
                    "alive": tick < len(agent.action_history),
                    "energy": agent.perception_history[tick].to_dict() if tick < len(agent.perception_history) else None
                })
        
        with open(output_path, 'w') as f:
            json.dump(stats_over_time, f, indent=2)

