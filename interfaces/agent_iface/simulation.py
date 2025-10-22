import numpy as np
import json
import os
from typing import List, Dict, Any, Type
from .banded_agent import BandedAgent
from ..ui_iface.runner.agent_api import EnvironmentGrid
from ..ui_iface.runner.predators import PredatorSystem

class AgentSimulation:
    """
    Manages agent-environment simulation with predators.
    Integrates banded agents, environment state, and predator dynamics.
    """
    
    def __init__(self, run_dir: str, num_predators: int = 5, seed: int = 42):
        self.run_dir = run_dir
        self.env = EnvironmentGrid(run_dir)
        self.world_width = self.env.w
        self.world_height = self.env.h
        
        self.predators = PredatorSystem(
            world_width=self.world_width,
            world_height=self.world_height,
            num_predators=num_predators,
            seed=seed + 1000
        )
        
        self.agents: List[BandedAgent] = []
        self.rng = np.random.default_rng(seed)
        self.current_tick = 0
        
        self.population_stats = []
        self.predation_events = []
        
    def spawn_agents(self, num_agents: int, initial_energy: float = 100.0, agent_seed_base: int = 2000):
        """Spawn banded agents at random locations."""
        for i in range(num_agents):
            x = self.rng.integers(0, self.world_width)
            y = self.rng.integers(0, self.world_height)
            agent_seed = agent_seed_base + i
            
            agent = BandedAgent(
                agent_id=i,
                x=x,
                y=y,
                initial_energy=initial_energy,
                seed=agent_seed
            )
            self.agents.append(agent)
    
    def step(self):
        """Execute one simulation tick."""
        self.env.load_tick(self.current_tick)
        
        alive_agents = [a for a in self.agents if a.state.alive]
        agent_positions = [(a.state.x, a.state.y) for a in alive_agents]
        
        self.predators.update(agent_positions, self.current_tick)
        
        for agent in alive_agents:
            env_state = self._get_env_state_for_agent(agent)
            agent.step(env_state, self.world_width, self.world_height)
        
        caught_indices = self.predators.check_predation(
            [(a.state.x, a.state.y) for a in alive_agents]
        )
        
        for idx in caught_indices:
            agent = alive_agents[idx]
            agent.handle_predation()
            
            self.predation_events.append({
                "tick": self.current_tick,
                "agent_id": agent.state.agent_id,
                "position": (agent.state.x, agent.state.y),
                "energy_after": agent.state.energy,
                "died": not agent.state.alive
            })
        
        self.population_stats.append(self._compute_population_stats())
        
        self.current_tick += 1
    
    def _get_env_state_for_agent(self, agent: BandedAgent) -> Dict[str, Any]:
        """Get environment state at agent's location including threat."""
        fields = self.env.get_all_fields_at(agent.state.x, agent.state.y)
        
        local_threat = self.predators.get_threat_at(agent.state.x, agent.state.y)
        neighborhood_threat = self.predators.get_local_threat(agent.state.x, agent.state.y, radius=3)
        
        return {
            "temperature": fields.get("temperature", 0.5),
            "hydration": fields.get("hydration", 0.5),
            "vegetation": fields.get("vegetation", 0.0),
            "movement_cost": fields.get("movement_cost", 0.0),
            "threat": local_threat,
            "neighborhood_threat": neighborhood_threat
        }
    
    def run(self, num_ticks: int, verbose: bool = False):
        """Run simulation for specified ticks."""
        for tick in range(num_ticks):
            self.step()
            
            if verbose and tick % 10 == 0:
                alive = sum(1 for a in self.agents if a.state.alive)
                print(f"Tick {tick}: {alive}/{len(self.agents)} alive, "
                      f"{len(self.predation_events)} predation events")
    
    def _compute_population_stats(self) -> Dict[str, Any]:
        """Compute population statistics."""
        alive_agents = [a for a in self.agents if a.state.alive]
        
        if not alive_agents:
            return {
                "tick": self.current_tick,
                "alive_count": 0,
                "mean_energy": 0.0,
                "total_predation_events": len(self.predation_events)
            }
        
        energies = [a.state.energy for a in alive_agents]
        
        band_urgencies = []
        for agent in alive_agents:
            if agent.bands:
                band_urgencies.append(agent.bands[0].state.urgency)
        
        return {
            "tick": self.current_tick,
            "alive_count": len(alive_agents),
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "min_energy": float(np.min(energies)),
            "max_energy": float(np.max(energies)),
            "mean_band1_urgency": float(np.mean(band_urgencies)) if band_urgencies else 0.0,
            "total_predation_events": len(self.predation_events),
            "predator_threat_mean": self.predators.threat_field.mean()
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete simulation results."""
        return {
            "population_stats": self.population_stats,
            "predation_events": self.predation_events,
            "final_alive_count": sum(1 for a in self.agents if a.state.alive),
            "agent_trajectories": {
                a.state.agent_id: a.get_trajectory()
                for a in self.agents
            },
            "agent_band_dominance": {
                a.state.agent_id: a.get_band_dominance()
                for a in self.agents if a.state.alive
            }
        }
    
    def save_results(self, output_path: str):
        """Save simulation results to JSON."""
        results = self.get_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_survival_rate(self) -> float:
        """Get final survival rate."""
        alive = sum(1 for a in self.agents if a.state.alive)
        return alive / len(self.agents) if self.agents else 0.0

