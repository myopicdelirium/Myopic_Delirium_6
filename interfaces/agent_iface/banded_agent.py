import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .band import Band, Action, ActionProposal
from .band_physiological import PhysiologicalBand
from .arbiter import Arbiter

@dataclass
class AgentState:
    agent_id: int
    x: int
    y: int
    energy: float
    tick: int
    alive: bool = True
    times_caught: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "tick": self.tick,
            "alive": self.alive,
            "times_caught": self.times_caught
        }

class BandedAgent:
    """
    Agent with banded controller architecture.
    Currently implements Band 1 (Physiological) only.
    Additional bands will be added progressively.
    """
    
    def __init__(self, agent_id: int, x: int, y: int, initial_energy: float = 100.0, 
                 seed: int = None, band_seeds: Optional[Dict[int, int]] = None):
        self.state = AgentState(
            agent_id=agent_id,
            x=x,
            y=y,
            energy=initial_energy,
            tick=0
        )
        
        self.rng = np.random.default_rng(seed)
        
        if band_seeds is None:
            band_seeds = {
                1: (seed + 1000) if seed is not None else None,
            }
        
        self.bands: List[Band] = [
            PhysiologicalBand(band_id=1, initial_gain=2.0, seed=band_seeds.get(1))
        ]
        
        arbiter_seed = (seed + 2000) if seed is not None else None
        self.arbiter = Arbiter(inertia=0.3, temperature=2.0, seed=arbiter_seed)
        
        self.decision_history = []
        self.trajectory = []
        
    def step(self, env_state: Dict[str, Any], world_width: int, world_height: int):
        """Execute one timestep: perceive → decide → act → learn."""
        if not self.state.alive:
            return
        
        agent_state_dict = {
            "energy": self.state.energy,
            "position": (self.state.x, self.state.y),
            "tick": self.state.tick
        }
        
        all_perceptions = []
        all_proposals = []
        
        for band in self.bands:
            perception = band.perceive(env_state, agent_state_dict)
            all_perceptions.append(perception)
            
            urgency = band.compute_urgency(perception)
            
            proposals = band.propose_actions(perception)
            all_proposals.append(proposals)
        
        selected_action, dominant_band_id, selected_proposal = self.arbiter.select_action(
            self.bands, all_proposals, agent_state_dict
        )
        
        self.decision_history.append({
            "tick": self.state.tick,
            "position": (self.state.x, self.state.y),
            "action": selected_action.name,
            "dominant_band": dominant_band_id,
            "urgencies": [band.state.urgency for band in self.bands],
            "energy": self.state.energy
        })
        
        old_x, old_y = self.state.x, self.state.y
        self._execute_action(selected_action, world_width, world_height)
        
        outcome = self._compute_outcome(env_state, selected_action, old_x, old_y)
        
        for i, band in enumerate(self.bands):
            perception = all_perceptions[i]
            band.update_state(perception, selected_action, outcome)
            
            learning_signal = band.compute_learning_signal(perception, selected_action, outcome)
            affect = learning_signal
            
            band.write_memory(perception, selected_action, outcome, affect)
            
            band.update_gain()
        
        self.state.tick += 1
    
    def _execute_action(self, action: Action, world_width: int, world_height: int):
        """Execute action and update position."""
        dx, dy = 0, 0
        
        if action == Action.MOVE_NORTH:
            dy = -1
        elif action == Action.MOVE_SOUTH:
            dy = 1
        elif action == Action.MOVE_EAST:
            dx = 1
        elif action == Action.MOVE_WEST:
            dx = -1
        
        self.state.x = (self.state.x + dx) % world_width
        self.state.y = (self.state.y + dy) % world_height
    
    def _compute_outcome(self, env_state: Dict[str, Any], action: Action, 
                        old_x: int, old_y: int) -> Dict[str, Any]:
        """Compute outcome of action for learning."""
        base_cost = -1.0
        
        if action in [Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST]:
            movement_cost = env_state.get("movement_cost", 0.0)
            base_cost -= 2.0 * movement_cost
        
        energy_gain = 0.0
        
        if action == Action.FORAGE:
            vegetation = env_state.get("vegetation", 0.0)
            energy_gain = vegetation * 10.0
        elif action == Action.DRINK:
            hydration = env_state.get("hydration", 0.0)
            energy_gain = hydration * 5.0
        elif action == Action.REST:
            energy_gain = 2.0
        elif action == Action.STAY:
            base_cost = -0.5
        
        energy_delta = base_cost + energy_gain
        self.state.energy = max(0.0, min(150.0, self.state.energy + energy_delta))
        
        if self.state.energy <= 0:
            self.state.alive = False
        
        return {
            "tick": self.state.tick,
            "energy_delta": energy_delta,
            "new_energy": self.state.energy,
            "new_position": (self.state.x, self.state.y),
            "old_position": (old_x, old_y),
            "dominant_band": self.arbiter.previous_band
        }
    
    def handle_predation(self):
        """Handle being caught by predator."""
        self.state.times_caught += 1
        self.state.energy = max(0.0, self.state.energy - 50.0)
        
        if self.state.energy <= 0:
            self.state.alive = False
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get complete decision trajectory."""
        return self.decision_history
    
    def get_band_dominance(self) -> Dict[int, float]:
        """Get distribution of which bands dominated decisions."""
        return self.arbiter.get_dominant_band_distribution()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary."""
        return {
            **self.state.to_dict(),
            "band_urgencies": [band.state.urgency for band in self.bands],
            "band_gains": [band.state.gain for band in self.bands],
            "band_frustrations": [band.state.frustration_accumulator for band in self.bands]
        }

