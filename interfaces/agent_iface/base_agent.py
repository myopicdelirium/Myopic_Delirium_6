import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum

class Action(Enum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    STAY = 4

@dataclass
class AgentState:
    agent_id: int
    x: int
    y: int
    energy: float
    tick: int
    alive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "tick": self.tick,
            "alive": self.alive
        }

@dataclass
class Perception:
    local_temperature: float
    local_hydration: float
    local_vegetation: float
    local_movement_cost: float
    neighborhood_temperature: np.ndarray
    neighborhood_hydration: np.ndarray
    neighborhood_vegetation: np.ndarray
    position: Tuple[int, int]
    tick: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_temperature": float(self.local_temperature),
            "local_hydration": float(self.local_hydration),
            "local_vegetation": float(self.local_vegetation),
            "local_movement_cost": float(self.local_movement_cost),
            "position": self.position,
            "tick": self.tick,
            "neighborhood_mean_temp": float(self.neighborhood_temperature.mean()),
            "neighborhood_mean_hydration": float(self.neighborhood_hydration.mean())
        }

class BaseAgent:
    def __init__(self, agent_id: int, x: int, y: int, initial_energy: float = 100.0, seed: int = None):
        self.state = AgentState(
            agent_id=agent_id,
            x=x,
            y=y,
            energy=initial_energy,
            tick=0
        )
        self.rng = np.random.default_rng(seed)
        self.perception_history = []
        self.action_history = []
        
    def perceive(self, env) -> Perception:
        from ..ui_iface.runner.agent_api import EnvironmentGrid
        
        if not isinstance(env, EnvironmentGrid):
            raise TypeError("env must be EnvironmentGrid instance")
        
        fields = env.get_all_fields_at(self.state.x, self.state.y)
        neighborhood = env.get_neighborhood(self.state.x, self.state.y, radius=2)
        
        perception = Perception(
            local_temperature=fields["temperature"],
            local_hydration=fields["hydration"],
            local_vegetation=fields["vegetation"],
            local_movement_cost=fields.get("movement_cost", 0.0),
            neighborhood_temperature=neighborhood["temperature"],
            neighborhood_hydration=neighborhood["hydration"],
            neighborhood_vegetation=neighborhood["vegetation"],
            position=(self.state.x, self.state.y),
            tick=self.state.tick
        )
        
        self.perception_history.append(perception)
        return perception
    
    def decide(self, perception: Perception) -> Action:
        raise NotImplementedError("Subclasses must implement decide()")
    
    def execute_action(self, action: Action, world_width: int, world_height: int):
        dx, dy = 0, 0
        
        if action == Action.MOVE_NORTH:
            dy = -1
        elif action == Action.MOVE_SOUTH:
            dy = 1
        elif action == Action.MOVE_EAST:
            dx = 1
        elif action == Action.MOVE_WEST:
            dx = -1
        
        new_x = (self.state.x + dx) % world_width
        new_y = (self.state.y + dy) % world_height
        
        self.state.x = new_x
        self.state.y = new_y
    
    def update_energy(self, delta: float):
        self.state.energy += delta
        if self.state.energy <= 0:
            self.state.alive = False
            self.state.energy = 0.0
    
    def step(self, env, world_width: int, world_height: int):
        if not self.state.alive:
            return
        
        perception = self.perceive(env)
        action = self.decide(perception)
        
        self.action_history.append({
            "tick": self.state.tick,
            "action": action.name,
            "position_before": (self.state.x, self.state.y)
        })
        
        self.execute_action(action, world_width, world_height)
        
        energy_cost = self._compute_energy_cost(action, perception)
        self.update_energy(energy_cost)
        
        self.state.tick += 1
    
    def _compute_energy_cost(self, action: Action, perception: Perception) -> float:
        base_cost = -1.0
        
        if action != Action.STAY:
            movement_penalty = -2.0 * perception.local_movement_cost
            base_cost += movement_penalty
        
        energy_gain = 5.0 * perception.local_vegetation
        
        return base_cost + energy_gain
    
    def get_trajectory(self) -> list:
        trajectory = []
        for i, action_record in enumerate(self.action_history):
            if i < len(self.perception_history):
                perception = self.perception_history[i]
                trajectory.append({
                    **action_record,
                    **perception.to_dict()
                })
        return trajectory

class RandomAgent(BaseAgent):
    def decide(self, perception: Perception) -> Action:
        return self.rng.choice(list(Action))

class GradientAgent(BaseAgent):
    def decide(self, perception: Perception) -> Action:
        temp_grad = np.gradient(perception.neighborhood_temperature)
        hydration_grad = np.gradient(perception.neighborhood_hydration)
        
        temp_direction = np.argmax(np.abs(temp_grad))
        hydration_direction = np.argmax(np.abs(hydration_grad))
        
        if perception.local_hydration < 0.5:
            if hydration_grad[hydration_direction] > 0:
                return Action.MOVE_NORTH if hydration_direction == 0 else Action.MOVE_EAST
            else:
                return Action.MOVE_SOUTH if hydration_direction == 0 else Action.MOVE_WEST
        
        return Action.STAY

