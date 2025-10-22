import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class Predator:
    predator_id: int
    x: int
    y: int
    hunt_radius: int
    speed: int
    active: bool = True
    
class PredatorSystem:
    def __init__(self, world_width: int, world_height: int, num_predators: int, seed: int = None):
        self.world_width = world_width
        self.world_height = world_height
        self.predators: List[Predator] = []
        self.rng = np.random.default_rng(seed)
        self.threat_field = np.zeros((world_height, world_width), dtype=np.float32)
        
        for i in range(num_predators):
            x = self.rng.integers(0, world_width)
            y = self.rng.integers(0, world_height)
            hunt_radius = self.rng.integers(5, 15)
            speed = self.rng.integers(1, 3)
            
            self.predators.append(Predator(
                predator_id=i,
                x=x,
                y=y,
                hunt_radius=hunt_radius,
                speed=speed
            ))
    
    def update(self, agent_positions: List[Tuple[int, int]], tick: int):
        """Update predator positions and threat field."""
        self.threat_field.fill(0.0)
        
        for predator in self.predators:
            if not predator.active:
                continue
            
            closest_agent = self._find_closest_agent(predator, agent_positions)
            
            if closest_agent is not None:
                self._move_toward(predator, closest_agent)
            else:
                self._random_patrol(predator)
            
            self._update_threat_field(predator)
    
    def _find_closest_agent(self, predator: Predator, agent_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Find closest agent within hunt radius."""
        if not agent_positions:
            return None
        
        min_dist = float('inf')
        closest = None
        
        for ax, ay in agent_positions:
            dx = min(abs(ax - predator.x), self.world_width - abs(ax - predator.x))
            dy = min(abs(ay - predator.y), self.world_height - abs(ay - predator.y))
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist and dist <= predator.hunt_radius:
                min_dist = dist
                closest = (ax, ay)
        
        return closest
    
    def _move_toward(self, predator: Predator, target: Tuple[int, int]):
        """Move predator toward target."""
        tx, ty = target
        
        dx = tx - predator.x
        dy = ty - predator.y
        
        if abs(dx) > self.world_width / 2:
            dx = -(self.world_width - abs(dx)) * np.sign(dx)
        if abs(dy) > self.world_height / 2:
            dy = -(self.world_height - abs(dy)) * np.sign(dy)
        
        if abs(dx) > abs(dy):
            step_x = np.sign(dx) * min(predator.speed, abs(dx))
            step_y = 0
        else:
            step_x = 0
            step_y = np.sign(dy) * min(predator.speed, abs(dy))
        
        predator.x = (predator.x + step_x) % self.world_width
        predator.y = (predator.y + step_y) % self.world_height
    
    def _random_patrol(self, predator: Predator):
        """Random patrol movement."""
        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)
        
        predator.x = (predator.x + dx) % self.world_width
        predator.y = (predator.y + dy) % self.world_height
    
    def _update_threat_field(self, predator: Predator):
        """Update threat field with predator influence."""
        threat_radius = predator.hunt_radius + 5
        
        for dy in range(-threat_radius, threat_radius + 1):
            for dx in range(-threat_radius, threat_radius + 1):
                x = (predator.x + dx) % self.world_width
                y = (predator.y + dy) % self.world_height
                
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= threat_radius:
                    threat = max(0.0, 1.0 - dist / threat_radius)
                    self.threat_field[y, x] = max(self.threat_field[y, x], threat)
    
    def get_threat_at(self, x: int, y: int) -> float:
        """Get threat level at position."""
        return float(self.threat_field[y, x])
    
    def get_local_threat(self, x: int, y: int, radius: int = 3) -> np.ndarray:
        """Get threat field in local neighborhood."""
        y_min = max(0, y - radius)
        y_max = min(self.world_height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(self.world_width, x + radius + 1)
        
        return self.threat_field[y_min:y_max, x_min:x_max].copy()
    
    def check_predation(self, agent_positions: List[Tuple[int, int]]) -> List[int]:
        """Check which agents are caught by predators (return agent indices)."""
        caught_indices = []
        
        for i, (ax, ay) in enumerate(agent_positions):
            for predator in self.predators:
                if not predator.active:
                    continue
                
                dx = min(abs(ax - predator.x), self.world_width - abs(ax - predator.x))
                dy = min(abs(ay - predator.y), self.world_height - abs(ay - predator.y))
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist <= 1.0:
                    caught_indices.append(i)
                    break
        
        return caught_indices
    
    def get_state(self) -> Dict[str, Any]:
        """Get current predator system state."""
        return {
            "num_active": sum(1 for p in self.predators if p.active),
            "positions": [(p.x, p.y) for p in self.predators if p.active],
            "threat_mean": float(self.threat_field.mean()),
            "threat_max": float(self.threat_field.max())
        }

