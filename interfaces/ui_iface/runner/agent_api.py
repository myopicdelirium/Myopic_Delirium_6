import numpy as np
import json
import os
from .hydrator import replay_frame
from .registry import build_registry

class EnvironmentGrid:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        with open(os.path.join(run_dir, "scenario.json"), "r") as f:
            self.cfg = json.load(f)
        self.registry = build_registry(self.cfg)
        self.h = self.cfg["world"]["height"]
        self.w = self.cfg["world"]["width"]
        self.f = len(self.registry["names"])
        self.current_tick = 0
        self.tensor = None
    
    def load_tick(self, tick: int):
        self.current_tick = tick
        self.tensor = replay_frame(self.run_dir, tick, self.h, self.w, self.f)
        return self.tensor
    
    def get_field(self, field_name: str) -> np.ndarray:
        if self.tensor is None:
            raise ValueError("Call load_tick() first")
        idx = self.registry["indices"][field_name]
        return self.tensor[:, :, idx]
    
    def get_cell(self, x: int, y: int, field_name: str) -> float:
        if self.tensor is None:
            raise ValueError("Call load_tick() first")
        idx = self.registry["indices"][field_name]
        return float(self.tensor[y, x, idx])
    
    def get_all_fields_at(self, x: int, y: int) -> dict:
        if self.tensor is None:
            raise ValueError("Call load_tick() first")
        return {name: float(self.tensor[y, x, idx]) 
                for name, idx in self.registry["indices"].items()}
    
    def get_neighborhood(self, x: int, y: int, radius: int = 1) -> dict:
        if self.tensor is None:
            raise ValueError("Call load_tick() first")
        y_min = max(0, y - radius)
        y_max = min(self.h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(self.w, x + radius + 1)
        neighborhood = {}
        for name, idx in self.registry["indices"].items():
            neighborhood[name] = self.tensor[y_min:y_max, x_min:x_max, idx]
        return neighborhood
    
    @property
    def shape(self):
        return (self.h, self.w, self.f)
    
    @property
    def field_names(self):
        return self.registry["names"]

def get_agent_grid(run_dir: str, tick: int = 0) -> EnvironmentGrid:
    env = EnvironmentGrid(run_dir)
    env.load_tick(tick)
    return env
