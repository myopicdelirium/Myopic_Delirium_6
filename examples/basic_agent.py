from interfaces.ui_iface.runner.agent_api import get_agent_grid
import random

class SimpleAgent:
    def __init__(self, x, y, env):
        self.x = x
        self.y = y
        self.env = env
    
    def step(self):
        nbr = self.env.get_neighborhood(self.x, self.y, radius=1)
        hydration = nbr['hydration']
        best_y, best_x = 0, 0
        best_water = -float('inf')
        for dy in range(hydration.shape[0]):
            for dx in range(hydration.shape[1]):
                if hydration[dy, dx] > best_water:
                    best_water = hydration[dy, dx]
                    best_y, best_x = dy, dx
        offset_y = best_y - 1
        offset_x = best_x - 1
        self.y = max(0, min(self.env.h - 1, self.y + offset_y))
        self.x = max(0, min(self.env.w - 1, self.x + offset_x))
        return self.x, self.y

if __name__ == "__main__":
    env = get_agent_grid('runs/run-evolution', tick=255)
    agent = SimpleAgent(128, 128, env)
    print(f"Agent starts at ({agent.x}, {agent.y})")
    for i in range(10):
        x, y = agent.step()
        cell = env.get_all_fields_at(x, y)
        print(f"Step {i+1}: ({x}, {y}) - Hydration: {cell['hydration']:.3f}")
