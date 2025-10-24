"""
FAST migration visualization using static environment.
Runs in ~10 seconds instead of hours.

Uses a single environment snapshot - agents move on fixed terrain.
Perfect for validating gradient-following behavior without I/O overhead.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from interfaces.agent_iface.banded_agent import BandedAgent
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
from interfaces.ui_iface.runner.predators import PredatorSystem
import tempfile

class FastStaticSimulation:
    """Lightweight simulation with static environment - no disk I/O per tick."""
    
    def __init__(self, vegetation, temperature, hydration, world_width, world_height, 
                 num_predators=2, seed=42):
        self.vegetation = vegetation
        self.temperature = temperature
        self.hydration = hydration
        self.world_width = world_width
        self.world_height = world_height
        self.agents = []
        pred_rng = np.random.default_rng(seed + 1000)
        self.predators = PredatorSystem(num_predators, world_width, world_height, pred_rng)
        self.predators.threat_field = np.zeros((world_height, world_width), dtype=np.float32)
        self.current_tick = 0
        self.rng = np.random.default_rng(seed)
    
    def spawn_agents(self, num_agents, initial_energy=50.0):
        """Spawn agents at random positions."""
        for i in range(num_agents):
            x = self.rng.integers(0, self.world_width)
            y = self.rng.integers(0, self.world_height)
            agent = BandedAgent(agent_id=i, x=x, y=y, initial_energy=initial_energy, 
                               seed=self.rng.integers(0, 1000000))
            self.agents.append(agent)
    
    def _get_env_state(self, agent):
        """Get static environment state at agent position with neighborhood."""
        x, y = agent.state.x, agent.state.y
        
        # Get neighborhood (5x5 centered on agent)
        radius = 2
        y_min, y_max = max(0, y-radius), min(self.world_height, y+radius+1)
        x_min, x_max = max(0, x-radius), min(self.world_width, x+radius+1)
        
        neighborhood_veg = self.vegetation[y_min:y_max, x_min:x_max]
        neighborhood_hyd = self.hydration[y_min:y_max, x_min:x_max]
        
        # Get threat
        local_threat = self.predators.get_threat_at(x, y)
        neighborhood_threat = self.predators.get_local_threat(x, y, radius=3)
        
        return {
            "temperature": float(self.temperature[y, x]),
            "hydration": float(self.hydration[y, x]),
            "vegetation": float(self.vegetation[y, x]),
            "movement_cost": 0.0,
            "threat": local_threat,
            "neighborhood_threat": neighborhood_threat,
            "neighborhood_vegetation": neighborhood_veg,
            "neighborhood_hydration": neighborhood_hyd
        }
    
    def step(self):
        """Execute one tick - fast, no disk I/O."""
        alive_agents = [a for a in self.agents if a.state.alive]
        agent_positions = [(a.state.x, a.state.y) for a in alive_agents]
        
        # Update predators
        self.predators.update(agent_positions, self.current_tick)
        
        # Update each agent
        for agent in alive_agents:
            env_state = self._get_env_state(agent)
            agent.step(env_state, self.world_width, self.world_height)
        
        # Check predation
        caught = self.predators.check_predation([(a.state.x, a.state.y) for a in alive_agents])
        for idx in caught:
            alive_agents[idx].handle_predation()
        
        self.current_tick += 1

def create_fast_visualization(num_agents=40, num_ticks=100, num_predators=2, 
                              initial_energy=50.0, seed=42):
    """Create before/after visualization in ~10 seconds."""
    
    print('Loading environment snapshot...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate just 1 tick to get initial state
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='static')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        temperature = tensor[:, :, 0]
        hydration = tensor[:, :, 1]
        h, w = vegetation.shape
        
        print(f'Running fast simulation ({num_agents} agents, {num_ticks} ticks)...')
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators, seed)
        sim.spawn_agents(num_agents, initial_energy)
        
        # Record initial state
        initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
        initial_energy = [a.state.energy for a in sim.agents]
        trajectories = [[(a.state.x, a.state.y)] for a in sim.agents]
        
        # Run simulation
        for tick in range(num_ticks):
            sim.step()
            for i, agent in enumerate(sim.agents):
                if agent.state.alive:
                    trajectories[i].append((agent.state.x, agent.state.y))
            
            if tick % 20 == 0:
                alive = sum(1 for a in sim.agents if a.state.alive)
                print(f'  Tick {tick}: {alive}/{num_agents} alive')
        
        # Record final state
        final_pos = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
        final_energy = [a.state.energy for a in sim.agents if a.state.alive]
        
        initial_veg = [vegetation[y, x] for x, y in initial_pos]
        final_veg = [vegetation[y, x] for x, y in final_pos]
        
        print('\nCreating visualization...')
        fig = plt.figure(figsize=(20, 6))
        
        # Panel 1: Initial positions
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
        ax1.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
                   c=initial_energy, cmap='RdYlGn', s=80, edgecolors='black',
                   vmin=0, vmax=100, linewidth=1.5)
        ax1.set_title(f'Initial (t=0)\nμ_veg={np.mean(initial_veg):.3f}', 
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Panel 2: Final positions with trajectories
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
        
        # Draw trajectories
        for traj in trajectories:
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax2.plot(xs, ys, 'gray', alpha=0.3, linewidth=0.5)
        
        ax2.scatter([p[0] for p in final_pos], [p[1] for p in final_pos],
                   c=final_energy, cmap='RdYlGn', s=80, edgecolors='black',
                   vmin=0, vmax=100, linewidth=1.5)
        ax2.set_title(f'Final (t={num_ticks})\nμ_veg={np.mean(final_veg):.3f}', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('X')
        
        # Panel 3: Vegetation distribution
        ax3 = plt.subplot(1, 4, 3)
        ax3.hist(initial_veg, bins=20, alpha=0.6, label=f'Initial', 
                color='red', edgecolor='black')
        ax3.hist(final_veg, bins=20, alpha=0.6, label=f'Final', 
                color='green', edgecolor='black')
        ax3.axvline(np.mean(initial_veg), color='red', linestyle='--', linewidth=2, 
                   label=f'μ_init={np.mean(initial_veg):.3f}')
        ax3.axvline(np.mean(final_veg), color='green', linestyle='--', linewidth=2,
                   label=f'μ_final={np.mean(final_veg):.3f}')
        ax3.set_xlabel('Vegetation Level', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Location Distribution', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Movement vectors
        ax4 = plt.subplot(1, 4, 4)
        ax4.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
        
        for i, (init, final) in enumerate(zip(initial_pos, final_pos)):
            dx = final[0] - init[0]
            dy = final[1] - init[1]
            init_v = vegetation[init[1], init[0]]
            final_v = vegetation[final[1], final[0]]
            color = 'green' if final_v > init_v else 'red'
            ax4.arrow(init[0], init[1], dx*0.8, dy*0.8, 
                     head_width=2, head_length=2, fc=color, ec=color, 
                     alpha=0.5, linewidth=1)
        
        ax4.set_title('Net Movement\n(green=toward food)', fontsize=13, fontweight='bold')
        ax4.set_xlabel('X')
        
        improvement = np.mean(final_veg) - np.mean(initial_veg)
        survival_pct = 100 * len(final_pos) / num_agents
        
        plt.suptitle(f'Band 1 Gradient-Following Migration: Static Environment Test\n' +
                    f'Survival: {len(final_pos)}/{num_agents} ({survival_pct:.0f}%) | ' +
                    f'Vegetation Δ: {improvement:+.3f} | ' +
                    f'{"✓ Food-seeking confirmed" if improvement > 0.05 else "~ Weak signal"}',
                    fontsize=15, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('migration_fast.png', dpi=150, bbox_inches='tight')
        print('✓ Saved: migration_fast.png')
        
        # Statistical test
        from scipy import stats
        if len(final_veg) > 1 and len(initial_veg) > 1:
            t_stat, p_value = stats.ttest_ind(final_veg, initial_veg)
            print(f'\n=== RESULTS ===')
            print(f'Survival: {len(final_pos)}/{num_agents} ({survival_pct:.1f}%)')
            print(f'Initial μ_veg: {np.mean(initial_veg):.3f} (σ={np.std(initial_veg):.3f})')
            print(f'Final μ_veg:   {np.mean(final_veg):.3f} (σ={np.std(final_veg):.3f})')
            print(f'Improvement:   {improvement:+.3f}')
            print(f'\nt-test: t={t_stat:.3f}, p={p_value:.4f}')
            if p_value < 0.05:
                print('✓ SIGNIFICANT migration toward food (p < 0.05)')
            else:
                print('✗ Not statistically significant')
        
        return improvement, survival_pct

if __name__ == '__main__':
    create_fast_visualization(
        num_agents=40,
        num_ticks=100,
        num_predators=2,
        initial_energy=50.0,
        seed=42
    )

