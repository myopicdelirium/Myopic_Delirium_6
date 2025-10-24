"""
Publication-quality visualization: Band 1 agents migrating toward food sources.

Creates an animated plot showing:
1. Vegetation field (background heatmap)
2. Agent trajectories (colored by energy level)
3. Predator positions (red circles)
4. Statistical overlay (mean vegetation, survival count)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from interfaces.agent_iface.simulation import AgentSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile
import os

def create_migration_animation(
    num_agents=30,
    num_ticks=100,
    num_predators=3,
    initial_energy=60.0,
    seed=42,
    output_path='band1_migration.gif',
    fps=10
):
    """Create animated visualization of Band 1 migration behavior."""
    
    print('Setting up environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'Generating environment ({num_ticks} ticks)...')
        run_dir = run_headless(cfg, ticks=num_ticks, out_dir=tmpdir, label='migration_viz')
        
        tensor = hydrate_tick(run_dir, 0)
        vegetation = tensor[:, :, 2]
        temperature = tensor[:, :, 0]
        
        print(f'Spawning {num_agents} agents with {num_predators} predators...')
        sim = AgentSimulation(run_dir, num_predators=num_predators, seed=seed)
        sim.spawn_agents(num_agents=num_agents, initial_energy=initial_energy)
        
        trajectories = [[] for _ in range(num_agents)]
        energy_history = [[] for _ in range(num_agents)]
        alive_history = []
        mean_veg_history = []
        predator_history = []
        
        print(f'Running simulation...')
        for tick in range(num_ticks):
            for i, agent in enumerate(sim.agents):
                if agent.state.alive:
                    trajectories[i].append((agent.state.x, agent.state.y))
                    energy_history[i].append(agent.state.energy)
            
            alive_count = sum(1 for a in sim.agents if a.state.alive)
            alive_history.append(alive_count)
            
            alive_positions = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
            if alive_positions:
                mean_veg = np.mean([vegetation[y, x] for x, y in alive_positions])
            else:
                mean_veg = 0.0
            mean_veg_history.append(mean_veg)
            
            predator_positions = [(p.x, p.y) for p in sim.predators.predators]
            predator_history.append(predator_positions)
            
            sim.step()
            
            if tick % 10 == 0:
                print(f'  Tick {tick}: {alive_count}/{num_agents} alive, mean_veg={mean_veg:.3f}')
        
        print('Creating animation...')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            ax1.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.8)
            ax1.set_title(f'Band 1 Migration - Tick {frame}\n{alive_history[frame]}/{num_agents} alive', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            
            trail_length = 15
            for i, traj in enumerate(trajectories):
                if frame < len(traj):
                    start_idx = max(0, frame - trail_length)
                    trail = traj[start_idx:frame+1]
                    energies = energy_history[i][start_idx:frame+1]
                    
                    if len(trail) > 1:
                        xs, ys = zip(*trail)
                        
                        energy_norm = np.array(energies) / 100.0
                        colors = plt.cm.RdYlGn(energy_norm)
                        
                        for j in range(len(trail) - 1):
                            alpha = 0.3 + 0.7 * (j / len(trail))
                            ax1.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]], 
                                   color=colors[j], alpha=alpha, linewidth=1.5)
                    
                    if len(trail) > 0:
                        x, y = trail[-1]
                        energy = energies[-1]
                        color = plt.cm.RdYlGn(energy / 100.0)
                        ax1.scatter(x, y, c=[color], s=50, edgecolors='black', linewidth=1, zorder=5)
            
            if frame < len(predator_history):
                for px, py in predator_history[frame]:
                    ax1.scatter(px, py, c='red', s=200, marker='*', 
                              edgecolors='darkred', linewidth=2, zorder=6, label='Predator')
            
            veg_text = f'Mean Vegetation: {mean_veg_history[frame]:.3f}'
            ax1.text(0.02, 0.98, veg_text, transform=ax1.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ticks = list(range(frame + 1))
            ax2.plot(ticks, alive_history[:frame+1], 'b-', linewidth=2, label='Alive Count')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(ticks, mean_veg_history[:frame+1], 'g-', linewidth=2, label='Mean Vegetation')
            
            ax2.set_xlabel('Tick', fontsize=12)
            ax2.set_ylabel('Alive Count', color='b', fontsize=12)
            ax2_twin.set_ylabel('Mean Vegetation', color='g', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='b')
            ax2_twin.tick_params(axis='y', labelcolor='g')
            ax2.set_xlim(0, num_ticks)
            ax2.set_ylim(0, num_agents)
            ax2_twin.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Population & Resource Quality', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, update, frames=num_ticks, interval=1000//fps)
        
        print(f'Saving animation to {output_path}...')
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()
        
        print(f'✓ Animation saved: {output_path}')
        
        print('\n=== FINAL RESULTS ===')
        print(f'Survival: {alive_history[-1]}/{num_agents} ({100*alive_history[-1]/num_agents:.1f}%)')
        print(f'Initial mean vegetation: {mean_veg_history[0]:.3f}')
        print(f'Final mean vegetation: {mean_veg_history[-1]:.3f}')
        print(f'Improvement: {mean_veg_history[-1] - mean_veg_history[0]:+.3f}')
        
        if mean_veg_history[-1] > mean_veg_history[0] + 0.05:
            print('✓ Strong evidence of food-seeking migration!')
        elif mean_veg_history[-1] > mean_veg_history[0]:
            print('~ Weak preference for higher vegetation')
        else:
            print('✗ No clear migration toward food')
        
        return {
            'alive_history': alive_history,
            'mean_veg_history': mean_veg_history,
            'trajectories': trajectories,
            'energy_history': energy_history
        }

if __name__ == '__main__':
    create_migration_animation(
        num_agents=30,
        num_ticks=150,
        num_predators=3,
        initial_energy=50.0,
        seed=42,
        output_path='band1_migration.gif',
        fps=10
    )

