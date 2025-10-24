"""
Quick static visualization: Before/After migration comparison.
Much faster than animation - runs in ~1 minute instead of hours.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from interfaces.agent_iface.simulation import AgentSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def quick_migration_test(num_agents=40, num_ticks=80, output='migration_summary.png'):
    print('Generating environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=num_ticks, out_dir=tmpdir, label='quick_test')
        tensor = hydrate_tick(run_dir, 0)
        vegetation = tensor[:, :, 2]
        
        print(f'Running simulation ({num_agents} agents, {num_ticks} ticks)...')
        sim = AgentSimulation(run_dir, num_predators=2, seed=42)
        sim.spawn_agents(num_agents=num_agents, initial_energy=50.0)
        
        initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
        initial_energy = [a.state.energy for a in sim.agents]
        
        for tick in range(num_ticks):
            sim.step()
            if tick % 20 == 0:
                alive = sum(1 for a in sim.agents if a.state.alive)
                print(f'  Tick {tick}: {alive}/{num_agents} alive')
        
        final_pos = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
        final_energy = [a.state.energy for a in sim.agents if a.state.alive]
        
        initial_veg = [vegetation[y, x] for x, y in initial_pos]
        final_veg = [vegetation[y, x] for x, y in final_pos]
        
        print('\nCreating visualization...')
        fig = plt.figure(figsize=(18, 6))
        
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8)
        ax1.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
                   c=initial_energy, cmap='RdYlGn', s=100, edgecolors='black',
                   vmin=0, vmax=100, alpha=0.8)
        ax1.set_title(f'Initial Positions (t=0)\nMean vegetation: {np.mean(initial_veg):.3f}',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, label='Vegetation')
        
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8)
        sc = ax2.scatter([p[0] for p in final_pos], [p[1] for p in final_pos],
                        c=final_energy, cmap='RdYlGn', s=100, edgecolors='black',
                        vmin=0, vmax=100, alpha=0.8)
        ax2.set_title(f'Final Positions (t={num_ticks})\nMean vegetation: {np.mean(final_veg):.3f}',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        cbar = plt.colorbar(sc, ax=ax2, label='Agent Energy')
        
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(initial_veg, bins=20, alpha=0.5, label=f'Initial (μ={np.mean(initial_veg):.3f})',
                color='red', edgecolor='black')
        ax3.hist(final_veg, bins=20, alpha=0.5, label=f'Final (μ={np.mean(final_veg):.3f})',
                color='green', edgecolor='black')
        ax3.axvline(np.mean(initial_veg), color='red', linestyle='--', linewidth=2)
        ax3.axvline(np.mean(final_veg), color='green', linestyle='--', linewidth=2)
        ax3.set_xlabel('Vegetation Level', fontsize=12)
        ax3.set_ylabel('Number of Agents', fontsize=12)
        ax3.set_title('Distribution of Agent Locations', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Band 1 Migration: Gradient-Following Food-Seeking Behavior\n' +
                    f'Survival: {len(final_pos)}/{num_agents} ({100*len(final_pos)/num_agents:.0f}%) | ' +
                    f'Vegetation Improvement: {np.mean(final_veg)-np.mean(initial_veg):+.3f}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {output}')
        
        print(f'\n=== RESULTS ===')
        print(f'Survival: {len(final_pos)}/{num_agents} ({100*len(final_pos)/num_agents:.1f}%)')
        print(f'Initial mean vegetation: {np.mean(initial_veg):.3f}')
        print(f'Final mean vegetation: {np.mean(final_veg):.3f}')
        print(f'Improvement: {np.mean(final_veg)-np.mean(initial_veg):+.3f}')
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(final_veg, initial_veg)
        print(f'\nStatistical Test (t-test):')
        print(f'  t-statistic: {t_stat:.3f}')
        print(f'  p-value: {p_value:.4f}')
        if p_value < 0.05:
            print(f'  ✓ Significant migration toward food (p < 0.05)')
        else:
            print(f'  ✗ Not statistically significant')

if __name__ == '__main__':
    quick_migration_test(num_agents=40, num_ticks=80, output='migration_summary.png')

