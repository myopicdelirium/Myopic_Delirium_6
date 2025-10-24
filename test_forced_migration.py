"""
Controlled test: Spawn agents in LOW vegetation, see if they migrate to HIGH vegetation.
This removes random starting position confound.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def test_forced_migration():
    print('Loading environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='forced')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        temperature = tensor[:, :, 0]
        hydration = tensor[:, :, 1]
        h, w = vegetation.shape
        
        # Find low and high vegetation zones
        low_veg_mask = vegetation < 0.3
        high_veg_mask = vegetation > 0.6
        
        low_coords = np.argwhere(low_veg_mask)
        high_coords = np.argwhere(high_veg_mask)
        
        print(f'Low vegetation cells: {len(low_coords)}')
        print(f'High vegetation cells: {len(high_coords)}')
        
        # Create simulation
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators=1, seed=42)
        
        # Spawn ALL agents in LOW vegetation areas
        num_agents = 30
        rng = np.random.default_rng(42)
        spawn_indices = rng.choice(len(low_coords), size=num_agents, replace=False)
        
        print(f'\nSpawning {num_agents} agents in LOW vegetation areas...')
        for i, idx in enumerate(spawn_indices):
            y, x = low_coords[idx]
            from interfaces.agent_iface.banded_agent import BandedAgent
            agent = BandedAgent(agent_id=i, x=int(x), y=int(y), initial_energy=40.0,
                               seed=rng.integers(0, 1000000))
            sim.agents.append(agent)
        
        initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
        initial_veg = [vegetation[y, x] for x, y in initial_pos]
        
        print(f'Initial mean vegetation: {np.mean(initial_veg):.3f}')
        
        # Run simulation
        num_ticks = 120
        print(f'\nRunning {num_ticks} ticks...')
        for tick in range(num_ticks):
            sim.step()
            if tick % 30 == 0:
                alive = sum(1 for a in sim.agents if a.state.alive)
                positions = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
                mean_veg = np.mean([vegetation[y, x] for x, y in positions]) if positions else 0
                print(f'  Tick {tick}: {alive}/{num_agents} alive, μ_veg={mean_veg:.3f}')
        
        final_pos = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
        final_veg = [vegetation[y, x] for x, y in final_pos]
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        ax = axes[0]
        ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8)
        ax.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
                  c='red', s=100, edgecolors='black', linewidth=2, label='Start (low veg)')
        ax.set_title(f'Initial: All agents in LOW vegetation\nμ={np.mean(initial_veg):.3f}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        ax = axes[1]
        ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8)
        ax.scatter([p[0] for p in final_pos], [p[1] for p in final_pos],
                  c='yellow', s=100, edgecolors='black', linewidth=2, label='End')
        ax.set_title(f'Final (t={num_ticks}): Migration result\nμ={np.mean(final_veg):.3f}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        ax = axes[2]
        ax.hist([initial_veg, final_veg], bins=15, label=['Initial', 'Final'],
               color=['red', 'green'], alpha=0.6, edgecolor='black')
        ax.axvline(np.mean(initial_veg), color='red', linestyle='--', linewidth=3)
        ax.axvline(np.mean(final_veg), color='green', linestyle='--', linewidth=3)
        ax.set_xlabel('Vegetation Level', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        improvement = np.mean(final_veg) - np.mean(initial_veg)
        plt.suptitle(f'Forced Migration Test: Start LOW → End ???\n' +
                    f'Δμ_veg = {improvement:+.3f} | Survival: {len(final_pos)}/{num_agents}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig('forced_migration.png', dpi=150)
        print('\n✓ Saved: forced_migration.png')
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(final_veg, initial_veg)
        
        print(f'\n=== RESULTS ===')
        print(f'Initial μ_veg: {np.mean(initial_veg):.3f} (started in desert)')
        print(f'Final μ_veg:   {np.mean(final_veg):.3f}')
        print(f'Improvement:   {improvement:+.3f}')
        print(f't-test: t={t_stat:.3f}, p={p_value:.4f}')
        
        if improvement > 0.1 and p_value < 0.05:
            print('✓✓ STRONG migration toward food!')
        elif improvement > 0.05:
            print('✓ Moderate migration')
        elif improvement > 0:
            print('~ Weak migration')
        else:
            print('✗ No migration (agents stayed in desert)')

if __name__ == '__main__':
    test_forced_migration()

