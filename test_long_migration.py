"""
Long-term migration test: 1000 ticks to see if agents can travel far distances.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def test_long_migration():
    print('Loading environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='long')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        temperature = tensor[:, :, 0]
        hydration = tensor[:, :, 1]
        h, w = vegetation.shape
        
        print(f'Grid size: {w} x {h}')
        
        # Find low and high vegetation zones
        low_veg_mask = vegetation < 0.25
        high_veg_mask = vegetation > 0.65
        
        low_coords = np.argwhere(low_veg_mask)
        high_coords = np.argwhere(high_veg_mask)
        
        print(f'Low vegetation cells: {len(low_coords)}')
        print(f'High vegetation cells: {len(high_coords)}')
        
        # Create simulation
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators=1, seed=42)
        
        # Spawn agents in LOW vegetation areas
        num_agents = 40
        rng = np.random.default_rng(42)
        spawn_indices = rng.choice(len(low_coords), size=num_agents, replace=False)
        
        print(f'\nSpawning {num_agents} agents in LOW vegetation areas...')
        for i, idx in enumerate(spawn_indices):
            y, x = low_coords[idx]
            from interfaces.agent_iface.banded_agent import BandedAgent
            agent = BandedAgent(agent_id=i, x=int(x), y=int(y), initial_energy=60.0,
                               seed=rng.integers(0, 1000000))
            sim.agents.append(agent)
        
        initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
        initial_veg = [vegetation[y, x] for x, y in initial_pos]
        
        print(f'Initial mean vegetation: {np.mean(initial_veg):.3f}')
        
        # Track migration over time
        num_ticks = 1000
        checkpoint_ticks = [0, 100, 200, 500, 1000]
        checkpoints = {}
        
        print(f'\nRunning {num_ticks} ticks...')
        for tick in range(num_ticks + 1):
            if tick in checkpoint_ticks:
                positions = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
                energies = [a.state.energy for a in sim.agents if a.state.alive]
                veg_values = [vegetation[y, x] for x, y in positions]
                checkpoints[tick] = {
                    'positions': positions,
                    'energies': energies,
                    'vegetation': veg_values,
                    'alive': len(positions)
                }
                print(f'  Tick {tick}: {len(positions)}/{num_agents} alive, μ_veg={np.mean(veg_values):.3f}')
            
            if tick < num_ticks:
                sim.step()
        
        # Calculate distances traveled
        final_pos = checkpoints[1000]['positions']
        distances = []
        for i, (init, final) in enumerate(zip(initial_pos, final_pos)):
            if i < len(final_pos):  # Agent survived
                dist = np.sqrt((final[0] - init[0])**2 + (final[1] - init[1])**2)
                distances.append(dist)
        
        # Visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Panel 1: Initial positions
        ax = plt.subplot(2, 3, 1)
        ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
        ax.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
                  c='red', s=80, edgecolors='black', linewidth=1.5, label='Start')
        ax.set_title(f'T=0: Initial\nμ_veg={np.mean(initial_veg):.3f}', fontsize=13, fontweight='bold')
        ax.legend()
        
        # Panels 2-5: Checkpoints at T=100, 200, 500, 1000
        for idx, tick in enumerate([100, 200, 500, 1000], start=2):
            ax = plt.subplot(2, 3, idx)
            ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
            
            cp = checkpoints[tick]
            ax.scatter([p[0] for p in cp['positions']], [p[1] for p in cp['positions']],
                      c=cp['energies'], cmap='RdYlGn', s=80, edgecolors='black',
                      vmin=0, vmax=100, linewidth=1.5)
            
            ax.set_title(f'T={tick}: {cp["alive"]}/{num_agents} alive\nμ_veg={np.mean(cp["vegetation"]):.3f}',
                        fontsize=13, fontweight='bold')
        
        # Panel 6: Vegetation over time
        ax = plt.subplot(2, 3, 6)
        ticks_list = sorted(checkpoints.keys())
        mean_vegs = [np.mean(checkpoints[t]['vegetation']) for t in ticks_list]
        alive_counts = [checkpoints[t]['alive'] for t in ticks_list]
        
        ax.plot(ticks_list, mean_vegs, 'g-o', linewidth=3, markersize=8, label='Mean Vegetation')
        ax.axhline(np.mean(initial_veg), color='red', linestyle='--', linewidth=2, label='Initial')
        ax.set_xlabel('Tick', fontsize=12)
        ax.set_ylabel('Mean Vegetation', fontsize=12, color='g')
        ax.tick_params(axis='y', labelcolor='g')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        ax2 = ax.twinx()
        ax2.plot(ticks_list, alive_counts, 'b-s', linewidth=2, markersize=6, label='Alive')
        ax2.set_ylabel('Alive Count', fontsize=12, color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.legend(loc='upper right')
        
        ax.set_title('Migration Progress Over Time', fontsize=13, fontweight='bold')
        
        final_veg = np.mean(checkpoints[1000]['vegetation'])
        improvement = final_veg - np.mean(initial_veg)
        mean_distance = np.mean(distances) if distances else 0
        
        plt.suptitle(f'Long-Term Migration Test: 1000 Ticks\n' +
                    f'Δμ_veg = {improvement:+.3f} | Mean Distance = {mean_distance:.1f} cells | ' +
                    f'Survival: {len(final_pos)}/{num_agents} ({100*len(final_pos)/num_agents:.0f}%)',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('long_migration.png', dpi=150)
        print('\n✓ Saved: long_migration.png')
        
        # Statistics
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(checkpoints[1000]['vegetation'], initial_veg)
        
        print(f'\n=== FINAL RESULTS ===')
        print(f'Initial μ_veg:  {np.mean(initial_veg):.3f}')
        print(f'Final μ_veg:    {final_veg:.3f}')
        print(f'Improvement:    {improvement:+.3f}')
        print(f'Mean distance:  {mean_distance:.1f} cells')
        print(f'Max distance:   {max(distances) if distances else 0:.1f} cells')
        print(f't-test: t={t_stat:.3f}, p={p_value:.6f}')
        
        if improvement > 0.15 and p_value < 0.001:
            print('✓✓✓ VERY STRONG migration!')
        elif improvement > 0.1 and p_value < 0.01:
            print('✓✓ STRONG migration!')
        elif improvement > 0.05 and p_value < 0.05:
            print('✓ Moderate migration')
        else:
            print('~ Weak migration')

if __name__ == '__main__':
    test_long_migration()

