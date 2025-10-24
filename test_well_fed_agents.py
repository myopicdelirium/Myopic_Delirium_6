"""
Control test: Agents in GOOD conditions should NOT migrate much.
This proves migration is emergent from desperation, not hard-coded.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def test_well_fed_agents():
    print('Loading environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='well_fed')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        temperature = tensor[:, :, 0]
        hydration = tensor[:, :, 1]
        h, w = vegetation.shape
        
        # Find RICH zones (high vegetation)
        rich_mask = vegetation > 0.65
        rich_coords = np.argwhere(rich_mask)
        
        print(f'Rich zones (veg > 0.65): {len(rich_coords)}')
        
        # Create simulation
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators=1, seed=42)
        
        # Spawn agents in RICH AREAS with HIGH energy (well-fed)
        num_agents = 20
        rng = np.random.default_rng(42)
        spawn_indices = rng.choice(len(rich_coords), size=num_agents, replace=False)
        
        print(f'\nSpawning {num_agents} agents in RICH areas with HIGH energy...')
        for i, idx in enumerate(spawn_indices):
            y, x = rich_coords[idx]
            from interfaces.agent_iface.banded_agent import BandedAgent
            # High initial energy = low desperation
            agent = BandedAgent(agent_id=i, x=int(x), y=int(y), initial_energy=90.0,
                               seed=rng.integers(0, 1000000))
            # Set low hunger (well-fed)
            agent.bands[0].state.internal_state["hunger"] = 0.1
            sim.agents.append(agent)
        
        initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
        initial_veg = [vegetation[y, x] for x, y in initial_pos]
        
        # Track metrics over time
        num_ticks = 300
        tick_data = []
        
        print(f'\nRunning {num_ticks} ticks with well-fed agents...')
        for tick in range(num_ticks + 1):
            alive_agents = [a for a in sim.agents if a.state.alive]
            if not alive_agents:
                print(f'  All agents dead at tick {tick}')
                break
            
            positions = [(a.state.x, a.state.y) for a in alive_agents]
            energies = [a.state.energy for a in alive_agents]
            veg_values = [vegetation[y, x] for x, y in positions]
            
            # Track Band 1 internal states
            hungers = [a.bands[0].state.internal_state.get("hunger", 0.0) for a in alive_agents]
            desperations = [a.bands[0].state.internal_state.get("desperation_level", 0.0) for a in alive_agents]
            focuses = [a.bands[0].state.internal_state.get("current_focus", "none") for a in alive_agents]
            
            tick_data.append({
                'tick': tick,
                'alive': len(alive_agents),
                'mean_veg': np.mean(veg_values),
                'mean_energy': np.mean(energies),
                'mean_hunger': np.mean(hungers),
                'mean_desperation': np.mean(desperations),
                'focus_hunger_pct': sum(1 for f in focuses if f == "hunger") / len(focuses) if focuses else 0
            })
            
            if tick % 50 == 0:
                print(f'  T={tick}: {len(alive_agents)}/{num_agents} alive, '
                      f'μ_veg={np.mean(veg_values):.3f}, '
                      f'μ_hunger={np.mean(hungers):.2f}, '
                      f'μ_desp={np.mean(desperations):.2f}')
            
            if tick < num_ticks:
                sim.step()
        
        final_pos = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
        final_veg = [vegetation[y, x] for x, y in final_pos] if final_pos else []
        
        # Calculate distances
        distances = []
        for init in initial_pos:
            # Find closest final position (some agents may have died)
            if final_pos:
                min_dist = min(np.sqrt((fp[0] - init[0])**2 + (fp[1] - init[1])**2) for fp in final_pos)
                distances.append(min_dist)
        
        # Visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Panel 1: Initial positions
        ax = plt.subplot(2, 3, 1)
        ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
        ax.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
                  c='gold', s=150, marker='*', edgecolors='black', linewidth=2, label='Start (rich area)')
        ax.set_title(f'T=0: Well-Fed in RICH Areas\nμ_veg={np.mean(initial_veg):.3f}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        # Panel 2: Final positions
        ax = plt.subplot(2, 3, 2)
        ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8, alpha=0.9)
        if final_pos:
            final_energies = [a.state.energy for a in sim.agents if a.state.alive]
            ax.scatter([p[0] for p in final_pos], [p[1] for p in final_pos],
                      c=final_energies, cmap='RdYlGn', s=150, edgecolors='black',
                      vmin=0, vmax=100, linewidth=2)
            ax.set_title(f'T={num_ticks}: Final Positions\nμ_veg={np.mean(final_veg):.3f}',
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'T={num_ticks}: All Dead', fontsize=14, fontweight='bold', color='red')
        
        # Panel 3: Vegetation over time
        ax = plt.subplot(2, 3, 3)
        ticks = [d['tick'] for d in tick_data]
        mean_vegs = [d['mean_veg'] for d in tick_data]
        ax.plot(ticks, mean_vegs, 'g-', linewidth=3, label='Mean Vegetation')
        ax.axhline(np.mean(initial_veg), color='gold', linestyle='--', linewidth=2, label='Initial')
        ax.set_xlabel('Tick', fontsize=12)
        ax.set_ylabel('Mean Vegetation', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Resource Quality Over Time', fontsize=13, fontweight='bold')
        
        # Panel 4: Hunger & Desperation
        ax = plt.subplot(2, 3, 4)
        hungers = [d['mean_hunger'] for d in tick_data]
        desperations = [d['mean_desperation'] for d in tick_data]
        ax.plot(ticks, hungers, 'orange', linewidth=2, label='Hunger')
        ax.plot(ticks, desperations, 'red', linewidth=2, label='Desperation')
        ax.set_xlabel('Tick', fontsize=12)
        ax.set_ylabel('Level (0-1)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Homeostatic Drives (Should Stay LOW)', fontsize=13, fontweight='bold')
        
        # Panel 5: Population & Energy
        ax = plt.subplot(2, 3, 5)
        alive_counts = [d['alive'] for d in tick_data]
        mean_energies = [d['mean_energy'] for d in tick_data]
        ax.plot(ticks, alive_counts, 'b-', linewidth=2, label='Alive')
        ax.set_ylabel('Alive Count', fontsize=12, color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax2 = ax.twinx()
        ax2.plot(ticks, mean_energies, 'purple', linewidth=2, label='Mean Energy')
        ax2.set_ylabel('Energy', fontsize=12, color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax.set_xlabel('Tick', fontsize=12)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Population Dynamics', fontsize=13, fontweight='bold')
        
        # Panel 6: Focus distribution
        ax = plt.subplot(2, 3, 6)
        focus_pcts = [d['focus_hunger_pct'] * 100 for d in tick_data]
        ax.plot(ticks, focus_pcts, 'darkgreen', linewidth=2)
        ax.set_xlabel('Tick', fontsize=12)
        ax.set_ylabel('% Focused on Hunger', fontsize=12)
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        ax.set_title('Attentional Focus (Should be LOW)', fontsize=13, fontweight='bold')
        
        if final_veg:
            improvement = np.mean(final_veg) - np.mean(initial_veg)
            mean_distance = np.mean(distances) if distances else 0
        else:
            improvement = 0
            mean_distance = 0
        
        survival_rate = (len(final_pos) / num_agents * 100) if num_agents > 0 else 0
        
        plt.suptitle(f'Control Test: Well-Fed Agents Should NOT Migrate\n' +
                    f'Δμ_veg = {improvement:+.3f} | Mean Distance = {mean_distance:.1f} cells | ' +
                    f'Survival: {len(final_pos)}/{num_agents} ({survival_rate:.0f}%)',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('well_fed_control.png', dpi=150)
        print('\n✓ Saved: well_fed_control.png')
        
        print(f'\n=== RESULTS ===')
        print(f'Initial μ_veg:  {np.mean(initial_veg):.3f} (rich area)')
        if final_veg:
            print(f'Final μ_veg:    {np.mean(final_veg):.3f}')
            print(f'Change:         {improvement:+.3f}')
            print(f'Mean distance:  {mean_distance:.1f} cells')
        else:
            print(f'Final μ_veg:    N/A (all dead)')
        print(f'Survival rate:  {survival_rate:.1f}%')
        
        print(f'\nINTERPRETATION:')
        if abs(improvement) < 0.05 and mean_distance < 30:
            print('✓✓✓ CORRECT: Well-fed agents stayed put (no migration pressure)')
        elif abs(improvement) < 0.1 and mean_distance < 50:
            print('✓✓ Good: Minimal migration (low desperation)')
        elif mean_distance < 80:
            print('✓ Some migration (hunger accumulated over time)')
        else:
            print('⚠ Warning: High migration despite good conditions')
        
        # Compare to desert test
        print(f'\nCOMPARISON TO DESERT TEST:')
        print(f'Desert agents: migrated ~92.6 cells (desperate)')
        print(f'Well-fed agents: migrated {mean_distance:.1f} cells')
        print(f'Ratio: {mean_distance / 92.6:.2f}x less migration')

if __name__ == '__main__':
    test_well_fed_agents()

