"""
Test multi-resource optimization: Do agents balance food AND water needs?
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def test_multi_resource():
    print('Loading environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='multi')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        hydration = tensor[:, :, 1]
        temperature = tensor[:, :, 0]
        h, w = vegetation.shape
        
        print(f'\nEnvironment statistics:')
        print(f'Vegetation: min={vegetation.min():.3f}, max={vegetation.max():.3f}, mean={vegetation.mean():.3f}')
        print(f'Hydration:  min={hydration.min():.3f}, max={hydration.max():.3f}, mean={hydration.mean():.3f}')
        print(f'Temperature: min={temperature.min():.3f}, max={temperature.max():.3f}, mean={temperature.mean():.3f}')
        
        # Find areas with HIGH food but LOW water (trade-off zones)
        high_food_low_water = (vegetation > 0.6) & (hydration < 0.7)
        low_food_high_water = (vegetation < 0.3) & (hydration > 0.9)
        
        print(f'\nResource distribution:')
        print(f'High food, low water: {high_food_low_water.sum()} cells')
        print(f'Low food, high water: {low_food_high_water.sum()} cells')
        
        # Spawn agents in area with food but no water
        high_food_coords = np.argwhere(high_food_low_water)
        
        if len(high_food_coords) < 20:
            print('Not enough high-food/low-water areas, using high food areas instead')
            high_food_coords = np.argwhere(vegetation > 0.6)
        
        print(f'\nSpawning 20 agents in HIGH FOOD, LOW WATER areas...')
        
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators=1, seed=42)
        
        rng = np.random.default_rng(42)
        spawn_indices = rng.choice(len(high_food_coords), size=20, replace=False)
        
        for i, idx in enumerate(spawn_indices):
            y, x = high_food_coords[idx]
            from interfaces.agent_iface.banded_agent import BandedAgent
            agent = BandedAgent(agent_id=i, x=int(x), y=int(y), initial_energy=70.0,
                               seed=rng.integers(0, 1000000))
            # Set initial thirst HIGH to force water-seeking
            agent.bands[0].state.internal_state["thirst"] = 0.7
            sim.agents.append(agent)
        
        # Track what agents are focused on
        num_ticks = 100
        focus_history = []
        
        print(f'\nRunning {num_ticks} ticks and tracking focus...')
        for tick in range(num_ticks):
            alive = [a for a in sim.agents if a.state.alive]
            if not alive:
                break
            
            # Track what each agent is focused on
            hunger_focused = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "hunger")
            thirst_focused = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "thirst")
            other_focused = len(alive) - hunger_focused - thirst_focused
            
            # Track average drives
            avg_hunger = np.mean([a.bands[0].state.internal_state.get("hunger", 0) for a in alive])
            avg_thirst = np.mean([a.bands[0].state.internal_state.get("thirst", 0) for a in alive])
            
            focus_history.append({
                'tick': tick,
                'hunger_focused': hunger_focused,
                'thirst_focused': thirst_focused,
                'other_focused': other_focused,
                'avg_hunger': avg_hunger,
                'avg_thirst': avg_thirst,
                'alive': len(alive)
            })
            
            if tick % 20 == 0:
                print(f'  T={tick}: {len(alive)} alive, hunger_focus={hunger_focused}, thirst_focus={thirst_focused}, ' +
                      f'hunger={avg_hunger:.2f}, thirst={avg_thirst:.2f}')
            
            sim.step()
        
        # Visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Panel 1: Vegetation field
        ax = plt.subplot(2, 3, 1)
        im = ax.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=0.8)
        ax.set_title('Vegetation (Food)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # Panel 2: Hydration field  
        ax = plt.subplot(2, 3, 2)
        im = ax.imshow(hydration, cmap='Blues', origin='upper', vmin=0.4, vmax=1.0)
        ax.set_title('Hydration (Water)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # Panel 3: Combined (resource overlap)
        ax = plt.subplot(2, 3, 3)
        # Create RGB image: R=vegetation, B=hydration
        rgb = np.zeros((h, w, 3))
        rgb[:,:,1] = vegetation / 0.8  # Green channel for food
        rgb[:,:,2] = (hydration - 0.4) / 0.6  # Blue channel for water
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(rgb, origin='upper')
        ax.set_title('Combined Resources\n(Green=Food, Blue=Water)', fontsize=14, fontweight='bold')
        
        # Panel 4: Focus over time
        ax = plt.subplot(2, 3, 4)
        ticks = [d['tick'] for d in focus_history]
        hunger_counts = [d['hunger_focused'] for d in focus_history]
        thirst_counts = [d['thirst_focused'] for d in focus_history]
        other_counts = [d['other_focused'] for d in focus_history]
        
        ax.plot(ticks, hunger_counts, 'orange', linewidth=2, label='Hunger Focus')
        ax.plot(ticks, thirst_counts, 'blue', linewidth=2, label='Thirst Focus')
        ax.plot(ticks, other_counts, 'gray', linewidth=2, label='Other Focus')
        ax.set_xlabel('Tick', fontsize=12)
        ax.set_ylabel('Number of Agents', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Attentional Focus Distribution', fontsize=14, fontweight='bold')
        
        # Panel 5: Drive levels over time
        ax = plt.subplot(2, 3, 5)
        avg_hungers = [d['avg_hunger'] for d in focus_history]
        avg_thirsts = [d['avg_thirst'] for d in focus_history]
        
        ax.plot(ticks, avg_hungers, 'orange', linewidth=2, label='Hunger')
        ax.plot(ticks, avg_thirsts, 'blue', linewidth=2, label='Thirst')
        ax.set_xlabel('Tick', fontsize=12)
        ax.set_ylabel('Drive Level (0-1)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Drive Levels', fontsize=14, fontweight='bold')
        
        # Panel 6: Summary text
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        total_hunger_focus = sum(d['hunger_focused'] for d in focus_history)
        total_thirst_focus = sum(d['thirst_focused'] for d in focus_history)
        total_focus = total_hunger_focus + total_thirst_focus
        
        if total_focus > 0:
            hunger_pct = 100 * total_hunger_focus / total_focus
            thirst_pct = 100 * total_thirst_focus / total_focus
        else:
            hunger_pct = thirst_pct = 0
        
        summary = f"""MULTI-RESOURCE OPTIMIZATION TEST

Question: Do agents balance food AND water needs?

Focus Distribution:
  Hunger: {hunger_pct:.1f}% of focus
  Thirst: {thirst_pct:.1f}% of focus

Interpretation:
"""
        
        if thirst_pct > 10:
            summary += "  ✓ Agents ARE considering thirst\n"
            summary += "  ✓ Multi-need optimization working\n"
        else:
            summary += "  ✗ Agents ignoring thirst (<10%)\n"
            summary += "  ✗ Only optimizing for food\n"
        
        if abs(hunger_pct - thirst_pct) < 20:
            summary += "  ✓ Balanced attention to both needs\n"
        elif hunger_pct > thirst_pct:
            summary += f"  ⚠ Hunger-dominant ({hunger_pct:.0f}% vs {thirst_pct:.0f}%)\n"
        else:
            summary += f"  ⚠ Thirst-dominant ({thirst_pct:.0f}% vs {hunger_pct:.0f}%)\n"
        
        ax.text(0.1, 0.5, summary, fontsize=13, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Band 1: Multi-Resource Optimization Test', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('multi_resource_test.png', dpi=150)
        print('\n✓ Saved: multi_resource_test.png')
        
        print(f'\n=== ANALYSIS ===')
        print(f'Total hunger-focused decisions: {total_hunger_focus} ({hunger_pct:.1f}%)')
        print(f'Total thirst-focused decisions: {total_thirst_focus} ({thirst_pct:.1f}%)')
        
        if thirst_pct < 5:
            print('\n⚠ WARNING: Agents are NOT seeking water!')
            print('Thirst mechanism may not be working properly.')
        elif thirst_pct > 30:
            print('\n✓ Good: Agents actively balance food and water needs')
        else:
            print('\n~ Moderate: Some water-seeking but hunger-dominant')

if __name__ == '__main__':
    test_multi_resource()

