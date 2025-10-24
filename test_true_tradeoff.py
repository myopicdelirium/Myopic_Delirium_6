"""
True trade-off test: Environment with food in south, water in north.
Agents must choose between satisfying hunger or thirst.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def test_true_tradeoff():
    print('Creating trade-off environment (food south, water north)...')
    scenario_path = 'interfaces/ui_iface/scenarios/tradeoff-env.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='tradeoff')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        hydration = tensor[:, :, 1]
        temperature = tensor[:, :, 0]
        h, w = vegetation.shape
        
        print(f'\nEnvironment verification:')
        print(f'Top half (y<128) - Water zone:')
        print(f'  Hydration: {hydration[:128, :].mean():.3f}')
        print(f'  Vegetation: {vegetation[:128, :].mean():.3f}')
        print(f'Bottom half (y>=128) - Food zone:')
        print(f'  Hydration: {hydration[128:, :].mean():.3f}')
        print(f'  Vegetation: {vegetation[128:, :].mean():.3f}')
        
        # Spawn agents in CENTER (moderate food and water)
        center_y = h // 2
        center_band = 20  # 20 pixels above/below center
        center_mask = (np.arange(h)[:, None] > center_y - center_band) & \
                      (np.arange(h)[:, None] < center_y + center_band)
        center_mask = np.broadcast_to(center_mask, (h, w))
        center_coords = np.argwhere(center_mask)
        
        print(f'\nSpawning 30 agents in CENTER zone...')
        
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators=1, seed=42)
        
        num_agents = 30
        rng = np.random.default_rng(42)
        spawn_indices = rng.choice(len(center_coords), size=num_agents, replace=False)
        
        for i, idx in enumerate(spawn_indices):
            y, x = center_coords[idx]
            from interfaces.agent_iface.banded_agent import BandedAgent
            agent = BandedAgent(agent_id=i, x=int(x), y=int(y), initial_energy=50.0,
                               seed=rng.integers(0, 1000000))
            # Start with moderate hunger and thirst
            agent.bands[0].state.internal_state["hunger"] = 0.4
            agent.bands[0].state.internal_state["thirst"] = 0.4
            sim.agents.append(agent)
        
        initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
        
        # Track trajectories and focus
        num_ticks = 200
        trajectories = [[] for _ in range(num_agents)]
        focus_history = []
        focus_switches = 0
        
        print(f'\nRunning {num_ticks} ticks...')
        for tick in range(num_ticks):
            alive = [a for a in sim.agents if a.state.alive]
            if not alive:
                print(f'All agents dead at tick {tick}')
                break
            
            # Track trajectories
            for i, agent in enumerate(sim.agents):
                if agent.state.alive:
                    trajectories[i].append((agent.state.x, agent.state.y))
            
            # Track focus switches
            for a in alive:
                current_focus = a.bands[0].state.internal_state.get("current_focus")
                last_focus = a.bands[0].state.internal_state.get("last_focus", current_focus)
                if current_focus != last_focus and last_focus is not None:
                    focus_switches += 1
                a.bands[0].state.internal_state["last_focus"] = current_focus
            
            hunger_focused = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "hunger")
            thirst_focused = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "thirst")
            
            avg_hunger = np.mean([a.bands[0].state.internal_state.get("hunger", 0) for a in alive])
            avg_thirst = np.mean([a.bands[0].state.internal_state.get("thirst", 0) for a in alive])
            avg_y = np.mean([a.state.y for a in alive])
            
            focus_history.append({
                'tick': tick,
                'hunger_focused': hunger_focused,
                'thirst_focused': thirst_focused,
                'avg_hunger': avg_hunger,
                'avg_thirst': avg_thirst,
                'avg_y': avg_y,
                'alive': len(alive)
            })
            
            if tick % 40 == 0:
                print(f'  T={tick}: {len(alive)} alive, H={hunger_focused}, T={thirst_focused}, ' +
                      f'avg_y={avg_y:.0f}, hunger={avg_hunger:.2f}, thirst={avg_thirst:.2f}')
            
            sim.step()
        
        # Analysis
        final_positions = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
        
        if final_positions:
            final_y_positions = [y for x, y in final_positions]
            went_north = sum(1 for y in final_y_positions if y < center_y - 30)  # Water seekers
            went_south = sum(1 for y in final_y_positions if y > center_y + 30)  # Food seekers
            stayed_center = len(final_positions) - went_north - went_south
        else:
            went_north = went_south = stayed_center = 0
        
        total_hunger = sum(d['hunger_focused'] for d in focus_history)
        total_thirst = sum(d['thirst_focused'] for d in focus_history)
        total_focus = total_hunger + total_thirst
        
        print(f'\n=== RESULTS ===')
        print(f'Survival: {len(final_positions)}/{num_agents} ({100*len(final_positions)/num_agents:.0f}%)')
        print(f'Migration:')
        print(f'  Went NORTH (water): {went_north}')
        print(f'  Went SOUTH (food): {went_south}')
        print(f'  Stayed CENTER: {stayed_center}')
        print(f'Focus distribution:')
        if total_focus > 0:
            print(f'  Hunger: {100*total_hunger/total_focus:.1f}%')
            print(f'  Thirst: {100*total_thirst/total_focus:.1f}%')
        print(f'Focus switches: {focus_switches} ({focus_switches/num_agents:.1f} per agent)')
        
        # Visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Panel 1: Environment with trajectories
        ax = plt.subplot(2, 3, 1)
        # Create RGB: Green=food, Blue=water
        rgb = np.zeros((h, w, 3))
        rgb[:,:,1] = vegetation / 0.8
        rgb[:,:,2] = hydration
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(rgb, origin='upper', aspect='auto')
        
        # Draw trajectories
        for traj in trajectories:
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, 'yellow', alpha=0.3, linewidth=1)
        
        # Draw initial and final positions
        ax.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
                  c='red', s=50, marker='x', linewidth=2, label='Start', zorder=5)
        if final_positions:
            ax.scatter([p[0] for p in final_positions], [p[1] for p in final_positions],
                      c='white', s=100, marker='*', edgecolors='black', linewidth=2, label='End', zorder=5)
        
        ax.axhline(center_y, color='white', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(10, 30, 'WATER ZONE', color='white', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
        ax.text(10, h-30, 'FOOD ZONE', color='white', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        ax.set_title('Agent Trajectories\n(Green=Food, Blue=Water)', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Panel 2: Focus over time
        ax = plt.subplot(2, 3, 2)
        ticks = [d['tick'] for d in focus_history]
        ax.plot(ticks, [d['hunger_focused'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
        ax.plot(ticks, [d['thirst_focused'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
        ax.set_xlabel('Tick')
        ax.set_ylabel('# Agents Focused')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Attentional Focus Distribution', fontsize=14, fontweight='bold')
        
        # Panel 3: Drives over time
        ax = plt.subplot(2, 3, 3)
        ax.plot(ticks, [d['avg_hunger'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
        ax.plot(ticks, [d['avg_thirst'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
        ax.set_xlabel('Tick')
        ax.set_ylabel('Drive Level (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Drive Levels', fontsize=14, fontweight='bold')
        
        # Panel 4: Average Y position (migration direction)
        ax = plt.subplot(2, 3, 4)
        ax.plot(ticks, [d['avg_y'] for d in focus_history], 'purple', linewidth=2)
        ax.axhline(center_y, color='red', linestyle='--', label='Center')
        ax.axhline(center_y - 30, color='blue', linestyle=':', alpha=0.7, label='Water Zone')
        ax.axhline(center_y + 30, color='green', linestyle=':', alpha=0.7, label='Food Zone')
        ax.set_xlabel('Tick')
        ax.set_ylabel('Average Y Position')
        ax.set_ylim([0, h])
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Population Movement\n(Lower=North/Water, Higher=South/Food)', fontsize=14, fontweight='bold')
        
        # Panel 5: Final distribution
        ax = plt.subplot(2, 3, 5)
        categories = ['North\n(Water)', 'Center', 'South\n(Food)']
        counts = [went_north, stayed_center, went_south]
        colors = ['blue', 'gray', 'green']
        ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Number of Agents')
        ax.set_title('Final Locations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 6: Summary
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        switches_per_agent = focus_switches / num_agents
        
        summary = f'''TRUE TRADE-OFF TEST

Environment:
  North: HIGH water, LOW food
  South: HIGH food, LOW water
  Center: Moderate both

Results:
  Survival: {len(final_positions)}/{num_agents}
  Went to water: {went_north}
  Went to food: {went_south}
  Stayed center: {stayed_center}
  
  Focus switches: {switches_per_agent:.1f}/agent
  
Interpretation:
'''
        
        if switches_per_agent > 3:
            summary += '  ✓ HIGH switching (dynamic optimization)\\n'
        elif switches_per_agent > 1:
            summary += '  ✓ Moderate switching\\n'
        else:
            summary += '  ~ Low switching\\n'
        
        if went_north > 0 and went_south > 0:
            summary += '  ✓ Agents chose DIFFERENT strategies\\n'
        else:
            summary += '  ⚠ All agents made same choice\\n'
        
        if stayed_center > len(final_positions) * 0.3:
            summary += '  ✓ Some balanced both needs\\n'
        
        ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        plt.suptitle('True Multi-Resource Trade-Off: Spatially Separated Food and Water',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('true_tradeoff.png', dpi=150)
        print('\n✓ Saved: true_tradeoff.png')

if __name__ == '__main__':
    test_true_tradeoff()

