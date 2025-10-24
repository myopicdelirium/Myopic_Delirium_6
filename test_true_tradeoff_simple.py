"""
True trade-off test: Create synthetic environment with food south, water north.
Tests improved focus switching with real spatial trade-offs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
import tempfile

def create_tradeoff_environment(size=256):
    """Create synthetic environment with food south, water north."""
    y_coords = np.arange(size)[:, None]
    x_coords = np.arange(size)[None, :]
    
    # Hydration: HIGH in north (y=0), LOW in south (y=255)
    hydration = 1.0 - (y_coords / size) * 0.7  # 1.0 at top -> 0.3 at bottom
    hydration = np.broadcast_to(hydration, (size, size)).copy()
    hydration += np.random.rand(size, size) * 0.1 - 0.05  # Small noise
    hydration = np.clip(hydration, 0.2, 1.0)
    
    # Vegetation: LOW in north, HIGH in south (OPPOSITE!)
    vegetation = (y_coords / size) * 0.7 + 0.1  # 0.1 at top -> 0.8 at bottom
    vegetation = np.broadcast_to(vegetation, (size, size)).copy()
    vegetation += np.random.rand(size, size) * 0.15 - 0.075  # Noise
    vegetation = np.clip(vegetation, 0.05, 0.8)
    
    # Temperature: mild everywhere
    temperature = np.ones((size, size)) * 0.5 + np.random.rand(size, size) * 0.1 - 0.05
    temperature = np.clip(temperature, 0.4, 0.6)
    
    return vegetation, hydration, temperature

def test_true_tradeoff():
    print('Creating synthetic trade-off environment...')
    size = 256
    vegetation, hydration, temperature = create_tradeoff_environment(size)
    
    print(f'\nEnvironment verification:')
    print(f'North (y=0-85):')
    print(f'  Water: {hydration[:85, :].mean():.3f}, Food: {vegetation[:85, :].mean():.3f}')
    print(f'Center (y=85-170):')
    print(f'  Water: {hydration[85:170, :].mean():.3f}, Food: {vegetation[85:170, :].mean():.3f}')
    print(f'South (y=170-256):')
    print(f'  Water: {hydration[170:, :].mean():.3f}, Food: {vegetation[170:, :].mean():.3f}')
    
    # Create simulation
    sim = FastStaticSimulation(vegetation, temperature, hydration, size, size, 
                               num_predators=1, seed=42)
    
    # Spawn agents in CENTER
    num_agents = 30
    center_y = size // 2
    rng = np.random.default_rng(42)
    
    print(f'\nSpawning {num_agents} agents in CENTER...')
    for i in range(num_agents):
        x = rng.integers(size//4, 3*size//4)
        y = center_y + rng.integers(-20, 20)
        
        from interfaces.agent_iface.banded_agent import BandedAgent
        agent = BandedAgent(agent_id=i, x=x, y=y, initial_energy=50.0,
                           seed=rng.integers(0, 1000000))
        # Both needs moderately urgent
        agent.bands[0].state.internal_state["hunger"] = 0.4
        agent.bands[0].state.internal_state["thirst"] = 0.4
        sim.agents.append(agent)
    
    initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
    
    # Run simulation
    num_ticks = 200
    trajectories = [[] for _ in range(num_agents)]
    focus_history = []
    focus_switches = 0
    
    print(f'\nRunning {num_ticks} ticks with adaptive focus...')
    for tick in range(num_ticks):
        alive = [a for a in sim.agents if a.state.alive]
        if not alive:
            break
        
        for i, agent in enumerate(sim.agents):
            if agent.state.alive:
                trajectories[i].append((agent.state.x, agent.state.y))
        
        # Track switches
        for a in alive:
            curr = a.bands[0].state.internal_state.get("current_focus")
            last = a.bands[0].state.internal_state.get("last_focus", curr)
            if curr != last and last is not None:
                focus_switches += 1
            a.bands[0].state.internal_state["last_focus"] = curr
        
        h_focus = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "hunger")
        t_focus = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "thirst")
        avg_h = np.mean([a.bands[0].state.internal_state.get("hunger", 0) for a in alive])
        avg_t = np.mean([a.bands[0].state.internal_state.get("thirst", 0) for a in alive])
        avg_y = np.mean([a.state.y for a in alive])
        
        focus_history.append({
            'tick': tick, 'h_focus': h_focus, 't_focus': t_focus,
            'avg_h': avg_h, 'avg_t': avg_t, 'avg_y': avg_y, 'alive': len(alive)
        })
        
        if tick % 40 == 0:
            print(f'  T={tick}: {len(alive)} alive, H={h_focus}, T={t_focus}, y={avg_y:.0f}, h={avg_h:.2f}, t={avg_t:.2f}')
        
        sim.step()
    
    # Analysis
    final_pos = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
    
    if final_pos:
        went_north = sum(1 for x, y in final_pos if y < center_y - 40)
        went_south = sum(1 for x, y in final_pos if y > center_y + 40)
        stayed_center = len(final_pos) - went_north - went_south
    else:
        went_north = went_south = stayed_center = 0
    
    total_h = sum(d['h_focus'] for d in focus_history)
    total_t = sum(d['t_focus'] for d in focus_history)
    
    print(f'\n=== RESULTS ===')
    print(f'Survival: {len(final_pos)}/{num_agents}')
    print(f'Migration: North={went_north}, South={went_south}, Center={stayed_center}')
    print(f'Focus: Hunger={100*total_h/(total_h+total_t+0.001):.1f}%, Thirst={100*total_t/(total_h+total_t+0.001):.1f}%')
    print(f'Switches: {focus_switches} total, {focus_switches/num_agents:.1f} per agent')
    
    # Visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Environment + trajectories
    ax = plt.subplot(2, 4, 1)
    rgb = np.zeros((size, size, 3))
    rgb[:,:,1] = vegetation / 0.8  # Green for food
    rgb[:,:,2] = hydration  # Blue for water
    rgb = np.clip(rgb, 0, 1)
    ax.imshow(rgb, origin='upper')
    
    for traj in trajectories:
        if len(traj) > 1:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, 'yellow', alpha=0.4, linewidth=1)
    
    ax.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos],
              c='red', s=50, marker='x', linewidth=2, zorder=5)
    if final_pos:
        ax.scatter([p[0] for p in final_pos], [p[1] for p in final_pos],
                  c='white', s=100, marker='*', edgecolors='black', linewidth=2, zorder=5)
    
    ax.axhline(center_y, color='white', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_title('Trajectories\\n(Green=Food, Blue=Water)', fontweight='bold')
    
    # Panel 2: Focus over time
    ax = plt.subplot(2, 4, 2)
    ticks = [d['tick'] for d in focus_history]
    ax.plot(ticks, [d['h_focus'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
    ax.plot(ticks, [d['t_focus'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Focus Distribution', fontweight='bold')
    ax.set_xlabel('Tick')
    
    # Panel 3: Drives
    ax = plt.subplot(2, 4, 3)
    ax.plot(ticks, [d['avg_h'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
    ax.plot(ticks, [d['avg_t'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Drive Levels', fontweight='bold')
    ax.set_xlabel('Tick')
    
    # Panel 4: Y position
    ax = plt.subplot(2, 4, 4)
    ax.plot(ticks, [d['avg_y'] for d in focus_history], 'purple', linewidth=2)
    ax.axhline(center_y, color='red', linestyle='--', label='Start')
    ax.axhline(center_y - 40, color='blue', linestyle=':', label='Water')
    ax.axhline(center_y + 40, color='green', linestyle=':', label='Food')
    ax.set_ylim([0, size])
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Avg Y Position', fontweight='bold')
    ax.set_xlabel('Tick')
    
    # Panel 5: Final distribution
    ax = plt.subplot(2, 4, 5)
    ax.bar(['North\\n(Water)', 'Center', 'South\\n(Food)'], 
          [went_north, stayed_center, went_south],
          color=['blue', 'gray', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title('Final Locations', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 6-8: Summary
    ax = plt.subplot(2, 4, (6, 8))
    ax.axis('off')
    
    switches_per_agent = focus_switches / num_agents
    
    summary = f'''ADAPTIVE FOCUS + SPATIAL TRADE-OFF

Environment:
  North: Water={hydration[:85, :].mean():.2f}, Food={vegetation[:85, :].mean():.2f}
  South: Water={hydration[170:, :].mean():.2f}, Food={vegetation[170:, :].mean():.2f}

Results:
  Survival: {len(final_pos)}/{num_agents} ({100*len(final_pos)/num_agents:.0f}%)
  
Migration:
  → North (water): {went_north} agents
  → South (food): {went_south} agents
  ○ Center: {stayed_center} agents
  
Focus:
  Hunger: {100*total_h/(total_h+total_t+0.001):.1f}%
  Thirst: {100*total_t/(total_h+total_t+0.001):.1f}%
  Switches: {switches_per_agent:.1f} per agent

Interpretation:
'''
    
    if switches_per_agent > 5:
        summary += '  ✓✓ HIGH switching (adaptive optimization)\\n'
    elif switches_per_agent > 2:
        summary += '  ✓ GOOD switching (dynamic needs)\\n'
    elif switches_per_agent > 0.5:
        summary += '  ~ Moderate switching\\n'
    else:
        summary += '  ✗ Low switching (hysteresis too strong)\\n'
    
    if went_north > 0 and went_south > 0:
        summary += '  ✓ Diverse strategies (some chose water, some food)\\n'
    
    if abs(total_h - total_t) / (total_h + total_t + 1) < 0.3:
        summary += '  ✓ Balanced attention to both needs\\n'
    
    ax.text(0.05, 0.5, summary, fontsize=12, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Multi-Resource Optimization with Adaptive Focus & Spatial Trade-Offs',
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('true_tradeoff.png', dpi=150)
    print('\\n✓ Saved: true_tradeoff.png')

if __name__ == '__main__':
    test_true_tradeoff()

