"""
EXTREME trade-off: Very stark separation of food and water.
Tests if agents will actually MIGRATE between zones based on their needs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation

def create_extreme_tradeoff_environment(size=256):
    """Create environment with EXTREME food/water separation."""
    y_coords = np.arange(size)[:, None]
    
    # Hydration: VERY HIGH in north, VERY LOW in south (EXTREME gradient)
    hydration = 1.0 - (y_coords / size) * 0.85  # 1.0 at top -> 0.15 at bottom
    hydration = np.broadcast_to(hydration, (size, size)).copy()
    hydration += np.random.rand(size, size) * 0.08 - 0.04
    hydration = np.clip(hydration, 0.15, 1.0)
    
    # Vegetation: VERY LOW in north, VERY HIGH in south (EXTREME opposite)
    vegetation = (y_coords / size) * 0.75  # 0.0 at top -> 0.75 at bottom
    vegetation = np.broadcast_to(vegetation, (size, size)).copy()
    vegetation += np.random.rand(size, size) * 0.12 - 0.06
    vegetation = np.clip(vegetation, 0.0, 0.75)
    
    # Temperature: mild
    temperature = np.ones((size, size)) * 0.5
    
    return vegetation, hydration, temperature

def test_extreme_tradeoff():
    print('Creating EXTREME trade-off environment...')
    size = 256
    vegetation, hydration, temperature = create_extreme_tradeoff_environment(size)
    
    print(f'\nEnvironment (EXTREME separation):')
    print(f'NORTH (y=0-64):    Water={hydration[:64, :].mean():.2f}, Food={vegetation[:64, :].mean():.2f}')
    print(f'NORTH-MID (y=64-128): Water={hydration[64:128, :].mean():.2f}, Food={vegetation[64:128, :].mean():.2f}')
    print(f'SOUTH-MID (y=128-192): Water={hydration[128:192, :].mean():.2f}, Food={vegetation[128:192, :].mean():.2f}')
    print(f'SOUTH (y=192-256): Water={hydration[192:, :].mean():.2f}, Food={vegetation[192:, :].mean():.2f}')
    
    # Create simulation
    sim = FastStaticSimulation(vegetation, temperature, hydration, size, size, 
                               num_predators=1, seed=42)
    
    # Spawn agents in TWO GROUPS:
    # Group A: North (water-rich, food-poor) - should migrate SOUTH
    # Group B: South (food-rich, water-poor) - should migrate NORTH
    
    num_agents = 40
    rng = np.random.default_rng(42)
    
    print(f'\nSpawning {num_agents} agents in TWO GROUPS...')
    print(f'  Group A (20 agents): NORTH zone (high water, low food)')
    print(f'  Group B (20 agents): SOUTH zone (high food, low water)')
    
    group_a_start = []
    group_b_start = []
    
    for i in range(num_agents):
        from interfaces.agent_iface.banded_agent import BandedAgent
        
        if i < 20:
            # Group A: North (water zone, should need food)
            x = rng.integers(size//4, 3*size//4)
            y = rng.integers(10, 50)  # Very north
            agent = BandedAgent(agent_id=i, x=x, y=y, initial_energy=40.0,
                               seed=rng.integers(0, 1000000))
            agent.bands[0].state.internal_state["hunger"] = 0.6  # Hungry!
            agent.bands[0].state.internal_state["thirst"] = 0.2  # Not thirsty
            group_a_start.append((x, y))
        else:
            # Group B: South (food zone, should need water)
            x = rng.integers(size//4, 3*size//4)
            y = rng.integers(200, 245)  # Very south
            agent = BandedAgent(agent_id=i, x=x, y=y, initial_energy=40.0,
                               seed=rng.integers(0, 1000000))
            agent.bands[0].state.internal_state["hunger"] = 0.2  # Not hungry
            agent.bands[0].state.internal_state["thirst"] = 0.6  # Thirsty!
            group_b_start.append((x, y))
        
        sim.agents.append(agent)
    
    # Run simulation
    num_ticks = 300
    trajectories = [[] for _ in range(num_agents)]
    focus_history = []
    
    print(f'\nRunning {num_ticks} ticks...')
    for tick in range(num_ticks):
        alive = [a for a in sim.agents if a.state.alive]
        if not alive:
            break
        
        for i, agent in enumerate(sim.agents):
            if agent.state.alive:
                trajectories[i].append((agent.state.x, agent.state.y))
        
        h_focus = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "hunger")
        t_focus = sum(1 for a in alive if a.bands[0].state.internal_state.get("current_focus") == "thirst")
        
        avg_y = np.mean([a.state.y for a in alive])
        avg_h = np.mean([a.bands[0].state.internal_state.get("hunger", 0) for a in alive])
        avg_t = np.mean([a.bands[0].state.internal_state.get("thirst", 0) for a in alive])
        
        focus_history.append({
            'tick': tick, 'h_focus': h_focus, 't_focus': t_focus,
            'avg_y': avg_y, 'avg_h': avg_h, 'avg_t': avg_t, 'alive': len(alive)
        })
        
        if tick % 60 == 0:
            print(f'  T={tick}: {len(alive)} alive, H={h_focus}, T={t_focus}, avg_y={avg_y:.0f}')
        
        sim.step()
    
    # Analysis: Did groups migrate toward their needed resources?
    group_a_final = [(a.state.x, a.state.y) for a in sim.agents[:20] if a.state.alive]
    group_b_final = [(a.state.x, a.state.y) for a in sim.agents[20:] if a.state.alive]
    
    group_a_y_change = np.mean([y for x, y in group_a_final]) - np.mean([y for x, y in group_a_start]) if group_a_final else 0
    group_b_y_change = np.mean([y for x, y in group_b_final]) - np.mean([y for x, y in group_b_start]) if group_b_final else 0
    
    total_h = sum(d['h_focus'] for d in focus_history)
    total_t = sum(d['t_focus'] for d in focus_history)
    
    print(f'\n=== RESULTS ===')
    print(f'Group A (started NORTH, hungry):')
    print(f'  Survival: {len(group_a_final)}/20')
    print(f'  Y change: {group_a_y_change:+.1f} pixels (positive = moved SOUTH toward food)')
    print(f'Group B (started SOUTH, thirsty):')
    print(f'  Survival: {len(group_b_final)}/20')
    print(f'  Y change: {group_b_y_change:+.1f} pixels (negative = moved NORTH toward water)')
    print(f'Focus balance: Hunger={100*total_h/(total_h+total_t+1):.1f}%, Thirst={100*total_t/(total_h+total_t+1):.1f}%')
    
    # Visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Environment + trajectories
    ax = plt.subplot(2, 3, 1)
    rgb = np.zeros((size, size, 3))
    rgb[:,:,1] = vegetation / 0.75  # Green for food
    rgb[:,:,2] = hydration  # Blue for water
    rgb = np.clip(rgb, 0, 1)
    ax.imshow(rgb, origin='upper')
    
    # Draw Group A trajectories (red)
    for i, traj in enumerate(trajectories[:20]):
        if len(traj) > 1:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, 'red', alpha=0.5, linewidth=2)
    
    # Draw Group B trajectories (blue)
    for i, traj in enumerate(trajectories[20:]):
        if len(traj) > 1:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, 'cyan', alpha=0.5, linewidth=2)
    
    # Start positions
    if group_a_start:
        ax.scatter([p[0] for p in group_a_start], [p[1] for p in group_a_start],
                  c='darkred', s=80, marker='x', linewidth=3, label='Group A Start', zorder=5)
    if group_b_start:
        ax.scatter([p[0] for p in group_b_start], [p[1] for p in group_b_start],
                  c='darkblue', s=80, marker='x', linewidth=3, label='Group B Start', zorder=5)
    
    # Final positions
    if group_a_final:
        ax.scatter([p[0] for p in group_a_final], [p[1] for p in group_a_final],
                  c='red', s=120, marker='o', edgecolors='black', linewidth=2, label='Group A End', zorder=6)
    if group_b_final:
        ax.scatter([p[0] for p in group_b_final], [p[1] for p in group_b_final],
                  c='cyan', s=120, marker='o', edgecolors='black', linewidth=2, label='Group B End', zorder=6)
    
    ax.axhline(64, color='white', linestyle=':', alpha=0.5)
    ax.axhline(192, color='white', linestyle=':', alpha=0.5)
    ax.set_title('Migration Trajectories\\n(Red=Hungry/North, Cyan=Thirsty/South)', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9)
    
    # Panel 2: Y position over time
    ax = plt.subplot(2, 3, 2)
    ticks = [d['tick'] for d in focus_history]
    ax.plot(ticks, [d['avg_y'] for d in focus_history], 'purple', linewidth=3)
    ax.axhline(size//2, color='red', linestyle='--', label='Center', linewidth=2)
    ax.axhline(64, color='cyan', linestyle=':', alpha=0.7, label='North-Mid')
    ax.axhline(192, color='orange', linestyle=':', alpha=0.7, label='South-Mid')
    ax.set_ylim([0, size])
    ax.invert_yaxis()
    ax.set_xlabel('Tick')
    ax.set_ylabel('Average Y Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Population Movement', fontweight='bold', fontsize=13)
    
    # Panel 3: Focus distribution
    ax = plt.subplot(2, 3, 3)
    ax.plot(ticks, [d['h_focus'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
    ax.plot(ticks, [d['t_focus'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
    ax.set_xlabel('Tick')
    ax.set_ylabel('# Agents')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Focus Distribution', fontweight='bold', fontsize=13)
    
    # Panel 4: Drive levels
    ax = plt.subplot(2, 3, 4)
    ax.plot(ticks, [d['avg_h'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
    ax.plot(ticks, [d['avg_t'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Drive Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Average Drives', fontweight='bold', fontsize=13)
    
    # Panel 5: Migration summary
    ax = plt.subplot(2, 3, 5)
    ax.barh(['Group A\\n(Hungry/North)', 'Group B\\n(Thirsty/South)'],
           [group_a_y_change, group_b_y_change],
           color=['red', 'cyan'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Y Position Change\\n(+ = South, - = North)')
    ax.set_title('Net Migration Direction', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 6: Summary
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    
    summary = f'''EXTREME TRADE-OFF TEST

Environment:
  North: Water=0.90, Food=0.10
  South: Water=0.20, Food=0.65
  
Groups:
  A: Hungry in north (need food)
     → Moved {group_a_y_change:+.0f} pixels
     → {"SOUTH (toward food) ✓" if group_a_y_change > 20 else "Stayed/wrong direction"}
     
  B: Thirsty in south (need water)
     → Moved {group_b_y_change:+.0f} pixels
     → {"NORTH (toward water) ✓" if group_b_y_change < -20 else "Stayed/wrong direction"}

Focus Balance:
  Hunger: {100*total_h/(total_h+total_t+1):.0f}%
  Thirst: {100*total_t/(total_h+total_t+1):.0f}%
  {"✓ BALANCED" if abs(total_h - total_t)/(total_h+total_t+1) < 0.3 else "~ Imbalanced"}

Overall:
'''
    
    if group_a_y_change > 20 and group_b_y_change < -20:
        summary += '  ✓✓✓ Both groups migrated correctly!\\n'
        summary += '  TRUE multi-resource optimization'
    elif group_a_y_change > 10 or group_b_y_change < -10:
        summary += '  ✓ Some migration observed\\n'
        summary += '  Needs tuning'
    else:
        summary += '  ✗ Little migration\\n'
        summary += '  Agents not responding to needs'
    
    ax.text(0.05, 0.5, summary, fontsize=11, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Extreme Trade-Off: Do Agents Migrate Toward Their Needed Resources?',
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('extreme_tradeoff.png', dpi=150)
    print('\\n✓ Saved: extreme_tradeoff.png')

if __name__ == '__main__':
    test_extreme_tradeoff()

