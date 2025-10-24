"""
Force a food/water trade-off: Spawn in DRY area with HIGH initial thirst.
This tests if agents can balance competing needs when both are urgent.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fast_migration_viz import FastStaticSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def test_forced_tradeoff():
    print('Loading environment...')
    scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=1, out_dir=tmpdir, label='tradeoff')
        tensor = hydrate_tick(run_dir, 0)
        
        vegetation = tensor[:, :, 2]
        hydration = tensor[:, :, 1]
        temperature = tensor[:, :, 0]
        h, w = vegetation.shape
        
        # Find DRIEST areas (even though still relatively wet)
        driest_mask = hydration < 0.6
        driest_coords = np.argwhere(driest_mask)
        
        print(f'Driest areas (hydration < 0.6): {len(driest_coords)} cells')
        
        sim = FastStaticSimulation(vegetation, temperature, hydration, w, h, 
                                   num_predators=1, seed=42)
        
        # Spawn agents in dry areas
        num_agents = 20
        rng = np.random.default_rng(42)
        spawn_indices = rng.choice(len(driest_coords), size=num_agents, replace=False)
        
        print(f'\\nSpawning {num_agents} agents in DRIEST areas...')
        print('Setting BOTH hunger and thirst to 0.5 (competing needs)')
        
        for i, idx in enumerate(spawn_indices):
            y, x = driest_coords[idx]
            from interfaces.agent_iface.banded_agent import BandedAgent
            agent = BandedAgent(agent_id=i, x=int(x), y=int(y), initial_energy=50.0,
                               seed=rng.integers(0, 1000000))
            # Set BOTH drives to moderate levels (force competition)
            agent.bands[0].state.internal_state["hunger"] = 0.5
            agent.bands[0].state.internal_state["thirst"] = 0.5
            sim.agents.append(agent)
        
        # Track focus switches
        num_ticks = 150
        focus_history = []
        focus_switches = 0
        
        print(f'\\nRunning {num_ticks} ticks...')
        for tick in range(num_ticks):
            alive = [a for a in sim.agents if a.state.alive]
            if not alive:
                break
            
            # Count focus switches
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
            
            focus_history.append({
                'tick': tick,
                'hunger_focused': hunger_focused,
                'thirst_focused': thirst_focused,
                'avg_hunger': avg_hunger,
                'avg_thirst': avg_thirst,
                'alive': len(alive)
            })
            
            if tick % 30 == 0:
                print(f'  T={tick}: {len(alive)} alive, H-focus={hunger_focused}, T-focus={thirst_focused}, ' +
                      f'hunger={avg_hunger:.2f}, thirst={avg_thirst:.2f}')
            
            sim.step()
        
        # Analysis
        total_hunger = sum(d['hunger_focused'] for d in focus_history)
        total_thirst = sum(d['thirst_focused'] for d in focus_history)
        total_focus = total_hunger + total_thirst
        
        print(f'\\n=== RESULTS ===')
        print(f'Hunger-focused ticks: {total_hunger} ({100*total_hunger/total_focus:.1f}%)')
        print(f'Thirst-focused ticks: {total_thirst} ({100*total_thirst/total_focus:.1f}%)')
        print(f'Focus switches: {focus_switches}')
        print(f'Switches per agent-lifetime: {focus_switches / (num_agents * len(focus_history)):.3f}')
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax = axes[0, 0]
        ticks = [d['tick'] for d in focus_history]
        ax.plot(ticks, [d['hunger_focused'] for d in focus_history], 'orange', linewidth=2, label='Hunger Focus')
        ax.plot(ticks, [d['thirst_focused'] for d in focus_history], 'blue', linewidth=2, label='Thirst Focus')
        ax.set_xlabel('Tick')
        ax.set_ylabel('Number of Agents')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Focus Distribution Over Time', fontweight='bold')
        
        ax = axes[0, 1]
        ax.plot(ticks, [d['avg_hunger'] for d in focus_history], 'orange', linewidth=2, label='Hunger')
        ax.plot(ticks, [d['avg_thirst'] for d in focus_history], 'blue', linewidth=2, label='Thirst')
        ax.set_xlabel('Tick')
        ax.set_ylabel('Drive Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Drive Levels', fontweight='bold')
        
        ax = axes[1, 0]
        hunger_pcts = [100 * d['hunger_focused'] / max(1, d['alive']) for d in focus_history]
        thirst_pcts = [100 * d['thirst_focused'] / max(1, d['alive']) for d in focus_history]
        ax.stackplot(ticks, hunger_pcts, thirst_pcts, labels=['Hunger', 'Thirst'], 
                    colors=['orange', 'blue'], alpha=0.7)
        ax.set_xlabel('Tick')
        ax.set_ylabel('% of Population')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Attentional Focus %', fontweight='bold')
        
        ax = axes[1, 1]
        ax.axis('off')
        
        if total_focus > 0:
            hunger_pct = 100 * total_hunger / total_focus
            thirst_pct = 100 * total_thirst / total_focus
        else:
            hunger_pct = thirst_pct = 0
        
        switches_per_agent = focus_switches / num_agents if num_agents > 0 else 0
        
        summary = f'''MULTI-NEED OPTIMIZATION TEST

Started with BOTH needs at 0.5
(Forced competition scenario)

Results:
  Hunger focus: {hunger_pct:.1f}%
  Thirst focus: {thirst_pct:.1f}%
  Total switches: {focus_switches}
  Switches/agent: {switches_per_agent:.1f}

Interpretation:
'''
        
        if abs(hunger_pct - thirst_pct) < 15:
            summary += '  ✓ BALANCED attention\\n'
        else:
            summary += f'  ⚠ Imbalanced ({max(hunger_pct, thirst_pct):.0f}% vs {min(hunger_pct, thirst_pct):.0f}%)\\n'
        
        if switches_per_agent > 2:
            summary += f'  ✓ Dynamic switching ({switches_per_agent:.1f}/agent)\\n'
        elif switches_per_agent > 0.5:
            summary += f'  ~ Some switching ({switches_per_agent:.1f}/agent)\\n'
        else:
            summary += f'  ✗ Stuck on one need ({switches_per_agent:.1f}/agent)\\n'
        
        ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        plt.suptitle('Multi-Need Optimization: Can Agents Balance Competing Drives?', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('forced_tradeoff.png', dpi=150)
        print('\\n✓ Saved: forced_tradeoff.png')

if __name__ == '__main__':
    test_forced_tradeoff()

