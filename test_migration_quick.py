"""Quick test: Do Band 1 agents migrate toward food (vegetation)?"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from interfaces.agent_iface.simulation import AgentSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

print('Generating environment...')
scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
cfg = load_scenario(scenario_path)

with tempfile.TemporaryDirectory() as tmpdir:
    run_dir = run_headless(cfg, ticks=60, out_dir=tmpdir, label='migration_test')
    
    tensor = hydrate_tick(run_dir, 0)
    vegetation = tensor[:, :, 2]
    
    print(f'Vegetation range: {vegetation.min():.3f} to {vegetation.max():.3f}')
    
    print('Running simulation (20 agents, 50 ticks, low initial energy)...')
    sim = AgentSimulation(run_dir, num_predators=1, seed=42)
    sim.spawn_agents(num_agents=20, initial_energy=80.0)  # Lower energy = hunger faster
    
    initial_pos = [(a.state.x, a.state.y) for a in sim.agents]
    
    for tick in range(50):
        sim.step()
    
    final_pos = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
    
    initial_veg = np.mean([vegetation[y, x] for x, y in initial_pos])
    final_veg = np.mean([vegetation[y, x] for x, y in final_pos]) if final_pos else 0
    
    print(f'\nRESULTS:')
    print(f'Survived: {len(final_pos)}/20')
    print(f'Initial avg vegetation: {initial_veg:.3f}')
    print(f'Final avg vegetation: {final_veg:.3f}')
    print(f'Improvement: {(final_veg - initial_veg):.3f}')
    
    if final_veg > initial_veg + 0.05:
        print('✓ SUCCESS: Agents migrated toward food!')
    elif final_veg > initial_veg:
        print('~ Slight preference for higher vegetation')
    else:
        print('✗ No clear food-seeking behavior')
    
    # Visual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=1)
    ax1.scatter([p[0] for p in initial_pos], [p[1] for p in initial_pos], 
               c='red', s=20, alpha=0.7, label='Initial')
    ax1.set_title('Initial Positions')
    ax1.legend()
    
    ax2.imshow(vegetation, cmap='Greens', origin='upper', vmin=0, vmax=1)
    ax2.scatter([p[0] for p in final_pos], [p[1] for p in final_pos], 
               c='yellow', s=40, alpha=0.9, edgecolors='black', label='Survivors')
    ax2.set_title(f'Final Positions (t=50)\nAvg vegetation: {final_veg:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('migration_test.png', dpi=150)
    print('Saved: migration_test.png')

