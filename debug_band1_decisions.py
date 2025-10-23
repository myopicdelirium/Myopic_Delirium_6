"""Debug: What decisions is Band 1 actually making?"""
import numpy as np
from interfaces.agent_iface.simulation import AgentSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile
from collections import Counter

print('Generating environment...')
scenario_path = 'interfaces/ui_iface/scenarios/env-b.yaml'
cfg = load_scenario(scenario_path)

with tempfile.TemporaryDirectory() as tmpdir:
    run_dir = run_headless(cfg, ticks=30, out_dir=tmpdir, label='debug_test')
    
    print('Running simulation (5 agents, 20 ticks, very low energy)...')
    sim = AgentSimulation(run_dir, num_predators=0, seed=42)  # No predators to isolate foraging
    sim.spawn_agents(num_agents=5, initial_energy=50.0)  # Very low energy = immediate hunger
    
    action_counts = Counter()
    urgency_samples = []
    
    for tick in range(20):
        print(f'\n--- Tick {tick} ---')
        sim.env.load_tick(tick)  # Load environment state for this tick
        for i, agent in enumerate(sim.agents):
            if not agent.state.alive:
                continue
                
            # Get Band 1's decision
            band1 = agent.bands[0]  # Physiological band
            env_state = sim._get_env_state_for_agent(agent)
            agent_state_dict = {
                "energy": agent.state.energy,
                "position": (agent.state.x, agent.state.y),
                "tick": agent.state.tick
            }
            perception = band1.perceive(env_state, agent_state_dict)
            urgency = band1.compute_urgency(perception)
            proposals = band1.propose_actions(perception)
            
            if proposals:
                top_action = proposals[0]
                action_counts[str(top_action.action)] += 1
                urgency_samples.append(urgency)
                
                if tick < 3 or tick % 5 == 0:  # Sample output
                    print(f'  Agent {i}: energy={agent.state.energy:.1f}, '
                          f'urgency={urgency:.2f}, action={top_action.action}, '
                          f'local_veg={perception.get("local_vegetation", 0):.2f}')
        
        sim.step()
    
    print(f'\n\nACTION DISTRIBUTION (100 agent-ticks):')
    total = sum(action_counts.values())
    for action, count in action_counts.most_common():
        pct = 100 * count / total
        print(f'  {action:12s}: {count:3d} ({pct:5.1f}%)')
    
    if urgency_samples:
        print(f'\nURGENCY STATS:')
        print(f'  Mean: {np.mean(urgency_samples):.3f}')
        print(f'  Min:  {np.min(urgency_samples):.3f}')
        print(f'  Max:  {np.max(urgency_samples):.3f}')
    
    print(f'\nFINAL STATES:')
    for i, agent in enumerate(sim.agents):
        if agent.state.alive:
            print(f'  Agent {i}: energy={agent.state.energy:.1f}, alive={agent.state.alive}')

