"""
Visualization: Agent Migration Toward Water
Tests if Band 1 (Physiological) agents actually migrate toward high-hydration areas.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from interfaces.agent_iface.simulation import AgentSimulation
from interfaces.ui_iface.runner.engine import load_scenario, run_headless
from interfaces.ui_iface.runner.hydrator import hydrate_tick
import tempfile

def visualize_agent_migration(num_agents=30, num_predators=2, num_ticks=50, output_file="agent_migration.png"):
    """
    Visualize agents migrating toward water over time.
    Should see agents cluster in high-hydration (blue) areas.
    """
    print("Setting up environment...")
    scenario_path = "interfaces/ui_iface/scenarios/env-b.yaml"
    cfg = load_scenario(scenario_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_headless(cfg, ticks=num_ticks + 10, out_dir=tmpdir, label="migration_viz")
        
        print("Loading environment state...")
        tensor = hydrate_tick(run_dir, 0)
        hydration = tensor[:, :, 1]  # Hydration field
        
        print(f"Spawning {num_agents} agents...")
        sim = AgentSimulation(run_dir, num_predators=num_predators, seed=42)
        sim.spawn_agents(num_agents=num_agents, initial_energy=150.0)
        
        initial_positions = [(a.state.x, a.state.y) for a in sim.agents]
        
        print(f"Running {num_ticks} ticks...")
        for tick in range(num_ticks):
            sim.step()
            if tick % 10 == 0:
                alive = sum(1 for a in sim.agents if a.state.alive)
                print(f"  Tick {tick}: {alive}/{num_agents} alive")
        
        final_positions = [(a.state.x, a.state.y) for a in sim.agents if a.state.alive]
        trajectories = [a.get_trajectory() for a in sim.agents if a.state.alive]
        
        print(f"Creating visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Initial positions
        ax1 = axes[0]
        ax1.imshow(hydration, cmap='Blues', origin='upper', alpha=0.6, vmin=0, vmax=1)
        init_x = [pos[0] for pos in initial_positions]
        init_y = [pos[1] for pos in initial_positions]
        ax1.scatter(init_x, init_y, c='red', s=30, alpha=0.8, edgecolors='black', linewidths=0.5, label='Agents')
        ax1.set_title(f'Initial Positions (t=0)\n{num_agents} agents spawned', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        # Panel 2: Final positions
        ax2 = axes[1]
        ax2.imshow(hydration, cmap='Blues', origin='upper', alpha=0.6, vmin=0, vmax=1)
        final_x = [pos[0] for pos in final_positions]
        final_y = [pos[1] for pos in final_positions]
        ax2.scatter(final_x, final_y, c='green', s=40, alpha=0.8, edgecolors='black', linewidths=0.5, label='Survivors')
        ax2.set_title(f'Final Positions (t={num_ticks})\n{len(final_positions)}/{num_agents} survived', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        # Panel 3: Trajectories overlaid
        ax3 = axes[2]
        ax3.imshow(hydration, cmap='Blues', origin='upper', alpha=0.5, vmin=0, vmax=1)
        
        for traj in trajectories[:15]:  # Show first 15 trajectories to avoid clutter
            x_coords = [d['position'][0] for d in traj]
            y_coords = [d['position'][1] for d in traj]
            ax3.plot(x_coords, y_coords, 'white', alpha=0.3, linewidth=0.8)
        
        ax3.scatter(final_x, final_y, c='yellow', s=50, alpha=0.9, edgecolors='black', 
                   linewidths=1, label='Final positions', marker='*')
        ax3.set_title(f'Agent Trajectories (showing 15)\nMigration toward water (blue areas)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
        
        # Analyze migration toward water
        print("\n" + "="*60)
        print("MIGRATION ANALYSIS")
        print("="*60)
        
        initial_hydration_values = [hydration[y, x] for x, y in initial_positions]
        final_hydration_values = [hydration[y, x] for x, y in final_positions]
        
        avg_initial_hydration = np.mean(initial_hydration_values)
        avg_final_hydration = np.mean(final_hydration_values)
        
        print(f"Initial average hydration at agent positions: {avg_initial_hydration:.3f}")
        print(f"Final average hydration at survivor positions: {avg_final_hydration:.3f}")
        print(f"Improvement: {(avg_final_hydration - avg_initial_hydration):.3f}")
        
        if avg_final_hydration > avg_initial_hydration + 0.05:
            print("\n✓ SUCCESS: Agents migrated toward higher-hydration areas!")
            print("  Band 1 (Physiological) is working correctly.")
        elif avg_final_hydration > avg_initial_hydration:
            print("\n~ PARTIAL: Agents slightly favored higher-hydration areas.")
        else:
            print("\n✗ UNEXPECTED: No clear migration toward water.")
            print("  May need to tune Band 1 urgencies or increase ticks.")
        
        # Check if high-hydration areas have more survivors
        high_hydration_threshold = 0.85
        survivors_in_high_hydration = sum(1 for x, y in final_positions if hydration[y, x] > high_hydration_threshold)
        total_high_hydration_area = (hydration > high_hydration_threshold).sum()
        total_area = hydration.size
        
        expected_random = (total_high_hydration_area / total_area) * len(final_positions)
        
        print(f"\nSurvivors in high-hydration areas (>{high_hydration_threshold}): {survivors_in_high_hydration}/{len(final_positions)}")
        print(f"Expected if random distribution: {expected_random:.1f}")
        
        if survivors_in_high_hydration > expected_random * 1.2:
            print("✓ Survivors cluster in high-hydration zones (>20% above random)")
        
        print("="*60)

if __name__ == "__main__":
    visualize_agent_migration(
        num_agents=40,
        num_predators=3,
        num_ticks=100,
        output_file="agent_migration.png"
    )

