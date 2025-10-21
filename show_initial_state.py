import sys
import os
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python show_initial_state.py <scenario_yaml>")
    sys.exit(1)

scenario_path = sys.argv[1]

from interfaces.ui_iface.runner.engine import load_scenario, assemble_initial_tensor, build_seed_partitions
from interfaces.ui_iface.runner.registry import build_registry

cfg = load_scenario(scenario_path)
seeds = build_seed_partitions(cfg["randomness"]["seed"], cfg["randomness"]["partitions"])
registry = build_registry(cfg)
result = assemble_initial_tensor(cfg, seeds, registry)
tensor = result["tensor"]
aux = result["aux"]

print("=" * 60)
print("INITIAL STATE ANALYSIS")
print("=" * 60)
print(f"Grid: {tensor.shape[0]}x{tensor.shape[1]} with {tensor.shape[2]} fields")
print()

for fname, idx in registry["indices"].items():
    field_data = tensor[:, :, idx]
    print(f"{fname.upper()}:")
    print(f"  Range: [{field_data.min():.3f}, {field_data.max():.3f}]")
    print(f"  Mean: {field_data.mean():.3f}")
    print(f"  Std Dev: {field_data.std():.3f}")
    
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(field_data, bins=bins)
    total = field_data.size
    print(f"  Distribution:")
    for i in range(len(bins)-1):
        pct = 100.0 * hist[i] / total
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:6d} cells ({pct:5.1f}%)")
    print()

print("HYDROLOGY:")
print(f"  Elevation Range: [{aux['E'].min():.3f}, {aux['E'].max():.3f}]")
print(f"  Precipitation Range: [{aux['P'].min():.3f}, {aux['P'].max():.3f}]")
print(f"  Flow Accumulation Range: [{aux['A'].min():.0f}, {aux['A'].max():.0f}]")
print(f"  Lake Coverage: {aux['lake_mask'].sum()} cells ({100.0 * aux['lake_mask'].sum() / aux['lake_mask'].size:.1f}%)")
print()
print("=" * 60)

