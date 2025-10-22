import sys
import os
import numpy as np
from interfaces.ui_iface.runner.hydrator import hydrate_tick, get_field_index, get_field_names

if len(sys.argv) < 2:
    print("Usage: python analyze_hydration.py <run_dir> [tick]")
    sys.exit(1)

run_dir = sys.argv[1]
tick = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print("=" * 60)
print("HYDRATION ANALYSIS")
print("=" * 60)

tensor = hydrate_tick(run_dir, tick)
field_names = get_field_names(run_dir)

h_idx = get_field_index(run_dir, "hydration")
hydration = tensor[:, :, h_idx]

print(f"Tick: {tick}")
print(f"Range: [{hydration.min():.3f}, {hydration.max():.3f}]")
print(f"Mean: {hydration.mean():.3f}")
print(f"Std Dev: {hydration.std():.3f}")
print()

try:
    w_idx = get_field_index(run_dir, "water_body")
    water = tensor[:, :, w_idx]
    land_mask = water < 0.5
    print("Land Coverage:")
    if land_mask.any():
        print(f"  Hydration Range on Land: [{hydration[land_mask].min():.3f}, {hydration[land_mask].max():.3f}]")
        print(f"  Hydration Mean on Land: {hydration[land_mask].mean():.3f}")
    else:
        print("  No land cells found")
    print()
except ValueError:
    pass

print("Hydration Distribution:")
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
hist, _ = np.histogram(hydration, bins=bins)
total = hydration.size
for i in range(len(bins)-1):
    pct = 100.0 * hist[i] / total
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:6d} cells ({pct:5.1f}%)")
print()

try:
    v_idx = get_field_index(run_dir, "vegetation")
    vegetation = tensor[:, :, v_idx]
    print(f"Vegetation Stats:")
    print(f"  Range: [{vegetation.min():.3f}, {vegetation.max():.3f}]")
    print(f"  Mean: {vegetation.mean():.3f}")
    print()
except ValueError:
    pass

print("=" * 60)
