# Environment Tuning Guide

## Adjusting Field Distributions

### Temperature
Edit `heat_profile` in `interfaces/ui_iface/scenarios/env-b.yaml`:
- `direction`: "north_hot" or "south_hot"
- `amplitude`: 0.0-1.0 (gradient strength)
- `noise_amp`: 0.0-0.2 (local variation)

### Hydrology
Edit `water_profile`:
- `elevation_scale`: 64-128 (larger = bigger features)
- `ridge_strength`: 0.0-1.0 (mountain ridges)
- `river_percentile`: 0.85-0.95 (river density)
- `lake_fill_threshold`: 0.1-0.3 (lake formation)

### Vegetation
Edit `vegetation_profile`:
- `k`: 0.01-0.2 (growth rate)
- `water_half`: 0.2-0.5 (water dependency)
- `heat_optimum`: 0.4-0.8 (ideal temperature)
- `heat_sigma`: 0.1-0.3 (temperature tolerance)

## Testing Changes

1. Edit `interfaces/ui_iface/scenarios/env-b.yaml`
2. Run: `tholos run interfaces/ui_iface/scenarios/env-b.yaml --ticks 128 --out runs --label test`
3. Visualize: `python -m interfaces.ui_iface.runner.cli visualize runs/run-test --plot-type hydrology`
4. Check metrics for realism

## Common Adjustments

**More rivers:** Decrease `river_percentile` to 0.88
**Bigger lakes:** Increase `lake_fill_threshold` to 0.2
**Faster vegetation growth:** Increase `k` to 0.15
**Hotter climate:** Increase `amplitude` to 0.8
