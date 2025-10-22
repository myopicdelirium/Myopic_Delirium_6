# Tholos API Reference

## Environment Grid API

### `EnvironmentGrid`

Primary interface for agents to query environment state.

```python
from interfaces.ui_iface.runner.agent_api import EnvironmentGrid

env = EnvironmentGrid(run_dir="runs/my_run")
env.load_tick(0)
temp = env.get_field("temperature")
```

#### Methods

**`__init__(run_dir: str)`**
- Initialize from a run directory
- Loads scenario configuration and builds field registry

**`load_tick(tick: int) -> np.ndarray`**
- Load environment state at specified tick
- Returns full tensor [H, W, F]
- Must be called before other query methods

**`get_field(field_name: str) -> np.ndarray`**
- Get 2D array for a single field
- Returns shape [H, W]
- Raises ValueError if load_tick() not called

**`get_cell(x: int, y: int, field_name: str) -> float`**
- Get single cell value for a field
- Returns scalar float

**`get_all_fields_at(x: int, y: int) -> dict`**
- Get all field values at a location
- Returns dict mapping field names to values

**`get_neighborhood(x: int, y: int, radius: int = 1) -> dict`**
- Get local neighborhood around a point
- Returns dict mapping field names to 2D arrays
- Each array has shape [(2*radius+1), (2*radius+1)]

#### Properties

**`shape`** - Tuple (height, width, num_fields)

**`field_names`** - List of field names

**`current_tick`** - Currently loaded tick number

---

## Simulation Engine

### `run_headless(cfg: dict, ticks: int, out_dir: str, label: str = None) -> str`

Execute a simulation run.

**Parameters:**
- `cfg`: Scenario configuration dictionary
- `ticks`: Number of simulation ticks to run
- `out_dir`: Output directory for artifacts
- `label`: Optional run label

**Returns:** Path to run directory

**Example:**
```python
from interfaces.ui_iface.runner.engine import load_scenario, run_headless

cfg = load_scenario("scenarios/env-b.yaml")
run_dir = run_headless(cfg, ticks=1000, out_dir="runs", label="experiment_1")
```

---

## Hydrator

### `hydrate_tick(run_dir: str, tick: int) -> np.ndarray`

Reconstruct full environment state at any tick.

**Parameters:**
- `run_dir`: Path to run directory
- `tick`: Tick number to reconstruct

**Returns:** Tensor of shape [H, W, F] with absolute field values

**Example:**
```python
from interfaces.ui_iface.runner.hydrator import hydrate_tick

tensor = hydrate_tick("runs/my_run", tick=500)
temperature = tensor[:, :, 0]  # Field index 0
```

### `get_field_index(run_dir: str, field_name: str) -> int`

Get field index by name.

### `get_field_names(run_dir: str) -> list[str]`

Get list of all field names in run.

### `get_tick_range(run_dir: str) -> tuple[int, int]`

Get available tick range as (min_tick, max_tick).

---

## CLI Commands

### `tholos run`

Execute a simulation.

```bash
tholos run SCENARIO_PATH [OPTIONS]
```

**Options:**
- `--ticks INTEGER`: Number of ticks (default: 256)
- `--out TEXT`: Output directory (default: runs)
- `--label TEXT`: Run label

**Example:**
```bash
tholos run scenarios/env-b.yaml --ticks 1000 --label my_experiment
```

### `tholos visualize`

Generate visualizations from a run.

```bash
tholos visualize RUN_DIR [OPTIONS]
```

**Options:**
- `--field TEXT`: Field to visualize (default: hydration)
- `--plot-type TEXT`: Type - field/hydrology/metrics/animation
- `--tick INTEGER`: Which tick to visualize (default: 0)
- `--save TEXT`: Save to file

**Examples:**
```bash
# Single field at specific tick
tholos visualize runs/my_run --field temperature --tick 500 --save temp500.png

# All fields overview
tholos visualize runs/my_run --plot-type hydrology --save overview.png

# Metrics over time
tholos visualize runs/my_run --plot-type metrics --save metrics.png

# Animation
tholos visualize runs/my_run --field vegetation --plot-type animation --save veg.gif
```

### `tholos inspect`

Inspect run metadata and statistics.

```bash
tholos inspect RUN_DIR
```

Outputs manifest information and final field statistics.

---

## Field Specifications

### Temperature
- **Range:** [0.0, 1.0]
- **Distribution:** Equator-hot (symmetric, cold at poles)
- **Dynamics:** Diffusion (0.18), no decay

### Hydration
- **Range:** [0.0, 1.0]
- **Distribution:** Based on hydrology (rivers, lakes, elevation)
- **Dynamics:** Diffusion (0.12), coupling with evaporation

### Vegetation
- **Range:** [0.0, 1.0]
- **Distribution:** Function of temperature optimum and water availability
- **Dynamics:** Diffusion (0.05), coupling with growth/consumption, decay

### Movement Cost (Derived)
- **Range:** [0.0, 1.0]
- **Distribution:** Computed from terrain
- **Dynamics:** None (derived field)

---

## Determinism Guarantees

1. **Identical seeds → Identical outputs**
   - Same scenario with same seed produces bit-identical results
   - Guaranteed across Python versions 3.11.x

2. **Seed partitioning**
   - Different subsystems use separate RNG streams
   - Prevents coupling between terrain, precipitation, temperature, etc.

3. **Floating-point stability**
   - All operations use float32
   - Bounded fields clipped to [0.0, 1.0]
   - No NaN or Inf values

---

## Testing

Run test suite:

```bash
pytest tests/ -v
```

**Test categories:**
- `test_agent_api.py`: Agent interface validation
- `test_determinism.py`: Reproducibility verification
- `test_field_quality.py`: Field initialization quality gates
- `test_artifacts.py`: Artifact completeness
- `test_hydrator.py`: State reconstruction accuracy

**Quality gates:**
- Deterministic initialization and simulation
- Field bounds [0.0, 1.0] preserved
- Temperature: equator-hot, symmetric poles
- Hydration: mean > 0.5, majority > 0.8
- Vegetation: positive correlation with temp and water
- Spatial coherence: neighbor similarity

---

## Performance Targets

- **Initialization:** < 2 seconds for 256x256 grid
- **Tick execution:** < 50ms average
- **1000 ticks:** < 60 seconds
- **Hydration:** < 0.5 seconds for any tick
- **Visualization:** < 2 seconds per plot

---

## Artifact Structure

```
runs/run-{label}/
├── manifest.json           # Run metadata
├── scenario.json          # Scenario configuration
├── grid/
│   └── deltas.parquet    # Sparse field changes
├── metrics/
│   ├── field_stats.parquet      # Mean, var, min, max per tick
│   ├── hydrology.parquet        # River/lake metrics
│   └── structure.parquet        # Spatial coherence
├── streams/
│   └── events.ndjson     # Lifecycle events
└── checksums/
    └── *.blake3          # File integrity hashes
```

---

## Extension Points

### Adding New Fields

1. Add to scenario `fields` array
2. Implement initialization in `initgen.py`
3. Add to `assemble_initial_tensor()` in `engine.py`
4. Define dynamics coefficients

### Adding New Kernels

1. Implement kernel function in `kernels.py`
2. Add pass to `step_kernels()` 
3. Enable in scenario `dynamics.passes`

### Custom Derived Fields

1. Set `derived: true` in field definition
2. Implement computation in `compute_derived_fields()`
3. Kernel passes will not modify derived fields

