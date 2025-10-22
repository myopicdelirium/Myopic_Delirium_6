# Tholos - Myopic Delirium Environment Generator

**A research-grade environment simulator for agent-based models that reject pure utilitarian frameworks.**

Tholos generates deterministic, spatially-coherent environments with realistic hydrology, temperature gradients, and vegetation dynamics. Designed for rigorous research into agent behavior that incorporates non-utilitarian decision-making substrates.

---

## Features

- **Deterministic Simulation**: Identical seeds → identical outputs
- **Natural Environmental Dynamics**: Hydrology (rivers, lakes), equator-hot temperature, vegetation coupling
- **Unified Grid Tensor**: Single [H, W, F] tensor for all fields - no overlays
- **Sparse Delta Storage**: Efficient artifact storage with full state reconstruction
- **Complete Observability**: Agent API for querying any tick, any location
- **Research-Grade Testing**: Comprehensive test suite with quality gates
- **Production CLI**: Run simulations, generate visualizations, inspect results

---

## Installation

**Requirements**: Python 3.11

```bash
# Clone repository
git clone https://github.com/myopicdelirium/Myopic_Delirium_6.git
cd Myopic_Delirium_6

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install -e .
```

---

## Quick Start

### Run a Simulation

```bash
# Run default scenario for 1000 ticks
tholos run interfaces/ui_iface/scenarios/env-b.yaml --ticks 1000 --label my_run

# Output: runs/run-my_run/
```

### Visualize Results

```bash
# Single field
tholos visualize runs/run-my_run --field temperature --save temp.png

# All fields
tholos visualize runs/run-my_run --plot-type hydrology --save overview.png

# Metrics over time
tholos visualize runs/run-my_run --plot-type metrics --save metrics.png
```

### Use Agent API

```python
from interfaces.ui_iface.runner.agent_api import EnvironmentGrid

# Load environment state
env = EnvironmentGrid("runs/run-my_run")
env.load_tick(500)

# Query fields
temperature = env.get_field("temperature")
location_data = env.get_all_fields_at(x=128, y=128)
neighborhood = env.get_neighborhood(x=128, y=128, radius=3)

# Access specific values
temp_value = env.get_cell(x=100, y=100, field_name="temperature")
```

---

## Environment Fields

### Temperature
- **Distribution**: Equator-hot, symmetric cold poles
- **Range**: [0.0, 1.0]
- **Dynamics**: Diffusion (0.18)

### Hydration
- **Distribution**: Hydrology-based (rivers, lakes, elevation)
- **Range**: [0.0, 1.0]
- **Dynamics**: Diffusion (0.12), evaporation coupling

### Vegetation
- **Distribution**: Function of temperature optimum and water availability
- **Range**: [0.0, 1.0]
- **Dynamics**: Diffusion (0.05), growth/decay coupling

### Movement Cost (Derived)
- **Distribution**: Terrain-based
- **Range**: [0.0, 1.0]
- **Dynamics**: None (derived field)

---

## Architecture

```
interfaces/ui_iface/
├── runner/
│   ├── engine.py        # Simulation loop, initialization
│   ├── kernels.py       # Diffusion, advection, coupling
│   ├── initgen.py       # Field initialization (hydrology, temperature)
│   ├── hydrator.py      # State reconstruction from deltas
│   ├── registry.py      # Field metadata and indexing
│   ├── agent_api.py     # Agent interface
│   ├── viz.py           # Visualization functions
│   └── cli.py           # Command-line interface
├── schemas/
│   └── schema.py        # Scenario validation
└── scenarios/
    └── env-b.yaml       # Default scenario

tests/
├── test_agent_api.py          # Agent interface validation
├── test_determinism.py        # Reproducibility tests
├── test_field_quality.py      # Initialization quality gates
├── test_artifacts.py          # Artifact completeness
└── test_hydrator.py           # State reconstruction accuracy
```

---

## Determinism

**Guaranteed reproducibility:**
1. Same scenario + same seed → bit-identical results
2. Seed partitioning ensures independent RNG streams for each subsystem
3. All operations use float32 with bounded clipping
4. Cross-platform reproducibility within Python 3.11.x

**Validation:**
```bash
pytest tests/test_determinism.py -v
```

---

## Testing

Run full test suite:
```bash
pytest tests/ -v
```

**Quality Gates:**
- ✓ Deterministic initialization and simulation
- ✓ Field bounds [0.0, 1.0] strictly enforced
- ✓ Temperature: equator-hot, symmetric poles
- ✓ Hydration: mean > 0.5, 80%+ cells > 0.8
- ✓ Vegetation: positive correlation with temp and water
- ✓ Spatial coherence maintained
- ✓ Artifact completeness (deltas, metrics, checksums)
- ✓ State reconstruction accuracy

---

## Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Tuning Guide](docs/TUNING.md)**: Parameter adjustment guide

---

## Performance

**Targets** (256x256 grid on modern CPU):
- Initialization: < 2 seconds
- Tick execution: < 50ms average
- 1000 ticks: < 60 seconds
- State reconstruction: < 0.5 seconds
- Visualization: < 2 seconds per plot

---

## Citation

If you use Tholos in your research:

```bibtex
@software{tholos2025,
  title={Tholos: Environment Generator for Non-Utilitarian Agent Research},
  author={Myopic Delirium Research},
  year={2025},
  url={https://github.com/myopicdelirium/Myopic_Delirium_6}
}
```

---

## Research Context

This tool is part of a research platform examining agent-based models that incorporate non-utilitarian substrates of cognition. Traditional agent-based models assume purely rational, utility-maximizing behavior. This environment enables research into agents with bounded rationality, emotional states, social dynamics, and other deviations from strict utilitarianism.

---

## License

[Specify license]

---

## Contributing

This is a research tool. Contributions should maintain:
1. Deterministic reproducibility
2. Research-grade testing coverage
3. Performance targets
4. API stability

For major changes, open an issue first to discuss the proposed modification.
