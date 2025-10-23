# Tholos Development Status

**Date**: October 23, 2025  
**Status**: Band 1 (Physiological) Complete - Foundation Ready

---

## ✓ COMPLETED: Research Infrastructure

### 1. Environment System (Complete)
- **Deterministic simulation engine** with reproducible outputs given same seed
- **Multi-field environment**: temperature, hydration, vegetation, movement cost
- **Delta-based storage** for efficient long runs (initial state + per-tick changes)
- **Hydrator system** for reconstructing full state at any tick
- **Comprehensive testing**: 82 tests covering determinism, field quality, artifacts
- **Visualization CLI**: field plots, animations, hydrology analysis, metrics timeseries

### 2. Banded Controller Architecture (Foundation Complete)
**Abstract Band System**:
- `Band` base class defining perception/urgency/proposal/learning interface
- `BandState` with urgency, internal state, gain, frustration accumulator
- `ActionProposal` with action, urgency, expected value, band_id, params
- Memory system with relevance-based retrieval and decay

**Arbiter**:
- Softmax blending over band urgencies with temperature parameter
- Hysteresis to prevent action oscillation
- Budget constraints (energy limits from physiological band)
- Safety veto system (predator avoidance overrides)
- Decision history tracking for analysis

**Band 1: Physiological (Implemented)**:
- **Homeostatic tracking**: hunger, thirst, temperature discomfort, energy/hydration budgets
- **Perception**: Local fields + 5x5 neighborhood for gradient sensing
- **Threat response**: FLEE action with highest priority when predator detected
- **Metabolic behavior**:
  - Critical hunger (energy < 20): Move or forage urgently
  - Moderate hunger (energy < 70): Gradient-following toward vegetation
  - Low vegetation (< 0.25): Active search behavior
  - High vegetation (> 0.4): Forage in place
- **Gradient following**: Uses neighborhood vegetation to move toward better resources
- **Learning signal**: Homeostatic error reduction

### 3. Predator System (Complete)
- **Dynamic threat field** updated each simulation tick
- **Hunting behavior**: Predators move toward nearest agents within hunt_radius
- **Predation**: Reduces agent energy by 50%, can kill if energy drops to 0
- **Threat perception**: Agents sense local threat + 7x7 neighborhood threat gradient
- **Configurable**: num_predators, speed, hunt_radius, threat_decay

### 4. Agent Simulation Framework (Complete)
- `BandedAgent`: Orchestrates bands, arbiter, memory, learning, trajectory
- `AgentManager`: Manages agent populations, spawning, state tracking
- `AgentSimulation`: Full integration of environment, predators, agents
- **State tracking**: Energy, position, alive status, decision history
- **Population statistics**: Survival rates, energy distributions, predation events

### 5. Testing & Validation (18 tests passing)
- Agent creation and initialization
- Band perception and decision-making  
- Arbiter blending and veto logic
- Predator system and threat detection
- Full simulation runs with survival dynamics
- Movement and resource-seeking behavior

---

## CURRENT CAPABILITIES

### Emergent Behaviors Demonstrated:
1. **Survival under pressure**: Agents balance hunger vs. predator avoidance
2. **Resource-seeking**: Gradient following moves agents toward high-vegetation areas
3. **Threat response**: Immediate flee behavior when predators nearby
4. **Metabolic regulation**: Energy budgeting drives foraging decisions

### Research-Grade Features:
- **Deterministic**: Same seed → same outcomes (critical for reproducibility)
- **Extensible**: Band architecture allows adding new Maslow layers without refactoring
- **Observable**: Full decision history, population stats, trajectory tracking
- **Validated**: Comprehensive test suite ensures correctness

---

## NEXT STEPS

### Phase 1: Enhanced Band 1 & Validation (High Priority)
**Goal**: Ensure Band 1 is publication-ready before adding complexity

1. **Enhanced Migration Visualization**
   - Create animated plots showing agent trajectories over time
   - Overlay vegetation field to show gradient-following behavior
   - Color-code agents by energy level (red=starving, green=satiated)
   - Track individual agent paths to verify food-seeking

2. **Quantitative Validation**
   - Measure migration toward high-vegetation zones over 100+ ticks
   - Compare random-walk control vs. gradient-following agents
   - Statistical significance testing (t-tests on final vegetation location)
   - Generate publication-quality figures

3. **Stress Testing**
   - Sparse vegetation scenarios (10% coverage)
   - High predator density (10+ predators per 100 agents)
   - Energy depletion rates (test survival with different metabolic costs)
   - Population collapse vs. sustainable equilibria

4. **Documentation**
   - Methods section: Band 1 perception, urgency, proposal logic
   - Results section: Migration behavior, survival curves, emergent patterns
   - Discussion: Pure utilitarianism vs. bounded rationality

### Phase 2: Band 2 - Safety (Next Major Milestone)
**Goal**: Introduce spatial memory and territory avoidance

**Capabilities**:
- **Spatial memory**: Remember high-threat areas, avoid returning
- **Safe haven seeking**: Identify and return to low-threat zones
- **Risk assessment**: Trade off food quality vs. predator proximity
- **Territory formation**: Emergent safe zones where agents cluster

**Learning Signal**: Risk prediction error (expected vs. actual threat)

**Expected Emergent Behavior**: 
- Agents form "safe corridors" between resources
- Predators force agents into suboptimal but safe locations
- Trade-off between optimal foraging and safety

### Phase 3: Band 3 - Affiliation (Social Architecture)
**Goal**: Introduce agent-agent perception and social behavior

**Capabilities**:
- **Agent detection**: Perceive nearby agents within radius
- **Proximity preference**: Urgency to stay near others when threatened
- **Herding**: Emergent group formation under predation pressure
- **Social information**: Copy movement decisions of successful neighbors

**Learning Signal**: Social reward (proximity to others reduces threat urgency)

**Expected Emergent Behavior**:
- Spontaneous herd formation near predators
- Information cascades (one agent flees → others follow)
- Density-dependent predation (predators more effective on isolated agents)

### Phase 4: Institutional Emergence (Research Goal)
**Goal**: Observe proto-institutions from multi-band interactions

**Phenomena to Measure**:
1. **Territorial norms**: Do agents respect each other's foraging zones?
2. **Alarm signaling**: Do fleeing agents inadvertently warn others?
3. **Resource competition**: Escalation vs. sharing strategies
4. **Leadership**: Do some agents lead group movements?

**Key Research Question**: 
*"Can institutions (stable behavioral patterns constraining individual choice) emerge from psychological need hierarchies under environmental pressure, without explicit utility maximization?"*

---

## TECHNICAL DEBT & CLEANUP

Before moving to Band 2:

1. **Remove test files**: 
   - `debug_band1_decisions.py`
   - `test_migration_quick.py` 
   - `visualize_agent_migration.py`
   
2. **Consolidate into proper test suite**:
   - `tests/test_band1_migration.py` (quantitative validation)
   - `tests/test_band1_stress.py` (population dynamics under pressure)

3. **Code review**:
   - Add docstrings to all Band 1 methods
   - Type hints for all function signatures
   - Inline comments explaining urgency/proposal logic

4. **Performance optimization**:
   - Profile simulation loop (current bottleneck: hydration loading)
   - Consider caching neighborhood queries
   - Batch agent updates if needed for large populations

---

## FILES STRUCTURE

```
tholos/
├── interfaces/
│   ├── agent_iface/
│   │   ├── band.py                    # Abstract Band base class
│   │   ├── band_physiological.py      # Band 1 implementation
│   │   ├── arbiter.py                 # Action blending & veto logic
│   │   ├── banded_agent.py            # Agent orchestration
│   │   ├── agent_manager.py           # Population management
│   │   └── simulation.py              # Environment-agent integration
│   └── ui_iface/
│       ├── runner/
│       │   ├── engine.py              # Simulation engine
│       │   ├── hydrator.py            # State reconstruction
│       │   ├── predators.py           # Predator system
│       │   ├── agent_api.py           # EnvironmentGrid interface
│       │   └── viz.py                 # Visualization tools
│       ├── schemas/                   # YAML validation schemas
│       └── scenarios/
│           └── env-b.yaml             # Test environment config
├── tests/
│   ├── test_banded_survival.py        # Band 1 + predator integration
│   ├── test_minimal_agent.py          # Basic agent functionality
│   └── (82 other environment tests)
├── docs/
│   ├── BANDED_AGENT_DESIGN.md         # Full 7-band architecture spec
│   └── DEVELOPMENT_STATUS.md          # This file
└── demo_survival.py                   # Quick demo script
```

---

## RESEARCH TIMELINE

**Current Milestone**: Band 1 Complete ✓  
**Next Milestone**: Band 1 Validated (2-3 days)  
**Following Milestone**: Band 2 Implemented (1 week)  
**Research Goal**: 3-4 Bands Implemented (2-3 weeks)

**Publication Target**: 
- Demonstrate institutional emergence from Band 1-3 interactions
- Quantitative comparison: banded agents vs. utility-maximizing baseline
- Theoretical contribution: Maslow hierarchy as alternative to rational choice

---

## NOTES FOR COLLABORATORS

This is **research-grade infrastructure**, not a toy project. Every design decision serves the goal of demonstrating that:

1. Complex social behaviors can emerge from simple psychological needs
2. Institutions arise from environmental pressure on bounded agents
3. Pure utilitarianism is insufficient for modeling real agent behavior

**Critical Properties**:
- **Determinism**: Must be reproducible for peer review
- **Observability**: Must track every decision for analysis
- **Modularity**: Must add bands without breaking existing code
- **Validation**: Must quantitatively verify each emergent behavior

**DO NOT**:
- Add features "because they're cool"
- Skip testing for speed
- Hard-code agent strategies
- Break determinism for "realism"

**DO**:
- Question every design choice
- Demand quantitative evidence
- Document all emergent behaviors
- Compare to theoretical predictions

