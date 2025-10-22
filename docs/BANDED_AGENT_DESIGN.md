# Banded Controller Architecture - Design Specification

## Overview

Agents operate as **banded controllers** where each band corresponds to a Maslow hierarchy layer. Each band owns:
- Sensors (perception transform)
- Action repertoire  
- Internal models (state, beliefs, predictions)
- Learning signals (band-specific reward functions)

A global **arbiter** blends action proposals using soft priority with hysteresis to prevent thrashing.

---

## Band Hierarchy

### Band 1: Physiological (Metabolic Governor)
**Status**: Implemented
**Homeostats**: hunger, thirst, temperature, fatigue, reproductive arousal
**Actions**: FORAGE, DRINK, REST, SEEK_SHELTER
**Learning Signal**: Homeostatic error reduction
**Perception**: Narrow, high-fidelity local environment
**Memory Decay**: Uniform (short-term focus)
**Failure Modes**: Starvation loops, dehydration spirals
**Mitigation**: Strong hysteresis, path-planning with movement cost

### Band 2: Safety (Threat & Stability Manager)
**Status**: To Implement
**State**: Predation risk, disease exposure, injury likelihood, environmental volatility
**Actions**: FLEE, FREEZE, GROUP_UP, SHELTER_REINFORCEMENT, AVOID_CONTAMINATED
**Learning Signal**: Asymmetric (negative outcomes update faster - loss aversion)
**Perception**: Risk assessment, threat detection
**Memory Decay**: Negative events persist longer
**Special**: Maintains "safe-return map" of refuges
**Priority**: Can pre-empt consumption and social overtures

### Band 3: Affiliation & Belonging (Social Glue)
**Status**: To Implement
**State**: Dyadic trust, reciprocity histories, kinship tags, group identity, norms
**Actions**: JOIN/FOLLOW, SHARE/TRADE, GROOM/HELP, IMITATE, SYNCHRONIZE
**Learning Signal**: Reduction in social uncertainty, increased coalition support
**Perception**: Partner-specific cues, group signals
**Memory Decay**: Affiliation warmth decays slower within kin clusters
**Special**: Manages communication primitives, learns conventions

### Band 4: Esteem & Status (Prestige & Role-Finding)
**Status**: To Implement
**State**: Comparative advantage, reputation scores, positional goods access
**Actions**: DEMONSTRATE_COMPETENCE, FAIR_PUNISHMENT, LEADERSHIP_BID, APPRENTICE, TERRITORIAL_SIGNAL
**Learning Signal**: Reputation delta, increased deference/access
**Perception**: Skill comparisons, reputation feedback
**Memory Decay**: Esteem wins fade unless reinforced by practice
**Special**: Shapes division of labor via marginal esteem gains

### Band 5: Cognitive Play & Understanding (Curiosity & Modeling)
**Status**: To Implement
**State**: Environment regularities, other-agent models, causal hypotheses
**Actions**: EXPLORE, INFORMATION_GATHER, SIMULATE_TACTICS, TOOL_EXPERIMENT
**Learning Signal**: Prediction error reduction (discounted by risk/opportunity cost)
**Perception**: Pattern detection, anomaly identification
**Memory Decay**: Significant discoveries persist
**Special**: Refines other bands' forecasts, improves long-horizon planning

### Band 6: Self-Actualization & Craft Mastery (Skill Deepening)
**Status**: To Implement
**State**: Craft niche selection, competence levels, flow states
**Actions**: PRACTICE_CRAFT, TOOL_IMPROVEMENT, MENTORSHIP
**Learning Signal**: Competence growth, flow-like states, productivity gains
**Perception**: Skill progress tracking
**Memory Decay**: Mastery milestones persist
**Special**: Coordinates with Band 4 (esteem) and Band 3 (institutionalize roles)

### Band 7: Transcendence & Ideology (Value-System Editor)
**Status**: To Implement
**State**: Belief structures, symbolic markers, sacred values
**Actions**: NORM_CODIFY, RITUAL_COST, PUNITIVE_COALITION, PROSELYTIZE
**Learning Signal**: Identity coherence, perceived meaning, group durability
**Perception**: Cultural exposure, ritual participation
**Memory Decay**: Sacred events highly persistent
**Special**: Can rescale weights of ALL bands via reward mutation
**Guardrails**: Arbiter prevents suicidal cascades unless culture evolved martyr rules

---

## Arbiter Design

### Action Blending Algorithm

```python
def blend_actions(band_proposals, current_action, inertia=0.3):
    # Soft-max over urgencies with hysteresis
    urgencies = [p.urgency * band.state.gain for p in band_proposals]
    
    # Add inertia bonus to current dominant band
    if current_action is not None:
        for i, p in enumerate(band_proposals):
            if p.action == current_action:
                urgencies[i] *= (1.0 + inertia)
    
    # Safety band veto hook
    safety_veto = check_safety_veto(band_proposals)
    if safety_veto:
        return safety_veto_action
    
    # Physiological budget constraints
    energy_sufficient = check_energy_budget(band_proposals)
    if not energy_sufficient:
        return force_physiological_action()
    
    # Softmax blend
    probs = softmax(urgencies)
    composite_action = weighted_blend(band_proposals, probs)
    
    return composite_action
```

### Composite Actions

Arbiter can produce composite actions:
- "Move toward water via covered route while signaling to allies"
- "Forage while monitoring for threats"
- "Practice craft in safe area near group"

### Meta-Controller

Monitors chronic frustration:
```python
def check_band_starvation():
    for band in bands:
        if band.state.frustration_accumulator > threshold:
            # Force lifestyle change
            if band_id in [1, 2]:  # Survival bands
                trigger_migration()
            elif band_id in [3, 4]:  # Social bands
                trigger_group_switch()
            elif band_id in [5, 6]:  # Mastery bands
                trigger_craft_change()
```

---

## Memory System

### Structure
```python
memory_entry = {
    "band_id": int,              # Which band wrote this
    "tick": int,                 # When
    "perception_summary": dict,  # Compressed percept
    "action": Action,            # What was done
    "outcome_summary": dict,     # What happened
    "affect": float,             # Emotional valence
    "dominant_band": int,        # Which band was in control
    "partners": List[int]        # Other agents involved (if any)
}
```

### Cross-Indexing

Bands query memory with their own priors:
- Band 1: Similar homeostatic states
- Band 2: Threat-relevant contexts, negative affect
- Band 3: Partner-specific histories
- Band 4: Reputation-relevant interactions
- Band 5: Novel/surprising patterns
- Band 6: Skill-practice outcomes
- Band 7: Identity-relevant events

Same episode can feed different lessons across bands → cultural drift.

### Decay Rules

```python
band_decay_bias = {
    1: "uniform",                # Short-term physiological
    2: "negative_persistent",    # Safety events persist
    3: "kin_slower",            # Affiliation with kin
    4: "practice_gated",        # Esteem needs reinforcement
    5: "prediction_error",      # Surprising memories persist
    6: "milestone_persistent",  # Mastery achievements
    7: "sacred_immortal"        # Sacred events eternal
}
```

---

## Developmental Stages

Bands unlock progressively:

| Age Stage | Active Bands | Behavior Profile |
|-----------|-------------|------------------|
| Juvenile | 1-2 + proto-3 | Survival + attachment |
| Adolescent | 1-5 | + Esteem seeking + exploration |
| Adult | 1-6 | + Craft specialization |
| Elder | 1-7 | + Ideological transmission |

Gated by:
- Physiology (age, health)
- Social exposure (group membership, mentors)
- Resource abundance (Band 6-7 need stable affluence)

---

## Learning Mechanisms

| Bands | Mechanism |
|-------|-----------|
| 1-2 | Fast TD updates, hard constraints |
| 3-4 | Reputation-weighted imitation, Bayesian partner models |
| 5-6 | Model-based planning, skill consolidation |
| 7 | Cultural selection (memes with higher coalition payoff spread) |

Noise and partial observability ensure no band achieves perfect control.

---

## Coordination & Emergence

### Signal Externalization

Each band naturally broadcasts:
- **Safety**: Alarm calls
- **Affiliation**: Food sharing, gossip
- **Esteem**: Demonstrations
- **Cognition**: Teachable discoveries
- **Transcendence**: Symbols, rituals

### Institutional Formation

Power structures emerge when agents can set arbitration context for others:
- Saturate field with sacred symbols → raise Band 7 urgency
- Control resources → permanently bias Band 1
- Monopolize expertise → gate Band 6 access

Path from psychology → institutions → hierarchy → ideological conflict.

---

## Implementation Phases

### Phase 1: Core Architecture (Current)
- [x] Band base class with perception/urgency/proposal/learning
- [x] Band 1 (Physiological) implementation
- [ ] Arbiter with softmax blending + hysteresis
- [ ] Memory system with cross-indexing
- [ ] Meta-controller for frustration monitoring

### Phase 2: Essential Bands
- [ ] Band 2 (Safety) - critical for survival
- [ ] Band 3 (Affiliation) - critical for coordination
- [ ] Simple multi-agent testing

### Phase 3: Higher Bands
- [ ] Band 4 (Esteem) - division of labor
- [ ] Band 5 (Cognition) - learning & adaptation
- [ ] Band 6 (Mastery) - specialization
- [ ] Band 7 (Transcendence) - culture & ideology

### Phase 4: Institutional Emergence
- [ ] Partner tracking & reputation
- [ ] Skill differentiation
- [ ] Norm formation & enforcement
- [ ] Cultural transmission
- [ ] Power structure dynamics

---

## Testing Strategy

### Unit Tests
- Each band's urgency computation
- Action proposal generation
- Learning signal calculation
- Memory decay mechanics

### Integration Tests
- Arbiter blending with conflicting urgencies
- Band switching with hysteresis
- Frustration accumulation → gain adaptation
- Cross-band memory queries

### Behavioral Tests
- Starvation avoidance (Band 1 dominance)
- Threat response (Band 2 override)
- Social cooperation emergence (Band 3)
- Division of labor (Band 4)
- Knowledge accumulation (Band 5)
- Specialization (Band 6)
- Norm enforcement (Band 7)

### Emergent Phenomena Tests
- Trade networks form
- Leadership structures emerge
- Cultural variants drift
- Institutional stability under shocks

---

## Next Steps

1. Implement Arbiter (softmax blending with hysteresis)
2. Implement Band 2 (Safety)
3. Implement Band 3 (Affiliation) 
4. Create BandedAgent class that orchestrates all bands
5. Test with multi-agent scenarios
6. Iterate based on emergent behaviors

This architecture is designed for **institutional-grade research** into non-utilitarian decision-making, emergent social structures, and cultural evolution.

