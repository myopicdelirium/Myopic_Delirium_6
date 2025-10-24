# Band 1 Redesign: True Homeostatic Drive System

## Current Problems

### 1. No Continuous Depletion
```python
hunger = max(0.0, 1.0 - energy / 100.0)  # Instant calculation, not accumulation
```
- Hunger is just a function of energy, not an independent drive
- No passive metabolic cost - agents don't get hungrier over time
- Agents only move when energy < 70, so they stay put if energy is stable

### 2. No Attentional Focus
- All needs are computed in parallel
- No mechanism for "I'm focusing on hunger right now"
- No hysteresis or commitment to pursuing one goal before switching

### 3. Instant Gratification
- Foraging immediately satisfies hunger (energy goes up)
- No search/travel cost before reward
- No desperation gradient (mild hunger = extreme hunger in behavior)

## Proposed Solution: Homeostatic Drive Architecture

### Core Principles
1. **Drives deplete continuously** (hunger += 0.01 per tick, regardless of action)
2. **Actions cost resources** (moving drains energy faster)
3. **Focus tracks the dominant drive** (attention mechanism)
4. **Desperation escalates behavior** (wider search, more risk as deficit grows)

### New Internal State Variables

```python
{
    # Primary drives (0.0 = satisfied, 1.0 = critical)
    "hunger": 0.0,          # Increases ~0.01/tick, satisfied by eating
    "thirst": 0.0,          # Increases ~0.015/tick, satisfied by drinking
    "fatigue": 0.0,         # Increases ~0.005/tick, satisfied by resting
    "temperature_stress": 0.0,  # Increases in extreme temps
    
    # Resources (agent state)
    "energy": 100.0,        # Depleted by action, replenished by food
    "hydration": 100.0,     # Depleted over time, replenished by water
    
    # Focus/Attention
    "current_focus": None,  # Which drive is currently dominant
    "focus_strength": 0.0,  # How committed to this goal (hysteresis)
    "ticks_since_satisfaction": 0,  # How long since last success
    
    # Desperation
    "desperation_level": 0.0,  # 0-1, increases with unmet needs
    "search_radius": 2,     # Expands with desperation
    "risk_tolerance": 0.1   # Increases with desperation
}
```

### Metabolic Depletion Rules

**Every Tick (Passive)**:
```python
hunger += 0.01  # Base metabolic rate
thirst += 0.015  # Dehydration faster than starvation
fatigue += 0.005  # Slow accumulation
```

**Action Costs**:
```python
MOVE: energy -= 2.0, hunger += 0.02, thirst += 0.01, fatigue += 0.01
FORAGE: energy -= 1.0, hunger += 0.01, fatigue += 0.015 (searching is tiring)
REST: hunger += 0.005, fatigue -= 0.1 (reduced metabolism)
STAY: hunger += 0.01 (no extra cost)
```

**Rewards**:
```python
FORAGE (if vegetation > 0.3): 
    hunger -= vegetation * 0.2  # Proportional to food quality
    energy += vegetation * 10.0  # Energy from calories

DRINK (if hydration > 0.7):
    thirst -= hydration * 0.3
    hydration_state += 20.0
```

### Attentional Focus Mechanism

```python
def compute_focus(self):
    """Determine which drive should dominate attention."""
    drives = {
        "hunger": self.state.internal_state["hunger"],
        "thirst": self.state.internal_state["thirst"],
        "fatigue": self.state.internal_state["fatigue"],
        "threat": self.state.internal_state.get("immediate_threat", 0.0) * 2.0  # Threats get 2x weight
    }
    
    current_focus = self.state.internal_state.get("current_focus", None)
    focus_strength = self.state.internal_state.get("focus_strength", 0.0)
    
    # Hysteresis: harder to switch focus if currently committed
    if current_focus and current_focus in drives:
        drives[current_focus] += focus_strength * 0.3  # Bonus to current focus
    
    # Select most urgent drive
    dominant_drive = max(drives, key=drives.get)
    dominant_urgency = drives[dominant_drive]
    
    # Update focus
    if dominant_drive == current_focus:
        focus_strength = min(1.0, focus_strength + 0.1)  # Strengthen commitment
    else:
        if dominant_urgency > drives.get(current_focus, 0.0) + 0.2:  # Need 0.2 threshold to switch
            current_focus = dominant_drive
            focus_strength = 0.3  # Start with moderate commitment
        # else: maintain current focus
    
    self.state.internal_state["current_focus"] = current_focus
    self.state.internal_state["focus_strength"] = focus_strength
    
    return current_focus, dominant_urgency
```

### Desperation-Based Behavior Escalation

```python
def compute_desperation(self):
    """Desperation increases with unmet needs and failed searches."""
    hunger = self.state.internal_state["hunger"]
    thirst = self.state.internal_state["thirst"]
    ticks_since_satisfaction = self.state.internal_state.get("ticks_since_satisfaction", 0)
    
    # Desperation from deficits
    deficit_desperation = (hunger ** 2 + thirst ** 2) / 2.0  # Quadratic - gets severe fast
    
    # Desperation from time without success
    time_desperation = min(1.0, ticks_since_satisfaction / 100.0)
    
    desperation = max(deficit_desperation, time_desperation)
    self.state.internal_state["desperation_level"] = desperation
    
    # Desperation changes behavior
    base_search_radius = 2
    self.state.internal_state["search_radius"] = int(base_search_radius + desperation * 3)  # 2 -> 5
    self.state.internal_state["risk_tolerance"] = 0.1 + desperation * 0.4  # 0.1 -> 0.5
    
    return desperation
```

### Propose Actions (Emergent Behavior)

```python
def propose_actions(self, perception: Dict[str, Any]) -> List[ActionProposal]:
    """Actions emerge from focused drive and desperation level."""
    
    # 1. Update drives (depletion)
    self._update_drives()
    
    # 2. Compute focus
    focus, urgency = self.compute_focus()
    
    # 3. Compute desperation
    desperation = self.compute_desperation()
    
    # 4. Propose action based on focused drive
    if focus == "threat":
        return self._propose_flee_action(perception, urgency)
    elif focus == "hunger":
        return self._propose_hunger_action(perception, urgency, desperation)
    elif focus == "thirst":
        return self._propose_thirst_action(perception, urgency, desperation)
    elif focus == "fatigue":
        return self._propose_rest_action(perception, urgency)
    else:
        return [ActionProposal(action=Action.STAY, urgency=0.1, expected_value=0.0, 
                              band_id=self.band_id, params={})]

def _propose_hunger_action(self, perception, urgency, desperation):
    """Hunger-driven behavior: forage locally or search with increasing radius."""
    local_veg = perception.get("local_vegetation", 0.0)
    neighborhood_veg = perception.get("neighborhood_vegetation", None)
    
    # If food here, eat it
    if local_veg > 0.3:  # Good food
        return [ActionProposal(action=Action.FORAGE, urgency=urgency, 
                              expected_value=local_veg * 5.0, band_id=self.band_id,
                              params={"food_quality": local_veg})]
    
    # If desperate enough, even bad food is acceptable
    if local_veg > 0.1 and desperation > 0.5:
        return [ActionProposal(action=Action.FORAGE, urgency=urgency * 1.5,
                              expected_value=local_veg * 3.0, band_id=self.band_id,
                              params={"desperate_foraging": True})]
    
    # Otherwise, search for food (radius expands with desperation)
    search_radius = self.state.internal_state.get("search_radius", 2)
    best_direction = self._find_vegetation_direction(perception, search_radius)
    
    return [ActionProposal(action=best_direction, urgency=urgency * (1 + desperation),
                          expected_value=1.0, band_id=self.band_id,
                          params={"searching_food": True, "desperation": desperation})]
```

## Expected Emergent Behaviors

### Low Desperation (Well-Fed)
- Agents stay near current location
- Forage opportunistically
- Small movements to slightly better areas
- **Won't see much migration** (correctly!)

### Medium Desperation (Hungry)
- Active search behavior (radius 3-4 cells)
- Willing to travel 10-20 cells to better food
- Accept moderate-quality vegetation (0.2-0.4)
- Committed focus on hunger

### High Desperation (Starving)
- Wide-area search (radius 5+ cells)
- Willing to travel 50+ cells
- Accept any food (vegetation > 0.1)
- Risk tolerance increases (might ignore mild threats)
- **Strong migration behavior emerges**

## Implementation Steps

1. **Add continuous depletion to `update_state()`** (after each action)
2. **Implement `compute_focus()` with hysteresis**
3. **Implement `compute_desperation()` with escalation**
4. **Refactor `propose_actions()` to use focus + desperation**
5. **Test with low initial energy** (force immediate desperation)
6. **Test with adequate energy** (should see minimal migration until needs build up)

## Key Insight

**Migration should be RARE in well-resourced environments.**  
**Migration should be INEVITABLE when resources are scarce.**

This is emergent - we don't tell agents to migrate. They migrate because:
- Hunger accumulates over time
- Local food is insufficient
- Desperation expands search radius
- Focus commits them to finding food
- They travel until they find sufficient resources or die

Do you want me to implement this redesign?

