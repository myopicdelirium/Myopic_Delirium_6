import numpy as np
from typing import Dict, Any, List, Optional
from .band import Band, Action, ActionProposal

class PhysiologicalBand(Band):
    """
    Band 1: Physiological - True Homeostatic Drive System
    
    Key features:
    - Continuous drive depletion (hunger/thirst increase every tick)
    - Attentional focus (commit to satisfying one drive at a time)
    - Desperation escalation (search radius expands with unmet needs)
    - Metabolic costs (actions drain resources)
    
    Emergence: Migration occurs when local resources cannot sustain needs,
    not because we programmed "seek food."
    """
    
    # Metabolic constants (tuned for better migration)
    PASSIVE_HUNGER_RATE = 0.008     # Slower hunger accumulation
    PASSIVE_THIRST_RATE = 0.012     # Slower thirst
    PASSIVE_FATIGUE_RATE = 0.004    # Slower fatigue
    
    MOVE_ENERGY_COST = 1.0          # Reduced from 2.0 - cheaper to move
    MOVE_HUNGER_COST = 0.01         # Reduced from 0.02
    MOVE_THIRST_COST = 0.005        # Reduced from 0.01
    MOVE_FATIGUE_COST = 0.005       # Reduced from 0.01
    
    FORAGE_ENERGY_COST = 1.0
    FORAGE_FATIGUE_COST = 0.015
    
    REST_FATIGUE_RECOVERY = 0.1
    
    FOCUS_SWITCH_THRESHOLD = 0.2    # How much more urgent to switch focus
    FOCUS_BUILDUP_RATE = 0.1        # Commitment strengthens over time
    FOCUS_HYSTERESIS_BONUS = 0.3    # Bonus to current focus for stability
    
    def __init__(self, band_id: int = 1, initial_gain: float = 2.0, seed: int = None):
        super().__init__(band_id, initial_gain, seed)
        self.state.internal_state = {
            # Primary drives (0.0 = satisfied, 1.0 = critical)
            "hunger": 0.0,
            "thirst": 0.0,
            "fatigue": 0.0,
            "temperature_stress": 0.0,
            
            # Resources
            "energy": 100.0,
            "hydration": 100.0,
            
            # Focus/Attention mechanism
            "current_focus": None,
            "focus_strength": 0.0,
            "ticks_since_satisfaction": 0,
            
            # Desperation
            "desperation_level": 0.0,
            "search_radius": 2,
            "risk_tolerance": 0.1,
            
            # Tracking
            "last_action": None,
            "successful_forages": 0,
            "failed_searches": 0
        }
    
    def perceive(self, env_state: Dict[str, Any], agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """High-fidelity perception of local environment and neighborhood gradients."""
        return {
            "local_temperature": env_state.get("temperature", 0.5),
            "local_hydration": env_state.get("hydration", 0.5),
            "local_vegetation": env_state.get("vegetation", 0.0),
            "local_threat": env_state.get("threat", 0.0),
            "neighborhood_threat": env_state.get("neighborhood_threat", np.zeros((5, 5))),
            "neighborhood_vegetation": env_state.get("neighborhood_vegetation", None),
            "neighborhood_hydration": env_state.get("neighborhood_hydration", None),
            "energy": agent_state.get("energy", 0.0),
            "position": agent_state.get("position", (0, 0)),
            "tick": agent_state.get("tick", 0)
        }
    
    def compute_urgency(self, perception: Dict[str, Any]) -> float:
        """Urgency emerges from focused drive and desperation level."""
        # Update drives (passive depletion)
        self._apply_passive_depletion()
        
        # Compute focus (which drive dominates attention)
        focus, focus_urgency = self._compute_focus(perception)
        
        # Compute desperation (escalates with unmet needs)
        desperation = self._compute_desperation()
        
        # Overall urgency = focused drive urgency * desperation multiplier
        urgency = focus_urgency * (1.0 + desperation)
        self.state.urgency = urgency * self.state.gain
        
        return self.state.urgency
    
    def propose_actions(self, perception: Dict[str, Any]) -> List[ActionProposal]:
        """Actions emerge from focused drive and desperation level."""
        focus = self.state.internal_state.get("current_focus", None)
        urgency = self.state.urgency
        desperation = self.state.internal_state.get("desperation_level", 0.0)
        
        # Propose action based on focused drive
        if focus == "threat":
            return self._propose_flee_action(perception, urgency)
        elif focus == "hunger":
            return self._propose_hunger_action(perception, urgency, desperation)
        elif focus == "thirst":
            return self._propose_thirst_action(perception, urgency, desperation)
        elif focus == "fatigue":
            return self._propose_rest_action(perception, urgency)
        else:
            # No urgent needs - just stay
            return [ActionProposal(
                action=Action.STAY,
                urgency=0.1,
                expected_value=0.0,
                band_id=self.band_id,
                params={"reason": "content"}
            )]
    
    def update_state(self, perception: Dict[str, Any], action_taken: Action, outcome: Dict[str, Any]):
        """Update internal state based on action costs and rewards."""
        self.state.internal_state["last_action"] = action_taken
        
        # Apply action costs
        if action_taken in [Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST]:
            self.state.internal_state["energy"] = max(0.0, 
                self.state.internal_state["energy"] - self.MOVE_ENERGY_COST)
            self.state.internal_state["hunger"] = min(1.0,
                self.state.internal_state["hunger"] + self.MOVE_HUNGER_COST)
            self.state.internal_state["thirst"] = min(1.0,
                self.state.internal_state["thirst"] + self.MOVE_THIRST_COST)
            self.state.internal_state["fatigue"] = min(1.0,
                self.state.internal_state["fatigue"] + self.MOVE_FATIGUE_COST)
            
        elif action_taken == Action.FORAGE:
            self.state.internal_state["energy"] = max(0.0,
                self.state.internal_state["energy"] - self.FORAGE_ENERGY_COST)
            self.state.internal_state["fatigue"] = min(1.0,
                self.state.internal_state["fatigue"] + self.FORAGE_FATIGUE_COST)
            
            # Apply rewards if successful
            local_veg = perception.get("local_vegetation", 0.0)
            if local_veg > 0.2:
                hunger_reduction = local_veg * 0.2
                energy_gain = local_veg * 10.0
                
                self.state.internal_state["hunger"] = max(0.0,
                    self.state.internal_state["hunger"] - hunger_reduction)
                self.state.internal_state["energy"] = min(100.0,
                    self.state.internal_state["energy"] + energy_gain)
                
                self.state.internal_state["successful_forages"] += 1
                self.state.internal_state["ticks_since_satisfaction"] = 0
            else:
                self.state.internal_state["failed_searches"] += 1
                
        elif action_taken == Action.DRINK:
            # Apply thirst reduction if water available
            local_hydration = perception.get("local_hydration", 0.5)
            if local_hydration > 0.7:
                thirst_reduction = (local_hydration - 0.7) * 0.5  # Proportional to water quality
                self.state.internal_state["thirst"] = max(0.0,
                    self.state.internal_state["thirst"] - thirst_reduction)
                self.state.internal_state["ticks_since_satisfaction"] = 0
            
        elif action_taken == Action.REST:
            self.state.internal_state["fatigue"] = max(0.0,
                self.state.internal_state["fatigue"] - self.REST_FATIGUE_RECOVERY)
            self.state.internal_state["hunger"] = min(1.0,
                self.state.internal_state["hunger"] + self.PASSIVE_HUNGER_RATE * 0.5)
        
        # Increment ticks since last satisfaction
        if action_taken not in [Action.FORAGE, Action.DRINK] or \
           (action_taken == Action.FORAGE and perception.get("local_vegetation", 0.0) < 0.2) or \
           (action_taken == Action.DRINK and perception.get("local_hydration", 0.5) < 0.7):
            self.state.internal_state["ticks_since_satisfaction"] += 1
        
        # Frustration accumulation
        if self.state.internal_state.get("desperation_level", 0.0) > 0.6:
            self.state.frustration_accumulator = min(1.0, self.state.frustration_accumulator + 0.05)
        else:
            self.state.frustration_accumulator = max(0.0, self.state.frustration_accumulator - 0.02)
    
    def compute_learning_signal(self, perception: Dict[str, Any], action: Action, outcome: Dict[str, Any]) -> float:
        """Learning signal from homeostatic error reduction."""
        # Positive signal if action reduced focused drive
        focus = self.state.internal_state.get("current_focus", None)
        if focus == "hunger" and action == Action.FORAGE:
            local_veg = perception.get("local_vegetation", 0.0)
            return local_veg * 0.5  # Reward proportional to food quality
        return 0.0
    
    # ========== Internal Helper Methods ==========
    
    def _apply_passive_depletion(self):
        """Drives continuously deplete over time (base metabolism)."""
        self.state.internal_state["hunger"] = min(1.0,
            self.state.internal_state["hunger"] + self.PASSIVE_HUNGER_RATE)
        self.state.internal_state["thirst"] = min(1.0,
            self.state.internal_state["thirst"] + self.PASSIVE_THIRST_RATE)
        self.state.internal_state["fatigue"] = min(1.0,
            self.state.internal_state["fatigue"] + self.PASSIVE_FATIGUE_RATE)
    
    def _compute_focus(self, perception: Dict[str, Any]) -> tuple[Optional[str], float]:
        """Determine which drive should dominate attention (with adaptive hysteresis)."""
        # Compute all drive urgencies
        hunger = self.state.internal_state["hunger"]
        thirst = self.state.internal_state["thirst"]
        fatigue = self.state.internal_state["fatigue"]
        threat = perception.get("local_threat", 0.0)
        
        drives = {
            "hunger": hunger * 2.0,      # Base weights
            "thirst": thirst * 1.3,      # Reduced from 1.5 to balance with hunger
            "fatigue": fatigue * 0.8,
            "threat": threat * 10.0      # Threats get highest priority
        }
        
        current_focus = self.state.internal_state.get("current_focus", None)
        focus_strength = self.state.internal_state.get("focus_strength", 0.0)
        
        # ADAPTIVE HYSTERESIS: weaker when drives are extreme
        max_drive = max(drives.values())
        hysteresis_multiplier = 1.0
        if max_drive > 2.0:  # Critical level (e.g., hunger=1.0 * 2.0 weight)
            hysteresis_multiplier = 0.3  # Much easier to switch when desperate
        elif max_drive > 1.5:
            hysteresis_multiplier = 0.6  # Moderate resistance
        
        # Apply hysteresis bonus (reduced when desperate)
        if current_focus and current_focus in drives:
            hysteresis_bonus = focus_strength * self.FOCUS_HYSTERESIS_BONUS * hysteresis_multiplier
            drives[current_focus] += hysteresis_bonus
        
        # Find most urgent drive
        dominant_drive = max(drives, key=drives.get)
        dominant_urgency = drives[dominant_drive]
        
        # Decide whether to switch focus
        if dominant_drive == current_focus:
            # Strengthen commitment to current focus (slower buildup when desperate)
            buildup_rate = self.FOCUS_BUILDUP_RATE * (1.0 if max_drive < 1.5 else 0.5)
            focus_strength = min(1.0, focus_strength + buildup_rate)
        else:
            # Switch threshold adapts: lower when drives are extreme
            switch_threshold = self.FOCUS_SWITCH_THRESHOLD * hysteresis_multiplier
            
            current_urgency = drives.get(current_focus, 0.0) if current_focus else 0.0
            
            # CRITICAL OVERRIDE: If any drive is critical (>0.9), force switch if it's higher
            critical_drives = {k: v for k, v in 
                             [("hunger", hunger), ("thirst", thirst), ("fatigue", fatigue)]
                             if v > 0.9}
            
            if critical_drives and dominant_drive in critical_drives:
                # Force switch to critical need regardless of hysteresis
                current_focus = dominant_drive
                focus_strength = 0.2  # Low initial commitment (ready to switch again)
            elif dominant_urgency > current_urgency + switch_threshold:
                # Normal switch
                current_focus = dominant_drive
                focus_strength = 0.3
            # else: maintain current focus
        
        self.state.internal_state["current_focus"] = current_focus
        self.state.internal_state["focus_strength"] = focus_strength
        
        return current_focus, dominant_urgency
    
    def _compute_desperation(self) -> float:
        """Desperation increases with unmet needs and failed searches."""
        hunger = self.state.internal_state["hunger"]
        thirst = self.state.internal_state["thirst"]
        ticks_since_satisfaction = self.state.internal_state.get("ticks_since_satisfaction", 0)
        
        # Desperation from deficits (quadratic - gets severe quickly)
        deficit_desperation = (hunger ** 2 + thirst ** 2) / 2.0
        
        # Desperation from prolonged failure to satisfy needs
        time_desperation = min(1.0, ticks_since_satisfaction / 50.0)
        
        desperation = max(deficit_desperation, time_desperation)
        self.state.internal_state["desperation_level"] = desperation
        
        # Desperation changes search behavior (wider radius for desperate agents)
        base_search_radius = 2
        self.state.internal_state["search_radius"] = int(base_search_radius + desperation * 8)  # 2 -> 10
        self.state.internal_state["risk_tolerance"] = 0.1 + desperation * 0.5  # 0.1 -> 0.6
        
        return desperation
    
    # ========== Action Proposal Methods ==========
    
    def _propose_flee_action(self, perception: Dict[str, Any], urgency: float) -> List[ActionProposal]:
        """Flee from immediate threats."""
        threat_gradient = perception.get("neighborhood_threat", np.zeros((5, 5)))
        safe_direction = self._find_safest_direction(threat_gradient)
        
        return [ActionProposal(
            action=safe_direction,
            urgency=urgency,
            expected_value=1.0,
            band_id=self.band_id,
            params={"reason": "flee_predator", "threat_level": perception.get("local_threat", 0.0)}
        )]
    
    def _propose_hunger_action(self, perception: Dict[str, Any], urgency: float, desperation: float) -> List[ActionProposal]:
        """Hunger-driven behavior: forage locally or search with expanding radius."""
        local_veg = perception.get("local_vegetation", 0.0)
        
        # Threshold for acceptable food decreases with desperation
        acceptable_threshold = 0.3 - desperation * 0.2  # 0.3 -> 0.1
        
        # If acceptable food here, forage
        if local_veg > acceptable_threshold:
            return [ActionProposal(
                action=Action.FORAGE,
                urgency=urgency,
                expected_value=local_veg * 5.0,
                band_id=self.band_id,
                params={"food_quality": local_veg, "desperate": desperation > 0.5}
            )]
        
        # Otherwise, search for food (radius expands with desperation)
        best_direction = self._find_vegetation_direction(perception)
        
        return [ActionProposal(
            action=best_direction,
            urgency=urgency * (1.0 + desperation * 0.5),
            expected_value=1.0,
            band_id=self.band_id,
            params={"searching_food": True, "desperation": desperation,
                   "search_radius": self.state.internal_state.get("search_radius", 2)}
        )]
    
    def _propose_thirst_action(self, perception: Dict[str, Any], urgency: float, desperation: float) -> List[ActionProposal]:
        """Thirst-driven behavior: drink or search for water."""
        local_hydration = perception.get("local_hydration", 0.5)
        
        if local_hydration > 0.7:
            return [ActionProposal(
                action=Action.DRINK,
                urgency=urgency,
                expected_value=local_hydration * 4.0,
                band_id=self.band_id,
                params={"water_quality": local_hydration}
            )]
        
        # Search for water
        best_direction = self._find_water_direction(perception)
        return [ActionProposal(
            action=best_direction,
            urgency=urgency,
            expected_value=1.0,
            band_id=self.band_id,
            params={"searching_water": True}
        )]
    
    def _propose_rest_action(self, perception: Dict[str, Any], urgency: float) -> List[ActionProposal]:
        """Rest to recover from fatigue."""
        return [ActionProposal(
            action=Action.REST,
            urgency=urgency,
            expected_value=0.5,
            band_id=self.band_id,
            params={"fatigue_level": self.state.internal_state["fatigue"]}
        )]
    
    # ========== Navigation Methods ==========
    
    def _find_safest_direction(self, threat_field: np.ndarray) -> Action:
        """Find direction with lowest threat."""
        if threat_field.size == 0:
            return Action.STAY
        
        center = threat_field.shape[0] // 2
        
        directions = {
            Action.MOVE_NORTH: threat_field[center-1, center] if center > 0 else 1.0,
            Action.MOVE_SOUTH: threat_field[center+1, center] if center < threat_field.shape[0]-1 else 1.0,
            Action.MOVE_EAST: threat_field[center, center+1] if center < threat_field.shape[1]-1 else 1.0,
            Action.MOVE_WEST: threat_field[center, center-1] if center > 0 else 1.0
        }
        
        return min(directions, key=directions.get)
    
    def _find_vegetation_direction(self, perception: Dict[str, Any]) -> Action:
        """Move toward higher vegetation using gradient following."""
        neighborhood_veg = perception.get("neighborhood_vegetation", None)
        
        if neighborhood_veg is None or neighborhood_veg.size == 0:
            return self.rng.choice([Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST])
        
        center_y, center_x = neighborhood_veg.shape[0] // 2, neighborhood_veg.shape[1] // 2
        current_veg = neighborhood_veg[center_y, center_x]
        
        directions = {}
        if center_y > 0:
            directions[Action.MOVE_NORTH] = neighborhood_veg[center_y - 1, center_x]
        if center_y < neighborhood_veg.shape[0] - 1:
            directions[Action.MOVE_SOUTH] = neighborhood_veg[center_y + 1, center_x]
        if center_x < neighborhood_veg.shape[1] - 1:
            directions[Action.MOVE_EAST] = neighborhood_veg[center_y, center_x + 1]
        if center_x > 0:
            directions[Action.MOVE_WEST] = neighborhood_veg[center_y, center_x - 1]
        
        if not directions:
            return Action.STAY
        
        best_direction = max(directions, key=directions.get)
        
        # Move toward better vegetation (threshold decreases with desperation)
        desperation = self.state.internal_state.get("desperation_level", 0.0)
        gradient_threshold = 0.03 * (1.0 - desperation * 0.7)  # 0.03 -> 0.009 when desperate
        
        if directions[best_direction] > current_veg + gradient_threshold:
            return best_direction
        else:
            # When desperate, still follow small gradients; when content, explore randomly
            if desperation > 0.5:
                return best_direction  # Follow any upward gradient when desperate
            else:
                return self.rng.choice(list(directions.keys()))  # Random when content
    
    def _find_water_direction(self, perception: Dict[str, Any]) -> Action:
        """Move toward higher hydration."""
        neighborhood_hyd = perception.get("neighborhood_hydration", None)
        
        if neighborhood_hyd is None or neighborhood_hyd.size == 0:
            return self.rng.choice([Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST])
        
        center_y, center_x = neighborhood_hyd.shape[0] // 2, neighborhood_hyd.shape[1] // 2
        current_hyd = neighborhood_hyd[center_y, center_x]
        
        directions = {}
        if center_y > 0:
            directions[Action.MOVE_NORTH] = neighborhood_hyd[center_y - 1, center_x]
        if center_y < neighborhood_hyd.shape[0] - 1:
            directions[Action.MOVE_SOUTH] = neighborhood_hyd[center_y + 1, center_x]
        if center_x < neighborhood_hyd.shape[1] - 1:
            directions[Action.MOVE_EAST] = neighborhood_hyd[center_y, center_x + 1]
        if center_x > 0:
            directions[Action.MOVE_WEST] = neighborhood_hyd[center_y, center_x - 1]
        
        if not directions:
            return Action.STAY
        
        best_direction = max(directions, key=directions.get)
        
        if directions[best_direction] > current_hyd + 0.05:
            return best_direction
        else:
            return self.rng.choice(list(directions.keys()))
    
    def _get_decay_probabilities(self) -> np.ndarray:
        """Uniform decay for physiological memories - short-term focus."""
        if not self.memory:
            return np.array([])
        return np.ones(len(self.memory))
    
    def _compute_relevance(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Relevance based on similar homeostatic states."""
        if "perception_summary" not in memory:
            return 0.0
        
        mem_hunger = memory["perception_summary"].get("hunger", 0.0)
        ctx_hunger = self.state.internal_state.get("hunger", 0.0)
        
        relevance = 1.0 - abs(mem_hunger - ctx_hunger)
        
        if memory.get("affect", 0.0) > 0:
            relevance *= 1.2
        
        return max(0.0, relevance)
