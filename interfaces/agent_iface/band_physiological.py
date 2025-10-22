import numpy as np
from typing import Dict, Any, List
from .band import Band, Action, ActionProposal

class PhysiologicalBand(Band):
    """
    Band 1: Physiological
    Metabolic governor tracking hunger, thirst, temperature, fatigue, reproductive arousal.
    Maintains energy and hydration budgets, predicts depletion, proposes short-horizon actions.
    Learning signal: homeostatic error reduction.
    """
    
    def __init__(self, band_id: int = 1, initial_gain: float = 2.0, seed: int = None):
        super().__init__(band_id, initial_gain, seed)
        self.state.internal_state = {
            "hunger": 0.0,
            "thirst": 0.0,
            "temperature_discomfort": 0.0,
            "fatigue": 0.0,
            "energy_budget": 100.0,
            "hydration_budget": 100.0,
            "reproductive_arousal": 0.0
        }
    
    def perceive(self, env_state: Dict[str, Any], agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """High-fidelity, narrow perception field for metabolic needs and immediate threats."""
        return {
            "local_temperature": env_state.get("temperature", 0.5),
            "local_hydration": env_state.get("hydration", 0.5),
            "local_vegetation": env_state.get("vegetation", 0.0),
            "local_threat": env_state.get("threat", 0.0),
            "neighborhood_threat": env_state.get("neighborhood_threat", np.zeros((5, 5))),
            "energy": agent_state.get("energy", 0.0),
            "position": agent_state.get("position", (0, 0)),
            "tick": agent_state.get("tick", 0)
        }
    
    def compute_urgency(self, perception: Dict[str, Any]) -> float:
        """Urgency from homeostatic deficits AND immediate survival threats."""
        energy = perception.get("energy", 0.0)
        
        hunger = max(0.0, 1.0 - energy / 100.0)
        self.state.internal_state["hunger"] = hunger
        
        local_threat = perception.get("local_threat", 0.0)
        self.state.internal_state["immediate_threat"] = local_threat
        
        local_temp = perception.get("local_temperature", 0.5)
        temp_optimal = 0.6
        temp_discomfort = abs(local_temp - temp_optimal) / 0.5
        self.state.internal_state["temperature_discomfort"] = temp_discomfort
        
        local_hydration = perception.get("local_hydration", 0.5)
        thirst = max(0.0, 1.0 - local_hydration)
        self.state.internal_state["thirst"] = thirst
        
        threat_urgency = local_threat * 10.0
        metabolic_urgency = hunger * 2.0 + thirst * 1.5 + temp_discomfort * 0.5
        
        urgency = max(threat_urgency, metabolic_urgency)
        self.state.urgency = urgency * self.state.gain
        
        return self.state.urgency
    
    def propose_actions(self, perception: Dict[str, Any]) -> List[ActionProposal]:
        """Propose survival actions: FLEE from threats, then forage/drink/rest."""
        proposals = []
        
        immediate_threat = self.state.internal_state.get("immediate_threat", 0.0)
        if immediate_threat > 0.3:
            threat_gradient = perception.get("neighborhood_threat", np.zeros((5, 5)))
            safe_direction = self._find_safest_direction(threat_gradient)
            
            proposals.append(ActionProposal(
                action=safe_direction,
                urgency=immediate_threat * 10.0,
                expected_value=1.0 - immediate_threat,
                band_id=self.band_id,
                params={"reason": "flee_predator"}
            ))
            return proposals
        
        energy = perception.get("energy", 0.0)
        if energy < 20.0:
            vegetation_value = perception.get("local_vegetation", 0.0)
            proposals.append(ActionProposal(
                action=Action.FORAGE if vegetation_value > 0.1 else self._find_vegetation_direction(perception),
                urgency=5.0,
                expected_value=vegetation_value * 5.0,
                band_id=self.band_id,
                params={"reason": "critical_hunger"}
            ))
            return proposals
        
        hunger = self.state.internal_state["hunger"]
        if hunger > 0.3:
            vegetation_value = perception.get("local_vegetation", 0.0)
            if vegetation_value > 0.2:
                proposals.append(ActionProposal(
                    action=Action.FORAGE,
                    urgency=hunger * 2.0,
                    expected_value=vegetation_value * 5.0,
                    band_id=self.band_id,
                    params={"target": "vegetation"}
                ))
        
        thirst = self.state.internal_state["thirst"]
        if thirst > 0.4:
            hydration_value = perception.get("local_hydration", 0.0)
            proposals.append(ActionProposal(
                action=Action.DRINK if hydration_value > 0.5 else self._find_water_direction(perception),
                urgency=thirst * 1.5,
                expected_value=hydration_value * 4.0,
                band_id=self.band_id,
                params={"target": "hydration"}
            ))
        
        if not proposals:
            proposals.append(ActionProposal(
                action=Action.STAY,
                urgency=0.1,
                expected_value=0.0,
                band_id=self.band_id,
                params={}
            ))
        
        return proposals
    
    def _find_safest_direction(self, threat_field: np.ndarray) -> Action:
        """Find direction with lowest threat."""
        center = threat_field.shape[0] // 2
        
        directions = {
            Action.MOVE_NORTH: threat_field[center-1, center] if center > 0 else 1.0,
            Action.MOVE_SOUTH: threat_field[center+1, center] if center < threat_field.shape[0]-1 else 1.0,
            Action.MOVE_EAST: threat_field[center, center+1] if center < threat_field.shape[1]-1 else 1.0,
            Action.MOVE_WEST: threat_field[center, center-1] if center > 0 else 1.0
        }
        
        return min(directions, key=directions.get)
    
    def _find_vegetation_direction(self, perception: Dict[str, Any]) -> Action:
        """Move toward higher vegetation if available."""
        return self.rng.choice([Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST])
    
    def _find_water_direction(self, perception: Dict[str, Any]) -> Action:
        """Move toward water if available."""
        return self.rng.choice([Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST])
    
    def update_state(self, perception: Dict[str, Any], action_taken: Action, outcome: Dict[str, Any]):
        """Update homeostatic budgets based on action outcome."""
        energy_delta = outcome.get("energy_delta", 0.0)
        self.state.internal_state["energy_budget"] = max(0.0, 
            self.state.internal_state["energy_budget"] + energy_delta
        )
        
        if action_taken in [Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST]:
            self.state.internal_state["fatigue"] = min(1.0, 
                self.state.internal_state.get("fatigue", 0.0) + 0.05
            )
        elif action_taken == Action.REST:
            self.state.internal_state["fatigue"] = max(0.0,
                self.state.internal_state.get("fatigue", 0.0) - 0.2
            )
        
        if action_taken != Action.FORAGE and action_taken != Action.DRINK:
            self.state.frustration_accumulator += 0.1
        else:
            self.state.frustration_accumulator = max(0.0, self.state.frustration_accumulator - 0.5)
    
    def compute_learning_signal(self, perception: Dict[str, Any], action: Action, outcome: Dict[str, Any]) -> float:
        """Homeostatic error reduction."""
        hunger_before = self.state.internal_state.get("hunger", 0.0)
        thirst_before = self.state.internal_state.get("thirst", 0.0)
        
        energy_delta = outcome.get("energy_delta", 0.0)
        hunger_reduction = energy_delta / 10.0 if energy_delta > 0 else 0.0
        
        learning_signal = hunger_reduction
        return learning_signal
    
    def _get_decay_probabilities(self) -> np.ndarray:
        """Uniform decay for physiological memories - short-term focus."""
        if not self.memory:
            return np.array([])
        return np.ones(len(self.memory))
    
    def _compute_relevance(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Relevance based on similar homeostatic states."""
        if "perception_summary" not in memory:
            return 0.0
        
        mem_energy = memory["perception_summary"].get("energy", 0.0)
        ctx_energy = context.get("energy", 0.0)
        
        relevance = 1.0 - abs(mem_energy - ctx_energy) / 100.0
        
        if memory.get("affect", 0.0) > 0:
            relevance *= 1.2
        
        return max(0.0, relevance)

