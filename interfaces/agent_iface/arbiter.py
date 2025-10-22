import numpy as np
from typing import List, Optional
from .band import Band, Action, ActionProposal

class Arbiter:
    """
    Global arbiter that blends band action proposals using soft priority with hysteresis.
    Prevents thrashing between bands when stimuli fluctuate.
    """
    
    def __init__(self, inertia: float = 0.3, temperature: float = 2.0, seed: int = None):
        self.inertia = inertia
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.previous_action: Optional[Action] = None
        self.previous_band: Optional[int] = None
        self.dominant_band_history = []
    
    def select_action(self, bands: List[Band], all_proposals: List[List[ActionProposal]], 
                      agent_state: dict) -> tuple[Action, int, ActionProposal]:
        """
        Select action from band proposals using softmax with hysteresis.
        
        Returns: (selected_action, dominant_band_id, selected_proposal)
        """
        if not all_proposals or all(not proposals for proposals in all_proposals):
            return Action.STAY, 0, None
        
        flat_proposals = [p for proposals in all_proposals for p in proposals if p is not None]
        
        if not flat_proposals:
            return Action.STAY, 0, None
        
        safety_veto = self._check_safety_veto(flat_proposals)
        if safety_veto is not None:
            self.previous_action = safety_veto[0]
            self.previous_band = safety_veto[1]
            self.dominant_band_history.append(safety_veto[1])
            return safety_veto
        
        energy_constraint = self._check_energy_budget(flat_proposals, agent_state)
        if energy_constraint is not None:
            self.previous_action = energy_constraint[0]
            self.previous_band = energy_constraint[1]
            self.dominant_band_history.append(energy_constraint[1])
            return energy_constraint
        
        urgencies = np.array([p.urgency for p in flat_proposals])
        
        for i, p in enumerate(flat_proposals):
            if self.previous_band is not None and p.band_id == self.previous_band:
                urgencies[i] *= (1.0 + self.inertia)
        
        if urgencies.max() == 0:
            probs = np.ones(len(urgencies)) / len(urgencies)
        else:
            probs = self._softmax(urgencies / self.temperature)
        
        selected_idx = self.rng.choice(len(flat_proposals), p=probs)
        selected = flat_proposals[selected_idx]
        
        self.previous_action = selected.action
        self.previous_band = selected.band_id
        self.dominant_band_history.append(selected.band_id)
        
        return selected.action, selected.band_id, selected
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _check_safety_veto(self, proposals: List[ActionProposal]) -> Optional[tuple]:
        """Safety band (Band 2) can veto when threat is immediate."""
        for p in proposals:
            if p.band_id == 2 and p.urgency > 8.0:
                return (p.action, p.band_id, p)
        return None
    
    def _check_energy_budget(self, proposals: List[ActionProposal], agent_state: dict) -> Optional[tuple]:
        """Force physiological action if energy critically low."""
        energy = agent_state.get("energy", 0.0)
        
        if energy < 10.0:
            for p in proposals:
                if p.band_id == 1 and p.params.get("reason") == "critical_hunger":
                    return (p.action, p.band_id, p)
        
        return None
    
    def get_dominant_band_distribution(self) -> dict:
        """Get distribution of which bands have been dominant."""
        if not self.dominant_band_history:
            return {}
        
        unique, counts = np.unique(self.dominant_band_history, return_counts=True)
        total = len(self.dominant_band_history)
        
        return {int(band_id): count / total for band_id, count in zip(unique, counts)}
    
    def reset_history(self):
        """Reset arbitration history."""
        self.previous_action = None
        self.previous_band = None
        self.dominant_band_history = []

