import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

class Action(Enum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    STAY = 4
    FORAGE = 5
    DRINK = 6
    REST = 7
    SEEK_SHELTER = 8
    FLEE = 9
    GROUP_UP = 10
    SHARE_RESOURCE = 11
    SIGNAL = 12
    DEMONSTRATE_SKILL = 13
    EXPLORE = 14
    PRACTICE_CRAFT = 15
    PERFORM_RITUAL = 16

@dataclass
class ActionProposal:
    action: Action
    urgency: float
    expected_value: float
    band_id: int
    params: Dict[str, Any]

@dataclass
class BandState:
    urgency: float
    internal_state: Dict[str, Any]
    gain: float
    frustration_accumulator: float

class Band(ABC):
    def __init__(self, band_id: int, initial_gain: float = 1.0, seed: int = None):
        self.band_id = band_id
        self.state = BandState(
            urgency=0.0,
            internal_state={},
            gain=initial_gain,
            frustration_accumulator=0.0
        )
        self.rng = np.random.default_rng(seed)
        self.memory = []
        
    @abstractmethod
    def perceive(self, env_state: Dict[str, Any], agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw environment and agent state into band-specific perception."""
        pass
    
    @abstractmethod
    def compute_urgency(self, perception: Dict[str, Any]) -> float:
        """Compute urgency from homeostatic deficits, opportunities, and learned expectations."""
        pass
    
    @abstractmethod
    def propose_actions(self, perception: Dict[str, Any]) -> List[ActionProposal]:
        """Generate action proposals with urgency and expected values."""
        pass
    
    @abstractmethod
    def update_state(self, perception: Dict[str, Any], action_taken: Action, outcome: Dict[str, Any]):
        """Update internal state based on action outcome."""
        pass
    
    @abstractmethod
    def compute_learning_signal(self, perception: Dict[str, Any], action: Action, outcome: Dict[str, Any]) -> float:
        """Compute band-specific learning signal."""
        pass
    
    def write_memory(self, perception: Dict[str, Any], action: Action, outcome: Dict[str, Any], affect: float):
        """Write episodic memory with band-specific tags."""
        memory_entry = {
            "band_id": self.band_id,
            "tick": outcome.get("tick", 0),
            "perception_summary": self._compress_perception(perception),
            "action": action.name,
            "outcome_summary": self._compress_outcome(outcome),
            "affect": affect,
            "dominant_band": outcome.get("dominant_band", self.band_id)
        }
        self.memory.append(memory_entry)
        self._decay_memory()
    
    def _compress_perception(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Compress perception for memory storage."""
        return {k: v for k, v in perception.items() if isinstance(v, (int, float, str, bool))}
    
    def _compress_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Compress outcome for memory storage."""
        return {k: v for k, v in outcome.items() if isinstance(v, (int, float, str, bool))}
    
    def _decay_memory(self, max_memories: int = 1000):
        """Decay old memories with band-specific bias."""
        if len(self.memory) > max_memories:
            decay_prob = self._get_decay_probabilities()
            indices_to_keep = self.rng.choice(
                len(self.memory),
                size=max_memories,
                replace=False,
                p=decay_prob / decay_prob.sum()
            )
            self.memory = [self.memory[i] for i in sorted(indices_to_keep)]
    
    @abstractmethod
    def _get_decay_probabilities(self) -> np.ndarray:
        """Get band-specific memory decay probabilities."""
        pass
    
    def query_memory(self, query_context: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        """Query memory with band-specific priors."""
        if not self.memory:
            return []
        
        relevance_scores = np.array([
            self._compute_relevance(mem, query_context)
            for mem in self.memory
        ])
        
        if relevance_scores.sum() == 0:
            top_k_indices = self.rng.choice(len(self.memory), size=min(k, len(self.memory)), replace=False)
        else:
            probs = relevance_scores / relevance_scores.sum()
            top_k_indices = np.argsort(relevance_scores)[-k:]
        
        return [self.memory[i] for i in top_k_indices]
    
    @abstractmethod
    def _compute_relevance(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute relevance of a memory to current context."""
        pass
    
    def update_gain(self, frustration_threshold: float = 10.0, gain_increment: float = 0.1):
        """Adapt gain if band is chronically frustrated."""
        if self.state.frustration_accumulator > frustration_threshold:
            self.state.gain = min(self.state.gain + gain_increment, 5.0)
            self.state.frustration_accumulator = 0.0
        elif self.state.urgency < 0.1:
            self.state.gain = max(self.state.gain - gain_increment * 0.5, 0.1)

