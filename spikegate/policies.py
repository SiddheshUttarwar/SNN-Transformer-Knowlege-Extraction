"""
Gating policy definitions for the SpikeGate framework.

All hardware gating policies are defined here as an Enum to prevent
silent failures from typos in raw strings.
"""

from enum import Enum


class GatingPolicy(str, Enum):
    """Hardware gating policies that control K/V compute bypass decisions.
    
    Inherits from ``str`` so policies serialize to/from JSON naturally
    (e.g., ``json.dumps(GatingPolicy.ACTIVE_NO_GATE)`` → ``"ACTIVE_NO_GATE"``).
    """
    ACTIVE_NO_GATE = "ACTIVE_NO_GATE"
    STATICALLY_PRUNE_OR_EARLY_EXIT_T1 = "STATICALLY_PRUNE_OR_EARLY_EXIT_T1"
    STATICALLY_GATED_BY_REDUNDANCY = "STATICALLY_GATED_BY_REDUNDANCY"
    LATE_WAKEUP_GATE = "LATE_WAKEUP_GATE"
    DYNAMIC_KEY_EXIT_WAIT_T2 = "DYNAMIC_KEY_EXIT_WAIT_T2"
    DYNAMIC_ONLINE_PRUNING = "DYNAMIC_ONLINE_PRUNING"

    @classmethod
    def from_string(cls, value: str) -> "GatingPolicy":
        """Safely converts a string to a GatingPolicy, raising ValueError on typos."""
        try:
            return cls(value)
        except ValueError:
            valid = [p.value for p in cls]
            raise ValueError(
                f"Unknown gating policy '{value}'. Valid policies: {valid}"
            )
