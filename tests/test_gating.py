import os
import json
import tempfile
from spikegate.gating import DynamicGateController
from spikegate.policies import GatingPolicy

def test_gate_controller_default():
    ctrl = DynamicGateController(profile_path=None)
    policy = ctrl.get_head_policy(block_idx=0, head_idx=0)
    assert policy == GatingPolicy.ACTIVE_NO_GATE.value

def test_gate_controller_with_profile():
    profile = {
        "block_0": {
            "head_0": {"HARDWARE_GATING_POLICY": "LATE_WAKEUP_GATE"}
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(profile, f)
        path = f.name
    
    try:
        ctrl = DynamicGateController(profile_path=path)
        policy = ctrl.get_head_policy(block_idx=0, head_idx=0)
        assert policy == "LATE_WAKEUP_GATE"
        
        # Fallback for undefined heads
        policy2 = ctrl.get_head_policy(block_idx=0, head_idx=1)
        assert policy2 == GatingPolicy.ACTIVE_NO_GATE.value
    finally:
        os.remove(path)
