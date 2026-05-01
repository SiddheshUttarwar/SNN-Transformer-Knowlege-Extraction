from spikegate.policies import GatingPolicy

def test_gating_policies_exist():
    assert hasattr(GatingPolicy, "ACTIVE_NO_GATE")
    assert hasattr(GatingPolicy, "STATICALLY_PRUNE_OR_EARLY_EXIT_T1")
    assert hasattr(GatingPolicy, "STATICALLY_GATED_BY_REDUNDANCY")
    assert hasattr(GatingPolicy, "LATE_WAKEUP_GATE")
    assert hasattr(GatingPolicy, "DYNAMIC_KEY_EXIT_WAIT_T2")
    assert hasattr(GatingPolicy, "DYNAMIC_ONLINE_PRUNING")

def test_policy_values():
    assert GatingPolicy.ACTIVE_NO_GATE.value == "ACTIVE_NO_GATE"
    assert GatingPolicy.LATE_WAKEUP_GATE.value == "LATE_WAKEUP_GATE"
