import torch
from spikegate.neurons import LIFNode

def test_lif_node_initialization():
    node = LIFNode(tau=2.0)
    assert node.decay == 0.5
    assert node.v_th == 0.5

def test_lif_node_forward():
    node = LIFNode(tau=2.0)
    # Provide input that crosses threshold
    x = torch.ones(1, 10) * 1.5
    spikes = node(x)
    assert spikes.shape == x.shape
    assert (spikes == 1.0).all()

def test_lif_node_reset():
    node = LIFNode(tau=2.0)
    x = torch.ones(1, 10) * 1.5
    node(x)
    assert node.u is not None
    node.reset_state()
    assert node.u is None

