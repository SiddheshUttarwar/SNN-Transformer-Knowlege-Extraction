import torch
import timm
from spikegate import convert_to_gated_snn, AutoProfiler

def test_full_pipeline():
    print("1. Loading standard pretrained ViT from timm...")
    # Load a small vision transformer
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    
    # Dummy input
    x = torch.randn(2, 3, 224, 224)
    
    print("2. Wrapping model with SpikeGate framework...")
    # Wrap it to convert Attention to SpikingGatedAttention
    # Since we don't have a profile yet, it defaults to ACTIVE_NO_GATE everywhere
    spiking_model = convert_to_gated_snn(model, profile_path=None, T=4)
    
    print("3. Testing inference forward pass...")
    # It should automatically repeat the input T=4 times and average
    out = spiking_model(x)
    print(f"   Output shape: {out.shape}")
    assert out.shape == (2, 1000)
    
    print("4. Testing AutoProfiler hook logic...")
    # Setup the profiler
    profiler = AutoProfiler(spiking_model)
    
    # Create a dummy dataloader
    dummy_dataset = [(torch.randn(3, 224, 224), 0) for _ in range(5)]
    dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2)
    
    # Run calibration
    profiler.calibrate(dataloader, device='cpu', num_blocks=12, num_heads=3, out_file="test_gating_profile.json")
    
    print("5. Reloading model with newly generated gating policy...")
    gated_model = convert_to_gated_snn(model, profile_path="test_gating_profile.json", T=4)
    
    out_gated = gated_model(x)
    print(f"   Gated Output shape: {out_gated.shape}")
    
    print("6. Testing Gating Ablation Evaluator...")
    from spikegate import GatingAblationStudy
    # Recreate the gate controller reference to pass to the evaluator
    # Since convert_to_gated_snn wraps the model, we extract the controller from the first SpikingGatedAttention
    gate_ctrl = None
    for m in gated_model.modules():
        if hasattr(m, 'gate_controller'):
            gate_ctrl = m.gate_controller
            break
            
    evaluator = GatingAblationStudy(gated_model, gate_ctrl)
    evaluator.run_study(dataloader, device='cpu', out_file="test_gating_impact.json")
    
    print("\n7. Testing Head Importance Estimation Study...")
    # Run with small num_blocks/heads for quick testing
    evaluator.run_head_importance_study(dataloader, device='cpu', num_blocks=1, num_heads=3, out_file="test_head_importance.json")
    
    print("\nPipeline Test Completed Successfully!")

