import torch
from datasets import load_dataset
from torchvision import transforms

# Spikegate imports
from spikegate.profiler import AutoProfiler

# From the user's qkv_sparsity_extractor
from qkv_sparsity_extractor import load_maxformer, IMG_SIZE, BATCH_SIZE

def main():
    print("--- Loading MaxFormer for SpikeGate Analysis ---")
    model, _, device = load_maxformer()
    
    # 1. Setup Data Loader (using a smaller set for rapid analysis)
    print("\n--- Initializing ImageNet-Subset streaming ---")
    dataset = load_dataset('mrm8488/ImageNet1K-val', split='train', streaming=True, trust_remote_code=True)
    
    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def data_generator():
        b = []
        for item in dataset:
            img = item['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            b.append(tfs(img).unsqueeze(0))
            if len(b) == BATCH_SIZE:
                yield torch.cat(b, dim=0), torch.zeros(BATCH_SIZE) # Dummy labels
                b = []
                
    # Create a small static dataloader for the profiler (e.g., 5 batches = 50 images)
    print("Pre-fetching 5 batches of ImageNet data for Profiler Calibration...")
    gen = data_generator()
    calibration_data = [next(gen) for _ in range(5)]
    
    print("\n--- Running AutoProfiler on MaxFormer ---")
    # T=4 as defined in qkv_sparsity_extractor
    profiler = AutoProfiler(model, T=4)
    
    profiler.calibrate(calibration_data, device=device, num_blocks=10, num_heads=6, out_file="maxformer_gating_profile.json")
    
    print("\nAnalysis complete! Check maxformer_gating_profile.json for Head Importance, Sparsity, and Gating rules.")

if __name__ == "__main__":
    main()
