"""
Entry-point script that uses the SpikeGate framework to run a comprehensive
MaxFormer analysis (10 runs x 500 images by default).

Usage:
    python generate_comprehensive_report.py --runs 10 --images 500
    python generate_comprehensive_report.py --fast   # quick test (2x50)
"""

import sys
import os
import argparse
import warnings

import torch
from torchvision import transforms
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from qkv_sparsity_extractor import load_maxformer, IMG_SIZE, BATCH_SIZE, NUM_HEADS
from spikegate import AutoProfiler, DynamicGateController, ReportGenerator

NUM_BLOCKS = 10
T_STEPS = 4
REPORT_DIR = "analysis_outputs/comprehensive_report"


def create_data_chunks(num_runs: int, images_per_run: int, batch_size: int):
    """Streams ImageNet-1K and yields lists of batches (one list per run)."""
    dataset = load_dataset(
        "mrm8488/ImageNet1K-val", split="train",
        streaming=True, trust_remote_code=True,
    )
    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    batches_per_run = images_per_run // batch_size
    buf, run_batches = [], []

    for item in dataset:
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf.append(tfs(img).unsqueeze(0))

        if len(buf) == batch_size:
            run_batches.append((torch.cat(buf, dim=0), torch.zeros(batch_size)))
            buf = []
            if len(run_batches) == batches_per_run:
                yield run_batches
                run_batches = []
                num_runs -= 1
                if num_runs <= 0:
                    return


def main():
    parser = argparse.ArgumentParser(description="SpikeGate Comprehensive Report")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--fast", action="store_true", help="Quick test (2x50)")
    args = parser.parse_args()

    if args.fast:
        args.runs, args.images = 2, 50

    # 1. Load Model
    print("=" * 60)
    print("  SpikeGate — Comprehensive MaxFormer Analysis")
    print("=" * 60)
    model, _, device = load_maxformer()

    # 2. Stream Data
    print(f"\nStreaming {args.runs} x {args.images} = "
          f"{args.runs * args.images} ImageNet images...")
    chunks = list(create_data_chunks(args.runs, args.images, BATCH_SIZE))
    print(f"Loaded {len(chunks)} data chunks ({len(chunks[0])} batches each)\n")

    # 3. AutoProfiler Calibration (first chunk)
    print("=" * 60)
    print("  Phase 1: AutoProfiler Calibration")
    print("=" * 60)
    os.makedirs(REPORT_DIR, exist_ok=True)
    profiler = AutoProfiler(model, T=T_STEPS)
    profile = profiler.calibrate(
        chunks[0], device=device,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
        out_file=os.path.join(REPORT_DIR, "gating_profile.json"),
    )

    # 4. Generate Comprehensive Report (via framework)
    gate_ctrl = DynamicGateController(
        os.path.join(REPORT_DIR, "gating_profile.json")
    )
    reporter = ReportGenerator(
        model, gate_ctrl, profile,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
        output_dir=REPORT_DIR,
    )
    report_path = reporter.run_full_pipeline(
        chunks, device=device, images_per_run=args.images,
    )

    print(f"\nReport:  {report_path}")
    print(f"Profile: {REPORT_DIR}/gating_profile.json")
    print(f"Figures: {REPORT_DIR}/figures/")


if __name__ == "__main__":
    main()
