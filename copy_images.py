import os
import shutil
import glob

src_dir = r"d:\Research_Brain\Research_Brain\Siddhesh\Software_Projects\ExplainableAI\MaxFormer\analysis_outputs\Images"
dst_dir = r"C:\Users\SiddheshPU\.gemini\antigravity\brain\babec454-6abd-45ef-8911-fc58c9778847\Images"

os.makedirs(dst_dir, exist_ok=True)

images = glob.glob(os.path.join(src_dir, "*.png"))
for img in images:
    dst_img = os.path.join(dst_dir, os.path.basename(img))
    shutil.copy2(img, dst_img)
    print(f"Copied {img} to {dst_img}")
