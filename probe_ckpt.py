import torch, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ck = torch.load('./checkpoints/10-384-T4.pth.tar', map_location='cpu')
sd = ck.get('state_dict', ck)

# Find stage block counts
stages = set()
for k in sd:
    k2 = k.replace('module.', '')
    if k2.startswith('stage'):
        parts = k2.split('.')
        stages.add((parts[0], int(parts[1])))

from collections import defaultdict
stage_counts = defaultdict(int)
for s, idx in stages:
    stage_counts[s] = max(stage_counts[s], idx + 1)

print("Stage block counts:", dict(stage_counts))
print("Head shape:", sd.get('head.weight', sd.get('module.head.weight', 'N/A')).shape if hasattr(sd.get('head.weight', sd.get('module.head.weight', None)), 'shape') else 'N/A')
print("\nAll top-level keys (first part):")
top = set(k.replace('module.','').split('.')[0] for k in sd)
for t in sorted(top):
    print(' ', t)

print("\nArchitecture params from ckpt dict (non state_dict keys):")
for k, v in ck.items():
    if k != 'state_dict':
        print(f"  {k}: {v}")
