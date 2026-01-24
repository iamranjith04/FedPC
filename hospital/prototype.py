import torch
from collections import defaultdict

def build_prototypes(model, loader):
    buckets = defaultdict(list)
    with torch.no_grad():
        for x, _, yg in loader:
            _, feats = model(x, return_feat=True)
            for f,y in zip(feats,yg):
                buckets[int(y)].append(f)
    return {k: torch.stack(v).mean(0) for k,v in buckets.items()}
