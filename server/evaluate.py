import os
import torch
from torch.utils.data import DataLoader
from hospital.dataset import NPZDataset
from hospital.model import CNN

BACKBONE_PATH = "artifacts/encoder.pt"

def evaluate_global(global_protos, device="cpu"):
    if not os.path.exists(BACKBONE_PATH):
        print("⚠️ Encoder not found — skipping global evaluation")
        return None

    model = CNN(num_classes=1)  # classifier unused
    model.encoder.load_state_dict(
        torch.load(BACKBONE_PATH, map_location=device)
    )
    model.eval()


    correct, total = 0, 0

    for h in ["hospital_1", "hospital_2", "hospital_3"]:
        ds = NPZDataset(f"fedpc_bloodmnist_npz/{h}/test.npz")
        dl = DataLoader(ds, batch_size=128)

        with torch.no_grad():
            for x, _, yg in dl:
                _, feats = model(x, return_feat=True)
                for f, y in zip(feats, yg):
                    sims = {
                        cid: torch.cosine_similarity(f, proto, dim=0)
                        for cid, proto in global_protos.items()
                    }
                    pred = max(sims, key=sims.get)
                    if pred == y.item():
                        correct += 1
                    total += 1

    return 100.0 * correct / total
