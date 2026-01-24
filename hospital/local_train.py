import torch
import torch.nn.functional as F

def train(model, loader, opt, global_protos=None, lam=0.0):
    model.train()
    correct, total = 0, 0

    for x, yl, yg in loader:
        opt.zero_grad()
        out, feat = model(x, return_feat=True)

        yl = torch.tensor(yl)
        loss = F.cross_entropy(out, yl)

        if global_protos:
            for f, y in zip(feat, yg):
                if y in global_protos:
                    loss += lam * torch.norm(f - global_protos[y])

        loss.backward()
        opt.step()

        preds = out.argmax(1)
        correct += (preds == yl).sum().item()
        total += yl.size(0)

    acc = 100.0 * correct / total
    return acc
