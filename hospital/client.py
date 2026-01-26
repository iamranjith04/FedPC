import socket
import pickle
import sys
import os
import torch
from torch.utils.data import DataLoader

from hospital.config import *
from hospital.model import CNN
from hospital.dataset import NPZDataset
from hospital.local_train import train
from hospital.prototype import build_prototypes


hospital = sys.argv[1]
print(f"🏥 {hospital} started")

ds = NPZDataset(f"fedpc_bloodmnist_npz/{hospital}/train.npz")
dl = DataLoader(ds, BATCH_SIZE, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN(len(ds.classes), device=device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

sock = socket.socket()
sock.connect((SERVER_HOST, SERVER_PORT))

global_protos = None
for r in range(ROUNDS):
    acc = train(model, dl, opt, global_protos, LAMBDA_PROTO)
    print(f"[{hospital}] ROUND {r} | Local Train Acc: {acc:.2f}%")

    # 🔑 Save backbone ONLY from hospital_1 (reference)
    if hospital == "hospital_1" and r == ROUNDS - 1:
        os.makedirs("artifacts", exist_ok=True)
        tmp_path = "artifacts/encoder.tmp"
        final_path = "artifacts/encoder.pt"

        torch.save(model.encoder.state_dict(), tmp_path)
        os.replace(tmp_path, final_path)

        print("💾 Saved reference ENCODER (hospital_1)")


    protos = build_prototypes(model, dl)
    sock.sendall(pickle.dumps(protos))
    print(f"[{hospital}] Sent prototypes")

    global_protos = pickle.loads(sock.recv(10_000_000))
    print(f"[{hospital}] Received global prototypes")

print(f"✅ {hospital} finished")
sock.close()
