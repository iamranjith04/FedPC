import socket, pickle
from server.cluster import cluster_prototypes
from server.config import *
import os
import torch

from server.evaluate import evaluate_global
os.makedirs("global_models", exist_ok=True)


print("🚀 FedPC Server Started")

sock = socket.socket()
sock.bind((HOST, PORT))
sock.listen(NUM_HOSPITALS)

print("Waiting for hospitals...")

connections = []
for i in range(NUM_HOSPITALS):
    conn, _ = sock.accept()
    connections.append(conn)
    print(f"Hospital connected ({i+1}/{NUM_HOSPITALS})")


for rnd in range(ROUNDS):
    print(f"\n🌐 ROUND {rnd}")

    all_prototypes = {}

    for i, conn in enumerate(connections):
        data = pickle.loads(conn.recv(10_000_000))
        hospital_id = f"hospital_{i+1}"
        all_prototypes[hospital_id] = data

        classes = list(data.keys())
        print(f"📥 Received prototypes from {hospital_id}: {classes}")

    print("Clustering prototypes...")
    global_protos, assignments = cluster_prototypes(all_prototypes, N_CLUSTERS)

    print("📊 Prototype → Cluster mapping:")
    for h, cls, cid in assignments:
        print(f"   {h} | class {cls} → cluster {cid}")


    for conn in connections:
        conn.sendall(pickle.dumps(global_protos))
    acc = evaluate_global(global_protos)
    if acc is not None:
        print(f"🌍 Global Prototype Accuracy (Round {rnd}): {acc:.2f}%")


    print("Sending global prototypes...")

torch.save(
    global_protos,
    f"global_models/global_prototypes_round_{ROUNDS}.pt"
)

print("💾 Final global prototypes saved")
print("✅ Server finished")
sock.close()
