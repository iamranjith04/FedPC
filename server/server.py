import socket
import pickle
from server.cluster import cluster_prototypes
from server.config import *
import os
import torch
from server.evaluate import evaluate_global

os.makedirs("global_models", exist_ok=True)

print("🚀 FedPC Server Started")

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

        classes = sorted(data.keys())
        print(f"📥 Received prototypes from {hospital_id}: classes {classes}")

    print("🔄 Clustering hospitals and aggregating prototypes...")
    global_protos, hospital_clusters, cluster_protos = cluster_prototypes(
        all_prototypes, N_CLUSTERS
    )

    # print("📊 Hospital → Cluster mapping:")
    # for hospital, cluster_id in hospital_clusters.items():
    #     print(f"   {hospital} → cluster {cluster_id}")
    
    print(f"📦 Global prototypes computed for {len(global_protos)} classes")

    for conn in connections:
        conn.sendall(pickle.dumps(global_protos))
    
    print("✅ Sent global prototypes to all hospitals")

    acc = evaluate_global(global_protos)
    if acc is not None and rnd==ROUNDS-1:
        print(f"🎯 Global Prototype Accuracy (Round {rnd}): {acc:.2f}%")

torch.save(
    {
        'global_prototypes': global_protos,
        'hospital_clusters': hospital_clusters,
        'cluster_prototypes': cluster_protos
    },
    f"global_models/final_model_round_{ROUNDS}.pt"
)

print("💾 Final model saved")
print("✅ Server finished")
sock.close()