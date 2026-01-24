import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def cluster_prototypes(all_prototypes, n_clusters):
    vectors, meta = [], []

    for hospital, protos in all_prototypes.items():
        for cls, vec in protos.items():
            vectors.append(vec.numpy())
            meta.append((hospital, cls))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    cluster_ids = kmeans.fit_predict(vectors)

    clusters = defaultdict(list)
    assignments = []

    for cid, vec, (hospital, cls) in zip(cluster_ids, vectors, meta):
        clusters[cid].append(torch.tensor(vec))
        assignments.append((hospital, cls, cid))

    global_protos = {
        cid: torch.stack(v).mean(0)
        for cid, v in clusters.items()
    }

    return global_protos, assignments

