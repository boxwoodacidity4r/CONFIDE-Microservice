import os
import json
import numpy as np
import networkx as nx
import community  # python-louvain
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

# SYSTEMS = ['acmeair', 'daytrader', 'jpetstore', 'plants']
SYSTEMS = ['acmeair']
ALPHAS = [0.3, 0.5, 0.7]
TAU = 0.3

DATA_DIR = os.path.join('data', 'processed', 'fusion')

for system in SYSTEMS:
    S_SEM_PATH = os.path.join(DATA_DIR, f"{system}_S_sem.npy")
    S_STRUCT_PATH = os.path.join(DATA_DIR, f"{system}_S_struct.npy")
    CLASS_NAMES_PATH = os.path.join(DATA_DIR, f"{system}_class_order.json")
    OUT_DIR = os.path.join(DATA_DIR, f"phase1_{system}")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    name2idx = {n: i for i, n in enumerate(class_names)}
    S_sem = np.load(S_SEM_PATH)
    S_struct = np.load(S_STRUCT_PATH)
    N = len(class_names)

    # Sanity check: ordering consistency and normalization
    print(f"[{system}] Sample: {class_names[0]}, S_sem[0][:5]={S_sem[0][:5]}, S_struct[0][:5]={S_struct[0][:5]}")
    assert S_sem.shape == (N, N) and S_struct.shape == (N, N)
    assert S_sem.min() >= 0 and S_sem.max() <= 1, "S_sem not normalized"
    assert S_struct.min() >= 0 and S_struct.max() <= 1, "S_struct not normalized"

    for alpha in ALPHAS:
        S = alpha * S_sem + (1 - alpha) * S_struct
        np.fill_diagonal(S, 1.0)

        # Sparsified weighted graph
        G = nx.Graph()
        for i in range(N):
            G.add_node(i, label=class_names[i])
        edge_count = 0
        for i in range(N):
            for j in range(i+1, N):
                if S[i, j] >= TAU:
                    G.add_edge(i, j, weight=float(S[i, j]))
                    edge_count += 1

        # Isolated node check
        isolated = list(nx.isolates(G))
        print(f"[{system} alpha={alpha}] Edges with S >= {TAU}: {edge_count}, Isolated nodes: {len(isolated)}")
        if len(isolated) > 0:
            print(f"Warning: {len(isolated)} isolated nodes, consider lowering tau.")

        # Louvain community detection
        partition = community.best_partition(G, weight='weight', random_state=42)
        clusters = {}
        for idx, cid in partition.items():
            clusters.setdefault(f"cluster_{cid}", []).append(class_names[idx])
        # Save cluster->classes
        cluster_path = os.path.join(OUT_DIR, f"clusters_phase1_alpha{alpha}.json")
        with open(cluster_path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2, ensure_ascii=False)
        # Save class->cluster id
        class2cid = {class_names[idx]: cid for idx, cid in partition.items()}
        with open(os.path.join(OUT_DIR, f"partition_phase1_alpha{alpha}.json"), "w", encoding="utf-8") as f:
            json.dump(class2cid, f, indent=2, ensure_ascii=False)

        # Metrics
        cluster_sizes = [len(v) for v in clusters.values()]
        intra_sims = []
        inter_sims = []
        cluster_list = list(clusters.values())
        for nodes in cluster_list:
            idxs = [name2idx[n] for n in nodes]
            if len(idxs) > 1:
                intra = S[np.ix_(idxs, idxs)]
                intra_sims.append(np.mean(intra[np.triu_indices_from(intra, k=1)]))
        # inter-cluster similarity uses upper triangle only
        for i in range(len(cluster_list)):
            for j in range(i+1, len(cluster_list)):
                idxs1 = [name2idx[n] for n in cluster_list[i]]
                idxs2 = [name2idx[n] for n in cluster_list[j]]
                inter = S[np.ix_(idxs1, idxs2)]
                inter_sims.append(np.mean(inter))
        # modularity
        mod = community.modularity(partition, G, weight="weight")
        # edge density
        density = 2 * edge_count / (N * (N - 1)) if N > 1 else 0.0
        stats = {
            "system": system,
            "alpha": alpha,
            "num_clusters": len(clusters),
            "cluster_size_distribution": cluster_sizes,
            "mean_intra_cluster_similarity": float(np.mean(intra_sims)) if intra_sims else 0.0,
            "mean_inter_cluster_similarity": float(np.mean(inter_sims)) if inter_sims else 0.0,
            "modularity": float(mod),
            "edge_density": float(density),
            "num_isolated_nodes": len(isolated),
        }
        stats_path = os.path.join(OUT_DIR, f"phase1_stats_alpha{alpha}.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        # Visualization
        try:
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(G, seed=42)
            colors = [partition[n] for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab20, node_size=60)
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            plt.title(f"{system} Phase1 Clustering (alpha={alpha})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"phase1_graph_alpha{alpha}.png"))
            plt.close()
        except Exception as e:
            print(f"Visualization failed for {system} alpha={alpha}: {e}")

    # Write a short markdown description
    readme = f'''
# Phase 1: Coarse Initialization Clustering ({system})

**Goal:** Obtain a coarse but usable initial clustering for microservice extraction by fusing structural and semantic similarity.

**Fusion:**  
S = α·S_sem + (1−α)·S_struct  
Default α = 0.5, ablation α ∈ {{0.3, 0.5, 0.7}}

**Sparsification & Community Detection:**  
- Only keep S[i,j] ≥ {TAU} to denoise and avoid trivial fully-connected graphs.
- Louvain community detection is used for clustering.

**Note:**  
We intentionally allow imperfect clusters in Phase 1 to expose conflicts across modalities.

**Outputs:**  
- clusters_phase1_alpha*.json: Cluster assignments
- partition_phase1_alpha*.json: Class-to-cluster mapping
- phase1_stats_alpha*.json: Clustering statistics
- phase1_graph_alpha*.png: Visualization (optional)
'''
    with open(os.path.join(OUT_DIR, "phase1_readme.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"Phase 1 clustering complete for {system}. See outputs in {OUT_DIR}")
