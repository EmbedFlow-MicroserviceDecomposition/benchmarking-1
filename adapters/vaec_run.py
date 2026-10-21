"""Run VAE-C's autoencoder + KMeans pipeline (ported from tools/vaec/
machine-learning/ae.ipynb, unmodified algorithm) on the output of
vaec_adapter.py.

Pipeline (matches the notebook cell-for-cell):
  1. Build entrypoint-reachability matrix E and co-entrypoint-path matrix C
     via DFS from each entrypoint over the adjacency matrix A.
  2. Feature matrix X = D^-1/2 . A . D^-1/2 . [E | C]  (one GCN-style
     propagation step), standardized.
  3. Train a small autoencoder: reconstruction loss + "architecture loss"
     (Z.Z^T should approximate A) - same AE class as the notebook.
  4. Extract 2D embeddings Z = encoder(X).
  5. KMeans sweep k=2..20, pick k via the SSE-elbow (max distance from the
     line connecting the first and last (k, SSE) points) - VAE-C's own paper
     doesn't bake in one fixed k (see result-analyser/ folder names varying
     per app), so we automate its own stated model-selection heuristic
     rather than hand-pick a k per dataset.

Deviation from the notebook, documented: the co-entrypoint path DFS
(dfs_for_paths) enumerates ALL simple paths from each entrypoint, which is
exponential on graphs with branching/cycles - fine for VAE-C's own small
example apps but infeasible on our larger datasets (JForum 352 classes,
Roller 533 classes). We cap total explored paths per entrypoint
(--max-paths, default 20000) and print a WARNING when capped - matches
the tool's own algorithm exactly below the cap, degrades gracefully above it.

Usage:
    python3 adapters/vaec_run.py --deps app_dependency.json \
        --entrypoints app_entrypoints.json --out result.json
"""

import argparse
import json
import sys

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_graph(deps):
    not_keys = []
    for key, vals in deps.items():
        for cls in vals:
            if cls not in deps and cls not in not_keys:
                not_keys.append(cls)
    for key in not_keys:
        deps[key] = []

    node_to_class = {}
    class_to_node = {}
    for i, cls in enumerate(deps.keys()):
        node_to_class[i] = cls
        class_to_node[cls] = i

    n_nodes = len(deps)
    A = np.zeros((n_nodes, n_nodes))
    for key, lst in deps.items():
        for cls in lst:
            A[class_to_node[key], class_to_node[cls]] = 1
    return A, node_to_class, class_to_node


def dfs_for_E(node, visited, graph, E, ep_idx):
    if visited[node] == 1:
        return
    visited[node] = 1
    for i in range(graph.shape[1]):
        if graph[node, i] == 1:
            if E[i, ep_idx] == 0:
                E[i, ep_idx] = 1
            dfs_for_E(i, visited, graph, E, ep_idx)


def dfs_for_paths(node, graph, ep_idx, cur_path, paths, budget):
    if budget[0] <= 0:
        return
    if node in cur_path:
        paths[ep_idx].append(cur_path)
        budget[0] -= 1
        return
    tmp_path = cur_path + [node]
    if np.sum(graph[node]) == 0:
        paths[ep_idx].append(tmp_path)
        budget[0] -= 1
        return
    for i in range(graph.shape[1]):
        if budget[0] <= 0:
            return
        if graph[node, i] == 1:
            dfs_for_paths(i, graph, ep_idx, tmp_path, paths, budget)


def build_features(A, entrypoints, max_paths):
    n_nodes = A.shape[0]
    E = np.zeros((n_nodes, len(entrypoints)))
    for idx, ep in enumerate(entrypoints):
        visited = np.zeros(n_nodes)
        dfs_for_E(ep, visited, A, E, idx)

    paths = [[] for _ in range(len(entrypoints))]
    capped = False
    for idx, ep in enumerate(entrypoints):
        budget = [max_paths]
        dfs_for_paths(ep, A, idx, [], paths, budget)
        if budget[0] <= 0:
            capped = True
    if capped:
        print("WARNING: co-entrypoint path exploration capped at %d paths/entrypoint "
              "(graph too large/cyclic for exhaustive enumeration) - degraded but not "
              "incorrect for the paths that were explored" % max_paths, file=sys.stderr)

    C = np.zeros((n_nodes, n_nodes))
    for ep_idx in range(len(entrypoints)):
        for path in paths[ep_idx]:
            path_set = list(dict.fromkeys(path))
            for i in path_set:
                for j in path_set:
                    if i != j:
                        C[i, j] += 1
                    else:
                        C[i, j] += 1

    for i in range(n_nodes):
        A[i, i] = 1
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        D[i, i] = np.sum(A[i])
    D_half_norm = fractional_matrix_power(D, -0.5)
    X = np.concatenate((E, C), axis=1)
    X = D_half_norm.dot(A).dot(D_half_norm).dot(X)
    return np.asarray(X, dtype=np.float64)


class AE(keras.Model):
    def __init__(self, encoder, decoder, A, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.A = A
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.architecture_loss_tracker = keras.metrics.Mean(name="architecture_loss")

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.architecture_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=(1,)))
            architecture_loss = tf.reduce_mean(tf.square(self.A - tf.matmul(z, z, transpose_b=True)))
            total_loss = reconstruction_loss + architecture_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.architecture_loss_tracker.update_state(architecture_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "architecture_loss": self.architecture_loss_tracker.result(),
        }


def train_ae(X, A, n_embeddings=2, n_epochs=8000, lr=0.001):
    n_features = X.shape[1]
    input_encoder = Input(shape=(n_features,))
    n_neurons = n_features // 2
    x = Dense(n_neurons, activation="tanh")(input_encoder)
    n_neurons = n_neurons // 2
    while n_neurons >= n_embeddings * 2:
        x = Dense(n_neurons, activation="tanh")(x)
        n_neurons = n_neurons // 2
    z = Dense(n_embeddings, name="z")(x)

    input_decoder = Input(shape=(n_embeddings,))
    n_neurons = n_embeddings * 2
    x = Dense(n_neurons, activation="tanh")(input_decoder)
    n_neurons = n_neurons * 2
    while n_neurons <= n_features // 2:
        x = Dense(n_neurons, activation="tanh")(x)
        n_neurons = n_neurons * 2
    recon = Dense(n_features)(x)

    encoder = Model(input_encoder, z, name="encoder")
    decoder = Model(input_decoder, recon, name="decoder")
    ae = AE(encoder, decoder, tf.constant(A, dtype=tf.float32))
    ae.compile(optimizer=Adam(lr))
    ae.built = True
    ae.fit(X, epochs=n_epochs, batch_size=X.shape[0], verbose=0)
    return ae.encoder(X).numpy()


def elbow_k(sse_list, ks):
    x1, y1 = ks[0], sse_list[0]
    x2, y2 = ks[-1], sse_list[-1]
    best_k, best_dist = ks[0], -1
    for k, sse in zip(ks, sse_list):
        num = abs((y2 - y1) * k - (x2 - x1) * sse + x2 * y1 - y2 * x1)
        den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        dist = num / den if den else 0
        if dist > best_dist:
            best_dist = dist
            best_k = k
    return best_k


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deps", required=True)
    p.add_argument("--entrypoints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-paths", type=int, default=20000)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    with open(args.deps) as f:
        deps = json.load(f)
    with open(args.entrypoints) as f:
        entrypoint_names = json.load(f)

    A, node_to_class, class_to_node = build_graph(deps)
    entrypoints = [class_to_node[e] for e in entrypoint_names if e in class_to_node]
    if not entrypoints:
        raise SystemExit("no entrypoints resolved against dependency graph keys")

    X = build_features(A.copy(), entrypoints, args.max_paths)
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)

    Z = train_ae(X, A)

    n_nodes = A.shape[0]
    max_k = min(args.max_k, n_nodes - 1)
    ks = list(range(2, max_k + 1))
    sse_list = []
    labels_by_k = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z)
        sse = sum(np.sum((Z[i] - km.cluster_centers_[km.labels_[i]]) ** 2) for i in range(len(Z)))
        sse_list.append(sse)
        labels_by_k[k] = km.labels_

    best_k = elbow_k(sse_list, ks)
    labels = labels_by_k[best_k]

    clusters = {}
    for node_id, cluster_id in enumerate(labels):
        clusters.setdefault(str(cluster_id), []).append(node_to_class[node_id])

    with open(args.out, "w") as f:
        json.dump(clusters, f, indent=1)

    print("classes=%d chosen_k=%d clusters=%d" % (n_nodes, best_k, len(clusters)))


if __name__ == "__main__":
    main()
