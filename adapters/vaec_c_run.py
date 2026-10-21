"""Run the REAL VAE-C pipeline (variational autoencoder + fuzzy c-means,
ported from tools/vaec/machine-learning/code.ipynb, unmodified algorithm) on
the output of vaec_adapter.py.

Correction: adapters/vaec_run.py (our earlier "VAE-C" script) actually ported
tools/vaec/machine-learning/ae.ipynb, whose AE class is a plain deterministic
autoencoder (no KL loss, no reparameterization) clustered with hard KMeans -
that is the paper's AE-K BASELINE, not VAE-C. The real VAE-C is defined in
code.ipynb: a variational encoder (z_mean, z_log_var, reparameterization
Sampling layer, KL-divergence loss term) whose embeddings are clustered with
fuzzy c-means (skfuzzy.cluster.cmeans), then hardened via argmax over the
membership matrix. See Rokin Maharjan's M.S. thesis (Baylor, "Microservice
Decomposition and Message Queues"), Ch.2, which defines all three models:
AE-K (baseline), AE-C (AE + fuzzy c-means), VAE-C (this script, their
approach) - Table 2.3 reports VAE-C results as "the average of five attempts
since VAE's latent space is based on the normal distribution."

Graph/feature construction (build_graph, build_features) is unchanged from
adapters/vaec_run.py - reused directly (same DFS/GCN-propagation code, same
--max-paths cap deviation, documented there).

Usage:
    python3 adapters/vaec_c_run.py --deps app_dependency.json \
        --entrypoints app_entrypoints.json --out result.json [--k 5]
"""

import argparse
import json

import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from adapters.vaec_run import build_graph, build_features, elbow_k


class Sampling(layers.Layer):
    """Reparameterization trick: samples z from (z_mean, z_log_var)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, A, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.A = A
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.architecture_loss_tracker = keras.metrics.Mean(name="architecture_loss")

    def call(self, inputs):
        mean, log_var, x = self.encoder(inputs)
        return self.decoder(x)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker,
                self.kl_loss_tracker, self.architecture_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=(1,)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            architecture_loss = tf.reduce_mean(tf.square(self.A - tf.matmul(z, z, transpose_b=True)))
            total_loss = reconstruction_loss + kl_loss + architecture_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.architecture_loss_tracker.update_state(architecture_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "architecture_loss": self.architecture_loss_tracker.result(),
        }


def train_vae(X, A, n_embeddings=2, n_epochs=8000, lr=0.001):
    n_features = X.shape[1]
    input_encoder = Input(shape=(n_features,))
    n_neurons = n_features // 2
    x = Dense(n_neurons, activation="tanh")(input_encoder)
    n_neurons = n_neurons // 2
    while n_neurons >= n_embeddings * 2:
        x = Dense(n_neurons, activation="tanh")(x)
        n_neurons = n_neurons // 2
    z_mean = Dense(n_embeddings, name="z_mean")(x)
    z_log_var = Dense(n_embeddings, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    input_decoder = Input(shape=(n_embeddings,))
    n_neurons = n_embeddings * 2
    x = Dense(n_neurons, activation="tanh")(input_decoder)
    n_neurons = n_neurons * 2
    while n_neurons <= n_features // 2:
        x = Dense(n_neurons, activation="tanh")(x)
        n_neurons = n_neurons * 2
    recon = Dense(n_features)(x)

    encoder = Model(input_encoder, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(input_decoder, recon, name="decoder")
    vae = VAE(encoder, decoder, tf.constant(A, dtype=tf.float32))
    vae.compile(optimizer=Adam(lr))
    vae.built = True
    vae.fit(X, epochs=n_epochs, batch_size=X.shape[0], verbose=0)
    _, _, z_final = vae.encoder(X)
    return z_final.numpy()


def fuzzy_sse(Z, cntr, u):
    sse = 0.0
    for i in range(len(Z)):
        z_loc = np.asarray(Z[i])
        for j in range(cntr.shape[0]):
            c_loc = np.asarray(cntr[j])
            sse += u[j][i] * np.sum(np.square(z_loc - c_loc))
    return sse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deps", required=True)
    p.add_argument("--entrypoints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-paths", type=int, default=20000)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--k", type=int, default=None,
                   help="fix cluster count instead of SSE-elbow selection "
                        "(paper uses a fixed k=5 for Bearboard/Autocare/Pharmacy)")
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

    Z = train_vae(X, A)

    n_nodes = A.shape[0]
    max_k = min(args.max_k, n_nodes - 1)

    if args.k is not None:
        best_k = args.k
        cntr, u, *_ = fuzz.cluster.cmeans(Z.T, best_k, 2, error=0.005, maxiter=1000, init=None)
    else:
        ks = list(range(2, max_k + 1))
        sse_list = []
        u_by_k = {}
        for k in ks:
            cntr, u, *_ = fuzz.cluster.cmeans(Z.T, k, 2, error=0.005, maxiter=1000, init=None)
            sse_list.append(fuzzy_sse(Z, cntr, u))
            u_by_k[k] = u
        best_k = elbow_k(sse_list, ks)
        u = u_by_k[best_k]

    labels = np.argmax(u.T, axis=1)

    clusters = {}
    for node_id, cluster_id in enumerate(labels):
        clusters.setdefault(str(cluster_id), []).append(node_to_class[node_id])

    with open(args.out, "w") as f:
        json.dump(clusters, f, indent=1)

    print("classes=%d chosen_k=%d clusters=%d" % (n_nodes, best_k, len(clusters)))


if __name__ == "__main__":
    main()
