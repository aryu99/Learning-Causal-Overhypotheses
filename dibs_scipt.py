import argparse
from os.path import join

import jax
import jax.random as random
from jax import numpy as jnp

from easydict import EasyDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import gym
from adjustText import adjust_text

from sklearn.manifold import TSNE
from umap import UMAP

from causal_env_v0 import CausalEnv_v0
from dibs.inference import JointDiBS, VQDiBS
from dibs.models import DenseNonlinearGaussian
from dibs.target import make_graph_model
from dibs.metrics import expected_shd, neg_ave_log_likelihood, threshold_metrics

from hypotheses import *

from policy import RandomLLPolicy, RandomPolicy, EGreedyLLPolicy, BALDPolicy
from utils import (
    Logger,
    class_to_hyp,
    compress_pickle,
    decompress_pickle,
    evaluate_policy,
    make_data_env,
    make_dirs,
    History,
    GaussianNoiseEnv,
    read_experiment_config,
    read_env_config,
)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rcParams["legend.frameon"] = True
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

GRAPH_ARRAY = [*SINGLE, *DOUBLE, *ALL]

def reshape_theta(theta, num_particles):
    """Reshape theta to be a list of length num_particles, where each element
    is a list of length len(theta)

    Args:
        theta (list): List of tuples of length num_particles
        num_particles (int): Number of particles
    """
    thetas = []
    for k in range(num_particles):
        single_theta = []
        for i in range(len(theta)):
            if len(theta[i]) > 0:
                single_theta.append((theta[i][0][k, ...], theta[i][1][k, ...]))
            else:
                single_theta.append(tuple([]))
        thetas.append(single_theta)
    return thetas


def compute_log_likelihood(z, t=2000):
    """Compute the log likelihood of a graph given the sampled latent parameters

    Args:
        z (np.ndarray): Array of shape (NUM_PARTICLES, N_VARS, PARTICLE_DIM, 2)
    """
    num_particles = z.shape[0]
    p_g_z = {}
    for name, G in zip(GRAPH_LABELS, GRAPH_ARRAY):
        p_g_z[name] = np.zeros((num_particles,))
        for i in range(num_particles):
            p_g_z[name][i] = dibs.latent_log_prob(G, z[i, ...], t=t)
            print(f"log p({name} | Z_{i}) = {p_g_z[name][i]}")
        print()
    return p_g_z


def compute_best_particles(p_g_z, k=10):
    """Identify the best particles for each hypotheses"""
    best_particles = {}
    for name, _ in zip(GRAPH_LABELS, GRAPH_ARRAY):
        idx = np.argpartition(p_g_z[name], -k)[-k:][::-1]
        print(f"Top {k} particles for {name}: {idx}")
        best_particles[name] = idx
    return best_particles


def plot_marginal_log_likelihoods(gs, theta, *, method, save_path):
    """Plot marginal log likelihoods of each particle"""
    dibs_mixture = dibs.get_mixture(gs, theta)
    for idx, logp_z in enumerate(dibs_mixture.logp):
        print(f"log p(G, theta | z_{idx}): {logp_z}")

    _, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.arange(len(best_particles)))

    num_particles = len(dibs_mixture.logp)

    marked = np.zeros((num_particles,))
    for idx, graph_name in enumerate(best_particles):
        particle_idx = []
        for i in range(num_particles):
            if i in best_particles[graph_name] and not marked[i] and i != 2:
                marked[i] = 1
                particle_idx.append(i)
        particle_idx = np.array(particle_idx).astype(np.int32)
        ax.scatter(
            particle_idx,
            dibs_mixture.logp[particle_idx],
            color=colors[idx],
            label=graph_name,
        )

    ax.set_xticks(np.arange(num_particles))
    ax.set_xticklabels(np.arange(num_particles), rotation=90)
    ax.set_xlabel("Particle index " + r"$i$")
    ax.set_ylabel(r"$\log p(G, \Theta | Z^{(i)})$")
    ax.set_title("Mixture log likelihood " + r"$\log p(G, \Theta | Z^{(i)})$")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(
        join(save_path, f"{method}-mixture_log_likelihoods.png"), bbox_inches="tight"
    )


def compute_node_embedding_cosine_distance(u, v):
    """Compute the cosine distance between node embeddings

    Args:
        u: U matrix
        v: V matrix

    Returns:
        np.ndarray: Cosine distance of shape (num_particles, n_vars, n_vars)
    """
    num_particles, n_vars, _ = u.shape
    grid = np.zeros((num_particles, n_vars, n_vars))

    for k in range(num_particles):
        for i in range(n_vars):
            for j in range(n_vars):
                Z = jnp.concatenate([u[k, i, :], v[k, i, :]], axis=0)
                Z_norm = jnp.expand_dims(jnp.linalg.norm(Z), axis=0)

                B = jnp.concatenate([u[k, j, :], v[k, j, :]], axis=0).T
                B_norm = jnp.expand_dims(jnp.linalg.norm(B), axis=0)

                cosine_similarity = ((Z @ B.T) / (Z_norm @ B_norm)).T
                cosine_distance = 1 - cosine_similarity
                grid[k, i, j] = cosine_distance
    return grid


def plot_grid(grid, method, save_path, n_cols=7, size=2.5):
    """Plot a grid of matrices"""
    N = grid.shape[0]
    n_rows = N // n_cols
    if N % n_cols:
        n_rows += 1

    plt.rcParams["figure.figsize"] = [size * n_cols, size * n_rows]
    fig, axs = plt.subplots(n_rows, n_cols)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i < N:
            ax.matshow(grid[i, :, :])
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_title(r"$Z^{(" f"{i}" r")}$", pad=3)
        ax.tick_params(axis="both", which="both", length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(join(save_path, f"{method}-node_embeddings.png"), bbox_inches="tight")


def eval_dibs(gs, theta):
    """Evaluate DiBS and DiBS+ on SHD, AUROC and NLL"""
    dibs_empirical = dibs.get_empirical(gs, theta)
    dibs_mixture = dibs.get_mixture(gs, theta)
    results = {'DiBS': {}, 'DiBS+': {}}
    for descr, dist in [("DiBS", dibs_empirical), ("DiBS+", dibs_mixture)]:

        eshd = expected_shd(dist=dist, g=data.g)
        auroc = threshold_metrics(dist=dist, g=data.g)["roc_auc"]
        negll = neg_ave_log_likelihood(
            dist=dist,
            x=data.x_ho,
            eltwise_log_likelihood=dibs.eltwise_log_likelihood_observ,
        )
        results[descr]['eshd'], results[descr]['auroc'], results[descr]['negll'] = eshd, auroc, negll
        print(
            f"{descr} |  E-SHD: {eshd:4.1f}    AUROC: {auroc:5.2f}    neg. MLL {negll:5.2f}"
        )
    return results


def particle_to_u_v_z_idx(z):
    """Convert particles Z to components U and V. Stack as single vector.
    Also return particle indices.

    Returns:
        X: node embeddings
        Y: node labels
        Zidx: particle indices
    """
    num_particles, n_vars, particle_dim, _ = z.shape
    u, v = z[..., 0], z[..., 1]

    X = np.zeros((num_particles, n_vars, particle_dim * 2))
    Y = np.zeros((num_particles, n_vars, 1))
    Z = np.zeros((num_particles, n_vars, 1))
    for k in range(num_particles):
        for i in range(n_vars):
            X[k, i, :] = np.concatenate([u[k, i, :], v[k, i, :]], axis=0)
            Y[k, i, :] = i
            Z[k, i, :] = k
    return (
        X.reshape(-1, particle_dim * 2),
        Y.reshape(-1).astype(int),
        Z.reshape(-1).astype(int),
    )


def plot_embeddings(
    embedding,
    Y,
    label_frmt,
    method,
    save_path,
    dibs_method,
    prefix=None,
    point_labels=None,
    filter_=None,
):
    plt.figure(figsize=(8, 5), dpi=300)

    for g in np.unique(Y):
        idx = np.where(Y == g)
        x, y = embedding[idx, 0], embedding[idx, 1]
        plt.scatter(x, y, s=25, label=label_frmt.format(int(g)))

    texts = []
    # Annotate some points in the scatter plot
    if point_labels is not None:

        # for each point plotted
        for idx in range(embedding.shape[0]):
            # if that index is to be annotated
            if point_labels[idx] in filter_:
                texts.append(
                    plt.text(
                        embedding[idx, 0],
                        embedding[idx, 1],
                        r"$Z^{(" + f"{int(point_labels[idx])}" + r")}$",
                    )
                )

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(f"[{dibs_method}] {method} projection of Node Embeddings")

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    plt.tight_layout()
    if prefix is None:
        plt.savefig(
            join(save_path, f"{dibs_method}-{method}-projection.png"),
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            join(save_path, f"{dibs_method}-{method}-{prefix}-projection.png"),
            bbox_inches="tight",
        )


def particle_to_single_vec(z):
    """Convert particles Z to components U and V. Stack as single vector.
    Also return particle indices."""
    num_particles, n_vars, particle_dim, _ = z.shape
    u, v = z[..., 0], z[..., 1]

    X = np.zeros((num_particles, n_vars, particle_dim * 2))
    Y = np.zeros((num_particles, 1))
    for k in range(num_particles):
        for i in range(n_vars):
            X[k, i, :] = np.concatenate([u[k, i, :], v[k, i, :]], axis=0)
        Y[k, :] = k
    return (
        X.reshape(-1, n_vars * particle_dim * 2),
        Y.reshape(-1).astype(int),
    )


def plot_wrt_functional_mechanisms(best_particles, num_particles, n_vars):
    Y_g = np.zeros((num_particles, n_vars, 1))
    for i in range(num_particles):
        for g_idx, key in enumerate(best_particles):
            for idx in best_particles[key]:
                if i == idx:
                    if g_idx < 3:
                        Y_g[idx, :, :] = 0
                    elif g_idx > 3 and g_idx < 6:
                        Y_g[idx, :, :] = 1
                    else:
                        Y_g[idx, :, :] = 2

    Y_g = Y_g.reshape(-1).astype(int)
    return Y_g


def plot_wrt_functional_mechanisms_per_particle(best_particles, num_particles):
    Y_g = np.zeros((num_particles, 1))
    for i in range(num_particles):
        for g_idx, key in enumerate(best_particles):
            for idx in best_particles[key]:
                if i == idx:
                    if g_idx < 3:
                        Y_g[idx, :] = 0
                    elif g_idx > 3 and g_idx < 6:
                        Y_g[idx, :] = 1
                    else:
                        Y_g[idx, :] = 2

    Y_g = Y_g.reshape(-1).astype(int)
    return Y_g


def plot_embeddings_3d(df, x, y, z, color):
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=color.astype(str),
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig.show()


def embedding_to_df(embedding, **kwargs):
    df = pd.DataFrame(embedding, columns=["x", "y", "z"])
    for k, v in kwargs.items():
        df[k] = v
    return df


def make_node_embedding_plots(z, best_particles, *, method, save_path):
    X, Y, Zidx = particle_to_u_v_z_idx(z)

    X_umap = UMAP(n_neighbors=4, n_components=2, metric="cosine").fit_transform(X)
    X_tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        perplexity=7,
        n_iter=5000,
        metric="cosine",
    ).fit_transform(X)

    plot_embeddings(
        X_umap,
        Y,
        label_frmt="Node {}",
        method="UMAP",
        point_labels=None,
        filter_=None,
        dibs_method=method,
        save_path=save_path,
    )

    plot_embeddings(
        X_tsne,
        Y,
        label_frmt="Node {}",
        method="TSNE",
        point_labels=None,
        filter_=None,
        dibs_method=method,
        save_path=save_path,
    )

    # X_, Y_ = particle_to_single_vec(z)

    # X_umap = UMAP(n_neighbors=4, n_components=2, metric="cosine").fit_transform(X_)
    # X_tsne = TSNE(
    #     n_components=2,
    #     learning_rate="auto",
    #     init="random",
    #     perplexity=7,
    #     n_iter=5000,
    #     metric="cosine",
    # ).fit_transform(X_)

    # Y_func = plot_wrt_functional_mechanisms_per_particle(best_particles, z.shape[0])
    # plot_embeddings(
    #     X_umap,
    #     Y_func,
    #     label_frmt="Func. {}",
    #     method="UMAP",
    #     prefix="functional",
    #     point_labels=None,
    #     filter_=None,
    #     dibs_method=method,
    #     save_path=save_path,
    # )

    # plot_embeddings(
    #     X_tsne,
    #     Y_func,
    #     label_frmt="Func. {}",
    #     method="TSNE",
    #     prefix="functional",
    #     point_labels=None,
    #     filter_=None,
    #     dibs_method=method,
    #     save_path=save_path,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of observation samples to collect",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=30,
        help="Number of particles to use for $Z$",
    )
    parser.add_argument("--n_vars", type=int, default=4, help="Number of nodes in SCM")
    parser.add_argument(
        "--particle_dim",
        type=int,
        default=4,
        help="Dimension of embedding. If `PARTICLE_DIM = N_VARS`, then embedding will be full-rank.",
    )
    parser.add_argument(
        "--num_codebook_per_node",
        type=int,
        default=8,
        help="Number of codebooks per node",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method to use for embedding",
        choices=["VQDiBS", "JointDiBS"],
    )
    parser.add_argument(
        "--plot_dir", type=str, default="./plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--env_config", type=str, required=True, help="Environment config file"
    )
    parser.add_argument("--exp_config", type=str, help="Experiment config file")
    args = parser.parse_args()
    make_dirs(args.plot_dir)

    key = random.PRNGKey(123)
    print(f"JAX backend: {jax.default_backend()}")

    key, subk = random.split(key)

    env_config = read_env_config(args.env_config)
    env = GaussianNoiseEnv(CausalEnv_v0(env_config))
    random_policy = RandomPolicy(action_space=env.action_space)

    data = make_data_env(
        policy=random_policy,
        env=env,
        num_samples=args.num_samples,
        n_vars=args.n_vars,
        key=key,
        g=A,
    )
    print(f"Hypotheses:\n{env_config.hypotheses}")

    graph_dist = make_graph_model(
        n_vars=args.n_vars, graph_prior_str="", edges_per_node=2
    )
    inference_model = DenseNonlinearGaussian(
        hidden_layers=[5, 5], activation="sigmoid", graph_dist=graph_dist
    )

    dummy_subkeys = jnp.zeros((args.n_vars, 2), dtype=jnp.uint32)
    _, dummy_theta = inference_model.eltwise_nn_init_random_params(
        dummy_subkeys, (args.n_vars,)
    )
    print(
        f"Inference model parameters shape: {inference_model.get_theta_shape(n_vars=args.n_vars)}"
    )

    inter_mask = np.concatenate(
        [np.ones((args.num_samples, 3)), np.zeros((args.num_samples, 1))], axis=-1
    ).astype(np.int32)

    key, subk = random.split(key)

    if args.method == "VQDiBS":
        dibs = VQDiBS(
            x=data.x,
            interv_mask=inter_mask,
            inference_model=inference_model,
            verbose=True,
            num_codebook_per_node=args.num_codebook_per_node,
            num_nodes=args.n_vars,
            particle_dim=args.particle_dim,
            key=subk,
        )
    elif args.method == "JointDiBS":
        dibs = JointDiBS(
            x=data.x,
            interv_mask=inter_mask,
            inference_model=inference_model,
            verbose=True,
        )

    key, subk = random.split(key)

    gs, theta, z = dibs.sample(
        key=subk,
        n_particles=args.num_particles,
        steps=2000,
        callback_every=500,
        return_z=True,
        callback=None,  # dibs.visualize_callback(ipython=False, save_path=args.plot_dir),
        n_dim_particles=args.particle_dim,
    )

    u, v = z[..., 0], z[..., 1]
    scores = jnp.einsum("...ik,...jk->...ij", u, v)
    plot_grid(compute_node_embedding_cosine_distance(u, v), method=args.method, save_path=args.plot_dir)
    eval_dibs(gs, theta)

    plt.style.use("seaborn-v0_8-darkgrid")
    thetas = reshape_theta(theta, args.num_particles)
    best_particles = compute_best_particles(compute_log_likelihood(z, t=2000), k=5)

    print(f"Best particles: {best_particles}")
    make_dirs(f"{args.method}-params")
    compress_pickle(f"{args.method}-params/{args.method}-z.pkl", z)
    compress_pickle(f"{args.method}-params/{args.method}-gs.pkl", gs)
    compress_pickle(f"{args.method}-params/{args.method}-thetas.pkl", thetas)

    plot_marginal_log_likelihoods(
        gs, theta, method=args.method, save_path=args.plot_dir
    )

    make_node_embedding_plots(
        z, best_particles, method=args.method, save_path=args.plot_dir
    )

    exp_config = read_experiment_config(args.exp_config)
    eval_env = GaussianNoiseEnv(CausalEnv_v0(exp_config.env))

    policy_bald = BALDPolicy(
        inference_model=dibs.inference_model,
        theta=theta,
        thetas=thetas,
        gs=gs,
        inter_mask=inter_mask,
        action_space=env.action_space,
        n_step=5,
        force_exploration=5,
        buffer=History(n_vars=env.observation_space.shape[0]),
        best_particles=best_particles,
        topk=6,
        key=subk,
    )

    _, log = evaluate_policy(
        policy=policy_bald,
        env=eval_env,
        n_episodes=exp_config.get("n_episodes", 10),
        log=Logger(log_dir=f"./logs/bald_{args.method}/"),
    )
    log.save(file_name="bald.pkl")

    # policy_greedy = EGreedyLLPolicy(
    #     inference_model=dibs.inference_model,
    #     thetas=theta,
    #     gs=gs,
    #     inter_mask=inter_mask,
    #     action_space=env.action_space,
    #     n_step=5,
    #     force_exploration=5,
    #     buffer=History(n_vars=env.observation_space.shape[0]),
    #     best_particles=best_particles,
    #     topk=2,
    # )

    # _, log = evaluate_policy(
    #     policy=policy_greedy,
    #     env=eval_env,
    #     n_episodes=exp_config.get("n_episodes", 10),
    #     log=Logger(log_dir=f"./logs/greedy_{args.method}/"),
    # )
    # log.save(file_name="greedy.pkl")

    policy_random = RandomLLPolicy(
        inference_model=dibs.inference_model,
        thetas=theta,
        gs=gs,
        inter_mask=inter_mask,
        action_space=env.action_space,
        n_step=5,
        force_exploration=5,
        buffer=History(n_vars=env.observation_space.shape[0]),
        best_particles=best_particles,
    )

    _, log = evaluate_policy(
        policy=policy_random,
        env=eval_env,
        n_episodes=exp_config.get("n_episodes", 10),
        log=Logger(log_dir=f"./logs/random_{args.method}/"),
    )
    log.save(file_name="random.pkl")
