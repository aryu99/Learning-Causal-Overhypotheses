import matplotlib.pyplot as plt
import numpy as np
import glob

import bz2
import pickle
import _pickle as cPickle
from pathlib import Path

from hypotheses import HYPS

from os.path import join
from cdt.metrics import SHD, SID, SHD_CPDAG, SID_CPDAG

from utils import make_dirs, decompress_pickle

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["legend.frameon"] = True
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_rewards(paths, names, save_path):
    mean = {}
    var = {}
    for path, name in zip(paths, names):
        files = glob.glob(join(path, "*.pkl"))
        data = decompress_pickle(files[-1])
        make_dirs(save_path)

        rewards = np.zeros((50, 22))
        for i in range(50):
            dat = np.array([d["reward"] for d in data[i][1]])
            rewards[i, : dat.shape[0]] = dat

        mean[name] = rewards.mean(axis=0)
        var[name] = rewards.std(axis=0)

    plt.figure(dpi=300)
    for name in names:
        plt.plot(mean[name], label=name)
        plt.fill_between(
            np.arange(22),
            mean[name] - 0.5 * var[name],
            mean[name] + 0.5 * var[name],
            alpha=0.2,
        )

    plt.xlabel("Intervention Steps")
    plt.ylabel("Reward")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(join(save_path, "rewards.png"), bbox_inches="tight")
    plt.close()


def plot_right_hypotheses(paths, names, save_path, metric, metric_name):
    mean = {}
    var = {}

    for path, name in zip(paths, names):
        files = glob.glob(join(path, "*.pkl"))
        data = decompress_pickle(files[-1])

        rewards = np.zeros((50, 22))
        for i in range(50):
            gt = data[i][0]

            dat_shd = []
            for t in range(22):
                dat_shd.append(metric(gt["hyp"], data[i][1][t]["hyp"]))
            rewards[i, : len(dat_shd)] = dat_shd

        mean[name] = rewards.mean(axis=0)
        var[name] = rewards.std(axis=0)

    plt.figure(figsize=(6, 4))
    for name in names:
        plt.plot(mean[name], label=name)
        plt.fill_between(
            np.arange(22),
            mean[name] - 0.5 * var[name],
            mean[name] + 0.5 * var[name],
            alpha=0.2,
        )

    plt.xlabel("Intervention Steps")
    plt.ylabel(f"{metric_name}")
    plt.legend()
    plt.tight_layout()
    make_dirs(save_path)
    plt.savefig(join(save_path, f"{metric_name}.png"), bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    save_path = "./plots/"

    paths = [
        # "./logs/greedy_JointDiBS/",
        "./logs/random_JointDiBS/",
        "./logs/bald_JointDiBS/",
        # "./logs/greedy_VQDiBS/",
        "./logs/random_VQDiBS/",
        "./logs/bald_VQDiBS/",
        "./logs/pc_random/",
        "./logs/ppo2_lstm/",
    ]
    names = [
        # "DiBS-Greedy",
        "DiBS-Rand",
        "DiBS-BALD",
        # "VQDiBS-Greedy",
        "VQDiBS-Rand",
        "VQDiBS-BALD",
        "PC-Rand",
        "PPO2-LSTM"
    ]

    # plot_rewards(paths, names, save_path)

    plot_right_hypotheses(
        paths,
        names,
        save_path,
        SHD,
        'SHD'
    )
