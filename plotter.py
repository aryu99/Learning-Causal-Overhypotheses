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

def decompress_pickle(fname: str):
    """Decompress pickle file at given path
    Args:
        fname (str): Path to file
    Returns:
        Serializable object
    """
    data = bz2.BZ2File(fname, "rb")
    data = cPickle.load(data)
    return data


def make_dirs(directory: str):
    """Make dir path if it does not exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def plot_rewards(paths, names, save_path):
    mean = {}
    var = {}
    for path, name in zip(paths, names):
        files = glob.glob(join(path, '*.pkl'))
        data = decompress_pickle(files[-1])
        make_dirs(save_path)

        rewards = np.zeros((50, 25))
        for i in range(50):
            dat = np.array([d['reward'] for d in data[i][1]])
            rewards[i, :dat.shape[0]] = dat

        mean[name] = rewards.mean(axis=0)
        var[name] = rewards.std(axis=0)

    plt.figure(figsize=(6, 4))
    for name in names:
        plt.plot(mean[name], label=name)
        plt.fill_between(np.arange(25), mean[name] - 0.5*var[name], mean[name] + 0.5*var[name], alpha=0.2)

    plt.xlabel('Intervention Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_path, 'rewards.png'), bbox_inches='tight')
    plt.close()


def plot_right_hypotheses(paths, names, save_path, metric, metric_name):
    mean = {}
    var = {}
    for path, name in zip(paths, names):
        files = glob.glob(join(path, '*.pkl'))
        data = decompress_pickle(files[-1])
        make_dirs(save_path)

        rewards = np.zeros((50, 22))
        for i in range(50):
            gt = data[i][0]
            
            dat_shd = []
            for t in range(6, 22):
                try:
                    dat_shd.append(metric(gt['hyp'], HYPS[data[i][1][t]['hyp']]))
                except:
                    print(gt['hyp'], data[i][1][t]['hyp'])
                    dat_shd.append(metric(gt['hyp'], data[i][1][t]['hyp']))
            rewards[i, :len(dat_shd)] = dat_shd

        mean[name] = rewards.mean(axis=0)
        var[name] = rewards.std(axis=0)

    plt.figure(figsize=(6, 4))
    for name in names:
        plt.plot(mean[name], label=name)
        plt.fill_between(np.arange(22), mean[name] - 0.5*var[name], mean[name] + 0.5*var[name], alpha=0.2)

    plt.xlabel('Intervention Steps')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_path, f'{metric_name}.png'), bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    save_path = './plots/'

    paths = ['./logs/p_greedy/', './logs/p_random/', './logs/p_ppo2/', './logs/p_ps_rand/']
    names = ['Greedy (LVM)', 'Random (LVM)', 'PPO-LSTM', 'PS-Rand']

    plot_rewards(paths, names, save_path)

    # plot_right_hypotheses(
    #     ['./logs/p_greedy/', './logs/p_random/'],
    #     ['Greedy (LVM)', 'Random (LVM)'],
    #     save_path,
    #     SHD,
    #     'SHD'
    # )