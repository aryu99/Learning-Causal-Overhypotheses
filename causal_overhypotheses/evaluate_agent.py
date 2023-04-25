from sb3_contrib import RecurrentPPO
from envs.causal_env_v0 import *

import argparse, yaml, easydict
from tqdm import trange
import torch, random
import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import bz2
import pickle
import _pickle as cPickle
from os.path import join

from pathlib import Path

import numpy as np

A = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

B = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

C = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)


AB = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

BC = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

CA = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

ABC = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

N = np.array(
    [[0, 0, 0, 0],
	 [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

FULL = np.array(
    [[0, 1, 1, 1],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

SINGLE = [A, B, C]
DOUBLE = [AB, BC, CA]
NONE = [N]
ALL = [ABC]

HYPS = {
    "A": A,
    "B": B,
    "C": C,
    "AB": AB,
    "BC": BC,
    "AC": CA,
    "CA": CA,
    "ABC": ABC,
}

def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces
    :param seed: (int) the seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def compress_pickle(fname: str, data):
    """Compress & save data as pickle file at given path
    Args:
        fname (str): File path
        data (Any): Serializable object
    """
    with bz2.BZ2File(fname, "wb") as f:
        cPickle.dump(data, f)


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


class Logger:
    def __init__(self, *, log_dir: str) -> None:
        """Logger to store episode information

        Args:
            log_dir (str): Path to directory to store logs
        """
        self.log_dir = log_dir
        make_dirs(self.log_dir)
        self.log = []
        self.current_log = []

    def reset(self, **kwargs) -> None:
        """Reset logger

        Args:
            **kwargs: Keyword arguments to store in log
        """
        self.current_param = kwargs
        self.current_log = []

    def push(self) -> None:
        """Push current log to log list"""
        self.log.append([self.current_param, self.current_log])

    def step(self, **kwargs) -> None:
        """Add step information to current log

        Args:
            **kwargs: Keyword arguments to store in log
        """
        self.current_log.append(kwargs)

    def save(self, *, file_name: str) -> None:
        """Save log to file

        Args:
            file_name (str): File name to save log to
        """
        compress_pickle(join(self.log_dir, file_name), self.log)

def class_to_hyp(name: str) -> str:
    """Convert class name to hypothesis string.

    Args:
        name (str): Class name

    Returns:
        str: Hypothesis string (e.g. "A")
    """
    if "conj" in name:
        name = name.split("conj")[0][-2:]
    else:
        name = name.split("disj")[0][-1]
    return name

def evaluate_policy(*, policy, env_cfg: dict, n_episodes: int, log: Logger):
    """Evaluate agent using given policy

    Args:
        policy (Policy): Policy to evaluate
        env_cfg (dict): environment configuration
        n_episodes (int): number of episodes to evaluate for
        log (Logger): Logger to store evaluation information

    Returns:
        list: rewards for each episode
        Logger: Logger with evaluation information
    """
    def vec_env():
        env = DummyVecEnv([lambda : CausalEnv_v0(env_cfg)])
        return env

    def get_graph(action_seq):
        s = np.zeros((4, 4), dtype=int)
        if action_seq[0] == 1:
            s += HYPS["A"]
        if action_seq[1] == 1:
            s += HYPS["B"]
        if action_seq[2] == 1:
            s += HYPS["C"]
        return s

    env = vec_env()
    eps_rew = []
    for e in trange(n_episodes):

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        state = env.reset()
        # Reset policy
        log.reset(hyp=HYPS[class_to_hyp(str(env.envs[0]._current_gt_hypothesis))])
        ep_rew = 0
        done = False
        while not done:
            action, lstm_states = policy.predict(state, state=lstm_states, episode_start=episode_starts, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_starts = done

            with torch.no_grad():
                state_ = np.concatenate([state[:, :4], np.ones((1, 1))], axis=1)
                action_, lstm_states_ = policy.predict(state, state=lstm_states, episode_start=episode_starts, deterministic=True)
                # print(f"state: {state}, state_: {state_}, action: {action}, action_: {action_}")

            log.step(
                state=state[0, :4],
                action=action,
                next_state=next_state[0, :4],
                reward=reward[0],
                hyp=get_graph(action_[0]),
            )
            
            # print(get_graph(action_[0]))

            state = next_state
            ep_rew += reward
        log.push()
        eps_rew.append(ep_rew)
    return eps_rew, log


def read_experiment_config(path: str) -> dict:
    """Read experiment configuration from file.

    Args:
        path (str): path to experiment configuration file

    Returns:
        dict: experiment configuration
    """
    with open(path, "r") as f:
        exp_config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(exp_config.env.hypotheses, globals(), ldict)
        exp_config.env.hypotheses = ldict["hypotheses"]

    return exp_config


def main(args, exp_cfg):
    model = RecurrentPPO.load(args.load_path)
    _, log = evaluate_policy(policy=model, env_cfg=exp_cfg['env'], n_episodes=exp_cfg.n_episodes, log=Logger(log_dir=args.log_dir))
    log.save(file_name='ppo2_lstm.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--exp_config', type=str, help='Path to experiment config')
    parser.add_argument('--load_path', type=str, help='Path to load model from')
    parser.add_argument('--log_dir', type=str, help='Path to save logs')
    args = parser.parse_args()

    exp_cfg = read_experiment_config(args.exp_config)
    main(args, exp_cfg)
