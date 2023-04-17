from typing import Callable
from tqdm import trange

from causal_env_v0 import CausalEnv_v0
from policy import Policy

from os.path import join
import bz2
import pickle
import _pickle as cPickle

from pathlib import Path


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


def collect_observational_data(
    *,
    env_config: dict,
    num_samples: int,
    store_observational_data_callback: Callable,
    store_eps_info: bool = False,
    policy: Policy
):
    """Collect observational data from environment using given policy

    Args:
        env_config (dict): Env configuration
        num_samples (int): Number of timesteps
        store_observational_data_callback (Callable): format trajectory information to store in numpy array
            takes (state, action, next_state, reward, done, info, t_step, store_eps_info) as arguments
        store_eps_info (bool): Store information about episode
        policy (Policy): policy to follow
    """
    env = CausalEnv_v0(env_config)

    t_step = 0
    while t_step < num_samples:
        state = env.reset()
        policy.reset()
        done = False
        while not done and t_step < num_samples:
            action = policy(state=state)
            next_state, reward, done, info = env.step(action)
            store_observational_data_callback(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                info=info,
                t_step=t_step,
                store_eps_info=store_eps_info,
            )
            state = next_state
            t_step += 1


def make_dirs(directory: str):
    """Make dir path if it does not exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

class Logger:
    def __init__(self, *, log_dir: str) -> None:
        self.log_dir = log_dir
        make_dirs(self.log_dir)
        self.log = []
        self.current_log = []

    def reset(self, **kwargs) -> None:
        self.current_param = kwargs
        self.current_log = []

    def push(self) -> None:
        self.log.append([self.current_param, self.current_log])

    def step(self, **kwargs) -> None:
        self.current_log.append(kwargs)

    def save(self, *, file_name: str) -> None:
        compress_pickle(join(self.log_dir, file_name), self.log)


def evaluate_agent(*, policy: Policy, env_cfg: dict, n_eval: int, log: Logger):
    """Evaluate agent using given policy

    Args:
        policy (Policy): Policy to evaluate
        env_cfg (dict): environment configuration
        n_eval (int): number of episodes to evaluate for
        log (Logger): Logger to store evaluation information

    Returns:
        list: rewards for each episode
        Logger: Logger with evaluation information
    """
    env = CausalEnv_v0(env_cfg)
    eps_rew = []
    for e in trange(n_eval):
        state = env.reset()
        policy.reset()
        log.reset(hyp=env._current_gt_hypothesis)
        ep_rew = 0
        done = False
        while not done:
            action, hyp = policy(state=state)
            next_state, reward, done, info = env.step(action)
            log.step(state=state, reward=reward, hyp=hyp)
            state = next_state
            ep_rew += reward
        log.push()
        eps_rew.append(ep_rew)
    return eps_rew, log
