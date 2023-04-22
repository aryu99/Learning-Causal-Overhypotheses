from typing import Callable
from tqdm import trange

import numpy as np

from os.path import join
import bz2
import pickle
import _pickle as cPickle

from dibs.target import Data
from copy import deepcopy
from pathlib import Path

from hypotheses import HYPS


def make_dirs(directory: str):
    """Make dir path if it does not exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


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
    env,
    num_samples: int,
    store_observational_data_callback: Callable,
    store_eps_info: bool = False,
    policy
):
    """Collect observational data from environment using given policy

    Args:
        env (gym.Env): Environment to collect data from.
        num_samples (int): Number of timesteps
        store_observational_data_callback (Callable): format trajectory information to store in numpy array
            takes (state, action, next_state, reward, done, info, t_step, store_eps_info) as arguments
        store_eps_info (bool): Store information about episode
        policy (Policy): policy to follow
    """

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


def evaluate_policy(*, policy, env, n_episodes: int, log: Logger):
    """Evaluate agent using given policy

    Args:
        policy (Policy): Policy to evaluate
        env (gy.Env): Environment to evaluate policy in
        n_episodes (int): number of episodes to evaluate for
        log (Logger): Logger to store evaluation information

    Returns:
        list: rewards for each episode
        Logger: Logger with evaluation information
    """
    eps_rew = []
    for e in trange(n_episodes):
        state = env.reset()
        policy.reset()
        log.reset(hyp=HYPS[class_to_hyp(str(env._current_gt_hypothesis))])
        ep_rew = 0
        done = False
        while not done:
            action, hyp = policy(state=state)
            next_state, reward, done, info = env.step(action)
            log.step(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                hyp=hyp,
            )
            state = next_state
            ep_rew += reward
        log.push()
        eps_rew.append(ep_rew)
    return eps_rew, log


def make_data_env(
    *, policy, env, num_samples: int, n_vars: int, key, g: np.ndarray
) -> Data:
    """Collect observational data from environment.

    Args:
        policy (Policy): Policy to follow
        env (gym.Env): Environment to collect data from.
        num_samples (int): Number of samples to collect
        n_vars (int): Number of variables in the environment
        key (jnp.ndarray): JAX PRNG key
        g (np.ndarray): Adjacency matrix of ground truth SCM (can be arbitrary)

    Returns:
        Data: Returns a Data object containing the collected data
    """

    # Create numpy array to store training data
    x = np.zeros((num_samples, n_vars))

    # Define simple method to collect training data
    def store_state(
        *, state, action, next_state, reward, done, info, t_step, store_eps_info
    ) -> None:
        obs = np.concatenate([action, next_state[-1:]]).astype(
            np.float32
        ) + np.random.normal(0, 0.1, n_vars)
        assert obs.shape[0] == n_vars, "Observation shape does not match n_vars!"
        x[t_step, :] = obs

    # collect training (observational) data from environment
    collect_observational_data(
        env=env,
        num_samples=num_samples,
        store_observational_data_callback=store_state,
        store_eps_info=False,
        policy=policy,
    )

    # Repeat for holdout & intervention data
    x_ho = np.zeros((num_samples, n_vars))

    def store_state(
        *, state, action, next_state, reward, done, info, t_step, store_eps_info
    ) -> None:
        obs = np.concatenate([action, next_state[-1:]]).astype(
            np.float32
        ) + np.random.normal(0, 0.1, n_vars)
        assert obs.shape[0] == n_vars, "Observation shape does not match n_vars!"
        x_ho[t_step, :] = obs

    collect_observational_data(
        env=env,
        num_samples=num_samples,
        store_observational_data_callback=store_state,
        store_eps_info=False,
        policy=policy,
    )

    x_interv = []

    def store_state(
        *, state, action, next_state, reward, done, info, t_step, store_eps_info
    ) -> None:
        x_interv.append(
            (action, np.concatenate([action, next_state[-1:]]).astype(np.float32))
        )

    collect_observational_data(
        env=env,
        num_samples=num_samples,
        store_observational_data_callback=store_state,
        store_eps_info=False,
        policy=policy,
    )

    data = Data(
        passed_key=key,
        n_vars=n_vars,
        n_observations=num_samples,
        n_ho_observations=num_samples,
        g=g,  # NOTE: Can be arbitrary adjacency matrix.
        theta=None,
        x=x,
        x_ho=x_ho,
        x_interv=x_interv,
    )

    return data


class NstepHistory:
    """N-step history of intervention data.

    Args:
        n_step: number of steps to store.

    Attributes:
        data: numpy array of shape (n_step, n_vars).
    """

    def __init__(self, *, n_step: int) -> None:
        self.n_step = n_step
        self.reset()

    def __call__(self, **kwargs):
        """Update history with new intervention data.

        Args:
            state: numpy array of shape (n_vars,).
        """
        if self.data is None:
            self.data = np.tile(deepcopy(kwargs["state"]), (self.n_step, 1))
        else:
            self.data = np.roll(self.data, -1, axis=0)
            self.data[-1:, ...] = deepcopy(kwargs["state"])

    def reset(self):
        """Reset history to None."""
        self.data = None


class History:
    """History of intervention data.

    Args:
        n_vars: number of variables.

    Attributes:
        data: numpy array of shape (steps, n_vars).
    """

    def __init__(self, *, n_vars: int) -> None:
        self.n_vars = n_vars
        self.reset()

    def __call__(self, **kwargs):
        """Update history with new intervention data.

        Args:
            state: numpy array of shape (n_vars,).
        """
        self.data = np.concatenate(
            [self.data, deepcopy(kwargs["state"][None, ...])], axis=0
        )

    def reset(self):
        """Reset history to None."""
        self.data = np.empty((0, self.n_vars))
