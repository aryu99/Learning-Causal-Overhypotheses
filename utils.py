from typing import Callable
from tqdm import trange

from causal_env_v0 import CausalEnv_v0
from policy import Policy


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


def evaluate_agent(*, policy: Policy, env_cfg: dict, n_eval: int) -> list:
    """Evaluate agent using given policy

    Args:
        policy (Policy): Policy to evaluate
        env_cfg (dict): environment configuration
        n_eval (int): number of episodes to evaluate for

    Returns:
        list: rewards for each episode
    """
    env = CausalEnv_v0(env_cfg)
    eps_rew = []
    for e in trange(n_eval):
        state = env.reset()
        policy.reset()
        ep_rew = 0
        done = False
        while not done:
            action = policy(state=state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_rew += reward
        eps_rew.append(ep_rew)
    return eps_rew
