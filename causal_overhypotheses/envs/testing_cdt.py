from typing import Tuple, Callable

import yaml, easydict, argparse, gym, copy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange
import cdt
from cdt.metrics import SID
from cdt.causality.pairwise import ANM, IGCI

from causal_env_v0 import CausalEnv_v0
from hypotheses import *


class Policy:
    """Abstract class for policy"""

    def __init__(self) -> None:
        pass

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): agent state

        Returns:
            np.ndarray: action
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs) -> None:
        """Reset policy"""
        pass


class RandomPolicy(Policy):
    """Random samples from action space

    Args:
        action_space (gym.Space): action space

    Attributes:
        action_space (gym.Space): action space
    """

    def __init__(self, *, action_space) -> None:
        self.action_space = action_space

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        return self.action_space.sample()


class CausalPolicy(Policy):
    """_summary_

    Args:
        obs_shape (int): observation shape
        action_shape (int): action shape
        clusters (list): list of hypotheses clustered by overhypotheses
    """

    def __init__(self, *, obs_shape: int, action_shape: int, clusters: list) -> None:
        super().__init__()
        self.clusters = clusters
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        self.obs_buffer = np.append(
            self.obs_buffer, state[None, : self.obs_shape], axis=0
        )
        data = pd.DataFrame(copy.deepcopy(self.obs_buffer))


        if state.shape[-1] > self.obs_shape and self.step_cnt > 0:

            idx = 0
            fig, ax = plt.subplots(nrows=2, ncols=8)
            for cluster in self.clusters:
                for graph in cluster:
                    gph = nx.from_numpy_array(
                            graph, parallel_edges=False, create_using=nx.Graph
                        )

                    output_graph = self.model.orient_undirected_graph(
                        data,
                        gph,
                    )

                    # glasso = cdt.independence.graph.Glasso()
                    # skeleton = glasso.predict(data)
                    # print("Glasso: ", nx.adjacency_matrix(skeleton).todense())

                    # new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg="aracne")
                    # print("New skeleton (alg=aracne): ", nx.adjacency_matrix(new_skeleton).todense())

                    nx.draw_networkx(copy.deepcopy(gph), ax=ax[0, idx], font_size=8, label='template')
                    nx.draw_networkx(copy.deepcopy(output_graph), ax=ax[1, idx], font_size=8, label='output')
                    idx += 1

            plt.tight_layout()
            plt.savefig('{}.png'.format(self.step_cnt))

            self.step_cnt += 1
            return self.action_space.sample()
        else:
            self.step_cnt += 1
            return self.action_space.sample()

    def reset(self) -> None:
        """Reset policy"""
        self.model = cdt.causality.graph.GES()
        self.obs_buffer = np.empty((0, self.obs_shape))
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        self.step_cnt = 0

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
    """_summary_

    Args:
        policy (Policy): _description_
        env_cfg (dict): _description_
        n_eval (int): _description_

    Returns:
        list: _description_
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


def cluster_graphs(list_of_graphs: list):
    """Cluster causal graphs based on overhypotheses

    Args:
        list_of_graphs (list): List of causal graphs
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Causal Overhypotheses")
    parser.add_argument("-e", "--env_config", required=True, help="path to env config")
    parser.add_argument("-x", "--exp_config", required=True, help="path to exp config")
    args = parser.parse_args()

    # Load env config
    with open(args.env_config, "r") as f:
        env_config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(env_config.hypotheses, globals(), ldict)
        env_config.hypotheses = ldict["hypotheses"]

    env = CausalEnv_v0(env_config)
    action_shape = len(env.action_space)
    obs_shape = env.observation_space._shape[0]

    # Load exp config
    with open(args.exp_config, "r") as f:
        exp_config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(exp_config.env.hypotheses, globals(), ldict)
        exp_config.env.hypotheses = ldict["hypotheses"]

    # Create numpy array to store training data
    data_np = np.zeros((exp_config.num_samples, obs_shape), dtype=np.int32)

    # Define simple method to collect training data
    def store_state(
        *, state, action, next_state, reward, done, info, t_step, store_eps_info
    ) -> None:
        # data_np[t_step, :action_shape] = action
        data_np[t_step, :] = next_state

    # initialize random policy
    random_policy = RandomPolicy(action_space=env.action_space)

    # collect training (observational) data from environment
    collect_observational_data(
        env_config=env_config,
        num_samples=exp_config.num_samples,
        store_observational_data_callback=store_state,
        store_eps_info=False,
        policy=random_policy,
    )

    # Convert numpy.ndarray to dataframe, set column names
    data = pd.DataFrame(data_np, columns=exp_config.labels)

    model = cdt.causality.graph.GES()
    output_graph = model.predict(data)
    print("GES: ", nx.adjacency_matrix(output_graph).todense())

    # Draw the graph output from using GES
    # nx.draw_networkx(output_graph, font_size=8)
    # plt.show()

    # Initialize policy with overhypotheses clusters
    c_agent = CausalPolicy(
        obs_shape=obs_shape,
        action_shape=action_shape,
        clusters=[SINGLE, DOUBLE, NONE, ALL],
    )
    exp_config.env.n_blickets = env_config.n_blickets

    # Evaluate Agent
    evaluate_agent(policy=c_agent, env_cfg=exp_config.env, n_eval=exp_config.n_eval)
