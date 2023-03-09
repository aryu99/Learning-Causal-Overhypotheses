from typing import Tuple, Callable
import numpy as np
import pandas as pd
import yaml, easydict, argparse, gym
from causal_env_v0 import CausalEnv_v0
import cdt
import networkx as nx
import matplotlib.pyplot as plt

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


class RandomPolicy:
    """Random samples from action space"""

    def __init__(self, *, action_space) -> None:
        self.action_space = action_space

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


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
                store_eps_info=store_eps_info
            )
            state = next_state
            t_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Causal Discovery")
    parser.add_argument("-c", "--config", required=True, help="path to env config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        env_config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(env_config.hypotheses, globals(), ldict)
        env_config.hypotheses = ldict["hypotheses"]

    data = np.zeros((1000, 4), dtype=np.int32)
    def store_state(*,
        state, action, next_state, reward, done, info, t_step, store_eps_info
    ) -> None:
        data[t_step, :] = next_state

    random_policy = RandomPolicy(action_space=CausalEnv_v0(env_config).action_space)

    collect_observational_data(
        env_config=env_config,
        num_samples=1000,
        store_observational_data_callback=store_state,
        store_eps_info=False,
        policy=random_policy,
    )

    data = pd.DataFrame(data, columns = ['A','B','C', 'L'])
    print(data)

    glasso = cdt.independence.graph.Glasso()
    skeleton = glasso.predict(data)
    print(nx.adjacency_matrix(skeleton).todense())
    new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
    print(nx.adjacency_matrix(new_skeleton).todense())
    model = cdt.causality.graph.GES()
    output_graph = model.predict(data)
    print('GES: ', nx.adjacency_matrix(output_graph).todense())

    nx.draw_networkx(output_graph, font_size=8)

    plt.show()

    # model2 = cdt.causality.graph.CAM()
    # output_graph_nc = model2.predict(data)
    # print('CAM: ', nx.adjacency_matrix(output_graph_nc).todense())
