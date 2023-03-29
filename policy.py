import numpy as np
import pandas as pd
import networkx as nx
import cdt, copy, gym


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

    def __init__(self, *, action_space: gym.Space) -> None:
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
        self.clusters = []
        hyp_cnt = 0
        for cluster in clusters:
            k = []
            for graph in cluster:
                gph = nx.from_numpy_array(
                    graph, parallel_edges=False, create_using=nx.Graph
                )
                k.append(gph)
                hyp_cnt += 1
            self.clusters.append(k)
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
            # idx = 0
            # fig, ax = plt.subplots(nrows=2, ncols=8)
            for cluster in self.clusters:
                for graph in cluster:
                    # output_graph = self.model.orient_undirected_graph(
                    #     data,
                    #     graph,
                    # )
                    output_graph = self.model.orient_graph(
                        data,
                        graph,
                    )

                    print(nx.adjacency_matrix(output_graph).todense())

                    output_graph = cdt.utils.graph.remove_indirect_links(
                        output_graph, alg="aracne"
                    )
                    print(
                        "New skeleton (alg=aracne): ",
                        nx.adjacency_matrix(output_graph).todense(),
                    )

                    # nx.draw_networkx(copy.deepcopy(graph), ax=ax[0, idx], font_size=8, label='template')
                    # nx.draw_networkx(copy.deepcopy(output_graph), ax=ax[1, idx], font_size=8, label='output')
                    # idx += 1

            # plt.tight_layout()
            # plt.savefig('{}.png'.format(self.step_cnt))

            self.step_cnt += 1
            return self.action_space.sample()
        else:
            self.step_cnt += 1
            return self.action_space.sample()

    def reset(self) -> None:
        """Reset policy"""
        # self.model = cdt.causality.graph.GES()
        self.model = cdt.causality.pairwise.ANM()
        self.obs_buffer = np.empty((0, self.obs_shape))
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        self.step_cnt = 0
