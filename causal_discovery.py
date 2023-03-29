import yaml, easydict, argparse
import numpy as np
import pandas as pd
import networkx as nx
import cdt
import matplotlib.pyplot as plt

from causal_env_v0 import CausalEnv_v0
from hypotheses import *
from policy import RandomPolicy
from utils import collect_observational_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Causal Discovery")
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
    nx.draw_networkx(output_graph, font_size=8)
    plt.tight_layout()
    plt.savefig('GES.png', dpi=300, bbox_inches='tight')