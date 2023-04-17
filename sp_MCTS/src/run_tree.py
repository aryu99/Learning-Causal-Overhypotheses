import copy
import pickle as pkl

from node import Node
# from MCTS import MCTS
import glue
# import utils
from causal_env_v0 import CausalEnv_v0
import yaml, pickle
from easydict import EasyDict

def setup_tree():
    obs = glue.env.reset()
    print(obs)

    root_state = copy.deepcopy(obs)
    root = Node(root_state)
    root.visits = 1

    # sol = MCTS(root)


if __name__ == "__main__":
    setup_tree()