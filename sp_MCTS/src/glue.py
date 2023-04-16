# from run_search import simulation
import numpy as np
import copy
import numpy as np
import itertools

from causal_env_v0 import CausalEnv_v0
import yaml, pickle
from easydict import EasyDict


# Load facilities
with open('config/env_config.yaml', "r") as f:
        env_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        ldict = {}
        exec(env_config.hypotheses, globals(), ldict)
        env_config.hypotheses = ldict["hypotheses"]

env_config = env_config
env = CausalEnv_v0(env_config)

# ------------------------------------------------------------------------------------------------------------------------------

# Simulator API
def GetNextState(CurrState):
    '''
    Returns the next state of the simulation/rollout step in MCTS
    
    Parameters
    ----------
    CurrState : game state

    Returns
    -------
    NextState : game state
    '''

    if CurrState == 'Disj':
        possibleStates = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif CurrState == 'Conj':
        possibleStates = [[1, 1, 0], [1, 0, 1]]
    elif CurrState == 'All':
        possibleStates = [[1, 1, 1]]
    elif CurrState == 'Null':
        possibleStates = [[0, 0, 0]]
    
    i = np.random.randint(0, len(possibleStates))
    NextState = possibleStates[i]
    obs, reward, _, _ = env.step(NextState)
    return obs[-1]

def EvalNextStates(CurrState):
    '''
    Returns the next children states from a given unexpanded node state in the expansion step of MCTS

    Parameters
    ----------
    CurrState : game state (unexpanded node)

    Returns
    -------
    NextStates : list of game states
    '''
    print("\n Evaluating the new states \n")
    State = copy.deepcopy(CurrState)

    childStates = []

    if len(State) == 4: # Root State
        possibleStates = ['Disj', 'Conj', 'All', 'Null']
        for i in possibleStates:
            childStates.append([i])

    else: # Non-Root State - Second Level:
        if State == 'Disj':
            possibleStates = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif State == 'Conj':
            possibleStates = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        elif State == 'All':
            possibleStates = [[1, 1, 1]]
        elif State == 'Null':
            possibleStates = [[0, 0, 0]]
        
        for i in possibleStates:
            childStates.append(i)
    
    return childStates
        