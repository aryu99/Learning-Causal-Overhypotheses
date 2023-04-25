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
    print("\n Getting the next state \n")
    print("Current State: ", CurrState)

    CurrentState = copy.deepcopy(CurrState[0])


    if CurrentState == 'Disj':
        possibleStates = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif CurrentState == 'Conj':
        possibleStates = [[1, 1, 0], [1, 0, 1]]
    elif CurrentState == 'All':
        possibleStates = [[1, 1, 1]]
    elif CurrentState == 'Null':
        possibleStates = [[0, 0, 0]]

    print("Possible States: ", possibleStates)
    
    i = np.random.randint(0, len(possibleStates))
    NextState = possibleStates[i]
    return NextState

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
        if State == ['Disj']:
            possibleStates = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif State == ['Conj']:
            possibleStates = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        elif State == ['All']:
            possibleStates = [[1, 1, 1]]
        elif State == ['Null']:
            possibleStates = [[0, 0, 0]]
        
        for i in possibleStates:
            childStates.append(i)
    
    return childStates


def calcResult(action):

    # Get the result of the action
    print("Action: ", action)
    obs, _, _, _ = env.step(action)
    print("Observation: ", obs)
    result = obs[-1]
    print("Result: ", result)
    return result