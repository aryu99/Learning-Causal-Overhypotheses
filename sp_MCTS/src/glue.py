# from run_search import simulation
import numpy as np
import copy
import numpy as np
import itertools

import sys
sys.path.append('../../simulation/src')

from load_config import load_config_files
import state as state

# Sim Initialization

SIM_CONFIG_FILE = 'basecase.yaml'
VIZ_CONFIG_FILE = 'basecase_viz.yaml'

# Load facilities
(
    simulation,
    visualizer,
    vehicles, # vehicles
    facilities,
    _, # sim_config
    _, # viz_config
) = load_config_files(SIM_CONFIG_FILE, VIZ_CONFIG_FILE)

num_vechicles = len(vehicles)
num_facilities = len(facilities) - 1
desired_resources = [facility.desired_resource_levels for facility in facilities if facility.identifier != 'SupplyDepot1']

def get_sim_state():
    '''
    Returns the current state of the simulation
    '''
    print("\n Getting the current state \n")
    return simulation.get_game_state()

def run_visualizer(game_states, save_to_file=False, saveGS = False):
    '''
    Runs the visualizer on the given game states

    Parameters
    ----------
    game_states : list of game states
        The game states to visualize
    save_to_file : bool
        Whether to save the visualization to a file
    saveGS : bool
        Whether to save the game states to a pickle file
    '''
    if saveGS:
        simulation.save_to_file('test_0.pkl', gs_list = game_states)
    print("\n Running the visualizer \n")
    # visualizer.run(game_states, save_to_file=save_to_file, dir_name = 'W:/Droneconia/simulation/images/test1')

def run_save(game_states):
    simulation.save_to_file('test.pkl', gs_list = game_states)
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
    simulation.reset(CurrState)
    Actions = simulation.get_vehicle_actions()
    Action = {}

    # Get a random action for each vehicle
    for key,value in Actions.items():
        i = np.random.randint(0, len(value))
        Action[key] = value[i]
    
    # Get the next state
    NextState = simulation.next_state(Action)  

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
    
    simulation.reset(State)
    Actions = simulation.get_vehicle_actions()

    # Get all possible combinations of actions for all vehicles
    storeActions = []    
    jointActionList = list(Actions.values())
    combinatonList = [p for p in itertools.product(*jointActionList)]
    vehicleList = list(Actions.keys())    
    
    for i in range(len(combinatonList)):
        storeAction = {}
        for j in range(len(vehicleList)):        
            storeAction[vehicleList[j]] = combinatonList[i][j]
        storeActions.append(storeAction)

    # Get the next state for each combination of actions
    NextStates = []
    for m in range(len(storeActions)):
        simulation.reset(State)        
        NextStates.append(simulation.next_state(storeActions[m])) 

    return NextStates