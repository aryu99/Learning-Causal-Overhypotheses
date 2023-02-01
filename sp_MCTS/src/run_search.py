import copy
import pickle as pkl

from node import Node
from MCTS import MCTS
import glue
import utils

# Modifiable variables for MCTS
MaxIteration = 10 #maximum number of iterations for selecting one action
numActions = 50 # number of actions to be selected

def load_tree(filename):
    '''
    Starter code for MCTS with a loaded tree
    
    Parameters
    ----------
    filename : str
    '''
    with open(filename, 'rb') as f:
        sol = pkl.load(f)
    time = utils.timer()
    sol.Run(MaxIteration, numActions, del_children=True, limit_del = True, clear_root=False, load=True, time_thresh=1200)
    end_time = utils.timer()
    print("Time taken: ", end_time - time)
    gameStates = sol.storeGameStates
    glue.run_visualizer(gameStates, save_to_file=True, saveGS = True)


def start_mcts():
    '''
    Starter code for MCTS with a new tree
    '''
    root_state = copy.deepcopy(glue.get_sim_state())
    root = Node(root_state)
    root.visits = 1

    sol = MCTS(root)
    time = utils.timer()
    sol.Run(MaxIteration, numActions, del_children=True, limit_del = True, clear_root=False, time_thresh=30)
    end_time = utils.timer()
    print("Time taken: ", end_time - time)
    gameStates = sol.storeGameStates
    glue.run_visualizer(gameStates, save_to_file=True, saveGS = True)

if __name__ == "__main__":

    profiling_flag = input("Do you want to profile the code? (y/n): ")
    if profiling_flag == 'y':
        import cProfile
        from pstats import Stats

        pr = cProfile.Profile()
        pr.enable()
    load_flag = input("Load Tree? (y/n): ")
    if load_flag == 'y':
        filename = input("Enter filename: ")
        load_tree(filename)
    else:
        start_mcts()

    if profiling_flag == 'y':
        pr.disable()
        stats = Stats(pr)
        stats.sort_stats('tottime').print_stats(20)
        
    
    

    

