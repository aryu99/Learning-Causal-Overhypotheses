import os
import time
import matplotlib.pyplot as plt
import glue

def calc_result(obs):

    num_blickets = 0
    for blicket in obs[:3]:
        if blicket == 1:
            num_blickets += 1
    
    if obs[-1] == 1:
        result = 1/num_blickets
    
    else:
        result = 0

    return result


def SaveResult(Text, name="final_result"):
    '''
    Saves the result of the game to a text file

    Parameters
    ----------
    Text : whatever you want to save, gets converted to string within this function (for example, result of the game)
    name : String (name of the file)
    '''
    filename = name + ".txt"
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    f = open(filename, append_write)
    f.write(str(Text) + '\n')
    f.close()
    # pass

def timer():
    '''
    Returns the current time in seconds

    Returns
    -------
    Current time in seconds (float)
    '''
    curr_time = time.time()
    return curr_time


def plotter(x, total_cost, dist_cost=None, res_cost=None, food_cost=None, water_cost=None, equip_cost=None,
            max_cost=None, name="plot"):
    '''
    Plots the total cost, distance cost, resource cost, food cost, water cost, equipment cost and max cost
    for a given game state

    Parameters
    ----------
    x : list of x values (list)
    total_cost : list of total cost values (list)
    dist_cost : list of distance cost values (list)
    res_cost : list of resource cost values (list)
    food_cost : list of food cost values (list)
    water_cost : list of water cost values (list)
    equip_cost : list of equipment cost values (list)
    max_cost : list of max cost values (list)
    name : String (name of the file)
    '''
    
    print("Plotting")
    print(x)
    print(total_cost)
    print(dist_cost)
    print(res_cost)
    fig1 = plt.figure("Figure 1")
    plt.plot(x, total_cost, label = "Cummulative cost")
    if dist_cost is not None:
        plt.plot(x, dist_cost, label = "Distance Cost")
    if res_cost is not None:
        plt.plot(x, res_cost, label = "Resources Cost")
    if food_cost is not None:
        plt.plot(x, food_cost, label = "Food Cost")
    if water_cost is not None:
        plt.plot(x, water_cost, label = "Water Cost")
    if equip_cost is not None:
        plt.plot(x, equip_cost, label = "Equipment Cost")
    if max_cost is not None:
        plt.hlines(y=[max_cost], xmin=[0], xmax=[len(x)], colors='purple', linestyles='--', lw=2, label='Max Cummulative Cost')
    plt.xlabel('Action Number')
    plt.ylabel('Cost')
    plt.title('Action No. vs Cost')
    plt.legend()
    plt.savefig(name + ".png")
    plt.show()

    pass