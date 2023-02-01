import os
import time
import matplotlib.pyplot as plt
import glue

# Modifiable variables to weigh the cost of different resources
w_food = 1
w_water = 1
w_equipment = 1

'''
Game state
np.ndarray of type np.float32
[
 (for each vehicle, sorted in alphabetical order by id:)
 "Vehicle state":
      x,
      y,
      destination_x,
      destination_y,
      distance_traveled,
      water_amount,
      food_amount,
      equipment_amount,
 (for each field facility, sorted in alphabetical order by id:) 
"Field facility state":
      water_amount,
      food_amount,
      equipment_amount,
]
'''

num_vehicles = glue.num_vechicles
num_facilities = glue.num_facilities
desired_resources = glue.desired_resources #List[Dict]

def calcRes(State):
    '''
    Calculates the resource cost for a given game state
    
    Parameters
    ----------
    State : np.ndarray (game state)

    Returns
    -------
    Resource cost dictionary (Dict)
    '''
    gs_state = glue.state.decompose_game_state(State, num_vehicles, num_facilities)
    facilities = gs_state[1]
    res = {"TOTAL":0, "FOOD":0, "WATER":0, "EQUIPMENT":0, "MAX_RES":0}
    count = 0
    for facility in facilities:
        facility_dict = facility
        resources_dict = facility_dict["resource_levels"]
        for keys, values in resources_dict.items():
            if keys.name == 'FOOD':
                w = w_food
                cost = w * (min(values, desired_resources[count][keys]))
                res["FOOD"] += cost
                res["TOTAL"] += cost
                res["MAX_RES"] += w*desired_resources[count][keys]
            elif keys.name == 'WATER':
                w = w_water
                cost = w * (min(values, desired_resources[count][keys]))
                res["WATER"] += cost
                res["TOTAL"] += cost
                res["MAX_RES"] += w*desired_resources[count][keys]
            elif keys.name == 'EQUIPMENT':
                w = w_equipment
                cost = w * (min(values, desired_resources[count][keys]))
                res["EQUIPMENT"] += cost
                res["TOTAL"] += cost
                res["MAX_RES"] += w*desired_resources[count][keys]
        count += 1
    return res

def calcDistCost(State):
    '''
    Calculates the distance cost for a given game state
    
    Parameters
    ----------
    State : np.ndarray (game state)

    Returns
    -------
    Distance cost (float)
    '''
    vehicle_list = glue.state.decompose_game_state(State, num_vehicles, num_facilities)[0]
    cost = 0
    for vehicle in vehicle_list:
        vehicle_dict = vehicle
        cost += vehicle_dict["distance_traveled"]
    return cost

def facilitiesAtLevel(State):
    '''
    For a given game state, checks if all facilities have reached the desired resource level

    Parameters
    ----------
    State : np.ndarray (game state)

    Returns
    -------
    True if all facilities have reached the desired resource level, False otherwise (Bool)
    '''
    facilities = glue.state.decompose_game_state(State, num_vehicles, num_facilities)[1]
    print(facilities)
    count = 0
    for facility in facilities:
        facility_dict = facility
        resources_dict = facility_dict["resource_levels"]
        for keys, values in resources_dict.items():
            if values < desired_resources[count][keys]:
                return False
        count += 1
    return True

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