# core
MCTS backend core

This repository contains files for the MCTS solution core. 

It consists of 5 primary files:

1. [`run_search.py`](src/run_search.py) is main starter file of MCTS. In this file you can modify and define the `MaxIteration` variable that determines the number of iterations to do for selection an action. Also, the variable `numActions` can be modified depending on the number of Actions you want MCTS to produce.

There are two functions in this file:

a. `load_tree` runs MCTS by loading a previously saved Tree, ad continues the iteration.

b. `start_mcts` start MCTS fresh.

Further, this file calls [`glue.py`](src/glue.py) for initializing the simulation, running the visualizer, as well as saving the action game states into a pickle file.

2. [`glue.py`](src/glue.py) handles interaction with the simulation. It initializes the simulation and the visualizer, as well as provides an API for MCTS to interact with the simulation using two functions:

a. `GetNextState()` takes in a state, queries the simulator for possible actions from the argument state, takes a random action and then return the next state. This function is called in the simulation/rollout phase of MCTS.

b. `EvalNextStates()` takes in a state, queries the simulator for possible actions from the argument state and queries the simulator to take all possible combinations of actions for all vehicles, thereby, producing the list of all possible next states for the argument state.

3. [`utils.py`](src/utils.py) contains complementary functions for MCTS, as well as utility functions. The list of functions are as follows:
a. `calcRes()` calculates the resource cost for a given state. This functions returns a `dict` that contains information about the cummulative levels of each resource on all islands, the desired resource levels, and the total resource cost which is a summation of all resources on all islands. You can modify weights for indiviual resource by modifying these [lines](https://github.com/droneconia/core/blob/67e6ace02271b51e9d00e7daec49ddaedf35b45d/src/utils.py#L7-L9).

b. `calcDistCost()` calculates the distance cost by summing the distance travelled for all vehicles.

c. `facilitiesAtLevel()` checks if all facilities have reached their desired levels or not for a state (whether the state is terminal or not). Returns a `bool`.

d. `SaveResult()` saves whatever you want to save in a text file.

e. `timer()` function to keep track of time

f. `plotter` plots results after MCTS is done.

4. [`node.py`](src/node.py) contains the node structure the MCTS tree. Modifiable parameters for controlling the weight of resource, distance cost during total cost calculation and the level till which simulation need to be performed is given by these [lines](https://github.com/droneconia/core/blob/67e6ace02271b51e9d00e7daec49ddaedf35b45d/src/node.py#L4-L6).  The primary methods for the object are:
a. `GetLevel()` Calculates the tree depth level of the node (called when deleting children)

b. `AppendChild()` appends a child node to the node's `children` attribute.

c. `IsNodeTerminal()` checks if the state is a terminal state (all resources at desired value).

d. `IsTerminal()` checks if the simulation rollout has reaches the set level or not.

e. `GetResult()` multiplies the resource and distance costs by their weights and then calculates the cummulative cost with both the terms included.

5. [`MCTS.py`](src/MCTS.py) contains the implementation of the MCTS object. The starter method that initiates all the steps/function of the algorithm is [`Run`](https://github.com/droneconia/core/blob/67e6ace02271b51e9d00e7daec49ddaedf35b45d/src/MCTS.py#L488). This method handles the execution, i.e. the number of iteration, number of actions, time allocated for each action. This method performs MCTS by calling `Selection`, `Expansion`, `Simuation` and `Backpropagation` methods, which have been defined in the same file. In [line](https://github.com/droneconia/core/blob/67e6ace02271b51e9d00e7daec49ddaedf35b45d/src/MCTS.py#L561), the method handles deletion of data and children for the next iteration. This method is reponsible for storing information and then plotting them (plotter called [here](https://github.com/droneconia/core/blob/67e6ace02271b51e9d00e7daec49ddaedf35b45d/src/MCTS.py#L579))