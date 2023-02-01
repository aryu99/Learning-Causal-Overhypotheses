#------------------------------------------------------------------------#
#
# Written by sergeim19 (Created June 21, 2017), modified by Aryaman Samyal
# https://github.com/sergeim19/

# https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
#
#------------------------------------------------------------------------#
import copy
import numpy as np
import os
import pickle as pkl
# from numba import jit

import node as nd
import glue

import utils

#------------------------------------------------------------------------#
# Class for Single Player Monte Carlo Tree Search implementation.
#------------------------------------------------------------------------#


class MCTS:

    # Modifiable variable that detemines the number of rollouts to be performed (and averaged) in the simulation step of MCTS.
    num_sim = 10
    #-----------------------------------------------------------------------#
    # Description: Constructor.
    # Node 	  - Root node of the tree of class Node.
    # Verbose - True: Print details of search during execution.
    # 			False: Otherwise
    #-----------------------------------------------------------------------#
    def __init__(self, Node, Verbose=False):
        self.root = Node
        self.verbose = Verbose

    #-----------------------------------------------------------------------#
    # Description: Performs selection phase of the MCTS.
    #-----------------------------------------------------------------------#
    def Selection(self):
        '''
        Selection phase of MCTS

        Returns
        -------
        Node
            Selected Node.
        '''

        print("\n ---SELECTION--- \n")
        SelectedChild = self.root
        HasChild = False

        # Check if child nodes exist.
        if (len(SelectedChild.children) > 0):
            HasChild = True
        else:
            HasChild = False
        while (HasChild):
            SelectedChild = self.SelectChild(SelectedChild)
            if (len(SelectedChild.children) == 0):
                HasChild = False
            # SelectedChild.visits += 1.0

        if (self.verbose):
            print("\nSelected: ", nd.Node.GetStateRepresentation(SelectedChild))

        return SelectedChild

    #-----------------------------------------------------------------------#
    # Description:
    #	Given a Node, selects the first unvisited child Node, or if all
    # 	children are visited, selects the Node with greatest UTC value.
    # Node	- Node from which to select child Node from.
    #-----------------------------------------------------------------------#
    # @jit(target_backend='cuda')  
    def SelectChild(self, Node):
        '''
        Given a Node, selects the first unvisited child Node, or if all
        children are visited, selects the Node with greatest UTC value.

        Parameters
        ----------
        Node : Node
            Node from which to select child Node from.
        
        Returns
        -------
        Node
            Selected child Node.
        '''
        if (len(Node.children) == 0):
            return Node

        for Child in Node.children:
            if (Child.visits > 0.0):
                continue
            else:
                if (self.verbose):
                    print("Considered child", nd.Node.GetStateRepresentation(
                        Child.state), "UTC: inf",)
                return Child

        MaxWeight = -100000.0
        for Child in Node.children:
            Weight = Child.sputc
            if (self.verbose):
                print("Considered child:", nd.Node.GetStateRepresentation(
                    Child.state), "UTC:", Weight)
            if (Weight >= MaxWeight):
                MaxWeight = Weight
                SelectedChild = Child
        return SelectedChild

    #-----------------------------------------------------------------------#
    # Description: Performs expansion phase of the MCTS.
    # Leaf	- Leaf Node to expand.
    #-----------------------------------------------------------------------#
    def Expansion(self, Leaf):
        '''
        Performs expansion phase of the MCTS.
        
        Parameters
        ----------
        Leaf : Node
            Leaf Node to expand.
        
        Returns
        -------
        Node
            Expanded Node's selected child.
        '''
        print("\n ---EXPANSION--- \n")
        if (self.IsTerminal((Leaf))):
            print("Is Terminal.")
            return False
        elif (Leaf.visits == 0):
            return Leaf
        else:
            # Expand.
            print("\n ---EXPANDING--- \n")
            if (len(Leaf.children) == 0):
                Children = self.EvalChildren(Leaf)
                for NewChild in Children:
                    if (np.all(NewChild.state == Leaf.state)):
                        continue
                    Leaf.AppendChild(NewChild)
            assert (len(Leaf.children) > 0), "Error"
            Child = self.SelectChildNode(Leaf)
        if (self.verbose):
            print("Expanded: ", nd.Node.GetStateRepresentation(Child.state))
        return Child

    #-----------------------------------------------------------------------#
    # Description: Checks if a Node is terminal (it has no more children).
    # Node	- Node to check.
    #-----------------------------------------------------------------------#
    def IsTerminal(self, Node):
        '''
        Checks if a Node is terminal (leaf node) (it has no more children).

        Parameters
        ----------
        Node : Node
            Node to check.
        
        Returns
        -------
        bool
            True if Node is terminal, False otherwise.
        '''    
        if (nd.Node.IsNodeTerminal(Node)):
            return True
        else:
            return False

    #-----------------------------------------------------------------------#
    # Description:
    #	Evaluates all the possible children states given a Node state
    #	and returns the possible children Nodes.
    # Node	- Node from which to evaluate children.
    #-----------------------------------------------------------------------#
    def EvalChildren(self, Node):
        '''
        Evaluates all the possible children states given a Node state
        and returns the possible children Nodes.

        Parameters
        ----------
        Node : Node
            Node from which to evaluate children.
        
        Returns
        -------
        Children : list 
            List of children Nodes.
        '''
        NextStates = glue.EvalNextStates(Node.state)
        Children = []
        for State in NextStates:            
            ChildNode = nd.Node(State)
            Children.append(ChildNode)
        return Children

    #-----------------------------------------------------------------------#
    # Description:
    #	Selects a child node randomly.
    # Node	- Node from which to select a random child.
    #-----------------------------------------------------------------------#
    def SelectChildNode(self, Node):
        '''
        Randomly selects a child node.
        
        Parameters
        ----------
        Node : Node
            Node from which to select a random child.
        
        Returns
        -------
        Node
            Randomly selected child node.
        '''
        Len = len(Node.children)
        assert Len > 0, "Incorrect length"
        i = np.random.randint(0, Len)
        return Node.children[i]

    #-----------------------------------------------------------------------#
    # Description:
    #	Performs the simulation phase of the MCTS.
    # Node	- Node from which to perform simulation.
    #-----------------------------------------------------------------------#
    def Simulation(self, Node):
        '''
        Performs the simulation phase of the MCTS. Currently setup so that multiple simulations are done (according to num_sim) 
        and the average score is returned.
        
        Parameters
        ----------
        Node : Node
            Node from which to perform simulation.
        
        Returns
        -------
        float
            Average score/result of the simulations.
        '''
        num_sim = MCTS.num_sim
        i = 0
        simResult = 0
        while i<num_sim:
            CurrentState = copy.deepcopy(Node.state)
            print ("\n ---SIMULATION--- \n")
            if (self.verbose):
                print ("Begin Simulation")

            relativeLevel = 1
            Result = nd.Node.GetResult(utils.calcRes(CurrentState)["TOTAL"], utils.calcDistCost(CurrentState))
            # Perform simulation.
            while (nd.Node.IsTerminal(relativeLevel)):
                relativeLevel += 1.0
                CurrentState = glue.GetNextState(CurrentState)
                Result += nd.Node.GetResult(utils.calcRes(CurrentState)["TOTAL"], utils.calcDistCost(CurrentState))
                if (self.verbose):
                    print ("CurrentState:", nd.Node.GetStateRepresentation(CurrentState))
                    nd.Node.PrintTablesScores(CurrentState)

            Result = nd.Node.GetResult(utils.calcRes(CurrentState)["TOTAL"], utils.calcDistCost(CurrentState))
            simResult += Result
            i += 1
        return simResult/num_sim

    #-----------------------------------------------------------------------#
    # Description:
    #	Performs the backpropagation phase of the MCTS.
    # Node		- Node from which to perform Backpropagation.
    # Result	- Result of the simulation performed at Node.
    #-----------------------------------------------------------------------#
    def Backpropagation(self, Node, Result): 
        '''
        The backpropagation phase of the MCTS.

        Parameters
        ----------
        Node : Node
            Node from which to perform Backpropagation.
        Result : float
            Result of the simulation performed at Node.
        
        Returns
        -------
        None.
        '''
        print("\n ---BACKPROPAGATION--- \n")
        # Update Node's weight.
        CurrentNode = Node
        CurrentNode.visits += 1

        CurrentNode.val += Result
        self.EvalUTC(CurrentNode)

        while (self.HasParent(CurrentNode)):
            # Update parent node's weight.
            CurrentNode = CurrentNode.parent
            CurrentNode.visits += 1
            CurrentNode.val += Result
            self.EvalUTC(CurrentNode)


    #-----------------------------------------------------------------------#
    # Description:
    #	Checks if Node has a parent..
    # Node - Node to check.
    #-----------------------------------------------------------------------#
    def HasParent(self, Node):
        '''
        Checks if Node has a parent.
        
        Parameters
        ----------
        Node : Node
            Node to check.
        
        Returns
        -------
        bool
            True if Node has a parent, False otherwise.
        '''
        if (Node.parent == None):
            return False
        else:
            return True

    #-----------------------------------------------------------------------#
    # Description:
    #	Evaluates the Single Player modified UTC. See:
    #	https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
    # Node - Node to evaluate.
    #-----------------------------------------------------------------------#
    def EvalUTC(self, Node):
        '''
        Evaluates the UTC of a Node.
        
        Parameters
        ----------
        Node : Node
            Node to evaluate.
    '''
        c = 0.5
        n = Node.visits
        if (Node.parent == None):
            t = Node.visits
        else:
            t = Node.parent.visits

        Modification = Node.val
        UTC = Modification/n + c * np.sqrt(np.log(t)/n)

        Node.sputc = UTC
        return Node.sputc

    #-----------------------------------------------------------------------#
    # Description:
    #	Gets the level of the node in the tree.
    # Node - Node to evaluate the level.
    #-----------------------------------------------------------------------#
    def GetLevel(self, Node):
        '''
        Gets the level of the node in the tree.
        
        Parameters
        ----------
        Node : Node
            Node to evaluate the level.
        
        Returns
        -------
        int
            Level of the node in the tree.
        '''
        Level = 0.0
        while (Node.parent):
            Level += 1.0
            Node = Node.parent
        return Level

    #-----------------------------------------------------------------------#
    # Description:
    #	Prints the tree to file.
    #-----------------------------------------------------------------------#
    def PrintTree(self):
        f = open('Tree.txt', 'w')
        Node = self.root
        self.PrintNode(f, Node, "", False)
        f.close()

    #-----------------------------------------------------------------------#
    # Description:
    #	Prints the tree Node and its details to file.
    # Node			- Node to print.
    # Indent		- Indent character.
    # IsTerminal	- True: Node is terminal. False: Otherwise.
    #-----------------------------------------------------------------------#
    def PrintNode(self, file, Node, Indent, IsTerminal):
        file.write(Indent)
        if (IsTerminal):
            file.write("\-")
            Indent += "  "
        else:
            file.write("|-")
            Indent += "| "

        string = str(self.GetLevel(Node)) + ") (["
        # for i in Node.state.bins: # game specific (scrap)
        # 	string += str(i) + ", "
        string += str(nd.Node.GetStateRepresentation(Node.state))
        string += "], W: " + str(Node.wins) + ", N: " + \
            str(Node.visits) + ", UTC: " + str(Node.sputc) + ") \n"
        file.write(string)

        for Child in Node.children:
            self.PrintNode(file, Child, Indent, self.IsTerminal(Child))

    def PrintResult(self, Result):
        filename = 'Results.txt'
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        f = open(filename, append_write)
        f.write(str(Result) + '\n')
        f.close()

    def delete_children(self, Node, level):
        '''
        Deletes all children of a node after a certain level
        
        Parameters
        ----------
        Node : Node
            Node to delete children from.
        level : int
            Level to delete children from.
        '''
        try:
            if self.GetLevel(Node.children[0]) > level:
                Node.children = []
            else:
                for child in Node.children:
                    self.delete_children(child, level)
        except:
            pass
    
    def BestChild(self):
        '''
        Returns the best child of the root node
        
        Returns:
            Node: Best child of the root node
        '''
        Node = self.root
        MaxWeight = -100000000000.0
        if (len(Node.children) == 0):
            return Node
        for Child in Node.children: 
            Weight = Child.val/Child.visits            
            if (self.verbose):
                print("Considered child:", nd.Node.GetStateRepresentation(
                    Child.state), "UTC:", Weight)
            if (Weight >= MaxWeight):
                MaxWeight = Weight
                SelectedChild = Child
        return SelectedChild

    def save_tree(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self, f)

    #-----------------------------------------------------------------------#
    # Description:
    #	Runs the SP-MCTS.
    # MaxIter	- Maximum iterations to run the search algorithm.
    #-----------------------------------------------------------------------#
    def Run(self, MaxIteration=5, numActions=1, del_children=False, limit_del = True, clear_root = False, load = False, time_thresh = 1200):
        if load == False:
            # This handles the case where the tree is not loaded from a file
            self.storeGameStates = [self.root.state]
            self.counter = 0
            self.MaxIter = MaxIteration
            self.store_cost = []
            self.store_dist = []
            self.store_res = []
            self.store_res_food = []
            self.store_res_water = []
            self.store_res_equip = []
            self.store_action_id = []
            self.action_timeThresh = time_thresh/numActions
        while (self.counter < numActions):
            # Loop for each action
            start_time = utils.timer()
            for i in range(int(self.MaxIter)):
                # Loop for each iteration for an action
                print ("\n===== Begin iteration: {} for Action No. {} =====".format(i, self.counter))
                if (self.verbose):
                    print ("\n===== Begin iteration:", i, "=====")
                X = self.Selection()

                Y = self.Expansion(X)
                if (Y):
                    Result = self.Simulation(Y)
                    if (self.verbose):
                        print ("Result: ", Result)
                    self.Backpropagation(Y, Result)
                    root = copy.deepcopy(self.root)
                    while root.children:
                        level = 0
                        child_idx = 0
                        for child in root.children:
                            child_idx += 1
                        child_idx = 0
                        level += 1
                        root = child
                
                # This is if we reach terminal state
                else:
                    Result = nd.Node.GetResult(utils.calcRes(X.state)["TOTAL"], utils.calcDistCost(X.state))
                    if (self.verbose):
                        print ("Result: ", Result)
                    self.Backpropagation(X, Result)

                curr_time = utils.timer()

                # stop loop if time exceeds threshold
                if (curr_time - start_time) > self.action_timeThresh:
                    break   
                    
            #     self.PrintResult(Result)
            # self.PrintTree()
            print ("Search complete.")
            print ("Iterations:", i)

            
            self.storeGameStates.append(self.BestChild().state) # Get the best child and append it to the list of game states
            self.root = self.BestChild() # Set the root to the best child

            # Handle storage for plotting
            self.store_cost.append(nd.Node.GetResult(utils.calcRes(self.root.state)["TOTAL"], utils.calcDistCost(self.root.state)))
            self.store_dist.append(nd.Node.w_dist*utils.calcDistCost(self.root.state))
            self.store_res.append(nd.Node.w_res*utils.calcRes(self.root.state)["TOTAL"])
            self.store_res_food.append(utils.calcRes(self.root.state)["FOOD"])
            self.store_res_water.append(utils.calcRes(self.root.state)["WATER"])
            self.store_res_equip.append(utils.calcRes(self.root.state)["EQUIPMENT"])
            self.store_action_id.append(self.counter)
            self.counter += 1

            # delete tree from memory
            print('Clearing memory by deleting the parents')
            if clear_root:
                self.root.visits = 0
                self.root.sputc = 0
            if del_children and limit_del:
                self.delete_children(self.root, numActions+1)
            elif del_children and not limit_del:
                self.root.children = []
            self.root.parent = None
            print('root visited: ', self.root.visits)

            # Handle number of iteration for next action (MaxIteration - number of times that node has been visited)
            if not clear_root:
                self.MaxIter = MaxIteration - self.root.visits

            self.save_tree('tree.pkl') # Save the tree to a pickle file
        
        # Plot the results
        utils.plotter(self.store_action_id, self.store_cost, self.store_dist, self.store_res, self.store_res_food, self.store_res_water, self.store_res_equip,
        max_cost=utils.calcRes(self.root.state)["MAX_RES"], name='cost_components_long_different_res_weights')
        count = 0
        for i in self.storeGameStates:
            utils.SaveResult('\n State: {}'.format(i))
            count += 1
            print("\n Action State {}: {}".format(count, glue.state.decompose_game_state(i, glue.num_vechicles, glue.num_facilities)))
