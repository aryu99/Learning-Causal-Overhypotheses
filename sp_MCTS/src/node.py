import utils

class Node:
    w_res = 1
    w_dist = 1
    levelTerminal = 20

    def __init__(self, state):
        self.state = state
        self.visits = 0.0
        self.parent = None
        self.children = []
        self.sputc = 0.0

        self.val = 0.0

    def AppendChild(self, child):
        '''
        Appends a child to the node

        Parameters
        ----------
        child : Node to be appended as a child.
        
        Returns
        -------
        None
        '''
        self.children.append(child)
        child.parent = self

    @staticmethod
    def GetLevel(node):
        '''
        Returns the level of a given node
        
        Parameters
        ----------
        node : Node
        
        Returns
        -------
        Level of the node (int)
        '''
        Level = 0.0
        while (node.parent):
            Level += 1.0
            node = node.parent
        return Level

    @staticmethod
    def IsNodeTerminal(node):
        '''
        Checks if a given node is terminal

        Parameters
        ----------
        node : Node
        
        Returns
        -------
        True if the node is terminal, False otherwise (bool)
        '''
        return utils.facilitiesAtLevel(node.state)

    @staticmethod
    def IsTerminal(level:int):
        '''
        This checks if the level is terminal during the simulation phase of the MCTS algorithm.

        Parameters
        ----------
        level : current level of the simulation (int)

        Returns
        -------
        True if the level is terminal, False otherwise (bool)
        '''
        return level < Node.levelTerminal

    @staticmethod
    def GetResult(resCost, distCost):
        '''
        Returns the result of the simulation

        Parameters
        ----------
        resCost : cost of resources (float)
        distCost : cost of distance (float)

        Returns
        -------
        Result of the simulation (float)
        '''
        return Node.w_res*resCost - Node.w_dist*distCost


    @staticmethod
    def GetStateRepresentation(node):
        '''
        Returns the state representation of a given node

        Parameters
        ----------
        node : Node

        Returns
        -------
        State representation of the node (np.ndarray)
        '''
        return node.state
    

