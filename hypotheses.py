import numpy as np

A = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

B = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

C = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)


AB = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

BC = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

CA = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

ABC = np.array(
    [[0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
)

N = np.array(
    [[0, 0, 0, 0],
	 [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)

SINGLE = [A, B, C]
DOUBLE = [AB, BC, CA]
NONE = [N]
ALL = [ABC]

HYPS = {
    'A': A,
    'B': B,
    'C': C,
    'AB': AB,
    'BC': BC,
    'AC': CA,
    'CA': CA,
    'ABC': ABC,
}

def gen_rand_hypotheses_structures(num_nodes: int, num_hypotheses: int = 1) -> list:
    """Generate random [num_nodes x num_nodes] adjacency graphs representing
    hypotheses structures, 0's on & below diagonal

    Args:
        num_nodes (int): Number of nodes
        num_hypotheses (int, optional): Number of hypotheses to generate.
            Defaults to 1.

    Returns:
        list: List of hypotheses
    """
    return [
        np.triu(np.random.randint(2, size=(num_nodes, num_nodes)), 1) for _ in range(num_hypotheses)
    ]
