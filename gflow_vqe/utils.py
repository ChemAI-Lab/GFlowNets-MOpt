from gflownet.envs.graph_building_env import *
import random
from tequila.hamiltonian import QubitHamiltonian, paulis
from tequila.grouping.binary_rep import BinaryHamiltonian
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

def get_terms(bin_H):
    """ Gets the terms from a Binary Hamiltonian excluding the constant term"""
    terms=[]
    for i in range(1,len(bin_H.binary_terms)):
        terms.append(bin_H.binary_terms[i])  
    return terms

def FC_CompMatrix(terms):
    """ Generates the Fully commuting complementary matrix"""
    FC_CompMatrix=[]
    rows=len(terms)
    cols=len(terms)
    FC_CompMatrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if terms[i].commute(terms[j]):
                FC_CompMatrix[i][j]=1
                FC_CompMatrix[j][j]=1 #Avoids self loops
            else:
                FC_CompMatrix[i][j]=0
    return FC_CompMatrix

def QWC_CompMatrix(terms):
    """ Generates the Qubit-wise commuting complementary matrix"""
    QWC_CompMatrix=[]
    rows=len(terms)
    cols=len(terms)
    QWC_CompMatrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if terms[i].qubit_wise_commute(terms[j]):
                QWC_CompMatrix[i][j]=1
                QWC_CompMatrix[j][j]=1 #Avoids self loops
            else:
                QWC_CompMatrix[i][j]=0
    return QWC_CompMatrix

def obj_to_comp_graph(terms, matrix) -> Graph:
        """Convert a CommMatrix to a Graph"""
        g = Graph()
        matrix = matrix  # Make a copy
        terms = terms
        for a in range(len(terms)):
            g.add_node(
                a,
                v=terms[a].to_pauli_strings(),
            )
        for i in range(len(terms)):
            for j in range(i,len(terms)):
                if matrix[i][j] == 0:
                    g.add_edge(
                    i,
                    j,
                )
        return g

def graph_hash(graph):
    """Returns a binary hash for each submitted graph."""
    colors_dict = nx.get_node_attributes(graph, "color")
    FEATURE_KEYS = list(colors_dict.values())

    return tuple(FEATURE_KEYS)

def graph_to_tensor(graph, verbose=False):
  """Encodes a graph as a binary tensor (converted to float32)."""
  if verbose:
      print("graph={}, hash={}, tensor={}".format(
          graph,
          graph_hash(graph),
          torch.tensor(graph_hash(graph)).float(),
          )
      )
  return torch.tensor(graph_hash(graph)).float()#.to(device)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_not_valid(graph):
    """Checks whether the graph coloring is valid or not.
    Returns True if not valid, False if valid."""
    colors_dict = nx.get_node_attributes(graph, "color")
    colors_list = list(colors_dict.values())

    for i in range(nx.number_of_nodes(graph)):
        node_color = colors_list[i]
        # Check if any neighbor has the same color as the node
        same_color = any(node_color == colors_list[n] for n in graph.neighbors(i))

        # If any neighbor has the same color, the graph coloring is not valid
        if same_color:
            #print(f"Node {i} has the same color as at least one of its neighbors.")
            return True

    # If no conflicts were found, the graph coloring is valid
    #print("Graph coloring is valid. No neighboring nodes share the same color.")
    return False

def max_color(graph):
    colors_dict = nx.get_node_attributes(graph, "color")
    return np.max(list(colors_dict.values()))+1

def color_reward(graph):
  """Reward is based on the number of colors we have. The lower the better. Invalid configs give 0"""
  if is_not_valid(graph):
    return 0
  else:
    for i in range(nx.number_of_nodes(graph)+1):
        if max_color(graph) == i:
            return nx.number_of_nodes(graph)-i

def flow_matching_loss(incoming_flows, outgoing_flows, reward):
    """Flow matching objective converted into mean squared error loss."""
    return (incoming_flows.sum() - outgoing_flows.sum() - reward).pow(2)

def extract_hamiltonian_by_color(graph):
    """
    Extracts the hamiltonian groups based on the coloring of the graph.
    """	
    hamiltonian_terms = defaultdict(list)

    for node, data in graph.nodes(data=True):
        color = data['color']
        pauli_string = data['v']
        #print(type(pauli_string))
        hamiltonian_terms[color].append(pauli_string)

    return dict(hamiltonian_terms)

def generate_groups(hamiltonian_by_color):
    """
    This generates the groups from a colored Hamiltonian in an openfermion format
    """
    groups = []

    for color, pauli_strings in hamiltonian_by_color.items():
        # Create a QubitHamiltonian from the PauliString list
        qubit_hamiltonian = QubitHamiltonian.from_paulistrings(pauli_strings)
        # Convert to OpenFermion QubitOperator
        openfermion_operator = qubit_hamiltonian.to_openfermion()
        # Append to the groups list
        groups.append(openfermion_operator)

    return groups

def convert_operators(groups):
    """In the current implementation we need to convert operator to Pennylane 
    to estimate the total number of shots. """
    imported_operators = []

    for hamiltonian in groups:
        # Import the QubitOperator into PennyLane format
        operator = qml.import_operator(hamiltonian, format='openfermion')
        imported_operators.append(operator)

    return imported_operators

def estimate_shots(imported_operators):
    """
    Estimate the number of shots required based on a list of imported PennyLane operators.

    Parameters:
    - imported_operators (list of qml.PauliSum): List of PennyLane operators.

    Returns:
    - list of float: Estimated number of shots for each operator.
    """
    # Initialize list to hold estimated shots for each operator
    estimated_shots_list = []

    # Iterate over each imported operator
    for operator in imported_operators:
        # Extract coefficients and observables from each operator
        coeffs, ops = operator.terms()

        # Convert coefficients to numpy arrays for compatibility with PennyLane
        coeffs = [np.array(c) for c in coeffs]

        # Estimate shots for the current operator
        estimated_shots = qml.resource.estimate_shots(coeffs)

        # Append the result to the list
        estimated_shots_list.append(estimated_shots)

    return sum(estimated_shots_list)

def shots_estimator(graph):
    """
    FUll function that estimates the required number of shots to achieve chemical accuracy.
    Classifies the H by color, generates groups, translates to pennylane and estimates shots
    """
    h_by_color = extract_hamiltonian_by_color(graph)
    groups = generate_groups(h_by_color)
    imported_operators = convert_operators(groups)
    estimated_shots = estimate_shots(imported_operators)
    return estimated_shots

def vqe_reward(graph):
    """Reward is based on the number of colors we have. The lower cliques the better.
    Invalid configs give 0. Additionally, employs 10^6/Nshots to achieve chemical accuracy
    as reward function. The lower number of shots, the better."""
    reward=color_reward(graph) + 10**6/shots_estimator(graph)

    return reward

def shots_per_group(graph):
    """
    Estimate the number of shots required based on a list of imported PennyLane operators.

    """
    h_by_color = extract_hamiltonian_by_color(graph)
    groups = generate_groups(h_by_color)
    imported_operators = convert_operators(groups)
    # Initialize list to hold estimated shots for each operator
    estimated_shots_list = []

    # Iterate over each imported operator
    for operator in imported_operators:
        # Extract coefficients and observables from each operator
        coeffs, ops = operator.terms()

        # Convert coefficients to numpy arrays for compatibility with PennyLane
        coeffs = [np.array(c) for c in coeffs]

        # Estimate shots for the current operator
        estimated_shots = qml.resource.estimate_shots(coeffs)

        # Append the result to the list
        estimated_shots_list.append(estimated_shots)

    return estimated_shots_list

def graph_parents(state):
    parent_states = []  # States that are parents of state.
    parent_actions = []  # Actions that lead from those parents to state.
    colors_dict = nx.get_node_attributes(state, "color")
    daddy_state=state.copy()
    for node, color in colors_dict.items():
        action={}
        if node != color:
            parent_dict=colors_dict.copy()
            parent_dict[node] = node
            #parent_states.append(parent_dict) #Potentially useful if we want to use only dicts
            nx.set_node_attributes(daddy_state, parent_dict, "color")
            parent_states.append(daddy_state)
            action[node] = colors_dict[node]
            parent_actions.append(tuple(list(action.values())))
        else: #This part allows to use as initial state a colored graph which the color is the same as the node
            parent_states.append(daddy_state)
            action[node] = colors_dict[node]
            parent_actions.append(tuple(list(action.values())))
    return parent_states, parent_actions

def graph_parents_precolored(state):
    parent_states = []  # States that are parents of state.
    parent_actions = []  # Actions that lead from those parents to state.
    colors_dict = nx.get_node_attributes(state, "color")
    daddy_state=state.copy()
    for node, color in colors_dict.items():
        action={}
        if node != color:
            parent_dict=colors_dict.copy()
            parent_dict[node] = node
            #parent_states.append(parent_dict) #Potentially useful if we want to use only dicts
            nx.set_node_attributes(daddy_state, parent_dict, "color")
            parent_states.append(daddy_state)
            action[node] = colors_dict[node]
            parent_actions.append(tuple(list(action.values())))
    return parent_states, parent_actions
