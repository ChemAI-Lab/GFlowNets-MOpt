from tqdm import tqdm, trange
from torch.distributions.categorical import Categorical
from gflow_vqe.utils import *
from gflow_vqe.gflow_utils import *

def precolored_flow_match_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed):
    
    set_seed(seed)
    # Instantiate model n_hid_units optimizer
    F_sa = FlowModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(F_sa.parameters(), learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs = [], []
    minibatch_loss = 0
    # Dictionary to store the color assigned to each node
    #color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    #nx.set_node_attributes(graph, color_map, 'color')

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        color_map = nx.coloring.greedy_color(state, strategy="random_sequential")
        nx.set_node_attributes(state, color_map, 'color')
        bound=max(color_map.values())
        edge_flow_preds = F_sa(graph_to_tensor(state))  # Predict F(s, a).
        #print(edge_flow_preds)
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            edge_flow_preds = edge_flow_preds * mask
            # Sample the action and compute the new state.
            policy = edge_flow_preds / edge_flow_preds.sum()
            # We want to sample from the policy here (a Categorical distribution...)
            action = Categorical(probs=policy).sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            #print(new_state.nodes[t]['color'])

            # To compute the loss, we'll first enumerate the parents, then compute
            # the edge flows F(s, a) of each parent, indexing to get relevant flows.
            parent_states, parent_actions = graph_parents(new_state)
            ps = torch.stack([graph_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(ps)[torch.arange(len(parent_states)), pa] 

            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward and set F(s,a) = 0 \forall a, since there
            # are no children of this state.
            #print(nx.get_node_attributes(new_state, "color"))
            #reward = color_reward(new_state)
                reward = vqe_reward(new_state)
                edge_flow_preds = torch.zeros(nx.number_of_nodes(state))
            #print(reward)
            else:
            # We compute F(s, a) and set the reward to zero.
                reward = 0
                edge_flow_preds = F_sa(graph_to_tensor(new_state))


            minibatch_loss += flow_matching_loss(  # Accumulate.
                parent_edge_flow_preds,
                edge_flow_preds,
                reward,
            )
            state = new_state  # Continue iterating.

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:

            # Normalize accumulated loss.
            minibatch_loss = minibatch_loss / (update_freq)
            losses.append(minibatch_loss.item())
            tbar.set_description("Training iter (reward={:.3f})".format(reward))
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0

    return sampled_graphs, losses
