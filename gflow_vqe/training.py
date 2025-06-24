from tqdm import tqdm, trange
from torch.distributions.categorical import Categorical
from gflow_vqe.utils import *
from gflow_vqe.gflow_utils import *

def precolored_flow_match_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
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
            parent_states, parent_actions = graph_parents_precolored(new_state)
            ps = torch.stack([graph_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(ps)[torch.arange(len(parent_states)), pa] 

            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward and set F(s,a) = 0 \forall a, since there
            # are no children of this state.
            #print(nx.get_node_attributes(new_state, "color"))
            #reward = color_reward(new_state)
                #reward = vqe_reward(new_state)
                reward = meas_reward(new_state,wfn,n_q)
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
            torch.save({
            'epoch': episode,
            'model_state_dict': F_sa.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_precoloredFMmodel.pth")

    return sampled_graphs, losses

def pure_flow_match_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    # Instantiate model n_hid_units optimizer
    for i in range(nx.number_of_nodes(graph)):
        graph.nodes[i]['color'] = i
    
    F_sa = FlowModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(F_sa.parameters(), learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs = [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+3

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        
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
                #reward = vqe_reward(new_state)
                reward = meas_reward(new_state,wfn,n_q)
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
            torch.save({
            'epoch': episode,
            'model_state_dict': F_sa.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_pureFMmodel.pth")

    return sampled_graphs, losses

def colored_initial_flow_match_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    
    F_sa = FlowModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(F_sa.parameters(), learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs = [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    nx.set_node_attributes(graph, color_map, "color")
    bound=max(color_map.values())+2

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        
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
            parent_states, parent_actions = graph_parents_precolored(new_state)
            ps = torch.stack([graph_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(ps)[torch.arange(len(parent_states)), pa] 

            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward and set F(s,a) = 0 \forall a, since there
            # are no children of this state.
            #print(nx.get_node_attributes(new_state, "color"))
                reward = meas_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
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
            torch.save({
            'epoch': episode,
            'model_state_dict': F_sa.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_colored_initial_FMmodel.pth")

    return sampled_graphs, losses

def colored_initial_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    
    # Instantiate model and optimizer
    model = TBModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    nx.set_node_attributes(graph, color_map, "color")
    bound=max(color_map.values())+5

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        P_F_s, P_B_s = model(graph_to_tensor(state),n_terms)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0
        
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = meas_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
            
            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = model(graph_to_tensor(new_state),n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`. 
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)
            
            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_colored_initial_TBmodel.pth")            

    return sampled_graphs, losses

def precolored_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    
    # Instantiate model and optimizer
    model = TBModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        color_map = nx.coloring.greedy_color(state, strategy="random_sequential")
        nx.set_node_attributes(state, color_map, 'color')
        bound=max(color_map.values())
        P_F_s, P_B_s = model(graph_to_tensor(state),n_terms)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0
        
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = meas_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
            
            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = model(graph_to_tensor(new_state),n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`. 
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)
            
            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_precoloredTBmodel.pth")            

    return sampled_graphs, losses

def pure_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    
    # Instantiate model and optimizer
    model = TBModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+10

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        P_F_s, P_B_s = model(graph_to_tensor(state),n_terms)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0
        
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = meas_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
            
            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = model(graph_to_tensor(new_state),n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`. 
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)
            
            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_pureTBmodel.pth")

    return sampled_graphs, losses

def seq_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    
    # Instantiate model and optimizer
    model = TBModel_seq(n_hid_units,n_terms)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")
    bound=max(color_map.values())+ math.floor(0.1*max(color_map.values())) #

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        P_F_s = model(graph_to_tensor(state),n_terms)  # Forward and backward policy
        total_log_P_F = 0
        
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            #P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = meas_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
            
            # We recompute P_F and P_B for new_state.
            P_F_s = model(graph_to_tensor(new_state),n_terms)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss_seq(
            model.logZ,
            total_log_P_F,
            reward,
        )

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_pureTBmodel_seq.pth")

    return sampled_graphs, losses

def emb_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name):
    
    set_seed(seed)
    
    # Instantiate model and optimizer
    model = embTBModel(n_hid_units,n_terms)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+10

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        P_F_s, P_B_s = model(graph_to_tensor(state),n_terms)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0
        
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = my_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
            
            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = model(graph_to_tensor(new_state),n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`. 
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)
            
            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )
    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, fig_name + "_embTBmodel.pth")

    return sampled_graphs, losses

def GIN_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name, n_emb):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    # Instantiate model and optimizer
    model = GIN(n_hid_units, n_terms, n_emb).to(device)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)
    # Attributes and batch for GNN
    edge_index = generate_edge_index(graph).to(device)
    # let's try to see if we can "negativly" affect nodes that are connected
    edge_attr = -1. * torch.ones((edge_index.shape[1],1),dtype=torch.long, device=device)
    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+2

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        x = graph_to_tensor(state).unsqueeze(1).long().to(device)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #data = Data(x=graph_to_tensor(state).unsqueeze(1).long(), edge_index=edge_index, edge_attr=edge_attr)
        P_F_s, P_B_s = model(data.x, data.edge_index, data.edge_attr, data.batch)
        #P_F_s, P_B_s = model(graph_to_tensor(state).unsqueeze(1).long(),n_terms, edge_attr, batch=1)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound).to(device)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = my_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)

            # We recompute P_F and P_B for new_state.
            x = graph_to_tensor(new_state).unsqueeze(1).long().to(device)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            P_F_s, P_B_s = model(data.x, data.edge_index, data.edge_attr, data.batch)
            #P_F_s, P_B_s = model(graph_to_tensor(new_state).unsqueeze(1).long(),n_terms, edge_attr, batch=1)
            mask = calculate_backward_mask_from_state(new_state, t, bound).to(device)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`.
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )
    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
                'epoch': episode,
                'model_state_dict': model.cpu().state_dict(),  # move to CPU before saving
                'optimizer_state_dict': opt.state_dict(),
                'loss': losses,
                }, fig_name + "_ginTBmodel.pth")
            model.to(device)  # Move it back if training will continue

    return sampled_graphs, losses

def GIN_2GPU_TB_training(graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q, fig_name,n_emb, device_ids):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    # Instantiate model and optimizer
    model = GIN_2GPUParallel(n_hid_units, n_terms, n_emb)
    #model = torch.nn.DataParallel(model, device_ids)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)
    # Attributes and batch for GNN
    edge_index = generate_edge_index(graph)
    # let's try to see if we can "negativly" affect nodes that are connected
    edge_attr = -1. * torch.ones((edge_index.shape[1],1),dtype=torch.long)
    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+2

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        x = graph_to_tensor(state).unsqueeze(1).long()
        batch = torch.zeros(x.size(0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        #data = Data(x=graph_to_tensor(state).unsqueeze(1).long(), edge_index=edge_index, edge_attr=edge_attr)
        P_F_s.to('cuda:1'), P_B_s.to('cuda:1') = model(data.x, data.edge_index, data.edge_attr, data.batch)
        #P_F_s, P_B_s = model(graph_to_tensor(state).unsqueeze(1).long(),n_terms, edge_attr, batch=1)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = my_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)

            # We recompute P_F and P_B for new_state.
            x = graph_to_tensor(new_state).unsqueeze(1).long()
            batch = torch.zeros(x.size(0), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            P_F_s.to('cuda:1'), P_B_s.to('cuda:1') = model(data.x, data.edge_index, data.edge_attr, data.batch)
            #P_F_s, P_B_s = model(graph_to_tensor(new_state).unsqueeze(1).long(),n_terms, edge_attr, batch=1)
            mask = calculate_backward_mask_from_state(new_state, t, bound)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`.
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )
    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.module.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save({
                'epoch': episode,
                'model_state_dict': model.state_dict(),  # move to CPU before saving
                'optimizer_state_dict': opt.state_dict(),
                'loss': losses,
                }, fig_name + "_ginTBmodel.pth")

    return sampled_graphs, losses