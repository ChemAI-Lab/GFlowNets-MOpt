from gflow_vqe.utils import *
from matplotlib.ticker import MaxNLocator

def plot_loss_curve(figure, losses_A, title=""):
    filename = f"{figure}_loss.svg"
    plt.figure(figsize=(10,5))
    plt.plot(losses_A, color="black")
    plt.savefig(filename, format='svg', dpi=600)

def check_sampled_graphs_vqe(sampled_graphs):
    """Check sampled graphs with no duplicates based on number of shots. No Graph."""
    #fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    n_plot = 16  # 4 x 4

    print('Proportion of valid graphs:{}, ideal=1.'.format(
        sum([color_reward(i) > 0 for i in sampled_graphs]) / len(sampled_graphs)
    ))

    # Sort graphs by reward, but filter out those with duplicate color dictionaries
    unique_graphs = []
    seen_color_dicts = set()

    for graph in sorted(sampled_graphs, key=lambda i: shots_estimator(i), reverse=False):
        color_dict = frozenset(nx.get_node_attributes(graph, "color").items())
        if color_dict not in seen_color_dicts:
            seen_color_dicts.add(color_dict)
            unique_graphs.append(graph)

    print('Number of unique graphs ={}'.format(len(unique_graphs)))
    print('Number of shots for the best {} graphs'.format(n_plot))

    for i in range(n_plot):
      print('Number of shots={} and max color {}'.format(shots_estimator(unique_graphs[i]), max_color(unique_graphs[i])))

def check_sampled_graphs_vqe_plot(figure, sampled_graphs):
    filename = f"{figure}_graphs.png"

    """Check sampled graphs with no duplicates based on vqe/number of shots and graphs them"""
    fig, ax = plt.subplots(4, 4, figsize=(40, 40))
    n_plot = 16  # 4 x 4

    print('Proportion of valid graphs:{}, ideal=1'.format(
        sum([color_reward(i) > 0 for i in sampled_graphs]) / len(sampled_graphs)
    ))

    # Sort graphs by reward, but filter out those with duplicate color dictionaries
    unique_graphs = []
    seen_color_dicts = set()

    for graph in sorted(sampled_graphs, key=lambda i: shots_estimator(i), reverse=False):
        color_dict = frozenset(nx.get_node_attributes(graph, "color").items())
        if color_dict not in seen_color_dicts:
            seen_color_dicts.add(color_dict)
            unique_graphs.append(graph)

    print('Number of unique graphs ={}'.format(len(unique_graphs)))
    print('Number of shots for the best {} graphs'.format(n_plot))

    for i in range(n_plot):
      print('Number of shots={} and max color {}'.format(shots_estimator(unique_graphs[i]), max_color(unique_graphs[i])))
      plt.sca(ax[i//4, i%4])
      plot_graph_wcolor(unique_graphs[i])

    plt.savefig(filename, format='png')

def plot_graph_wcolor(graph):

    colors_dict = nx.get_node_attributes(graph, "color")
    vector = list(colors_dict.values())
    pos = nx.circular_layout(graph)
    #pos = nx.kamada_kawai_layout(graph)
    options = {
    "pos": pos,
    "node_color": vector,
    "node_size": 300,
    "edge_color": "gray",
    "alpha": 0.8,
    "width": 6,
    "labels": {n: n for n in graph}
    }
    #Options for larger graphs (up to BeH2).Use kamada_kawai, node_size=35, width 0.02, "linewidths": 0. 
    #For H2 add labels and circ layout. node_size=300, width=6
    nx.draw(graph, cmap=plt.cm.rainbow, **options)
    shots = shots_estimator(graph)
    plt.text(0.01, 0.98, f'Shots:',
             horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=9, color='black')
    plt.text(0.01, 0.92, f'{shots}',
             horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=9, color='black')

def check_sampled_graphs_color_reward_plot(sampled_graphs):
    """Check sampled graphs with no duplicates based on color reward."""
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    n_plot = 16  # 4 x 4

    print('Proportion of valid graphs:{}, ideal=1'.format(
        sum([color_reward(i) > 0 for i in sampled_graphs]) / len(sampled_graphs)
    ))

    # Sort graphs by reward, but filter out those with duplicate color dictionaries
    unique_graphs = []
    seen_color_dicts = set()

    for graph in sorted(sampled_graphs, key=lambda i: color_reward(i), reverse=True):
        color_dict = frozenset(nx.get_node_attributes(graph, "color").items())
        if color_dict not in seen_color_dicts:
            seen_color_dicts.add(color_dict)
            unique_graphs.append(graph)

    print('Number of unique graphs ={}'.format(len(unique_graphs)))
    print('Number of colors for the best {} graphs'.format(n_plot))

    for i in range(n_plot):
      print('Number of shots={} and max color {}'.format(shots_estimator(unique_graphs[i]), max_color(unique_graphs[i])))
      plt.sca(ax[i//4, i%4])
      plot_graph_wcolor(unique_graphs[i])

def histogram_last(sampled_graphs):
    n_shots = [shots_estimator(i) for i in sampled_graphs]
    color = [max_color(i) for i in sampled_graphs]
    color_graph = color[-1000:]
    n_shots_graph = np.array(n_shots[-1000:])*1E-6
    x_bins = np.arange(min(color_graph) - 0.5, max(color_graph) + 1.5, 1)  # Center the bars on integer ticks
    y_bins = np.linspace(min(n_shots_graph), max(n_shots_graph), 100)  # You can adjust the number of bins
    plt.figure(figsize=(10,5))
# Create 2D histogram
    plt.hist2d(color_graph, n_shots_graph, bins=[x_bins, y_bins])
# Add color bar for intensity reference
    plt.colorbar(label='Sampled graphs')
# Label axes
    plt.xlabel('Max Color')
    plt.ylabel(r'$M_{est}\  \ [\times 10^{6}]$')
    plt.savefig('histo_last.png', format='png', dpi=600)
    
def histogram_all(figure, sampled_graphs):
    filename = f"{figure}_histo_all.svg"
    n_shots = [shots_estimator(i)*1E-6 for i in sampled_graphs]
    color = [max_color(i) for i in sampled_graphs]
    print('Minimum number of groups found {}'.format(min(color)))
    x_bins = np.arange(min(color) - 0.5, max(color)  + 1.5, 1)  # Center the bars on integer ticks
    y_bins = np.linspace(min(n_shots), max(n_shots), 50)  # You can adjust the number of bins
# Create 2D histogram
    plt.figure(figsize=(10,5))
    plt.hist2d(color, n_shots, bins=[x_bins, y_bins])
# Add color bar for intensity reference
    plt.colorbar(label='Sampled graphs')
# Label axes
    plt.xticks(np.arange(min(color),max(color)+1,step=1,dtype=np.int32)) #Requests only integers on x
    plt.xlabel('Max Color')
    plt.ylabel(r'$M_{est}\  \ [\times 10^{6}]$')
    plt.savefig(filename, format='svg', dpi=600)

def check_sampled_graphs_fci(sampled_graphs, wfn, n_qubit):
    """Check sampled graphs with no duplicates based on number of shots. No Graph."""
    #fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    n_plot = 16  # 4 x 4

    print('Proportion of valid graphs:{}, ideal=1.'.format(
        sum([color_reward(i) > 0 for i in sampled_graphs]) / len(sampled_graphs)
    ))

    # Sort graphs by reward, but filter out those with duplicate color dictionaries
    unique_graphs = []
    seen_color_dicts = set()

    for graph in sorted(sampled_graphs, key=lambda i: meas_reward(i, wfn, n_qubit), reverse=True):
        color_dict = frozenset(nx.get_node_attributes(graph, "color").items())
        if color_dict not in seen_color_dicts:
            seen_color_dicts.add(color_dict)
            unique_graphs.append(graph)

    print('Number of unique graphs ={}'.format(len(unique_graphs)))
    print('Number of shots for the best {} graphs'.format(n_plot))

    for i in range(n_plot):
      print('Number of shots={} and max color {}. Reward:{}'.format(get_groups_measurement(unique_graphs[i], wfn, n_qubit), max_color(unique_graphs[i]),meas_reward(unique_graphs[i], wfn, n_qubit)))

def plot_graph_wcolor_fci(graph, wfn, n_qubit):

    colors_dict = nx.get_node_attributes(graph, "color")
    vector = list(colors_dict.values())
    pos = nx.circular_layout(graph)
    #pos = nx.kamada_kawai_layout(graph)
    options = {
    "pos": pos,
    "node_color": vector,
    "node_size": 300,
    "edge_color": "gray",
    "alpha": 0.8,
    "width": 6,
    "labels": {n: n for n in graph}
    }
    #Options for larger graphs (up to BeH2).Use kamada_kawai, node_size=35, width 0.02, "linewidths": 0. 
    #For H2 add labels and circ layout. node_size=300, width=6
    nx.draw(graph, cmap=plt.cm.rainbow, **options)
    shots = get_groups_measurement(graph, wfn, n_qubit)/(0.0016**2)
    plt.text(0.01, 0.98, f'Shots:',
             horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=9, color='black')
    plt.text(0.01, 0.92, f'{shots}',
             horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=9, color='black')

def check_sampled_graphs_fci_plot(figure, sampled_graphs, wfn, n_qubit):
    filename = f"{figure}_graphs.png"

    """Check sampled graphs with no duplicates based on vqe/number of shots and graphs them"""
    fig, ax = plt.subplots(4, 4, figsize=(40, 40))
    n_plot = 16  # 4 x 4

    print('Proportion of valid graphs:{}, ideal=1'.format(
        sum([color_reward(i) > 0 for i in sampled_graphs]) / len(sampled_graphs)
    ))

    # Sort graphs by reward, but filter out those with duplicate color dictionaries
    unique_graphs = []
    seen_color_dicts = set()

    for graph in sorted(sampled_graphs, key=lambda i: meas_reward(i, wfn, n_qubit), reverse=True):
        color_dict = frozenset(nx.get_node_attributes(graph, "color").items())
        if color_dict not in seen_color_dicts:
            seen_color_dicts.add(color_dict)
            unique_graphs.append(graph)

    print('Number of unique graphs ={}'.format(len(unique_graphs)))
    print('Number of shots for the best {} graphs'.format(n_plot))

    for i in range(n_plot):
      print('eps^2 M={} and max color {}. Reward= {}'.format(get_groups_measurement(unique_graphs[i], wfn, n_qubit), max_color(unique_graphs[i]),meas_reward(unique_graphs[i], wfn, n_qubit)))
      plt.sca(ax[i//4, i%4])
      plot_graph_wcolor_fci(unique_graphs[i], wfn, n_qubit)

    plt.savefig(filename, format='png')

def histogram_all_fci(figure, sampled_graphs, wfn, n_qubit):
    filename = f"{figure}_histo_all.svg"
    n_shots = [get_groups_measurement(i, wfn, n_qubit)*1E-3/(0.0016**2) for i in sampled_graphs]
    color = [max_color(i) for i in sampled_graphs]
    print('Minimum number of groups found {}'.format(min(color)))
    x_bins = np.arange(min(color) - 0.5, max(color)  + 1.5, 1)  # Center the bars on integer ticks
    y_bins = np.linspace(min(n_shots), max(n_shots), 50)  # You can adjust the number of bins
# Create 2D histogram
    plt.figure(figsize=(10,5))
    plt.hist2d(color, n_shots, bins=[x_bins, y_bins])
# Add color bar for intensity reference
    plt.colorbar(label='Sampled graphs')
# Label axes
    plt.xticks(np.arange(min(color),max(color)+1,step=1,dtype=np.int32)) #Requests only integers on x
    plt.xlabel('Max Color')
    plt.ylabel(r'$M_{est}\  \ [\times 10^{3}]$')
    plt.savefig(filename, format='svg', dpi=600)