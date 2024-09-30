from utils import *

def plot_loss_curve(losses_A, title=""):
    plt.figure(figsize=(10,3))
    plt.plot(losses_A, color="black")

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

def check_sampled_graphs_vqe_plot(sampled_graphs):
    """Check sampled graphs with no duplicates based on vqe/number of shots and graphs them"""
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
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

def plot_graph_wcolor(graph):

    colors_dict = nx.get_node_attributes(graph, "color")
    vector = list(colors_dict.values())
    pos = nx.circular_layout(graph)
    options = {
    "pos": pos,
    "node_color": vector,
    "node_size": 300,
    "edge_color": "gray",
    "alpha": 0.8,
    "width": 6,
    "labels": {n: n for n in graph}
    }
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

