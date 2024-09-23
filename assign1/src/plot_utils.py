import matplotlib.pyplot as plt
import networkx as nx

def plot_full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    plt.figure(figsize=figsize)
    ax = plt.axes((0, 0, 1, 1), frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def get_position(nx_graph):
    """
    Normalize the positions of the nodes in the graph.
    """
    pos = {node: (nx_graph.nodes[node]['pos'][0], nx_graph.nodes[node]['pos'][1]) for node in nx_graph.nodes}
    return pos


def visualize_route(nx_graph, route_node_list):
    """
    Visualize the graph and highlight the path.
    """
    pos = get_position(nx_graph)
    plot_full_frame(12, 14)
    # Draw Seoul Map
    plt.imshow(plt.imread("seoul_map.png"), extent=(126.4, 127.8, 36.7, 38.0), aspect='auto')
    # Draw the entire graph
    nx.draw(nx_graph, pos, node_size=10, edge_color='gray', with_labels=False)

    # Highlight the path
    nx_route = nx_graph.subgraph(route_node_list)
    weights = [3 for _ in nx_route.edges()]
    nx.draw(nx_route, pos, node_size=50, edge_color='red', node_color='red', width=weights, with_labels=False)

    plt.show()