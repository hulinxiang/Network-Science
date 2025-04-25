import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def empirical_CCDF(xaxis, data):
    lis = []
    for x in xaxis:
        lis.append(len(data[data >= x])/len(data))
    return lis

    
def plot_eigs(ax_val, ax_vecs, vals, vecs, X, title_prefix, n, m):
    """
    ax_val : Axes for plotting eigenvalues
    ax_vecs : A list of 5 axes, used to plot the first 5 eigenvectors
    vals, vecs : Corresponding eigenvalues and eigenvectors
    X : Used as the x-axis when plotting eigenvectors (1D data in this example)
    title_prefix : Prefix for the figure title
    n: Number of eigen values to show
    m: Number of eigen vectors to show
    """
    num_eigs_to_show = n

    # (1) Eigenvalue scatter plot: only show the first 10
    ax_val.scatter(range(1, num_eigs_to_show+1), vals[:num_eigs_to_show], marker='*')
    ax_val.set_title(f"{title_prefix}\nEigenvalues (1-{num_eigs_to_show})")
    ax_val.set_xlabel("Eigenvalue index")
    ax_val.set_ylabel("Eigenvalue")

    # (2) The first m eigenvectors
    for i in range(m):
        ax = ax_vecs[i]
        ax.plot(X, vecs[:, i])
        ax.set_title(f"{title_prefix}\nEigenvector {i+1}")
        ax.set_xlabel("x_i")
        ax.set_ylabel("u_i")


def read_graph_from_files(node_file, edge_file):
    """
    Build and return an undirected graph G from node_file and edge_file.
      - node_file: one node ID per line (convertible to int)
      - edge_file: each line in the form 'u v' indicates an edge
    """
    G = nx.Graph()

    # Read nodes
    with open(node_file, 'r') as fn:
        for line in fn:
            node_id = line.strip()
            node_id = int(node_id)  # If the node ID was originally a numeric string
            G.add_node(node_id)

    # Read edges
    with open(edge_file, 'r') as fe:
        for line in fe:
            uv = line.strip().split()
            if len(uv) == 2:
                u, v = uv
                u, v = int(u), int(v)
                G.add_edge(u, v)
    
    return G


def plot_adjacency_matrix(G, node_order=None, title=""):
    """
    Use matplotlib to draw the adjacency matrix of G on an n x n dot matrix plot.
      - node_order: specify the order in which nodes appear; if None, uses sorted(G.nodes()).
      - White indicates 0 (no edge), Black (dark gray) indicates 1 (edge).
      - A: adjacency matrix
    """
    if node_order is None:
        node_order = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=node_order, dtype=int)

    plt.figure()
    plt.imshow(A, cmap='gray_r', interpolation='nearest')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def read_graph_from_files_with_color(node_file, edge_file):
    """
    Read the graph structure from files and the color attribute for each node.

    Parameters:
        node_file: str - each line formatted as "node_id color_id"
        edge_file: str - each line formatted as "u v"

    Returns:
        G: networkx.Graph
    """
    G = nx.Graph()

    # Read nodes and their colors
    with open(node_file, 'r') as fn:
        for line in fn:
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, color_id = int(parts[0]), int(parts[1])
                G.add_node(node_id, true_color=color_id)

    # Read edges
    with open(edge_file, 'r') as fe:
        for line in fe:
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)

    return G


def color_adjacency_matrix(A, color_vector):
    """
    Generate a color-encoded adjacency matrix for visualization.

    Each non-zero entry in the adjacency matrix A is colored based on the node's color label 
    (given by color_vector). The resulting 3D matrix can be used to display a color-coded 
    dot matrix plot (RGB image) of the graph structure.

    Parameters:
        A (np.ndarray): A square adjacency matrix (2D numpy array) representing the graph.
        color_vector (List[int]): A list of color labels (integers) corresponding to each node in A.

    Returns:
        color_matrix (np.ndarray): A 3D RGB matrix (size x size x 3) where each non-zero entry 
                                   in A is represented with an assigned RGB color based on the node's color.
    """
    size = A.shape[0]
    color_matrix = np.ones((size, size, 3), dtype=int) * 255  # Initialize as white
    unique_colors = sorted(set(color_vector))
    
    # Predefined RGB colors for visual distinction (cycled if needed)
    predefined_colors = [
        (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
        (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153)
    ]
    
    # Map color IDs to RGB values
    color_map = {col: predefined_colors[col % len(predefined_colors)] for col in unique_colors}
    
    # Assign colors to non-zero entries in A based on the color of the source node (row)
    for i in range(size):
        for j in range(size):
            if A[i, j] > 0:
                color_matrix[i, j] = color_map[color_vector[i]]
    
    return color_matrix
