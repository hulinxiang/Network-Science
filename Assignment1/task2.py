import networkx as nx
import matplotlib.pyplot as plt
import collections
import random


def build_graph(cities_file, edges_file):
    """
    This function reads city information and edge data,
    then builds and returns a NetworkX Graph object.
    """
    # Dictionary to map node_id -> city/airport name
    id_to_name = {}
    # Dictionary to map airport_code -> node_id
    code_to_id = {}

    # Read city data
    with open(cities_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines if any
            if not line:
                continue
            parts = line.split('|')
            # parts should contain [airport_code, node_id, city_name...]
            if len(parts) < 3:
                continue
            airport_code = parts[0]
            node_id_str = parts[1]
            city_name = parts[2]

            # Convert node_id from string to integer
            try:
                node_id = int(node_id_str)
            except ValueError:
                # If node_id is not an integer, skip or handle error
                continue

            # Store the mapping (node_id -> city_name)
            id_to_name[node_id] = city_name

            # Also store the mapping (airport_code -> node_id)
            code_to_id[airport_code] = node_id

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes from id_to_name dictionary
    # (This step is optional, because adding edges will also create nodes automatically,
    #  but we do it here to ensure all nodes are present in the graph, even if they have no edges.)
    for nid in id_to_name:
        G.add_node(nid)

    # Read edge data
    with open(edges_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines if any
            if not line:
                continue
            # Assume each line has exactly two integers separated by space
            parts = line.split()
            if len(parts) < 2:
                continue

            # Parse node IDs
            try:
                source_id = int(parts[0])
                target_id = int(parts[1])
            except ValueError:
                # If either is not an integer, skip or handle error
                continue

            # Add an undirected edge
            G.add_edge(source_id, target_id)

    return G, id_to_name, code_to_id


def analyze_connected_components(G):
    """
    This function calculates the number of connected components in the graph G
    and identifies the largest connected component (by number of nodes).
    It prints out the total number of components and the size (nodes, edges)
    of the largest component. It returns the subgraph corresponding to
    the largest connected component.
    """
    # Get all connected components as a list of sets (each set contains node IDs)
    components = list(nx.connected_components(G))

    # Print how many connected components there are
    print("Number of connected components:", len(components))

    # Find the largest connected component based on the number of nodes
    largest_cc = max(components, key=len)

    # Create a subgraph of the largest connected component
    G_lcc = G.subgraph(largest_cc).copy()

    # Print the number of nodes and edges in the largest component
    print("Largest connected component has",
          G_lcc.number_of_nodes(), "nodes and",
          G_lcc.number_of_edges(), "edges.")

    # Return the largest connected component subgraph for further analysis
    return G_lcc


def print_top_degree_nodes(G_lcc, id_to_name, top_n=10):
    """
    This function finds the nodes in G_lcc with the highest degree,
    then prints their city/airport names (from id_to_name) and
    the number of edges (degree) they have.
    Only the top 'top_n' nodes are displayed.
    """
    # Create a list of (node_id, degree) pairs
    degree_list = [(node, G_lcc.degree(node)) for node in G_lcc.nodes()]

    # Sort the list by degree in descending order
    degree_list.sort(key=lambda x: x[1], reverse=True)

    # Select the top 'top_n' nodes
    top_nodes = degree_list[:top_n]

    # Print the results with city/airport names instead of IDs
    print(f"Top {top_n} nodes by degree in the largest component:")
    for node_id, deg in top_nodes:
        city_name = id_to_name.get(node_id, "Unknown")  # fallback if not found
        print(f" - {city_name}: degree {deg}")


def plot_degree_distribution(G_lcc):
    """
    This function computes the degree distribution of G_lcc
    and plots it in both linear and log-log scales.
    Each data point (x, y) represents:
      x = a positive integer degree
      y = fraction of nodes in G_lcc having that degree.
    """
    # Get degrees of all nodes
    degrees = [G_lcc.degree(n) for n in G_lcc.nodes()]

    # Count how many times each degree occurs
    degree_count = collections.Counter(degrees)

    # Sort degrees (x-values) in ascending order
    x_vals = sorted(degree_count.keys())

    # Calculate the fraction y = count / total_number_of_nodes
    total_nodes = G_lcc.number_of_nodes()
    y_vals = [degree_count[d] / total_nodes for d in x_vals]

    # --- Plot in linear scale ---
    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, y_vals, color='blue', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.title('Degree Distribution (Linear Scale)')

    # --- Plot in log-log scale ---
    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, y_vals, color='red', alpha=0.7)
    plt.xscale('log', base=10)  # Use base 10 for the x-axis
    plt.yscale('log', base=10)  # Use base 10 for the y-axis
    plt.xlabel('Degree (log scale, base 10)')
    plt.ylabel('Fraction of Nodes (log scale, base 10)')
    plt.title('Degree Distribution (Log-Log Scale)')

    plt.show()


def compute_diameter_and_longest_path(G_lcc, id_to_name):
    """
    This function estimates the unweighted diameter of G_lcc by performing
    a two-phase BFS:
      1) Pick a random node in G_lcc and run BFS to find the farthest node.
      2) From that farthest node, run BFS again to find the absolute farthest node.
    The distance between these two farthest nodes is the (unweighted) diameter
    of the largest connected component. We also retrieve the longest shortest path
    (a sequence of node IDs), convert them into city/airport names, and print them.

    If the graph is not too large, you could alternatively use:
        diameter_value = nx.diameter(G_lcc)
        # to get the diameter directly
        # and then you can identify a pair of nodes in the periphery
        # (nx.periphery(G_lcc)) and get the actual path.
    """
    # If G_lcc is empty, handle the edge case
    if G_lcc.number_of_nodes() == 0:
        print("The graph is empty. No diameter can be computed.")
        return

    # 1) Pick a random node and run BFS/shortest_path_length
    some_node = random.choice(list(G_lcc.nodes()))
    dist_map_1 = nx.single_source_shortest_path_length(G_lcc, some_node)
    # Find the node that is farthest from 'some_node'
    farthest_node_1 = max(dist_map_1, key=dist_map_1.get)

    # 2) From that farthest node, run BFS again
    dist_map_2 = nx.single_source_shortest_path_length(G_lcc, farthest_node_1)
    farthest_node_2 = max(dist_map_2, key=dist_map_2.get)

    # The distance between farthest_node_1 and farthest_node_2 is the diameter
    diameter_val = dist_map_2[farthest_node_2]

    # Retrieve the actual path
    longest_path_ids = nx.shortest_path(G_lcc, farthest_node_1, farthest_node_2)

    # Convert node IDs to city/airport names
    longest_path_names = [id_to_name.get(nid, "Unknown") for nid in longest_path_ids]

    # Print results
    print("Estimated diameter of the largest component:", diameter_val)
    print("An example of the longest shortest path between two nodes:")
    print(" -> ".join(longest_path_names))

    # Return the diameter and path
    return diameter_val, longest_path_names


def find_shortest_route_cbr_to_cpt(G_lcc, code_to_id, id_to_name):
    """
    This function finds the shortest path (in terms of number of flights)
    from Canberra (CBR) to Cape Town (CPT) within G_lcc,
    assuming we already have a dictionary code_to_id that maps airport codes
    (e.g. 'CBR', 'CPT') to their corresponding node IDs, and a dictionary
    id_to_name that maps node IDs to city/airport names.

    It prints the minimum number of flights (edges) and the route in terms of
    city/airport names only (no node IDs).
    """
    # Retrieve the node IDs from the airport codes
    cbr_id = code_to_id.get("CBR", None)
    cpt_id = code_to_id.get("CPT", None)

    if cbr_id is None or cpt_id is None:
        print("Either CBR or CPT code was not found in code_to_id mapping.")
        return

    # Compute the shortest path
    try:
        path_ids = nx.shortest_path(G_lcc, source=cbr_id, target=cpt_id)
    except nx.NetworkXNoPath:
        print("No path found between CBR and CPT in the graph.")
        return

    # The number of flights is the number of edges, which is len(path) - 1
    num_flights = len(path_ids) - 1

    # Convert each node ID in the path to a city/airport name
    path_names = [id_to_name.get(nid, "Unknown") for nid in path_ids]

    # Print the results
    print(f"Smallest number of flights from Canberra (CBR) to Cape Town (CPT): {num_flights}")
    print("Route (city/airport names):")
    print(" -> ".join(path_names))

def print_top_betweenness_nodes(G_lcc, id_to_name, top_n=10):
    """
    This function calculates the betweenness centrality of all nodes in G_lcc,
    then finds the top 'top_n' nodes with the highest betweenness values.
    It prints each node's city/airport name (from id_to_name) and its betweenness score.
    """
    # Calculate betweenness centrality for each node
    bc_dict = nx.betweenness_centrality(G_lcc)
    # bc_dict is a dictionary {node_id: betweenness_value}

    # Sort the items by betweenness (value) in descending order
    sorted_bc = sorted(bc_dict.items(), key=lambda x: x[1], reverse=True)

    # Take the top 'top_n'
    top_nodes = sorted_bc[:top_n]

    # Print the results, using the city/airport name instead of the node ID
    print(f"Top {top_n} nodes by betweenness centrality in the largest component:")
    for node_id, bc_val in top_nodes:
        city_name = id_to_name.get(node_id, "Unknown")
        print(f" - {city_name}: betweenness = {bc_val:.6f}")

def main():
    # File names
    cities_file = r"airports\global-cities.dat"
    edges_file = r"airports\global-net.dat"

    # Build the graph
    G, id_to_name, code_to_id = build_graph(cities_file, edges_file)

    # Check basic info
    print("============Question1============")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Retrieve the largest connected component
    print("\n============Question2============")
    G_lcc = analyze_connected_components(G)

    # List the top 10 nodes in G having the highest
    # degree, and how many other nodes are they connected to.
    print("\n============Question3============")
    print_top_degree_nodes(G_lcc, id_to_name, top_n=10)

    # Plot distribution
    print("\n============Question4============")
    plot_degree_distribution(G_lcc)
    print("Finished Plotting!")

    # Calculate the (unweighted) diameter of the giant component G
    print("\n============Question5============")
    diameter_val, path_names = compute_diameter_and_longest_path(G_lcc, id_to_name)
    # Optional: you can continue performing other graph analyses here

    print("\n============Question6============")
    find_shortest_route_cbr_to_cpt(G_lcc, code_to_id, id_to_name)

    print("\n============Question7============")
    print_top_betweenness_nodes(G_lcc, id_to_name, top_n=10)

if __name__ == "__main__":
    main()
