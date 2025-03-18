import networkx as nx
import matplotlib.pyplot as plt
import collections
import random


def build_graph(cities_file, edges_file):
    # Dictionary to map node_id -> city/airport name
    id_to_name = {}
    # Dictionary to map airport_code -> node_id
    code_to_id = {}

    # Read city data
    with open(cities_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            parts = line.split('|')

            if len(parts) < 3:
                continue
            airport_code = parts[0]
            node_id_str = parts[1]
            city_name = parts[2]

            # Convert node_id from string to integer
            try:
                node_id = int(node_id_str)
            except ValueError:
                continue

            # Store the mapping (node_id -> city_name)
            id_to_name[node_id] = city_name

            # Also store the mapping (airport_code -> node_id)
            code_to_id[airport_code] = node_id

    # Create an undirected graph
    G = nx.Graph()

    for nid in id_to_name:
        G.add_node(nid)

    # Read edge data
    with open(edges_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            # Parse node IDs
            try:
                source_id = int(parts[0])
                target_id = int(parts[1])
            except ValueError:
                continue

            G.add_edge(source_id, target_id)

    return G, id_to_name, code_to_id


def analyze_connected_components(G):
    # Get all connected components
    components = list(nx.connected_components(G))

    # Print the number of connected components
    print("Number of connected components:", len(components))

    # Find the largest connected component
    largest_cc = max(components, key=len)

    # Create a subgraph of the largest connected component
    G_lcc = G.subgraph(largest_cc).copy()

    # Print
    print("Largest connected component has",
          G_lcc.number_of_nodes(), "nodes and",
          G_lcc.number_of_edges(), "edges.")

    return G_lcc


def print_top_degree_nodes(G_lcc, id_to_name, top_n=10):
    # Create a list of (node_id, degree) pairs
    degree_list = [(node, G_lcc.degree(node)) for node in G_lcc.nodes()]

    # Sort the list by degree in descending order
    degree_list.sort(key=lambda x: x[1], reverse=True)

    # Select the top nodes
    top_nodes = degree_list[:top_n]

    # Print the results
    print(f"Top {top_n} nodes by degree in the largest component:")
    for node_id, deg in top_nodes:
        city_name = id_to_name.get(node_id, "Unknown")  # fallback if not found
        print(f" - {city_name}: degree {deg}")


def plot_degree_distribution(G_lcc):
    # Get degrees of all nodes
    degrees = [G_lcc.degree(n) for n in G_lcc.nodes()]

    # Count how many times each degree occurs
    degree_count = collections.Counter(degrees)

    # Remove degree 0 entries
    degree_count = {k: v for k, v in degree_count.items() if k > 0}

    # Sort degrees in ascending order
    x_vals = sorted(degree_count.keys())

    # Calculate the fraction
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
    if G_lcc.number_of_nodes() == 0:
        print("The graph is empty. No diameter can be computed.")
        return

    # Pick a random node and run BFS
    some_node = random.choice(list(G_lcc.nodes()))
    dist_map_1 = nx.single_source_shortest_path_length(G_lcc, some_node)
    # Find the node that is farthest from 'some_node'
    farthest_node_1 = max(dist_map_1, key=dist_map_1.get)

    # From that farthest node, run BFS again
    dist_map_2 = nx.single_source_shortest_path_length(G_lcc, farthest_node_1)
    farthest_node_2 = max(dist_map_2, key=dist_map_2.get)

    # The distance between farthest_node_1 and farthest_node_2 is the diameter
    diameter_val = dist_map_2[farthest_node_2]

    # Get the actual path
    longest_path_ids = nx.shortest_path(G_lcc, farthest_node_1, farthest_node_2)

    longest_path_names = [id_to_name.get(nid, "Unknown") for nid in longest_path_ids]

    # Print results
    print("Estimated diameter of the largest component:", diameter_val)
    print("An example of the longest shortest path between two nodes:")
    print(" -> ".join(longest_path_names))

    # Return the diameter and path
    return diameter_val, longest_path_names


def find_shortest_route_cbr_to_cpt(G_lcc, code_to_id, id_to_name):
    # Get the node IDs from the airport codes
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

    num_flights = len(path_ids) - 1

    path_names = [id_to_name.get(nid, "Unknown") for nid in path_ids]

    # Print the results
    print(f"Smallest number of flights from Canberra (CBR) to Cape Town (CPT): {num_flights}")
    print("Route (city/airport names):")
    print(" -> ".join(path_names))


def print_top_betweenness_nodes(G_lcc, id_to_name, top_n=10):
    # Calculate betweenness for each node
    bc_dict = nx.betweenness_centrality(G_lcc)
    # bc_dict is a dictionary {node_id: betweenness_value}

    # Sort the items by betweenness in descending order
    sorted_bc = sorted(bc_dict.items(), key=lambda x: x[1], reverse=True)

    # Take the top 'top_n'
    top_nodes = sorted_bc[:top_n]

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

    print("\n============Question6============")
    find_shortest_route_cbr_to_cpt(G_lcc, code_to_id, id_to_name)

    print("\n============Question7============")
    print_top_betweenness_nodes(G_lcc, id_to_name, top_n=10)


if __name__ == "__main__":
    main()
