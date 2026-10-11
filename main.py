
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

# Load dataset
df = pd.read_csv("data/archaeological_sites.csv")

# Create graph
G = nx.Graph()

# Add sites as nodes
for _, row in df.iterrows():
    G.add_node(
        row["site_name"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        period=row["period"],
        site_type=row["site_type"]
    )

# Calculate distance between two coordinates in km
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# Connect sites within 500 km
threshold = 500
sites = list(G.nodes(data=True))

for i in range(len(sites)):
    for j in range(i + 1, len(sites)):
        name1, data1 = sites[i]
        name2, data2 = sites[j]

        distance = calculate_distance(
            data1["latitude"], data1["longitude"],
            data2["latitude"], data2["longitude"]
        )

        if distance <= threshold:
            G.add_edge(name1, name2, distance=round(distance, 2))

# Print results
print("Number of sites:", G.number_of_nodes())
print("Number of connections:", G.number_of_edges())

print("\nConnections:")
for u, v, data in G.edges(data=True):
    print(f"{u} -- {v}: {data['distance']} km")

# Draw graph
plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    with_labels=True,
    node_color="lightgreen",
    node_size=2200,
    font_size=8,
    font_weight="bold",
    edge_color="gray",
    width=1.5
)

plt.title("Archaeological Site Network")
plt.show()

# Graph analysis
print("\n--- Graph Analysis ---")

# Degree centrality
degree = nx.degree_centrality(G)

# Betweenness centrality
betweenness = nx.betweenness_centrality(G)

print("\nSite Importance Ranking:")

for site in G.nodes():
    print(f"\nSite: {site}")
    print(f"Degree Centrality: {degree[site]:.3f}")
    print(f"Betweenness Centrality: {betweenness[site]:.3f}")

# Find the most connected site
most_connected = max(degree, key=degree.get)

print("\nMost connected site:", most_connected)


# Community detection
print("\n--- Archaeological Site Communities ---")

communities = nx.community.greedy_modularity_communities(G)

for i, community in enumerate(communities, start=1):
    print(f"Community {i}: {', '.join(sorted(community))}")
