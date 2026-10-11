
import os
import tempfile
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Global Archaeological Network",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Global Archaeological Site Network")
st.markdown(
    """
    Explore archaeological sites across the world through
    interactive geographical networks, shortest paths,
    and statistical analysis.
    """
)


# =========================================================
# 2. LOAD AND VALIDATE DATA
# =========================================================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "archaeological_sites.csv"
)

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        "Dataset not found. Check that "
        "data/archaeological_sites.csv exists."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()


required_columns = [
    "site_name",
    "latitude",
    "longitude",
    "period",
    "site_type",
    "country"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required CSV columns: {missing_columns}"
    )
    st.info(
        "Your CSV should contain site_name, latitude, longitude, "
        "period, site_type, and country."
    )
    st.stop()


# Convert coordinates to numeric values.
df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
)
df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
)

# Clean text columns.
for column in [
    "site_name", "period", "site_type", "country"
]:
    df[column] = df[column].fillna("Unknown").astype(str).str.strip()
    df[column] = df[column].replace("", "Unknown")

# Retain optional columns if available.
if "site_id" not in df.columns:
    df["site_id"] = range(1, len(df) + 1)

if "source" not in df.columns:
    df["source"] = "Not provided"

if "source_url" not in df.columns:
    df["source_url"] = ""

# Remove invalid coordinates.
df = df.dropna(
    subset=["latitude", "longitude"]
).copy()

df = df[
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
].copy()

# Avoid duplicate site names in the graph.
df = df.drop_duplicates(
    subset=["site_name"],
    keep="first"
)

if df.empty:
    st.error("No valid archaeological site records were found.")
    st.stop()


# =========================================================
# 3. SIDEBAR FILTERS
# =========================================================

st.sidebar.title("Explore the Dataset")

st.sidebar.caption(
    "Filters apply to the network, map, charts, and analysis."
)

all_countries = sorted(df["country"].unique())

selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=all_countries,
    default=all_countries
)

country_filtered_df = df[
    df["country"].isin(selected_countries)
].copy()

all_periods = sorted(country_filtered_df["period"].unique())

selected_periods = st.sidebar.multiselect(
    "Select historical periods",
    options=all_periods,
    default=all_periods
)

period_filtered_df = country_filtered_df[
    country_filtered_df["period"].isin(selected_periods)
].copy()

all_types = sorted(period_filtered_df["site_type"].unique())

selected_types = st.sidebar.multiselect(
    "Select site types",
    options=all_types,
    default=all_types
)

filtered_df = period_filtered_df[
    period_filtered_df["site_type"].isin(selected_types)
].copy()

threshold = st.sidebar.slider(
    "Maximum connection distance (km)",
    min_value=100,
    max_value=2000,
    value=500,
    step=50
)

show_edge_labels = st.sidebar.checkbox(
    "Show distances on network edges",
    value=False
)

st.sidebar.divider()

st.sidebar.caption(
    "Connections represent geographical proximity, "
    "not confirmed ancient travel or trade routes."
)


# =========================================================
# 4. DISTANCE CALCULATION
# =========================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometres."""
    earth_radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2 * earth_radius * atan2(
        sqrt(a), sqrt(1 - a)
    )


# =========================================================
# 5. BUILD THE NETWORK
# =========================================================

G = nx.Graph()

for _, row in filtered_df.iterrows():
    G.add_node(
        row["site_name"],
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        period=row["period"],
        site_type=row["site_type"],
        country=row["country"]
    )

site_records = list(G.nodes(data=True))

for i in range(len(site_records)):
    for j in range(i + 1, len(site_records)):
        name1, data1 = site_records[i]
        name2, data2 = site_records[j]

        distance = calculate_distance(
            data1["latitude"],
            data1["longitude"],
            data2["latitude"],
            data2["longitude"]
        )

        if distance <= threshold:
            G.add_edge(
                name1,
                name2,
                distance=round(distance, 2)
            )


# =========================================================
# 6. OVERVIEW METRICS
# =========================================================

st.subheader("Global Network Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Archaeological Sites",
    G.number_of_nodes()
)

m2.metric(
    "Geographical Connections",
    G.number_of_edges()
)

component_count = (
    nx.number_connected_components(G)
    if G.number_of_nodes() > 0 else 0
)

m3.metric(
    "Connected Components",
    component_count
)

average_degree = (
    sum(dict(G.degree()).values()) / G.number_of_nodes()
    if G.number_of_nodes() > 0 else 0
)

m4.metric(
    "Average Connections per Site",
    f"{average_degree:.2f}"
)

st.caption(
    f"Showing {len(filtered_df)} sites from "
    f"{filtered_df['country'].nunique()} countries or "
    "territories matching the selected filters."
)


# =========================================================
# 7. SITE EXPLORER
# =========================================================

st.subheader("Explore an Archaeological Site")

if G.number_of_nodes() > 0:

    site_names = sorted(G.nodes())

    selected_site = st.selectbox(
        "Search for a site",
        options=site_names,
        key="selected_archaeological_site"
    )

    site_data = G.nodes[selected_site]

    site_col1, site_col2, site_col3 = st.columns(3)

    site_col1.markdown(f"**Site**\n\n{selected_site}")
    site_col2.markdown(
        f"**Country**\n\n{site_data['country']}"
    )
    site_col3.markdown(
        f"**Historical period**\n\n{site_data['period']}"
    )

    st.write(f"**Site type:** {site_data['site_type']}")

    neighbors = sorted(G.neighbors(selected_site))

    st.write(
        "**Directly connected sites:** "
        + (
            ", ".join(neighbors)
            if neighbors else "No connections at this threshold"
        )
    )

    st.write(f"**Number of connections:** {G.degree(selected_site)}")

    selected_row = filtered_df[
        filtered_df["site_name"] == selected_site
    ].iloc[0]

    source_value = str(selected_row["source"])
    source_url_value = str(selected_row["source_url"]).strip()

    st.write(f"**Source reference:** {source_value}")

    if source_url_value and source_url_value.lower() != "nan":
        st.markdown(f"[Open source reference]({source_url_value})")

else:
    selected_site = None
    neighbors = []
    st.info("No sites match the selected filters.")


# =========================================================
# 8. SHORTEST-DISTANCE PATH
# =========================================================

st.subheader("Shortest-Distance Path Finder")

shortest_path = []
path_exists = False
total_distance = 0.0
source_site = None
destination_site = None

if G.number_of_nodes() >= 2:

    path_options = sorted(G.nodes())

    path_col1, path_col2 = st.columns(2)

    with path_col1:
        source_site = st.selectbox(
            "Starting archaeological site",
            options=path_options,
            index=0,
            key="path_source"
        )

    with path_col2:
        destination_site = st.selectbox(
            "Destination archaeological site",
            options=path_options,
            index=len(path_options) - 1,
            key="path_destination"
        )

    if source_site == destination_site:
        st.info("Choose two different sites.")

    elif nx.has_path(G, source_site, destination_site):

        shortest_path = nx.shortest_path(
            G,
            source=source_site,
            target=destination_site,
            weight="distance",
            method="dijkstra"
        )

        total_distance = nx.shortest_path_length(
            G,
            source=source_site,
            target=destination_site,
            weight="distance",
            method="dijkstra"
        )

        path_exists = True

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Total Path Distance",
            f"{total_distance:,.2f} km"
        )

        result_col2.metric(
            "Connections Along Path",
            len(shortest_path) - 1
        )

        st.success(
            " → ".join(shortest_path)
        )

        # Display individual edge distances.
        path_details = []

        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]

            path_details.append({
                "From": u,
                "To": v,
                "Distance (km)": G[u][v]["distance"]
            })

        st.dataframe(
            pd.DataFrame(path_details),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "This is the shortest path in the current distance-based "
            "graph, not a reconstructed historical travel itinerary."
        )

    else:
        st.warning(
            "No path exists between these sites under the current "
            "filters and connection threshold. Try increasing the "
            "threshold or selecting different sites."
        )

elif G.number_of_nodes() == 1:
    st.info("At least two sites are needed to calculate a path.")

else:
    st.info("Add sites or change the filters to find a path.")


path_nodes = set(shortest_path)

path_edges = {
    frozenset([shortest_path[i], shortest_path[i + 1]])
    for i in range(len(shortest_path) - 1)
}


# =========================================================
# 9. INTERACTIVE NETWORK GRAPH
# =========================================================

st.subheader("Interactive Archaeological Network")

if G.number_of_nodes() > 0:

    net = Network(
        height="650px",
        width="100%",
        bgcolor="#111827",
        font_color="#FFFFFF",
        directed=False,
        notebook=False
    )

    for node, data in G.nodes(data=True):

        if path_exists and node == source_site:
            color = "#22C55E"
            size = 38

        elif path_exists and node == destination_site:
            color = "#EF4444"
            size = 38

        elif path_exists and node in path_nodes:
            color = "#F59E0B"
            size = 32

        elif node == selected_site:
            color = "#A855F7"
            size = 34

        elif node in neighbors:
            color = "#06B6D4"
            size = 27

        else:
            color = "#94A3B8"
            size = 20

        net.add_node(
            node,
            label=node,
            title=(
                f"Site: {node}\n"
                f"Country: {data['country']}\n"
                f"Period: {data['period']}\n"
                f"Type: {data['site_type']}\n"
                f"Connections: {G.degree(node)}"
            ),
            color=color,
            size=size,
            borderWidth=2,
            font={
                "color": "#FFFFFF",
                "size": 15
            }
        )

    for u, v, data in G.edges(data=True):

        edge_is_path = frozenset([u, v]) in path_edges

        edge_color = "#FDE047" if edge_is_path else "#64748B"
        edge_width = 5 if edge_is_path else 1

        edge_options = {
            "title": f"Distance: {data['distance']} km",
            "color": edge_color,
            "width": edge_width
        }

        if show_edge_labels:
            edge_options["label"] = f"{data['distance']} km"

        net.add_edge(u, v, **edge_options)

    net.barnes_hut()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        encoding="utf-8"
    ) as temp_file:
        html_path = temp_file.name

    try:
        net.save_graph(html_path)

        with open(html_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        components.html(
            html_content,
            height=670,
            scrolling=True
        )

    finally:
        if os.path.exists(html_path):
            os.remove(html_path)

    st.markdown(
        """
        **Network legend**

        🟢 Starting site · 🔴 Destination · 🟠 Shortest-path sites ·
        🟣 Selected site · 🔵 Direct neighbors · ⚪ Other sites ·
        🟡 Shortest-path edges
        """
    )

    st.caption(
        "Node colors indicate selection and path status, not the "
        "country or archaeological importance of a site."
    )

else:
    st.info("No sites are available for the network graph.")


# =========================================================
# 10. CENTRALITY ANALYSIS
# =========================================================

st.subheader("Network Centrality Analysis")

if G.number_of_nodes() > 0:

    degree_centrality = nx.degree_centrality(G)

    betweenness_centrality = nx.betweenness_centrality(
        G,
        weight="distance",
        normalized=True
    )

    closeness_centrality = nx.closeness_centrality(
        G,
        distance="distance"
    )

    centrality_df = pd.DataFrame({
        "Site": list(G.nodes()),
        "Country": [
            G.nodes[node]["country"] for node in G.nodes()
        ],
        "Connections": [
            G.degree(node) for node in G.nodes()
        ],
        "Degree Centrality": [
            round(degree_centrality[node], 4)
            for node in G.nodes()
        ],
        "Betweenness Centrality": [
            round(betweenness_centrality[node], 4)
            for node in G.nodes()
        ],
        "Closeness Centrality": [
            round(closeness_centrality[node], 4)
            for node in G.nodes()
        ]
    })

    centrality_df = centrality_df.sort_values(
        by="Degree Centrality",
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(
        centrality_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("**Understanding the results**")

    st.write(
        "- **Degree centrality:** Measures the proportion of "
        "other sites directly connected to a site."
    )

    st.write(
        "- **Betweenness centrality:** Measures how often a site "
        "lies on shortest paths between pairs of sites."
    )

    st.write(
        "- **Closeness centrality:** Measures how close a site "
        "is to other reachable sites using network distances."
    )

    chart_df = centrality_df.head(15).sort_values(
        "Degree Centrality"
    )

    fig_centrality = px.bar(
        chart_df,
        x="Degree Centrality",
        y="Site",
        color="Country",
        orientation="h",
        title="Top 15 Sites by Degree Centrality",
        hover_data=["Connections", "Betweenness Centrality"]
    )

    st.plotly_chart(
        fig_centrality,
        use_container_width=True
    )


# =========================================================
# 11. GLOBAL GEOGRAPHICAL MAP
# =========================================================

st.subheader("Global Archaeological Site Map")

if not filtered_df.empty:

    fig_map = px.scatter_map(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="country",
        hover_name="site_name",
        hover_data={
            "country": True,
            "period": True,
            "site_type": True,
            "latitude": False,
            "longitude": False
        },
        zoom=0,
        height=650,
        map_style="open-street-map",
        title="Archaeological Sites Across the World"
    )

    fig_map.update_traces(
        marker={"size": 11, "opacity": 0.85}
    )

    fig_map.update_layout(
        dragmode="pan",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        legend_title_text="Country"
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True
        }
    )

else:
    st.info("No sites match the current filters.")


# =========================================================
# 12. ANALYTICAL CHARTS
# =========================================================

st.subheader("Archaeological Data Insights")

if not filtered_df.empty:

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "Sites by Country",
        "Sites by Period",
        "Sites by Type"
    ])

    # ---------------- Country chart ----------------

    with chart_tab1:

        country_counts = (
            filtered_df["country"]
            .value_counts()
            .rename_axis("Country")
            .reset_index(name="Number of Sites")
        )

        country_counts = country_counts.sort_values(
            "Number of Sites",
            ascending=True
        )

        fig_country = px.bar(
            country_counts,
            x="Number of Sites",
            y="Country",
            orientation="h",
            title="Number of Archaeological Sites by Country",
            text="Number of Sites"
        )

        st.plotly_chart(
            fig_country,
            use_container_width=True
        )

    # ---------------- Period chart ----------------

    with chart_tab2:

        period_counts = (
            filtered_df["period"]
            .value_counts()
            .rename_axis("Historical Period")
            .reset_index(name="Number of Sites")
        )

        fig_period = px.bar(
            period_counts,
            x="Historical Period",
            y="Number of Sites",
            title="Sites by Historical Period",
            text="Number of Sites"
        )

        fig_period.update_layout(
            xaxis_tickangle=-35
        )

        st.plotly_chart(
            fig_period,
            use_container_width=True
        )

    # ---------------- Site type chart ----------------

    with chart_tab3:

        type_counts = (
            filtered_df["site_type"]
            .value_counts()
            .rename_axis("Site Type")
            .reset_index(name="Number of Sites")
        )

        fig_type = px.pie(
            type_counts,
            names="Site Type",
            values="Number of Sites",
            title="Distribution of Archaeological Site Types",
            hole=0.35
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True
        )


# =========================================================
# 13. DATASET EXPLORER
# =========================================================

st.subheader("Archaeological Dataset Explorer")

search_text = st.text_input(
    "Search by site name, country, period, or type",
    placeholder="For example: India, Roman, temple..."
)

search_df = filtered_df.copy()

if search_text.strip():

    query = search_text.strip().lower()

    search_mask = (
        search_df["site_name"].str.lower().str.contains(
            query, na=False, regex=False
        )
        | search_df["country"].str.lower().str.contains(
            query, na=False, regex=False
        )
        | search_df["period"].str.lower().str.contains(
            query, na=False, regex=False
        )
        | search_df["site_type"].str.lower().str.contains(
            query, na=False, regex=False
        )
    )

    search_df = search_df[search_mask]

display_columns = [
    "site_id",
    "site_name",
    "latitude",
    "longitude",
    "period",
    "site_type",
    "country",
    "source"
]

if "source_url" in search_df.columns:
    display_columns.append("source_url")

with st.expander("View and search archaeological records", expanded=True):

    st.write(f"**Records displayed:** {len(search_df)}")

    st.dataframe(
        search_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    csv_data = search_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered dataset as CSV",
        data=csv_data,
        file_name="archaeological_sites_filtered.csv",
        mime="text/csv"
    )


# =========================================================
# 14. DATA QUALITY INFORMATION
# =========================================================

st.subheader("Dataset Quality")

quality_col1, quality_col2, quality_col3 = st.columns(3)

quality_col1.metric(
    "Records in Dataset",
    len(df)
)

quality_col2.metric(
    "Distinct Countries / Territories",
    df["country"].nunique()
)

source_available = (
    df["source"].notna()
    & df["source"].astype(str).str.strip().ne("")
    & df["source"].astype(str).str.lower().ne("not provided")
)

quality_col3.metric(
    "Records with Source Labels",
    int(source_available.sum())
)

st.caption(
    "A source label is not the same as a verified source URL. "
    "Check coordinates, site dates, classifications, and references "
    "before using the dataset for academic conclusions."
)


# =========================================================
# 15. PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader("About This Project")

st.markdown(
    """
    **Project:** Global Archaeological Site Network Analysis
    and Interactive Visualization

    **Technologies used**
    - Python for data processing and calculations
    - Pandas for dataset handling
    - NetworkX for graph construction and network analysis
    - PyVis for the interactive network graph
    - Plotly for maps and analytical charts
    - Streamlit for the dashboard

    **Methods**
    - Haversine formula for geographical distances
    - Dijkstra's algorithm for weighted shortest paths
    - Degree, betweenness, and closeness centrality
    - Country, historical-period, and site-type comparisons
    """
)

st.warning(
    "Research limitation: Sites are connected when their "
    "great-circle distance is within the chosen threshold. "
    "These connections do not prove historical travel, trade, "
    "cultural contact, or direct interaction. Results depend on "
    "dataset accuracy, site selection, and the chosen threshold."
)
