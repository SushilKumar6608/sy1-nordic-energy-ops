import os
import json
import random
import numpy as np
import pandas as pd
import networkx as nx

SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")

# Grid parameters grounded in Swedish district heating engineering standards
# Source basis: Svensk Fjärrvärme technical guidelines
# Pipe flow velocity: 0.5 - 2.0 m/s
# Supply temperature: 80-120°C, Return: 40-60°C
# Typical substation capacity: 0.5 - 5.0 MW

CITY_CONFIGS = {
    "Stockholm": {"nodes": 30, "seed": 42},
    "Gothenburg": {"nodes": 25, "seed": 43},
    "Malmö": {"nodes": 22, "seed": 44},
    "Uppsala": {"nodes": 18, "seed": 45},
    "Umeå": {"nodes": 16, "seed": 46},
    "Kiruna": {"nodes": 12, "seed": 47},
    "Åre-Östersund": {"nodes": 14, "seed": 48},
}


def generate_city_grid(city: str, n_nodes: int, seed: int) -> nx.DiGraph:
    """
    Generate a physically grounded district heating grid for one city.

    Grid structure:
    - Node 0: Central heat plant (source)
    - Nodes 1 to n_nodes-1: Substations or building clusters
    - Edges: Pipes with capacity, length, heat loss attributes
    - Topology: Tree-like with some redundancy loops (realistic)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    G = nx.DiGraph()

    # Add central plant node
    G.add_node(
        0,
        type="plant",
        city=city,
        capacity_mw=50.0,
        supply_temp_c=110.0,
        return_temp_c=50.0,
    )

    # Add substation nodes
    for i in range(1, n_nodes):
        node_type = "substation" if i % 4 != 0 else "building_cluster"
        G.add_node(
            i,
            type=node_type,
            city=city,
            capacity_mw=float(rng.uniform(0.5, 5.0)),
            demand_mw=float(rng.uniform(0.3, 4.5)),
            supply_temp_c=float(rng.uniform(80.0, 110.0)),
            return_temp_c=float(rng.uniform(40.0, 60.0)),
        )

    # Build tree topology from plant outward (realistic for district heating)
    # Each node connects to the plant or a closer node
    for i in range(1, n_nodes):
        parent = rng.integers(0, i)
        pipe_length_m = float(rng.uniform(200, 2000))
        pipe_diameter_mm = float(rng.choice([100, 150, 200, 250, 300, 400]))
        flow_velocity = float(rng.uniform(0.5, 2.0))
        # Heat loss: ~15-30 W/m for insulated pipes (Svensk Fjärrvärme standard)
        heat_loss_w_per_m = float(rng.uniform(15.0, 30.0))
        total_heat_loss_kw = (heat_loss_w_per_m * pipe_length_m) / 1000.0

        G.add_edge(
            int(parent),
            i,
            pipe_length_m=pipe_length_m,
            pipe_diameter_mm=pipe_diameter_mm,
            flow_velocity_m_s=flow_velocity,
            heat_loss_kw=total_heat_loss_kw,
            max_flow_m3_h=float(rng.uniform(10, 200)),
        )

    # Add a small number of redundancy edges (loops) — realistic for larger grids
    n_loops = max(1, n_nodes // 10)
    added = 0
    attempts = 0
    while added < n_loops and attempts < 100:
        u = int(rng.integers(1, n_nodes))
        v = int(rng.integers(1, n_nodes))
        if u != v and not G.has_edge(u, v) and not G.has_edge(v, u):
            pipe_length_m = float(rng.uniform(300, 1500))
            G.add_edge(
                u,
                v,
                pipe_length_m=pipe_length_m,
                pipe_diameter_mm=float(rng.choice([100, 150, 200])),
                flow_velocity_m_s=float(rng.uniform(0.5, 1.5)),
                heat_loss_kw=(25.0 * pipe_length_m) / 1000.0,
                max_flow_m3_h=float(rng.uniform(10, 80)),
                is_redundancy=True,
            )
            added += 1
        attempts += 1

    return G


def graph_to_dataframes(G: nx.DiGraph, city: str):
    """Convert networkx graph to node and edge DataFrames for storage."""
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        row = {"node_id": node_id, "city": city}
        row.update(attrs)
        nodes.append(row)

    edges = []
    for u, v, attrs in G.edges(data=True):
        row = {"source": u, "target": v, "city": city}
        row.update(attrs)
        edges.append(row)

    return pd.DataFrame(nodes), pd.DataFrame(edges)


def generate_all_grids() -> dict:
    """Generate grids for all cities and save to data/synthetic/."""
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)

    all_nodes = []
    all_edges = []
    graphs = {}

    for city, config in CITY_CONFIGS.items():
        print(f"Generating grid for {city} ({config['nodes']} nodes)...")
        G = generate_city_grid(city, config["nodes"], config["seed"])
        graphs[city] = G

        nodes_df, edges_df = graph_to_dataframes(G, city)
        all_nodes.append(nodes_df)
        all_edges.append(edges_df)

        print(
            f"  Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | "
            f"Total pipe length: {edges_df['pipe_length_m'].sum():.0f}m"
        )

    nodes_combined = pd.concat(all_nodes, ignore_index=True)
    edges_combined = pd.concat(all_edges, ignore_index=True)

    nodes_path = os.path.join(SYNTHETIC_DIR, "grid_nodes.csv")
    edges_path = os.path.join(SYNTHETIC_DIR, "grid_edges.csv")

    nodes_combined.to_csv(nodes_path, index=False)
    edges_combined.to_csv(edges_path, index=False)

    print(f"\nSaved grid_nodes.csv: {len(nodes_combined)} nodes")
    print(f"Saved grid_edges.csv: {len(edges_combined)} edges")

    return graphs


if __name__ == "__main__":
    print("=== District Heating Grid Generator ===")
    graphs = generate_all_grids()
    print("\nDone.")
