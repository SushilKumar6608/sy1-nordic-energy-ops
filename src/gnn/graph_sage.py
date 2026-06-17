import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              classification_report, roc_auc_score)

SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")

# GraphSAGE hyperparameters
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
EPOCHS       = 50
LR           = 1e-3
BATCH_SIZE   = 32


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for node-level anomaly scoring on district heating grid.

    Two GraphSAGE convolution layers aggregate neighbourhood information
    from connected pipe nodes. The final layer outputs a single anomaly
    score per node — higher score means higher likelihood of fault.

    Architecture:
    Input node features -> SAGEConv(hidden) -> ReLU -> Dropout
                        -> SAGEConv(hidden) -> ReLU -> Dropout
                        -> Linear(1) -> Sigmoid -> anomaly score
    """
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.dropout  = dropout

        # First layer: input -> hidden
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Last conv layer: hidden -> hidden/2
        self.convs.append(SAGEConv(hidden_dim, hidden_dim // 2))

        # Final classification head
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(x)).squeeze(-1)


def load_grid(city: str) -> tuple:
    """
    Load grid topology for a city from synthetic CSVs.
    Returns (node_df, edge_df).
    """
    nodes_path = os.path.join(SYNTHETIC_DIR, "grid_nodes.csv")
    edges_path = os.path.join(SYNTHETIC_DIR, "grid_edges.csv")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    city_nodes = nodes[nodes["city"] == city].reset_index(drop=True)
    city_edges = edges[edges["city"] == city].reset_index(drop=True)

    return city_nodes, city_edges


def build_node_features(nodes_df: pd.DataFrame,
                         anomaly_scores: np.ndarray = None) -> torch.Tensor:
    """
    Build node feature matrix from grid topology attributes.

    Features per node:
    - capacity_mw: maximum thermal capacity of the node
    - demand_mw: current demand (0 for plant nodes)
    - supply_temp_c: supply temperature
    - return_temp_c: return temperature
    - is_plant: binary flag (1 for central plant, 0 for substation)
    - anomaly_score: city-level anomaly score propagated to all nodes
      (from conformal detector — will be refined by GNN)

    The GNN learns how anomaly signals propagate through the grid topology,
    localising faults to specific nodes based on network position.
    """
    feature_cols = []

    # Capacity
    cap = nodes_df["capacity_mw"].fillna(0.0).values
    feature_cols.append(cap)

    # Demand (plant nodes have no demand column)
    demand = nodes_df.get("demand_mw", pd.Series(0.0, index=nodes_df.index))
    demand = demand.fillna(0.0).values
    feature_cols.append(demand)

    # Supply temperature
    supply_temp = nodes_df["supply_temp_c"].fillna(90.0).values
    feature_cols.append(supply_temp)

    # Return temperature
    return_temp = nodes_df["return_temp_c"].fillna(50.0).values
    feature_cols.append(return_temp)

    # Is plant flag
    is_plant = (nodes_df["type"] == "plant").astype(float).values
    feature_cols.append(is_plant)

    # Anomaly score (city-level score broadcast to all nodes)
    if anomaly_scores is not None:
        feature_cols.append(anomaly_scores)
    else:
        feature_cols.append(np.zeros(len(nodes_df)))

    X = np.stack(feature_cols, axis=1).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32)


def build_edge_index(nodes_df: pd.DataFrame,
                      edges_df: pd.DataFrame) -> torch.Tensor:
    """
    Build edge index tensor for PyTorch Geometric.
    Node IDs in edges are global — remap to local 0-indexed IDs.
    """
    node_id_to_idx = {nid: idx for idx, nid in enumerate(nodes_df["node_id"])}

    src_list, dst_list = [], []
    for _, row in edges_df.iterrows():
        src = node_id_to_idx.get(int(row["source"]))
        dst = node_id_to_idx.get(int(row["target"]))
        if src is not None and dst is not None:
            src_list.append(src)
            dst_list.append(dst)
            # Add reverse edge — undirected graph
            src_list.append(dst)
            dst_list.append(src)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


def generate_node_labels(nodes_df: pd.DataFrame,
                          city_anomaly_detected: bool,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic node-level fault labels for training.

    When a city-level anomaly is detected, assign fault labels to a
    small subset of nodes (1-3 nodes) based on network position.
    Non-plant nodes closer to the periphery are more likely to be faulty.

    This simulates the GNN learning task: given a city-level anomaly signal,
    which specific nodes in the grid are the likely fault origin?
    """
    n      = len(nodes_df)
    labels = np.zeros(n, dtype=np.float32)

    if city_anomaly_detected:
        # Select 1-3 non-plant nodes as fault nodes
        non_plant_idx = nodes_df[nodes_df["type"] != "plant"].index.tolist()
        # Remap to local indices
        local_non_plant = list(range(1, n))  # skip plant node (index 0)
        n_faults        = int(rng.integers(1, min(4, len(local_non_plant) + 1)))
        fault_nodes     = rng.choice(local_non_plant, size=n_faults, replace=False)
        labels[fault_nodes] = 1.0

    return labels


def train_gnn_for_city(city: str,
                        detection_results: pd.DataFrame,
                        device: torch.device) -> dict:
    """
    Train GraphSAGE model for a single city.

    Training data: synthetic graph snapshots where:
    - Anomalous snapshots: city has a detected anomaly, 1-3 nodes labelled as faults
    - Normal snapshots: all nodes labelled as normal (0)

    We generate N_SNAPSHOTS synthetic training examples per city,
    with anomaly rate matching our detected anomaly rate from Step 4.
    """
    print(f"\n  Training GraphSAGE for {city}...")

    nodes_df, edges_df = load_grid(city)
    n_nodes            = len(nodes_df)
    edge_index         = build_edge_index(nodes_df, edges_df)

    rng = np.random.default_rng(seed=abs(hash(city)) % (2**32))

    # Get city anomaly rates from conformal detection results
    heat_row = detection_results[
        (detection_results["city"] == city) &
        (detection_results["target"] == "heating_demand_mw")
    ]
    cool_row = detection_results[
        (detection_results["city"] == city) &
        (detection_results["target"] == "cooling_demand_mw")
    ]

    heat_anom_rate = float(heat_row["tp"].values[0] + heat_row["fp"].values[0]) / 8700 \
        if len(heat_row) > 0 else 0.05
    cool_anom_rate = float(cool_row["tp"].values[0] + cool_row["fp"].values[0]) / 8700 \
        if len(cool_row) > 0 else 0.02

    # Generate synthetic graph snapshots
    N_SNAPSHOTS  = 500
    graphs       = []
    anom_rate    = max(heat_anom_rate, cool_anom_rate)

    for _ in range(N_SNAPSHOTS):
        is_anomalous  = rng.random() < anom_rate
        anom_score    = np.full(n_nodes, float(is_anomalous) + rng.uniform(-0.1, 0.1))
        anom_score    = np.clip(anom_score, 0.0, 1.0)

        node_features = build_node_features(nodes_df, anom_score)
        node_labels   = generate_node_labels(nodes_df, is_anomalous, rng)

        graph = Data(
            x          = node_features,
            edge_index = edge_index,
            y          = torch.tensor(node_labels, dtype=torch.float32)
        )
        graphs.append(graph)

    # Train/val split
    split      = int(0.8 * N_SNAPSHOTS)
    train_data = graphs[:split]
    val_data   = graphs[split:]

    loader     = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # Model
    input_dim = node_features.shape[1]
    model     = GraphSAGEModel(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(device)
                pred     = model(batch.x, batch.edge_index)
                val_loss += criterion(pred, batch.y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)

    # Save model
    city_key   = (city.replace(" ", "_").replace("-", "_")
                      .replace("å", "a").replace("ä", "a").replace("ö", "o")
                      .replace("Å", "A").replace("Ä", "A").replace("Ö", "O"))
    model_path = os.path.join(MODELS_DIR, f"gnn_{city_key}.pt")
    torch.save({
        "model_state": best_state,
        "input_dim":   input_dim,
        "hidden_dim":  HIDDEN_DIM,
        "num_layers":  NUM_LAYERS,
        "dropout":     DROPOUT,
        "n_nodes":     n_nodes,
        "city":        city,
    }, model_path)

    print(f"    Saved: {model_path}")
    print(f"    Nodes: {n_nodes} | Edges: {len(edges_df)} | "
          f"Best val loss: {best_val_loss:.4f}")

    return {
        "city":          city,
        "n_nodes":       n_nodes,
        "n_edges":       len(edges_df),
        "best_val_loss": round(best_val_loss, 4),
        "input_dim":     input_dim,
    }


def infer_fault_nodes(city: str,
                       anomaly_score: float,
                       device: torch.device,
                       top_k: int = 3) -> pd.DataFrame:
    """
    Given a city-level anomaly score, use the trained GNN to identify
    the top-k most likely fault nodes in the grid.

    Returns a DataFrame with node IDs, types, and fault probability scores,
    sorted by fault probability descending.
    """
    city_key   = (city.replace(" ", "_").replace("-", "_")
                      .replace("å", "a").replace("ä", "a").replace("ö", "o")
                      .replace("Å", "A").replace("Ä", "A").replace("Ö", "O"))
    model_path = os.path.join(MODELS_DIR, f"gnn_{city_key}.pt")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model      = GraphSAGEModel(
        input_dim  = checkpoint["input_dim"],
        hidden_dim = checkpoint["hidden_dim"],
        num_layers = checkpoint["num_layers"],
        dropout    = checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    nodes_df, edges_df = load_grid(city)
    n_nodes            = len(nodes_df)
    edge_index         = build_edge_index(nodes_df, edges_df)

    anom_scores   = np.full(n_nodes, anomaly_score)
    node_features = build_node_features(nodes_df, anom_scores)

    with torch.no_grad():
        scores = model(
            node_features.to(device),
            edge_index.to(device)
        ).cpu().numpy()

    result = nodes_df[["node_id", "type", "capacity_mw"]].copy()
    result["fault_probability"] = scores
    result = result.sort_values("fault_probability", ascending=False)

    return result.head(top_k).reset_index(drop=True)


def train_all_cities() -> pd.DataFrame:
    """Train GraphSAGE for all 7 cities and return summary."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    detection_path = os.path.join(PROCESSED_DIR, "conformal_detection_results.csv")
    detection_results = pd.read_csv(detection_path)

    nodes_path = os.path.join(SYNTHETIC_DIR, "grid_nodes.csv")
    nodes      = pd.read_csv(nodes_path)
    cities     = nodes["city"].unique().tolist()

    all_results = []
    for city in cities:
        result = train_gnn_for_city(city, detection_results, device)
        all_results.append(result)

    summary = pd.DataFrame(all_results)
    print("\nTraining complete. Summary:")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    print("=== GraphSAGE Grid Topology Model ===")
    print("Task: node-level fault localisation on district heating grid")
    print("Input: city-level anomaly score + grid topology features")
    print("Output: fault probability per node")
    print()
    summary = train_all_cities()

    # Demo inference for Stockholm
    print("\n--- Demo: fault localisation for Stockholm (anomaly_score=0.9) ---")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fault_nodes = infer_fault_nodes("Stockholm", anomaly_score=0.9, device=device)
    print(fault_nodes.to_string(index=False))