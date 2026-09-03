"""
=========================================================
Project : A Hybrid CNN-LSTM-GCN Framework for Spatio-Temporal Crime Prediction
Module  : Graph Utilities & Adjacency Construction
File    : src/graph_utils.py

Description:
Builds and persists a static spatial adjacency graph for the 850 unique
(STATE/UT, DISTRICT) composite district nodes present in the NCRB crime dataset.
Adjacency is derived from geographic centroid coordinates (LATITUDE, LONGITUDE)
using strict k-Nearest Neighbors (k=5 default, configurable) with deterministic
tie-breaking to prevent artificial dense cliques among coincident state-centroid
nodes. Computes Kipf-Welling symmetric normalization (D^-1/2 * (A + I) * D^-1/2)
and persists artifacts to artifacts/adjacency.pkl.

Author: B.Tech Final Year Project
=========================================================
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Deterministic seeding
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------- #
# Logging Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GraphUtils")


class CrimeGraphBuilder:
    """
    Constructs and persists a static district-level spatial adjacency graph
    for the GCN branch of the Hybrid CNN-LSTM-GCN crime prediction system.

    Attributes:
        dataset_path (Path): Path to the engineered feature dataset CSV.
        artifacts_dir (Path): Directory where adjacency.pkl will be stored.
        k_neighbors (int): Number of nearest neighbors to connect per node.
        distance_threshold (Optional[float]): Distance threshold fallback.
        node_df (pd.DataFrame): Unique 850 district nodes and their coordinates.
        adj_matrix (np.ndarray): Raw static adjacency matrix (850 x 850).
        norm_adj_matrix (np.ndarray): Symmetrically normalized adjacency matrix (850 x 850).
        node_to_idx (Dict[Tuple[str, str], int]): (State, District) -> node index mapping.
        idx_to_node (Dict[int, Tuple[str, str]]): Node index -> (State, District) mapping.
    """

    NON_FEATURE_COLUMNS = ["Id", "STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES"]

    def __init__(
        self,
        dataset_path: str = "dataset/Crimes_in_india_2001-2013_features.csv",
        artifacts_dir: str = "artifacts",
        k_neighbors: int = 5,
        distance_threshold: Optional[float] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

        self.df: Optional[pd.DataFrame] = None
        self.node_df: Optional[pd.DataFrame] = None
        self.adj_matrix: Optional[np.ndarray] = None
        self.norm_adj_matrix: Optional[np.ndarray] = None
        self.node_to_idx: Dict[Tuple[str, str], int] = {}
        self.enc_pair_to_idx: Dict[Tuple[float, float], int] = {}
        self.idx_to_node: Dict[int, Tuple[str, str]] = {}
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------ #
    # Step 1: Load Data & Extract Unique Composite Nodes
    # ------------------------------------------------------------------ #
    def load_nodes(self) -> pd.DataFrame:
        """
        Extracts the 850 unique composite (STATE/UT, DISTRICT) nodes from the
        engineered dataset, along with their coordinates and encodings.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Feature dataset not found at: {self.dataset_path}")

        logger.info("Loading dataset from '%s'...", self.dataset_path)
        self.df = pd.read_csv(self.dataset_path)

        # Determine feature columns used by the models (excluding identifiers and target)
        self.feature_columns = [
            c for c in self.df.columns if c not in self.NON_FEATURE_COLUMNS
        ]

        # Extract unique composite nodes
        node_cols = [
            "STATE/UT",
            "DISTRICT",
            "STATE_ENCODED",
            "DISTRICT_ENCODED",
            "LATITUDE",
            "LONGITUDE",
        ]
        unique_nodes = (
            self.df[node_cols]
            .drop_duplicates(subset=["STATE/UT", "DISTRICT"])
            .sort_values(by=["STATE_ENCODED", "DISTRICT_ENCODED"])
            .reset_index(drop=True)
        )

        unique_nodes["node_id"] = np.arange(len(unique_nodes), dtype=int)
        self.node_df = unique_nodes

        # Build bidirectional mapping dictionaries
        for idx, row in unique_nodes.iterrows():
            state = str(row["STATE/UT"]).strip().upper()
            district = str(row["DISTRICT"]).strip().upper()
            state_enc = float(row["STATE_ENCODED"])
            dist_enc = float(row["DISTRICT_ENCODED"])

            self.node_to_idx[(state, district)] = idx
            self.enc_pair_to_idx[(state_enc, dist_enc)] = idx
            self.idx_to_node[idx] = (state, district)

        logger.info(
            "Identified %d unique composite district nodes across %d states/UTs.",
            len(self.node_df),
            self.node_df["STATE/UT"].nunique(),
        )
        return self.node_df

    # ------------------------------------------------------------------ #
    # Step 2: Build Weighted Spatial Adjacency Matrix with Non-Geo Filter
    # ------------------------------------------------------------------ #
    def build_adjacency_matrix(self) -> np.ndarray:
        """
        Constructs the static weighted spatial adjacency matrix A (N x N) using:
          1. Distance-weighted k-NN (k=5, w = 1 / (1 + distance)) among real geographic nodes only.
          2. Non-geographic nodes (CID, GRP, Railways, STF, etc.) excluded from spatial k-NN
             and connected strictly to their own state's real geographic districts.
          3. Symmetrization and Kipf-Welling normalization.
        """
        import re

        if self.node_df is None:
            self.load_nodes()

        num_nodes = len(self.node_df)
        coordinates = self.node_df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64)

        # Defensively handle any missing coordinates with state-centroid fallback
        if np.isnan(coordinates).any():
            logger.warning("Found NaN coordinates; applying state-centroid fallback...")
            state_avg = (
                self.node_df.groupby("STATE/UT")[["LATITUDE", "LONGITUDE"]].transform("mean")
            ).to_numpy()
            coordinates = np.where(np.isnan(coordinates), state_avg, coordinates)

        # Identify non-geographic nodes
        non_geo_pattern = re.compile(
            r'(CID|C\.I\.D|CBCID|GRP|G\.R\.P|RAILWAY|RAILWAYS|RLY|W\.RLY|STF|CRIME|SPECIAL|METRO|TRAFFIC|^EAST$|^WEST$|^NORTH$|^SOUTH$|^CENTRAL$)',
            re.IGNORECASE,
        )
        is_non_geo = np.zeros(num_nodes, dtype=bool)
        for idx, row in self.node_df.iterrows():
            d = str(row["DISTRICT"]).strip()
            if non_geo_pattern.search(d):
                is_non_geo[idx] = True

        geo_indices = np.where(~is_non_geo)[0]
        non_geo_indices = np.where(is_non_geo)[0]

        logger.info(
            "Graph Partition: %d real geographic district nodes, %d non-geographic/special nodes.",
            len(geo_indices),
            len(non_geo_indices),
        )

        # Deterministic micro-jitter (1e-5) for tie-breaking among coincident state-centroid points
        rng = np.random.RandomState(42)
        jittered_coords = coordinates + rng.normal(0, 1e-5, size=coordinates.shape)

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 1. Distance-weighted k-NN (k=5) among real geographic nodes only
        k = min(self.k_neighbors, len(geo_indices) - 1)
        geo_coords = jittered_coords[geo_indices]
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(geo_coords)
        distances, indices = nn.kneighbors(geo_coords)

        for i_idx, (dists, neighs) in enumerate(zip(distances, indices)):
            global_i = geo_indices[i_idx]
            for d, n_idx in zip(dists[1:], neighs[1:]):  # Skip self-loop
                global_j = geo_indices[n_idx]
                weight = 1.0 / (1.0 + float(d))
                adj[global_i, global_j] = max(adj[global_i, global_j], weight)
                adj[global_j, global_i] = max(adj[global_j, global_i], weight)

        # 2. Non-geographic nodes: connect strictly to their own state's real geographic districts
        for u_idx in non_geo_indices:
            state_u = self.node_df.iloc[u_idx]["STATE/UT"]
            state_geo_indices = [
                g for g in geo_indices if self.node_df.iloc[g]["STATE/UT"] == state_u
            ]
            if len(state_geo_indices) > 0:
                for v_idx in state_geo_indices:
                    d = np.linalg.norm(coordinates[u_idx] - coordinates[v_idx])
                    weight = 1.0 / (1.0 + float(d))
                    adj[u_idx, v_idx] = max(adj[u_idx, v_idx], weight)
                    adj[v_idx, u_idx] = max(adj[v_idx, u_idx], weight)
            else:
                # If state has no real districts, connect to state peers
                same_state = [
                    g for g in range(num_nodes)
                    if self.node_df.iloc[g]["STATE/UT"] == state_u and g != u_idx
                ]
                for v_idx in same_state:
                    adj[u_idx, v_idx] = 1.0
                    adj[v_idx, u_idx] = 1.0

        # Remove diagonal self-loops in raw adjacency (will be added in normalization)
        np.fill_diagonal(adj, 0.0)

        # Ensure no node is completely disconnected (minimum degree >= 1)
        isolated_nodes = np.where(adj.sum(axis=1) == 0)[0]
        if len(isolated_nodes) > 0:
            logger.warning("Connecting %d isolated nodes to nearest spatial neighbor.", len(isolated_nodes))
            for iso in isolated_nodes:
                distances = np.linalg.norm(coordinates - coordinates[iso], axis=1)
                distances[iso] = np.inf
                nearest = np.argmin(distances)
                d_near = distances[nearest]
                w = 1.0 / (1.0 + float(d_near))
                adj[iso, nearest] = w
                adj[nearest, iso] = w

        self.adj_matrix = adj
        degrees = (adj > 0).sum(axis=1)
        weighted_degrees = adj.sum(axis=1)
        edge_count = int(np.sum(adj > 0) / 2)
        logger.info(
            "Constructed static spatial adjacency matrix (%d x %d): Total Edges = %d, Min Degree = %d, "
            "Max Degree = %d, Median Degree = %.1f, Mean Degree = %.2f, Mean Weighted Degree = %.2f.",
            num_nodes,
            num_nodes,
            edge_count,
            degrees.min(),
            degrees.max(),
            np.median(degrees),
            np.mean(degrees),
            np.mean(weighted_degrees),
        )
        return self.adj_matrix

    # ------------------------------------------------------------------ #
    # Step 3: Kipf-Welling Symmetric Normalization
    # ------------------------------------------------------------------ #
    def normalize_adjacency(self) -> np.ndarray:
        """
        Computes Kipf-Welling symmetric normalization:
            A_tilde = A + I (adds self-loops)
            D_tilde = diag(sum(A_tilde, axis=1))
            A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
        """
        if self.adj_matrix is None:
            self.build_adjacency_matrix()

        num_nodes = self.adj_matrix.shape[0]

        # Add self-loops (A_tilde = A + I)
        a_tilde = self.adj_matrix + np.eye(num_nodes, dtype=np.float32)

        # Compute degree vector and inverted square root: D_tilde^(-1/2)
        degrees = np.sum(a_tilde, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        d_inv_sqrt[degrees == 0] = 0.0

        # A_hat = D^(-1/2) * A_tilde * D^(-1/2)
        d_mat = np.diag(d_inv_sqrt)
        norm_adj = d_mat @ a_tilde @ d_mat

        self.norm_adj_matrix = norm_adj.astype(np.float32)
        logger.info(
            "Computed symmetric normalized adjacency matrix A_hat (%d x %d).",
            self.norm_adj_matrix.shape[0],
            self.norm_adj_matrix.shape[1],
        )
        return self.norm_adj_matrix

    # ------------------------------------------------------------------ #
    # Step 4: Persist Artifacts
    # ------------------------------------------------------------------ #
    def save_artifacts(self, filename: str = "adjacency.pkl") -> Path:
        """
        Persists the graph dictionary (raw adjacency, normalized adjacency,
        node lookup maps, node dataframe, and metadata) to artifacts/adjacency.pkl.
        """
        if self.norm_adj_matrix is None:
            self.normalize_adjacency()

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.artifacts_dir / filename

        degrees = self.adj_matrix.sum(axis=1)

        artifact_data: Dict[str, Any] = {
            "adj_matrix": self.adj_matrix,
            "norm_adj_matrix": self.norm_adj_matrix,
            "node_to_idx": self.node_to_idx,
            "enc_pair_to_idx": self.enc_pair_to_idx,
            "idx_to_node": self.idx_to_node,
            "node_df": self.node_df,
            "feature_columns": self.feature_columns,
            "metadata": {
                "num_nodes": len(self.node_df),
                "k_neighbors": self.k_neighbors,
                "distance_threshold": self.distance_threshold,
                "total_edges": int(np.sum(self.adj_matrix > 0) / 2),
                "degree_min": float(degrees.min()),
                "degree_max": float(degrees.max()),
                "degree_median": float(np.median(degrees)),
                "degree_mean": float(np.mean(degrees)),
                "degree_std": float(np.std(degrees)),
                "created_at": datetime.now().isoformat(),
                "notes": (
                    "Static weighted spatial graph over 850 unique (STATE/UT, DISTRICT) composite nodes. "
                    "Distance-weighted k-NN (k=5, w=1/(1+d)) among real geographic district nodes; "
                    "non-geographic units (CID, GRP, Railways, STF, etc.) connected strictly to their own state's real districts."
                ),
            },
        }

        joblib.dump(artifact_data, save_path)
        logger.info("Saved graph adjacency artifacts to '%s' (file size: %.2f KB).", save_path, save_path.stat().st_size / 1024)
        return save_path

    # ------------------------------------------------------------------ #
    # Helper: Build Annual Node Feature Matrix X_t (850 x F)
    # ------------------------------------------------------------------ #
    def build_annual_node_features(self, year_index: int) -> np.ndarray:
        """
        Constructs the node feature matrix X_t in R^(850 x 33) for a specified
        year_index (0=2001, ..., 12=2013). Missing district records in that year
        are filled with zeros.
        """
        if self.df is None or self.node_df is None:
            self.load_nodes()

        num_nodes = len(self.node_df)
        num_features = len(self.feature_columns)
        x_t = np.zeros((num_nodes, num_features), dtype=np.float32)

        year_df = self.df[self.df["YEAR_INDEX"] == year_index]

        for _, row in year_df.iterrows():
            state = str(row["STATE/UT"]).strip().upper()
            district = str(row["DISTRICT"]).strip().upper()
            node_idx = self.node_to_idx.get((state, district))

            if node_idx is not None:
                x_t[node_idx] = row[self.feature_columns].to_numpy(dtype=np.float32)

        return x_t

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> Path:
        """Executes full graph construction and persistence pipeline."""
        self.load_nodes()
        self.build_adjacency_matrix()
        self.normalize_adjacency()
        return self.save_artifacts()


def load_adjacency_artifacts(artifacts_path: str = "artifacts/adjacency.pkl") -> Dict[str, Any]:
    """
    Convenience loader for downstream modules (gcn_model.py, hybrid_model_v2.py)
    to load the precomputed static graph adjacency artifacts.
    """
    path = Path(artifacts_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency artifact not found at '{path}'. Run 'python src/graph_utils.py' first."
        )
    return joblib.load(path)


if __name__ == "__main__":
    builder = CrimeGraphBuilder(k_neighbors=5)
    artifact_path = builder.run()
    logger.info("Graph adjacency pipeline executed successfully.")
