"""
Data Loaders
============
All dataset loading functions for optimal design experiments.
Each function returns (M, A_dict, m, n) — the standard problem setup.
"""

import numpy as np
import itertools
import io
import zipfile
import requests
import networkx as nx

from .data_utils import DataGenerator


# ========================================================
# Optimal Design Datasets
# ========================================================

def load_synthetic(m=900, n=50, sd_big=10, sd_small=5):
    """
    Synthetic block-diagonal normal data.

    Args:
        m (int): Number of design points (must be even)
        n (int): Feature dimension (must be even)
        sd_big (float): Std for large-variance block
        sd_small (float): Std for small-variance block

    Returns:
        M (np.ndarray): Zero matrix (n x n)
        A_dict (dict): Outer product matrices
        m (int): Number of candidates
        n (int): Feature dimension
    """
    print(f"[Data] Generating synthetic data: m={m}, n={n}, sd_big={sd_big}, sd_small={sd_small}")
    A_dict, _ = DataGenerator.generate_small_big_normal(m, n, sd_big, sd_small)
    M = np.zeros((n, n))
    return M, A_dict, m, n


def load_housing():
    """
    Housing dataset from LIBSVM (housing_scale).
    506 samples, 13 features.

    Returns:
        M, A_dict, m, n
    """
    from sklearn.datasets import load_svmlight_file
    from scipy import sparse
    from io import BytesIO

    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale"
    print("[Data] Downloading housing_scale from LIBSVM...")
    response = requests.get(url)
    X_raw, _ = load_svmlight_file(BytesIO(response.content))
    if sparse.issparse(X_raw):
        X_raw = X_raw.toarray()

    m, n = X_raw.shape
    print(f"[Data] Loaded: m={m}, n={n}")
    A_dict = DataGenerator.dictionarize(X_raw)
    M = np.zeros((n, n))
    return M, A_dict, m, n


def load_regensburg():
    """
    Regensburg Pediatric Appendicitis dataset (UCI id=938).
    Applies paper-style cleaning: drop cols with >40 missing,
    drop rows with any missing, one-hot encode, standardize.

    Returns:
        M, A_dict, m, n
    """
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    print("[Data] Loading Regensburg dataset from UCI (id=938)...")
    dataset = fetch_ucirepo(id=938)
    X_df = dataset.data.features.copy()
    print(f"  Original shape: {X_df.shape}")

    # Drop columns with >40 missing
    cols_to_drop = X_df.isnull().sum()[X_df.isnull().sum() > 40].index
    X_df = X_df.drop(columns=cols_to_drop)

    # Drop rows with any missing
    X_df = X_df.dropna(axis=0, how='any')
    print(f"  After cleaning: {X_df.shape}")

    # Drop single-value columns
    X_df = X_df.drop(columns=X_df.nunique()[X_df.nunique() <= 1].index)

    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
    num_cols = X_df.select_dtypes(exclude=['object', 'category']).columns

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X_cat = encoder.fit_transform(X_df[cat_cols])
        X_final = np.hstack([X_df[num_cols].values, X_cat])
    else:
        X_final = X_df[num_cols].values

    X_final = StandardScaler().fit_transform(X_final)
    m, n = X_final.shape
    print(f"  Final shape: m={m}, n={n}")

    A_dict = {i: np.outer(X_final[i], X_final[i]) for i in range(m)}
    M = np.eye(n) * 1e-7
    return M, A_dict, m, n


# ========================================================
# Graph Augmentation Datasets
# ========================================================

def _setup_graph_problem(G):
    """
    Convert a NetworkX graph into the standard (M, A_dict, m, n) problem format.
    Candidate edges = all possible edges minus existing edges (capped at 1500).
    """
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = nx.convert_node_labels_to_integers(G.subgraph(largest).copy())

    B_base = nx.incidence_matrix(G, oriented=True).toarray().T
    m_base, N_nodes = B_base.shape
    n = N_nodes - 1

    M = np.zeros((n, n))
    X_base = B_base[:, :-1]
    for i in range(m_base):
        M += np.outer(X_base[i], X_base[i])

    edges_existing = set(tuple(sorted(e)) for e in G.edges())
    all_possible = set(itertools.combinations(range(N_nodes), 2))
    edges_candidates = list(all_possible - edges_existing)

    if len(edges_candidates) > 1500:
        import random
        random.seed(42)
        edges_candidates = random.sample(edges_candidates, 1500)

    A_dict = {}
    for idx, (u, v) in enumerate(edges_candidates):
        vec = np.zeros(n)
        if u < n: vec[u] = 1
        if v < n: vec[v] = -1
        A_dict[idx] = np.outer(vec, vec)

    m = len(A_dict)
    print(f"[Setup] Nodes={N_nodes}, Base Edges={m_base}, Candidates={m}")
    return M, A_dict, m, n


def load_graph_synthetic(graph_type='WS', n_nodes=30):
    """
    Synthetic graph: Watts-Strogatz or Barabasi-Albert.

    Args:
        graph_type (str): 'WS' or 'BA'
        n_nodes (int): Number of nodes

    Returns:
        M, A_dict, m, n
    """
    if graph_type == 'BA':
        print(f"[Graph] Barabasi-Albert (n={n_nodes})")
        G = nx.barabasi_albert_graph(n_nodes, 2, seed=42)
    elif graph_type == 'WS':
        print(f"[Graph] Watts-Strogatz (n={n_nodes})")
        for seed in range(42, 100):
            G = nx.connected_watts_strogatz_graph(n_nodes, 4, 0.3, seed=seed)
            if nx.is_connected(G):
                break
    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Choose 'WS' or 'BA'.")

    return _setup_graph_problem(G)


def load_graph_real(name):
    """
    Real-world graph datasets.

    Args:
        name (str): 'Karate', 'Dolphins', or 'TrainTerrorists'

    Returns:
        M, A_dict, m, n
    """
    import pandas as pd

    print(f"[Graph] Loading '{name}'...")

    if name == 'Karate':
        G = nx.karate_club_graph()

    elif name == 'Dolphins':
        url = "https://websites.umich.edu/~mejn/netdata/dolphins.zip"
        response = requests.get(url, timeout=15)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            gml_files = [f for f in z.namelist() if f.endswith('.gml')]
            with z.open(gml_files[0]) as f:
                G = nx.read_gml(f)

    elif name == 'TrainTerrorists':
        url = "https://networks.skewed.de/net/train_terrorists/files/train_terrorists.csv.zip"
        response = requests.get(url, timeout=30)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            with z.open(csv_files[0]) as f:
                edges_df = pd.read_csv(f)
                G = nx.Graph()
                cols = edges_df.columns[:2]
                for _, row in edges_df.iterrows():
                    G.add_edge(int(row[cols[0]]), int(row[cols[1]]))
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'Karate', 'Dolphins', or 'TrainTerrorists'.")

    G = nx.convert_node_labels_to_integers(G)
    print(f"  Loaded: N={G.number_of_nodes()}, E={G.number_of_edges()}")
    return _setup_graph_problem(G)
