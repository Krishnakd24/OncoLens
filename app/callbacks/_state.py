"""
OncoLens Callbacks - Shared State.

Global data stores populated once at server startup.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
from dash import Input, Output, html, Dash, callback_context, no_update
from app.layouts.theme import (
    PLOT_TEMPLATE,
    PLOT_PAPER_BG, PLOT_PLOT_BG, PLOT_GRID, PLOT_ZEROLINE,
    PLOT_TICK_COLOR, PLOT_TITLE_COLOR, PLOT_AXIS_LABEL_COLOR,
    COLOR_TEXT_SECONDARY, COLOR_BORDER,
)

# Global data stores
df_expr_global: pd.DataFrame = None
df_annot_global: pd.DataFrame = None
df_di_global: pd.DataFrame = None
df_rank_global: pd.DataFrame = None
df_top_pairs_global: pd.DataFrame = None
df_network_edges_global: pd.DataFrame = None
network_layout_pos: dict = None
DEFAULT_GENE_X: str = None
DEFAULT_GENE_Y: str = None
default_profile_probe_global: str = None
top_20_options_global: list = []
gene_options_global: list = []
gene_cols_global: list = []
centroids_matrix_global: np.ndarray = None
patient_options_global: list = []
subtype_names_list: list = ["normal", "ependymoma", "glioblastoma", "medulloblastoma", "pilocytic_astrocytoma"]


def initialise_globals(df_expression, df_annotated, df_patient_di, df_variance_ranking, df_top_pairs, df_network_edges):
    """Populate all global state from the loaded DataFrames."""
    global df_expr_global, df_annot_global, df_di_global, df_rank_global, df_top_pairs_global
    global df_network_edges_global, network_layout_pos
    global DEFAULT_GENE_X, DEFAULT_GENE_Y, default_profile_probe_global, top_20_options_global, gene_options_global
    global centroids_matrix_global, gene_cols_global, patient_options_global

    # Store dataframes globally in the module
    df_expr_global = df_expression
    df_annot_global = df_annotated
    df_di_global = df_patient_di
    df_rank_global = df_variance_ranking
    df_top_pairs_global = df_top_pairs
    df_network_edges_global = df_network_edges

    # Initialize simulator configurations
    gene_cols_global = [col for col in df_expression.columns if col not in ["samples", "type"]]
    patient_options_global = [{"label": f"Patient {pid}", "value": str(pid)} for pid in sorted(df_expression["samples"].unique())]

    # Calculate subtype centroids
    print(" -> Computing diagnostic centroids for Virtual Expression Assayer...")
    centroids_list = []
    for subtype in subtype_names_list:
        df_sub = df_expression[df_expression["type"] == subtype]
        centroids_list.append(df_sub[gene_cols_global].mean(axis=0).values)
    centroids_matrix_global = np.array(centroids_list)

    # Pre-generate co-expression network spring layout positions once at startup
    print(" -> Constructing co-expression network graph for spring layout calculations...")
    G_base = nx.Graph()
    connected_probes = set(df_network_edges["Probe X"]).union(set(df_network_edges["Probe Y"]))

    df_nodes = df_annotated[df_annotated["ProbeID"].isin(connected_probes)]
    for _, row in df_nodes.iterrows():
        G_base.add_node(row["ProbeID"])

    for _, row in df_network_edges.iterrows():
        G_base.add_edge(row["Probe X"], row["Probe Y"])

    print(" -> Precomputing network layout coordinates using spring_layout...")
    network_layout_pos = nx.spring_layout(G_base, seed=42, k=0.12, iterations=50)

    # Extract Rank 1 dynamically as the initial default pair
    DEFAULT_GENE_X = df_top_pairs.iloc[0]["Probe X"]
    DEFAULT_GENE_Y = df_top_pairs.iloc[0]["Probe Y"]

    # Resolve the highest-variance annotated gene dynamically as default profile explorer gene
    df_annot_sorted = df_annotated.dropna(subset=["Rank"]).sort_values(by="Rank")
    for _, row in df_annot_sorted.iterrows():
        symbol = row["Gene Symbol"]
        if pd.notna(symbol) and str(symbol).strip() != "" and str(symbol) != "nan":
            default_profile_probe_global = row["ProbeID"]
            break
    if not default_profile_probe_global:
        default_profile_probe_global = df_annot_sorted.iloc[0]["ProbeID"]

    # Build Top 20 suggested options
    top_20 = []
    for rank, row in df_top_pairs.head(20).iterrows():
        label = f"Rank {rank}: {row['Gene X']} & {row['Gene Y']} (Sil: {row['Silhouette Score']:.3f})"
        top_20.append({"label": label, "value": rank})
    top_20_options_global = top_20

    # Pre-generate dropdown options (ONLY display genes with valid annotations/symbols)
    options = []
    for _, row in df_annotated.iterrows():
        probe_id = row["ProbeID"]
        symbol = row["Gene Symbol"]
        if pd.notna(symbol) and str(symbol).strip() != "" and str(symbol) != "nan":
            label = f"{symbol} ({probe_id})"
            options.append({"label": label, "value": probe_id})

    # Sort options by symbol alphabetically
    gene_options_global = sorted(options, key=lambda x: x["label"])
