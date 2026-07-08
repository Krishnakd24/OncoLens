"""
Evaluation Utility for OncoLens.

This script performs an exhaustive grid search over all unique pairs of annotated
genes in the Top-1000 variance-filtered dataset. For each pair, it computes the
2D Silhouette Score using true clinical subtype labels, ranks them by separation
strength, and saves the results to a CSV. It uses multiprocessing for speed.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

# Import configurations
from src.config import (
    PROCESSED_DATA_DIR,
    TOP_GENE_COUNT
)

# ------------------------------------------------------------------------------
# Process-Level Global Cache for Multiprocessing Workers
# ------------------------------------------------------------------------------
_X_scaled: np.ndarray = None
_masks: list = None
_sizes: np.ndarray = None
_N: int = None


def init_worker(X_scaled: np.ndarray, masks: list, sizes: np.ndarray, N: int) -> None:
    """
    Initializes global variables for each worker process to avoid IPC transmission overhead.
    """
    global _X_scaled, _masks, _sizes, _N
    _X_scaled = X_scaled
    _masks = masks
    _sizes = sizes
    _N = N


def evaluate_pair(indices: tuple) -> tuple:
    """
    Computes the 2D Silhouette score for a specific pair of gene indices.
    
    Uses squared Euclidean distances internally. The Silhouette coefficient
    compares ratios of a_i vs b_i — since squaring is a monotonic transform
    on non-negative values, the relative ordering and sign of (b-a)/max(a,b)
    is preserved, while avoiding ~478K sqrt() calls.
    
    Includes guards for:
    - Single-sample clusters (a_i = 0 by definition)
    - Empty inter-cluster distance sets
    - NaN/Inf in the final score
    
    Args:
        indices (tuple): A tuple containing (i, j) column indices.
        
    Returns:
        tuple: (i, j, silhouette_score)
    """
    i, j = indices
    c1 = _X_scaled[:, i]
    c2 = _X_scaled[:, j]
    
    # 1. Compute pairwise squared Euclidean distance matrix D²
    #    Avoiding sqrt: silhouette ratios are preserved under monotonic transforms.
    dx = c1[:, np.newaxis] - c1
    dy = c2[:, np.newaxis] - c2
    D = dx * dx + dy * dy  # Squared distances (no sqrt needed)
    
    # 2. Calculate Silhouette elements (a_i and b_i)
    a = np.zeros(_N)
    b = np.full(_N, np.inf)  # Initialize to inf so np.minimum works correctly
    
    for label_idx, mask in enumerate(_masks):
        n_c = _sizes[label_idx]
        
        # a_i: mean intra-cluster distance
        if n_c > 1:
            a[mask] = D[mask][:, mask].sum(axis=1) / (n_c - 1)
        else:
            # Single-sample cluster: a_i = 0 by Silhouette definition
            a[mask] = 0.0
        
        # b_i: minimum mean inter-cluster distance across all other clusters
        for other_idx, other_mask in enumerate(_masks):
            if other_idx == label_idx:
                continue
            n_other = _sizes[other_idx]
            if n_other == 0:
                continue
            mean_d = D[mask][:, other_mask].mean(axis=1)
            b[mask] = np.minimum(b[mask], mean_d)
    
    # Fallback: if a sample had no valid other-cluster (shouldn't happen with 5 subtypes)
    b[np.isinf(b)] = 0.0
    
    denom = np.maximum(a, b)
    denom[denom == 0] = 1.0
    score = float(((b - a) / denom).mean())
    
    # NaN safety: return 0.0 for degenerate cases
    if np.isnan(score) or np.isinf(score):
        score = 0.0
    
    return i, j, score


def main() -> None:
    # 1. Define input/output paths
    path_expression = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}.csv"
    path_annotated = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}_annotated.csv"
    path_output = PROCESSED_DATA_DIR / "top_gene_pairs.csv"
    
    if not path_expression.exists() or not path_annotated.exists():
        raise FileNotFoundError(
            "Required preprocessed files do not exist. Please run preprocessing and annotation first."
        )

    print("Loading preprocessed expression matrix...")
    df_expression = pd.read_csv(path_expression)
    df_annotated = pd.read_csv(path_annotated)

    # 2. Filter for annotated genes only (must have a valid Gene Symbol)
    df_valid = df_annotated.dropna(subset=["Gene Symbol"])
    df_valid = df_valid[df_valid["Gene Symbol"].str.strip() != ""]
    df_valid = df_valid[df_valid["Gene Symbol"] != "nan"]
    
    genes_list = df_valid[["ProbeID", "Gene Symbol"]].to_dict(orient="records")
    num_genes = len(genes_list)
    
    print(f"Isolated {num_genes} annotated genes from Top-1000 dataset.")
    
    # 3. Standardize expression values for all annotated genes
    probes = [g["ProbeID"] for g in genes_list]
    X = df_expression[probes].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    
    # Extract labels
    y = df_expression["type"].to_numpy()
    unique_labels = np.unique(y)
    masks = [y == label for label in unique_labels]
    sizes = np.array([mask.sum() for mask in masks])
    N = len(y)
    
    # 4. Generate all unique pairs of indices
    pairs = []
    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            pairs.append((i, j))
            
    num_pairs = len(pairs)
    print(f"Generated {num_pairs:,} unique pairs for grid-search evaluation.")
    
    # 5. Execute evaluation using multiprocessing Pool
    start_time = time.time()
    
    # Use available CPU cores minus one to keep system responsive
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting multiprocessing Pool using {num_workers} parallel workers...")
    
    results = []
    with Pool(processes=num_workers, initializer=init_worker, initargs=(X_scaled, masks, sizes, N)) as pool:
        # Using imap_unordered for speed and low memory usage, chunksize to optimize context switching
        for i, j, score in pool.imap_unordered(evaluate_pair, pairs, chunksize=1000):
            results.append((i, j, score))
            
            # Print periodic updates
            if len(results) % 50000 == 0:
                pct = (len(results) / num_pairs) * 100
                print(f" -> Evaluated {len(results):,} / {num_pairs:,} pairs ({pct:.1f}% complete)...")
                
    runtime = time.time() - start_time
    print(f"Grid-search completed in {runtime:.2f} seconds.")
    
    # 6. Map indices back to labels and save the ranked results
    print("Formatting and ranking results...")
    formatted_results = []
    for i, j, score in results:
        gene_x = genes_list[i]["Gene Symbol"]
        probe_x = genes_list[i]["ProbeID"]
        gene_y = genes_list[j]["Gene Symbol"]
        probe_y = genes_list[j]["ProbeID"]
        formatted_results.append({
            "Gene X": gene_x,
            "Probe X": probe_x,
            "Gene Y": gene_y,
            "Probe Y": probe_y,
            "Silhouette Score": score
        })
        
    df_results = pd.DataFrame(formatted_results)
    # Sort descending by score
    df_results = df_results.sort_values(by="Silhouette Score", ascending=False).reset_index(drop=True)
    df_results.index += 1  # 1-based ranking index
    df_results.index.name = "Rank"
    
    # Save output CSV
    df_results.to_csv(path_output)
    print(f"Successfully saved ranked results to: {path_output}")
    
    # 7. Print the final console report
    best_row = df_results.iloc[0]
    best_pair_str = f"{best_row['Gene X']} ({best_row['Probe X']}) & {best_row['Gene Y']} ({best_row['Probe Y']})"
    
    print("\n" + "="*50)
    print("PAIR EVALUATION REPORT")
    print("="*50)
    print(f"Genes evaluated       : {num_genes}")
    print(f"Pairs evaluated       : {num_pairs:,}")
    print(f"Runtime               : {runtime:.2f} seconds")
    print(f"Best pair             : {best_pair_str}")
    print(f"Best silhouette score : {best_row['Silhouette Score']:.6f}")
    print("-"*50)
    print("TOP 20 PAIRS:")
    print("-"*50)
    
    top_20 = df_results.head(20)
    print(f"{'Rank':<6}{'Gene X (Probe)':<22}{'Gene Y (Probe)':<22}{'Silhouette':<12}")
    print("-"*62)
    for rank, row in top_20.iterrows():
        gx = f"{row['Gene X']} ({row['Probe X']})"
        gy = f"{row['Gene Y']} ({row['Probe Y']})"
        print(f"{rank:<6}{gx:<22}{gy:<22}{row['Silhouette Score']:.6f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
