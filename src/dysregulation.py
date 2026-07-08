"""
Dysregulation Metric Calculations Module for OncoLens.

This module computes the Dysregulation Index (DI) for every patient using the
annotated expression dataset. It calculates the mean and standard deviation of
each gene across the healthy "normal" tissue samples, standardizes the cancer
patient expression profiles relative to this baseline, and sums the absolute
z-scores. It outputs a patient-level DI table and a boxplot visualization.
"""

import os
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import path constants from src.config
from src.config import (
    PROCESSED_DATA_DIR,
    TOP_GENE_COUNT,
    TARGET_COLUMN
)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the annotated dataset and the preprocessed expression metadata.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_annotated, df_preprocessed)
    """
    annotated_path = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}_annotated.csv"
    preprocessed_path = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}.csv"
    
    if not annotated_path.exists():
        raise FileNotFoundError(f"Annotated dataset not found at: {annotated_path}. Run annotation first.")
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at: {preprocessed_path}. Run preprocessing first.")
        
    print(f"Loading annotated dataset from {annotated_path}...")
    df_annotated = pd.read_csv(annotated_path)
    
    print(f"Loading preprocessed metadata from {preprocessed_path}...")
    df_preprocessed = pd.read_csv(preprocessed_path)
    
    return df_annotated, df_preprocessed


def get_sample_mappings(df_preprocessed: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """
    Maps each sample ID to its clinical tumor type and isolates the normal sample IDs.

    Args:
        df_preprocessed (pd.DataFrame): The preprocessed expression DataFrame.

    Returns:
        Tuple[Dict[str, str], List[str]]:
            - sample_to_type: Dictionary mapping sample ID string to tumor type string.
            - normal_samples: List of sample ID strings corresponding to the 'normal' class.
    """
    sample_to_type = df_preprocessed.set_index(df_preprocessed["samples"].astype(str))[TARGET_COLUMN].to_dict()
    normal_samples = [s for s, t in sample_to_type.items() if t == "normal"]
    return sample_to_type, normal_samples


def compute_normal_stats(
    df_annotated: pd.DataFrame, 
    normal_samples: List[str], 
    percentile_floor: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes mean and standard deviation across normal controls for each gene,
    applying robust regularization against near-zero-variance genes.

    Instead of an arbitrary epsilon threshold that masks genes out, we compute a
    biologically meaningful floor: the 5th percentile of all gene standard
    deviations across normal samples. Genes with std < floor get their std
    clamped UP to the floor, ensuring they still contribute to the DI but cannot
    dominate through division-by-near-zero (which amplifies microarray noise).

    Args:
        df_annotated (pd.DataFrame): The annotated gene expression DataFrame (genes as rows).
        normal_samples (List[str]): List of normal sample ID columns.
        percentile_floor (float): Percentile of std distribution to use as the
            regularization floor. Default 5.0 (the 5th percentile).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - mean_normal: Array of means for each gene.
            - std_normal: Array of regularized standard deviations for each gene.
            - valid_mask: Boolean mask (all True after regularization; kept for API compat).
    """
    # Extract only the normal sample columns
    normal_data = df_annotated[normal_samples].to_numpy()
    
    mean_normal = np.mean(normal_data, axis=1)
    std_normal = np.std(normal_data, axis=1, ddof=1)  # Sample standard deviation (ddof=1)
    
    # ── Robust Regularization ────────────────────────────────────────────────
    # Compute the 5th percentile of all gene standard deviations as a robust
    # biological floor.  Any gene with std < floor is clamped up to floor.
    # This avoids division-by-near-zero while keeping ALL genes in the DI.
    sigma_floor = float(np.percentile(std_normal[std_normal > 0], percentile_floor))
    sigma_floor = max(sigma_floor, 1e-8)  # Absolute safety net
    
    n_clamped = int(np.sum(std_normal < sigma_floor))
    std_normal = np.maximum(std_normal, sigma_floor)
    
    print(f"  Regularization floor (P{percentile_floor:.0f}): {sigma_floor:.6f}")
    print(f"  Genes clamped to floor: {n_clamped} / {len(std_normal)}")
    
    # All genes are now valid after clamping (no masking needed)
    valid_mask = np.ones(len(std_normal), dtype=bool)
    
    return mean_normal, std_normal, valid_mask


def calculate_dysregulation_index(
    df_annotated: pd.DataFrame,
    patient_cols: List[str],
    mean_normal: np.ndarray,
    std_normal: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Vectorized computation of the Dysregulation Index (DI) for all patients.

    Formula:
        DI_p = Sum_g | (x_pg - Mean_normal_g) / Std_normal_g |

    Args:
        df_annotated (pd.DataFrame): The annotated gene expression DataFrame (genes as rows).
        patient_cols (List[str]): List of patient sample ID columns.
        mean_normal (np.ndarray): Normal mean vector of shape (num_genes,).
        std_normal (np.ndarray): Normal std dev vector of shape (num_genes,).
        valid_mask (np.ndarray): Boolean mask of shape (num_genes,) indicating non-flat genes.

    Returns:
        np.ndarray: Array of DI values for each patient in patient_cols.
    """
    # Filter genes that have valid variation
    valid_mean = mean_normal[valid_mask][:, np.newaxis]
    valid_std = std_normal[valid_mask][:, np.newaxis]
    
    # Extract expression matrix (valid genes, all patients)
    expr_matrix = df_annotated.loc[valid_mask, patient_cols].to_numpy()
    
    # Compute z-scores and sum absolute values across columns
    z_scores = np.abs((expr_matrix - valid_mean) / valid_std)
    dis = np.sum(z_scores, axis=0)
    
    return dis


def save_results(df_di: pd.DataFrame) -> Path:
    """
    Saves the computed patient-level DI table to a CSV file.

    Args:
        df_di (pd.DataFrame): Patient DI DataFrame.

    Returns:
        Path: Path to the saved file.
    """
    output_path = PROCESSED_DATA_DIR / "patient_DI.csv"
    df_di.to_csv(output_path, index=False)
    return output_path


def generate_validation_plot(df_di: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Generates a boxplot/violin plot of the Dysregulation Index grouped by tumor type.
    Saves the figure to processed data and artifact directory.

    Args:
        df_di (pd.DataFrame): Patient DI DataFrame.

    Returns:
        Tuple[Path, Path]: (processed_plot_path, artifact_plot_path)
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Sort tumor types by median DI for a cleaner visual representation
    order = df_di.groupby("TumorType")["DysregulationIndex"].median().sort_values().index
    
    # Create boxplot layout
    sns.boxplot(
        x="TumorType", 
        y="DysregulationIndex", 
        data=df_di, 
        order=order,
        palette="viridis",
        hue="TumorType",
        legend=False,
        width=0.6
    )
    sns.stripplot(
        x="TumorType", 
        y="DysregulationIndex", 
        data=df_di, 
        order=order,
        color="black", 
        alpha=0.4, 
        size=5, 
        jitter=0.2
    )
    
    plt.title("Genomic Dysregulation Index (DI) across Brain Tumor Subtypes", fontsize=16, pad=15)
    plt.xlabel("Tissue/Tumor Subtype", fontsize=12)
    plt.ylabel("Dysregulation Index (DI)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Save to data/processed
    processed_plot_path = PROCESSED_DATA_DIR / "patient_DI_boxplot.png"
    plt.savefig(processed_plot_path, dpi=300, bbox_inches='tight')
    
    # Save to artifact directory
    artifact_dir = Path("C:/Users/dhruv/.gemini/antigravity/brain/c2eb589e-1013-4280-97e6-898b9b06146e")
    artifact_plot_path = artifact_dir / "patient_DI_boxplot.png"
    if artifact_dir.exists():
        plt.savefig(artifact_plot_path, dpi=300, bbox_inches='tight')
        
    plt.close()
    return processed_plot_path, artifact_plot_path


def main() -> None:
    # 1. Load data
    df_annotated, df_preprocessed = load_datasets()
    
    # Identify patient columns (all columns except the metadata ones)
    metadata_cols = {"ProbeID", "Gene Symbol", "Chromosome", "Genomic Start", "Cytoband"}
    patient_cols = [col for col in df_annotated.columns if col not in metadata_cols]
    
    # 2. Get sample clinical type mappings
    sample_to_type, normal_samples = get_sample_mappings(df_preprocessed)
    
    # 3. Compute baseline mean and std dev across healthy controls (normal samples)
    mean_normal, std_normal, valid_mask = compute_normal_stats(df_annotated, normal_samples)
    
    total_genes = len(df_annotated)
    skipped_genes = int(np.sum(~valid_mask))
    used_genes = int(np.sum(valid_mask))
    
    # 4. Compute Dysregulation Index for all patients
    dis = calculate_dysregulation_index(df_annotated, patient_cols, mean_normal, std_normal, valid_mask)
    
    # 5. Create final DI DataFrame
    df_di = pd.DataFrame({
        "SampleID": patient_cols,
        "TumorType": [sample_to_type[col] for col in patient_cols],
        "DysregulationIndex": dis
    })
    
    # 6. Save results to disk
    output_csv = save_results(df_di)
    
    # 7. Generate boxplot visualization
    processed_plot, _ = generate_validation_plot(df_di)
    
    # 8. Print report
    print("=====================================")
    print("DYSREGULATION INDEX REPORT")
    print("=====================================")
    print(f"Number of patients       : {len(patient_cols)}")
    print(f"Number of genes used     : {used_genes}")
    print(f"Genes skipped            : {skipped_genes}")
    print(f"Minimum DI               : {dis.min():.2f}")
    print(f"Maximum DI               : {dis.max():.2f}")
    print(f"Mean DI                  : {dis.mean():.2f}")
    print(f"Standard Deviation of DI : {dis.std():.2f}")
    print(f"Output file location     : {output_csv}")
    print("=====================================")


if __name__ == "__main__":
    main()
