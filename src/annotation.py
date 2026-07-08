"""
Genomic Annotation Module for OncoLens.

This module is responsible for:
1. Downloading the GPL570 platform annotation file from NCBI GEO if not already present.
2. Parsing probe set annotations (Gene Symbol, Chromosome, Genomic Start, and Cytoband).
3. Merging these annotations with our preprocessed top 1000 gene expression dataset.
4. Transposing the expression matrix so that rows correspond to genes (annotated) and
   columns correspond to samples, preserving genomic coordinate mapping.
5. Printing a standardized genomic annotation report.
"""

import os
import gzip
import urllib.request
import shutil
import re
import pandas as pd
from pathlib import Path
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TOP_GENE_COUNT
)

def download_annotation_file(target_path: Path) -> None:
    """
    Downloads the GPL570 annotation file from NCBI GEO FTP server.
    """
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
    print(f"Annotation file not found. Downloading from {url}...")
    
    # Use standard urllib with a User-Agent to avoid blocking
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req, timeout=60) as response, open(target_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download completed successfully.")

def parse_annotations(annot_path: Path) -> pd.DataFrame:
    """
    Parses the GPL570.annot.gz file and extracts Probe ID, Gene Symbol, Chromosome,
    Genomic Start Position, and Cytoband.
    """
    print(f"Parsing annotation file: {annot_path}...")
    
    # Find the table start dynamically
    table_start = 0
    with gzip.open(annot_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "!platform_table_begin" in line:
                table_start = i + 1
                break
                
    # Read the tab-separated table
    df_annot = pd.read_csv(
        annot_path,
        compression="gzip",
        sep="\t",
        skiprows=table_start,
        comment="!",
        low_memory=False
    )
    
    parsed_records = []
    
    for _, row in df_annot.iterrows():
        probe_id = str(row["ID"])
        raw_symbol = str(row.get("Gene symbol", ""))
        raw_location = str(row.get("Chromosome location", ""))
        raw_annotation = str(row.get("Chromosome annotation", ""))
        
        # 1. Clean Gene Symbol (extract first of multi-gene mappings)
        gene_symbol = None
        if pd.notna(raw_symbol) and raw_symbol.strip() != "" and raw_symbol != "nan":
            gene_symbol = raw_symbol.split("///")[0].strip()
            
        # 2. Extract Cytoband (from Chromosome location)
        cytoband = None
        if pd.notna(raw_location) and raw_location.strip() != "" and raw_location != "nan":
            cytoband = raw_location.split("///")[0].strip()
            
        # 3. Extract Chromosome and Genomic Start from Chromosome annotation
        # Format: "Chromosome 10, NC_000010.11 (133527363..133539116)"
        chromosome = None
        genomic_start = None
        
        if pd.notna(raw_annotation) and raw_annotation.strip() != "" and raw_annotation != "nan":
            # Extract chromosome (e.g. 2, X, Y)
            chrom_match = re.search(r"Chromosome\s+([^,]+)", raw_annotation)
            if chrom_match:
                chromosome = chrom_match.group(1).strip().split("///")[0].strip()
                
            # Extract genomic start position
            coord_match = re.search(r"\((\d+)\.\.", raw_annotation)
            if coord_match:
                genomic_start = int(coord_match.group(1))
                
        parsed_records.append({
            "ProbeID": probe_id,
            "Gene Symbol": gene_symbol,
            "Chromosome": chromosome,
            "Genomic Start": genomic_start,
            "Cytoband": cytoband
        })
        
    return pd.DataFrame(parsed_records)

def main() -> None:
    # Paths configuration
    annot_file_path = RAW_DATA_DIR / "GPL570.annot.gz"
    processed_input_path = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}.csv"
    processed_output_path = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}_annotated.csv"
    
    # 1. Download annotation file if it doesn't exist
    if not annot_file_path.exists():
        download_annotation_file(annot_file_path)
        
    # 2. Parse the annotation file
    df_annotations = parse_annotations(annot_file_path)
    
    # 3. Load preprocessed expression dataset
    if not processed_input_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at: {processed_input_path}. Run preprocessing first.")
    df_expr = pd.read_csv(processed_input_path)
    
    # Extract gene columns, samples, and types
    samples = df_expr["samples"].tolist()
    types = df_expr["type"].tolist()
    gene_cols = [col for col in df_expr.columns if col not in ["samples", "type"]]
    
    # 4. Filter annotations to keep only our top genes
    df_annot_subset = df_annotations[df_annotations["ProbeID"].isin(gene_cols)].copy()
    
    # 5. Transpose the expression columns so that genes are rows and samples are columns
    df_transposed = df_expr[gene_cols].T
    df_transposed.index.name = "ProbeID"
    df_transposed.columns = [str(s) for s in samples]
    df_transposed = df_transposed.reset_index()
    
    # 6. Merge annotations with transposed expression data
    df_final = pd.merge(df_annot_subset, df_transposed, on="ProbeID", how="right")
    
    # Save the annotated matrix
    df_final.to_csv(processed_output_path, index=False)
    
    # 7. Compute dynamic statistics for the report
    total_genes = len(gene_cols)
    annotated_count = len(df_final.dropna(subset=["Gene Symbol"]))
    missing_count = total_genes - annotated_count
    coverage_pct = (annotated_count / total_genes) * 100 if total_genes > 0 else 0.0
    
    unique_chroms = df_final["Chromosome"].dropna().unique()
    num_chroms = len(unique_chroms)
    chrom_str = ", ".join(sorted([str(c) for c in unique_chroms])) if num_chroms <= 5 else f"{num_chroms} (including X, Y)"
    
    missing_cytoband = len(df_final[df_final["Cytoband"].isna() | (df_final["Cytoband"] == "None")])
    
    print("=========================================")
    print("GENOMIC ANNOTATION REPORT")
    print("=========================================")
    print("")
    print(f"Top genes before merge : {total_genes}")
    print(f"Successfully annotated : {annotated_count}")
    print(f"Missing annotation     : {missing_count}")
    print(f"Coverage               : {coverage_pct:.1f}%")
    print("")
    print(f"Unique chromosomes     : {num_chroms} found")
    print(f"Genes without cytoband : {missing_cytoband}")
    print("")
    print("Annotated dataset saved to:")
    print(f"data/processed/brain_top{TOP_GENE_COUNT}_annotated.csv")
    print("=========================================")

if __name__ == "__main__":
    main()
