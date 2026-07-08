"""
Main Application Entry Point for OncoLens.

This module initializes the Dash server, loads all processed datasets (top 1000 expressions,
genomic annotations, patient Dysregulation Index ratings, and variance rankings) into memory
during server startup, binds layouts, and registers the application callbacks.
"""

import dash
import pandas as pd
from pathlib import Path
from app.layouts import create_layout
from app.callbacks import register_callbacks

# Import configuration settings
from src.config import (
    PROCESSED_DATA_DIR,
    TOP_GENE_COUNT
)

# ------------------------------------------------------------------------------
# 1. Dataset Path Definitions
# ------------------------------------------------------------------------------
PATH_EXPRESSION = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}.csv"
PATH_ANNOTATED = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}_annotated.csv"
PATH_PATIENT_DI = PROCESSED_DATA_DIR / "patient_DI.csv"
PATH_VARIANCE_RANKING = PROCESSED_DATA_DIR / "gene_variance_ranking.csv"
PATH_TOP_PAIRS = PROCESSED_DATA_DIR / "top_gene_pairs.csv"
PATH_NETWORK_EDGES = PROCESSED_DATA_DIR / "gene_network_edges.csv"

# ------------------------------------------------------------------------------
# 2. Server Startup Data Loading
# ------------------------------------------------------------------------------
print("======================================================================")
print("ONCOLENS SERVER STARTUP: Loading Processed Datasets...")
print("======================================================================")

if not PATH_EXPRESSION.exists():
    raise FileNotFoundError(f"Expression file not found at: {PATH_EXPRESSION}. Run preprocessing first.")
if not PATH_ANNOTATED.exists():
    raise FileNotFoundError(f"Annotated file not found at: {PATH_ANNOTATED}. Run annotation first.")
if not PATH_PATIENT_DI.exists():
    raise FileNotFoundError(f"Patient DI file not found at: {PATH_PATIENT_DI}. Run dysregulation calculations first.")
if not PATH_VARIANCE_RANKING.exists():
    raise FileNotFoundError(f"Variance ranking file not found at: {PATH_VARIANCE_RANKING}. Run preprocessing first.")
if not PATH_TOP_PAIRS.exists():
    raise FileNotFoundError(f"Top gene pairs file not found at: {PATH_TOP_PAIRS}. Run pair evaluation utility first.")
if not PATH_NETWORK_EDGES.exists():
    raise FileNotFoundError(f"Gene co-expression network edges not found at: {PATH_NETWORK_EDGES}. Run preprocess_network.py first.")

# Load datasets into memory
print(f" -> Loading preprocessed expressions: {PATH_EXPRESSION.name}...")
df_expression = pd.read_csv(PATH_EXPRESSION)

print(f" -> Loading genomic annotations    : {PATH_ANNOTATED.name}...")
df_annotated = pd.read_csv(PATH_ANNOTATED)

print(f" -> Loading patient DI ratings     : {PATH_PATIENT_DI.name}...")
df_patient_di = pd.read_csv(PATH_PATIENT_DI)

print(f" -> Loading gene variance rankings : {PATH_VARIANCE_RANKING.name}...")
df_variance_ranking = pd.read_csv(PATH_VARIANCE_RANKING)

print(f" -> Loading top gene pairs         : {PATH_TOP_PAIRS.name}...")
df_top_pairs = pd.read_csv(PATH_TOP_PAIRS, index_col="Rank")

print(f" -> Loading co-expression edges    : {PATH_NETWORK_EDGES.name}...")
df_network_edges = pd.read_csv(PATH_NETWORK_EDGES)

# Merge variance values and 1-based rank indexing directly into annotations at startup
df_variance_ranking["Rank"] = df_variance_ranking.index + 1
df_annotated = df_annotated.merge(df_variance_ranking, on="ProbeID", how="left")

print("-" * 70)
print(f"Successfully loaded {len(df_expression)} samples and {df_expression.shape[1] - 2} features.")
print(f"Genomic annotations mapped for {len(df_annotated)} genes.")
print(f"Patient Dysregulation Index loaded for {len(df_patient_di)} patients.")
print(f"Top gene pairs loaded: {len(df_top_pairs)} pairs available.")
print(f"Co-expression edges preloaded: {len(df_network_edges)} edges available.")
print("======================================================================")

# ------------------------------------------------------------------------------
# 3. Pre-compute layout data (avoids duplicating logic in callbacks.py)
# ------------------------------------------------------------------------------
# Gene dropdown options (annotated genes only)
_gene_options_startup = []
for _, row in df_annotated.iterrows():
    sym = row.get("Gene Symbol", "")
    if pd.notna(sym) and str(sym).strip() not in ("", "nan"):
        _gene_options_startup.append({
            "label": f"{sym} ({row['ProbeID']})",
            "value": row["ProbeID"]
        })
_gene_options_startup = sorted(_gene_options_startup, key=lambda x: x["label"])

# Top-20 suggested pairs
_top_20_startup = []
for rank, row in df_top_pairs.head(20).iterrows():
    _top_20_startup.append({
        "label": f"Rank {rank}: {row['Gene X']} & {row['Gene Y']} (Sil: {row['Silhouette Score']:.3f})",
        "value": rank,
    })

# Patient options
_patient_options_startup = [
    {"label": f"Patient {pid}", "value": str(pid)}
    for pid in sorted(df_expression["samples"].unique())
]
_default_patient = _patient_options_startup[0]["value"] if _patient_options_startup else None

# Default gene pair (Rank 1)
_default_x = df_top_pairs.iloc[0]["Probe X"]
_default_y = df_top_pairs.iloc[0]["Probe Y"]

# Default profile gene + top-variance gene symbol.
# Both should refer to the SAME gene: the highest-ranked (by variance) probe
# that actually has an annotated symbol. Previously the KPI looked only at
# the single #1-ranked row, so if that specific probe lacked a symbol the
# "TOP VARIANCE GENE" card showed "—" even though a valid symbol existed
# a few ranks down — exactly what default_profile's loop already guards
# against. One shared search now drives both.
_default_profile = None
_top_gene_sym = "—"
_df_ann_sorted = df_annotated.dropna(subset=["Rank"]).sort_values("Rank")
for _, _row in _df_ann_sorted.iterrows():
    _sym = _row.get("Gene Symbol", "")
    if pd.notna(_sym) and str(_sym).strip() not in ("", "nan"):
        _default_profile = _row["ProbeID"]
        _top_gene_sym = str(_sym)
        break
if not _default_profile:
    _default_profile = _df_ann_sorted.iloc[0]["ProbeID"]

# Best pair display string
_best_pair_row = df_top_pairs.iloc[0]
_best_pair_str = f"{_best_pair_row['Gene X']} & {_best_pair_row['Gene Y']}"

# Simulator default slider values from default patient's baseline expression
from app.config import DEFAULT_SIM_GENE_1, DEFAULT_SIM_GENE_2, DEFAULT_SIM_GENE_3
_sim_val1 = _sim_val2 = _sim_val3 = 5.0
if _default_patient:
    _p_row = df_expression[df_expression["samples"].astype(str) == str(_default_patient)]
    if not _p_row.empty:
        _sim_val1 = float(_p_row[DEFAULT_SIM_GENE_1].values[0])
        _sim_val2 = float(_p_row[DEFAULT_SIM_GENE_2].values[0])
        _sim_val3 = float(_p_row[DEFAULT_SIM_GENE_3].values[0])

# ------------------------------------------------------------------------------
# 4. Dash Server Initialization
# ------------------------------------------------------------------------------
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap"
]

app = dash.Dash(
    __name__,
    assets_folder=str(Path(__file__).parent.parent / "assets"),
    external_stylesheets=external_stylesheets,
    title="OncoLens — Visual Analytics Dashboard",
    suppress_callback_exceptions=True
)

# Build the unified static dashboard layout
app.layout = create_layout(
    n_patients=len(df_expression),
    n_genes=df_expression.shape[1] - 2,
    n_classes=df_expression["type"].nunique(),
    best_pair=_best_pair_str,
    top_gene=_top_gene_sym,
    gene_options=_gene_options_startup,
    top_20_options=_top_20_startup,
    patient_options=_patient_options_startup,
    default_x=_default_x,
    default_y=_default_y,
    default_profile=_default_profile,
    default_patient=_default_patient,
    sim_val1=_sim_val1,
    sim_val2=_sim_val2,
    sim_val3=_sim_val3,
)

# Register event-driven callbacks
register_callbacks(app, df_expression, df_annotated, df_patient_di, df_variance_ranking, df_top_pairs, df_network_edges)

# Expose server wrapper for production WSGI setups
server = app.server

# ------------------------------------------------------------------------------
# 5. Local Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=True, load_dotenv=False)
