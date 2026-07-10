from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import json

from src.config import (
    PROCESSED_DATA_DIR,
    TOP_GENE_COUNT
)

app = FastAPI(title="OncoLens API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for datasets
df_expression = None
df_annotated = None
df_patient_di = None
df_variance_ranking = None
df_top_pairs = None
df_network_edges = None

# Startup Constants
app_config = {}

@app.on_event("startup")
def load_data():
    global df_expression, df_annotated, df_patient_di
    global df_variance_ranking, df_top_pairs, df_network_edges
    global app_config

    PATH_EXPRESSION = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}.csv"
    PATH_ANNOTATED = PROCESSED_DATA_DIR / f"brain_top{TOP_GENE_COUNT}_annotated.csv"
    PATH_PATIENT_DI = PROCESSED_DATA_DIR / "patient_DI.csv"
    PATH_VARIANCE_RANKING = PROCESSED_DATA_DIR / "gene_variance_ranking.csv"
    PATH_TOP_PAIRS = PROCESSED_DATA_DIR / "top_gene_pairs.csv"
    PATH_NETWORK_EDGES = PROCESSED_DATA_DIR / "gene_network_edges.csv"

    print("Loading data...")
    df_expression = pd.read_csv(PATH_EXPRESSION)
    df_annotated = pd.read_csv(PATH_ANNOTATED)
    df_patient_di = pd.read_csv(PATH_PATIENT_DI)
    df_variance_ranking = pd.read_csv(PATH_VARIANCE_RANKING)
    df_top_pairs = pd.read_csv(PATH_TOP_PAIRS, index_col="Rank")
    df_network_edges = pd.read_csv(PATH_NETWORK_EDGES)

    # Merge variance
    df_variance_ranking["Rank"] = df_variance_ranking.index + 1
    df_annotated = df_annotated.merge(df_variance_ranking, on="ProbeID", how="left")

    # Pre-compute layout options
    gene_options = []
    for _, row in df_annotated.iterrows():
        sym = row.get("Gene Symbol", "")
        if pd.notna(sym) and str(sym).strip() not in ("", "nan"):
            gene_options.append({
                "label": f"{sym} ({row['ProbeID']})",
                "value": row["ProbeID"]
            })
    gene_options = sorted(gene_options, key=lambda x: x["label"])

    top_20_options = []
    for rank, row in df_top_pairs.head(20).iterrows():
        top_20_options.append({
            "label": f"Rank {rank}: {row['Gene X']} & {row['Gene Y']} (Sil: {row['Silhouette Score']:.3f})",
            "value": rank,
        })

    patient_options = [
        {"label": f"Patient {pid}", "value": str(pid)}
        for pid in sorted(df_expression["samples"].unique())
    ]
    
    app_config = {
        "gene_options": gene_options,
        "top_20_options": top_20_options,
        "patient_options": patient_options,
        "default_patient": patient_options[0]["value"] if patient_options else None,
        "default_x": df_top_pairs.iloc[0]["Probe X"],
        "default_y": df_top_pairs.iloc[0]["Probe Y"]
    }
    print("Data loading complete.")

@app.get("/api/config")
def get_config():
    return app_config

@app.get("/api/ping")
def ping():
    return {"status": "ok"}

from api.routers import contour, hotspots, network, profiles
app.include_router(contour.router)
app.include_router(hotspots.router)
app.include_router(network.router)
app.include_router(profiles.router)


