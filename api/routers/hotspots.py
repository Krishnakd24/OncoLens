from fastapi import APIRouter
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pydantic import BaseModel
import json

router = APIRouter()

class HotspotsRequest(BaseModel):
    selected_chr: str = "All"
    selected_probe: str | None = None

@router.post("/api/hotspots")
def get_hotspots_plot(req: HotspotsRequest):
    import api.main as main_app
    df_annot_global = main_app.df_annotated

    selected_chr = req.selected_chr
    selected_probe = req.selected_probe

    # Filter for annotated genes that have valid chromosome coordinates
    df_plot = df_annot_global.dropna(subset=["Chromosome", "Genomic Start"]).copy()
    
    # Filter by chromosome selector
    if selected_chr != "All":
        df_plot = df_plot[df_plot["Chromosome"] == selected_chr]
        
    fig = go.Figure()
    
    # Determine chromosomal order on y-axis category track
    if selected_chr == "All":
        chroms_order = [f"Chr{c}" for c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']]
        chroms_order.reverse() # Reverse so Chr1 is at top, ChrY at bottom
    else:
        chroms_order = [f"Chr{selected_chr}"]

    df_plot["ChrTrack"] = "Chr" + df_plot["Chromosome"].astype(str)
    
    if selected_chr == "All":
        base_size = 3.0
        scale_coeff = 9.0
        selected_marker_size = 18.0
        selected_marker_outline_width = 2.0
    else:
        base_size = 8.0
        scale_coeff = 18.0
        selected_marker_size = 30.0
        selected_marker_outline_width = 3.0
        
    var_min = df_plot["Variance"].min()
    var_max = df_plot["Variance"].max()
    var_range = var_max - var_min if var_max > var_min else 1.0
    
    normalized_variance = (df_plot["Variance"] - var_min) / var_range
    marker_sizes = base_size + scale_coeff * np.sqrt(normalized_variance)
    
    fig.add_trace(go.Scatter(
        x=df_plot["Genomic Start"],
        y=df_plot["ChrTrack"],
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=df_plot["Rank"],
            colorscale="Plasma",
            showscale=True,
            reversescale=True,
            opacity=0.8,
            line=dict(width=0.5, color="#1e293b")
        ),
        text=df_plot["Gene Symbol"],
        customdata=df_plot["ProbeID"],
    ))
    
    # Highlighting Selected Gene Marker
    if not selected_probe or selected_probe not in df_plot["ProbeID"].values:
        if not df_plot.empty:
            df_sorted = df_plot.dropna(subset=["Rank"]).sort_values(by="Rank")
            if not df_sorted.empty:
                selected_probe = df_sorted.iloc[0]["ProbeID"]
                
    if selected_probe and selected_probe in df_plot["ProbeID"].values:
        df_sel = df_plot[df_plot["ProbeID"] == selected_probe]
        if not df_sel.empty:
            row_sel = df_sel.iloc[0]
            fig.add_trace(go.Scatter(
                x=[row_sel["Genomic Start"]],
                y=[row_sel["ChrTrack"]],
                mode="markers",
                marker=dict(
                    size=selected_marker_size,
                    color="rgba(0,0,0,0)",
                    line=dict(width=selected_marker_outline_width, color="#10B981")
                ),
                showlegend=False,
                hoverinfo="skip"
            ))
    
    title_text = "Genome-Wide Expression Variance Hotspots" if selected_chr == "All" else f"Chromosome {selected_chr} Hotspot Loci"
    
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text),
        yaxis=dict(type="category", categoryarray=chroms_order, categoryorder="array"),
        margin=dict(l=45, r=20, t=35, b=25),
        height=490
    )
    
    return {
        "figure": json.loads(fig.to_json()),
        "selected_probe": selected_probe
    }
