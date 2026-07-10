from fastapi import APIRouter, HTTPException
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json



router = APIRouter()


class ContourRequest(BaseModel):
    probe_x: str
    probe_y: str
    display_mode: str = "scatter"

@router.post("/api/contour")
def get_contour_plot(req: ContourRequest):
    import api.main as main_app

    probe_x = req.probe_x
    probe_y = req.probe_y
    display_mode = req.display_mode

    if not probe_x or not probe_y:
        return {"error": "Missing probe"}

    # Calculate expression statistics across all samples
    expr_x = main_app.df_expression[probe_x]
    expr_y = main_app.df_expression[probe_y]
    
    df_plot = main_app.df_expression[["samples", "type", probe_x, probe_y]].copy()
    X_subset = df_plot[[probe_x, probe_y]].to_numpy()
    X_scaled = StandardScaler().fit_transform(X_subset)
    
    sil = silhouette_score(X_scaled, df_plot["type"])
    
    fig = go.Figure()
    color_map = {
        "normal": "#10b981",
        "ependymoma": "#3b82f6",
        "glioblastoma": "#ef4444",
        "medulloblastoma": "#8b5cf6",
        "pilocytic_astrocytoma": "#f59e0b"
    }
    
    show_scatter = display_mode in ("scatter", "both")
    show_contour = display_mode == "both"
    
    for subtype in sorted(df_plot["type"].unique()):
        df_sub = df_plot[df_plot["type"] == subtype]
        x_vals = df_sub[probe_x].to_numpy()
        y_vals = df_sub[probe_y].to_numpy()
        subtype_color = color_map.get(subtype, "#94a3b8")
        display_name = subtype.replace("_", " ").title()
        
        if show_contour and len(df_sub) > 1:
            fig.add_trace(go.Histogram2dContour(
                x=x_vals, y=y_vals,
                name=f"{display_name} Contour",
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, subtype_color]],
                showlegend=False,
                contours=dict(coloring='none', showlines=True),
                line=dict(width=1.2, color=subtype_color),
                ncontours=7, opacity=0.3
            ))
            
        if show_scatter:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                name=display_name,
                marker=dict(color=subtype_color, size=9.5, opacity=0.78, line=dict(width=0.6, color="rgba(255,255,255,0.4)")),
            ))
            
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Joint Gene Phenotyping"),
        margin=dict(l=50, r=115, t=50, b=50),
    )
    
    return {
        "figure": json.loads(fig.to_json()),
        "sil_score": sil
    }
