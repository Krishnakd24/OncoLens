from fastapi import APIRouter
import plotly.graph_objects as go
from pydantic import BaseModel
import json

router = APIRouter()

class ProfilesRequest(BaseModel):
    selected_probe: str | None = None

@router.post("/api/profiles")
def get_profiles_plot(req: ProfilesRequest):
    import api.main as main_app
    
    df_expression = main_app.df_expression
    df_annotated = main_app.df_annotated
    selected_probe = req.selected_probe

    if not selected_probe or selected_probe not in df_expression.columns:
        # fallback
        selected_probe = main_app.app_config.get("default_x")

    fig = go.Figure()
    
    if selected_probe and selected_probe in df_expression.columns:
        ann = df_annotated[df_annotated["ProbeID"] == selected_probe]
        symbol = ann.iloc[0]["Gene Symbol"] if not ann.empty else selected_probe
        
        df_plot = df_expression[["samples", "type", selected_probe]].copy()
        
        color_map = {
            "normal": "#10b981",
            "ependymoma": "#3b82f6",
            "glioblastoma": "#ef4444",
            "medulloblastoma": "#8b5cf6",
            "pilocytic_astrocytoma": "#f59e0b"
        }
        
        for subtype in sorted(df_plot["type"].unique()):
            df_sub = df_plot[df_plot["type"] == subtype]
            fig.add_trace(go.Box(
                y=df_sub[selected_probe],
                name=subtype.replace("_", " ").title(),
                marker_color=color_map.get(subtype, "#94a3b8"),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
        fig.update_layout(
            template="plotly_white",
            title=dict(text=f"Expression Profile: {symbol} ({selected_probe})"),
            yaxis_title="Log2 Expression",
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        
    return {"figure": json.loads(fig.to_json())}
