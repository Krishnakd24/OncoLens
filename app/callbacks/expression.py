"""
OncoLens Callbacks - Gene Expression Profile Explorer Card.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, html
from app.layouts.theme import (
    PLOT_TEMPLATE, PLOT_PAPER_BG, PLOT_PLOT_BG, PLOT_GRID, PLOT_ZEROLINE,
    PLOT_TICK_COLOR, PLOT_TITLE_COLOR, PLOT_AXIS_LABEL_COLOR,
)
import app.callbacks._state as _state
from app.callbacks._shared import make_stats_card_content, _gene_stat_item


def register_expression(app) -> None:
    """Register all expression profile card callbacks."""

    # --------------------------------------------------------------------------
    # 8. Gene Expression Profile Explorer Callback
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("profiles-plot", "figure"),
         Output("profiles-details-card", "children")],
        [Input("profiles-gene-selector", "value")]
    )
    def update_profiles(probe_id: str) -> Tuple[go.Figure, list]:
        """
        Calculates subtype cohort statistics and renders a combined violin, box,
        and jittered scatter plot for the selected gene.
        """
        from scipy import stats

        # 1. Query annotations details
        ann_row = _state.df_annot_global[_state.df_annot_global["ProbeID"] == probe_id]
        if ann_row.empty:
            return go.Figure(), [html.P("Gene details unavailable.")]

        row = ann_row.iloc[0]
        symbol = row["Gene Symbol"] if pd.notna(row["Gene Symbol"]) else "Unknown"
        chrom = row["Chromosome"]
        cytoband = row["Cytoband"]
        rank = row["Rank"]
        variance = row["Variance"]

        # 2. Extract expression values and calculate cohort metrics
        expr_series = _state.df_expr_global[probe_id]
        mean_overall = expr_series.mean()
        median_overall = expr_series.median()
        std_overall = expr_series.std()
        min_overall = expr_series.min()
        max_overall = expr_series.max()
        n_samples = len(expr_series)

        # Standardized subtype ordering
        SUBTYPE_ORDER = ["normal", "ependymoma", "glioblastoma", "medulloblastoma", "pilocytic_astrocytoma"]

        # 3. Compute one-way ANOVA p-value across the 5 cohorts
        groups = [_state.df_expr_global[_state.df_expr_global["type"] == subtype][probe_id].values for subtype in SUBTYPE_ORDER]
        # Filter groups to make sure they are not empty
        groups_valid = [g for g in groups if len(g) > 0]

        if len(groups_valid) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups_valid)
                p_value_str = f"{p_value:.4e}" if p_value >= 1e-4 else f"{p_value:.2e}"
                if p_value < 1e-12:
                    p_value_str = "< 1e-12"
            except Exception:
                p_value_str = "Error calculating"
        else:
            p_value_str = "N/A"

        # 4. Generate combined Plotly violin, box, and jittered points figure
        fig = go.Figure()

        # Establish standard color mapping
        color_map = {
            "normal": "#10b981",          # Emerald
            "ependymoma": "#3b82f6",      # Blue
            "glioblastoma": "#ef4444",    # Crimson
            "medulloblastoma": "#8b5cf6", # Purple
            "pilocytic_astrocytoma": "#f59e0b" # Amber
        }

        # Add trace for each clinical class in standard sequence
        for subtype in SUBTYPE_ORDER:
            df_sub = _state.df_expr_global[_state.df_expr_global["type"] == subtype]
            if df_sub.empty:
                continue

            y_vals = df_sub[probe_id].values
            samples = df_sub["samples"].values

            # Format hover tooltip labels: Sample ID, Subtype, Expression
            customdata = np.stack((samples, [subtype.replace('_', ' ').title()]*len(samples)), axis=-1)

            display_name = subtype.replace("_", " ").title()
            subtype_color = color_map.get(subtype, "#cbd5e1")
            x_tick_label = f"{display_name}<br>(N={len(df_sub)})"

            fig.add_trace(go.Violin(
                x=[x_tick_label] * len(df_sub),
                y=y_vals,
                name=display_name,
                box_visible=True,
                meanline_visible=True,
                points='all',
                jitter=0.3,
                pointpos=-1.8, # Position points to the left of the violin
                marker=dict(size=5, opacity=0.7, color=subtype_color),
                line=dict(color=subtype_color, width=1.5),
                fillcolor=subtype_color,
                opacity=0.8,
                customdata=customdata,
                hovertemplate=(
                    "<b>Sample ID</b>: %{customdata[0]}<br>"
                    "<b>Subtype</b>: %{customdata[1]}<br>"
                    "<b>Expression Level</b>: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

        # 5. Add a faint horizontal reference line representing the overall mean
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(groups_valid) - 0.5,
            y0=mean_overall,
            y1=mean_overall,
            line=dict(
                color="#cbd5e1",
                width=1.2,
                dash="dash"
            ),
            xref="x",
            yref="y"
        )

        # Annotate overall mean line — anchored to paper space so it sits just
        # outside the plotting area instead of floating over the last violin
        fig.add_annotation(
            xref="paper",
            x=1.01,
            xanchor="left",
            y=mean_overall,
            yref="y",
            yanchor="middle",
            text=f"Overall Mean: {mean_overall:.4f}",
            showarrow=False,
            align="left",
            textangle=-90,
            font=dict(color=PLOT_TICK_COLOR, size=10, family="Inter")
        )

        title_text = f"Subtype Expression Profile: {symbol} ({probe_id})"

        fig.update_layout(
            template=PLOT_TEMPLATE,
            title=dict(
                text=title_text,
                font=dict(size=16, color=PLOT_TITLE_COLOR, family="Outfit"),
                y=0.98,
                yanchor="top"
            ),
            xaxis=dict(
                title=dict(text="Clinical Cohort", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=12), standoff=10),
                tickfont=dict(color=PLOT_TICK_COLOR, size=10),
                gridcolor=PLOT_GRID,
                categoryorder="array",
                categoryarray=[f"{s.replace('_', ' ').title()}<br>(N={len(_state.df_expr_global[_state.df_expr_global['type'] == s])})" for s in SUBTYPE_ORDER],
                tickangle=45,
                automargin=True
            ),
            yaxis=dict(
                title=dict(text="Log2 Normalized Expression Value", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=12), standoff=10),
                tickfont=dict(color=PLOT_TICK_COLOR),
                gridcolor=PLOT_GRID,
                zerolinecolor=PLOT_ZEROLINE,
                automargin=True
            ),
            plot_bgcolor=PLOT_PLOT_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            margin=dict(l=75, r=30, t=40, b=0),
            hovermode="closest",
            autosize=True
        )

        # 6. Render Statistics and ANOVA details card
        # Compacted so all 6 rows fit within the 220px-wide side panel without
        # needing to scroll: no header, Gene Symbol + Probe Set ID share a row,
        # and every remaining stat is built through the shared _gene_stat_item
        # helper. Colors are kept exactly as they were — only spacing/sizing
        # changed, since that's what was actually causing the overflow.
        pvalue_color = "#10b981" if p_value_str.startswith("<") or (
            not p_value_str.startswith("Error") and float(p_value_str.split('e')[0]) < 0.05
        ) else "#475569"

        stat_rows = [
            (("Gene Symbol", symbol, "#10b981", "1.05rem", "Outfit"),
             ("Probe Set ID", probe_id, "#334155", "0.82rem", None)),
            (("Chromosome", f"Chr {chrom}" if pd.notna(chrom) else "Unmapped", "#475569", "0.80rem", None),
             ("Cytoband", cytoband if pd.notna(cytoband) else "Unmapped", "#475569", "0.80rem", None)),
            (("Variance Rank", f"#{int(rank)}" if pd.notna(rank) else "N/A", "#f59e0b", "0.80rem", None),
             ("ANOVA p-value", p_value_str, pvalue_color, "0.80rem", None)),
            (("Mean Expression", f"{mean_overall:.4f}", "#475569", "0.80rem", None),
             ("Median Expression", f"{median_overall:.4f}", "#475569", "0.80rem", None)),
            (("Std Deviation", f"{std_overall:.4f}", "#475569", "0.80rem", None),
             ("Min Expression", f"{min_overall:.4f}", "#475569", "0.80rem", None)),
            (("Max Expression", f"{max_overall:.4f}", "#475569", "0.80rem", None),
             ("Sample Count", f"{n_samples}", "#475569", "0.80rem", None)),
        ]

        card_content = [
            html.Div(
                style={"display": "flex", "flexDirection": "column", "gap": "0.15rem", "marginTop": "0.1rem"},
                children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "0.2rem", "alignItems": "start"},
                        children=[_gene_stat_item(*left), _gene_stat_item(*right)]
                    )
                    for left, right in stat_rows
                ]
            )
        ]

        return fig, card_content
