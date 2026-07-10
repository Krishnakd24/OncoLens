"""
OncoLens Callbacks - Contour/Scatter Card.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback_context, no_update
from app.layouts.theme import (
    PLOT_TEMPLATE, PLOT_PAPER_BG, PLOT_PLOT_BG, PLOT_GRID, PLOT_ZEROLINE,
    PLOT_TICK_COLOR, PLOT_TITLE_COLOR, PLOT_AXIS_LABEL_COLOR,
    COLOR_BORDER,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import app.callbacks._state as _state
from app.callbacks._shared import make_pair_quality_card_content, make_contour_footer_content


def register_contour(app) -> None:
    """Register all contour card callbacks."""

    # --------------------------------------------------------------------------
    # NOTE: render_tab_content() has been removed.
    # The dashboard is now a single unified page — all five visualizations are
    # statically mounted at startup. No tab routing is required.
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # 3. Dynamic Dropdown Mutual Exclusion Callback (Prevents Duplication)
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("dropdown-x", "options"),
         Output("dropdown-y", "options")],
        [Input("dropdown-x", "value"),
         Input("dropdown-y", "value")]
    )
    def update_dropdown_options(val_x: str, val_y: str) -> Tuple[list, list]:
        """
        Excludes the selected gene of Dropdown X from Dropdown Y's options, and vice versa.
        """
        options_x = [opt for opt in _state.gene_options_global if opt["value"] != val_y]
        options_y = [opt for opt in _state.gene_options_global if opt["value"] != val_x]
        return options_x, options_y

    # --------------------------------------------------------------------------
    # 4. Multi-Gene Contour & Scatter Plot Update Callback
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("contour-plot", "figure"),
         Output("contour-footer", "children")],
        [Input("dropdown-x", "value"),
         Input("dropdown-y", "value"),
         Input("contour-display-toggle", "value")]
    )
    def update_contour_plot(probe_x: str, probe_y: str, display_mode: str) -> Tuple[go.Figure, list]:
        """
        Updates the 2D contour-scatter visualization, calculates statistics, and evaluates separation quality.
        """
        # A. Handle empty selections
        if not probe_x or not probe_y:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template=PLOT_TEMPLATE,
                title=dict(
                    text="Please select both X-axis and Y-axis genes to begin.",
                    font=dict(size=14, color=PLOT_TICK_COLOR)
                ),
                plot_bgcolor=PLOT_PLOT_BG,
                paper_bgcolor=PLOT_PAPER_BG,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig, make_contour_footer_content(
                symbol_x="No gene selected.",
                rank_x="—",
                chrom_x="—",
                symbol_y="No gene selected.",
                rank_y="—",
                chrom_y="—",
                sil_score=None,
                interpretation=None,
            )

        # B. Get annotations for selected genes
        ann_x = _state.df_annot_global[_state.df_annot_global["ProbeID"] == probe_x]
        ann_y = _state.df_annot_global[_state.df_annot_global["ProbeID"] == probe_y]

        symbol_x = ann_x.iloc[0]["Gene Symbol"] if not ann_x.empty else "Unknown"
        symbol_y = ann_y.iloc[0]["Gene Symbol"] if not ann_y.empty else "Unknown"

        # C. Retrieve variance ranking position
        rank_x_search = _state.df_rank_global[_state.df_rank_global["ProbeID"] == probe_x]
        rank_y_search = _state.df_rank_global[_state.df_rank_global["ProbeID"] == probe_y]
        rank_x = rank_x_search.index[0] + 1 if not rank_x_search.empty else "N/A"
        rank_y = rank_y_search.index[0] + 1 if not rank_y_search.empty else "N/A"

        # D. Validate presence of expression profiles
        if probe_x not in _state.df_expr_global.columns or probe_y not in _state.df_expr_global.columns:
            error_fig = go.Figure()
            error_fig.update_layout(
                template=PLOT_TEMPLATE,
                title=dict(text="Error: Selected probe not found in expression database.", font=dict(color="#ef4444")),
                plot_bgcolor=PLOT_PLOT_BG,
                paper_bgcolor=PLOT_PAPER_BG
            )
            return error_fig, make_contour_footer_content(
                symbol_x="Error loading gene.",
                rank_x="—",
                chrom_x="—",
                symbol_y="Error loading gene.",
                rank_y="—",
                chrom_y="—",
                sil_score=None,
                interpretation=None,
            )

        # E. Calculate expression statistics across all samples
        expr_x = _state.df_expr_global[probe_x]
        expr_y = _state.df_expr_global[probe_y]
        chrom_x = ann_x.iloc[0]["Chromosome"] if not ann_x.empty else None
        chrom_y = ann_y.iloc[0]["Chromosome"] if not ann_y.empty else None
        chrom_x_text = f"Chr{chrom_x}" if pd.notna(chrom_x) and str(chrom_x) != "None" else "Chr?"
        chrom_y_text = f"Chr{chrom_y}" if pd.notna(chrom_y) and str(chrom_y) != "None" else "Chr?"

        # G. Compute Silhouette separation score
        # Extract features and scale for scale independence
        df_plot = _state.df_expr_global[["samples", "type", probe_x, probe_y]].copy()
        X_subset = df_plot[[probe_x, probe_y]].to_numpy()
        X_scaled = StandardScaler().fit_transform(X_subset)

        sil = silhouette_score(X_scaled, df_plot["type"])

        # Select interpretation and styling
        if sil >= 0.5:
            interpretation = "Excellent separation"
            quality_color = "#10b981"  # Emerald
        elif sil >= 0.35:
            interpretation = "Good separation"
            quality_color = "#3b82f6"  # Blue
        elif sil >= 0.15:
            interpretation = "Moderate separation"
            quality_color = "#f59e0b"  # Amber
        else:
            interpretation = "Weak separation"
            quality_color = "#ef4444"  # Crimson

        card_quality_content = make_pair_quality_card_content(
            sil_score=sil,
            interpretation=interpretation,
            color=quality_color
        )

        # H. Generate Plotly Figure
        fig = go.Figure()

        # Dashboard clinical color map
        color_map = {
            "normal": "#10b981",          # Emerald
            "ependymoma": "#3b82f6",      # Blue
            "glioblastoma": "#ef4444",    # Crimson
            "medulloblastoma": "#8b5cf6", # Purple
            "pilocytic_astrocytoma": "#f59e0b" # Amber/Gold
        }

        show_scatter = display_mode in ("scatter", "both")
        show_contour = display_mode == "both"

        # Render each clinical class
        for subtype in sorted(df_plot["type"].unique()):
            df_sub = df_plot[df_plot["type"] == subtype]
            x_vals = df_sub[probe_x].to_numpy()
            y_vals = df_sub[probe_y].to_numpy()

            subtype_color = color_map.get(subtype, "#94a3b8")
            display_name = subtype.replace("_", " ").title()

            # 1. Subtle 2D background density contours
            if show_contour and len(df_sub) > 1:
                fig.add_trace(go.Histogram2dContour(
                    x=x_vals,
                    y=y_vals,
                    name=f"{display_name} Contour",
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, subtype_color]],
                    showlegend=False,
                    contours=dict(coloring='none', showlines=True),
                    line=dict(width=1.2, color=subtype_color),
                    ncontours=7,  # Simplified number of levels
                    opacity=0.3   # Highly transparent background guide
                ))

            # 2. Front scatter points
            if show_scatter:
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=display_name,
                    marker=dict(
                        color=subtype_color,
                        size=9,
                        opacity=0.78,
                        line=dict(width=0.6, color="rgba(255,255,255,0.4)")
                    ),
                    customdata=np.stack((df_sub["samples"], df_sub["type"], x_vals, y_vals), axis=-1),
                    hovertemplate=(
                        "<b>Sample ID</b>: %{customdata[0]}<br>"
                        "<b>Subtype</b>: %{customdata[1]}<br>"
                        f"<b>{symbol_x} ({probe_x})</b>: %{{customdata[2]:.4f}}<br>"
                        f"<b>{symbol_y} ({probe_y})</b>: %{{customdata[3]:.4f}}<br>"
                        "<extra></extra>"
                    )
                ))

        # I. Styling configurations
        fig.update_layout(
            template=PLOT_TEMPLATE,
            title=dict(
                text=f"Joint Gene Phenotyping: {symbol_x} vs {symbol_y}",
                font=dict(size=16, color=PLOT_TITLE_COLOR, family="Outfit"),
                y=0.98,
                yanchor="top"
            ),
            xaxis=dict(
                title=dict(text=f"{symbol_x} Expression ({probe_x})", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=12)),
                tickfont=dict(color=PLOT_TICK_COLOR),
                gridcolor=PLOT_GRID,
                zerolinecolor=PLOT_ZEROLINE,
            ),
            yaxis=dict(
                title=dict(text=f"{symbol_y} Expression ({probe_y})", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=12)),
                tickfont=dict(color=PLOT_TICK_COLOR),
                gridcolor=PLOT_GRID,
                zerolinecolor=PLOT_ZEROLINE,
            ),
            plot_bgcolor=PLOT_PLOT_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            legend=dict(
                font=dict(color=PLOT_TITLE_COLOR, size=9.5),
                bgcolor="rgba(240, 240, 240, 0.8)",
                bordercolor="#CFD0D3",
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=0.99,
                xanchor="center",
                x=0.498,
                traceorder="normal",
                itemwidth=30
            ),
            margin=dict(l=50, r=20, t=60, b=90),
            hovermode="closest",
            uirevision=f"{probe_x}_{probe_y}"
        )

        return fig, make_contour_footer_content(
            symbol_x=symbol_x,
            rank_x=str(rank_x),
            chrom_x=chrom_x_text,
            symbol_y=symbol_y,
            rank_y=str(rank_y),
            chrom_y=chrom_y_text,
            sil_score=sil,
            interpretation=interpretation,
        )
