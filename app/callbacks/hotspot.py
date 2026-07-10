"""
OncoLens Callbacks - Chromosomal Hotspot Card.
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
from app.callbacks._shared import make_stats_card_content


def register_hotspot(app) -> None:
    """Register all hotspot card callbacks."""

    # --------------------------------------------------------------------------
    # 5. Chromosomal Hotspot Plot Callback
    # --------------------------------------------------------------------------
    @app.callback(
        Output("hotspots-plot", "figure"),
        [Input("hotspots-chr-selector", "value"),
         Input("hotspots-plot", "clickData")]
    )
    def update_hotspots_plot(selected_chr: str, click_data: Optional[dict]) -> go.Figure:
        """
        Renders the chromosomal hotspot mapping chart with selection highlighting.
        """
        # Filter for annotated genes that have valid chromosome coordinates
        df_plot = _state.df_annot_global.dropna(subset=["Chromosome", "Genomic Start"]).copy()

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
        # Adaptive layout styling based on overview ("All") vs detail (single chromosome) selection
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

        # Size scaling: map variance to marker size
        var_min = df_plot["Variance"].min()
        var_max = df_plot["Variance"].max()
        var_range = var_max - var_min if var_max > var_min else 1.0

        # Square-root scaling to normalize the visual distribution of marker sizes
        normalized_variance = (df_plot["Variance"] - var_min) / var_range
        marker_sizes = base_size + scale_coeff * np.sqrt(normalized_variance)

        fig.add_trace(go.Scatter(
            x=df_plot["Genomic Start"],
            y=df_plot["ChrTrack"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=df_plot["Rank"],
                colorscale="Plasma", # Continuous color scale representing rank
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Variance Rank",
                        side="top",
                        font=dict(color="#475569", size=10)
                    ),
                    tickfont=dict(color="#64748b", size=9),
                    len=0.92,      # colorbar height prominence (92% of plot height)
                    thickness=15   # colorbar width prominence (reduced to 15px)
                ),
                reversescale=True,
                opacity=0.8,
                line=dict(width=0.5, color="#1e293b")
            ),
            text=df_plot["Gene Symbol"],
            customdata=df_plot["ProbeID"],
            hovertemplate=(
                "<b>Gene Symbol</b>: %{text}<br>"
                "<b>Probe ID</b>: %{customdata}<br>"
                "<b>Location</b>: %{y}: %{x:,} bp<br>"
                "<b>Variance Rank</b>: #%{marker.color}<br>"
                "<extra></extra>"
            )
        ))

        # ── Highlighting Selected Gene Marker ──
        selected_probe = None
        if click_data and "points" in click_data:
            pt = click_data["points"][0]
            if "customdata" in pt:
                selected_probe = pt["customdata"]

        # Validate selected probe belongs to the filtered subset, else fallback to top-ranked of selected chromosome
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
                        size=selected_marker_size,          # Selected marker size
                        color="rgba(0,0,0,0)",              # Transparent fill
                        line=dict(width=selected_marker_outline_width, color="#10B981") # Outline width and color
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                ))

        title_text = "Genome-Wide Expression Variance Hotspots" if selected_chr == "All" else f"Chromosome {selected_chr} Hotspot Loci"

        fig.update_layout(
            template=PLOT_TEMPLATE,
            showlegend=False,
            title=dict(
                text=title_text,
                font=dict(size=14, color=PLOT_TITLE_COLOR, family="Outfit")
            ),
            xaxis=dict(
                title=dict(text="Genomic Coordinate (Base Pairs)", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=11)),
                tickfont=dict(color=PLOT_TICK_COLOR, size=10),
                gridcolor=PLOT_GRID,
                zerolinecolor=PLOT_ZEROLINE,
                tickformat=","
            ),
            yaxis=dict(
                title=dict(text="Chromosomal Tracks", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=11)),
                tickfont=dict(color=PLOT_TICK_COLOR, size=10),
                gridcolor=PLOT_GRID,
                type="category",
                categoryarray=chroms_order,
                categoryorder="array"
            ),
            plot_bgcolor=PLOT_PLOT_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            margin=dict(l=45, r=20, t=25, b=0), # Tight margins to completely use the canvas
            hovermode="closest"
        )

        return fig

    # --------------------------------------------------------------------------
    # 6. Chromosomal Hotspot Gene Highlight Details Callback
    # --------------------------------------------------------------------------
    @app.callback(
        Output("hotspots-details-card", "children"),
        [Input("hotspots-plot", "clickData"),
         Input("hotspots-chr-selector", "value")]
    )
    def update_hotspot_gene_details(click_data: Optional[dict], chr_value: str) -> list:
        """
        Updates the hotspots details panel when a gene marker is clicked.
        Falls back to the top-ranked gene of the selected chromosome on startup or reset.
        """
        probe_id = None

        if click_data and "points" in click_data:
            pt = click_data["points"][0]
            if "customdata" in pt:
                probe_id = pt["customdata"]

        # Fallback default gene selection
        if not probe_id:
            if chr_value != "All":
                df_chr = _state.df_annot_global[_state.df_annot_global["Chromosome"] == chr_value]
            else:
                df_chr = _state.df_annot_global

            if not df_chr.empty:
                df_chr_sorted = df_chr.dropna(subset=["Rank"]).sort_values(by="Rank")
                if not df_chr_sorted.empty:
                    probe_id = df_chr_sorted.iloc[0]["ProbeID"]

            if not probe_id:
                probe_id = _state.df_rank_global.iloc[0]["ProbeID"]

        # Query details
        ann = _state.df_annot_global[_state.df_annot_global["ProbeID"] == probe_id]
        if ann.empty:
            return [html.P("No gene details available.")]

        row = ann.iloc[0]
        symbol = row["Gene Symbol"] if pd.notna(row["Gene Symbol"]) else "Unknown"
        chrom = row["Chromosome"]
        cytoband = row["Cytoband"]
        rank = row["Rank"]
        variance = row["Variance"]

        # Calculate expression statistics
        expr_vals = _state.df_expr_global[probe_id]
        mean_val = expr_vals.mean()
        std_val = expr_vals.std()
        min_val = expr_vals.min()
        max_val = expr_vals.max()

        return [
            html.Div(
                className="hotspots-footer-container",
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "width": "100%",
                    "fontSize": "0.75rem",
                    "color": "#4B5563"
                },
                children=[
                    html.Div(
                        children=[
                            html.Span("Selected Gene: ", style={"color": "#6B7280", "fontWeight": "500"}),
                            html.Strong(symbol, style={"color": "#3B82F6", "fontSize": "0.85rem", "fontFamily": "Outfit"}),
                            html.Span("  •  ", style={"color": "#D1D5DB"}),
                            html.Span("Rank ", style={"color": "#6B7280"}),
                            html.Strong(f"#{int(rank)}" if pd.notna(rank) else "#—", style={"color": "#F59E0B", "fontSize": "0.85rem"}),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Span(f"Chr {chrom} • " if pd.notna(chrom) else "Unmapped • ", style={"fontWeight": "600", "color": "#374151"}),
                            html.Span(f"Cytoband {cytoband} • " if pd.notna(cytoband) else ""),
                            html.Span("Variance ", style={"color": "#6B7280"}),
                            html.Strong(f"{variance:.4f}" if pd.notna(variance) else "—", style={"color": "#374151"}),
                            html.Span("  •  ", style={"color": "#D1D5DB"}),
                            html.Span("Mean Expression ", style={"color": "#6B7280"}),
                            html.Strong(f"{mean_val:.2f}", style={"color": "#374151"}),
                        ]
                    )
                ]
            )
        ]
