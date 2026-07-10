"""
OncoLens Callbacks - Gene Co-expression Network Card.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from dash import Input, Output, callback_context, html
from app.layouts.theme import (
    PLOT_TEMPLATE, PLOT_PAPER_BG, PLOT_PLOT_BG, PLOT_GRID, PLOT_ZEROLINE,
    PLOT_TICK_COLOR, PLOT_TITLE_COLOR, PLOT_AXIS_LABEL_COLOR,
)
import app.callbacks._state as _state
from app.callbacks._shared import make_stats_card_content


def register_network(app) -> None:
    """Register all network card callbacks."""

    # --------------------------------------------------------------------------
    # 7. Gene Interaction Network Callback
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("network-plot", "figure"),
         Output("network-details-card", "children")],
        [Input("network-threshold-selector", "value"),
         Input("network-plot", "clickData")]
    )
    def update_network(threshold: float, click_data: Optional[dict]) -> Tuple[go.Figure, list]:
        """
        Dynamically filters edges based on selected Pearson threshold,
        generates the co-expression network graph, and highlights neighbors on click.
        """
        ctx = callback_context
        clicked_probe = None

        # Determine if clickData was triggered
        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger_id == "network-plot" and click_data and "points" in click_data:
                pt = click_data["points"][0]
                if "customdata" in pt:
                    clicked_probe = pt["customdata"][0]

        # 1. Filter edges at the selected threshold
        df_edges_filtered = _state.df_network_edges_global[_state.df_network_edges_global["Correlation"] >= threshold]

        # Extract set of connected nodes at threshold r >= 0.70 (pre-computed in network_layout_pos)
        connected_probes = set(_state.df_network_edges_global["Probe X"]).union(set(_state.df_network_edges_global["Probe Y"]))
        df_nodes = _state.df_annot_global[_state.df_annot_global["ProbeID"].isin(connected_probes)].copy()

        # Node variance scaling bounds
        var_min = df_nodes["Variance"].min()
        var_max = df_nodes["Variance"].max()
        var_range = var_max - var_min if var_max > var_min else 1.0

        # Calculate degrees (neighbor counts) at the current threshold
        degrees = {px: 0 for px in connected_probes}
        for _, row in df_edges_filtered.iterrows():
            px = row["Probe X"]
            py = row["Probe Y"]
            if px in degrees:
                degrees[px] += 1
            if py in degrees:
                degrees[py] += 1

        # Identify neighbors of the clicked node
        neighbors = set()
        if clicked_probe:
            df_n1 = df_edges_filtered[df_edges_filtered["Probe X"] == clicked_probe]
            neighbors.update(df_n1["Probe Y"])
            df_n2 = df_edges_filtered[df_edges_filtered["Probe Y"] == clicked_probe]
            neighbors.update(df_n2["Probe X"])

        # Construct Plotly Figure
        fig = go.Figure()

        # 2. Draw co-expression edges grouped into dynamic bins for speed and styling
        # This approach avoids rendering thousands of individual traces which lags the browser
        step = (1.0 - threshold) / 3.0
        bins = [
            (threshold, threshold + step, 1.2, 0.15),
            (threshold + step, threshold + 2*step, 2.5, 0.35),
            (threshold + 2*step, 1.01, 4.2, 0.65)
        ]

        for bin_start, bin_end, width, base_opacity in bins:
            df_bin = df_edges_filtered[(df_edges_filtered["Correlation"] >= bin_start) & (df_edges_filtered["Correlation"] < bin_end)]
            edge_x = []
            edge_y = []
            for _, row in df_bin.iterrows():
                px = row["Probe X"]
                py = row["Probe Y"]
                if px in _state.network_layout_pos and py in _state.network_layout_pos:
                    x0, y0 = _state.network_layout_pos[px]
                    x1, y1 = _state.network_layout_pos[py]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            # Apply low background opacity to line traces if highlighting is active
            opacity = base_opacity * 0.10 if clicked_probe else base_opacity

            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=width, color="#475569"),
                opacity=opacity,
                hoverinfo="none",
                showlegend=False
            ))

        # 3. Draw active co-expression lines in foreground if a node is clicked
        if clicked_probe:
            df_high = df_edges_filtered[(df_edges_filtered["Probe X"] == clicked_probe) | (df_edges_filtered["Probe Y"] == clicked_probe)]
            high_x = []
            high_y = []
            for _, row in df_high.iterrows():
                px = row["Probe X"]
                py = row["Probe Y"]
                if px in _state.network_layout_pos and py in _state.network_layout_pos:
                    x0, y0 = _state.network_layout_pos[px]
                    x1, y1 = _state.network_layout_pos[py]
                    high_x.extend([x0, x1, None])
                    high_y.extend([y0, y1, None])

            fig.add_trace(go.Scatter(
                x=high_x,
                y=high_y,
                mode="lines",
                line=dict(width=3.5, color="#3b82f6"),
                opacity=0.9,
                hoverinfo="none",
                showlegend=False
            ))

        # 4. Map chromosome names to colors
        chr_colors = {
            "1": "#ff007f", "2": "#ff5500", "3": "#ffaa00", "4": "#aaff00", "5": "#00ff55",
            "6": "#00ffaa", "7": "#00aaff", "8": "#0055ff", "9": "#aa00ff", "10": "#ff00aa",
            "11": "#7f00ff", "12": "#00ff7f", "13": "#ff0055", "14": "#3b82f6", "15": "#10b981",
            "16": "#f59e0b", "17": "#8b5cf6", "18": "#ec4899", "19": "#14b8a6", "20": "#f43f5e",
            "21": "#84cc16", "22": "#06b6d4", "X": "#e11d48", "Y": "#4f46e5"
        }

        # 5. Populate node properties
        node_x = []
        node_y = []
        node_text = []
        node_customdata = []
        node_colors = []
        node_sizes = []
        node_opacities = []
        node_borders = []
        node_border_widths = []

        for _, row in df_nodes.iterrows():
            probe_id = row["ProbeID"]
            symbol = row["Gene Symbol"] if pd.notna(row["Gene Symbol"]) else probe_id
            if probe_id not in _state.network_layout_pos:
                continue

            x, y = _state.network_layout_pos[probe_id]
            node_x.append(x)
            node_y.append(y)
            node_text.append(symbol)

            deg = degrees.get(probe_id, 0)
            node_customdata.append([probe_id, symbol, row["Chromosome"], int(row["Rank"]), deg])

            # Base node properties
            base_color = chr_colors.get(str(row["Chromosome"]), "#cbd5e1")
            base_size = 6 + 14 * (row["Variance"] - var_min) / var_range

            if clicked_probe:
                if probe_id == clicked_probe:
                    node_colors.append("#ef4444") # Red highlight for selected
                    node_sizes.append(22)
                    node_opacities.append(1.0)
                    node_borders.append("#ffffff")
                    node_border_widths.append(2.0)
                elif probe_id in neighbors:
                    node_colors.append("#3b82f6") # Blue highlight for neighbors
                    node_sizes.append(15)
                    node_opacities.append(0.95)
                    node_borders.append("#ffffff")
                    node_border_widths.append(1.0)
                else:
                    node_colors.append("#334155") # Fade unconnected nodes
                    node_sizes.append(5)
                    node_opacities.append(0.12)
                    node_borders.append("#1e293b")
                    node_border_widths.append(0.5)
            else:
                node_colors.append(base_color)
                node_sizes.append(base_size)
                node_opacities.append(0.85)
                node_borders.append("#ffffff")
                node_border_widths.append(0.5)

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=node_opacities,
                line=dict(color=node_borders, width=node_border_widths)
            ),
            text=node_text,
            customdata=node_customdata,
            hovertemplate=(
                "<b>Gene Symbol</b>: %{text}<br>"
                "<b>Probe ID</b>: %{customdata[0]}<br>"
                "<b>Chromosome</b>: Chr %{customdata[2]}<br>"
                "<b>Variance Rank</b>: #%{customdata[3]}<br>"
                "<b>Degree (Neighbors)</b>: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))

        title_text = "Co-Expression network (r ≥ {0:.2f})".format(threshold)

        fig.update_layout(
            template=PLOT_TEMPLATE,
            title=dict(
                text=title_text,
                font=dict(size=16, color=PLOT_TITLE_COLOR, family="Outfit")
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor=PLOT_PAPER_BG,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="closest",
            dragmode="pan"
        )

        # 6. Render Details Card Sidebar Content
        target_probe = clicked_probe if clicked_probe else _state.df_rank_global.iloc[0]["ProbeID"]
        ann_row = _state.df_annot_global[_state.df_annot_global["ProbeID"] == target_probe]

        if not ann_row.empty:
            row = ann_row.iloc[0]
            symbol = row["Gene Symbol"] if pd.notna(row["Gene Symbol"]) else "Unknown"
            chrom = row["Chromosome"]
            cytoband = row["Cytoband"]
            rank = row["Rank"]
            variance = row["Variance"]

            deg = degrees.get(target_probe, 0)

            card_content = html.Div(
                className="network-footer",
                children=[
                    # Left column
                    html.Div(
                        className="network-footer-col network-footer-left",
                        children=[
                            html.Div(
                                className="network-footer-line",
                                children=[
                                    html.Span("Selected Gene: ", className="network-footer-label"),
                                    html.Span(symbol, className="network-footer-value network-footer-value--blue" if clicked_probe else "network-footer-value"),
                                ]
                            ),
                            html.Div(
                                className="network-footer-line",
                                children=[
                                    html.Span("Probe: ", className="network-footer-label"),
                                    html.Span(target_probe, className="network-footer-value"),
                                ]
                            ),
                            html.Div(
                                className="network-footer-line",
                                children=[
                                    html.Span(f"Rank #{int(rank)}" if pd.notna(rank) else "Rank N/A", className="network-footer-value--amber"),
                                    html.Span(" • ", style={"color": "#D1D5DB"}),
                                    html.Span(f"Chr {chrom}" if pd.notna(chrom) else "Chr N/A", className="network-footer-value"),
                                ]
                            ),
                        ]
                    ),

                    # Vertical divider line
                    html.Div(className="network-footer-divider"),

                    # Right column
                    html.Div(
                        className="network-footer-col network-footer-right",
                        children=[
                            html.Div(
                                className="network-footer-line",
                                style={"justifyContent": "flex-end"},
                                children=[
                                    html.Span("Neighbors: ", className="network-footer-label"),
                                    html.Span(str(deg), className="network-footer-value network-footer-value--green" if deg > 0 else "network-footer-value"),
                                ]
                            ),
                            html.Div(
                                className="network-footer-line",
                                style={"justifyContent": "flex-end"},
                                children=[
                                    html.Span("Degree: ", className="network-footer-label"),
                                    html.Span(str(deg), className="network-footer-value"),
                                ]
                            ),
                            html.Div(
                                className="network-footer-line",
                                style={"justifyContent": "flex-end"},
                                children=[
                                    html.Span("Threshold: ", className="network-footer-label"),
                                    html.Span("r ≥ {0:.2f}".format(threshold), className="network-footer-value network-footer-value--blue"),
                                ]
                            ),
                        ]
                    )
                ]
            )
        else:
            card_content = html.Div(
                className="network-footer",
                children=[
                    html.Div("Gene details unavailable.", className="network-footer-line", style={"fontSize": "10px", "color": "#6B7280"})
                ]
            )

        return fig, card_content
