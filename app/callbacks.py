"""
Dashboard Callbacks Module for OncoLens.

This module implements all reactive event handlers: contour/scatter visualization,
chromosomal hotspot mapping, co-expression network, gene expression profiles,
and the Interactive Virtual Expression Assayer (simulator).

Tab routing has been removed; all components are statically mounted at startup.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
from dash import Input, Output, html, Dash, callback_context, no_update
from app.layouts.theme import (
    PLOT_TEMPLATE,
    PLOT_PAPER_BG, PLOT_PLOT_BG, PLOT_GRID, PLOT_ZEROLINE,
    PLOT_TICK_COLOR, PLOT_TITLE_COLOR, PLOT_AXIS_LABEL_COLOR,
    COLOR_TEXT_SECONDARY, COLOR_BORDER,
)

# ==============================================================================
# Global Data Storage Placeholders (Populated at Server Startup)
# ==============================================================================
df_expr_global: pd.DataFrame = None
df_annot_global: pd.DataFrame = None
df_di_global: pd.DataFrame = None
df_rank_global: pd.DataFrame = None
df_top_pairs_global: pd.DataFrame = None
df_network_edges_global: pd.DataFrame = None
network_layout_pos: dict = None
DEFAULT_GENE_X: str = None
DEFAULT_GENE_Y: str = None
default_profile_probe_global: str = None
top_20_options_global: list = []
gene_options_global: list = []
gene_cols_global: list = []
centroids_matrix_global: np.ndarray = None
patient_options_global: list = []
subtype_names_list: list = ["normal", "ependymoma", "glioblastoma", "medulloblastoma", "pilocytic_astrocytoma"]


def make_stats_card_content(
    gene_symbol: str, 
    probe_id: str, 
    chrom: str, 
    cytoband: str, 
    rank: int,
    mean_val: float,
    std_val: float,
    min_val: float,
    max_val: float
) -> list:
    """
    Renders the metadata and expression statistics card for a selected gene.
    """
    return [
        html.H4(f"Gene: {gene_symbol}", className="card-title", style={"borderBottom": "1px solid #1e293b", "paddingBottom": "0.5rem", "marginBottom": "1rem"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "0.75rem"},
            children=[
                html.Div([
                    html.Span("Probe Set ID", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(probe_id, style={"color": "#f8fafc", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Variance Rank", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(f"#{rank}", style={"color": "#3b82f6", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Chromosome", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(f"Chr {chrom}" if pd.notna(chrom) and str(chrom) != "None" else "Unmapped", style={"color": "#f8fafc", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Cytoband", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(cytoband if pd.notna(cytoband) and str(cytoband) != "None" else "Unmapped", style={"color": "#f8fafc", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Mean Expression", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(f"{mean_val:.4f}", style={"color": "#cbd5e1", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Std Deviation", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(f"{std_val:.4f}", style={"color": "#cbd5e1", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Min Expression", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(f"{min_val:.4f}", style={"color": "#cbd5e1", "fontSize": "1.0rem"})
                ]),
                html.Div([
                    html.Span("Max Expression", style={"fontSize": "0.8rem", "color": "#64748b", "display": "block"}),
                    html.Strong(f"{max_val:.4f}", style={"color": "#cbd5e1", "fontSize": "1.0rem"})
                ])
            ]
        )
    ]


def _gene_stat_item(label: str, value, color: str, size: str = "0.85rem", family: Optional[str] = None) -> html.Div:
    """
    Compact label/value pair used in the Expression card's gene statistics panel.
    Kept as a small helper so each stat block stays visually consistent.
    """
    strong_style = {"color": color, "fontSize": size}
    if family:
        strong_style["fontFamily"] = family
    return html.Div([
        html.Span(label, style={"fontSize": "0.68rem", "color": "#64748b", "display": "block"}),
        html.Strong(str(value), style=strong_style)
    ])


def make_pair_quality_card_content(sil_score: float, interpretation: str, color: str) -> list:
    """
    Renders the metadata panel showing the 2D Silhouette separation quality.
    """
    return [
        html.H4("Pair Separation Quality", className="card-title", style={"borderBottom": "1px solid #1e293b", "paddingBottom": "0.5rem", "marginBottom": "1rem"}),
        html.Div(
            style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "height": "calc(100% - 40px)"},
            children=[
                html.Div(
                    style={"textAlign": "center", "marginBottom": "1.25rem", "marginTop": "0.5rem"},
                    children=[
                        html.Span("2D Silhouette Score", style={"fontSize": "0.85rem", "color": "#64748b", "display": "block", "marginBottom": "0.25rem"}),
                        html.Strong(f"{sil_score:.4f}", style={"fontSize": "2.4rem", "color": "#f8fafc", "fontFamily": "Outfit"})
                    ]
                ),
                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.Span("Clinical Separation Strength", style={"fontSize": "0.85rem", "color": "#64748b", "display": "block", "marginBottom": "0.5rem"}),
                        html.Div(
                            interpretation,
                            style={
                                "color": color,
                                "backgroundColor": f"{color}12",
                                "border": f"1px solid {color}30",
                                "borderRadius": "8px",
                                "padding": "8px 16px",
                                "fontSize": "1.05rem",
                                "fontWeight": "700",
                                "display": "inline-block",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.05em"
                            }
                        )
                    ]
                )
            ]
        )
    ]


def make_contour_footer_content(
    symbol_x: str,
    rank_x: str,
    chrom_x: str,
    symbol_y: str,
    rank_y: str,
    chrom_y: str,
    sil_score: Optional[float],
    interpretation: Optional[str],
) -> list:
    """
    Renders the compact two-column contour footer.
    """
    separation_text = f"★ {interpretation.replace(' separation', '')}" if interpretation else "—"
    silhouette_text = f"{sil_score:.3f}" if sil_score is not None else "—"

    return [
html.Div(
    className="contour-footer-col contour-footer-left",
    children=[

            html.Div(
                className="contour-footer-line",
                children=[
                    html.Span("Gene X: ", className="contour-footer-label"),
                    html.Span(symbol_x, className="contour-footer-value"),
                    html.Span(f" • Rank #{rank_x} • {chrom_x}",
                            className="contour-footer-meta"),
                ],
            ),

            html.Div(
                className="contour-footer-line",
                children=[
                    html.Span("Gene Y: ", className="contour-footer-label"),
                    html.Span(symbol_y, className="contour-footer-value"),
                    html.Span(f" • Rank #{rank_y} • {chrom_y}",
                            className="contour-footer-meta"),
                ],
            ),

        ],
    ),
        html.Div(className="contour-footer-divider"),
            html.Div(
            className="contour-footer-col contour-footer-right",
            children=[

                html.Div(
                    className="contour-footer-line",
                    children=[
                        html.Span("Separation: ", className="contour-footer-label"),
                        html.Span(separation_text,
                                className="contour-footer-quality"),
                    ],
                ),

                html.Div(
                    className="contour-footer-line",
                    children=[
                        html.Span("Silhouette: ", className="contour-footer-label"),
                        html.Span(silhouette_text,
                                className="contour-footer-sil"),
                    ],
                ),

            ],
        ),
    ]


def register_callbacks(
    app: Dash, 
    df_expression: pd.DataFrame, 
    df_annotated: pd.DataFrame, 
    df_patient_di: pd.DataFrame, 
    df_variance_ranking: pd.DataFrame,
    df_top_pairs: pd.DataFrame,
    df_network_edges: pd.DataFrame
) -> None:
    """
    Registers all reactive callbacks for the Dash application.
    """
    global df_expr_global, df_annot_global, df_di_global, df_rank_global, df_top_pairs_global
    global df_network_edges_global, network_layout_pos
    global DEFAULT_GENE_X, DEFAULT_GENE_Y, default_profile_probe_global, top_20_options_global, gene_options_global
    global centroids_matrix_global, gene_cols_global, patient_options_global
    
    # Store dataframes globally in the module
    df_expr_global = df_expression
    df_annot_global = df_annotated
    df_di_global = df_patient_di
    df_rank_global = df_variance_ranking
    df_top_pairs_global = df_top_pairs
    df_network_edges_global = df_network_edges
    
    # Initialize simulator configurations
    gene_cols_global = [col for col in df_expression.columns if col not in ["samples", "type"]]
    patient_options_global = [{"label": f"Patient {pid}", "value": str(pid)} for pid in sorted(df_expression["samples"].unique())]
    
    # Calculate subtype centroids
    print(" -> Computing diagnostic centroids for Virtual Expression Assayer...")
    centroids_list = []
    for subtype in subtype_names_list:
        df_sub = df_expression[df_expression["type"] == subtype]
        centroids_list.append(df_sub[gene_cols_global].mean(axis=0).values)
    centroids_matrix_global = np.array(centroids_list)
    
    # Pre-generate co-expression network spring layout positions once at startup
    print(" -> Constructing co-expression network graph for spring layout calculations...")
    G_base = nx.Graph()
    connected_probes = set(df_network_edges["Probe X"]).union(set(df_network_edges["Probe Y"]))
    
    df_nodes = df_annotated[df_annotated["ProbeID"].isin(connected_probes)]
    for _, row in df_nodes.iterrows():
        G_base.add_node(row["ProbeID"])
        
    for _, row in df_network_edges.iterrows():
        G_base.add_edge(row["Probe X"], row["Probe Y"])
        
    print(" -> Precomputing network layout coordinates using spring_layout...")
    network_layout_pos = nx.spring_layout(G_base, seed=42, k=0.12, iterations=50)
    
    # Extract Rank 1 dynamically as the initial default pair
    DEFAULT_GENE_X = df_top_pairs.iloc[0]["Probe X"]
    DEFAULT_GENE_Y = df_top_pairs.iloc[0]["Probe Y"]
    
    # Resolve the highest-variance annotated gene dynamically as default profile explorer gene
    df_annot_sorted = df_annotated.dropna(subset=["Rank"]).sort_values(by="Rank")
    for _, row in df_annot_sorted.iterrows():
        symbol = row["Gene Symbol"]
        if pd.notna(symbol) and str(symbol).strip() != "" and str(symbol) != "nan":
            default_profile_probe_global = row["ProbeID"]
            break
    if not default_profile_probe_global:
        default_profile_probe_global = df_annot_sorted.iloc[0]["ProbeID"]
    
    # Build Top 20 suggested options
    top_20 = []
    for rank, row in df_top_pairs.head(20).iterrows():
        label = f"Rank {rank}: {row['Gene X']} & {row['Gene Y']} (Sil: {row['Silhouette Score']:.3f})"
        top_20.append({"label": label, "value": rank})
    top_20_options_global = top_20
    
    # Pre-generate dropdown options (ONLY display genes with valid annotations/symbols)
    options = []
    for _, row in df_annotated.iterrows():
        probe_id = row["ProbeID"]
        symbol = row["Gene Symbol"]
        if pd.notna(symbol) and str(symbol).strip() != "" and str(symbol) != "nan":
            label = f"{symbol} ({probe_id})"
            options.append({"label": label, "value": probe_id})
            
    # Sort options by symbol alphabetically
    gene_options_global = sorted(options, key=lambda x: x["label"])


    # --------------------------------------------------------------------------
    # NOTE: render_tab_content() has been removed.
    # The dashboard is now a single unified page — all five visualizations are
    # statically mounted at startup. No tab routing is required.
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # 2. Toggle Visibility of the Demonstration Pair Badge

    # --------------------------------------------------------------------------
    @app.callback(
        Output("demo-badge", "style"),
        [Input("dropdown-x", "value"),
         Input("dropdown-y", "value")]
    )
    def toggle_demo_badge(val_x: str, val_y: str) -> dict:
        """
        Shows the demo badge only when the active gene pair matches the default demo pair.
        """
        if val_x == DEFAULT_GENE_X and val_y == DEFAULT_GENE_Y:
            return {
                "backgroundColor": "#3b82f612",
                "color": "#3b82f6",
                "border": "1px solid #3b82f625",
                "borderRadius": "6px",
                "padding": "6px 12px",
                "fontSize": "0.8rem",
                "fontWeight": "600",
                "display": "inline-block",
                "marginBottom": "1rem",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
                "cursor": "help"
            }
        return {"display": "none"}

    # --------------------------------------------------------------------------
    # 3. Suggested Pairs Selector & Gene Dropdowns Synchronization Callback
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("dropdown-x", "value", allow_duplicate=True),
         Output("dropdown-y", "value", allow_duplicate=True),
         Output("demo-pair-selector", "value", allow_duplicate=True)],
        [Input("demo-pair-selector", "value"),
         Input("dropdown-x", "value"),
         Input("dropdown-y", "value")],
        prevent_initial_call=True
    )
    def sync_gene_selections(selected_rank: Optional[int], val_x: Optional[str], val_y: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Bi-directionally synchronizes the Suggested Pairs Dropdown and individual X/Y dropdowns.
        Uses allow_duplicate=True to allow cross-synchronization safely.
        """
        trigger = callback_context.triggered[0]["prop_id"]
        
        if "demo-pair-selector" in trigger:
            if selected_rank is not None:
                row = df_top_pairs_global.loc[selected_rank]
                return row["Probe X"], row["Probe Y"], selected_rank
            return val_x, val_y, None
            
        elif "dropdown-x" in trigger or "dropdown-y" in trigger:
            matched_rank = None
            for rank in range(1, min(21, len(df_top_pairs_global) + 1)):
                row = df_top_pairs_global.loc[rank]
                if ((row["Probe X"] == val_x and row["Probe Y"] == val_y) or 
                    (row["Probe X"] == val_y and row["Probe Y"] == val_x)):
                    matched_rank = rank
                    break
            return val_x, val_y, matched_rank
            
        return val_x, val_y, selected_rank

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
        options_x = [opt for opt in gene_options_global if opt["value"] != val_y]
        options_y = [opt for opt in gene_options_global if opt["value"] != val_x]
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
        ann_x = df_annot_global[df_annot_global["ProbeID"] == probe_x]
        ann_y = df_annot_global[df_annot_global["ProbeID"] == probe_y]
        
        symbol_x = ann_x.iloc[0]["Gene Symbol"] if not ann_x.empty else "Unknown"
        symbol_y = ann_y.iloc[0]["Gene Symbol"] if not ann_y.empty else "Unknown"
        
        # C. Retrieve variance ranking position
        rank_x_search = df_rank_global[df_rank_global["ProbeID"] == probe_x]
        rank_y_search = df_rank_global[df_rank_global["ProbeID"] == probe_y]
        rank_x = rank_x_search.index[0] + 1 if not rank_x_search.empty else "N/A"
        rank_y = rank_y_search.index[0] + 1 if not rank_y_search.empty else "N/A"
        
        # D. Validate presence of expression profiles
        if probe_x not in df_expr_global.columns or probe_y not in df_expr_global.columns:
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
        expr_x = df_expr_global[probe_x]
        expr_y = df_expr_global[probe_y]
        chrom_x = ann_x.iloc[0]["Chromosome"] if not ann_x.empty else None
        chrom_y = ann_y.iloc[0]["Chromosome"] if not ann_y.empty else None
        chrom_x_text = f"Chr{chrom_x}" if pd.notna(chrom_x) and str(chrom_x) != "None" else "Chr?"
        chrom_y_text = f"Chr{chrom_y}" if pd.notna(chrom_y) and str(chrom_y) != "None" else "Chr?"
        
        # G. Compute Silhouette separation score
        # Extract features and scale for scale independence
        df_plot = df_expr_global[["samples", "type", probe_x, probe_y]].copy()
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
                        size=9.5,
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
                font=dict(size=16, color=PLOT_TITLE_COLOR, family="Outfit")
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
                font=dict(color=PLOT_TITLE_COLOR, size=10),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=COLOR_BORDER,
                borderwidth=1,
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(l=50, r=115, t=50, b=15),
            hovermode="closest"
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
                        side="right",
                        font=dict(color="#cbd5e1", size=10)
                    ),
                    tickfont=dict(color="#94a3b8", size=9),
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
            margin=dict(l=45, r=20, t=15, b=25), # Tight margins to completely use the canvas
            hovermode="closest",
            height=400 # height directly controls track-to-track row spacing
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
                df_chr = df_annot_global[df_annot_global["Chromosome"] == chr_value]
            else:
                df_chr = df_annot_global
                
            if not df_chr.empty:
                df_chr_sorted = df_chr.dropna(subset=["Rank"]).sort_values(by="Rank")
                if not df_chr_sorted.empty:
                    probe_id = df_chr_sorted.iloc[0]["ProbeID"]
            
            if not probe_id:
                probe_id = df_rank_global.iloc[0]["ProbeID"]
                
        # Query details
        ann = df_annot_global[df_annot_global["ProbeID"] == probe_id]
        if ann.empty:
            return [html.P("No gene details available.")]
            
        row = ann.iloc[0]
        symbol = row["Gene Symbol"] if pd.notna(row["Gene Symbol"]) else "Unknown"
        chrom = row["Chromosome"]
        cytoband = row["Cytoband"]
        rank = row["Rank"]
        variance = row["Variance"]
        
        # Calculate expression statistics
        expr_vals = df_expr_global[probe_id]
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
        df_edges_filtered = df_network_edges_global[df_network_edges_global["Correlation"] >= threshold]
        
        # Extract set of connected nodes at threshold r >= 0.70 (pre-computed in network_layout_pos)
        connected_probes = set(df_network_edges_global["Probe X"]).union(set(df_network_edges_global["Probe Y"]))
        df_nodes = df_annot_global[df_annot_global["ProbeID"].isin(connected_probes)].copy()
        
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
                if px in network_layout_pos and py in network_layout_pos:
                    x0, y0 = network_layout_pos[px]
                    x1, y1 = network_layout_pos[py]
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
                if px in network_layout_pos and py in network_layout_pos:
                    x0, y0 = network_layout_pos[px]
                    x1, y1 = network_layout_pos[py]
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
            if probe_id not in network_layout_pos:
                continue
                
            x, y = network_layout_pos[probe_id]
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
            height=360,
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
        target_probe = clicked_probe if clicked_probe else df_rank_global.iloc[0]["ProbeID"]
        ann_row = df_annot_global[df_annot_global["ProbeID"] == target_probe]
        
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
        ann_row = df_annot_global[df_annot_global["ProbeID"] == probe_id]
        if ann_row.empty:
            return go.Figure(), [html.P("Gene details unavailable.")]
            
        row = ann_row.iloc[0]
        symbol = row["Gene Symbol"] if pd.notna(row["Gene Symbol"]) else "Unknown"
        chrom = row["Chromosome"]
        cytoband = row["Cytoband"]
        rank = row["Rank"]
        variance = row["Variance"]
        
        # 2. Extract expression values and calculate cohort metrics
        expr_series = df_expr_global[probe_id]
        mean_overall = expr_series.mean()
        median_overall = expr_series.median()
        std_overall = expr_series.std()
        min_overall = expr_series.min()
        max_overall = expr_series.max()
        n_samples = len(expr_series)
        
        # Standardized subtype ordering
        SUBTYPE_ORDER = ["normal", "ependymoma", "glioblastoma", "medulloblastoma", "pilocytic_astrocytoma"]
        
        # 3. Compute one-way ANOVA p-value across the 5 cohorts
        groups = [df_expr_global[df_expr_global["type"] == subtype][probe_id].values for subtype in SUBTYPE_ORDER]
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
            df_sub = df_expr_global[df_expr_global["type"] == subtype]
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
                x=0.5,
                y=0.98,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(text="Clinical Cohort", font=dict(color=PLOT_AXIS_LABEL_COLOR, size=12), standoff=10),
                tickfont=dict(color=PLOT_TICK_COLOR, size=10),
                gridcolor=PLOT_GRID,
                categoryorder="array",
                categoryarray=[f"{s.replace('_', ' ').title()}<br>(N={len(df_expr_global[df_expr_global['type'] == s])})" for s in SUBTYPE_ORDER],
                tickangle=45
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
            margin=dict(l=75, r=40, t=50, b=70),
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
        ) else "#cbd5e1"

        stat_rows = [
            (("Gene Symbol", symbol, "#10b981", "1.05rem", "Outfit"),
             ("Probe Set ID", probe_id, "#f8fafc", "0.82rem", None)),
            (("Chromosome", f"Chr {chrom}" if pd.notna(chrom) else "Unmapped", "#cbd5e1", "0.85rem", None),
             ("Cytoband", cytoband if pd.notna(cytoband) else "Unmapped", "#cbd5e1", "0.85rem", None)),
            (("Variance Rank", f"#{int(rank)}" if pd.notna(rank) else "N/A", "#f59e0b", "0.85rem", None),
             ("ANOVA p-value", p_value_str, pvalue_color, "0.85rem", None)),
            (("Mean Expression", f"{mean_overall:.4f}", "#cbd5e1", "0.85rem", None),
             ("Median Expression", f"{median_overall:.4f}", "#cbd5e1", "0.85rem", None)),
            (("Std Deviation", f"{std_overall:.4f}", "#cbd5e1", "0.85rem", None),
             ("Min Expression", f"{min_overall:.4f}", "#cbd5e1", "0.85rem", None)),
            (("Max Expression", f"{max_overall:.4f}", "#cbd5e1", "0.85rem", None),
             ("Sample Count", f"{n_samples}", "#cbd5e1", "0.85rem", None)),
        ]

        card_content = [
            html.Div(
                style={"display": "flex", "flexDirection": "column", "gap": "0.55rem", "marginTop": "0.3rem"},
                children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "0.6rem", "alignItems": "start"},
                        children=[_gene_stat_item(*left), _gene_stat_item(*right)]
                    )
                    for left, right in stat_rows
                ]
            )
        ]
        
        return fig, card_content

    # --------------------------------------------------------------------------
    # 9. Virtual Expression Assayer (Simulator Widget) bounds & values reset
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("simulator-slider-1", "min"),
         Output("simulator-slider-1", "max"),
         Output("simulator-slider-1", "value"),
         Output("simulator-slider-2", "min"),
         Output("simulator-slider-2", "max"),
         Output("simulator-slider-2", "value"),
         Output("simulator-slider-3", "min"),
         Output("simulator-slider-3", "max"),
         Output("simulator-slider-3", "value")],
        [Input("simulator-patient-selector", "value"),
         Input("simulator-gene-1", "value"),
         Input("simulator-gene-2", "value"),
         Input("simulator-gene-3", "value"),
         Input("simulator-reset-btn", "n_clicks")]
    )
    def update_simulator_slider_bounds(patient_id, gene1, gene2, gene3, n_clicks):
        """
        Dynamically adjusts the min, max, and values of the three perturbation sliders
        based on the selected patient baseline, gene selection, or reset button click.
        """
        import datetime
        ctx = callback_context
        triggered_id = ctx.triggered_id if hasattr(ctx, "triggered_id") else (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
        now_str = datetime.datetime.now().isoformat()
        
        print(f"\n[CALLBACK EXECUTION] timestamp={now_str} name=update_simulator_slider_bounds")
        print(f"  triggered_id: {triggered_id}")
        print(f"  Inputs: patient_id={patient_id}, gene1={gene1}, gene2={gene2}, gene3={gene3}, n_clicks={n_clicks}")
        
        # Ignore reset button initial render/mount triggers (when n_clicks is 0 or None)
        if triggered_id == "simulator-reset-btn" and (n_clicks is None or n_clicks == 0):
            print("  update_simulator_slider_bounds: Ignored reset button initial render trigger.")
            return [no_update] * 9
            
        if not patient_id:
            results = (0, 15, 5, 0, 15, 5, 0, 15, 5)
            print(f"  Outputs: {results}")
            return results
            
        patient_row = df_expr_global[df_expr_global["samples"].astype(str) == str(patient_id)]
        if patient_row.empty:
            results = (0, 15, 5, 0, 15, 5, 0, 15, 5)
            print(f"  Outputs: {results}")
            return results
        results = []
        for gene in [gene1, gene2, gene3]:
            baseline = patient_row[gene].values[0]
            
            g_min = df_expr_global[gene].min()
            g_max = df_expr_global[gene].max()
            g_range = g_max - g_min if g_max > g_min else 1.0
            
            s_min = max(0.0, float(g_min - 0.1 * g_range))
            s_max = float(g_max + 0.1 * g_range)
            
            val = float(baseline)
            if val < s_min:
                s_min = max(0.0, val - 0.5)
            if val > s_max:
                s_max = val + 0.5
                
            results.extend([round(s_min, 2), round(s_max, 2), round(val, 3)])
            
        print(f"  Outputs: {results}")
        return results

    # --------------------------------------------------------------------------
    # 10. Virtual Expression Assayer Simulation Calculation & Plotting
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("simulator-plot", "figure"),
         Output("simulator-prediction-card", "children"),
         Output("simulator-distance-card", "children"),
         Output("simulator-slider-label-1", "children"),
         Output("simulator-slider-label-2", "children"),
         Output("simulator-slider-label-3", "children")],
        [Input("simulator-patient-selector", "value"),
         Input("simulator-gene-1", "value"),
         Input("simulator-gene-2", "value"),
         Input("simulator-gene-3", "value"),
         Input("simulator-slider-1", "value"),
         Input("simulator-slider-2", "value"),
         Input("simulator-slider-3", "value")]
    )
    def run_simulation(patient_id, gene1, gene2, gene3, val1, val2, val3):
        """
        Runs the centroid distance classifier on the perturbed expression vector
        and updates the horizontal probability plot, prediction card, and distances.
        """
        try:
            import datetime
            ctx = callback_context
            triggered_id = ctx.triggered_id if hasattr(ctx, "triggered_id") else (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
            now_str = datetime.datetime.now().isoformat()
            
            # Resolve patient ID if None
            if patient_id is None:
                patient_id = patient_options_global[0]["value"] if patient_options_global else "834"
                
            patient_row = df_expr_global[df_expr_global["samples"].astype(str) == str(patient_id)]
            if patient_row.empty:
                patient_row = df_expr_global.iloc[0:1]
                patient_id = str(patient_row["samples"].values[0])
                
            # Resolve None slider values to baseline expression values (avoiding blank figures on startup/transition)
            resolved_val1 = val1 if val1 is not None else float(patient_row[gene1].values[0])
            resolved_val2 = val2 if val2 is not None else float(patient_row[gene2].values[0])
            resolved_val3 = val3 if val3 is not None else float(patient_row[gene3].values[0])
            
            # Copy the original patient expression vector for the 1000 genes
            patient_vector = patient_row[gene_cols_global].iloc[0].values.copy()
            
            # Replace the 3 selected genes with the resolved slider values
            try:
                idx1 = gene_cols_global.index(gene1)
                idx2 = gene_cols_global.index(gene2)
                idx3 = gene_cols_global.index(gene3)
                
                patient_vector[idx1] = resolved_val1
                patient_vector[idx2] = resolved_val2
                patient_vector[idx3] = resolved_val3
            except ValueError:
                return go.Figure(), [html.P("Gene not found.")], [html.P("Gene not found.")], "", "", ""
                
            from app.config import SIMULATOR_TEMPERATURE
            
            # Compute Euclidean distance to each subtype centroid
            diffs = centroids_matrix_global - patient_vector
            distances = np.linalg.norm(diffs, axis=1) # Shape: (5,)
            
            # Shifted scores formulation: scores = -(distances - min(distances))
            scores = -(distances - np.min(distances))
            
            # Scale scores using temperature factor T (default 20.0)
            exp_scores = np.exp(scores / SIMULATOR_TEMPERATURE)
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Print formal run_simulation diagnostics
            print("\n=== RUN_SIMULATION DIAGNOSTICS ===")
            print(f"  timestamp                     : {now_str}")
            print(f"  triggered_id                  : {triggered_id}")
            print(f"  patient_id                    : {patient_id}")
            print(f"  genes                         : {gene1}, {gene2}, {gene3}")
            print(f"  slider values (input)         : val1={val1}, val2={val2}, val3={val3}")
            print(f"  first 5 values of baseline vec: {patient_row[gene_cols_global].iloc[0].values[:5].tolist()}")
            print(f"  modified indices              : idx1={idx1}, idx2={idx2}, idx3={idx3}")
            print(f"  values written to copy vec    : val1={resolved_val1:.4f}, val2={resolved_val2:.4f}, val3={resolved_val3:.4f}")
            print(f"  computed centroid distances   : {[float(d) for d in distances]}")
            print(f"  computed probabilities        : {[float(p) for p in probabilities]}")
            print(f"  Softmax temperature T         : {SIMULATOR_TEMPERATURE}")
            print("==================================\n")
            
            # Find highest probability index
            max_idx = int(np.argmax(probabilities))
            prediction_confidence = float(probabilities[max_idx] * 100)
            
            # Establish standard color mapping
            color_map = {
                "normal": "#10b981",          # Emerald
                "ependymoma": "#3b82f6",      # Blue
                "glioblastoma": "#ef4444",    # Crimson
                "medulloblastoma": "#8b5cf6", # Purple
                "pilocytic_astrocytoma": "#f59e0b" # Amber
            }
            
            # Helper to retrieve gene symbol
            def get_symbol(gene_id):
                match = df_annot_global[df_annot_global["ProbeID"] == gene_id]
                if not match.empty:
                    sym = match["Gene Symbol"].values[0]
                    if pd.notna(sym) and str(sym).strip() != "" and str(sym) != "nan":
                        return sym
                return gene_id
                
            # 1. Build labels displaying symbols, current slider values, and deviations
            labels = []
            for gene, val in [(gene1, resolved_val1), (gene2, resolved_val2), (gene3, resolved_val3)]:
                sym = get_symbol(gene)
                baseline = patient_row[gene].values[0]
                delta = val - baseline
                sign = "+" if delta >= 0 else ""
                
                label_el = [
                    html.Span(f"{sym} ({gene})"),
                    html.Span(
                        f"{val:.3f} ({sign}{delta:.3f})", 
                        style={"color": "#10b981" if delta >= 0 else "#ef4444", "fontWeight": "bold"}
                    )
                ]
                labels.append(label_el)
                
            # 2. Horizontal probability bar chart
            y_labels = [s.replace("_", " ").title() for s in subtype_names_list]
            prob_percentages = probabilities * 100
            
            # Highlight predicted class via color opacity boundaries
            opacities = [1.0 if i == max_idx else 0.55 for i in range(5)]
            bar_colors = [color_map.get(s, "#cbd5e1") for s in subtype_names_list]
            
            # Reverse all lists for horizontal plotting direction so that Normal is at the top
            y_labels_reversed = y_labels[::-1]
            prob_percentages_reversed = [float(p) for p in prob_percentages[::-1]]
            opacities_reversed = opacities[::-1]
            bar_colors_reversed = bar_colors[::-1]
            
            # Diagnostic prints as requested by user
            print("\n=== SIMULATOR DIAGNOSTIC REPORT ===")
            print(f"Selected Patient: {patient_id}")
            print(f"Selected Genes: {gene1}, {gene2}, {gene3}")
            print(f"Perturbed Expression Values: {resolved_val1:.4f}, {resolved_val2:.4f}, {resolved_val3:.4f}")
            print(f"Raw Euclidean Distances: {[float(d) for d in distances]}")
            print(f"Shifted Scores for Softmax: {[float(s) for s in scores]}")
            print(f"Final Probabilities: {[float(p) for p in probabilities]}")
            print(f"Probabilities Sum: {float(np.sum(probabilities)):.6f}")
            print(f"Has NaN: {bool(np.isnan(probabilities).any())}")
            print(f"Has Inf: {bool(np.isinf(probabilities).any())}")
            print(f"Lengths check - Subtypes: {len(y_labels_reversed)}, Probs: {len(prob_percentages_reversed)}, Colors: {len(bar_colors_reversed)}, Opacities: {len(opacities_reversed)}")
            
            # Sort subtypes alphabetically so they match the Contour plot legend ordering:
            # Ependymoma, Glioblastoma, Medulloblastoma, Normal, Pilocytic Astrocytoma
            alphabetical_subtypes = ["ependymoma", "glioblastoma", "medulloblastoma", "normal", "pilocytic_astrocytoma"]
            
            x_vals = []
            y_names = []
            colors = []
            opacities_list = []
            text_labels = []
            hover_texts = []
            
            for subtype in alphabetical_subtypes:
                idx = subtype_names_list.index(subtype)
                prob = probabilities[idx] * 100
                opacity = 1.0 if idx == max_idx else 0.55
                color = color_map.get(subtype, "#cbd5e1")
                display_name = subtype.replace("_", " ").title()
                
                x_vals.append(prob)
                y_names.append(display_name)
                colors.append(color)
                opacities_list.append(opacity)
                text_labels.append(f" {prob:.1f}%")
                hover_texts.append(f"<b>{display_name}</b><br>Similarity: {prob:.2f}%<extra></extra>")
                
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y_names,
                orientation="h",
                width=0.85,  # Maximized bar thickness (occupies 85% of vertical slot)
                marker=dict(
                    color=colors,
                    opacity=opacities_list,
                    line=dict(color="rgba(0,0,0,0.1)", width=1.0)
                ),
                text=text_labels,
                textposition="outside",
                textfont=dict(color=PLOT_TITLE_COLOR, size=12, family="Inter"),
                hoverinfo="text",
                hovertext=hover_texts,
                showlegend=False
            ))
            
            fig.update_layout(
                template=PLOT_TEMPLATE,
                showlegend=False,  # Plotly legend completely removed
                dragmode=False,    # Disable rubber-band zoom / pan gestures entirely
                bargap=0.15,       # Tighter gaps so bars fill more of the row height
                xaxis=dict(
                    showgrid=False,   # Hide grid lines
                    zeroline=False,
                    tickfont=dict(color=PLOT_TICK_COLOR, size=10),
                    range=[0, 100],   # Similarity scale up to 100%
                    fixedrange=True,  # Lock this chart — no zoom/pan like the other plots
                ),
                yaxis=dict(
                    showticklabels=False,  # Hide subtype names from Y-axis
                    showgrid=False,        # Hide grid lines
                    type="category",
                    fixedrange=True,  # Lock this chart — no zoom/pan like the other plots
                ),
                plot_bgcolor=PLOT_PLOT_BG,
                paper_bgcolor=PLOT_PAPER_BG,
                margin=dict(l=10, r=45, t=8, b=20)  # Wider right margin so outside labels (e.g. "72.8%") never clip
            )
            
            # 3. Predicted class card (redesigned for compact light theme)
            predicted_subtype = subtype_names_list[max_idx].replace("_", " ").title()
            subtype_color = color_map.get(subtype_names_list[max_idx], "#cbd5e1")
            
            prediction_card = [
                html.H4("Predicted Subtype", style={"color": "#6B7280", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.05em", "marginTop": "0", "marginBottom": "0.2rem"}),
                html.Strong(predicted_subtype, style={"color": subtype_color, "fontSize": "1.1rem", "fontFamily": "Outfit", "display": "block", "marginBottom": "0.15rem"}),
                html.Span([
                    html.Span("Confidence: ", style={"color": "#6B7280", "fontSize": "0.72rem"}),
                    html.Strong(f"{prediction_confidence:.1f}%", style={"color": "#1F2937", "fontSize": "0.8rem"})
                ])
            ]
            
            # 4. Centroid distances list (redesigned for compact light theme)
            distance_items = []
            for i, subtype in enumerate(subtype_names_list):
                display_name = subtype.replace("_", " ").title()
                dist = distances[i]
                
                distance_items.append(
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "0.15rem"},
                        children=[
                            html.Span(display_name, style={"color": "#374151", "fontSize": "0.72rem"}),
                            html.Span(
                                f"{dist:.2f}", 
                                style={
                                    "color": "#10b981" if i == max_idx else "#4B5563",
                                    "fontWeight": "bold" if i == max_idx else "normal",
                                    "fontSize": "0.75rem"
                                }
                            )
                        ]
                    )
                )
                
            distance_card = [
                html.H4("Centroid Distances", style={"color": "#6B7280", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.05em", "marginTop": "0", "marginBottom": "0.3rem", "borderBottom": "1px solid #E5E7EB", "paddingBottom": "0.2rem"}),
                html.Div(distance_items)
            ]
            

            outputs = (fig, prediction_card, distance_card, labels[0], labels[1], labels[2])
            print(f"  Outputs (Lengths prediction_card={len(prediction_card)}, distance_card={len(distance_card)}): Successfully returned.")
            return outputs
        except Exception:
            import traceback
            print("\n==================== EXCEPTION ====================")
            traceback.print_exc()
            print("===================================================\n")
            raise
