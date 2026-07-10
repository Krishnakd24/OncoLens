"""
OncoLens Callbacks - Shared helper functions.
"""
from typing import Tuple, Optional
from dash import html
import pandas as pd


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
