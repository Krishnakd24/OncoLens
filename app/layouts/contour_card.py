"""
Contour Card – Multi-Gene Phenotypic Contour Plot.
"""

from dash import html, dcc
from app.layouts.viz_card import viz_card


def ContourCard() -> html.Div:
    """
    Returns the viz card for the Multi-Gene Phenotypic Contour Plot.

    Controls (Gene X, Gene Y, Demo Pair, Display Mode) live in the global
    ControlBar. This card contains only the plot and the three stat cards
    that callbacks populate dynamically.
    """
    # Demo badge (hidden until default pair is active)
    demo_badge = html.Div(
        id="demo-badge",
        children="★ Best Demonstration Pair",
        title=(
            "This pair was automatically selected during preprocessing as one of "
            "the highest subtype-separating gene pairs within the variance-filtered dataset."
        ),
        style={"display": "none"}
    )

    plot_area = html.Div([
        demo_badge,
        dcc.Loading(
            type="circle",
            color="#2563EB",
            children=dcc.Graph(
                id="contour-plot",
                config={"displayModeBar": True, "responsive": True},
                style={"height": "100%", "minHeight": "280px"},
            )
        ),
    ], style={"flex": "1", "display": "flex", "flexDirection": "column"})

    # Three statistics sub-cards below the graph
    stats_row = html.Div(
        className="contour-stats-row",
        children=[
            html.Div(id="stats-card-x", className="mini-stat-card"),
            html.Div(id="stats-card-y", className="mini-stat-card"),
            html.Div(id="stats-card-quality", className="mini-stat-card"),
        ]
    )

    return viz_card(
        card_id="card-contour",
        title="Multi-Gene Phenotypic Contour",
        subtitle="Do the expression patterns of two genes cleanly separate tumor subtypes?",
        children=[plot_area, stats_row],
    )
