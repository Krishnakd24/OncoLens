"""
Expression Card – Gene Expression Profile Explorer.
"""

from dash import html, dcc
from app.layouts.viz_card import viz_card


def ExpressionCard(
    gene_options: list,
    default_profile: str,
) -> html.Div:
    """
    Returns the viz card for the Gene Expression Profile Explorer.

    Includes the profiles violin/box plot and the statistics details panel with the gene selector at the top.
    """
    dd_style = {"color": "#1F2937", "fontSize": "0.85rem"}
    
    # Expression Gene selector control styled for the details sidebar
    gene_ctrl = html.Div(
        style={
            "marginBottom": "0.75rem",
            "paddingBottom": "0.5rem",
            "borderBottom": "1px solid #E5E7EB"
        },
        children=[
            html.Label("Expression Gene", className="ctrl-label"),
            dcc.Dropdown(
                id="profiles-gene-selector",
                options=gene_options,
                value=default_profile,
                searchable=True,
                clearable=False,
                placeholder="Select gene…",
                style=dd_style,
            )
        ]
    )

    plot_col = html.Div(
        className="sim2-chart",
        style={"flex": "1", "minWidth": "0"},
        children=[
            dcc.Loading(
                type="circle",
                color="#2563EB",
                children=dcc.Graph(
                    id="profiles-plot",
                    className="dash-graph",
                    config={"displayModeBar": True, "responsive": True},
                    style={"height": "100%", "minHeight": "240px"},
                )
            )
        ]
    )

    details_col = html.Div(
        className="side-details-panel",
        style={
            "width": "260px",
            "flexShrink": "0",
            "display": "flex",
            "flexDirection": "column",
            "marginLeft": "0.5rem"
        },
        children=[
            gene_ctrl,
            html.Div(
                id="profiles-details-card",
                children=[],
                style={"flex": "1", "minHeight": "0"}
            )
        ]
    )

    return viz_card(
        card_id="card-expression",
        title="Gene Expression Profiles",
        subtitle="How does a gene's expression vary across the five clinical cohorts?",
        children=html.Div(
            style={"display": "flex", "gap": "1rem", "flex": "1", "minHeight": "0"},
            children=[plot_col, details_col]
        ),
    )
