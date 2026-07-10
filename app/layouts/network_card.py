"""
Network Card – Gene Interaction & Co-expression Network.
"""

from dash import html, dcc
from app.layouts.viz_card import viz_card


def NetworkCard(
    network_threshold_options: list,
    default_threshold: float,
) -> html.Div:
    """
    Returns the viz card for the Gene Interaction & Co-expression Network.

    Includes the network threshold selector in a compact toolbar at the top,
    a large plot area taking almost the full card, and a minimal two-column footer
    for selected node details.
    """
    dd_style = {"color": "#1F2937", "fontSize": "0.85rem"}
    
    # 1. Network Toolbar (compact top row, matching Contour toolbar)
    controls_row = html.Div(
        className="contour-toolbar",
        children=[
            html.Div(
                className="toolbar-left",
                children=[
                    html.Div(className="toolbar-control", children=[
                        html.Label("Network Threshold", className="ctrl-label"),
                        dcc.Dropdown(
                            id="network-threshold-selector",
                            options=network_threshold_options,
                            value=default_threshold,
                            clearable=False,
                            style=dd_style,
                        )
                    ]),
                ]
            )
        ]
    )

    # 2. Network Plot Area (takes up maximum available card space)
    plot_area = html.Div(
        style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "minWidth": "0",
            "minHeight": "0",
        },
        children=[
            dcc.Loading(
                type="circle",
                color="#2563EB",
                parent_style={"flex": "1", "display": "flex", "flexDirection": "column", "minHeight": "0"},
                style={"flex": "1", "display": "flex", "flexDirection": "column"},
                children=dcc.Graph(
                    id="network-plot",
                    className="dash-graph",
                    config={
                        "displayModeBar": True,
                        "responsive": True,
                    },
                    style={
                        "flex": "1",
                        "height": "100%",
                        "width": "100%",
                        "minHeight": "200px",
                        "display": "block",
                    },
                ),
            )
        ],
    )
    return viz_card(
        card_id="card-network",
        title="Gene Co-expression Network",
        subtitle="Which genes are strongly co-expressed, and how do they cluster by chromosome?",
        children=html.Div(
            style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"},
            children=[
                controls_row,
                plot_area,
            ]
        ),
        footer=html.Div(
            id="network-details-card",
            children=[],
            style={"width": "100%"}
        )
    )
