"""
Hotspot Card – Chromosomal Hotspot Mapping.
"""

from dash import html, dcc
from app.layouts.viz_card import viz_card


def HotspotCard(
    default_chr: str = "All",
) -> html.Div:
    """
    Returns the viz card for Chromosomal Hotspot & Cytoband Mapping.

    Chromosome selector dropdown is placed on the right side of the card header.
    """
    dd_style = {"color": "#1F2937", "fontSize": "0.85rem"}
    
    chr_dropdown = html.Div(
        style={"width": "180px", "marginTop": "2px"},
        children=[
            dcc.Dropdown(
                id="hotspots-chr-selector",
                options=(
                    [{"label": "All Chromosomes", "value": "All"}]
                    + [{"label": f"Chr {c}", "value": str(c)} for c in range(1, 23)]
                    + [{"label": "Chr X", "value": "X"}, {"label": "Chr Y", "value": "Y"}]
                ),
                value=default_chr,
                clearable=False,
                style=dd_style,
            )
        ]
    )

    # Plot area stretching to full width and height
    plot_area = html.Div(
        className="sim2-chart",
        style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "minWidth": "0",
            "minHeight": "0"
        },
        children=[
            dcc.Loading(
                type="circle",
                color="#2563EB",
                children=dcc.Graph(
                    id="hotspots-plot",
                    className="dash-graph",
                    config={"displayModeBar": True, "responsive": True},
                    style={"height": "100%", "minHeight": "240px"},
                )
            )
        ]
    )

    # Footer element for highlighted gene details (updated via callback)
    footer_details = html.Div(
        id="hotspots-details-card",
        className="hotspots-footer-content"
    )

    return viz_card(
        card_id="card-hotspot",
        title="Chromosomal Hotspot Mapping",
        subtitle="Where are the highest-variance genes physically located across the genome?",
        children=plot_area,
        footer=footer_details,
        header_right=chr_dropdown
    )
