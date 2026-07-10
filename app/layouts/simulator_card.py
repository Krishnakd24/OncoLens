"""
Simulator Card – Interactive Virtual Expression Assayer.

Two-panel scientific workspace: left control panel, right results panel.
"""

from dash import html, dcc
from app.layouts.viz_card import viz_card
from app.config import DEFAULT_SIM_GENE_1, DEFAULT_SIM_GENE_2, DEFAULT_SIM_GENE_3


def SimulatorCard(
    gene_options: list,
    patient_options: list,
    default_patient: str,
    val1: float = 5.0,
    val2: float = 5.0,
    val3: float = 5.0,
) -> html.Div:
    """
    Two-column simulator card.
    Left: compact experiment controls.  Right: similarity chart + result panels.
    """
    dd_style = {"color": "#1F2937", "fontSize": "0.82rem"}

    # ── LEFT COLUMN: Control Panel ────────────────────────────────────────────

    left_col = html.Div(
        className="sim2-left",
        children=[
            # Patient selector + Reset
            html.Div(
                className="sim2-patient-row",
                children=[
                    html.Div(style={"flex": "1"}, children=[
                        html.Label("Patient", className="ctrl-label"),
                        dcc.Dropdown(
                            id="simulator-patient-selector",
                            options=patient_options,
                            value=default_patient,
                            clearable=False,
                            searchable=True,
                            style=dd_style,
                        ),
                    ]),
                    html.Button(
                        "↺ Reset",
                        id="simulator-reset-btn",
                        n_clicks=0,
                        className="btn-reset",
                        style={"padding": "4px 10px", "fontSize": "0.72rem",
                               "alignSelf": "flex-end", "marginBottom": "1px"},
                    ),
                ]
            ),

            # Gene controls: each gene dropdown followed immediately by its slider
            _gene_block("Gene 1", "simulator-gene-1", DEFAULT_SIM_GENE_1,
                         "simulator-slider-label-1", "simulator-slider-1", val1, gene_options, dd_style),
            _gene_block("Gene 2", "simulator-gene-2", DEFAULT_SIM_GENE_2,
                         "simulator-slider-label-2", "simulator-slider-2", val2, gene_options, dd_style),
            _gene_block("Gene 3", "simulator-gene-3", DEFAULT_SIM_GENE_3,
                         "simulator-slider-label-3", "simulator-slider-3", val3, gene_options, dd_style),
        ]
    )

    # ── RIGHT COLUMN: Results Panel ───────────────────────────────────────────

    # Custom HTML legend centered horizontally, one single row
    legend_items = [
        ("Ependymoma", "#3b82f6"),
        ("Glioblastoma", "#ef4444"),
        ("Medulloblastoma", "#8b5cf6"),
        ("Normal", "#10b981"),
        ("Pilocytic Astrocytoma", "#f59e0b")
    ]
    
    html_legend = html.Div(
        className="sim2-html-legend",
        children=[
            html.Div(
                className="sim2-legend-item",
                children=[
                    html.Span(className="sim2-legend-dot", style={"backgroundColor": color}),
                    html.Span(name, className="sim2-legend-label"),
                ]
            ) for name, color in legend_items
        ]
    )

    right_col = html.Div(
        className="sim2-right",
        children=[
            # Custom HTML Legend
            html_legend,

            # Large similarity chart (primary visual)
            html.Div(
                className="sim2-chart",
                children=[
                    dcc.Loading(
                        type="circle",
                        color="#2563EB",
                        parent_style={"height": "100%", "flex": "1 1 0", "display": "flex", "flexDirection": "column"},
                        children=dcc.Graph(
                            id="simulator-plot",
                            className="dash-graph",
                            config={"responsive": True, "displayModeBar": False},
                            style={"height": "100%", "flex": "1 1 0"},
                        ),
                    ),
                ]
            ),

            # Compact result panels side-by-side
            html.Div(
                className="sim2-results",
                children=[
                    html.Div(id="simulator-prediction-card", className="sim2-result-panel"),
                    html.Div(id="simulator-distance-card", className="sim2-result-panel"),
                ]
            ),
        ]
    )

    # ── DISCLAIMER FOOTER ─────────────────────────────────────────────────────

    disclaimer = html.Div(
        className="sim2-disclaimer",
        children=[
            html.Strong("⚠ ", style={"color": "#D97706"}),
            "Research tool only — not a clinical diagnostic instrument.",
        ]
    )

    return viz_card(
        card_id="card-simulator",
        title="Interactive Virtual Expression Assayer",
        subtitle="How does perturbing a gene's expression level shift predicted subtype similarity?",
        children=html.Div(
            className="sim2-body",
            children=[left_col, right_col],
        ),
        footer=disclaimer,
        extra_class="viz-card--full-width",
    )


# ── Private helper ────────────────────────────────────────────────────────────

def _gene_block(label, dd_id, dd_default, lbl_id, slider_id, val, options, dd_style):
    """One gene dropdown stacked directly above its slider."""
    return html.Div(
        className="sim2-gene-block",
        children=[
            html.Label(label, className="ctrl-label"),
            dcc.Dropdown(
                id=dd_id, options=options, value=dd_default,
                clearable=False, searchable=True, style=dd_style,
            ),
            html.Div(id=lbl_id, className="sim2-slider-label"),
            dcc.Slider(
                id=slider_id, min=0, max=15, step=0.01, value=val,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
        ]
    )
