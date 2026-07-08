"""
OncoLens Sidebar Component.

Contains the global application parameters and the Virtual Assayer inputs,
arranged in a vertically scrolling column that stays fixed on the left side.
"""

from dash import html, dcc

def Sidebar(
    gene_options: list,
    top_20_options: list,
    network_threshold_options: list,
    patient_options: list,
    default_x: str,
    default_y: str,
    default_profile: str,
    default_threshold: float,
    default_patient: str,
    sim_val1: float,
    sim_val2: float,
    sim_val3: float,
    sim_gene_1: str,
    sim_gene_2: str,
    sim_gene_3: str,
) -> html.Div:
    """
    Returns the left sidebar containing all controls.
    """
    
    # ── Branding / Logo ───────────────────────────────────────────────────────
    logo = html.Div(
        className="sidebar-header",
        children=[
            html.Div(
                "⚕", className="sidebar-logo-icon"
            ),
            html.H1("OncoLens", className="sidebar-logo-text")
        ]
    )

    # ── Global Parameters ─────────────────────────────────────────────────────
    global_params = html.Section(
        className="sidebar-section",
        children=[
            html.Div(
                className="sidebar-section-title",
                children=[
                    html.Span("⚙", style={"marginRight": "0.375rem"}),
                    "Global Parameters"
                ]
            ),
            html.Div(
                className="sidebar-grid",
                children=[
                    _compact_select("Gene X", "dropdown-x", gene_options, default_x),
                    _compact_select("Gene Y", "dropdown-y", gene_options, default_y),
                ]
            ),
            _compact_select("Demo Pair", "demo-pair-selector", top_20_options, 1, clearable=True),
            
            html.Div(
                className="sidebar-control-group w-full",
                children=[
                    html.Label("Contour Mode", className="sidebar-label"),
                    html.Div(
                        className="contour-mode-toggle-container",
                        children=[
                            dcc.RadioItems(
                                id="contour-display-toggle",
                                options=[
                                    {"label": " Scatter", "value": "scatter"},
                                    {"label": " Contour", "value": "both"},
                                ],
                                value="scatter",
                                inline=True,
                                className="contour-mode-radio"
                            )
                        ]
                    )
                ]
            ),
            
            html.Div(
                className="sidebar-grid",
                children=[
                    _compact_select("Chromosome", "hotspots-chr-selector", (
                        [{"label": "All", "value": "All"}]
                        + [{"label": f"Chr {c}", "value": str(c)} for c in range(1, 23)]
                        + [{"label": "Chr X", "value": "X"}, {"label": "Chr Y", "value": "Y"}]
                    ), "All"),
                    _compact_select("Network Thr", "network-threshold-selector", network_threshold_options, default_threshold),
                ]
            ),
            
            html.Div(
                className="sidebar-grid",
                children=[
                    _compact_select("Patient Baseline", "simulator-patient-selector", patient_options, default_patient),
                    _compact_select("Expression Gene", "profiles-gene-selector", gene_options, default_profile),
                ]
            )
        ]
    )

    # ── Virtual Assayer (Simulator) ───────────────────────────────────────────
    virtual_assayer = html.Section(
        className="sidebar-section",
        children=[
            html.Div(
                className="sidebar-section-title",
                children=[
                    html.Span("⑂", style={"marginRight": "0.375rem"}),
                    "Virtual Assayer"
                ]
            ),
            html.Div(
                className="sim-controls-box",
                children=[
                    _sim_gene_row("simulator-gene-1", gene_options, sim_gene_1, "simulator-slider-label-1", "simulator-slider-1", sim_val1),
                    _sim_gene_row("simulator-gene-2", gene_options, sim_gene_2, "simulator-slider-label-2", "simulator-slider-2", sim_val2),
                    _sim_gene_row("simulator-gene-3", gene_options, sim_gene_3, "simulator-slider-label-3", "simulator-slider-3", sim_val3),
                ]
            ),
            html.Div(
                className="sim-output-box",
                children=[
                    html.Div("Similarity to Centroids", className="sidebar-label", style={"marginBottom": "0.5rem"}),
                    # This div is targeted by the callback to inject the HTML bars
                    html.Div(id="simulator-bars", className="sim-bars-container")
                ]
            ),
            html.Button(
                "↺ Reset Simulator",
                id="simulator-reset-btn",
                n_clicks=0,
                className="btn-reset-sidebar",
            )
        ]
    )

    return html.Aside(
        className="sidebar",
        children=[
            logo,
            html.Div(
                className="sidebar-body custom-scrollbar",
                children=[
                    global_params,
                    html.Hr(className="sidebar-divider"),
                    virtual_assayer
                ]
            )
        ]
    )


def _compact_select(label: str, id: str, options: list, default_val, clearable: bool = False) -> html.Div:
    return html.Div(
        className="sidebar-control-group",
        children=[
            html.Label(label, className="sidebar-label"),
            dcc.Dropdown(
                id=id,
                options=options,
                value=default_val,
                clearable=clearable,
                searchable=True,
                className="compact-dropdown",
            )
        ]
    )

def _sim_gene_row(dropdown_id: str, options: list, default_val, label_id: str, slider_id: str, slider_val: float) -> html.Div:
    return html.Div(
        className="sim-gene-row-sidebar",
        children=[
            html.Div(
                className="sim-gene-row-top",
                children=[
                    html.Div(
                        dcc.Dropdown(
                            id=dropdown_id,
                            options=options,
                            value=default_val,
                            clearable=False,
                            searchable=True,
                            className="sim-compact-dropdown",
                        ),
                        style={"flex": "1"}
                    ),
                    html.Div(id=label_id, className="sim-slider-val-label")
                ]
            ),
            dcc.Slider(
                id=slider_id,
                min=0,
                max=20,
                step=0.1,
                value=slider_val,
                updatemode="drag",
                tooltip={"placement": "bottom", "always_visible": False},
                className="sim-slider-sidebar",
            )
        ]
    )
