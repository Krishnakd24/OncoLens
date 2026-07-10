"""
OncoLens Dashboard Assembler.

`create_layout()` is the single entry point called by app.py.
It accepts all runtime data needed to pre-populate static values and
assembles the full unified dashboard from individual layout components.

No reactive callbacks are defined here — only static component assembly.
"""

from dash import html, dcc

from app.layouts.header import Header
from app.layouts.summary_cards import SummaryCards
from app.layouts.contour_card import ContourCard
from app.layouts.expression_card import ExpressionCard
from app.layouts.hotspot_card import HotspotCard
from app.layouts.network_card import NetworkCard
from app.layouts.simulator_card import SimulatorCard


# Default network threshold options — shared across layout components
NETWORK_THRESHOLD_OPTIONS = [
    {"label": "r ≥ 0.70 (Dense)",    "value": 0.70},
    {"label": "r ≥ 0.75",             "value": 0.75},
    {"label": "r ≥ 0.80 (Standard)", "value": 0.80},
    {"label": "r ≥ 0.85",             "value": 0.85},
    {"label": "r ≥ 0.90 (Sparse)",   "value": 0.90},
]
DEFAULT_NETWORK_THRESHOLD = 0.80


def create_layout(
    n_patients: int = 0,
    n_genes: int = 0,
    n_classes: int = 5,
    best_pair: str = "—",
    top_gene: str = "—",
    gene_options: list = None,
    top_20_options: list = None,
    patient_options: list = None,
    default_x: str = None,
    default_y: str = None,
    default_profile: str = None,
    default_patient: str = None,
    sim_val1: float = 5.0,
    sim_val2: float = 5.0,
    sim_val3: float = 5.0,
    network_threshold: float = 0.80,
) -> html.Div:
    """
    Assembles the complete unified OncoLens dashboard layout.

    All five visualizations are statically embedded; no tab routing required.
    Called once at application startup by app.py.
    """
    gene_options   = gene_options   or []
    top_20_options = top_20_options or []
    patient_options = patient_options or []

    # Resolve default Gene X and Y selections if not provided
    resolved_default_x = default_x or (gene_options[0]["value"] if gene_options else None)
    resolved_default_y = default_y or (gene_options[1]["value"] if len(gene_options) > 1 else None)

    # Format network threshold for display in SummaryCards
    _threshold_label = f"r ≥ {network_threshold:.2f}"

    return html.Div(
        id="oncolens-dashboard",
        className="oncolens-dashboard",
        children=[

            # ============================================================
            # Hidden Stores
            # ------------------------------------------------------------
            # Reserved for future cross-visualization synchronization.
            # ============================================================
            dcc.Store(id="selected-gene"),
            dcc.Store(id="selected-patient"),

            # ============================================================
            # Dashboard Header
            # ------------------------------------------------------------
            # Branding + dataset summary.
            # ============================================================
            Header(
                n_samples=n_patients,
                n_genes=n_genes,
                n_classes=n_classes,
            ),

            # ============================================================
            # KPI Summary Strip
            # ------------------------------------------------------------
            # Six static stat cards (computed once at startup):
            #   Patients | Genes | Classes | Best Pair | Top Gene | Network Thr
            # ============================================================
            SummaryCards(
                n_patients=n_patients,
                n_genes=n_genes,
                n_classes=n_classes,
                best_pair=best_pair,
                top_gene=top_gene,
                network_threshold=_threshold_label,
            ),

            # ============================================================
            # Main Dashboard
            # ------------------------------------------------------------
            # The dashboard is divided into two horizontal sections:
            #
            #   Row 1 (Primary Analysis)
            #       • Contour Plot
            #       • Gene Network
            #       • Simulator
            #
            #   Row 2 (Supporting Analysis)
            #       • Chromosomal Hotspot
            #       • Expression Explorer
            #
            # This hierarchy gives maximum space to the three most
            # interactive visualizations while keeping the supporting
            # analyses easily accessible.
            # ============================================================
            html.Div(
                className="dashboard-main",
                children=[

                    # ====================================================
                    # PRIMARY ANALYSIS ROW
                    # ====================================================
                    html.Div(
                        className="dashboard-row-top",
                        children=[

                            # --------------------------------------------
                            # Multi-Gene Phenotypic Contour
                            # --------------------------------------------
                            ContourCard(
                                gene_options=gene_options,
                                top_20_options=top_20_options,
                                default_x=resolved_default_x,
                                default_y=resolved_default_y,
                            ),

                            # --------------------------------------------
                            # Gene Co-expression Network
                            # --------------------------------------------
                            NetworkCard(
                                network_threshold_options=NETWORK_THRESHOLD_OPTIONS,
                                default_threshold=DEFAULT_NETWORK_THRESHOLD,
                            ),

                            # --------------------------------------------
                            # Interactive Virtual Expression Assayer
                            # --------------------------------------------
                            SimulatorCard(
                                gene_options=gene_options,
                                patient_options=patient_options,
                                default_patient=default_patient,
                                val1=sim_val1,
                                val2=sim_val2,
                                val3=sim_val3,
                            ),

                        ],
                    ),

                    # ====================================================
                    # SUPPORTING ANALYSIS ROW
                    # ====================================================
                    html.Div(
                        className="dashboard-row-bottom",
                        children=[

                            # --------------------------------------------
                            # Chromosomal Hotspot Mapping
                            # --------------------------------------------
                            HotspotCard(
                                default_chr="All",
                            ),

                            # --------------------------------------------
                            # Gene Expression Profile Explorer
                            # --------------------------------------------
                            ExpressionCard(
                                gene_options=gene_options,
                                default_profile=default_profile,
                            ),

                        ],
                    ),

                ],
            ),

            # ============================================================
            # Footer
            # ============================================================
            html.Footer(
                id="dash-footer",
                className="dash-footer",
                children=[
                    html.Span(
                        "OncoLens v1.0  •  CS661 Visual Analytics Project  •  Group 11  •  Research tool only — not a clinical diagnostic instrument.",
                        className="footer-text",
                    )
                ],
            ),

        ],
    )