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
from app.layouts.sidebar import Sidebar
from app.layouts.contour_card import ContourCard
from app.layouts.expression_card import ExpressionCard
from app.layouts.hotspot_card import HotspotCard
from app.layouts.network_card import NetworkCard


# Default network threshold options — kept here so ControlBar and summary cards
# share the same definition without duplication.
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
    sim_gene_1: str = None,
    sim_gene_2: str = None,
    sim_gene_3: str = None,
) -> html.Div:
    """
    Assembles the complete unified OncoLens dashboard layout.

    Follows the 2-column sidebar design.
    """
    if gene_options is None: gene_options = []
    if top_20_options is None: top_20_options = []
    if patient_options is None: patient_options = []

    # Resolve default simulator genes if none provided
    if sim_gene_1 is None and len(gene_options) > 0: sim_gene_1 = gene_options[0]["value"]
    if sim_gene_2 is None and len(gene_options) > 1: sim_gene_2 = gene_options[1]["value"]
    if sim_gene_3 is None and len(gene_options) > 2: sim_gene_3 = gene_options[2]["value"]

    return html.Div(
        className="app-container",
        children=[
            # ── 1. Left Sidebar ────────────────────────────────────────────────
            Sidebar(
                gene_options=gene_options,
                top_20_options=top_20_options,
                network_threshold_options=NETWORK_THRESHOLD_OPTIONS,
                patient_options=patient_options,
                default_x=default_x or (gene_options[0]["value"] if gene_options else None),
                default_y=default_y or (gene_options[1]["value"] if len(gene_options) > 1 else None),
                default_profile=default_profile,
                default_threshold=DEFAULT_NETWORK_THRESHOLD,
                default_patient=default_patient,
                sim_val1=sim_val1,
                sim_val2=sim_val2,
                sim_val3=sim_val3,
                sim_gene_1=sim_gene_1,
                sim_gene_2=sim_gene_2,
                sim_gene_3=sim_gene_3,
            ),

            # ── 2. Main Content Area ───────────────────────────────────────────
            html.Main(
                className="main-content",
                children=[
                    Header(
                        n_samples=n_patients,
                        n_genes=n_genes,
                        n_classes=n_classes,
                    ),
                    SummaryCards(
                        n_patients=n_patients,
                        n_genes=n_genes,
                        n_classes=n_classes,
                        best_pair=best_pair,
                        top_gene=top_gene,
                        network_threshold="0.80",
                    ),
                    # ── 3. Charts Grid (2x2) ───────────────────────────────────
                    html.Div(
                        className="charts-grid",
                        children=[
                            html.Div(ContourCard(), className="viz-slot"),
                            html.Div(HotspotCard(), className="viz-slot"),
                            html.Div(NetworkCard(), className="viz-slot"),
                            html.Div(ExpressionCard(), className="viz-slot"),
                        ]
                    )
                ]
            )
        ]
    )
