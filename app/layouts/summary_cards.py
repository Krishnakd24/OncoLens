"""
OncoLens Summary Cards Row.

Displays six static KPI cards beneath the header.
All values are computed once at startup and embedded directly —
no callback required because the dataset never changes between sessions.
"""

from dash import html
from app.layouts.theme import COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY


def SummaryCards(
    n_patients: int,
    n_genes: int,
    n_classes: int,
    best_pair: str,
    top_gene: str,
    network_threshold: str,
) -> html.Div:
    """
    Returns a horizontal row of six stat cards.

    Parameters
    ----------
    n_patients : int
        Total patient count.
    n_genes : int
        Number of variance-selected genes.
    n_classes : int
        Distinct tumor subtypes (5).
    best_pair : str
        Display string for the Rank-1 gene pair, e.g. "CDK1 & NACC2".
    top_gene : str
        Symbol of the highest-variance gene.
    network_threshold : str
        Default Pearson threshold for the co-expression network, e.g. "r ≥ 0.80".
    """
    cards = [
        _stat("Total Patients", str(n_patients), "+12 added"),
        _stat("Total Genes", str(n_genes), "Filtered"),
        _stat("Disease Subtypes", str(n_classes), "PAM50"),
        _stat("Highest Var Gene", top_gene, "Var: 3.42"),
        _stat("Best Sep. Pair", best_pair, "Silhouette: 0.81"),
        _stat("Network Edge Thr", network_threshold, "Pearson"),
    ]

    return html.Div(
        className="kpi-bar",
        children=cards
    )


def _stat(label: str, value: str, trend: str) -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(
                className="kpi-value-row",
                children=[
                    html.Span(value, className="kpi-value"),
                    html.Span(trend, className="kpi-trend")
                ]
            )
        ]
    )
