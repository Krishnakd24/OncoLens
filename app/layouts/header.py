"""
OncoLens Header Component.

Builds the compact top header bar containing the branding and a dataset
status pill on the right. Accepts no reactive inputs — fully static.
"""

from dash import html, dcc
from app.layouts.theme import COLOR_TEXT_SECONDARY


def Header(n_samples: int, n_genes: int, n_classes: int) -> html.Header:
    """
    Returns the top dashboard header.
    """
    return html.Header(
        className="top-header",
        children=[
            html.Div(
                className="header-left-group",
                children=[
                    html.H2("Brain Tumor Visual Analytics", className="header-title"),
                    html.Div(
                        className="header-pill",
                        children=[
                            html.Span(className="pulse-dot"),
                            html.Span(f"{n_samples} Samples", className="pill-text"),
                            html.Span("|", className="pill-divider"),
                            html.Span(f"{n_genes} Genes", className="pill-text"),
                            html.Span("|", className="pill-divider"),
                            html.Span(f"{n_classes} Subtypes", className="pill-text")
                        ]
                    )
                ]
            ),
            html.Div(
                className="header-right-group",
                children=[
                    html.Div(
                        className="search-container",
                        children=[
                            html.Span("🔍", className="search-icon"),
                            dcc.Input(
                                type="text",
                                placeholder="Search genes, pathways...",
                                className="search-input"
                            )
                        ]
                    )
                ]
            )
        ]
    )


def _pill(value: str, label: str) -> html.Div:
    return html.Div(
        className="header-pill",
        children=[
            html.Span(value, className="pill-value"),
            html.Span(label, className="pill-label"),
        ]
    )
