"""
OncoLens Callbacks Package.

All reactive callbacks are split by visualization card.
Call register_all() once from app.py to register everything.
"""

from app.callbacks._state import initialise_globals
from app.callbacks.contour import register_contour
from app.callbacks.hotspot import register_hotspot
from app.callbacks.network import register_network
from app.callbacks.expression import register_expression
from app.callbacks.simulator import register_simulator


def register_all(
    app,
    df_expression,
    df_annotated,
    df_patient_di,
    df_variance_ranking,
    df_top_pairs,
    df_network_edges,
) -> None:
    """Register all dashboard callbacks."""
    initialise_globals(
        df_expression, df_annotated, df_patient_di,
        df_variance_ranking, df_top_pairs, df_network_edges
    )
    register_contour(app)
    register_hotspot(app)
    register_network(app)
    register_expression(app)
    register_simulator(app)
