"""
OncoLens Callbacks - Virtual Expression Assayer (Simulator) Card.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback_context, no_update, html
from app.layouts.theme import (
    PLOT_TEMPLATE, PLOT_PAPER_BG, PLOT_PLOT_BG, PLOT_GRID, PLOT_ZEROLINE,
    PLOT_TICK_COLOR, PLOT_TITLE_COLOR, PLOT_AXIS_LABEL_COLOR,
)
import app.callbacks._state as _state


def register_simulator(app) -> None:
    """Register all simulator card callbacks."""

    # --------------------------------------------------------------------------
    # 9. Virtual Expression Assayer (Simulator Widget) bounds & values reset
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("simulator-slider-1", "min"),
         Output("simulator-slider-1", "max"),
         Output("simulator-slider-1", "value"),
         Output("simulator-slider-2", "min"),
         Output("simulator-slider-2", "max"),
         Output("simulator-slider-2", "value"),
         Output("simulator-slider-3", "min"),
         Output("simulator-slider-3", "max"),
         Output("simulator-slider-3", "value")],
        [Input("simulator-patient-selector", "value"),
         Input("simulator-gene-1", "value"),
         Input("simulator-gene-2", "value"),
         Input("simulator-gene-3", "value"),
         Input("simulator-reset-btn", "n_clicks")]
    )
    def update_simulator_slider_bounds(patient_id, gene1, gene2, gene3, n_clicks):
        """
        Dynamically adjusts the min, max, and values of the three perturbation sliders
        based on the selected patient baseline, gene selection, or reset button click.
        """
        import datetime
        ctx = callback_context
        triggered_id = ctx.triggered_id if hasattr(ctx, "triggered_id") else (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
        now_str = datetime.datetime.now().isoformat()

        print(f"\n[CALLBACK EXECUTION] timestamp={now_str} name=update_simulator_slider_bounds")
        print(f"  triggered_id: {triggered_id}")
        print(f"  Inputs: patient_id={patient_id}, gene1={gene1}, gene2={gene2}, gene3={gene3}, n_clicks={n_clicks}")

        # Ignore reset button initial render/mount triggers (when n_clicks is 0 or None)
        if triggered_id == "simulator-reset-btn" and (n_clicks is None or n_clicks == 0):
            print("  update_simulator_slider_bounds: Ignored reset button initial render trigger.")
            return [no_update] * 9

        if not patient_id:
            results = (0, 15, 5, 0, 15, 5, 0, 15, 5)
            print(f"  Outputs: {results}")
            return results

        patient_row = _state.df_expr_global[_state.df_expr_global["samples"].astype(str) == str(patient_id)]
        if patient_row.empty:
            results = (0, 15, 5, 0, 15, 5, 0, 15, 5)
            print(f"  Outputs: {results}")
            return results
        results = []
        for gene in [gene1, gene2, gene3]:
            baseline = patient_row[gene].values[0]

            g_min = _state.df_expr_global[gene].min()
            g_max = _state.df_expr_global[gene].max()
            g_range = g_max - g_min if g_max > g_min else 1.0

            s_min = max(0.0, float(g_min - 0.1 * g_range))
            s_max = float(g_max + 0.1 * g_range)

            val = float(baseline)
            if val < s_min:
                s_min = max(0.0, val - 0.5)
            if val > s_max:
                s_max = val + 0.5

            results.extend([round(s_min, 2), round(s_max, 2), round(val, 3)])

        print(f"  Outputs: {results}")
        return results

    # --------------------------------------------------------------------------
    # 10. Virtual Expression Assayer Simulation Calculation & Plotting
    # --------------------------------------------------------------------------
    @app.callback(
        [Output("simulator-plot", "figure"),
         Output("simulator-prediction-card", "children"),
         Output("simulator-distance-card", "children"),
         Output("simulator-slider-label-1", "children"),
         Output("simulator-slider-label-2", "children"),
         Output("simulator-slider-label-3", "children")],
        [Input("simulator-patient-selector", "value"),
         Input("simulator-gene-1", "value"),
         Input("simulator-gene-2", "value"),
         Input("simulator-gene-3", "value"),
         Input("simulator-slider-1", "value"),
         Input("simulator-slider-2", "value"),
         Input("simulator-slider-3", "value")]
    )
    def run_simulation(patient_id, gene1, gene2, gene3, val1, val2, val3):
        """
        Runs the centroid distance classifier on the perturbed expression vector
        and updates the horizontal probability plot, prediction card, and distances.
        """
        try:
            import datetime
            ctx = callback_context
            triggered_id = ctx.triggered_id if hasattr(ctx, "triggered_id") else (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
            now_str = datetime.datetime.now().isoformat()

            # Resolve patient ID if None
            if patient_id is None:
                patient_id = _state.patient_options_global[0]["value"] if _state.patient_options_global else "834"

            patient_row = _state.df_expr_global[_state.df_expr_global["samples"].astype(str) == str(patient_id)]
            if patient_row.empty:
                patient_row = _state.df_expr_global.iloc[0:1]
                patient_id = str(patient_row["samples"].values[0])

            # Resolve None slider values to baseline expression values (avoiding blank figures on startup/transition)
            resolved_val1 = val1 if val1 is not None else float(patient_row[gene1].values[0])
            resolved_val2 = val2 if val2 is not None else float(patient_row[gene2].values[0])
            resolved_val3 = val3 if val3 is not None else float(patient_row[gene3].values[0])

            # Copy the original patient expression vector for the 1000 genes
            patient_vector = patient_row[_state.gene_cols_global].iloc[0].values.copy()

            # Replace the 3 selected genes with the resolved slider values
            try:
                idx1 = _state.gene_cols_global.index(gene1)
                idx2 = _state.gene_cols_global.index(gene2)
                idx3 = _state.gene_cols_global.index(gene3)

                patient_vector[idx1] = resolved_val1
                patient_vector[idx2] = resolved_val2
                patient_vector[idx3] = resolved_val3
            except ValueError:
                return go.Figure(), [html.P("Gene not found.")], [html.P("Gene not found.")], "", "", ""

            from app.config import SIMULATOR_TEMPERATURE

            # Compute Euclidean distance to each subtype centroid
            diffs = _state.centroids_matrix_global - patient_vector
            distances = np.linalg.norm(diffs, axis=1) # Shape: (5,)

            # Shifted scores formulation: scores = -(distances - min(distances))
            scores = -(distances - np.min(distances))

            # Scale scores using temperature factor T (default 20.0)
            exp_scores = np.exp(scores / SIMULATOR_TEMPERATURE)
            probabilities = exp_scores / np.sum(exp_scores)

            # Print formal run_simulation diagnostics
            print("\n=== RUN_SIMULATION DIAGNOSTICS ===")
            print(f"  timestamp                     : {now_str}")
            print(f"  triggered_id                  : {triggered_id}")
            print(f"  patient_id                    : {patient_id}")
            print(f"  genes                         : {gene1}, {gene2}, {gene3}")
            print(f"  slider values (input)         : val1={val1}, val2={val2}, val3={val3}")
            print(f"  first 5 values of baseline vec: {patient_row[_state.gene_cols_global].iloc[0].values[:5].tolist()}")
            print(f"  modified indices              : idx1={idx1}, idx2={idx2}, idx3={idx3}")
            print(f"  values written to copy vec    : val1={resolved_val1:.4f}, val2={resolved_val2:.4f}, val3={resolved_val3:.4f}")
            print(f"  computed centroid distances   : {[float(d) for d in distances]}")
            print(f"  computed probabilities        : {[float(p) for p in probabilities]}")
            print(f"  Softmax temperature T         : {SIMULATOR_TEMPERATURE}")
            print("==================================\n")

            # Find highest probability index
            max_idx = int(np.argmax(probabilities))
            prediction_confidence = float(probabilities[max_idx] * 100)

            # Establish standard color mapping
            color_map = {
                "normal": "#10b981",          # Emerald
                "ependymoma": "#3b82f6",      # Blue
                "glioblastoma": "#ef4444",    # Crimson
                "medulloblastoma": "#8b5cf6", # Purple
                "pilocytic_astrocytoma": "#f59e0b" # Amber
            }

            # Helper to retrieve gene symbol
            def get_symbol(gene_id):
                match = _state.df_annot_global[_state.df_annot_global["ProbeID"] == gene_id]
                if not match.empty:
                    sym = match["Gene Symbol"].values[0]
                    if pd.notna(sym) and str(sym).strip() != "" and str(sym) != "nan":
                        return sym
                return gene_id

            # 1. Build labels displaying symbols, current slider values, and deviations
            labels = []
            for gene, val in [(gene1, resolved_val1), (gene2, resolved_val2), (gene3, resolved_val3)]:
                sym = get_symbol(gene)
                baseline = patient_row[gene].values[0]
                delta = val - baseline
                sign = "+" if delta >= 0 else ""

                label_el = [
                    html.Span(f"{sym} ({gene})"),
                    html.Span(
                        f"{val:.3f} ({sign}{delta:.3f})",
                        style={"color": "#10b981" if delta >= 0 else "#ef4444", "fontWeight": "bold"}
                    )
                ]
                labels.append(label_el)

            # 2. Horizontal probability bar chart
            y_labels = [s.replace("_", " ").title() for s in _state.subtype_names_list]
            prob_percentages = probabilities * 100

            # Highlight predicted class via color opacity boundaries
            opacities = [1.0 if i == max_idx else 0.55 for i in range(5)]
            bar_colors = [color_map.get(s, "#cbd5e1") for s in _state.subtype_names_list]

            # Reverse all lists for horizontal plotting direction so that Normal is at the top
            y_labels_reversed = y_labels[::-1]
            prob_percentages_reversed = [float(p) for p in prob_percentages[::-1]]
            opacities_reversed = opacities[::-1]
            bar_colors_reversed = bar_colors[::-1]

            # Diagnostic prints as requested by user
            print("\n=== SIMULATOR DIAGNOSTIC REPORT ===")
            print(f"Selected Patient: {patient_id}")
            print(f"Selected Genes: {gene1}, {gene2}, {gene3}")
            print(f"Perturbed Expression Values: {resolved_val1:.4f}, {resolved_val2:.4f}, {resolved_val3:.4f}")
            print(f"Raw Euclidean Distances: {[float(d) for d in distances]}")
            print(f"Shifted Scores for Softmax: {[float(s) for s in scores]}")
            print(f"Final Probabilities: {[float(p) for p in probabilities]}")
            print(f"Probabilities Sum: {float(np.sum(probabilities)):.6f}")
            print(f"Has NaN: {bool(np.isnan(probabilities).any())}")
            print(f"Has Inf: {bool(np.isinf(probabilities).any())}")
            print(f"Lengths check - Subtypes: {len(y_labels_reversed)}, Probs: {len(prob_percentages_reversed)}, Colors: {len(bar_colors_reversed)}, Opacities: {len(opacities_reversed)}")

            # Sort subtypes alphabetically so they match the Contour plot legend ordering:
            # Ependymoma, Glioblastoma, Medulloblastoma, Normal, Pilocytic Astrocytoma
            alphabetical_subtypes = ["ependymoma", "glioblastoma", "medulloblastoma", "normal", "pilocytic_astrocytoma"]

            x_vals = []
            y_names = []
            colors = []
            opacities_list = []
            text_labels = []
            hover_texts = []

            for subtype in alphabetical_subtypes:
                idx = _state.subtype_names_list.index(subtype)
                prob = probabilities[idx] * 100
                opacity = 1.0 if idx == max_idx else 0.55
                color = color_map.get(subtype, "#cbd5e1")
                display_name = subtype.replace("_", " ").title()

                x_vals.append(prob)
                y_names.append(display_name)
                colors.append(color)
                opacities_list.append(opacity)
                text_labels.append(f" {prob:.1f}%")
                hover_texts.append(f"<b>{display_name}</b><br>Similarity: {prob:.2f}%<extra></extra>")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y_names,
                orientation="h",
                width=0.85,  # Maximized bar thickness (occupies 85% of vertical slot)
                marker=dict(
                    color=colors,
                    opacity=opacities_list,
                    line=dict(color="rgba(0,0,0,0.1)", width=1.0)
                ),
                text=text_labels,
                textposition="outside",
                textfont=dict(color=PLOT_TITLE_COLOR, size=12, family="Inter"),
                hoverinfo="text",
                hovertext=hover_texts,
                showlegend=False
            ))

            fig.update_layout(
                template=PLOT_TEMPLATE,
                showlegend=False,  # Plotly legend completely removed
                dragmode=False,    # Disable rubber-band zoom / pan gestures entirely
                bargap=0.15,       # Tighter gaps so bars fill more of the row height
                xaxis=dict(
                    showgrid=False,   # Hide grid lines
                    zeroline=False,
                    tickfont=dict(color=PLOT_TICK_COLOR, size=10),
                    range=[0, 100],   # Similarity scale up to 100%
                    fixedrange=True,  # Lock this chart — no zoom/pan like the other plots
                ),
                yaxis=dict(
                    showticklabels=False,  # Hide subtype names from Y-axis
                    showgrid=False,        # Hide grid lines
                    type="category",
                    fixedrange=True,  # Lock this chart — no zoom/pan like the other plots
                ),
                plot_bgcolor=PLOT_PLOT_BG,
                paper_bgcolor=PLOT_PAPER_BG,
                margin=dict(l=10, r=45, t=8, b=20)  # Wider right margin so outside labels (e.g. "72.8%") never clip
            )

            # 3. Predicted class card (redesigned for compact light theme)
            predicted_subtype = _state.subtype_names_list[max_idx].replace("_", " ").title()
            subtype_color = color_map.get(_state.subtype_names_list[max_idx], "#cbd5e1")

            prediction_card = [
                html.H4("Predicted Subtype", style={"color": "#6B7280", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.05em", "marginTop": "0", "marginBottom": "0.2rem"}),
                html.Strong(predicted_subtype, style={"color": subtype_color, "fontSize": "1.1rem", "fontFamily": "Outfit", "display": "block", "marginBottom": "0.15rem"}),
                html.Span([
                    html.Span("Confidence: ", style={"color": "#6B7280", "fontSize": "0.72rem"}),
                    html.Strong(f"{prediction_confidence:.1f}%", style={"color": "#1F2937", "fontSize": "0.8rem"})
                ])
            ]

            # 4. Centroid distances list (redesigned for compact light theme)
            distance_items = []
            for i, subtype in enumerate(_state.subtype_names_list):
                display_name = subtype.replace("_", " ").title()
                dist = distances[i]

                distance_items.append(
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "0.15rem"},
                        children=[
                            html.Span(display_name, style={"color": "#374151", "fontSize": "0.72rem"}),
                            html.Span(
                                f"{dist:.2f}",
                                style={
                                    "color": "#10b981" if i == max_idx else "#4B5563",
                                    "fontWeight": "bold" if i == max_idx else "normal",
                                    "fontSize": "0.75rem"
                                }
                            )
                        ]
                    )
                )

            distance_card = [
                html.H4("Centroid Distances", style={"color": "#6B7280", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.05em", "marginTop": "0", "marginBottom": "0.3rem", "borderBottom": "1px solid #E5E7EB", "paddingBottom": "0.2rem"}),
                html.Div(distance_items)
            ]


            outputs = (fig, prediction_card, distance_card, labels[0], labels[1], labels[2])
            print(f"  Outputs (Lengths prediction_card={len(prediction_card)}, distance_card={len(distance_card)}): Successfully returned.")
            return outputs
        except Exception:
            import traceback
            print("\n==================== EXCEPTION ====================")
            traceback.print_exc()
            print("===================================================\n")
            raise
