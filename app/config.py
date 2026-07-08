"""
OncoLens Configuration Module.

Stores configurable constants such as default genes for the
Interactive Virtual Expression Assayer.
"""

# Default genes for the Virtual Expression Assayer (Simulator Widget)
# C9orf24 (ProbeID: 229012_at)
# NACC2 (ProbeID: 212993_at)
# RAB31 (ProbeID: 217763_s_at)
DEFAULT_SIM_GENE_1 = "229012_at"
DEFAULT_SIM_GENE_2 = "212993_at"
DEFAULT_SIM_GENE_3 = "217763_s_at"

# Softmax temperature for centroid distance → probability conversion.
# Temperature is now computed DYNAMICALLY at runtime as:
#   T = SIMULATOR_TEMPERATURE_SCALE * std(baseline_distances)
# This ensures the 3-gene perturbation signal is always amplified
# relative to the 1000-dimensional baseline noise floor.
# The static fallback is used only if std ≈ 0 (degenerate case).
SIMULATOR_TEMPERATURE_SCALE = 0.35   # Scaling factor applied to distance std dev
SIMULATOR_TEMPERATURE_FALLBACK = 5.0  # Fallback T if distance spread is degenerate
