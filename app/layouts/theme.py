"""
OncoLens Theme Constants.

Centralises all visual design tokens used across layout builders and callbacks.
Premium Dark Mode & Glassmorphism Theme.
"""

# ── Plotly chart template ─────────────────────────────────────────────────────
PLOT_TEMPLATE = "plotly_dark"

# ── Page / surface colors ─────────────────────────────────────────────────────
COLOR_BG        = "#0A0A0B"   # Deepest background
COLOR_SURFACE   = "#111113"   # Sidebar / Card surface
COLOR_BORDER    = "#1E293B"   # Border slate-800
COLOR_BORDER_MED = "#334155"  # slate-700

# ── Typography ────────────────────────────────────────────────────────────────
COLOR_TEXT_PRIMARY   = "#E2E8F0"   # slate-200
COLOR_TEXT_SECONDARY = "#94A3B8"   # slate-400
COLOR_TEXT_MUTED     = "#64748B"   # slate-500

# ── Accent palette ────────────────────────────────────────────────────────────
COLOR_ACCENT_PRIMARY   = "#3B82F6"  # Bright electric blue
COLOR_ACCENT_SECONDARY = "#2DD4BF"  # Neon teal
COLOR_ACCENT_WARN      = "#FBBF24"  # Bright amber
COLOR_ACCENT_DANGER    = "#F87171"  # Bright coral/red

# ── Plotly figure background / grid ──────────────────────────────────────────
PLOT_PAPER_BG = "rgba(0,0,0,0)"     # Transparent to let CSS surface show through
PLOT_PLOT_BG  = "rgba(0,0,0,0)"     # Transparent for depth
PLOT_GRID     = "#2A2A30"           # Design specific grid color
PLOT_ZEROLINE = "#334155"
PLOT_TICK_COLOR = COLOR_TEXT_SECONDARY
PLOT_TITLE_COLOR = COLOR_TEXT_PRIMARY
PLOT_AXIS_LABEL_COLOR = COLOR_TEXT_SECONDARY

# ── Clinical subtype palette ──────────────────────────────────────────────────
# Maintained for biological consistency, but brightened slightly for dark mode contrast
SUBTYPE_COLORS = {
    "normal":                "#10B981",   # Emerald
    "ependymoma":            "#3B82F6",   # Blue
    "glioblastoma":          "#EF4444",   # Crimson
    "medulloblastoma":       "#8B5CF6",   # Purple
    "pilocytic_astrocytoma": "#F59E0B",   # Amber
}
