import plotly.io as pio
import plotly.graph_objects as go

# Tokens
BG = "#0B0F14"
PANEL = "#121923"
BORDER = "#1E2A38"
TEXT = "#E6EDF3"
MUTED = "#8B98A9"
ACCENT = "#2DD4BF"
ACCEPT_COLOR = "#22C55E"
REJECT_COLOR = "#EF4444"
REVIEW_COLOR = "#F59E0B"
NOVEL_COLOR = "#A78BFA"

def inject_theme():
    # Register Plotly template
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        colorway=[ACCENT, ACCEPT_COLOR, REJECT_COLOR, REVIEW_COLOR, NOVEL_COLOR],
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    pio.templates["qualitytwin"] = template
    pio.templates.default = "qualitytwin"
