from fastapi import APIRouter
import plotly.graph_objects as go
import networkx as nx
from pydantic import BaseModel
import json

router = APIRouter()

class NetworkRequest(BaseModel):
    selected_probe: str | None = None

@router.post("/api/network")
def get_network_plot(req: NetworkRequest):
    import api.main as main_app
    
    df_network_edges = main_app.df_network_edges
    df_annotated = main_app.df_annotated
    selected_probe = req.selected_probe

    if not hasattr(main_app, 'G_base') or main_app.G_base is None:
        G_base = nx.Graph()
        connected_probes = set(df_network_edges["Probe X"]).union(set(df_network_edges["Probe Y"]))
        df_nodes = df_annotated[df_annotated["ProbeID"].isin(connected_probes)]
        for _, row in df_nodes.iterrows():
            G_base.add_node(row["ProbeID"], symbol=row["Gene Symbol"], rank=row["Rank"])
        for _, row in df_network_edges.iterrows():
            G_base.add_edge(row["Probe X"], row["Probe Y"], weight=row["Spearman"])
        main_app.G_base = G_base
        main_app.network_layout_pos = nx.spring_layout(G_base, seed=42, k=0.12, iterations=50)

    G_base = main_app.G_base
    pos = main_app.network_layout_pos

    edge_x = []
    edge_y = []
    for edge in G_base.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#cbd5e1'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G_base.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        attrs = G_base.nodes[node]
        node_text.append(f"{attrs.get('symbol', 'Unknown')} ({node})")
        node_color.append(attrs.get('rank', 1))
        
        # Highlight selected
        if selected_probe and node == selected_probe:
            node_size.append(25)
        else:
            node_size.append(10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Plasma',
            reversescale=True,
            color=node_color,
            size=node_size,
            line_width=1))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Gene Co-expression Network',
                titlefont_size=14,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
                
    return {"figure": json.loads(fig.to_json())}
