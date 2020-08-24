import networkx
from plotly import graph_objs
from plotly.offline import iplot


def plot_graph(g, edge_label_attr=None, pos=None, layout={}):
    pos = networkx.spring_layout(g) if pos is None else pos
    edge_traces = _draw_edges(g, pos)
    node_trace = _draw_nodes(g, pos)
    fig = graph_objs.Figure(data=edge_traces + [node_trace], layout=layout)
    if edge_label_attr is not None:
        _add_edge_annotation(g, pos, fig, edge_label_attr)
    iplot(fig)


def _draw_edges(g, pos):
    trace_list = [
        graph_objs.Scatter(
            x=[pos[e[0]][0], pos[e[1]][0]],
            y=[pos[e[0]][1], pos[e[1]][1]],
            mode="lines",
            showlegend=False
        )
        for e in g.edges()
    ]
    return trace_list


def _draw_nodes(g, pos):
    trace = graph_objs.Scatter(
        x=[pos[n][0] for n in g.nodes()],
        y=[pos[n][1] for n in g.nodes()],
        text=[n for n in g.nodes()],
        mode="markers+text",
        textposition="bottom center",
        marker=dict(size=20),
        showlegend=False
    )
    return trace


def _add_edge_annotation(g, pos, fig, edge_label_attr):
    for e in g.edges():
        center_x = (pos[e[0]][0] + pos[e[1]][0]) / 2
        center_y = (pos[e[0]][1] + pos[e[1]][1]) / 2
        fig.add_annotation(x=center_x, y=center_y, text=g.edges[e][edge_label_attr])
