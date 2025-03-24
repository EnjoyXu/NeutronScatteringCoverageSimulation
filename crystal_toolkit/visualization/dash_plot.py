import dash
from dash import dcc, html
import webbrowser


def dash_plot(figures):

    # 创建 Dash 应用
    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={
            "height": "100vh",
            "width": "100%",
            "margin": "0",
            "padding": "0",
            "overflow": "hidden",
        },
        children=[
            dcc.Graph(
                id="fullscreen-graph",
                figure=figures,
                style={"height": "100%", "width": "100%"},
            )
        ],
    )
    webbrowser.open("http://127.0.0.1:8050")  # 调用浏览器打开URL
    app.run()
