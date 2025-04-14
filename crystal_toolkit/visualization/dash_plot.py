import dash
from dash import dcc, html
import webbrowser
import os
import time
import signal
import sys


def signal_handler(sig, frame):
    sys.exit(0)


def dash_plot(figure):

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
                figure=figure,
                style={"height": "100%", "width": "100%"},
            )
        ],
    )
    webbrowser.open("http://127.0.0.1:8050")  # 调用浏览器打开URL

    signal.signal(signal.SIGALRM, signal_handler)  # 设置定时信号
    signal.alarm(5)  # 5秒后触发信号

    app.run()
