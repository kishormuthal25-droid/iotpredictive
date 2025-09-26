import dash
from dash import html

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Minimal Dash Test"),
    html.P("If you see this, Dash is working!")
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8060, debug=False)
