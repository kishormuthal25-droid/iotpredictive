import dash
from dash import html
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Dash + MLFlow Import Test"),
    html.P("If you see this, MLFlow import is not crashing!")
])

try:
    import mlflow
    logger.info("MLFlow imported successfully.")
except Exception as e:
    logger.error(f"MLFlow import failed: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8060, debug=False)
