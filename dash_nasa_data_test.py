import dash
from dash import html
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Dash + NASA Data Integration Test"),
    html.P("If you see this, NASA data integration import is not crashing!")
])

try:
    from src.data_ingestion.nasa_data_service import nasa_data_service
    status = nasa_data_service.get_equipment_status()
    logger.info(f"NASA equipment status: {status}")
except Exception as e:
    logger.error(f"NASA data integration import failed: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8060, debug=False)
