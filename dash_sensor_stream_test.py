import dash
from dash import html
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Dash + Sensor Streaming Test"),
    html.P("If you see this, sensor streaming import is not crashing!")
])

try:
    from src.dashboard.sensor_stream_manager import SensorStreamManager
    manager = SensorStreamManager(buffer_size=10, update_frequency=0.5)
    manager.start_streaming()
    logger.info("SensorStreamManager initialized and streaming started.")
except Exception as e:
    logger.error(f"Sensor streaming import failed: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8060, debug=False)
