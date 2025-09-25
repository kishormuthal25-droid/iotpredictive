# IoT Anomaly Detection System - Project Structure

```
iot_anomaly_detection_system/
│
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── settings.py
│
├── data/
│   ├── raw/
│   │   ├── smap/
│   │   └── msl/
│   ├── processed/
│   └── models/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── stream_simulator.py
│   │   ├── kafka_producer.py
│   │   ├── kafka_consumer.py
│   │   └── database_manager.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_preprocessor.py
│   │   ├── feature_engineering.py
│   │   └── normalizer.py
│   │
│   ├── anomaly_detection/
│   │   ├── __init__.py
│   │   ├── base_detector.py
│   │   ├── lstm_detector.py
│   │   ├── lstm_autoencoder.py
│   │   ├── lstm_vae.py
│   │   └── model_evaluator.py
│   │
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── base_forecaster.py
│   │   ├── transformer_forecaster.py
│   │   ├── lstm_forecaster.py
│   │   └── forecast_evaluator.py
│   │
│   ├── maintenance/
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── optimizer.py
│   │   ├── work_order_manager.py
│   │   └── priority_calculator.py
│   │
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── alert_manager.py
│   │   ├── email_sender.py
│   │   └── notification_queue.py
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── layouts/
│   │   │   ├── __init__.py
│   │   │   ├── overview.py
│   │   │   ├── anomaly_monitor.py
│   │   │   ├── forecast_view.py
│   │   │   ├── maintenance_scheduler.py
│   │   │   └── work_orders.py
│   │   ├── callbacks/
│   │   │   ├── __init__.py
│   │   │   └── dashboard_callbacks.py
│   │   └── assets/
│   │       └── style.css
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   ├── test_anomaly_detection.py
│   ├── test_forecasting.py
│   └── test_maintenance.py
│
├── scripts/
│   ├── download_data.sh
│   ├── train_models.py
│   ├── start_pipeline.py
│   └── run_dashboard.py
│
└── notebooks/
    ├── data_exploration.ipynb
    ├── model_experiments.ipynb
    └── visualization.ipynb
```

## Module Descriptions

### 1. **Config Module**
- Centralized configuration management
- YAML-based settings for easy modification
- Environment-specific configurations

### 2. **Data Ingestion Module**
- Handles .npy and .h5 file loading
- Simulates real-time streaming
- Kafka integration for scalability
- TimescaleDB management

### 3. **Preprocessing Module**
- Data cleaning and normalization
- Feature engineering for time series
- Sliding window generation

### 4. **Anomaly Detection Module**
- Base abstract class for all detectors
- LSTM, LSTM-AE, LSTM-VAE implementations
- Model comparison and evaluation

### 5. **Forecasting Module**
- Transformer-based forecasting
- LSTM baseline comparison
- Multi-step ahead predictions

### 6. **Maintenance Module**
- Optimization engine using PuLP
- Constraint-based scheduling
- Work order generation and tracking

### 7. **Alerts Module**
- Real-time alert management
- Email notifications
- Alert queue and prioritization

### 8. **Dashboard Module**
- Dash-based interactive UI
- Real-time monitoring
- Multiple views for different aspects

### 9. **Utils Module**
- Logging configuration
- Performance metrics
- Helper functions