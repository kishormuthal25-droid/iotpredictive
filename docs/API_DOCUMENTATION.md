# 🔌 API Documentation - IoT Predictive Maintenance System

**Version**: Production-Ready v3.0
**Last Updated**: December 2024
**System Status**: ✅ Production Ready

This document provides comprehensive API documentation for the IoT Predictive Maintenance System, covering all major components and interfaces.

---

## 📊 System Overview

The IoT Predictive Maintenance System provides several programmatic interfaces:

- **🎯 Model Registry API**: MLflow-based model management
- **📊 Data Services API**: NASA data access and processing
- **🚨 Anomaly Detection API**: Real-time anomaly detection
- **🔧 Maintenance API**: Work order and scheduling management
- **📈 Dashboard API**: Dashboard data and configuration

---

## 🎯 Model Registry API

### **Model Manager (`src.model_registry.model_manager`)**

#### **`get_model_manager() -> ModelManager`**
Returns the global model manager instance with lazy loading enabled.

```python
from src.model_registry.model_manager import get_model_manager

# Get model manager (singleton pattern)
model_manager = get_model_manager()
```

#### **`ModelManager.get_available_models() -> Dict[str, Dict]`**
Returns all available models in the MLflow registry.

```python
# Get all available models
models = model_manager.get_available_models()

# Example response:
# {
#     "SMAP_00": {
#         "name": "SMAP_00",
#         "version": "1",
#         "description": "SMAP satellite power subsystem model",
#         "stage": "Production",
#         "tags": {"subsystem": "power", "satellite": "smap"}
#     },
#     "MSL_01": {
#         "name": "MSL_01",
#         "version": "2",
#         "description": "MSL rover mobility system model",
#         "stage": "Production",
#         "tags": {"subsystem": "mobility", "rover": "msl"}
#     }
# }
```

#### **`ModelManager.get_model_info(model_id: str) -> Optional[Dict]`**
Retrieves detailed information for a specific model.

```python
# Get specific model information
model_info = model_manager.get_model_info("SMAP_00")

# Returns None if model not found, or dict with model details
if model_info:
    print(f"Model: {model_info['name']}")
    print(f"Accuracy: {model_info.get('accuracy', 'N/A')}")
```

#### **`ModelManager.load_model(model_id: str) -> Optional[object]`**
Loads a specific model for inference (lazy loading).

```python
# Load model (cached after first load)
model = model_manager.load_model("SMAP_00")

if model:
    # Model is ready for inference
    predictions = model.predict(data)
```

---

## 📊 Data Services API

### **Unified Data Access (`src.data_ingestion.unified_data_access`)**

#### **`UnifiedDataAccess.get_real_time_telemetry(**kwargs) -> List[Dict]`**
Retrieves real-time telemetry data from NASA sources.

```python
from src.data_ingestion.unified_data_access import UnifiedDataAccess

data_access = UnifiedDataAccess()

# Get recent telemetry data
telemetry = data_access.get_real_time_telemetry(
    limit=100,                    # Number of records
    time_range_minutes=60,        # Last 60 minutes
    sensor_types=['temperature', 'pressure']
)

# Example response:
# [
#     {
#         "timestamp": "2024-12-27T10:30:00Z",
#         "sensor_id": "SMAP_TEMP_001",
#         "value": 23.5,
#         "unit": "celsius",
#         "status": "normal"
#     }
# ]
```

#### **`UnifiedDataAccess.get_equipment_status() -> Dict[str, Dict]`**
Returns current status of all monitored equipment.

```python
# Get equipment status
status = data_access.get_equipment_status()

# Example response:
# {
#     "SMAP_POWER_UNIT_001": {
#         "status": "operational",
#         "last_maintenance": "2024-12-01",
#         "next_maintenance": "2024-12-31",
#         "health_score": 0.92
#     }
# }
```

### **NASA Data Ingestion Service (`src.data_ingestion.nasa_data_ingestion_service`)**

#### **`NASADataIngestionService.get_available_datasets() -> List[str]`**
Lists all available NASA datasets.

```python
from src.data_ingestion.nasa_data_ingestion_service import NASADataIngestionService

ingestion_service = NASADataIngestionService()

# Get available datasets
datasets = ingestion_service.get_available_datasets()

# Returns: ["SMAP", "MSL", "SMAP_Power", "MSL_Mobility", ...]
```

#### **`NASADataIngestionService.load_dataset_sample(dataset: str, limit: int) -> Optional[DataFrame]`**
Loads a sample from a specific dataset.

```python
# Load sample data
sample = ingestion_service.load_dataset_sample("SMAP", limit=1000)

if sample is not None:
    print(f"Sample shape: {sample.shape}")
    print(f"Columns: {sample.columns.tolist()}")
```

---

## 🚨 Anomaly Detection API

### **NASA Anomaly Engine (`src.anomaly_detection.nasa_anomaly_engine`)**

#### **`NASAAnomalyEngine.get_available_models() -> Dict[str, Dict]`**
Returns available anomaly detection models.

```python
from src.anomaly_detection.nasa_anomaly_engine import NASAAnomalyEngine

anomaly_engine = NASAAnomalyEngine()

# Get available anomaly detection models
models = anomaly_engine.get_available_models()

# Returns model information similar to model registry
```

#### **`NASAAnomalyEngine.detect_anomalies(model_id: str, data: np.ndarray) -> Optional[List]`**
Performs anomaly detection on provided data.

```python
import numpy as np

# Prepare data (example: 100 timesteps, 5 features)
sensor_data = np.random.randn(100, 5)

# Detect anomalies
anomalies = anomaly_engine.detect_anomalies("SMAP_00", sensor_data)

if anomalies:
    print(f"Detected {len(anomalies)} anomalies")
    # Process anomaly results
```

#### **`NASAAnomalyEngine.get_threshold(model_id: str) -> Optional[float]`**
Retrieves the anomaly threshold for a specific model.

```python
# Get model threshold
threshold = anomaly_engine.get_threshold("SMAP_00")

if threshold:
    print(f"Anomaly threshold: {threshold}")
```

---

## 🔧 Maintenance API

### **Work Order Manager (`src.maintenance.work_order_manager`)**

#### **`WorkOrderManager.create_work_order(**kwargs) -> Dict`**
Creates a new work order.

```python
from src.maintenance.work_order_manager import WorkOrderManager

work_order_manager = WorkOrderManager()

# Create work order
work_order = work_order_manager.create_work_order(
    equipment_id="SMAP_POWER_001",
    issue_type="preventive_maintenance",
    priority="medium",
    description="Scheduled power system maintenance",
    estimated_duration=120  # minutes
)

# Returns work order with ID and details
```

#### **`WorkOrderManager.get_work_orders(status: str = None) -> List[Dict]`**
Retrieves work orders, optionally filtered by status.

```python
# Get all open work orders
open_orders = work_order_manager.get_work_orders(status="open")

# Get all work orders
all_orders = work_order_manager.get_work_orders()
```

### **Maintenance Scheduler (`src.maintenance.scheduler`)**

#### **`MaintenanceScheduler.optimize_schedule(**kwargs) -> Dict`**
Optimizes maintenance schedule based on constraints and priorities.

```python
from src.maintenance.scheduler import MaintenanceScheduler

scheduler = MaintenanceScheduler()

# Optimize schedule
optimized_schedule = scheduler.optimize_schedule(
    time_horizon_days=30,
    available_technicians=5,
    max_daily_hours=8
)

# Returns optimized maintenance schedule
```

---

## 📈 Dashboard API

### **Unified Data Service (`src.dashboard.unified_data_service`)**

#### **`UnifiedDataService.get_sensor_data(**kwargs) -> List[Dict]`**
Retrieves sensor data for dashboard visualization.

```python
from src.dashboard.unified_data_service import UnifiedDataService

data_service = UnifiedDataService()

# Get sensor data for dashboard
sensor_data = data_service.get_sensor_data(
    sensor_id="SMAP_TEMP_001",
    limit=100,
    time_range="1h"  # Last 1 hour
)
```

#### **`UnifiedDataService.get_anomaly_data(**kwargs) -> List[Dict]`**
Retrieves anomaly data for dashboard alerts.

```python
# Get recent anomalies
anomaly_data = data_service.get_anomaly_data(
    limit=50,
    severity="high",
    time_range="24h"
)

# Example response:
# [
#     {
#         "timestamp": "2024-12-27T09:15:00Z",
#         "sensor_id": "SMAP_TEMP_001",
#         "anomaly_score": 0.85,
#         "severity": "high",
#         "model_id": "SMAP_00"
#     }
# ]
```

#### **`UnifiedDataService.get_system_status() -> Dict`**
Returns overall system health and status.

```python
# Get system status
system_status = data_service.get_system_status()

# Example response:
# {
#     "overall_health": "good",
#     "active_sensors": 157,
#     "active_alerts": 3,
#     "system_uptime": "72h 15m",
#     "memory_usage": "456MB",
#     "model_cache_hits": 0.94
# }
```

---

## 🎛️ Configuration API

### **Configuration Manager (`config.settings`)**

#### **`Config.get(section: str, key: str, default=None) -> Any`**
Retrieves configuration values.

```python
from config.settings import Config

config = Config()

# Get configuration values
dashboard_port = config.get('dashboard', 'port', 8060)
log_level = config.get('logging', 'level', 'INFO')
mlflow_uri = config.get('mlflow', 'tracking_uri', './mlruns')
```

#### **`Config.get_section(section: str) -> Dict`**
Retrieves entire configuration section.

```python
# Get entire section
dashboard_config = config.get_section('dashboard')

# Returns: {"host": "localhost", "port": 8060, "debug": false, ...}
```

---

## 🚦 Health & Monitoring API

### **System Health Endpoints**

The system exposes HTTP health endpoints when the dashboard is running:

#### **`GET /_health`**
Returns system health status.

```bash
curl http://localhost:8060/_health

# Response:
# {
#     "status": "healthy",
#     "timestamp": "2024-12-27T10:30:00Z",
#     "uptime": "2h 15m",
#     "version": "3.0"
# }
```

#### **`GET /_metrics`**
Returns system performance metrics.

```bash
curl http://localhost:8060/_metrics

# Response:
# {
#     "startup_time": 1.8,
#     "memory_usage_mb": 456,
#     "active_models": 15,
#     "cache_hit_rate": 0.94,
#     "request_count": 1247
# }
```

#### **`GET /_models`**
Returns model registry status.

```bash
curl http://localhost:8060/_models

# Response:
# {
#     "total_models": 97,
#     "loaded_models": 15,
#     "model_cache_size": 10,
#     "average_load_time": 0.45
# }
```

---

## 📝 Error Handling

### **Standard Error Response Format**

All APIs use consistent error response format:

```python
# Error response structure
{
    "error": True,
    "error_type": "ModelNotFoundError",
    "message": "Model 'INVALID_MODEL' not found in registry",
    "timestamp": "2024-12-27T10:30:00Z",
    "request_id": "req_12345"
}
```

### **Common Error Types**

- **`ModelNotFoundError`**: Requested model doesn't exist
- **`DataAccessError`**: Unable to access data source
- **`ValidationError`**: Invalid input parameters
- **`ServiceUnavailableError`**: Service temporarily unavailable
- **`ConfigurationError`**: Configuration issue

---

## 🔒 Authentication & Security

### **API Security**

- **Development**: No authentication required (localhost access)
- **Production**: JWT-based authentication (configure in `config/config.yaml`)
- **CORS**: Configurable cross-origin request handling
- **Rate Limiting**: Built-in request rate limiting

### **Environment Variables**

```bash
# Security configuration
export API_KEY=your_api_key_here
export JWT_SECRET=your_jwt_secret
export ENABLE_AUTH=true  # Enable authentication in production
```

---

## 📈 Performance Guidelines

### **API Performance Targets**

- **Model Loading**: <500ms per model (lazy loading)
- **Data Queries**: <200ms for typical requests
- **Anomaly Detection**: <100ms per prediction
- **Health Endpoints**: <50ms response time

### **Best Practices**

1. **Use Caching**: Model manager implements intelligent caching
2. **Batch Requests**: Combine multiple data requests when possible
3. **Limit Data**: Use `limit` parameters to control response size
4. **Monitor Performance**: Use `/_metrics` endpoint to track performance

---

## 🧪 API Testing

### **Test Your API Integration**

```python
# Example API test script
def test_api_integration():
    # Test model registry
    from src.model_registry.model_manager import get_model_manager

    model_manager = get_model_manager()
    models = model_manager.get_available_models()
    assert len(models) > 0, "No models available"

    # Test data access
    from src.data_ingestion.unified_data_access import UnifiedDataAccess

    data_access = UnifiedDataAccess()
    telemetry = data_access.get_real_time_telemetry(limit=10)
    # Process telemetry data

    print("✅ API integration test passed")

if __name__ == "__main__":
    test_api_integration()
```

### **Run API Tests**

```bash
# Run comprehensive API tests
python tests/run_all_phase3_tests.py

# Test specific API components
python tests/mlflow_tests/test_model_registry_integration.py
python tests/e2e_tests/test_real_data_processing.py
```

---

## 📞 API Support

### **Documentation Resources**
- **System Overview**: See `README.md`
- **Development Guide**: See `CLAUDE.md`
- **Test Examples**: See `tests/` directory
- **Configuration**: See `config/config.yaml`

### **API Versioning**
- **Current Version**: v3.0 (Production Ready)
- **Compatibility**: Backward compatible with v2.x APIs
- **Deprecation**: 6-month notice for breaking changes

### **Getting Help**
1. **Check Documentation**: Review relevant API section above
2. **Run Tests**: Use test suites to validate API usage
3. **Check Health**: Use `/_health` endpoint to verify system status
4. **Review Logs**: Check system logs for detailed error information

---

**API Documentation Status**: ✅ **Complete**
**Last Updated**: December 2024
**System Compatibility**: Production Ready v3.0

This API documentation covers all major interfaces and provides practical examples for integration with the IoT Predictive Maintenance System.