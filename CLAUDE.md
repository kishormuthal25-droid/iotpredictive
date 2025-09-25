# CLAUDE.md - Developer Guide

This file provides comprehensive guidance for Claude Code when working with the **IoT Predictive Maintenance System** - a production-ready platform with MLflow-optimized NASA data processing.

---

## 🚀 Quick Start (Production System)

**Primary Entry Point (RECOMMENDED):**
```bash
python launch_real_data_dashboard.py
```

**System Features:**
- ✅ **<2 second startup** (MLflow lazy loading optimization)
- ✅ **97+ NASA Telemanom models** (SMAP/MSL datasets)
- ✅ **Real-time dashboard** at http://localhost:8060
- ✅ **<512MB memory usage** (production optimized)
- ✅ **6 specialized monitoring layouts**

---

## 📊 System Architecture (Current - Optimized)

### **High-Level Architecture**
This is a **production-ready IoT Predictive Maintenance Platform** with aerospace-grade reliability:

```
NASA SMAP/MSL Data → MLflow Model Registry → Anomaly Detection Engine → Production Dashboard
```

**Key Components:**
- **NASA Data Integration**: Real satellite (SMAP) and Mars rover (MSL) telemetry
- **MLflow Model Registry**: 97+ pre-trained NASA Telemanom LSTM models
- **Lazy Loading System**: Models load on-demand for <2s startup
- **Production Dashboard**: 6 specialized layouts with <200ms response time
- **Comprehensive Testing**: 47+ test suites ensuring reliability

### **Directory Structure (Current)**
```
IOT Predictive Maintenance System/
├── launch_real_data_dashboard.py    # 🚀 MAIN ENTRY POINT
├── src/                             # Core application code
│   ├── dashboard/                   # Complete dashboard system (40+ files)
│   │   ├── layouts/                 # 6 specialized page layouts
│   │   ├── components/              # Reusable UI components
│   │   ├── callbacks/               # Dashboard interaction logic
│   │   └── app.py                   # Main dashboard application
│   ├── data_ingestion/              # NASA SMAP/MSL data processing
│   ├── anomaly_detection/           # ML models and detection engine
│   ├── model_registry/              # MLflow model management
│   ├── forecasting/                 # Time series forecasting
│   ├── maintenance/                 # Work order and scheduling
│   └── utils/                       # Shared utilities
├── tests/                           # 47+ comprehensive test suites
│   ├── mlflow_tests/                # MLflow integration tests
│   ├── e2e_tests/                   # End-to-end workflow tests
│   ├── system_tests/                # Performance and system tests
│   └── run_all_phase3_tests.py      # Test runner
├── data/                            # Data storage
│   ├── raw/smap/                    # NASA SMAP satellite data
│   ├── raw/msl/                     # NASA MSL rover data
│   └── models/                      # Trained model files (97+ models)
├── config/                          # Configuration management
└── PROJECT_MEMORY/                  # Project documentation
```

---

## 🛠️ Development Workflow

### **Environment Setup**
```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch system (auto-downloads data if needed)
python launch_real_data_dashboard.py
```

### **System Startup Options**
```bash
# Standard launch (recommended)
python launch_real_data_dashboard.py

# With environment specification
python launch_real_data_dashboard.py --environment production

# Custom port
python launch_real_data_dashboard.py --port 8070

# Debug mode
python launch_real_data_dashboard.py --debug
```

### **Data Management**
```bash
# Data is auto-downloaded on first run, but manual commands available:

# Download NASA datasets (if needed)
python scripts/download_data.sh

# Check data integrity
python scripts/validate_data.py

# Update model registry
python scripts/update_models.py
```

---

## 🧪 Testing & Quality Assurance

### **Comprehensive Test Suite (47+ Tests)**
```bash
# Run all Phase 3 tests (recommended)
python tests/run_all_phase3_tests.py

# Category-specific tests
python -m pytest tests/mlflow_tests/          # MLflow integration
python -m pytest tests/e2e_tests/             # End-to-end workflows
python -m pytest tests/system_tests/          # System performance
python -m pytest tests/integration/           # Legacy integration tests
python -m pytest tests/unit/                  # Unit tests
```

### **Performance Validation**
```bash
# Critical performance tests
python tests/mlflow_tests/test_model_lazy_loading.py     # <2s startup validation
python tests/system_tests/test_startup_sequence.py      # Memory usage validation
python tests/e2e_tests/test_complete_dashboard_flow.py  # Dashboard performance
```

### **Test Categories**
- **MLflow Tests**: Model registry, lazy loading, caching performance
- **E2E Tests**: Complete user workflows, dashboard interaction
- **System Tests**: Startup time, memory usage, resource management
- **Integration Tests**: Data pipeline, component integration
- **Unit Tests**: Individual component functionality

---

## ⚙️ Configuration System

### **Environment-Specific Configuration**
The system uses YAML-based configuration with environment overrides:

```yaml
# config/config.yaml (main configuration)
system:
  environment: development          # development, staging, production, testing
  log_level: INFO

anomaly_detection:
  enabled: true
  models:
    - nasa_telemanom
  thresholds:
    default: 0.5

dashboard:
  host: localhost
  port: 8060
  debug: false
  auto_reload: true

mlflow:
  tracking_uri: ./mlruns
  lazy_loading: true               # Critical for <2s startup
  model_cache_size: 10
```

### **Environment Variables**
```bash
# Set environment
export ENVIRONMENT=production        # Affects logging, caching, performance
export LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR
export DASHBOARD_PORT=8060          # Custom dashboard port
export MLFLOW_TRACKING_URI=./mlruns # MLflow model registry location
```

---

## 🔧 Development Guidelines

### **Code Style & Standards**
```bash
# Code formatting (if available)
black src/ tests/ scripts/

# Linting (if available)
flake8 src/ tests/ scripts/

# Type checking (if available)
mypy src/

# Import sorting (if available)
isort src/ tests/ scripts/
```

### **Performance Requirements**
When making changes, ensure these performance targets are maintained:

- **Startup Time**: <2 seconds (tested by `test_model_lazy_loading.py`)
- **Memory Usage**: <512MB baseline (tested by `test_startup_sequence.py`)
- **Dashboard Response**: <200ms for interactions
- **Model Loading**: <500ms per model (lazy loading)

### **Testing Requirements**
- Add tests for new functionality
- Run `python tests/run_all_phase3_tests.py` before commits
- Ensure performance tests pass
- Update documentation for significant changes

---

## 🎯 Key Components Guide

### **1. Dashboard System (`src/dashboard/`)**
- **Entry Point**: `src/dashboard/app.py`
- **Layouts**: 6 specialized pages in `src/dashboard/layouts/`
- **Components**: Reusable UI components in `src/dashboard/components/`
- **Data Service**: `src/dashboard/unified_data_service.py`

### **2. Model Management (`src/model_registry/`)**
- **Model Manager**: `src/model_registry/model_manager.py`
- **MLflow Integration**: Lazy loading with caching
- **97+ NASA Models**: Pre-trained Telemanom LSTM models

### **3. Anomaly Detection (`src/anomaly_detection/`)**
- **NASA Engine**: `src/anomaly_detection/nasa_anomaly_engine.py`
- **Telemanom LSTM**: `src/anomaly_detection/nasa_telemanom.py`
- **Real-time Detection**: Sub-second anomaly detection

### **4. Data Processing (`src/data_ingestion/`)**
- **NASA Data Service**: `src/data_ingestion/nasa_data_ingestion_service.py`
- **Unified Access**: `src/data_ingestion/unified_data_access.py`
- **Real-time Streaming**: `src/data_ingestion/realtime_streaming_service.py`

---

## 🚀 Production Deployment

### **System Requirements**
- **Minimum**: 2 cores, 4GB RAM, 5GB disk space
- **Recommended**: 4+ cores, 8GB+ RAM, 20GB+ disk space
- **OS**: Linux, macOS, Windows (cross-platform)

### **Production Configuration**
```bash
# Set production environment
export ENVIRONMENT=production
export LOG_LEVEL=WARNING

# Optional: Use PostgreSQL instead of SQLite
export DATABASE_URL=postgresql://user:pass@localhost/iot_maintenance

# Launch with production settings
python launch_real_data_dashboard.py --environment production
```

### **Health Monitoring**
```bash
# Built-in health endpoints
curl http://localhost:8060/_health        # System health check
curl http://localhost:8060/_metrics       # Performance metrics
curl http://localhost:8060/_models        # Model status
```

---

## 📊 Model Information

### **NASA Telemanom Models (97+ Models)**
- **SMAP Power Subsystem**: 35+ models for satellite power monitoring
- **SMAP Thermal Management**: 28+ models for temperature control
- **MSL Mobility System**: 20+ models for rover navigation
- **MSL Science Instruments**: 14+ models for scientific equipment

### **Model Performance**
- **Average Accuracy**: 92.9% across all models
- **Loading Time**: <500ms per model (lazy loading)
- **Memory Footprint**: Optimized for production deployment
- **Inference Speed**: <100ms per prediction

---

## 🐛 Troubleshooting

### **Common Issues**

**1. Slow Startup (>2 seconds)**
```bash
# Check if lazy loading is enabled
grep -r "lazy_loading: true" config/

# Run performance test
python tests/mlflow_tests/test_model_lazy_loading.py
```

**2. High Memory Usage (>512MB)**
```bash
# Check memory usage
python tests/system_tests/test_startup_sequence.py

# Monitor system resources
htop  # or Task Manager on Windows
```

**3. Dashboard Not Loading**
```bash
# Check if port 8060 is available
netstat -an | grep 8060

# Try different port
python launch_real_data_dashboard.py --port 8070

# Check logs
tail -f logs/dashboard.log
```

**4. Model Loading Issues**
```bash
# Validate MLflow setup
python -c "from src.model_registry.model_manager import get_model_manager; print(get_model_manager().get_available_models())"

# Check model files
ls -la data/models/

# Test model registry
python tests/mlflow_tests/test_model_registry_integration.py
```

---

## 📈 Performance Monitoring

### **Built-in Metrics**
The system includes comprehensive performance monitoring:

- **Startup Time**: Tracked and validated (<2s requirement)
- **Memory Usage**: Real-time monitoring (<512MB target)
- **Dashboard Response Time**: <200ms for all interactions
- **Model Loading Performance**: <500ms per model
- **Error Rates**: Comprehensive error tracking and recovery

### **Performance Testing**
```bash
# Complete performance validation
python tests/run_all_phase3_tests.py

# Specific performance benchmarks
python tests/mlflow_tests/test_model_lazy_loading.py --benchmark
```

---

## 🚦 Development Best Practices

### **When Adding New Features**
1. **Follow existing patterns** in the codebase
2. **Add comprehensive tests** (unit, integration, performance)
3. **Update documentation** as needed
4. **Validate performance impact** (<2s startup, <512MB memory)
5. **Run full test suite** before committing

### **When Modifying Existing Code**
1. **Understand the component** by reading related tests
2. **Check performance implications**
3. **Update tests** to reflect changes
4. **Verify backward compatibility**
5. **Test with real NASA data**

### **Code Organization**
- **Dashboard**: Keep UI components modular and reusable
- **Models**: Maintain MLflow integration and lazy loading
- **Data**: Preserve NASA data processing accuracy
- **Tests**: Add tests for new functionality, maintain coverage

---

## 📞 Support & Resources

### **Documentation**
- **README.md**: System overview and quick start
- **PROJECT_MEMORY/**: Detailed project status and history
- **tests/**: Examples of system usage and validation
- **config/**: Configuration examples and options

### **Key Files for Reference**
- **`launch_real_data_dashboard.py`**: Main system entry point
- **`src/dashboard/app.py`**: Dashboard application
- **`src/model_registry/model_manager.py`**: Model management
- **`tests/run_all_phase3_tests.py`**: Comprehensive testing
- **`config/config.yaml`**: System configuration

### **Performance Benchmarks**
- **Current System**: <2s startup, <512MB memory, <200ms response
- **Model Capacity**: 97+ NASA models with lazy loading
- **Test Coverage**: 47+ comprehensive test suites
- **Data Processing**: Real NASA SMAP/MSL datasets

---

**System Status**: ✅ **Production Ready**
**Last Updated**: December 2024
**Version**: Production-Ready v3.0

This system represents a production-grade IoT predictive maintenance platform with aerospace-level reliability and performance optimization.