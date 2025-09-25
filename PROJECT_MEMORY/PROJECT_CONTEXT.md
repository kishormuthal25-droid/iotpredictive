# PROJECT CONTEXT - IoT Predictive Maintenance Platform

## Core Mission
Build production-ready IoT platform for NASA SMAP/MSL equipment with:
- 80+ sensor anomaly detection using NASA Telemanom
- Real-time forecasting with Transformer models
- Automated maintenance scheduling
- Company demo by Day 26

## Key Requirements From Company
- Scalable data ingestion pipeline for IoT sensor streams
- AI anomaly detection for various sensor types (vibration, temperature, pressure)
- Predictive failure models using time series analysis
- Maintenance scheduling optimization engine
- Real-time monitoring dashboard
- Mobile alerts and work order integration

## Technical Architecture Decisions
- **Python 3.11** (staying, not upgrading to 3.13)
- **NASA Telemanom** LSTM with dynamic thresholding (official NASA algorithm)
- **Transformer** for forecasting
- **Current dashboard GUI** (keeping - it's perfect, minor improvements allowed)
- **SQLite** for development, PostgreSQL for production
- **No Git** - file-based progress tracking
- **Local storage** with manual save points

## NASA Equipment Structure
- **SMAP Satellite**: 25 sensors across 5 subsystems (Power, Communication, Attitude, Thermal, Payload)
- **MSL Mars Rover**: 55 sensors across 6 subsystems (Mobility, Power, Environmental, Scientific, Communication, Navigation)
- **Total**: 80 sensors requiring individual LSTM models

## Current System Status
- ✅ **Dashboard**: Working with beautiful GUI on localhost:8060
- ✅ **Data Pipeline**: NASA SMAP/MSL data ingestion operational
- ✅ **AI Models**: **REAL NASA TELEMANOM MODELS TRAINED** (5 SMAP sensors)
- ✅ **Model Infrastructure**: Complete training pipeline for 80-sensor scaling
- ✅ **Model Persistence**: Save/load functionality working perfectly
- 🔄 **Dashboard Integration**: Currently integrating real models (Phase 2.1)
- ❌ **Real-time Integration**: Models not yet connected to live dashboard
- ❌ **Work Orders**: Not automatically generated from anomalies

## Success Metrics
- >95% accuracy on NASA datasets with real Telemanom models
- <100ms real-time response for 80+ sensors
- Complete anomaly → forecast → work order automation
- All dashboard data from real models (no placeholders)
- Production-ready system for company demo

## Implementation Strategy
- ✅ **NASA Telemanom Integration** - COMPLETE with 5 trained models
- ✅ **Subset Model Training** - COMPLETE (5 SMAP power sensors)
- 🔄 **Dashboard Integration** - IN PROGRESS (Phase 2.1)
- ⏳ **Scale to All 80 Sensors** - Ready (infrastructure complete)
- ⏳ **Automated Maintenance Workflows** - Planned (Phase 4)
- ⏳ **End-to-End Testing** - Planned (Phase 6)

## Project Constraints
- Timeline: ~10 sessions to completion
- Keep existing GUI structure
- Focus on working software over documentation
- Pragmatic approach with flexibility for improvements