@echo off
echo Memory-Optimized Sequential Testing
echo ===================================

echo Step 1: Basic imports...
python -c "import sys; print('Python OK')"

echo Step 2: Dashboard test...
set CONFIG_FILE=config/memory_optimized.yaml
python -c "from src.dashboard.app import create_dash_app; print('Dashboard OK')"

echo Step 3: Model manager...
python -c "from src.model_registry.model_manager import get_model_manager; print('Models OK')"

echo Tests complete - Check output above
pause