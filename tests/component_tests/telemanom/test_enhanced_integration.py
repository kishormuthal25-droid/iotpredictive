"""
Enhanced Forecasting Integration Test
Quick test to verify all new components work with existing system
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration_loading():
    """Test that enhanced configuration loads correctly"""
    print("\n" + "="*60)
    print("Testing Configuration Loading")
    print("="*60)

    try:
        from config.settings import settings
        config = settings.to_dict()

        # Test new configuration sections
        assert 'forecasting' in config, "Forecasting config missing"
        assert 'failure_probability' in config, "Failure probability config missing"
        assert 'scenario_analysis' in config, "Scenario analysis config missing"
        assert 'risk_assessment' in config, "Risk assessment config missing"

        # Test enhanced forecasting config
        forecasting_config = config['forecasting']
        assert 'enhanced' in forecasting_config, "Enhanced forecasting config missing"
        assert 'general' in forecasting_config, "General forecasting config missing"

        print("[PASS] Configuration loading: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Configuration loading: FAILED - {e}")
        return False

def test_enhanced_forecaster():
    """Test enhanced forecaster instantiation"""
    print("\n" + "="*60)
    print("Testing Enhanced Forecaster")
    print("="*60)

    try:
        from src.forecasting.enhanced_forecaster import EnhancedForecaster
        from config.settings import get_config

        # Create forecaster
        forecaster = EnhancedForecaster(
            base_model="transformer",
            uncertainty_methods=["quantile", "ensemble"],
            confidence_levels=[0.8, 0.9, 0.95],
            horizon=24,
            lookback=50
        )

        print(f"[PASS] Enhanced Forecaster created: {forecaster}")

        # Test with mock data
        mock_data = np.random.normal(50, 10, (200, 3))

        # This would normally require training, so we'll just test instantiation
        print("[PASS] Enhanced forecaster: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Enhanced forecaster: FAILED - {e}")
        return False

def test_failure_probability_estimator():
    """Test failure probability estimator"""
    print("\n" + "="*60)
    print("Testing Failure Probability Estimator")
    print("="*60)

    try:
        from src.forecasting.failure_probability import FailureProbabilityEstimator, FailurePrediction, SeverityLevel

        # Create estimator
        estimator = FailureProbabilityEstimator()

        # Test with mock failure prediction
        mock_prediction = FailurePrediction(
            equipment_id='SMAP',
            component_id='power_system',
            failure_probability=0.3,
            time_to_failure=72,
            severity=SeverityLevel.HIGH
        )

        print(f"[PASS] Failure prediction created: {mock_prediction.equipment_id}/{mock_prediction.component_id}")
        print("[PASS] Failure probability estimator: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Failure probability estimator: FAILED - {e}")
        return False

def test_risk_matrix_system():
    """Test risk matrix system"""
    print("\n" + "="*60)
    print("Testing Risk Matrix System")
    print("="*60)

    try:
        from src.forecasting.risk_matrix import RiskMatrixSystem, RiskMatrixCalculator
        from src.forecasting.failure_probability import FailurePrediction, SeverityLevel

        # Create risk matrix system
        risk_system = RiskMatrixSystem()

        # Test with mock failure predictions
        mock_predictions = [
            FailurePrediction(
                equipment_id='SMAP',
                component_id='power_system',
                failure_probability=0.7,
                time_to_failure=48,
                severity=SeverityLevel.CRITICAL
            ),
            FailurePrediction(
                equipment_id='MSL',
                component_id='mobility_front',
                failure_probability=0.3,
                time_to_failure=168,
                severity=SeverityLevel.MEDIUM
            )
        ]

        # Update risk assessments
        assessments = risk_system.update_risk_assessments(mock_predictions)

        print(f"[PASS] Risk assessments created: {len(assessments)} components")
        print("[PASS] Risk matrix system: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Risk matrix system: FAILED - {e}")
        return False

def test_scenario_analysis():
    """Test scenario analysis system"""
    print("\n" + "="*60)
    print("Testing Scenario Analysis")
    print("="*60)

    try:
        from src.forecasting.scenario_analysis import WhatIfAnalyzer, MaintenanceStrategy
        from src.forecasting.failure_probability import FailureProbabilityEstimator
        from src.forecasting.enhanced_forecaster import EnhancedForecaster

        # Create components (mock)
        failure_estimator = FailureProbabilityEstimator()
        forecaster = EnhancedForecaster(base_model="transformer")

        # Create analyzer
        analyzer = WhatIfAnalyzer(failure_estimator, forecaster)

        # Test scenario creation
        scenario = analyzer.create_scenario(
            scenario_id="test_scenario",
            name="Test Maintenance Strategy",
            description="Testing scenario creation",
            maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
            time_horizon=168,
            budget_limit=50000
        )

        print(f"[PASS] Scenario created: {scenario.name}")
        print("[PASS] Scenario analysis: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Scenario analysis: FAILED - {e}")
        return False

def test_dashboard_integration():
    """Test dashboard component integration"""
    print("\n" + "="*60)
    print("Testing Dashboard Integration")
    print("="*60)

    try:
        from src.dashboard.layouts.enhanced_forecasting import EnhancedForecastingDashboard
        from src.dashboard.components.risk_matrix_dashboard import RiskMatrixDashboardComponent

        # Test enhanced forecasting dashboard
        forecast_dashboard = EnhancedForecastingDashboard()
        layout = forecast_dashboard.create_layout()

        print("[PASS] Enhanced forecasting dashboard layout created")

        # Test risk matrix dashboard component
        risk_component = RiskMatrixDashboardComponent()
        risk_layout = risk_component.create_full_layout()

        print("[PASS] Risk matrix dashboard component created")
        print("[PASS] Dashboard integration: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Dashboard integration: FAILED - {e}")
        return False

def test_data_pipeline_compatibility():
    """Test compatibility with existing data pipeline"""
    print("\n" + "="*60)
    print("Testing Data Pipeline Compatibility")
    print("="*60)

    try:
        # Test if we can import existing components
        from src.dashboard.app import real_time_data
        from src.dashboard.utils import DataManager

        # Test data manager compatibility
        data_manager = DataManager()

        print("[PASS] Existing data pipeline components accessible")
        print("[PASS] Data pipeline compatibility: PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Data pipeline compatibility: FAILED - {e}")
        return False

def main():
    """Run all integration tests"""
    print("\n" + "="*50)
    print("ENHANCED FORECASTING INTEGRATION TEST")
    print("="*50)

    tests = [
        test_configuration_loading,
        test_enhanced_forecaster,
        test_failure_probability_estimator,
        test_risk_matrix_system,
        test_scenario_analysis,
        test_dashboard_integration,
        test_data_pipeline_compatibility
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[CRASH] Test {test.__name__} crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! System ready for user testing.")
        print("\nNext Steps:")
        print("1. Run: python launch_real_data_dashboard.py")
        print("2. Test enhanced forecasting pages")
        print("3. Verify real-time data updates")
        print("4. Test risk matrix visualization")
    else:
        print(f"\n[WARNING] {total-passed} tests failed. Fix issues before proceeding.")
        print("\nPriority fixes needed for failed components.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)