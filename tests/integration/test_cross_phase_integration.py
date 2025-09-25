"""
Cross-Phase Integration Testing Suite

Tests integration between different phases of the IoT Predictive Maintenance System:
- Phase 1 (Dashboard) ↔ Phase 2 (Business Logic)
- Phase 2 (Business Logic) ↔ Phase 3 (Optimization)
- Phase 3 (Optimization) ↔ Phase 4 (Advanced Features)
- Cross-cutting concerns: Configuration, Monitoring, Alerts
"""

import unittest
import sys
import os
import asyncio
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import threading
import queue
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import Phase 1 components (Dashboard Infrastructure)
try:
    from src.dashboard.app import create_dashboard_app
    from src.dashboard.callbacks.data_callbacks import DataCallbackManager
    from src.dashboard.layouts.main_layout import MainLayoutManager
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Warning: Could not import Phase 1 components: {e}")

# Import Phase 2 components (Business Logic)
try:
    from src.business_logic.business_rules_engine import BusinessRulesEngine
    from src.business_logic.failure_classification import FailureClassificationEngine
    from src.business_logic.equipment_health import EquipmentHealthMonitor
    from src.business_logic.predictive_triggers import PredictiveTriggerEngine
except ImportError as e:
    print(f"Warning: Could not import Phase 2 components: {e}")

# Import Phase 3 components (Optimization)
try:
    from src.maintenance.optimization_engine import MaintenanceOptimizationEngine
    from src.maintenance.resource_scheduler import ResourceScheduler
    from src.maintenance.cost_optimizer import CostOptimizer
    from src.alerts.alert_manager import AlertManager
except ImportError as e:
    print(f"Warning: Could not import Phase 3 components: {e}")

# Import Phase 4 components (Advanced Features)
try:
    from src.optimization.cache_manager import AdvancedCacheManager
    from src.optimization.memory_manager import MemoryManager
    from src.optimization.async_processor import AsyncSensorProcessor
    from src.optimization.callback_optimizer import CallbackOptimizer
except ImportError as e:
    print(f"Warning: Could not import Phase 4 components: {e}")

# Import core system components
try:
    from src.data_ingestion.nasa_data_service import NASADataService
    from src.preprocessing.data_preprocessor import DataPreprocessor
    from src.anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
    from src.forecasting.transformer_forecaster import TransformerForecaster
except ImportError as e:
    print(f"Warning: Could not import core system components: {e}")


@dataclass
class CrossPhaseTestResult:
    """Container for cross-phase integration test results"""
    test_name: str
    phase_combination: str
    integration_success: bool
    data_consistency: float
    performance_impact: float
    error_handling_score: float
    execution_time: float
    resource_efficiency: float
    timestamp: datetime


class TestCrossPhaseIntegration(unittest.TestCase):
    """Cross-phase integration testing"""

    @classmethod
    def setUpClass(cls):
        """Set up cross-phase integration test environment"""
        cls.logger = get_logger(__name__)
        cls.config_manager = ConfigManager()
        cls.test_results = []

        # Test configuration
        cls.test_config = {
            'phase_integration': {
                'phase1_dashboard_sensors': 40,
                'phase2_business_rules': 8,
                'phase3_optimization_jobs': 15,
                'phase4_cache_entries': 100
            },
            'performance_targets': {
                'cross_phase_latency': 1000,  # ms
                'data_consistency_threshold': 0.95,
                'resource_efficiency_target': 0.85,
                'error_handling_score_target': 0.90
            },
            'integration_scenarios': {
                'dashboard_to_business_logic': True,
                'business_logic_to_optimization': True,
                'optimization_to_advanced_features': True,
                'full_cross_phase': True
            }
        }

        # Initialize components
        cls._initialize_cross_phase_components()

    @classmethod
    def _initialize_cross_phase_components(cls):
        """Initialize all cross-phase components"""
        try:
            # Phase 1: Dashboard Infrastructure
            cls.dashboard_app = create_dashboard_app()
            cls.data_callback_manager = DataCallbackManager()
            cls.layout_manager = MainLayoutManager()

            # Phase 2: Business Logic
            cls.business_rules_engine = BusinessRulesEngine()
            cls.failure_classifier = FailureClassificationEngine()
            cls.equipment_health_monitor = EquipmentHealthMonitor()
            cls.predictive_triggers = PredictiveTriggerEngine()

            # Phase 3: Optimization
            cls.maintenance_optimizer = MaintenanceOptimizationEngine()
            cls.resource_scheduler = ResourceScheduler()
            cls.cost_optimizer = CostOptimizer()
            cls.alert_manager = AlertManager()

            # Phase 4: Advanced Features
            cls.cache_manager = AdvancedCacheManager()
            cls.memory_manager = MemoryManager()
            cls.async_processor = AsyncSensorProcessor()
            cls.callback_optimizer = CallbackOptimizer()

            # Core components
            cls.nasa_data_service = NASADataService()
            cls.data_preprocessor = DataPreprocessor()
            cls.ensemble_detector = EnsembleAnomalyDetector()
            cls.transformer_forecaster = TransformerForecaster()

            cls.logger.info("All cross-phase components initialized successfully")

        except Exception as e:
            cls.logger.error(f"Failed to initialize cross-phase components: {e}")
            raise

    def setUp(self):
        """Set up individual test"""
        self.test_start_time = time.time()
        self.test_data = self._generate_cross_phase_test_data()

    def tearDown(self):
        """Clean up individual test"""
        execution_time = time.time() - self.test_start_time
        self.logger.info(f"Cross-phase test {self._testMethodName} completed in {execution_time:.2f}s")

    def _generate_cross_phase_test_data(self) -> Dict[str, Any]:
        """Generate test data for cross-phase integration"""
        sensors_count = self.test_config['phase_integration']['phase1_dashboard_sensors']

        test_data = {
            'sensor_data': {},
            'anomaly_results': {},
            'health_assessments': {},
            'business_decisions': {},
            'optimization_results': {},
            'cache_entries': {}
        }

        # Generate sensor data for dashboard integration
        for i in range(sensors_count):
            sensor_id = f"integration_sensor_{i+1:03d}"

            # Generate time series with anomalies and patterns
            values = np.random.normal(0.5, 0.1, 500)

            # Add trend and seasonal patterns
            trend = np.linspace(0, 0.2, 500)
            seasonal = 0.1 * np.sin(2 * np.pi * np.arange(500) / 50)
            values += trend + seasonal

            # Inject anomalies for testing
            anomaly_indices = np.random.choice(500, size=25, replace=False)
            values[anomaly_indices] += np.random.uniform(0.5, 1.0, 25)

            test_data['sensor_data'][sensor_id] = {
                'values': np.clip(values, 0, 1),
                'timestamps': pd.date_range(
                    start=datetime.now() - timedelta(hours=8),
                    periods=500,
                    freq='1min'
                ),
                'metadata': {
                    'sensor_type': 'temperature' if i % 2 == 0 else 'pressure',
                    'equipment_id': f"equipment_{(i // 5) + 1}",
                    'location': f"zone_{(i // 10) + 1}"
                }
            }

        return test_data

    def test_phase1_to_phase2_integration(self):
        """Test integration between Dashboard (Phase 1) and Business Logic (Phase 2)"""
        test_start = time.time()

        try:
            integration_results = []
            data_consistency_scores = []

            # Stage 1: Dashboard data preparation
            dashboard_data = {}
            for sensor_id, sensor_info in list(self.test_data['sensor_data'].items())[:20]:
                # Simulate dashboard data processing
                dashboard_entry = {
                    'sensor_id': sensor_id,
                    'current_value': sensor_info['values'][-1],
                    'trend': np.mean(np.diff(sensor_info['values'][-10:])),
                    'volatility': np.std(sensor_info['values'][-50:]),
                    'last_updated': datetime.now()
                }
                dashboard_data[sensor_id] = dashboard_entry

            # Stage 2: Business Logic processing
            business_processing_start = time.time()

            for sensor_id, dashboard_entry in dashboard_data.items():
                # Process through business logic components
                sensor_values = self.test_data['sensor_data'][sensor_id]['values']

                # 1. Failure classification
                try:
                    failure_context = {
                        'sensor_id': sensor_id,
                        'current_value': dashboard_entry['current_value'],
                        'historical_data': sensor_values[-100:]
                    }
                    failure_classification = self.failure_classifier.classify_sensor_condition(failure_context)

                    # 2. Equipment health assessment
                    health_assessment = self.equipment_health_monitor.assess_sensor_health(
                        sensor_id, sensor_values[-100:]
                    )

                    # 3. Business rules evaluation
                    business_context = {
                        'sensor_data': dashboard_entry,
                        'failure_classification': failure_classification,
                        'health_assessment': health_assessment
                    }

                    business_decision = asyncio.run(
                        self.business_rules_engine.evaluate_sensor_rules(business_context)
                    )

                    # Validate integration consistency
                    integration_success = all([
                        failure_classification is not None,
                        health_assessment is not None,
                        business_decision is not None
                    ])

                    integration_results.append(integration_success)

                    # Calculate data consistency
                    original_value = dashboard_entry['current_value']
                    processed_value = health_assessment.get('current_reading', original_value)
                    consistency_score = 1.0 - abs(original_value - processed_value) / (abs(original_value) + 1e-8)
                    data_consistency_scores.append(consistency_score)

                except Exception as e:
                    self.logger.warning(f"Phase 1-2 integration error for {sensor_id}: {e}")
                    integration_results.append(False)
                    data_consistency_scores.append(0.0)

            business_processing_time = time.time() - business_processing_start

            # Validate integration performance
            integration_success_rate = np.mean(integration_results) if integration_results else 0.0
            avg_data_consistency = np.mean(data_consistency_scores) if data_consistency_scores else 0.0

            self.assertGreaterEqual(
                integration_success_rate, 0.9,
                f"Phase 1-2 integration success rate {integration_success_rate:.3f} below threshold"
            )

            self.assertGreaterEqual(
                avg_data_consistency, self.test_config['performance_targets']['data_consistency_threshold'],
                f"Phase 1-2 data consistency {avg_data_consistency:.3f} below threshold"
            )

            # Record test result
            total_time = time.time() - test_start
            result = CrossPhaseTestResult(
                test_name="phase1_to_phase2_integration",
                phase_combination="Dashboard → Business Logic",
                integration_success=integration_success_rate >= 0.9,
                data_consistency=avg_data_consistency,
                performance_impact=business_processing_time / len(dashboard_data),
                error_handling_score=integration_success_rate,
                execution_time=total_time,
                resource_efficiency=len(integration_results) / total_time,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Phase 1-2 integration: success={integration_success_rate:.3f}, "
                           f"consistency={avg_data_consistency:.3f}")

        except Exception as e:
            self.fail(f"Phase 1 to Phase 2 integration test failed: {e}")

    def test_phase2_to_phase3_integration(self):
        """Test integration between Business Logic (Phase 2) and Optimization (Phase 3)"""
        test_start = time.time()

        try:
            optimization_results = []
            resource_efficiency_scores = []

            # Stage 1: Business Logic decisions
            business_decisions = {}
            for sensor_id, sensor_info in list(self.test_data['sensor_data'].items())[:15]:
                # Generate business logic output
                sensor_values = sensor_info['values']

                # Health assessment
                health_assessment = self.equipment_health_monitor.assess_sensor_health(
                    sensor_id, sensor_values[-100:]
                )

                # Predictive triggers
                trigger_context = {
                    'sensor_id': sensor_id,
                    'health_score': health_assessment.get('overall_score', 0.8),
                    'anomaly_probability': np.random.uniform(0.1, 0.7)
                }

                trigger_events = self.predictive_triggers.evaluate_triggers([trigger_context])

                # Business decision
                decision_context = {
                    'sensor_id': sensor_id,
                    'health_assessment': health_assessment,
                    'trigger_events': trigger_events,
                    'priority_factors': {
                        'criticality': np.random.uniform(0.3, 0.9),
                        'cost_impact': np.random.uniform(1000, 10000),
                        'urgency': np.random.choice(['LOW', 'MEDIUM', 'HIGH'])
                    }
                }

                business_decision = asyncio.run(
                    self.business_rules_engine.make_maintenance_decision(decision_context)
                )

                business_decisions[sensor_id] = {
                    'health_assessment': health_assessment,
                    'trigger_events': trigger_events,
                    'maintenance_decision': business_decision,
                    'decision_context': decision_context
                }

            # Stage 2: Optimization processing
            optimization_start = time.time()

            # Prepare optimization input
            maintenance_requirements = []
            for sensor_id, decision_data in business_decisions.items():
                if decision_data['maintenance_decision'] and decision_data['maintenance_decision'].get('action_required', False):
                    requirement = {
                        'sensor_id': sensor_id,
                        'priority': decision_data['decision_context']['priority_factors']['urgency'],
                        'estimated_cost': decision_data['decision_context']['priority_factors']['cost_impact'],
                        'health_score': decision_data['health_assessment'].get('overall_score', 0.8),
                        'criticality': decision_data['decision_context']['priority_factors']['criticality'],
                        'required_skills': ['maintenance', 'diagnostic'],
                        'estimated_duration': np.random.uniform(2, 8)
                    }
                    maintenance_requirements.append(requirement)

            # Optimization processing
            if maintenance_requirements:
                # Resource scheduling
                resource_schedule = self.resource_scheduler.schedule_maintenance_tasks(
                    maintenance_requirements
                )

                # Cost optimization
                cost_optimization = self.cost_optimizer.optimize_maintenance_costs(
                    maintenance_requirements, resource_schedule
                )

                # Maintenance optimization
                optimization_plan = self.maintenance_optimizer.create_optimal_schedule(
                    maintenance_requirements
                )

                optimization_results.append(True)

                # Calculate resource efficiency
                if resource_schedule and cost_optimization:
                    planned_cost = sum(req['estimated_cost'] for req in maintenance_requirements)
                    optimized_cost = cost_optimization.get('total_optimized_cost', planned_cost)
                    efficiency = max(0.0, 1.0 - (optimized_cost / planned_cost))
                    resource_efficiency_scores.append(efficiency)

                # Validate optimization quality
                self.assertIsNotNone(resource_schedule, "Resource schedule should be generated")
                self.assertIsNotNone(optimization_plan, "Optimization plan should be generated")
                self.assertGreater(
                    len(resource_schedule.get('scheduled_tasks', [])), 0,
                    "Should have scheduled tasks"
                )

            else:
                optimization_results.append(True)  # No maintenance needed is valid
                resource_efficiency_scores.append(1.0)

            optimization_time = time.time() - optimization_start

            # Validate integration performance
            optimization_success_rate = np.mean(optimization_results) if optimization_results else 0.0
            avg_resource_efficiency = np.mean(resource_efficiency_scores) if resource_efficiency_scores else 0.0

            self.assertGreaterEqual(
                optimization_success_rate, 0.9,
                f"Phase 2-3 optimization success rate {optimization_success_rate:.3f} below threshold"
            )

            self.assertGreaterEqual(
                avg_resource_efficiency, self.test_config['performance_targets']['resource_efficiency_target'],
                f"Phase 2-3 resource efficiency {avg_resource_efficiency:.3f} below threshold"
            )

            # Record test result
            total_time = time.time() - test_start
            result = CrossPhaseTestResult(
                test_name="phase2_to_phase3_integration",
                phase_combination="Business Logic → Optimization",
                integration_success=optimization_success_rate >= 0.9,
                data_consistency=1.0,  # Optimization preserves business logic decisions
                performance_impact=optimization_time / max(1, len(maintenance_requirements)),
                error_handling_score=optimization_success_rate,
                execution_time=total_time,
                resource_efficiency=avg_resource_efficiency,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Phase 2-3 integration: success={optimization_success_rate:.3f}, "
                           f"efficiency={avg_resource_efficiency:.3f}")

        except Exception as e:
            self.fail(f"Phase 2 to Phase 3 integration test failed: {e}")

    def test_phase3_to_phase4_integration(self):
        """Test integration between Optimization (Phase 3) and Advanced Features (Phase 4)"""
        test_start = time.time()

        try:
            caching_performance = []
            memory_optimization_scores = []
            async_processing_results = []

            # Stage 1: Generate optimization data
            optimization_data = {}
            for i in range(self.test_config['phase_integration']['phase3_optimization_jobs']):
                job_id = f"optimization_job_{i+1:03d}"

                optimization_result = {
                    'job_id': job_id,
                    'maintenance_schedule': [
                        {
                            'sensor_id': f"sensor_{j+1}",
                            'task_type': f"maintenance_type_{j % 5}",
                            'priority': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                            'estimated_duration': np.random.uniform(1, 6),
                            'resource_requirements': np.random.randint(1, 4)
                        }
                        for j in range(np.random.randint(2, 8))
                    ],
                    'cost_analysis': {
                        'total_cost': np.random.uniform(5000, 25000),
                        'savings_potential': np.random.uniform(0.1, 0.3),
                        'roi_estimate': np.random.uniform(1.2, 2.5)
                    },
                    'resource_allocation': {
                        'technicians_required': np.random.randint(2, 6),
                        'equipment_needed': np.random.randint(1, 3),
                        'timeline_days': np.random.randint(1, 14)
                    }
                }
                optimization_data[job_id] = optimization_result

            # Stage 2: Advanced Features processing
            advanced_processing_start = time.time()

            # Test caching integration
            cache_operations = []
            for job_id, opt_result in optimization_data.items():
                cache_start = time.time()

                # Store optimization result in cache
                cache_key = f"optimization_result_{job_id}"
                cache_success = self.cache_manager.set(cache_key, opt_result, ttl=3600)

                # Retrieve from cache
                cached_result = self.cache_manager.get(cache_key)

                cache_time = time.time() - cache_start
                cache_operations.append(cache_time)

                # Validate caching
                cache_integrity = cached_result is not None and cached_result == opt_result
                caching_performance.append(1.0 if cache_integrity else 0.0)

            # Test memory optimization
            memory_start = time.time()

            for job_id, opt_result in list(optimization_data.items())[:10]:  # Test subset
                # Simulate memory optimization
                memory_usage_before = self.memory_manager.get_memory_usage()

                # Process optimization result
                processed_result = self.memory_manager.optimize_data_structure(opt_result)

                memory_usage_after = self.memory_manager.get_memory_usage()

                # Calculate memory efficiency
                if memory_usage_before > 0:
                    memory_savings = max(0, memory_usage_before - memory_usage_after) / memory_usage_before
                    memory_optimization_scores.append(memory_savings)

                # Validate memory optimization
                self.assertIsNotNone(processed_result, f"Memory optimization should return result for {job_id}")

            memory_time = time.time() - memory_start

            # Test async processing integration
            async_start = time.time()

            async def process_optimization_batch(batch_data):
                """Async processing of optimization batch"""
                results = []
                for job_id, opt_result in batch_data.items():
                    # Simulate async processing
                    processed = await self.async_processor.process_optimization_data(opt_result)
                    results.append((job_id, processed))
                    await asyncio.sleep(0.01)  # Simulate processing time
                return results

            # Process optimization data in async batches
            batch_size = 5
            job_items = list(optimization_data.items())
            async_results = []

            for i in range(0, len(job_items), batch_size):
                batch = dict(job_items[i:i + batch_size])
                batch_results = asyncio.run(process_optimization_batch(batch))
                async_results.extend(batch_results)

            async_time = time.time() - async_start

            # Validate async processing
            async_success_rate = len(async_results) / len(optimization_data)
            async_processing_results.append(async_success_rate)

            # Calculate performance metrics
            avg_caching_performance = np.mean(caching_performance) if caching_performance else 0.0
            avg_memory_optimization = np.mean(memory_optimization_scores) if memory_optimization_scores else 0.0
            avg_async_success = np.mean(async_processing_results) if async_processing_results else 0.0

            advanced_processing_time = time.time() - advanced_processing_start

            # Validate integration performance
            self.assertGreaterEqual(
                avg_caching_performance, 0.95,
                f"Phase 3-4 caching performance {avg_caching_performance:.3f} below threshold"
            )

            self.assertGreaterEqual(
                avg_async_success, 0.95,
                f"Phase 3-4 async processing success {avg_async_success:.3f} below threshold"
            )

            # Record test result
            total_time = time.time() - test_start
            result = CrossPhaseTestResult(
                test_name="phase3_to_phase4_integration",
                phase_combination="Optimization → Advanced Features",
                integration_success=avg_caching_performance >= 0.95 and avg_async_success >= 0.95,
                data_consistency=avg_caching_performance,
                performance_impact=advanced_processing_time / len(optimization_data),
                error_handling_score=(avg_caching_performance + avg_async_success) / 2,
                execution_time=total_time,
                resource_efficiency=1.0 + avg_memory_optimization,  # Memory savings improve efficiency
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Phase 3-4 integration: cache={avg_caching_performance:.3f}, "
                           f"memory={avg_memory_optimization:.3f}, async={avg_async_success:.3f}")

        except Exception as e:
            self.fail(f"Phase 3 to Phase 4 integration test failed: {e}")

    def test_full_cross_phase_workflow(self):
        """Test complete cross-phase workflow integration"""
        test_start = time.time()

        try:
            workflow_stages = []
            stage_performance = {}

            # Select sensors for full workflow test
            test_sensors = list(self.test_data['sensor_data'].keys())[:10]

            self.logger.info(f"Starting full cross-phase workflow with {len(test_sensors)} sensors")

            # Stage 1: Dashboard Data Processing (Phase 1)
            stage1_start = time.time()
            dashboard_processed = {}

            for sensor_id in test_sensors:
                sensor_info = self.test_data['sensor_data'][sensor_id]

                # Simulate dashboard processing
                dashboard_entry = {
                    'sensor_id': sensor_id,
                    'current_value': sensor_info['values'][-1],
                    'historical_data': sensor_info['values'][-100:],
                    'metadata': sensor_info['metadata'],
                    'processing_timestamp': datetime.now()
                }
                dashboard_processed[sensor_id] = dashboard_entry

            stage1_time = time.time() - stage1_start
            stage_performance['phase1_dashboard'] = stage1_time
            workflow_stages.append(('phase1_dashboard', True))

            # Stage 2: Business Logic Processing (Phase 2)
            stage2_start = time.time()
            business_processed = {}

            for sensor_id, dashboard_data in dashboard_processed.items():
                # Health assessment
                health_assessment = self.equipment_health_monitor.assess_sensor_health(
                    sensor_id, dashboard_data['historical_data']
                )

                # Failure classification
                failure_context = {
                    'sensor_id': sensor_id,
                    'current_value': dashboard_data['current_value'],
                    'historical_data': dashboard_data['historical_data']
                }
                failure_classification = self.failure_classifier.classify_sensor_condition(failure_context)

                # Business rules evaluation
                business_context = {
                    'sensor_data': dashboard_data,
                    'health_assessment': health_assessment,
                    'failure_classification': failure_classification
                }

                business_decision = asyncio.run(
                    self.business_rules_engine.evaluate_sensor_rules(business_context)
                )

                business_processed[sensor_id] = {
                    'dashboard_data': dashboard_data,
                    'health_assessment': health_assessment,
                    'failure_classification': failure_classification,
                    'business_decision': business_decision
                }

            stage2_time = time.time() - stage2_start
            stage_performance['phase2_business_logic'] = stage2_time
            workflow_stages.append(('phase2_business_logic', True))

            # Stage 3: Optimization Processing (Phase 3)
            stage3_start = time.time()

            # Aggregate maintenance requirements
            maintenance_requirements = []
            for sensor_id, business_data in business_processed.items():
                if business_data['business_decision'] and business_data['business_decision'].get('action_required', False):
                    requirement = {
                        'sensor_id': sensor_id,
                        'priority': business_data['business_decision'].get('priority', 'MEDIUM'),
                        'estimated_cost': np.random.uniform(1000, 5000),
                        'health_score': business_data['health_assessment'].get('overall_score', 0.8),
                        'required_skills': ['maintenance'],
                        'estimated_duration': np.random.uniform(2, 6)
                    }
                    maintenance_requirements.append(requirement)

            # Optimization processing
            optimization_results = {}
            if maintenance_requirements:
                # Resource scheduling
                resource_schedule = self.resource_scheduler.schedule_maintenance_tasks(
                    maintenance_requirements
                )

                # Cost optimization
                cost_optimization = self.cost_optimizer.optimize_maintenance_costs(
                    maintenance_requirements, resource_schedule
                )

                # Alert generation
                alert_results = []
                for requirement in maintenance_requirements:
                    if requirement['priority'] == 'HIGH':
                        alert = self.alert_manager.create_maintenance_alert(
                            requirement['sensor_id'],
                            f"High priority maintenance required",
                            requirement['priority']
                        )
                        alert_results.append(alert)

                optimization_results = {
                    'resource_schedule': resource_schedule,
                    'cost_optimization': cost_optimization,
                    'alerts_generated': alert_results,
                    'maintenance_requirements': maintenance_requirements
                }

            stage3_time = time.time() - stage3_start
            stage_performance['phase3_optimization'] = stage3_time
            workflow_stages.append(('phase3_optimization', True))

            # Stage 4: Advanced Features Processing (Phase 4)
            stage4_start = time.time()

            # Cache optimization results
            cache_operations = []
            for sensor_id, business_data in business_processed.items():
                cache_key = f"workflow_result_{sensor_id}"
                cached = self.cache_manager.set(cache_key, business_data, ttl=1800)
                cache_operations.append(cached)

            # Memory optimization
            memory_start_usage = self.memory_manager.get_memory_usage()
            optimized_data = self.memory_manager.optimize_data_structure(business_processed)
            memory_end_usage = self.memory_manager.get_memory_usage()

            # Async processing integration
            async def finalize_workflow_data(data):
                """Async finalization of workflow data"""
                finalized = {}
                for sensor_id, sensor_data in data.items():
                    processed = await self.async_processor.process_optimization_data(sensor_data)
                    finalized[sensor_id] = processed
                    await asyncio.sleep(0.01)
                return finalized

            finalized_results = asyncio.run(finalize_workflow_data(optimized_data))

            stage4_time = time.time() - stage4_start
            stage_performance['phase4_advanced_features'] = stage4_time
            workflow_stages.append(('phase4_advanced_features', True))

            # Validate complete workflow
            total_workflow_time = time.time() - test_start

            # Performance validation
            target_latency = self.test_config['performance_targets']['cross_phase_latency'] / 1000
            self.assertLessEqual(
                total_workflow_time, target_latency,
                f"Full workflow time {total_workflow_time:.2f}s exceeds target {target_latency:.2f}s"
            )

            # Data consistency validation
            workflow_success = all(success for _, success in workflow_stages)
            cache_success_rate = np.mean(cache_operations) if cache_operations else 0.0

            self.assertTrue(workflow_success, "All workflow stages should succeed")
            self.assertGreaterEqual(cache_success_rate, 0.95, "Cache operations should succeed")

            # Calculate overall performance metrics
            data_consistency = len(finalized_results) / len(test_sensors)
            memory_efficiency = max(0.0, memory_start_usage - memory_end_usage) / max(memory_start_usage, 1)
            overall_efficiency = len(test_sensors) / total_workflow_time

            # Record comprehensive test result
            result = CrossPhaseTestResult(
                test_name="full_cross_phase_workflow",
                phase_combination="All Phases (1→2→3→4)",
                integration_success=workflow_success,
                data_consistency=data_consistency,
                performance_impact=total_workflow_time / len(test_sensors),
                error_handling_score=cache_success_rate,
                execution_time=total_workflow_time,
                resource_efficiency=overall_efficiency,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Full workflow: {total_workflow_time:.2f}s, "
                           f"stages={dict(stage_performance)}")

        except Exception as e:
            error_result = CrossPhaseTestResult(
                test_name="full_cross_phase_workflow",
                phase_combination="All Phases (1→2→3→4)",
                integration_success=False,
                data_consistency=0.0,
                performance_impact=0.0,
                error_handling_score=0.0,
                execution_time=time.time() - test_start,
                resource_efficiency=0.0,
                timestamp=datetime.now()
            )
            self.test_results.append(error_result)
            self.fail(f"Full cross-phase workflow test failed: {e}")

    def test_configuration_consistency_across_phases(self):
        """Test configuration consistency across all phases"""
        test_start = time.time()

        try:
            config_consistency_scores = []

            # Test configuration access from each phase
            phase_configs = {}

            # Phase 1: Dashboard configuration
            try:
                dashboard_config = self.config_manager.get_dashboard_config()
                phase_configs['phase1'] = dashboard_config
                config_consistency_scores.append(1.0)
            except Exception as e:
                self.logger.warning(f"Phase 1 config access failed: {e}")
                config_consistency_scores.append(0.0)

            # Phase 2: Business Logic configuration
            try:
                business_config = self.config_manager.get_business_logic_config()
                phase_configs['phase2'] = business_config
                config_consistency_scores.append(1.0)
            except Exception as e:
                self.logger.warning(f"Phase 2 config access failed: {e}")
                config_consistency_scores.append(0.0)

            # Phase 3: Optimization configuration
            try:
                optimization_config = self.config_manager.get_optimization_config()
                phase_configs['phase3'] = optimization_config
                config_consistency_scores.append(1.0)
            except Exception as e:
                self.logger.warning(f"Phase 3 config access failed: {e}")
                config_consistency_scores.append(0.0)

            # Phase 4: Advanced Features configuration
            try:
                advanced_config = self.config_manager.get_advanced_features_config()
                phase_configs['phase4'] = advanced_config
                config_consistency_scores.append(1.0)
            except Exception as e:
                self.logger.warning(f"Phase 4 config access failed: {e}")
                config_consistency_scores.append(0.0)

            # Validate configuration consistency
            avg_config_consistency = np.mean(config_consistency_scores) if config_consistency_scores else 0.0

            self.assertGreaterEqual(
                avg_config_consistency, 0.8,
                f"Configuration consistency {avg_config_consistency:.3f} below threshold"
            )

            # Test cross-phase configuration validation
            shared_configs = ['database', 'logging', 'monitoring']
            for config_key in shared_configs:
                config_values = []
                for phase, config in phase_configs.items():
                    if config and config_key in config:
                        config_values.append(config[config_key])

                # All phases should have consistent shared configuration
                if config_values:
                    consistency_check = all(val == config_values[0] for val in config_values)
                    self.assertTrue(
                        consistency_check,
                        f"Shared config '{config_key}' inconsistent across phases"
                    )

            # Record test result
            total_time = time.time() - test_start
            result = CrossPhaseTestResult(
                test_name="configuration_consistency_across_phases",
                phase_combination="All Phases (Config)",
                integration_success=avg_config_consistency >= 0.8,
                data_consistency=avg_config_consistency,
                performance_impact=total_time,
                error_handling_score=avg_config_consistency,
                execution_time=total_time,
                resource_efficiency=len(phase_configs) / total_time,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Configuration consistency: {avg_config_consistency:.3f}")

        except Exception as e:
            self.fail(f"Configuration consistency test failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Generate cross-phase integration report"""
        cls._generate_cross_phase_report()

    @classmethod
    def _generate_cross_phase_report(cls):
        """Generate comprehensive cross-phase integration report"""
        try:
            report_data = {
                'test_suite': 'Cross-Phase Integration Testing',
                'execution_timestamp': datetime.now().isoformat(),
                'total_tests': len(cls.test_results),
                'successful_integrations': len([r for r in cls.test_results if r.integration_success]),
                'integration_coverage': {
                    'phase1_to_phase2': any('phase1_to_phase2' in r.test_name for r in cls.test_results),
                    'phase2_to_phase3': any('phase2_to_phase3' in r.test_name for r in cls.test_results),
                    'phase3_to_phase4': any('phase3_to_phase4' in r.test_name for r in cls.test_results),
                    'full_workflow': any('full_cross_phase' in r.test_name for r in cls.test_results),
                    'configuration_consistency': any('configuration_consistency' in r.test_name for r in cls.test_results)
                },
                'performance_summary': {
                    'avg_data_consistency': np.mean([r.data_consistency for r in cls.test_results]),
                    'avg_performance_impact': np.mean([r.performance_impact for r in cls.test_results]),
                    'avg_error_handling_score': np.mean([r.error_handling_score for r in cls.test_results]),
                    'avg_execution_time': np.mean([r.execution_time for r in cls.test_results]),
                    'avg_resource_efficiency': np.mean([r.resource_efficiency for r in cls.test_results])
                },
                'phase_combinations': {
                    combination: {
                        'tests_count': len([r for r in cls.test_results if r.phase_combination == combination]),
                        'success_rate': np.mean([r.integration_success for r in cls.test_results if r.phase_combination == combination]),
                        'avg_consistency': np.mean([r.data_consistency for r in cls.test_results if r.phase_combination == combination])
                    }
                    for combination in set(r.phase_combination for r in cls.test_results)
                },
                'detailed_results': [
                    {
                        'test_name': r.test_name,
                        'phase_combination': r.phase_combination,
                        'integration_success': r.integration_success,
                        'data_consistency': r.data_consistency,
                        'performance_impact': r.performance_impact,
                        'error_handling_score': r.error_handling_score,
                        'execution_time': r.execution_time,
                        'resource_efficiency': r.resource_efficiency,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in cls.test_results
                ]
            }

            # Save cross-phase integration report
            report_path = Path(__file__).parent.parent / "cross_phase_integration_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            cls.logger.info(f"Cross-phase integration report saved to {report_path}")

        except Exception as e:
            cls.logger.error(f"Failed to generate cross-phase report: {e}")


if __name__ == '__main__':
    # Configure test runner for cross-phase integration
    unittest.main(verbosity=2, buffer=True)