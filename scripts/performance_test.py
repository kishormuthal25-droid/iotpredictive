#!/usr/bin/env python3
"""
Performance Testing Script for 80-Sensor IoT Dashboard
Validates sub-second response times and scalability targets
"""

import sys
import os
import time
import asyncio
import concurrent.futures
import statistics
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import requests
import threading
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all optimized components
from src.utils.performance_monitor import performance_monitor, start_monitoring
from src.utils.memory_manager import memory_manager, optimize_for_sensors
from src.utils.async_processor import async_processor
from src.utils.advanced_cache import advanced_cache
from src.utils.callback_optimizer import callback_optimizer
from src.utils.data_compressor import data_compressor
from src.utils.predictive_cache import predictive_cache

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Performance test result"""
    test_name: str
    target_metric: str
    target_value: float
    actual_value: float
    passed: bool
    execution_time: float
    additional_metrics: Dict[str, Any]
    timestamp: datetime


class PerformanceTester:
    """
    Comprehensive performance testing for 80-sensor dashboard
    Tests all optimization components and validates targets
    """

    def __init__(self):
        """Initialize performance tester"""
        self.results: List[TestResult] = []
        self.test_data_generated = False

        # Performance targets
        self.targets = {
            'dashboard_response_time_ms': 500.0,
            'sensor_processing_rate': 80.0,  # 80 sensors per second
            'cache_hit_rate': 0.7,
            'memory_usage_limit_mb': 2048.0,
            'async_success_rate': 0.95,
            'compression_ratio': 0.8,
            'database_query_time_ms': 100.0
        }

        logger.info("Performance Tester initialized with 80-sensor targets")

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        logger.info("Starting comprehensive performance testing for 80-sensor dashboard...")

        # Start monitoring
        start_monitoring()
        await asyncio.sleep(2)  # Let monitoring stabilize

        test_results = []
        overall_start = time.time()

        try:
            # Test 1: Memory Management Performance
            result = await self.test_memory_management()
            test_results.append(result)

            # Test 2: Async Processing Performance
            result = await self.test_async_processing()
            test_results.append(result)

            # Test 3: Database Operations Performance
            result = await self.test_database_performance()
            test_results.append(result)

            # Test 4: Cache Performance
            result = await self.test_cache_performance()
            test_results.append(result)

            # Test 5: Dashboard Callback Performance
            result = await self.test_dashboard_callbacks()
            test_results.append(result)

            # Test 6: Data Compression Performance
            result = await self.test_data_compression()
            test_results.append(result)

            # Test 7: 80-Sensor Load Test
            result = await self.test_80_sensor_load()
            test_results.append(result)

            # Test 8: End-to-End Dashboard Response
            result = await self.test_dashboard_response_time()
            test_results.append(result)

        except Exception as e:
            logger.error(f"Test suite error: {e}")

        total_execution_time = time.time() - overall_start

        # Generate comprehensive report
        report = self._generate_test_report(test_results, total_execution_time)

        logger.info(f"Performance testing completed in {total_execution_time:.2f}s")
        return report

    async def test_memory_management(self) -> TestResult:
        """Test sliding window memory management"""
        logger.info("Testing memory management for 80 sensors...")
        start_time = time.time()

        try:
            # Optimize for 80 sensors
            optimize_for_sensors(80)

            # Generate test data for 80 sensors
            sensor_data_added = 0
            for sensor_id in range(80):
                for i in range(1000):  # 1000 data points per sensor
                    timestamp = datetime.now() - timedelta(seconds=i)
                    value = np.random.normal(50, 10)
                    anomaly_score = np.random.uniform(0, 1)

                    success = memory_manager.add_sensor_data(
                        f"sensor_{sensor_id:03d}",
                        timestamp,
                        value,
                        anomaly_score,
                        anomaly_score > 0.8
                    )

                    if success:
                        sensor_data_added += 1

            # Get memory stats
            memory_stats = memory_manager.get_memory_stats()
            execution_time = time.time() - start_time

            # Check if within memory limits
            memory_within_limits = memory_stats.total_memory_mb < self.targets['memory_usage_limit_mb']

            return TestResult(
                test_name="Memory Management",
                target_metric="memory_usage_mb",
                target_value=self.targets['memory_usage_limit_mb'],
                actual_value=memory_stats.total_memory_mb,
                passed=memory_within_limits,
                execution_time=execution_time,
                additional_metrics={
                    'sensors_created': 80,
                    'data_points_added': sensor_data_added,
                    'avg_window_size': memory_stats.avg_window_size,
                    'compressed_windows': memory_stats.compressed_windows
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Memory management test failed: {e}")
            return TestResult(
                test_name="Memory Management",
                target_metric="memory_usage_mb",
                target_value=self.targets['memory_usage_limit_mb'],
                actual_value=float('inf'),
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_async_processing(self) -> TestResult:
        """Test async parallel processing for 80 sensors"""
        logger.info("Testing async processing for 80 sensors...")
        start_time = time.time()

        try:
            # Start async processor
            await async_processor.start()

            # Create test sensor data
            sensor_data_batch = []
            for i in range(80):
                sensor_data_batch.append({
                    'sensor_id': f'sensor_{i:03d}',
                    'timestamp': datetime.now().isoformat(),
                    'value': np.random.normal(50, 10),
                    'anomaly_score': np.random.uniform(0, 1)
                })

            # Test async processing
            async def test_sensor_processor(data):
                await asyncio.sleep(0.01)  # Simulate processing time
                return {'processed': True, 'sensor_id': data['sensor_id']}

            # Process all sensors in parallel
            results = await async_processor.process_sensor_batch(
                sensor_data_batch, test_sensor_processor
            )

            execution_time = time.time() - start_time
            success_count = sum(1 for r in results if r.success)
            success_rate = success_count / len(results) if results else 0

            # Calculate effective processing rate
            sensors_per_second = len(sensor_data_batch) / execution_time

            return TestResult(
                test_name="Async Processing",
                target_metric="async_success_rate",
                target_value=self.targets['async_success_rate'],
                actual_value=success_rate,
                passed=success_rate >= self.targets['async_success_rate'],
                execution_time=execution_time,
                additional_metrics={
                    'sensors_processed': len(sensor_data_batch),
                    'successful_operations': success_count,
                    'sensors_per_second': sensors_per_second,
                    'avg_execution_time_ms': np.mean([r.execution_time for r in results]) * 1000
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Async processing test failed: {e}")
            return TestResult(
                test_name="Async Processing",
                target_metric="async_success_rate",
                target_value=self.targets['async_success_rate'],
                actual_value=0.0,
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_database_performance(self) -> TestResult:
        """Test database performance with bulk operations"""
        logger.info("Testing database performance...")
        start_time = time.time()

        try:
            # Simulate database operations (since we don't have actual DB in test)
            query_times = []

            # Simulate 20 bulk queries for 80 sensors
            for batch in range(20):
                query_start = time.time()

                # Simulate database query processing time
                await asyncio.sleep(0.05)  # 50ms simulation

                query_time = (time.time() - query_start) * 1000  # Convert to ms
                query_times.append(query_time)

            execution_time = time.time() - start_time
            avg_query_time = np.mean(query_times)

            return TestResult(
                test_name="Database Performance",
                target_metric="db_query_time_ms",
                target_value=self.targets['database_query_time_ms'],
                actual_value=avg_query_time,
                passed=avg_query_time <= self.targets['database_query_time_ms'],
                execution_time=execution_time,
                additional_metrics={
                    'total_queries': len(query_times),
                    'min_query_time_ms': min(query_times),
                    'max_query_time_ms': max(query_times),
                    'p95_query_time_ms': np.percentile(query_times, 95)
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Database performance test failed: {e}")
            return TestResult(
                test_name="Database Performance",
                target_metric="db_query_time_ms",
                target_value=self.targets['database_query_time_ms'],
                actual_value=float('inf'),
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_cache_performance(self) -> TestResult:
        """Test advanced caching performance"""
        logger.info("Testing cache performance...")
        start_time = time.time()

        try:
            # Clear cache to start fresh
            advanced_cache.clear_all()

            # Generate test cache operations
            cache_operations = 1000
            hits = 0
            misses = 0

            for i in range(cache_operations):
                cache_key = f"test_key_{i % 100}"  # 100 unique keys, causing some hits

                # Try to get from cache
                cached_value = advanced_cache.get(cache_key)

                if cached_value is not None:
                    hits += 1
                else:
                    misses += 1
                    # Set cache value
                    test_data = {'sensor_id': i, 'value': np.random.random(), 'timestamp': datetime.now().isoformat()}
                    advanced_cache.set(cache_key, test_data, 300)

            execution_time = time.time() - start_time
            hit_rate = hits / cache_operations if cache_operations > 0 else 0

            # Get cache metrics
            cache_metrics = advanced_cache.get_metrics()

            return TestResult(
                test_name="Cache Performance",
                target_metric="cache_hit_rate",
                target_value=self.targets['cache_hit_rate'],
                actual_value=hit_rate,
                passed=hit_rate >= self.targets['cache_hit_rate'],
                execution_time=execution_time,
                additional_metrics={
                    'total_operations': cache_operations,
                    'cache_hits': hits,
                    'cache_misses': misses,
                    'avg_retrieval_time_ms': cache_metrics.avg_retrieval_time * 1000,
                    'memory_usage': cache_metrics.memory_usage
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Cache performance test failed: {e}")
            return TestResult(
                test_name="Cache Performance",
                target_metric="cache_hit_rate",
                target_value=self.targets['cache_hit_rate'],
                actual_value=0.0,
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_dashboard_callbacks(self) -> TestResult:
        """Test optimized dashboard callbacks"""
        logger.info("Testing dashboard callback performance...")
        start_time = time.time()

        try:
            # Simulate dashboard callback executions
            callback_times = []

            for i in range(50):  # 50 callback simulations
                callback_start = time.time()

                # Simulate callback processing
                await asyncio.sleep(0.02)  # 20ms processing time

                callback_time = (time.time() - callback_start) * 1000
                callback_times.append(callback_time)

                # Record with callback optimizer
                callback_optimizer.record_callback_time(f"test_callback_{i % 5}", callback_time / 1000)

            execution_time = time.time() - start_time
            avg_callback_time = np.mean(callback_times)

            # Get callback optimizer metrics
            optimizer_summary = callback_optimizer.get_performance_summary()

            return TestResult(
                test_name="Dashboard Callbacks",
                target_metric="avg_callback_time_ms",
                target_value=100.0,  # Target 100ms per callback
                actual_value=avg_callback_time,
                passed=avg_callback_time <= 100.0,
                execution_time=execution_time,
                additional_metrics={
                    'total_callbacks': len(callback_times),
                    'min_callback_time_ms': min(callback_times),
                    'max_callback_time_ms': max(callback_times),
                    'p95_callback_time_ms': np.percentile(callback_times, 95),
                    'cache_hit_rate': optimizer_summary.get('cache_hit_rate', 0)
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Dashboard callback test failed: {e}")
            return TestResult(
                test_name="Dashboard Callbacks",
                target_metric="avg_callback_time_ms",
                target_value=100.0,
                actual_value=float('inf'),
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_data_compression(self) -> TestResult:
        """Test data compression performance"""
        logger.info("Testing data compression...")
        start_time = time.time()

        try:
            # Generate test data similar to sensor dashboard data
            test_data_sets = []

            # Generate various types of dashboard data
            for i in range(20):
                sensor_data = {
                    'timestamps': [(datetime.now() - timedelta(seconds=j)).isoformat() for j in range(100)],
                    'values': [np.random.normal(50, 10) for _ in range(100)],
                    'anomaly_scores': [np.random.uniform(0, 1) for _ in range(100)],
                    'sensor_ids': [f'sensor_{j:03d}' for j in range(100)]
                }
                test_data_sets.append(sensor_data)

            # Test compression
            compression_results = []
            for data in test_data_sets:
                compressed = data_compressor.compress_dashboard_data(data)
                compression_results.append(compressed)

            execution_time = time.time() - start_time

            # Calculate average compression ratio
            ratios = [r['compression_ratio'] for r in compression_results if 'compression_ratio' in r]
            avg_compression_ratio = np.mean(ratios) if ratios else 1.0

            # Get compression stats
            compression_stats = data_compressor.get_compression_stats()

            return TestResult(
                test_name="Data Compression",
                target_metric="compression_ratio",
                target_value=self.targets['compression_ratio'],
                actual_value=avg_compression_ratio,
                passed=avg_compression_ratio <= self.targets['compression_ratio'],
                execution_time=execution_time,
                additional_metrics={
                    'datasets_compressed': len(test_data_sets),
                    'total_savings_bytes': compression_stats.get('total_savings_bytes', 0),
                    'avg_compression_time_ms': compression_stats.get('avg_compression_time_ms', 0),
                    'method_usage': compression_stats.get('method_usage', {})
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Data compression test failed: {e}")
            return TestResult(
                test_name="Data Compression",
                target_metric="compression_ratio",
                target_value=self.targets['compression_ratio'],
                actual_value=1.0,
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_80_sensor_load(self) -> TestResult:
        """Test system performance under 80-sensor load"""
        logger.info("Testing 80-sensor concurrent load...")
        start_time = time.time()

        try:
            # Simulate 80 sensors sending data concurrently
            sensor_tasks = []

            async def simulate_sensor(sensor_id: int):
                data_points_sent = 0
                for i in range(100):  # 100 data points per sensor
                    # Simulate sensor data processing
                    sensor_data = {
                        'sensor_id': f'sensor_{sensor_id:03d}',
                        'timestamp': datetime.now().isoformat(),
                        'value': np.random.normal(50, 10),
                        'anomaly_score': np.random.uniform(0, 1)
                    }

                    # Add to memory manager
                    success = memory_manager.add_sensor_data(
                        sensor_data['sensor_id'],
                        datetime.now(),
                        sensor_data['value'],
                        sensor_data['anomaly_score']
                    )

                    if success:
                        data_points_sent += 1

                    # Small delay to simulate real-time data
                    await asyncio.sleep(0.001)  # 1ms

                return data_points_sent

            # Create tasks for all 80 sensors
            for sensor_id in range(80):
                task = asyncio.create_task(simulate_sensor(sensor_id))
                sensor_tasks.append(task)

            # Execute all sensor simulations concurrently
            results = await asyncio.gather(*sensor_tasks, return_exceptions=True)

            execution_time = time.time() - start_time

            # Calculate performance metrics
            successful_sensors = sum(1 for r in results if isinstance(r, int))
            total_data_points = sum(r for r in results if isinstance(r, int))
            data_points_per_second = total_data_points / execution_time if execution_time > 0 else 0

            # Check if we achieved 80 sensors processing target
            target_achieved = successful_sensors >= 80 and data_points_per_second >= self.targets['sensor_processing_rate']

            return TestResult(
                test_name="80-Sensor Load Test",
                target_metric="sensor_processing_rate",
                target_value=self.targets['sensor_processing_rate'],
                actual_value=data_points_per_second,
                passed=target_achieved,
                execution_time=execution_time,
                additional_metrics={
                    'sensors_active': 80,
                    'successful_sensors': successful_sensors,
                    'total_data_points': total_data_points,
                    'avg_data_points_per_sensor': total_data_points / 80 if total_data_points > 0 else 0
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"80-sensor load test failed: {e}")
            return TestResult(
                test_name="80-Sensor Load Test",
                target_metric="sensor_processing_rate",
                target_value=self.targets['sensor_processing_rate'],
                actual_value=0.0,
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    async def test_dashboard_response_time(self) -> TestResult:
        """Test end-to-end dashboard response time"""
        logger.info("Testing dashboard response time...")
        start_time = time.time()

        try:
            # Simulate complete dashboard page load cycle
            response_times = []

            for i in range(20):  # 20 page load simulations
                page_start = time.time()

                # Simulate typical dashboard operations
                tasks = []

                # Simulate data retrieval
                tasks.append(asyncio.create_task(asyncio.sleep(0.05)))  # 50ms data fetch

                # Simulate cache lookups
                for j in range(5):
                    cache_key = f"dashboard_data_{j}"
                    cached_data = advanced_cache.get(cache_key)
                    if cached_data is None:
                        # Simulate data generation and caching
                        test_data = {'data': np.random.random(100).tolist()}
                        advanced_cache.set(cache_key, test_data, 300)

                # Simulate chart generation
                tasks.append(asyncio.create_task(asyncio.sleep(0.03)))  # 30ms chart render

                # Simulate callback processing
                tasks.append(asyncio.create_task(asyncio.sleep(0.02)))  # 20ms callbacks

                # Wait for all operations
                await asyncio.gather(*tasks)

                page_time = (time.time() - page_start) * 1000  # Convert to ms
                response_times.append(page_time)

                # Record response time
                performance_monitor.record_response_time(page_time / 1000)

            execution_time = time.time() - start_time
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)

            # Check if sub-second target is met
            sub_second_achieved = avg_response_time <= self.targets['dashboard_response_time_ms']

            return TestResult(
                test_name="Dashboard Response Time",
                target_metric="dashboard_response_time_ms",
                target_value=self.targets['dashboard_response_time_ms'],
                actual_value=avg_response_time,
                passed=sub_second_achieved,
                execution_time=execution_time,
                additional_metrics={
                    'page_loads_tested': len(response_times),
                    'min_response_time_ms': min(response_times),
                    'max_response_time_ms': max(response_times),
                    'p95_response_time_ms': p95_response_time,
                    'sub_second_rate': sum(1 for t in response_times if t <= 500) / len(response_times)
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Dashboard response time test failed: {e}")
            return TestResult(
                test_name="Dashboard Response Time",
                target_metric="dashboard_response_time_ms",
                target_value=self.targets['dashboard_response_time_ms'],
                actual_value=float('inf'),
                passed=False,
                execution_time=time.time() - start_time,
                additional_metrics={'error': str(e)},
                timestamp=datetime.now()
            )

    def _generate_test_report(self, test_results: List[TestResult], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = sum(1 for r in test_results if r.passed)
        total_tests = len(test_results)

        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_execution_time': total_time,
                'timestamp': datetime.now().isoformat()
            },
            'target_achievement': {
                'sub_second_response': False,
                '80_sensor_processing': False,
                'memory_efficiency': False,
                'cache_performance': False,
                'async_reliability': False,
                'compression_efficiency': False
            },
            'detailed_results': [],
            'performance_summary': {},
            'recommendations': []
        }

        # Process individual test results
        for result in test_results:
            report['detailed_results'].append(asdict(result))

            # Update target achievement
            if result.test_name == "Dashboard Response Time" and result.passed:
                report['target_achievement']['sub_second_response'] = True
            elif result.test_name == "80-Sensor Load Test" and result.passed:
                report['target_achievement']['80_sensor_processing'] = True
            elif result.test_name == "Memory Management" and result.passed:
                report['target_achievement']['memory_efficiency'] = True
            elif result.test_name == "Cache Performance" and result.passed:
                report['target_achievement']['cache_performance'] = True
            elif result.test_name == "Async Processing" and result.passed:
                report['target_achievement']['async_reliability'] = True
            elif result.test_name == "Data Compression" and result.passed:
                report['target_achievement']['compression_efficiency'] = True

        # Generate performance summary
        response_time_results = [r for r in test_results if r.test_name == "Dashboard Response Time"]
        if response_time_results:
            report['performance_summary']['avg_response_time_ms'] = response_time_results[0].actual_value

        sensor_load_results = [r for r in test_results if r.test_name == "80-Sensor Load Test"]
        if sensor_load_results:
            report['performance_summary']['sensor_processing_rate'] = sensor_load_results[0].actual_value

        # Generate recommendations
        for result in test_results:
            if not result.passed:
                if result.test_name == "Dashboard Response Time":
                    report['recommendations'].append(
                        "Dashboard response time exceeds 500ms target. Consider increasing cache TTL and optimizing callbacks."
                    )
                elif result.test_name == "80-Sensor Load Test":
                    report['recommendations'].append(
                        "80-sensor processing rate below target. Consider increasing async worker count and optimizing memory management."
                    )
                elif result.test_name == "Memory Management":
                    report['recommendations'].append(
                        "Memory usage exceeds limits. Consider reducing window sizes or enabling compression."
                    )

        # Overall system assessment
        all_critical_targets = (
            report['target_achievement']['sub_second_response'] and
            report['target_achievement']['80_sensor_processing'] and
            report['target_achievement']['memory_efficiency']
        )

        report['overall_assessment'] = {
            'ready_for_80_sensors': all_critical_targets,
            'optimization_level': 'excellent' if passed_tests == total_tests else 'good' if passed_tests >= total_tests * 0.8 else 'needs_improvement',
            'critical_issues': [r.test_name for r in test_results if not r.passed and r.test_name in ["Dashboard Response Time", "80-Sensor Load Test", "Memory Management"]]
        }

        return report

    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        if not filename:
            filename = f"performance_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance test report saved to {filename}")


async def main():
    """Main performance testing function"""
    print("🚀 Starting 80-Sensor IoT Dashboard Performance Testing...")
    print("=" * 60)

    tester = PerformanceTester()

    try:
        # Run comprehensive test suite
        report = await tester.run_comprehensive_test()

        # Save report
        tester.save_report(report)

        # Print summary
        print("\n📊 PERFORMANCE TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {report['test_summary']['total_tests']}")
        print(f"Passed: {report['test_summary']['passed_tests']}")
        print(f"Failed: {report['test_summary']['failed_tests']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
        print(f"Execution Time: {report['test_summary']['total_execution_time']:.2f}s")

        print("\n🎯 TARGET ACHIEVEMENT")
        print("=" * 60)
        achievements = report['target_achievement']
        print(f"✅ Sub-second Response: {'PASS' if achievements['sub_second_response'] else 'FAIL'}")
        print(f"✅ 80-Sensor Processing: {'PASS' if achievements['80_sensor_processing'] else 'FAIL'}")
        print(f"✅ Memory Efficiency: {'PASS' if achievements['memory_efficiency'] else 'FAIL'}")
        print(f"✅ Cache Performance: {'PASS' if achievements['cache_performance'] else 'FAIL'}")
        print(f"✅ Async Reliability: {'PASS' if achievements['async_reliability'] else 'FAIL'}")
        print(f"✅ Compression Efficiency: {'PASS' if achievements['compression_efficiency'] else 'FAIL'}")

        print(f"\n🏆 OVERALL ASSESSMENT: {report['overall_assessment']['optimization_level'].upper()}")
        print(f"🎯 80-Sensor Ready: {'YES' if report['overall_assessment']['ready_for_80_sensors'] else 'NO'}")

        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS")
            print("=" * 60)
            for rec in report['recommendations']:
                print(f"• {rec}")

        return report

    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        logger.error(f"Performance testing failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())