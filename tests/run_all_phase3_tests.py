#!/usr/bin/env python3
"""
Phase 3 Comprehensive Test Runner
Runs all Phase 3.1 test suites in sequence
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Suppress warnings during testing
warnings.filterwarnings('ignore')


def print_banner(title, char="=", width=80):
    """Print a formatted banner"""
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)


def print_section(title, char="-", width=80):
    """Print a section header"""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def run_test_suite(test_module_path, description):
    """Run a specific test suite and return success status"""
    print_section(f"TEST: {description}")

    start_time = time.time()

    try:
        # Import and run the test module
        sys.path.insert(0, str(test_module_path.parent))

        if test_module_path.name == "test_model_lazy_loading.py":
            from tests.mlflow_tests.test_model_lazy_loading import run_performance_benchmark
            success = run_performance_benchmark()

        elif test_module_path.name == "test_model_registry_integration.py":
            from tests.mlflow_tests.test_model_registry_integration import run_comprehensive_mlflow_test
            success = run_comprehensive_mlflow_test()

        elif test_module_path.name == "test_complete_dashboard_flow.py":
            from tests.e2e_tests.test_complete_dashboard_flow import run_e2e_dashboard_tests
            success = run_e2e_dashboard_tests()

        elif test_module_path.name == "test_real_data_processing.py":
            from tests.e2e_tests.test_real_data_processing import run_real_data_processing_e2e
            success = run_real_data_processing_e2e()

        elif test_module_path.name == "test_startup_sequence.py":
            from tests.system_tests.test_startup_sequence import run_system_startup_tests
            success = run_system_startup_tests()

        else:
            print(f"❌ Unknown test module: {test_module_path.name}")
            success = False

    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        success = False
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        success = False

    duration = time.time() - start_time

    if success:
        print(f"\n✅ {description} PASSED ({duration:.1f}s)")
    else:
        print(f"\n❌ {description} FAILED ({duration:.1f}s)")

    return success, duration


def run_all_phase3_tests():
    """Run all Phase 3.1 test suites"""

    print_banner("PHASE 3.1 COMPREHENSIVE TEST SUITE", "=")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")

    # Define test suites in execution order
    test_suites = [
        {
            'path': Path(__file__).parent / "mlflow_tests" / "test_model_lazy_loading.py",
            'description': "MLflow Lazy Loading Performance Tests",
            'category': 'performance',
            'critical': True
        },
        {
            'path': Path(__file__).parent / "mlflow_tests" / "test_model_registry_integration.py",
            'description': "MLflow Model Registry Integration Tests",
            'category': 'integration',
            'critical': True
        },
        {
            'path': Path(__file__).parent / "system_tests" / "test_startup_sequence.py",
            'description': "System Startup Sequence Tests",
            'category': 'system',
            'critical': True
        },
        {
            'path': Path(__file__).parent / "e2e_tests" / "test_complete_dashboard_flow.py",
            'description': "End-to-End Dashboard Flow Tests",
            'category': 'e2e',
            'critical': False
        },
        {
            'path': Path(__file__).parent / "e2e_tests" / "test_real_data_processing.py",
            'description': "Real Data Processing E2E Tests",
            'category': 'e2e',
            'critical': False
        }
    ]

    # Track results
    results = []
    total_start_time = time.time()

    # Run each test suite
    for suite in test_suites:
        if not suite['path'].exists():
            print(f"⚠️  Test file not found: {suite['path']}")
            results.append({
                'description': suite['description'],
                'success': False,
                'duration': 0,
                'category': suite['category'],
                'critical': suite['critical'],
                'error': 'File not found'
            })
            continue

        success, duration = run_test_suite(suite['path'], suite['description'])

        results.append({
            'description': suite['description'],
            'success': success,
            'duration': duration,
            'category': suite['category'],
            'critical': suite['critical']
        })

        # Brief pause between test suites
        time.sleep(1)

    total_duration = time.time() - total_start_time

    # Print comprehensive summary
    print_banner("📊 PHASE 3.1 TEST RESULTS SUMMARY", "=")

    # Results by category
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = {'passed': 0, 'failed': 0, 'total_time': 0}

        if result['success']:
            categories[cat]['passed'] += 1
        else:
            categories[cat]['failed'] += 1
        categories[cat]['total_time'] += result['duration']

    # Print category summary
    for category, stats in categories.items():
        total_tests = stats['passed'] + stats['failed']
        pass_rate = (stats['passed'] / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🏷️  {category.upper()} Tests:")
        print(f"    ✅ Passed: {stats['passed']}")
        print(f"    ❌ Failed: {stats['failed']}")
        print(f"    📊 Pass Rate: {pass_rate:.1f}%")
        print(f"    ⏱️  Total Time: {stats['total_time']:.1f}s")

    # Overall statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    overall_pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    print(f"\n📈 OVERALL STATISTICS:")
    print(f"    🧪 Total Test Suites: {total_tests}")
    print(f"    ✅ Passed: {passed_tests}")
    print(f"    ❌ Failed: {failed_tests}")
    print(f"    📊 Overall Pass Rate: {overall_pass_rate:.1f}%")
    print(f"    ⏱️  Total Execution Time: {total_duration:.1f}s")

    # Critical test analysis
    critical_tests = [r for r in results if r['critical']]
    critical_passed = sum(1 for r in critical_tests if r['success'])
    critical_failed = len(critical_tests) - critical_passed

    if critical_tests:
        critical_pass_rate = (critical_passed / len(critical_tests) * 100)
        print(f"\n🚨 CRITICAL TESTS:")
        print(f"    ✅ Passed: {critical_passed}")
        print(f"    ❌ Failed: {critical_failed}")
        print(f"    📊 Critical Pass Rate: {critical_pass_rate:.1f}%")

    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        critical_marker = "🚨" if result['critical'] else "  "
        print(f"    {critical_marker} {status} - {result['description']} ({result['duration']:.1f}s)")
        if not result['success'] and 'error' in result:
            print(f"        Error: {result['error']}")

    # Performance benchmarks
    performance_results = [r for r in results if r['category'] == 'performance']
    if performance_results:
        print(f"\n⚡ PERFORMANCE BENCHMARKS:")
        for result in performance_results:
            if result['success']:
                print(f"    ✅ {result['description']}: {result['duration']:.1f}s")
            else:
                print(f"    ❌ {result['description']}: FAILED")

    # Final verdict
    print_banner("🏆 FINAL VERDICT", "=")

    if critical_failed == 0 and overall_pass_rate >= 80:
        print("🎉 PHASE 3.1 TEST SUITE: EXCELLENT")
        print("   All critical tests passed, system ready for production!")
        verdict = "EXCELLENT"
    elif critical_failed == 0 and overall_pass_rate >= 60:
        print("✅ PHASE 3.1 TEST SUITE: GOOD")
        print("   Critical tests passed, minor issues to address")
        verdict = "GOOD"
    elif critical_failed <= 1 and overall_pass_rate >= 50:
        print("⚠️  PHASE 3.1 TEST SUITE: NEEDS ATTENTION")
        print("   Some critical issues found, requires fixes")
        verdict = "NEEDS_ATTENTION"
    else:
        print("❌ PHASE 3.1 TEST SUITE: MAJOR ISSUES")
        print("   Multiple critical failures, significant work needed")
        verdict = "MAJOR_ISSUES"

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if failed_tests == 0:
        print("   🎯 System is production ready!")
        print("   🚀 Proceed with Phase 3.2 (Documentation)")
    elif critical_failed == 0:
        print("   🔧 Fix non-critical issues for optimal performance")
        print("   🚀 Can proceed with Phase 3.2 while addressing issues")
    else:
        print("   🚨 Address critical test failures immediately")
        print("   ⏸️  Hold Phase 3.2 until critical issues resolved")

    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return verdict, overall_pass_rate, critical_failed == 0


if __name__ == '__main__':
    try:
        verdict, pass_rate, critical_success = run_all_phase3_tests()

        # Exit code based on results
        if verdict == "EXCELLENT":
            sys.exit(0)
        elif verdict == "GOOD":
            sys.exit(0)
        elif verdict == "NEEDS_ATTENTION":
            sys.exit(1)
        else:
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test runner failed with error: {e}")
        sys.exit(1)