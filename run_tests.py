#!/usr/bin/env python3
"""
Test runner script for IoT Predictive Maintenance System
Provides convenient commands to run different test suites
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="IoT Predictive Maintenance System Test Runner")
    parser.add_argument(
        "suite",
        choices=[
            "all", "unit", "integration", "performance", "security",
            "phase1", "phase2", "phase3", "phase4",
            "quick", "coverage", "components"
        ],
        help="Test suite to run"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")

    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    if args.verbose:
        cmd.append("-v")

    if args.parallel:
        cmd.extend(["-n", "auto"])

    if args.coverage or args.suite == "coverage":
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.html:
            cmd.append("--cov-report=html")

    # Test suite selection
    if args.suite == "all":
        cmd.append("tests/")
    elif args.suite == "unit":
        cmd.extend(["-m", "unit", "tests/unit/"])
    elif args.suite == "integration":
        cmd.extend(["-m", "integration", "tests/integration/"])
    elif args.suite == "performance":
        cmd.extend(["-m", "performance", "tests/performance/"])
    elif args.suite == "security":
        cmd.extend(["-m", "security", "tests/security/"])
    elif args.suite == "phase1":
        cmd.extend(["-m", "phase1", "tests/phase_tests/phase1/"])
    elif args.suite == "phase2":
        cmd.extend(["-m", "phase2", "tests/phase_tests/phase2/"])
    elif args.suite == "phase3":
        cmd.extend(["-m", "phase3", "tests/phase_tests/phase3/"])
    elif args.suite == "phase4":
        cmd.extend(["-m", "phase4", "tests/phase_tests/phase4/"])
    elif args.suite == "components":
        cmd.append("tests/component_tests/")
    elif args.suite == "quick":
        cmd.extend(["-m", "not slow", "tests/unit/", "tests/component_tests/"])
    elif args.suite == "coverage":
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing", "tests/"])

    print(f"IoT Predictive Maintenance System - Running {args.suite} tests")
    print("=" * 60)

    success = run_command(cmd)

    if success:
        print("\n" + "=" * 60)
        print(f"✅ {args.suite.upper()} tests completed successfully!")
    else:
        print("\n" + "=" * 60)
        print(f"❌ {args.suite.upper()} tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()