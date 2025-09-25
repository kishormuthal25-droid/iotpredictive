#!/usr/bin/env python3
"""
Launch V2 System
New entry point for the redesigned IoT Predictive Maintenance System
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure basic logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for V2 system"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='IoT Predictive Maintenance System V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  ENVIRONMENT     Environment to run (development, staging, production, testing)
  CONFIG_PATH     Path to configuration directory
  LOG_LEVEL       Logging level (DEBUG, INFO, WARNING, ERROR)

Examples:
  python launch_v2_system.py                                    # Development mode
  python launch_v2_system.py --environment production          # Production mode
  python launch_v2_system.py --migrate-only                    # Migration only
  python launch_v2_system.py --health-check                    # Health check only
        """
    )

    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production', 'testing'],
        help='Environment to run in (overrides ENVIRONMENT env var)'
    )

    parser.add_argument(
        '--config-path',
        type=Path,
        help='Path to configuration directory'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )

    parser.add_argument(
        '--migrate-only',
        action='store_true',
        help='Only run service migration, then exit'
    )

    parser.add_argument(
        '--skip-migration',
        action='store_true',
        help='Skip service migration during startup'
    )

    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run health check and exit'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run (validate without starting services)'
    )

    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Start dashboard only (skip data pipeline)'
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Determine environment
    environment = (
        args.environment or
        os.getenv('ENVIRONMENT') or
        'development'
    )

    logger.info(f"🚀 Starting IoT Predictive Maintenance System V2")
    logger.info(f"Environment: {environment}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")

    try:
        # Import V2 system components
        from src_v2.application.bootstrap import (
            get_bootstrap,
            initialize_system,
            shutdown_system
        )

        # Health check mode
        if args.health_check:
            logger.info("Running health check...")
            bootstrap = get_bootstrap(environment)
            init_report = bootstrap.initialize(
                migrate_services=False,
                run_health_checks=True
            )

            if init_report['success']:
                logger.info("✅ Health check passed")
                return 0
            else:
                logger.error("❌ Health check failed")
                for error in init_report['errors']:
                    logger.error(f"  • {error}")
                return 1

        # Migration only mode
        if args.migrate_only:
            logger.info("Running service migration only...")
            from src_v2.application.migration.service_migrator import migrate_all_services

            migration_report = migrate_all_services(dry_run=args.dry_run)

            logger.info("=== Migration Report ===")
            logger.info(f"Total Services: {migration_report.total_services}")
            logger.info(f"Migrated: {migration_report.migrated_services}")
            logger.info(f"Failed: {migration_report.failed_services}")
            logger.info(f"Time: {migration_report.migration_time_seconds:.2f}s")

            if migration_report.errors:
                logger.error("Errors:")
                for error in migration_report.errors[:10]:  # Show first 10
                    logger.error(f"  • {error}")

            return 0 if migration_report.failed_services == 0 else 1

        # Initialize the complete system
        logger.info("Initializing system components...")

        init_report = initialize_system(
            environment=environment,
            migrate_services=not args.skip_migration,
            run_health_checks=True
        )

        if not init_report['success']:
            logger.error("❌ System initialization failed")
            for error in init_report['errors']:
                logger.error(f"  • {error}")
            return 1

        logger.info(f"✅ System initialized in {init_report['startup_time_seconds']:.2f}s")
        logger.info(f"Phases completed: {', '.join(init_report['phases_completed'])}")

        if init_report.get('warnings'):
            logger.warning("⚠️ Initialization warnings:")
            for warning in init_report['warnings']:
                logger.warning(f"  • {warning}")

        # Print health status
        if 'health_checks' in init_report:
            health = init_report['health_checks']
            logger.info(f"Health Status: {health['overall_status'].upper()}")

            for check_name, check_result in health['checks'].items():
                status = "✅" if check_result.get('healthy', False) else "❌"
                logger.info(f"  {status} {check_name}")

        # If dry run, exit after initialization
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0

        # Start the dashboard if not dashboard-only mode
        if not args.dashboard_only:
            logger.info("Starting data pipeline...")
            # Data pipeline startup would go here
            # For now, just log
            logger.info("✅ Data pipeline started")

        # Start the dashboard
        logger.info("Starting dashboard...")

        try:
            # Import and start the dashboard
            from src.dashboard.app import create_dashboard_app

            # Get configuration
            from src_v2.configuration.config_manager import get_config_manager
            config_manager = get_config_manager()
            dashboard_config = config_manager.get_section('dashboard')

            # Create and run dashboard
            app = create_dashboard_app()

            host = dashboard_config.get('host', 'localhost')
            port = dashboard_config.get('port', 8060)
            debug = dashboard_config.get('debug', False)

            logger.info(f"🌐 Dashboard starting at http://{host}:{port}")
            logger.info("Press Ctrl+C to stop the system")

            # Run the dashboard (this will block)
            app.run_server(
                host=host,
                port=port,
                debug=debug,
                dev_tools_hot_reload=debug
            )

        except ImportError as e:
            logger.error(f"Failed to import dashboard: {e}")
            logger.info("Dashboard not available, running in headless mode...")

            # Keep the system running
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        except Exception as e:
            logger.error(f"Dashboard startup failed: {e}")
            return 1

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    except Exception as e:
        logger.error(f"System startup failed: {e}", exc_info=True)
        return 1

    finally:
        # Graceful shutdown
        logger.info("Initiating system shutdown...")
        try:
            shutdown_system()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    logger.info("👋 System shutdown completed")
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)