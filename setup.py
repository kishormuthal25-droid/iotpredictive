"""
Setup configuration for IoT Anomaly Detection System
Enables the project to be installed as a Python package
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
def read_requirements(filename='requirements.txt'):
    """Read requirements from file"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core dependencies (minimal set for basic functionality)
core_requirements = [
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'scipy>=1.11.0',
    'scikit-learn>=1.3.0',
    'h5py>=3.9.0',
    'tensorflow>=2.13.0',
    'pyyaml>=6.0.0',
    'loguru>=0.7.0',
    'tqdm>=4.65.0',
]

# Optional dependencies for different components
extras_require = {
    'dashboard': [
        'dash>=2.11.0',
        'dash-bootstrap-components>=1.4.0',
        'plotly>=5.15.0',
        'dash-daq>=0.5.0',
        'dash-extensions>=1.0.0',
    ],
    'streaming': [
        'kafka-python>=2.0.2',
        'redis>=4.6.0',
        'aioredis>=2.0.0',
    ],
    'database': [
        'psycopg2-binary>=2.9.0',
        'sqlalchemy>=2.0.0',
        'alembic>=1.11.0',
    ],
    'optimization': [
        'pulp>=2.7.0',
        'ortools>=9.6.0',
        'networkx>=3.1',
    ],
    'alerts': [
        'yagmail>=0.15.0',
        'email-validator>=2.0.0',
    ],
    'dev': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'black>=23.7.0',
        'flake8>=6.0.0',
        'mypy>=1.4.0',
        'isort>=5.12.0',
        'pre-commit>=3.3.0',
    ],
    'ml-extras': [
        'xgboost>=1.7.0',
        'lightgbm>=4.0.0',
        'prophet>=1.1.0',
        'imbalanced-learn>=0.11.0',
    ],
}

# All extras combined
extras_require['all'] = list(set(sum(extras_require.values(), [])))

# Package metadata
setup(
    name='iot-anomaly-detection-system',
    version='1.0.0',
    author='IoT Analytics Team',
    author_email='iot-analytics@example.com',
    description='Scalable IoT Anomaly Detection System with NASA SMAP/MSL Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/iot-anomaly-detection',
    license='MIT',
    
    # Package discovery
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        'config': ['*.yaml', '*.yml'],
        'dashboard': ['assets/*.css', 'assets/*.js'],
    },
    
    # Python version requirement
    python_requires='>=3.8,<3.12',
    
    # Dependencies
    install_requires=core_requirements,
    extras_require=extras_require,
    
    # Entry points for CLI commands
    entry_points={
        'console_scripts': [
            'iot-start-pipeline=scripts.start_pipeline:main',
            'iot-run-dashboard=scripts.run_dashboard:main',
            'iot-train-models=scripts.train_models:main',
            'iot-download-data=scripts.download_data:main',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Monitoring',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Framework :: Dash',
        'Environment :: Console',
        'Environment :: Web Environment',
    ],
    
    # Additional metadata
    keywords='iot anomaly-detection lstm autoencoder vae nasa smap msl predictive-maintenance',
    project_urls={
        'Documentation': 'https://github.com/your-org/iot-anomaly-detection/wiki',
        'Bug Reports': 'https://github.com/your-org/iot-anomaly-detection/issues',
        'Source': 'https://github.com/your-org/iot-anomaly-detection',
    },
    
    # Testing
    test_suite='tests',
    tests_require=[
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
    ],
    
    # Additional options
    zip_safe=False,  # Don't install as zip file
)

# Create necessary directories on installation
def post_install():
    """Create necessary directories after installation"""
    import os
    
    dirs_to_create = [
        'data/raw/smap',
        'data/raw/msl',
        'data/processed',
        'data/models',
        'logs',
        'config',
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

# Installation instructions
if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     IoT Anomaly Detection System - Installation Guide        ║
    ╚══════════════════════════════════════════════════════════════╝
    
    To install the package:
    
    1. Basic installation (core features only):
       pip install .
    
    2. Installation with all features:
       pip install .[all]
    
    3. Installation for development:
       pip install -e .[dev]
    
    4. Installation with specific features:
       pip install .[dashboard,streaming,optimization]
    
    Available feature groups:
    - dashboard: Interactive Dash-based UI components
    - streaming: Kafka and Redis for data streaming
    - database: PostgreSQL/TimescaleDB support
    - optimization: Maintenance scheduling optimization
    - alerts: Email notification system
    - dev: Development and testing tools
    - ml-extras: Additional ML algorithms for comparison
    
    After installation, run:
    - iot-download-data      : Download NASA datasets
    - iot-train-models       : Train anomaly detection models
    - iot-start-pipeline     : Start the data ingestion pipeline
    - iot-run-dashboard      : Launch the monitoring dashboard
    
    For development mode (editable installation):
       pip install -e .[all,dev]
    
    This allows you to modify the code without reinstalling.
    """)
    
    # Run post-installation tasks
    try:
        post_install()
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")