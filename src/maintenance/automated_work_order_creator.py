"""
Automated Work Order Creator Module for Phase 3.2
Advanced automation for creating work orders from anomaly detection with intelligent prioritization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import pickle
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path
from src.maintenance.work_order_manager import (
    WorkOrder, WorkOrderPriority, WorkOrderStatus, MaintenanceType,
    Equipment, Technician, WorkOrderManager
)

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AnomalyWorkOrderMapping:
    """Mapping configuration for anomaly to work order conversion"""
    anomaly_type: str
    equipment_type: str
    priority_base: WorkOrderPriority
    maintenance_type: MaintenanceType
    estimated_duration: float
    required_skills: List[str]
    parts_likely_needed: List[str]
    cost_multiplier: float = 1.0
    sla_hours_override: Optional[int] = None
    escalation_threshold: float = 0.9  # Confidence threshold for escalation


@dataclass
class EquipmentCriticalityProfile:
    """Equipment criticality assessment for priority calculation"""
    equipment_id: str
    equipment_type: str
    business_impact: str  # 'critical', 'high', 'medium', 'low'
    safety_impact: str    # 'critical', 'high', 'medium', 'low'
    operational_impact: str # 'critical', 'high', 'medium', 'low'
    downtime_cost_per_hour: float
    replacement_cost: float
    dependencies: List[str]  # List of dependent equipment/systems
    redundancy_available: bool
    maintenance_complexity: str  # 'simple', 'moderate', 'complex', 'specialized'


class AutomatedWorkOrderCreator:
    """Advanced automated work order creation system for Phase 3.2"""

    def __init__(self,
                 work_order_manager: WorkOrderManager,
                 anomaly_detection_config: Optional[Dict] = None,
                 ml_models_path: Optional[str] = None):
        """Initialize Automated Work Order Creator

        Args:
            work_order_manager: Main work order management system
            anomaly_detection_config: Configuration for anomaly integration
            ml_models_path: Path to ML models for priority prediction
        """
        self.work_order_manager = work_order_manager
        self.anomaly_config = anomaly_detection_config or {}
        self.ml_models_path = ml_models_path

        # Initialize components
        self.anomaly_mapper = AnomalyToWorkOrderMapper()
        self.priority_calculator = IntelligentPriorityCalculator()
        self.equipment_criticality_assessor = EquipmentCriticalityAssessor()
        self.automation_engine = WorkOrderAutomationEngine()

        # Configuration storage
        self.anomaly_mappings = {}
        self.equipment_profiles = {}
        self.automation_rules = {}

        # Performance tracking
        self.automation_metrics = {
            'total_automated_orders': 0,
            'automation_success_rate': 0.0,
            'average_creation_time': 0.0,
            'false_positive_rate': 0.0,
            'priority_accuracy': 0.0
        }

        # Background processing
        self.anomaly_queue = queue.Queue()
        self.processing_executor = ThreadPoolExecutor(max_workers=4)
        self.auto_processing_enabled = True

        # Initialize default mappings and profiles
        self._initialize_default_configurations()

        logger.info("Initialized Automated Work Order Creator")

    def process_anomaly_for_work_order(self,
                                     anomaly_data: Dict[str, Any],
                                     equipment_id: str,
                                     auto_assign: bool = True,
                                     force_creation: bool = False) -> Optional[WorkOrder]:
        """Process anomaly and automatically create work order if criteria met

        Args:
            anomaly_data: Anomaly detection results with enhanced metadata
            equipment_id: Equipment identifier
            auto_assign: Automatically assign technician
            force_creation: Force creation even if confidence is low

        Returns:
            Created work order or None if not created
        """
        try:
            # Enhanced anomaly analysis
            anomaly_analysis = self._analyze_anomaly_for_automation(anomaly_data, equipment_id)

            # Determine if work order should be created
            if not force_creation and not self._should_create_work_order(anomaly_analysis):
                logger.info(f"Anomaly {anomaly_data.get('anomaly_id')} does not meet automation criteria")
                return None

            # Get equipment criticality profile
            equipment_profile = self.equipment_profiles.get(
                equipment_id,
                self._create_default_equipment_profile(equipment_id)
            )

            # Calculate intelligent priority
            priority = self.priority_calculator.calculate_priority(
                anomaly_analysis,
                equipment_profile,
                anomaly_data
            )

            # Determine optimal maintenance type
            maintenance_type = self._determine_optimal_maintenance_type(
                anomaly_analysis,
                equipment_profile
            )

            # Create enhanced work order
            work_order = self._create_enhanced_work_order(
                anomaly_data,
                equipment_id,
                priority,
                maintenance_type,
                anomaly_analysis,
                equipment_profile
            )

            # Store in work order manager
            self.work_order_manager.work_orders[work_order.order_id] = work_order
            self.work_order_manager.metrics['total_orders'] += 1

            # Auto-assign if requested
            if auto_assign:
                self._intelligent_auto_assignment(work_order, equipment_profile)

            # Update automation metrics
            self._update_automation_metrics(work_order, anomaly_analysis)

            logger.info(f"Automatically created work order {work_order.order_id} from anomaly {anomaly_data.get('anomaly_id')}")

            return work_order

        except Exception as e:
            logger.error(f"Error processing anomaly for work order creation: {e}")
            return None

    def _analyze_anomaly_for_automation(self,
                                      anomaly_data: Dict[str, Any],
                                      equipment_id: str) -> Dict[str, Any]:
        """Analyze anomaly for automation decision making

        Args:
            anomaly_data: Raw anomaly data
            equipment_id: Equipment identifier

        Returns:
            Enhanced anomaly analysis
        """
        analysis = {
            'anomaly_id': anomaly_data.get('anomaly_id'),
            'raw_severity': anomaly_data.get('severity', 'medium'),
            'confidence': anomaly_data.get('confidence', 0.5),
            'anomaly_type': anomaly_data.get('type', 'unknown'),
            'sensor_involved': anomaly_data.get('sensor', 'unknown'),
            'timestamp': anomaly_data.get('timestamp', datetime.now()),
            'value': anomaly_data.get('value'),
            'threshold': anomaly_data.get('threshold'),
            'trend_analysis': anomaly_data.get('trend', {}),
            'historical_context': anomaly_data.get('historical_context', {}),
            'equipment_id': equipment_id
        }

        # Enhanced analysis
        analysis['severity_score'] = self._calculate_severity_score(anomaly_data)
        analysis['urgency_score'] = self._calculate_urgency_score(anomaly_data, equipment_id)
        analysis['impact_score'] = self._calculate_impact_score(anomaly_data, equipment_id)
        analysis['automation_confidence'] = self._calculate_automation_confidence(analysis)

        # Contextual analysis
        analysis['similar_anomalies'] = self._find_similar_historical_anomalies(anomaly_data)
        analysis['failure_probability'] = self._estimate_failure_probability(anomaly_data, equipment_id)
        analysis['recommended_actions'] = self._get_recommended_actions(analysis)

        return analysis

    def _should_create_work_order(self, anomaly_analysis: Dict[str, Any]) -> bool:
        """Determine if work order should be automatically created

        Args:
            anomaly_analysis: Enhanced anomaly analysis

        Returns:
            True if work order should be created
        """
        # Confidence threshold
        min_confidence = self.anomaly_config.get('min_automation_confidence', 0.7)
        if anomaly_analysis['confidence'] < min_confidence:
            return False

        # Severity threshold
        min_severity_score = self.anomaly_config.get('min_severity_score', 0.6)
        if anomaly_analysis['severity_score'] < min_severity_score:
            return False

        # Combined automation confidence
        min_automation_confidence = self.anomaly_config.get('min_automation_confidence_combined', 0.75)
        if anomaly_analysis['automation_confidence'] < min_automation_confidence:
            return False

        # Check for recent similar work orders (avoid duplicates)
        if self._has_recent_similar_work_order(anomaly_analysis):
            return False

        return True

    def _calculate_severity_score(self, anomaly_data: Dict[str, Any]) -> float:
        """Calculate numerical severity score from anomaly data

        Args:
            anomaly_data: Anomaly detection data

        Returns:
            Severity score (0.0 to 1.0)
        """
        severity = anomaly_data.get('severity', 'medium')
        confidence = anomaly_data.get('confidence', 0.5)

        # Base severity mapping
        severity_map = {
            'critical': 0.95,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.3
        }

        base_score = severity_map.get(severity.lower(), 0.5)

        # Adjust by confidence
        adjusted_score = base_score * confidence

        # Consider trend if available
        trend = anomaly_data.get('trend', {})
        if trend.get('direction') == 'increasing':
            adjusted_score *= 1.2
        elif trend.get('direction') == 'decreasing':
            adjusted_score *= 0.9

        return min(adjusted_score, 1.0)

    def _calculate_urgency_score(self, anomaly_data: Dict[str, Any], equipment_id: str) -> float:
        """Calculate urgency score based on time-sensitive factors

        Args:
            anomaly_data: Anomaly data
            equipment_id: Equipment identifier

        Returns:
            Urgency score (0.0 to 1.0)
        """
        urgency_score = 0.5

        # Time of day factor (higher urgency during business hours)
        hour = datetime.now().hour
        if 8 <= hour <= 17:
            urgency_score += 0.2

        # Equipment usage pattern
        equipment_profile = self.equipment_profiles.get(equipment_id)
        if equipment_profile and equipment_profile.business_impact == 'critical':
            urgency_score += 0.3

        # Trend acceleration
        trend = anomaly_data.get('trend', {})
        if trend.get('acceleration', 0) > 0:
            urgency_score += 0.2

        return min(urgency_score, 1.0)

    def _calculate_impact_score(self, anomaly_data: Dict[str, Any], equipment_id: str) -> float:
        """Calculate potential impact score

        Args:
            anomaly_data: Anomaly data
            equipment_id: Equipment identifier

        Returns:
            Impact score (0.0 to 1.0)
        """
        equipment_profile = self.equipment_profiles.get(equipment_id)
        if not equipment_profile:
            return 0.5

        # Business impact
        business_impact_scores = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }

        business_score = business_impact_scores.get(equipment_profile.business_impact, 0.5)

        # Safety impact
        safety_score = business_impact_scores.get(equipment_profile.safety_impact, 0.3)

        # Operational impact
        operational_score = business_impact_scores.get(equipment_profile.operational_impact, 0.5)

        # Redundancy factor
        redundancy_factor = 0.7 if equipment_profile.redundancy_available else 1.0

        # Combined impact
        combined_impact = (business_score * 0.4 + safety_score * 0.4 + operational_score * 0.2) * redundancy_factor

        return min(combined_impact, 1.0)

    def _calculate_automation_confidence(self, anomaly_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence in automation decision

        Args:
            anomaly_analysis: Enhanced anomaly analysis

        Returns:
            Automation confidence score (0.0 to 1.0)
        """
        # Weighted combination of factors
        weights = {
            'detection_confidence': 0.3,
            'severity_score': 0.25,
            'urgency_score': 0.2,
            'impact_score': 0.15,
            'historical_accuracy': 0.1
        }

        factors = {
            'detection_confidence': anomaly_analysis['confidence'],
            'severity_score': anomaly_analysis['severity_score'],
            'urgency_score': anomaly_analysis['urgency_score'],
            'impact_score': anomaly_analysis['impact_score'],
            'historical_accuracy': self._get_historical_accuracy(anomaly_analysis)
        }

        automation_confidence = sum(
            weights[factor] * factors[factor]
            for factor in weights
        )

        return min(automation_confidence, 1.0)

    def _create_enhanced_work_order(self,
                                  anomaly_data: Dict[str, Any],
                                  equipment_id: str,
                                  priority: WorkOrderPriority,
                                  maintenance_type: MaintenanceType,
                                  anomaly_analysis: Dict[str, Any],
                                  equipment_profile: EquipmentCriticalityProfile) -> WorkOrder:
        """Create enhanced work order with full automation metadata

        Args:
            anomaly_data: Original anomaly data
            equipment_id: Equipment identifier
            priority: Calculated priority
            maintenance_type: Determined maintenance type
            anomaly_analysis: Enhanced analysis
            equipment_profile: Equipment criticality profile

        Returns:
            Enhanced work order
        """
        work_order = WorkOrder(
            order_id=self._generate_enhanced_order_id(anomaly_data, equipment_id),
            equipment_id=equipment_id,
            anomaly_id=anomaly_data.get('anomaly_id'),
            type=maintenance_type,
            priority=priority,
            status=WorkOrderStatus.CREATED,
            created_at=datetime.now(),
            description=self._generate_enhanced_description(anomaly_analysis),
            estimated_duration_hours=self._estimate_enhanced_duration(anomaly_analysis, equipment_profile),
            anomaly_details=anomaly_analysis
        )

        # Enhanced SLA calculation
        work_order.sla_deadline = self._calculate_enhanced_sla_deadline(
            priority,
            equipment_profile,
            anomaly_analysis
        )

        # Enhanced cost estimation
        work_order.estimated_cost = self._estimate_enhanced_cost(
            work_order,
            equipment_profile,
            anomaly_analysis
        )

        # Add automation metadata
        work_order.automation_metadata = {
            'created_by_automation': True,
            'automation_confidence': anomaly_analysis['automation_confidence'],
            'priority_reasoning': self._get_priority_reasoning(priority, anomaly_analysis),
            'recommended_skills': self._get_required_skills(anomaly_analysis, equipment_profile),
            'estimated_parts': self._estimate_required_parts(anomaly_analysis, equipment_profile),
            'risk_assessment': self._assess_failure_risk(anomaly_analysis, equipment_profile)
        }

        return work_order

    def _initialize_default_configurations(self):
        """Initialize default automation configurations"""
        # Default anomaly to work order mappings
        self.anomaly_mappings = {
            'temperature_high': AnomalyWorkOrderMapping(
                anomaly_type='temperature_high',
                equipment_type='motor',
                priority_base=WorkOrderPriority.HIGH,
                maintenance_type=MaintenanceType.CORRECTIVE,
                estimated_duration=2.5,
                required_skills=['electrical', 'motor_repair'],
                parts_likely_needed=['cooling_fan', 'thermal_sensor'],
                cost_multiplier=1.2
            ),
            'vibration_excessive': AnomalyWorkOrderMapping(
                anomaly_type='vibration_excessive',
                equipment_type='pump',
                priority_base=WorkOrderPriority.MEDIUM,
                maintenance_type=MaintenanceType.PREDICTIVE,
                estimated_duration=3.0,
                required_skills=['mechanical', 'vibration_analysis'],
                parts_likely_needed=['bearing', 'coupling'],
                cost_multiplier=1.5
            ),
            'pressure_abnormal': AnomalyWorkOrderMapping(
                anomaly_type='pressure_abnormal',
                equipment_type='compressor',
                priority_base=WorkOrderPriority.HIGH,
                maintenance_type=MaintenanceType.CORRECTIVE,
                estimated_duration=4.0,
                required_skills=['pneumatic', 'pressure_systems'],
                parts_likely_needed=['pressure_valve', 'gasket'],
                cost_multiplier=1.3
            )
        }

        # Default equipment criticality profiles
        self._initialize_default_equipment_profiles()

    def _initialize_default_equipment_profiles(self):
        """Initialize default equipment criticality profiles"""
        self.equipment_profiles = {
            'default': EquipmentCriticalityProfile(
                equipment_id='default',
                equipment_type='generic',
                business_impact='medium',
                safety_impact='medium',
                operational_impact='medium',
                downtime_cost_per_hour=500.0,
                replacement_cost=10000.0,
                dependencies=[],
                redundancy_available=False,
                maintenance_complexity='moderate'
            )
        }

    def get_automation_metrics(self) -> Dict[str, Any]:
        """Get automation performance metrics

        Returns:
            Automation metrics
        """
        return self.automation_metrics.copy()

    def update_equipment_profile(self, equipment_id: str, profile: EquipmentCriticalityProfile):
        """Update equipment criticality profile

        Args:
            equipment_id: Equipment identifier
            profile: Updated profile
        """
        self.equipment_profiles[equipment_id] = profile
        logger.info(f"Updated equipment profile for {equipment_id}")


class AnomalyToWorkOrderMapper:
    """Maps anomalies to appropriate work order configurations"""

    def __init__(self):
        """Initialize mapper"""
        self.mapping_rules = {}
        self.learned_mappings = {}

    def get_mapping(self, anomaly_type: str, equipment_type: str) -> Optional[AnomalyWorkOrderMapping]:
        """Get mapping for anomaly and equipment combination

        Args:
            anomaly_type: Type of anomaly detected
            equipment_type: Type of equipment

        Returns:
            Mapping configuration or None
        """
        key = f"{anomaly_type}_{equipment_type}"
        return self.mapping_rules.get(key)


class IntelligentPriorityCalculator:
    """ML-based priority calculation system"""

    def __init__(self):
        """Initialize priority calculator"""
        self.priority_model = None
        self.feature_extractors = {}

    def calculate_priority(self,
                         anomaly_analysis: Dict[str, Any],
                         equipment_profile: EquipmentCriticalityProfile,
                         anomaly_data: Dict[str, Any]) -> WorkOrderPriority:
        """Calculate intelligent priority using ML and business rules

        Args:
            anomaly_analysis: Enhanced anomaly analysis
            equipment_profile: Equipment criticality profile
            anomaly_data: Original anomaly data

        Returns:
            Calculated priority level
        """
        # Extract features for ML model
        features = self._extract_priority_features(
            anomaly_analysis,
            equipment_profile,
            anomaly_data
        )

        # Calculate base priority score
        priority_score = self._calculate_priority_score(features)

        # Apply business rules
        final_priority = self._apply_priority_business_rules(
            priority_score,
            equipment_profile,
            anomaly_analysis
        )

        return final_priority

    def _extract_priority_features(self,
                                 anomaly_analysis: Dict[str, Any],
                                 equipment_profile: EquipmentCriticalityProfile,
                                 anomaly_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for priority calculation

        Args:
            anomaly_analysis: Enhanced anomaly analysis
            equipment_profile: Equipment profile
            anomaly_data: Original anomaly data

        Returns:
            Feature dictionary
        """
        features = {
            'severity_score': anomaly_analysis['severity_score'],
            'urgency_score': anomaly_analysis['urgency_score'],
            'impact_score': anomaly_analysis['impact_score'],
            'confidence': anomaly_analysis['confidence'],
            'business_impact_score': self._map_impact_to_score(equipment_profile.business_impact),
            'safety_impact_score': self._map_impact_to_score(equipment_profile.safety_impact),
            'operational_impact_score': self._map_impact_to_score(equipment_profile.operational_impact),
            'downtime_cost_factor': min(equipment_profile.downtime_cost_per_hour / 1000.0, 1.0),
            'redundancy_factor': 0.3 if equipment_profile.redundancy_available else 1.0,
            'complexity_factor': self._map_complexity_to_score(equipment_profile.maintenance_complexity)
        }

        return features

    def _calculate_priority_score(self, features: Dict[str, float]) -> float:
        """Calculate numerical priority score

        Args:
            features: Extracted features

        Returns:
            Priority score (0.0 to 1.0)
        """
        # Weighted feature combination
        weights = {
            'severity_score': 0.25,
            'urgency_score': 0.2,
            'impact_score': 0.15,
            'confidence': 0.1,
            'business_impact_score': 0.15,
            'safety_impact_score': 0.1,
            'operational_impact_score': 0.05
        }

        score = sum(weights[feature] * features[feature] for feature in weights)

        # Apply modifiers
        score *= features['redundancy_factor']
        score *= features['complexity_factor']

        return min(score, 1.0)

    def _apply_priority_business_rules(self,
                                     priority_score: float,
                                     equipment_profile: EquipmentCriticalityProfile,
                                     anomaly_analysis: Dict[str, Any]) -> WorkOrderPriority:
        """Apply business rules to determine final priority

        Args:
            priority_score: Calculated priority score
            equipment_profile: Equipment profile
            anomaly_analysis: Anomaly analysis

        Returns:
            Final priority level
        """
        # Safety-first rule
        if equipment_profile.safety_impact == 'critical':
            return WorkOrderPriority.CRITICAL

        # High confidence + high impact = critical
        if (anomaly_analysis['confidence'] > 0.9 and
            equipment_profile.business_impact == 'critical'):
            return WorkOrderPriority.CRITICAL

        # Score-based mapping
        if priority_score >= 0.85:
            return WorkOrderPriority.CRITICAL
        elif priority_score >= 0.65:
            return WorkOrderPriority.HIGH
        elif priority_score >= 0.4:
            return WorkOrderPriority.MEDIUM
        else:
            return WorkOrderPriority.LOW

    def _map_impact_to_score(self, impact: str) -> float:
        """Map impact level to numerical score

        Args:
            impact: Impact level string

        Returns:
            Numerical score
        """
        mapping = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
        return mapping.get(impact.lower(), 0.5)

    def _map_complexity_to_score(self, complexity: str) -> float:
        """Map maintenance complexity to score factor

        Args:
            complexity: Complexity level

        Returns:
            Score factor
        """
        mapping = {
            'simple': 0.8,
            'moderate': 1.0,
            'complex': 1.2,
            'specialized': 1.4
        }
        return mapping.get(complexity.lower(), 1.0)


class EquipmentCriticalityAssessor:
    """Assess and manage equipment criticality profiles"""

    def __init__(self):
        """Initialize assessor"""
        self.criticality_cache = {}

    def assess_equipment_criticality(self, equipment_id: str) -> EquipmentCriticalityProfile:
        """Assess equipment criticality

        Args:
            equipment_id: Equipment identifier

        Returns:
            Criticality profile
        """
        # Implementation would include real assessment logic
        # For now, return default profile
        return EquipmentCriticalityProfile(
            equipment_id=equipment_id,
            equipment_type='generic',
            business_impact='medium',
            safety_impact='medium',
            operational_impact='medium',
            downtime_cost_per_hour=500.0,
            replacement_cost=10000.0,
            dependencies=[],
            redundancy_available=False,
            maintenance_complexity='moderate'
        )


class WorkOrderAutomationEngine:
    """Core automation engine for work order processing"""

    def __init__(self):
        """Initialize automation engine"""
        self.automation_rules = {}
        self.processing_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def process_automation_queue(self):
        """Process automation queue in background"""
        while True:
            try:
                task = self.processing_queue.get(timeout=5)
                if task is None:
                    break

                # Process task
                self.executor.submit(self._process_automation_task, task)

            except queue.Empty:
                continue

    def _process_automation_task(self, task: Dict[str, Any]):
        """Process individual automation task

        Args:
            task: Automation task data
        """
        # Implementation for background task processing
        pass