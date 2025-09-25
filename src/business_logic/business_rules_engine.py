"""
Business Rules Engine for IoT Predictive Maintenance System

This module provides a comprehensive business rules engine that integrates
failure classification, health scoring, predictive triggers, and cost-benefit
analysis to automate intelligent maintenance decision-making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

# Local imports
from .failure_classification import FailureClassificationEngine, FailureMode, Severity, FailureClassification
from .equipment_health import EquipmentHealthScorer, SensorHealth, SubsystemHealth, SystemHealth, HealthStatus
from .predictive_triggers import PredictiveMaintenanceTriggerSystem, TriggerEvent, MaintenanceAction, Priority
from .cost_benefit_analysis import CostBenefitAnalyzer, FinancialAnalysis, MaintenanceStrategy
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of maintenance decisions"""
    IMMEDIATE_ACTION = "immediate_action"         # Emergency response required
    SCHEDULE_MAINTENANCE = "schedule_maintenance" # Schedule planned maintenance
    MONITOR_CLOSELY = "monitor_closely"           # Increase monitoring frequency
    CONTINUE_OPERATION = "continue_operation"     # No action needed
    INVESTIGATE_FURTHER = "investigate_further"   # More analysis required
    OPTIMIZE_STRATEGY = "optimize_strategy"       # Change maintenance strategy
    REPLACE_EQUIPMENT = "replace_equipment"       # Equipment replacement recommended


class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    VERY_LOW = "very_low"      # < 40%
    LOW = "low"                # 40-60%
    MEDIUM = "medium"          # 60-80%
    HIGH = "high"              # 80-95%
    VERY_HIGH = "very_high"    # > 95%


@dataclass
class BusinessRule:
    """Business rule definition"""
    rule_id: str
    rule_name: str
    description: str
    category: str                                 # Rule category

    # Rule conditions
    conditions: Dict[str, Any]                    # Condition parameters
    decision_logic: str                           # Python expression for decision

    # Rule outputs
    decision_type: DecisionType
    action_parameters: Dict[str, Any] = field(default_factory=dict)

    # Rule metadata
    priority: int = 1                             # Rule priority (1=highest)
    is_active: bool = True
    confidence_modifier: float = 1.0              # Modifier for decision confidence
    business_impact: str = "medium"               # Business impact level

    # Rule execution tracking
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    success_rate: float = 1.0                     # Historical success rate


@dataclass
class DecisionContext:
    """Context information for decision making"""
    timestamp: datetime

    # Input data
    sensor_data: Dict[str, float]
    anomaly_scores: Dict[str, float]
    time_window_data: Optional[Dict[str, np.ndarray]] = None

    # Analysis results
    failure_classifications: Dict[str, FailureClassification] = field(default_factory=dict)
    sensor_healths: Dict[str, SensorHealth] = field(default_factory=dict)
    subsystem_healths: Dict[str, SubsystemHealth] = field(default_factory=dict)
    system_health: Optional[SystemHealth] = None
    trigger_events: List[TriggerEvent] = field(default_factory=list)

    # External factors
    operational_context: Dict[str, Any] = field(default_factory=dict)
    resource_availability: Dict[str, Any] = field(default_factory=dict)
    business_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaintenanceDecision:
    """Maintenance decision output"""
    decision_id: str
    decision_type: DecisionType
    confidence: float                             # Decision confidence (0-1)
    confidence_level: ConfidenceLevel

    # Decision details
    primary_reason: str
    supporting_evidence: List[str]
    affected_equipment: List[str]
    recommended_actions: List[MaintenanceAction]

    # Economic justification
    estimated_cost: float
    estimated_benefit: float
    roi_projection: float
    payback_period_months: float

    # Implementation details
    urgency_level: Priority
    suggested_timeline: str
    resource_requirements: Dict[str, Any]

    # Risk assessment
    risk_if_deferred: float                       # Risk if decision is delayed
    alternative_options: List[str]

    # Decision metadata
    decision_timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    contributing_rules: List[str] = field(default_factory=list)


@dataclass
class RuleExecutionResult:
    """Result of business rule execution"""
    rule_id: str
    executed: bool
    decision: Optional[MaintenanceDecision]
    confidence: float
    execution_time_ms: float
    error_message: Optional[str] = None


class BusinessRulesEngine:
    """
    Comprehensive business rules engine for predictive maintenance decisions.

    Features:
    - Integration of all analysis components
    - Intelligent decision automation
    - Multi-criteria decision optimization
    - Risk-based decision making
    - Cost-benefit decision optimization
    - Real-time decision execution
    - Decision audit and learning
    """

    def __init__(self):
        self.config = get_config()

        # Initialize analysis components
        self.failure_classifier = FailureClassificationEngine()
        self.health_scorer = EquipmentHealthScorer(self.failure_classifier)
        self.trigger_system = PredictiveMaintenanceTriggerSystem(
            self.failure_classifier, self.health_scorer
        )
        self.cost_analyzer = CostBenefitAnalyzer()

        # Business rules
        self.rules: Dict[str, BusinessRule] = {}
        self.rule_execution_history: List[RuleExecutionResult] = []

        # Decision cache and history
        self.decision_cache: Dict[str, MaintenanceDecision] = {}
        self.decision_history: List[MaintenanceDecision] = []

        # Configuration
        self.max_concurrent_decisions = 10
        self.decision_cache_ttl_hours = 4
        self.min_confidence_threshold = 0.6

        # Initialize default business rules
        self._initialize_default_rules()

        logger.info("Initialized BusinessRulesEngine with integrated analysis components")

    def _initialize_default_rules(self):
        """Initialize default business rules"""

        # Rule 1: Critical Health Emergency Response
        critical_health_rule = BusinessRule(
            rule_id="critical_health_emergency",
            rule_name="Critical Health Emergency Response",
            description="Immediate action for critical equipment health",
            category="emergency_response",
            conditions={
                'min_health_score': 30.0,
                'max_confidence': 0.4,
                'affected_sensor_count': 1
            },
            decision_logic="any(h.health_score < 30 and h.status == HealthStatus.CRITICAL for h in context.sensor_healths.values())",
            decision_type=DecisionType.IMMEDIATE_ACTION,
            action_parameters={
                'shutdown_required': True,
                'emergency_team': True,
                'escalation_level': 'executive'
            },
            priority=1,
            confidence_modifier=1.2,
            business_impact="critical"
        )

        # Rule 2: Cascade Failure Prevention
        cascade_prevention_rule = BusinessRule(
            rule_id="cascade_failure_prevention",
            rule_name="Cascade Failure Prevention",
            description="Prevent cascade failures in interconnected systems",
            category="failure_prevention",
            conditions={
                'cascade_risk_threshold': 0.3,
                'affected_subsystems': 2,
                'time_window_hours': 24
            },
            decision_logic="context.system_health and context.system_health.cascade_failure_risk > 0.3",
            decision_type=DecisionType.SCHEDULE_MAINTENANCE,
            action_parameters={
                'coordination_required': True,
                'subsystem_isolation': True
            },
            priority=2,
            confidence_modifier=1.1,
            business_impact="high"
        )

        # Rule 3: Cost-Optimized Maintenance Timing
        cost_optimization_rule = BusinessRule(
            rule_id="cost_optimized_timing",
            rule_name="Cost-Optimized Maintenance Timing",
            description="Optimize maintenance timing based on cost-benefit analysis",
            category="cost_optimization",
            conditions={
                'min_roi': 20.0,
                'max_payback_months': 18,
                'health_threshold': 70.0
            },
            decision_logic="len([h for h in context.sensor_healths.values() if 50 < h.health_score < 70]) >= 3",
            decision_type=DecisionType.OPTIMIZE_STRATEGY,
            action_parameters={
                'strategy_analysis_required': True,
                'timing_optimization': True
            },
            priority=3,
            confidence_modifier=1.0,
            business_impact="medium"
        )

        # Rule 4: Predictive Trigger Integration
        predictive_integration_rule = BusinessRule(
            rule_id="predictive_trigger_integration",
            rule_name="Predictive Trigger Integration",
            description="Integrate predictive trigger recommendations",
            category="predictive_maintenance",
            conditions={
                'trigger_confidence': 0.8,
                'priority_threshold': Priority.HIGH
            },
            decision_logic="len([e for e in context.trigger_events if e.confidence > 0.8]) > 0",
            decision_type=DecisionType.SCHEDULE_MAINTENANCE,
            action_parameters={
                'follow_trigger_recommendations': True,
                'resource_allocation': 'automatic'
            },
            priority=4,
            confidence_modifier=0.9,
            business_impact="medium"
        )

        # Rule 5: Performance Degradation Response
        performance_degradation_rule = BusinessRule(
            rule_id="performance_degradation_response",
            rule_name="Performance Degradation Response",
            description="Respond to gradual performance degradation",
            category="performance_management",
            conditions={
                'performance_threshold': 0.85,
                'trend_period_days': 7,
                'degradation_rate': 0.05
            },
            decision_logic="context.system_health and context.system_health.performance_efficiency < 0.85",
            decision_type=DecisionType.INVESTIGATE_FURTHER,
            action_parameters={
                'performance_analysis': True,
                'root_cause_investigation': True
            },
            priority=5,
            confidence_modifier=0.8,
            business_impact="medium"
        )

        # Rule 6: Resource-Constrained Decision Making
        resource_constraint_rule = BusinessRule(
            rule_id="resource_constrained_decisions",
            rule_name="Resource-Constrained Decision Making",
            description="Make decisions under resource constraints",
            category="resource_management",
            conditions={
                'available_technicians': 2,
                'budget_threshold': 10000,
                'maintenance_backlog': 5
            },
            decision_logic="context.resource_availability.get('technicians', 0) < 2",
            decision_type=DecisionType.MONITOR_CLOSELY,
            action_parameters={
                'defer_non_critical': True,
                'prioritize_by_risk': True
            },
            priority=6,
            confidence_modifier=0.7,
            business_impact="low"
        )

        # Rule 7: Equipment Replacement Economics
        replacement_economics_rule = BusinessRule(
            rule_id="equipment_replacement_economics",
            rule_name="Equipment Replacement Economics",
            description="Recommend equipment replacement based on economics",
            category="asset_management",
            conditions={
                'age_threshold_years': 10,
                'maintenance_cost_ratio': 0.4,
                'reliability_threshold': 0.6
            },
            decision_logic="any(h.health_score < 40 and h.maintenance_urgency >= 4 for h in context.sensor_healths.values())",
            decision_type=DecisionType.REPLACE_EQUIPMENT,
            action_parameters={
                'replacement_analysis': True,
                'lifecycle_assessment': True
            },
            priority=7,
            confidence_modifier=0.9,
            business_impact="high"
        )

        # Rule 8: Regulatory Compliance
        compliance_rule = BusinessRule(
            rule_id="regulatory_compliance",
            rule_name="Regulatory Compliance Maintenance",
            description="Ensure regulatory compliance requirements",
            category="compliance",
            conditions={
                'compliance_deadline_days': 30,
                'safety_criticality': 4,
                'certification_required': True
            },
            decision_logic="any(h.maintenance_urgency >= 3 for h in context.sensor_healths.values() if 'safety' in h.sensor_id.lower())",
            decision_type=DecisionType.SCHEDULE_MAINTENANCE,
            action_parameters={
                'compliance_priority': True,
                'documentation_required': True
            },
            priority=2,  # High priority for compliance
            confidence_modifier=1.1,
            business_impact="high"
        )

        # Add rules to engine
        rules = [
            critical_health_rule, cascade_prevention_rule, cost_optimization_rule,
            predictive_integration_rule, performance_degradation_rule,
            resource_constraint_rule, replacement_economics_rule, compliance_rule
        ]

        for rule in rules:
            self.rules[rule.rule_id] = rule

    async def make_maintenance_decision(self, context: DecisionContext) -> List[MaintenanceDecision]:
        """
        Make comprehensive maintenance decisions based on all available data

        Args:
            context: Decision context with sensor data and analysis results

        Returns:
            List of maintenance decisions ordered by priority
        """
        logger.info("Making maintenance decisions based on current context")

        # Step 1: Perform comprehensive analysis if not already done
        await self._ensure_complete_analysis(context)

        # Step 2: Execute business rules
        rule_results = await self._execute_business_rules(context)

        # Step 3: Consolidate and prioritize decisions
        decisions = self._consolidate_decisions(rule_results, context)

        # Step 4: Validate decisions
        validated_decisions = self._validate_decisions(decisions, context)

        # Step 5: Cache and log decisions
        self._cache_decisions(validated_decisions)
        self._log_decisions(validated_decisions, context)

        logger.info(f"Generated {len(validated_decisions)} maintenance decisions")

        return validated_decisions

    async def _ensure_complete_analysis(self, context: DecisionContext):
        """Ensure all required analysis components are complete"""

        # Failure classification
        if not context.failure_classifications:
            logger.info("Performing failure classification analysis")
            context.failure_classifications = self.failure_classifier.classify_failure(
                context.sensor_data, context.anomaly_scores, context.time_window_data
            )

        # Health scoring
        if not context.sensor_healths:
            logger.info("Performing health scoring analysis")
            for sensor_id in context.sensor_data.keys():
                sensor_health = self.health_scorer.calculate_sensor_health(
                    sensor_id, context.sensor_data, context.anomaly_scores, context.time_window_data
                )
                context.sensor_healths[sensor_id] = sensor_health

        # Subsystem health calculation
        if not context.subsystem_healths:
            logger.info("Calculating subsystem health")
            # Group sensors by subsystem
            subsystem_sensors = {}
            for sensor_id, health in context.sensor_healths.items():
                if sensor_id in self.failure_classifier.sensor_mappings:
                    mapping = self.failure_classifier.sensor_mappings[sensor_id]
                    subsystem = mapping.subsystem
                    if subsystem not in subsystem_sensors:
                        subsystem_sensors[subsystem] = {}
                    subsystem_sensors[subsystem][sensor_id] = health

            # Calculate subsystem health for each subsystem
            for subsystem, sensors in subsystem_sensors.items():
                subsystem_health = self.health_scorer.calculate_subsystem_health(
                    subsystem, sensors, context.operational_context
                )
                context.subsystem_healths[subsystem] = subsystem_health

        # System health calculation
        if not context.system_health and context.subsystem_healths:
            logger.info("Calculating system health")
            system_metrics = {
                'availability': context.operational_context.get('availability', 1.0),
                'performance_efficiency': context.operational_context.get('performance_efficiency', 1.0)
            }
            context.system_health = self.health_scorer.calculate_system_health(
                context.subsystem_healths, system_metrics
            )

        # Trigger evaluation
        if not context.trigger_events:
            logger.info("Evaluating maintenance triggers")
            context.trigger_events = self.trigger_system.evaluate_triggers(
                context.sensor_data, context.anomaly_scores, context.sensor_healths, context.time_window_data
            )

    async def _execute_business_rules(self, context: DecisionContext) -> List[RuleExecutionResult]:
        """Execute all active business rules"""

        rule_results = []

        # Sort rules by priority
        sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority)

        for rule in sorted_rules:
            if not rule.is_active:
                continue

            start_time = datetime.now()

            try:
                # Evaluate rule conditions
                rule_triggered = self._evaluate_rule_conditions(rule, context)

                if rule_triggered:
                    # Generate decision
                    decision = self._generate_decision_from_rule(rule, context)

                    # Calculate execution time
                    execution_time = (datetime.now() - start_time).total_seconds() * 1000

                    # Update rule statistics
                    rule.execution_count += 1
                    rule.last_executed = datetime.now()

                    result = RuleExecutionResult(
                        rule_id=rule.rule_id,
                        executed=True,
                        decision=decision,
                        confidence=decision.confidence if decision else 0.0,
                        execution_time_ms=execution_time
                    )

                    rule_results.append(result)
                    logger.info(f"Rule '{rule.rule_name}' generated decision: {decision.decision_type.value}")

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                error_result = RuleExecutionResult(
                    rule_id=rule.rule_id,
                    executed=False,
                    decision=None,
                    confidence=0.0,
                    execution_time_ms=execution_time,
                    error_message=str(e)
                )
                rule_results.append(error_result)
                logger.error(f"Error executing rule '{rule.rule_name}': {e}")

        return rule_results

    def _evaluate_rule_conditions(self, rule: BusinessRule, context: DecisionContext) -> bool:
        """Evaluate if rule conditions are met"""

        try:
            # Create evaluation environment
            eval_env = {
                'context': context,
                'HealthStatus': HealthStatus,
                'Severity': Severity,
                'Priority': Priority,
                'DecisionType': DecisionType,
                'any': any,
                'all': all,
                'len': len,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs
            }

            # Evaluate the decision logic expression
            result = eval(rule.decision_logic, eval_env)
            return bool(result)

        except Exception as e:
            logger.error(f"Error evaluating rule conditions for '{rule.rule_name}': {e}")
            return False

    def _generate_decision_from_rule(self, rule: BusinessRule, context: DecisionContext) -> MaintenanceDecision:
        """Generate maintenance decision from triggered rule"""

        decision_id = f"decision_{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine affected equipment
        affected_equipment = []
        if context.failure_classifications:
            affected_equipment.extend(list(context.failure_classifications.keys()))
        if context.trigger_events:
            for event in context.trigger_events:
                affected_equipment.extend(event.triggering_sensors)
        affected_equipment = list(set(affected_equipment))  # Remove duplicates

        # Determine confidence based on rule and data quality
        base_confidence = self._calculate_rule_confidence(rule, context)
        confidence = base_confidence * rule.confidence_modifier
        confidence = max(0.0, min(1.0, confidence))

        # Determine confidence level
        confidence_level = self._determine_confidence_level(confidence)

        # Generate supporting evidence
        supporting_evidence = self._generate_supporting_evidence(rule, context)

        # Determine recommended actions
        recommended_actions = self._determine_recommended_actions(rule, context)

        # Economic analysis
        estimated_cost, estimated_benefit, roi, payback = self._calculate_decision_economics(
            rule, context, recommended_actions
        )

        # Determine urgency and timeline
        urgency_level, timeline = self._determine_urgency_and_timeline(rule, context)

        # Risk assessment
        risk_if_deferred = self._calculate_deferral_risk(rule, context)

        # Alternative options
        alternatives = self._generate_alternative_options(rule, context)

        # Resource requirements
        resource_requirements = self._calculate_resource_requirements(recommended_actions)

        # Set expiration
        expires_at = datetime.now() + timedelta(hours=self._get_decision_validity_hours(rule))

        return MaintenanceDecision(
            decision_id=decision_id,
            decision_type=rule.decision_type,
            confidence=confidence,
            confidence_level=confidence_level,
            primary_reason=rule.description,
            supporting_evidence=supporting_evidence,
            affected_equipment=affected_equipment,
            recommended_actions=recommended_actions,
            estimated_cost=estimated_cost,
            estimated_benefit=estimated_benefit,
            roi_projection=roi,
            payback_period_months=payback,
            urgency_level=urgency_level,
            suggested_timeline=timeline,
            resource_requirements=resource_requirements,
            risk_if_deferred=risk_if_deferred,
            alternative_options=alternatives,
            expires_at=expires_at,
            contributing_rules=[rule.rule_id]
        )

    def _calculate_rule_confidence(self, rule: BusinessRule, context: DecisionContext) -> float:
        """Calculate base confidence for rule execution"""

        confidence_factors = []

        # Data quality factor
        data_completeness = len(context.sensor_data) / 80  # Assuming 80 sensors total
        confidence_factors.append(data_completeness)

        # Health score confidence
        if context.sensor_healths:
            health_confidences = [h.metrics.confidence for h in context.sensor_healths.values()]
            avg_health_confidence = np.mean(health_confidences) if health_confidences else 0.5
            confidence_factors.append(avg_health_confidence)

        # Trigger event confidence
        if context.trigger_events:
            trigger_confidences = [e.confidence for e in context.trigger_events]
            avg_trigger_confidence = np.mean(trigger_confidences) if trigger_confidences else 0.5
            confidence_factors.append(avg_trigger_confidence)
        else:
            confidence_factors.append(0.7)  # Default when no triggers

        # Rule historical success rate
        confidence_factors.append(rule.success_rate)

        # Calculate weighted average
        base_confidence = np.mean(confidence_factors)

        return base_confidence

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level"""
        if confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_supporting_evidence(self, rule: BusinessRule, context: DecisionContext) -> List[str]:
        """Generate supporting evidence for decision"""

        evidence = []

        # Health score evidence
        if context.sensor_healths:
            critical_sensors = [sid for sid, h in context.sensor_healths.items() if h.health_score < 50]
            if critical_sensors:
                evidence.append(f"Critical health detected in {len(critical_sensors)} sensors: {', '.join(critical_sensors[:3])}")

        # Anomaly evidence
        if context.anomaly_scores:
            high_anomalies = [sid for sid, score in context.anomaly_scores.items() if score > 0.8]
            if high_anomalies:
                evidence.append(f"High anomaly scores detected in sensors: {', '.join(high_anomalies[:3])}")

        # Failure classification evidence
        if context.failure_classifications:
            failure_modes = set()
            for classification in context.failure_classifications.values():
                failure_modes.update([fm.value for fm in classification.failure_modes])
            if failure_modes:
                evidence.append(f"Predicted failure modes: {', '.join(list(failure_modes)[:3])}")

        # Trigger event evidence
        if context.trigger_events:
            urgent_triggers = [e for e in context.trigger_events if any(a.priority.value >= 4 for a in e.recommended_actions)]
            if urgent_triggers:
                evidence.append(f"Urgent maintenance triggers detected: {len(urgent_triggers)} events")

        # System health evidence
        if context.system_health:
            if context.system_health.overall_health_score < 70:
                evidence.append(f"System health score: {context.system_health.overall_health_score:.1f}%")

        return evidence

    def _determine_recommended_actions(self, rule: BusinessRule, context: DecisionContext) -> List[MaintenanceAction]:
        """Determine recommended maintenance actions"""

        actions = []

        # Get actions from trigger events
        for event in context.trigger_events:
            actions.extend(event.recommended_actions)

        # Add rule-specific actions based on decision type
        if rule.decision_type == DecisionType.IMMEDIATE_ACTION:
            # Emergency actions would be defined in maintenance action database
            pass  # Actions from triggers should cover this

        elif rule.decision_type == DecisionType.SCHEDULE_MAINTENANCE:
            # Scheduled maintenance actions
            pass  # Actions from triggers should cover this

        # Remove duplicates
        unique_actions = []
        seen_ids = set()
        for action in actions:
            if action.action_id not in seen_ids:
                unique_actions.append(action)
                seen_ids.add(action.action_id)

        return unique_actions

    def _calculate_decision_economics(self, rule: BusinessRule, context: DecisionContext,
                                    actions: List[MaintenanceAction]) -> Tuple[float, float, float, float]:
        """Calculate economic impact of decision"""

        # Sum up action costs
        estimated_cost = sum(action.estimated_cost for action in actions)

        # Estimate benefits based on failure prevention and performance improvement
        estimated_benefit = 0.0

        # Calculate benefit from avoided failures
        if context.failure_classifications:
            for classification in context.failure_classifications.values():
                # Estimate benefit based on failure severity
                if classification.severity == Severity.CRITICAL:
                    estimated_benefit += 50000  # High cost avoidance
                elif classification.severity == Severity.HIGH:
                    estimated_benefit += 25000
                elif classification.severity == Severity.MEDIUM:
                    estimated_benefit += 10000

        # Calculate benefit from health improvement
        if context.sensor_healths:
            poor_health_sensors = [h for h in context.sensor_healths.values() if h.health_score < 60]
            estimated_benefit += len(poor_health_sensors) * 5000  # Benefit per improved sensor

        # ROI calculation
        roi = ((estimated_benefit - estimated_cost) / estimated_cost * 100) if estimated_cost > 0 else 0

        # Payback period (simplified)
        payback = (estimated_cost / (estimated_benefit / 12)) if estimated_benefit > 0 else float('inf')

        return estimated_cost, estimated_benefit, roi, payback

    def _determine_urgency_and_timeline(self, rule: BusinessRule, context: DecisionContext) -> Tuple[Priority, str]:
        """Determine urgency level and suggested timeline"""

        # Base urgency on rule decision type
        if rule.decision_type == DecisionType.IMMEDIATE_ACTION:
            urgency = Priority.CRITICAL
            timeline = "Immediate (within 2 hours)"
        elif rule.decision_type == DecisionType.SCHEDULE_MAINTENANCE:
            # Determine based on health scores and trigger events
            if any(h.health_score < 40 for h in context.sensor_healths.values()):
                urgency = Priority.URGENT
                timeline = "Within 24 hours"
            elif any(h.health_score < 60 for h in context.sensor_healths.values()):
                urgency = Priority.HIGH
                timeline = "Within 1 week"
            else:
                urgency = Priority.MEDIUM
                timeline = "Within 1 month"
        else:
            urgency = Priority.LOW
            timeline = "As resources permit"

        return urgency, timeline

    def _calculate_deferral_risk(self, rule: BusinessRule, context: DecisionContext) -> float:
        """Calculate risk of deferring the decision"""

        risk_factors = []

        # Health degradation risk
        if context.sensor_healths:
            critical_count = sum(1 for h in context.sensor_healths.values() if h.health_score < 40)
            risk_factors.append(critical_count / len(context.sensor_healths))

        # Cascade failure risk
        if context.system_health:
            risk_factors.append(context.system_health.cascade_failure_risk)

        # Trigger urgency risk
        if context.trigger_events:
            urgent_events = sum(1 for e in context.trigger_events
                              if any(a.priority.value >= 4 for a in e.recommended_actions))
            risk_factors.append(urgent_events / max(1, len(context.trigger_events)))

        # Business impact multiplier
        impact_multipliers = {"critical": 2.0, "high": 1.5, "medium": 1.0, "low": 0.5}
        impact_multiplier = impact_multipliers.get(rule.business_impact, 1.0)

        base_risk = np.mean(risk_factors) if risk_factors else 0.1
        return min(1.0, base_risk * impact_multiplier)

    def _generate_alternative_options(self, rule: BusinessRule, context: DecisionContext) -> List[str]:
        """Generate alternative decision options"""

        alternatives = []

        if rule.decision_type == DecisionType.IMMEDIATE_ACTION:
            alternatives.extend([
                "Controlled shutdown for inspection",
                "Operate with enhanced monitoring",
                "Emergency repair in place"
            ])
        elif rule.decision_type == DecisionType.SCHEDULE_MAINTENANCE:
            alternatives.extend([
                "Defer to next scheduled outage",
                "Partial maintenance in phases",
                "Outsource to specialist contractor"
            ])
        elif rule.decision_type == DecisionType.REPLACE_EQUIPMENT:
            alternatives.extend([
                "Major overhaul instead of replacement",
                "Lease replacement equipment",
                "Gradual component replacement"
            ])

        return alternatives

    def _calculate_resource_requirements(self, actions: List[MaintenanceAction]) -> Dict[str, Any]:
        """Calculate resource requirements for actions"""

        total_hours = sum(action.estimated_duration_hours for action in actions)
        max_technicians = max([action.required_technicians for action in actions], default=1)

        all_skills = []
        all_parts = []
        for action in actions:
            all_skills.extend(action.required_skills)
            all_parts.extend(action.required_parts)

        return {
            'total_duration_hours': total_hours,
            'peak_technicians_needed': max_technicians,
            'required_skills': list(set(all_skills)),
            'required_parts': list(set(all_parts)),
            'shutdown_required': any(action.requires_system_shutdown for action in actions)
        }

    def _get_decision_validity_hours(self, rule: BusinessRule) -> int:
        """Get decision validity period based on rule type"""

        validity_map = {
            DecisionType.IMMEDIATE_ACTION: 2,
            DecisionType.SCHEDULE_MAINTENANCE: 24,
            DecisionType.MONITOR_CLOSELY: 72,
            DecisionType.CONTINUE_OPERATION: 168,
            DecisionType.INVESTIGATE_FURTHER: 48,
            DecisionType.OPTIMIZE_STRATEGY: 720,
            DecisionType.REPLACE_EQUIPMENT: 2160
        }

        return validity_map.get(rule.decision_type, 24)

    def _consolidate_decisions(self, rule_results: List[RuleExecutionResult],
                              context: DecisionContext) -> List[MaintenanceDecision]:
        """Consolidate multiple rule results into prioritized decisions"""

        # Extract successful decisions
        decisions = [result.decision for result in rule_results
                    if result.executed and result.decision is not None]

        # Remove low-confidence decisions
        decisions = [d for d in decisions if d.confidence >= self.min_confidence_threshold]

        # Group similar decisions
        grouped_decisions = self._group_similar_decisions(decisions)

        # Prioritize decisions
        prioritized_decisions = self._prioritize_decisions(grouped_decisions)

        return prioritized_decisions

    def _group_similar_decisions(self, decisions: List[MaintenanceDecision]) -> List[MaintenanceDecision]:
        """Group and merge similar decisions"""

        # Simple grouping by decision type for now
        # In a full implementation, this would use more sophisticated similarity analysis

        grouped = {}
        for decision in decisions:
            key = (decision.decision_type, tuple(sorted(decision.affected_equipment)))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(decision)

        # Merge decisions in each group
        merged_decisions = []
        for group in grouped.values():
            if len(group) == 1:
                merged_decisions.append(group[0])
            else:
                merged_decision = self._merge_decisions(group)
                merged_decisions.append(merged_decision)

        return merged_decisions

    def _merge_decisions(self, decisions: List[MaintenanceDecision]) -> MaintenanceDecision:
        """Merge multiple similar decisions into one"""

        # Use the highest confidence decision as base
        base_decision = max(decisions, key=lambda d: d.confidence)

        # Combine evidence and actions
        all_evidence = []
        all_actions = []
        all_rules = []

        for decision in decisions:
            all_evidence.extend(decision.supporting_evidence)
            all_actions.extend(decision.recommended_actions)
            all_rules.extend(decision.contributing_rules)

        # Remove duplicates
        unique_evidence = list(set(all_evidence))
        unique_actions = []
        seen_action_ids = set()
        for action in all_actions:
            if action.action_id not in seen_action_ids:
                unique_actions.append(action)
                seen_action_ids.add(action.action_id)

        # Update the base decision
        base_decision.supporting_evidence = unique_evidence
        base_decision.recommended_actions = unique_actions
        base_decision.contributing_rules = list(set(all_rules))

        # Adjust confidence (average of all decisions)
        base_decision.confidence = np.mean([d.confidence for d in decisions])

        return base_decision

    def _prioritize_decisions(self, decisions: List[MaintenanceDecision]) -> List[MaintenanceDecision]:
        """Prioritize decisions by urgency, confidence, and business impact"""

        def decision_priority_score(decision: MaintenanceDecision) -> float:
            # Priority score based on multiple factors
            urgency_weight = decision.urgency_level.value * 0.4
            confidence_weight = decision.confidence * 0.3
            roi_weight = min(1.0, max(0.0, decision.roi_projection / 100)) * 0.2
            risk_weight = decision.risk_if_deferred * 0.1

            return urgency_weight + confidence_weight + roi_weight + risk_weight

        # Sort by priority score (descending)
        prioritized = sorted(decisions, key=decision_priority_score, reverse=True)

        # Limit to maximum concurrent decisions
        return prioritized[:self.max_concurrent_decisions]

    def _validate_decisions(self, decisions: List[MaintenanceDecision],
                           context: DecisionContext) -> List[MaintenanceDecision]:
        """Validate decisions for consistency and feasibility"""

        validated_decisions = []

        for decision in decisions:
            # Check resource constraints
            if self._check_resource_feasibility(decision, context):
                # Check business constraints
                if self._check_business_constraints(decision, context):
                    validated_decisions.append(decision)
                else:
                    logger.warning(f"Decision {decision.decision_id} failed business constraint validation")
            else:
                logger.warning(f"Decision {decision.decision_id} failed resource feasibility check")

        return validated_decisions

    def _check_resource_feasibility(self, decision: MaintenanceDecision, context: DecisionContext) -> bool:
        """Check if decision is feasible given resource constraints"""

        resource_reqs = decision.resource_requirements
        available_resources = context.resource_availability

        # Check technician availability
        required_techs = resource_reqs.get('peak_technicians_needed', 1)
        available_techs = available_resources.get('technicians', 0)

        if required_techs > available_techs:
            return False

        # Check budget constraints
        if decision.estimated_cost > available_resources.get('budget', float('inf')):
            return False

        return True

    def _check_business_constraints(self, decision: MaintenanceDecision, context: DecisionContext) -> bool:
        """Check if decision meets business constraints"""

        constraints = context.business_constraints

        # Check minimum ROI requirement
        min_roi = constraints.get('min_roi', 0)
        if decision.roi_projection < min_roi:
            return False

        # Check maximum payback period
        max_payback = constraints.get('max_payback_months', float('inf'))
        if decision.payback_period_months > max_payback:
            return False

        return True

    def _cache_decisions(self, decisions: List[MaintenanceDecision]):
        """Cache decisions for future reference"""

        for decision in decisions:
            self.decision_cache[decision.decision_id] = decision

        # Clean up expired decisions
        current_time = datetime.now()
        expired_ids = [
            decision_id for decision_id, decision in self.decision_cache.items()
            if decision.expires_at and decision.expires_at < current_time
        ]

        for decision_id in expired_ids:
            del self.decision_cache[decision_id]

    def _log_decisions(self, decisions: List[MaintenanceDecision], context: DecisionContext):
        """Log decisions for audit and learning"""

        self.decision_history.extend(decisions)

        # Keep only recent history (last 1000 decisions)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        # Log to file/database in production
        for decision in decisions:
            logger.info(
                f"Decision: {decision.decision_type.value} | "
                f"Confidence: {decision.confidence:.2f} | "
                f"Equipment: {', '.join(decision.affected_equipment[:3])} | "
                f"Urgency: {decision.urgency_level.value}"
            )

    def get_active_decisions(self) -> List[MaintenanceDecision]:
        """Get currently active (non-expired) decisions"""

        current_time = datetime.now()
        active_decisions = [
            decision for decision in self.decision_cache.values()
            if not decision.expires_at or decision.expires_at > current_time
        ]

        return sorted(active_decisions, key=lambda d: d.urgency_level.value, reverse=True)

    def add_custom_rule(self, rule: BusinessRule) -> bool:
        """Add a custom business rule"""

        if rule.rule_id in self.rules:
            logger.warning(f"Rule {rule.rule_id} already exists")
            return False

        self.rules[rule.rule_id] = rule
        logger.info(f"Added custom rule: {rule.rule_name}")
        return True

    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""

        total_rules = len(self.rules)
        active_rules = sum(1 for rule in self.rules.values() if rule.is_active)

        # Recent execution statistics
        recent_executions = [
            result for result in self.rule_execution_history
            if result.execution_time_ms is not None
        ][-100:]  # Last 100 executions

        avg_execution_time = np.mean([r.execution_time_ms for r in recent_executions]) if recent_executions else 0
        success_rate = len([r for r in recent_executions if r.executed]) / len(recent_executions) if recent_executions else 0

        # Decision statistics
        recent_decisions = [
            d for d in self.decision_history
            if (datetime.now() - d.decision_timestamp).total_seconds() < 24 * 3600
        ]

        decision_types = {}
        for decision in recent_decisions:
            dt = decision.decision_type.value
            decision_types[dt] = decision_types.get(dt, 0) + 1

        return {
            'total_rules': total_rules,
            'active_rules': active_rules,
            'avg_execution_time_ms': avg_execution_time,
            'rule_success_rate': success_rate,
            'decisions_24h': len(recent_decisions),
            'decision_type_breakdown': decision_types,
            'active_decisions': len(self.get_active_decisions()),
            'system_status': 'operational'
        }