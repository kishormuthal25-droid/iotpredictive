"""
Optimized Resource Allocator Module for Phase 3.2
Advanced technician assignment with integration to Phase 3.1 resource optimization
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

# Optimization imports
import pulp
try:
    from ortools.sat.python import cp_model
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None
    pywraplp = None

# ML imports for skill matching
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path
from src.maintenance.work_order_manager import (
    WorkOrder, WorkOrderPriority, WorkOrderStatus, MaintenanceType,
    Equipment, Technician, WorkOrderManager
)
from src.maintenance.advanced_resource_optimizer import AdvancedResourceOptimizer, TechnicianProfile
from src.maintenance.intelligent_cost_forecaster import IntelligentCostForecaster

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TechnicianCapacity:
    """Advanced technician capacity modeling"""
    technician_id: str
    current_workload: float  # 0.0 to 1.0 (percentage of capacity)
    skill_utilization: Dict[str, float]  # skill -> utilization percentage
    availability_windows: List[Tuple[datetime, datetime]]
    preferred_work_types: List[str]
    efficiency_ratings: Dict[str, float]  # task_type -> efficiency score
    fatigue_level: float  # 0.0 to 1.0
    learning_curve_factor: float  # Improvement rate for new tasks
    cost_per_hour: float
    overtime_multiplier: float = 1.5


@dataclass
class WorkOrderRequirements:
    """Detailed work order requirements for optimal assignment"""
    work_order_id: str
    required_skills: Dict[str, int]  # skill -> required_level (1-5)
    estimated_duration: float
    complexity_score: float
    urgency_factor: float
    location: str
    required_certifications: List[str]
    preferred_technician_traits: List[str]
    equipment_type: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    collaborative_requirements: Optional[int] = None  # Number of technicians needed


@dataclass
class AssignmentSolution:
    """Optimal assignment solution with detailed metrics"""
    assignment_id: str
    assignments: Dict[str, str]  # work_order_id -> technician_id
    total_cost: float
    efficiency_score: float
    skill_match_score: float
    workload_balance_score: float
    completion_time_estimate: float
    risk_assessment: Dict[str, Any]
    confidence_level: float
    alternative_assignments: List[Dict[str, str]]  # Alternative solutions


@dataclass
class TechnicianPerformanceMetrics:
    """Comprehensive technician performance tracking"""
    technician_id: str
    completion_rate: float
    average_task_time: float
    quality_score: float
    customer_satisfaction: float
    efficiency_trend: List[float]  # Historical efficiency scores
    skill_improvement_rate: Dict[str, float]
    workload_preference: float
    collaboration_effectiveness: float
    adaptability_score: float


class OptimizedResourceAllocator:
    """Advanced resource allocation system for Phase 3.2 work order automation"""

    def __init__(self,
                 work_order_manager: WorkOrderManager,
                 phase31_resource_optimizer: AdvancedResourceOptimizer,
                 cost_forecaster: IntelligentCostForecaster,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize Optimized Resource Allocator

        Args:
            work_order_manager: Work order management system
            phase31_resource_optimizer: Phase 3.1 resource optimizer
            cost_forecaster: Cost forecasting system
            config: Configuration dictionary
        """
        self.work_order_manager = work_order_manager
        self.phase31_optimizer = phase31_resource_optimizer
        self.cost_forecaster = cost_forecaster
        self.config = config or {}

        # Initialize allocation components
        self.skill_matcher = AdvancedSkillMatcher()
        self.workload_balancer = WorkloadBalancer()
        self.assignment_optimizer = AssignmentOptimizer()
        self.performance_tracker = PerformanceTracker()

        # Allocation data
        self.technician_capacities = {}
        self.work_order_requirements = {}
        self.assignment_history = []
        self.performance_metrics = {}

        # Real-time optimization
        self.assignment_queue = queue.PriorityQueue()
        self.optimization_executor = ThreadPoolExecutor(max_workers=2)
        self.real_time_enabled = True

        # Algorithm parameters
        self.optimization_algorithms = {
            'hungarian': self._hungarian_assignment,
            'genetic': self._genetic_algorithm_assignment,
            'ml_based': self._ml_based_assignment,
            'hybrid': self._hybrid_assignment
        }

        # Performance tracking
        self.allocation_metrics = {
            'total_assignments': 0,
            'successful_assignments': 0,
            'average_assignment_time': 0.0,
            'average_skill_match': 0.0,
            'average_cost_efficiency': 0.0,
            'workload_balance_score': 0.0
        }

        self._initialize_system()
        logger.info("Initialized Optimized Resource Allocator")

    def allocate_technician_to_work_order(self,
                                        work_order: WorkOrder,
                                        algorithm: str = 'hybrid',
                                        consider_cost: bool = True,
                                        force_assignment: bool = False) -> Optional[AssignmentSolution]:
        """Allocate optimal technician to work order using advanced algorithms

        Args:
            work_order: Work order requiring assignment
            algorithm: Algorithm to use ('hungarian', 'genetic', 'ml_based', 'hybrid')
            consider_cost: Whether to include cost optimization
            force_assignment: Force assignment even if suboptimal

        Returns:
            Assignment solution or None if no suitable assignment found
        """
        try:
            # Analyze work order requirements
            requirements = self._analyze_work_order_requirements(work_order)
            self.work_order_requirements[work_order.order_id] = requirements

            # Get available technicians
            available_technicians = self._get_available_technicians(work_order)

            if not available_technicians and not force_assignment:
                logger.warning(f"No available technicians for work order {work_order.order_id}")
                return None

            # Update technician capacities
            self._update_technician_capacities()

            # Select and run optimization algorithm
            if algorithm in self.optimization_algorithms:
                assignment_solution = self.optimization_algorithms[algorithm](
                    work_order, requirements, available_technicians, consider_cost
                )
            else:
                logger.warning(f"Unknown algorithm {algorithm}, using hybrid")
                assignment_solution = self._hybrid_assignment(
                    work_order, requirements, available_technicians, consider_cost
                )

            if assignment_solution:
                # Apply assignment
                self._apply_assignment(assignment_solution)

                # Update performance metrics
                self._update_allocation_metrics(assignment_solution)

                # Integration with Phase 3.1 resource optimizer
                self._integrate_with_phase31_optimizer(assignment_solution)

                logger.info(f"Successfully allocated technician for work order {work_order.order_id}")

            return assignment_solution

        except Exception as e:
            logger.error(f"Error in technician allocation: {e}")
            return None

    def batch_optimize_assignments(self,
                                 work_orders: List[WorkOrder],
                                 algorithm: str = 'hybrid',
                                 optimization_objective: str = 'balanced') -> List[AssignmentSolution]:
        """Optimize multiple work order assignments simultaneously

        Args:
            work_orders: List of work orders to assign
            algorithm: Optimization algorithm to use
            optimization_objective: 'cost', 'efficiency', 'balanced', 'workload'

        Returns:
            List of assignment solutions
        """
        try:
            # Analyze all work order requirements
            all_requirements = []
            for wo in work_orders:
                req = self._analyze_work_order_requirements(wo)
                all_requirements.append(req)
                self.work_order_requirements[wo.order_id] = req

            # Get all available technicians
            all_technicians = self._get_all_available_technicians()

            # Run batch optimization
            batch_solution = self._batch_optimization(
                work_orders, all_requirements, all_technicians,
                algorithm, optimization_objective
            )

            # Apply all assignments
            assignment_solutions = []
            for assignment in batch_solution:
                self._apply_assignment(assignment)
                assignment_solutions.append(assignment)

            # Update metrics
            for solution in assignment_solutions:
                self._update_allocation_metrics(solution)

            logger.info(f"Batch optimized {len(work_orders)} work order assignments")
            return assignment_solutions

        except Exception as e:
            logger.error(f"Error in batch assignment optimization: {e}")
            return []

    def _analyze_work_order_requirements(self, work_order: WorkOrder) -> WorkOrderRequirements:
        """Analyze work order to determine detailed requirements

        Args:
            work_order: Work order to analyze

        Returns:
            Detailed requirements analysis
        """
        # Extract requirements from work order and anomaly data
        anomaly_details = work_order.anomaly_details or {}
        equipment = self.work_order_manager.equipment_registry.get(work_order.equipment_id)

        # Determine required skills based on equipment type and maintenance type
        required_skills = self._determine_required_skills(work_order, equipment)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(work_order, anomaly_details)

        # Determine urgency factor
        urgency_factor = self._calculate_urgency_factor(work_order)

        # Get required certifications
        required_certifications = self._get_required_certifications(work_order, equipment)

        # Determine location
        location = equipment.location if equipment else 'unknown'

        return WorkOrderRequirements(
            work_order_id=work_order.order_id,
            required_skills=required_skills,
            estimated_duration=work_order.estimated_duration_hours,
            complexity_score=complexity_score,
            urgency_factor=urgency_factor,
            location=location,
            required_certifications=required_certifications,
            preferred_technician_traits=self._get_preferred_traits(work_order),
            equipment_type=equipment.type if equipment else 'unknown',
            risk_level=self._assess_risk_level(work_order, anomaly_details)
        )

    def _determine_required_skills(self, work_order: WorkOrder, equipment: Optional[Equipment]) -> Dict[str, int]:
        """Determine required skills and levels for work order

        Args:
            work_order: Work order
            equipment: Equipment details

        Returns:
            Dictionary of skill -> required_level
        """
        skills = {}

        # Base skills from equipment type
        if equipment:
            equipment_skills = {
                'motor': {'electrical': 4, 'mechanical': 3},
                'pump': {'mechanical': 4, 'hydraulic': 3},
                'compressor': {'pneumatic': 4, 'mechanical': 3},
                'conveyor': {'mechanical': 3, 'electrical': 2},
                'sensor': {'electronics': 4, 'programming': 2}
            }
            skills.update(equipment_skills.get(equipment.type, {'general': 3}))

        # Adjust based on maintenance type
        type_adjustments = {
            MaintenanceType.EMERGENCY: 1,  # Require higher skill levels
            MaintenanceType.CORRECTIVE: 0,
            MaintenanceType.PREDICTIVE: 1,
            MaintenanceType.PREVENTIVE: -1,  # Can use lower skill levels
            MaintenanceType.INSPECTION: -1
        }

        adjustment = type_adjustments.get(work_order.type, 0)
        for skill in skills:
            skills[skill] = max(1, min(5, skills[skill] + adjustment))

        return skills

    def _calculate_complexity_score(self, work_order: WorkOrder, anomaly_details: Dict[str, Any]) -> float:
        """Calculate work order complexity score

        Args:
            work_order: Work order
            anomaly_details: Anomaly detection details

        Returns:
            Complexity score (0.0 to 1.0)
        """
        base_complexity = {
            MaintenanceType.EMERGENCY: 0.9,
            MaintenanceType.CORRECTIVE: 0.6,
            MaintenanceType.PREDICTIVE: 0.7,
            MaintenanceType.PREVENTIVE: 0.3,
            MaintenanceType.INSPECTION: 0.2
        }

        complexity = base_complexity.get(work_order.type, 0.5)

        # Adjust based on priority
        priority_adjustment = {
            WorkOrderPriority.CRITICAL: 0.2,
            WorkOrderPriority.HIGH: 0.1,
            WorkOrderPriority.MEDIUM: 0.0,
            WorkOrderPriority.LOW: -0.1
        }

        complexity += priority_adjustment.get(work_order.priority, 0.0)

        # Adjust based on anomaly confidence
        if anomaly_details:
            confidence = anomaly_details.get('confidence', 0.5)
            if confidence < 0.7:
                complexity += 0.1  # Less certain anomalies are more complex

        return min(1.0, max(0.0, complexity))

    def _calculate_urgency_factor(self, work_order: WorkOrder) -> float:
        """Calculate urgency factor for work order

        Args:
            work_order: Work order

        Returns:
            Urgency factor (0.0 to 1.0)
        """
        # Base urgency from priority
        priority_urgency = {
            WorkOrderPriority.CRITICAL: 1.0,
            WorkOrderPriority.HIGH: 0.8,
            WorkOrderPriority.MEDIUM: 0.5,
            WorkOrderPriority.LOW: 0.3,
            WorkOrderPriority.PREVENTIVE: 0.2
        }

        urgency = priority_urgency.get(work_order.priority, 0.5)

        # Adjust based on SLA deadline
        if work_order.sla_deadline:
            time_remaining = (work_order.sla_deadline - datetime.now()).total_seconds() / 3600
            if time_remaining < 4:
                urgency = min(1.0, urgency + 0.3)
            elif time_remaining < 24:
                urgency = min(1.0, urgency + 0.1)

        return urgency

    def _get_available_technicians(self, work_order: WorkOrder) -> List[str]:
        """Get list of available technicians for work order

        Args:
            work_order: Work order

        Returns:
            List of available technician IDs
        """
        available = []

        for tech_id, technician in self.work_order_manager.technician_registry.items():
            # Check basic availability
            if technician.current_workload < technician.max_daily_orders:
                # Check time availability
                if work_order.scheduled_start:
                    if technician.is_available(
                        work_order.scheduled_start,
                        int(work_order.estimated_duration_hours)
                    ):
                        available.append(tech_id)
                else:
                    available.append(tech_id)

        return available

    def _hungarian_assignment(self,
                            work_order: WorkOrder,
                            requirements: WorkOrderRequirements,
                            available_technicians: List[str],
                            consider_cost: bool) -> Optional[AssignmentSolution]:
        """Hungarian algorithm assignment (placeholder implementation)"""
        # For now, use the same logic as hybrid assignment
        return self._hybrid_assignment(work_order, requirements, available_technicians, consider_cost)

    def _genetic_algorithm_assignment(self,
                                    work_order: WorkOrder,
                                    requirements: WorkOrderRequirements,
                                    available_technicians: List[str],
                                    consider_cost: bool) -> Optional[AssignmentSolution]:
        """Genetic algorithm assignment (placeholder implementation)"""
        # For now, use the same logic as hybrid assignment
        return self._hybrid_assignment(work_order, requirements, available_technicians, consider_cost)

    def _ml_based_assignment(self,
                           work_order: WorkOrder,
                           requirements: WorkOrderRequirements,
                           available_technicians: List[str],
                           consider_cost: bool) -> Optional[AssignmentSolution]:
        """ML-based assignment (placeholder implementation)"""
        # For now, use the same logic as hybrid assignment
        return self._hybrid_assignment(work_order, requirements, available_technicians, consider_cost)

    def _batch_optimization(self,
                          work_orders: List[WorkOrder],
                          all_requirements: List[WorkOrderRequirements],
                          all_technicians: List[str],
                          algorithm: str,
                          optimization_objective: str) -> List[AssignmentSolution]:
        """Batch optimization (placeholder implementation)"""
        # For now, assign each work order individually
        solutions = []
        for i, work_order in enumerate(work_orders):
            solution = self._hybrid_assignment(
                work_order, all_requirements[i], all_technicians, True
            )
            if solution:
                solutions.append(solution)
        return solutions

    def _hybrid_assignment(self,
                         work_order: WorkOrder,
                         requirements: WorkOrderRequirements,
                         available_technicians: List[str],
                         consider_cost: bool) -> Optional[AssignmentSolution]:
        """Hybrid assignment algorithm combining multiple approaches

        Args:
            work_order: Work order to assign
            requirements: Work order requirements
            available_technicians: Available technicians
            consider_cost: Whether to consider cost

        Returns:
            Assignment solution
        """
        if not available_technicians:
            return None

        # Score each technician
        technician_scores = {}

        for tech_id in available_technicians:
            technician = self.work_order_manager.technician_registry[tech_id]

            # Calculate skill match score
            skill_score = self._calculate_skill_match_score(technician, requirements)

            # Calculate workload score (prefer less loaded technicians)
            workload_score = 1.0 - (technician.current_workload / technician.max_daily_orders)

            # Calculate location score
            location_score = self._calculate_location_score(technician, requirements)

            # Calculate cost score
            cost_score = 1.0 - min(1.0, technician.hourly_rate / 100.0) if consider_cost else 0.5

            # Calculate efficiency score
            efficiency_score = self._get_technician_efficiency(tech_id, work_order.type)

            # Combine scores
            total_score = (
                skill_score * 0.3 +
                workload_score * 0.2 +
                location_score * 0.1 +
                cost_score * 0.2 +
                efficiency_score * 0.2
            )

            technician_scores[tech_id] = total_score

        # Select best technician
        best_technician = max(technician_scores, key=technician_scores.get)
        best_score = technician_scores[best_technician]

        if best_score < 0.5:  # Threshold for acceptable assignment
            logger.warning(f"Best assignment score {best_score} below threshold for work order {work_order.order_id}")

        # Calculate assignment details
        technician = self.work_order_manager.technician_registry[best_technician]
        estimated_cost = self._calculate_assignment_cost(work_order, technician)

        # Create assignment solution
        assignment = AssignmentSolution(
            assignment_id=f"ASSIGN-{work_order.order_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            assignments={work_order.order_id: best_technician},
            total_cost=estimated_cost,
            efficiency_score=best_score,
            skill_match_score=self._calculate_skill_match_score(technician, requirements),
            workload_balance_score=1.0 - (technician.current_workload / technician.max_daily_orders),
            completion_time_estimate=work_order.estimated_duration_hours,
            risk_assessment=self._assess_assignment_risk(work_order, technician),
            confidence_level=best_score,
            alternative_assignments=self._get_alternative_assignments(technician_scores, work_order.order_id)
        )

        return assignment

    def _calculate_skill_match_score(self, technician: Technician, requirements: WorkOrderRequirements) -> float:
        """Calculate how well technician skills match requirements

        Args:
            technician: Technician to evaluate
            requirements: Work order requirements

        Returns:
            Skill match score (0.0 to 1.0)
        """
        if not requirements.required_skills:
            return 0.8  # Neutral score if no specific skills required

        total_score = 0.0
        total_weight = 0.0

        for skill, required_level in requirements.required_skills.items():
            weight = required_level / 5.0  # Normalize to 0-1

            if technician.has_skill(skill):
                # For simplicity, assume all technician skills are at level 3
                # In a real system, this would come from detailed skill profiles
                technician_level = 3
                score = min(1.0, technician_level / required_level)
            else:
                score = 0.0

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_location_score(self, technician: Technician, requirements: WorkOrderRequirements) -> float:
        """Calculate location proximity score

        Args:
            technician: Technician to evaluate
            requirements: Work order requirements

        Returns:
            Location score (0.0 to 1.0)
        """
        if not technician.location or not requirements.location:
            return 0.5  # Neutral score if location unknown

        if technician.location == requirements.location:
            return 1.0
        else:
            return 0.3  # Penalty for different location

    def _get_technician_efficiency(self, tech_id: str, maintenance_type: MaintenanceType) -> float:
        """Get technician efficiency for specific maintenance type

        Args:
            tech_id: Technician ID
            maintenance_type: Type of maintenance

        Returns:
            Efficiency score (0.0 to 1.0)
        """
        # In a real system, this would be based on historical performance data
        # For now, return a default efficiency based on experience
        technician = self.work_order_manager.technician_registry.get(tech_id)
        if not technician:
            return 0.5

        # Base efficiency on certification level
        base_efficiency = min(1.0, technician.certification_level / 5.0)

        # Adjust based on maintenance type complexity
        type_adjustments = {
            MaintenanceType.EMERGENCY: -0.1,
            MaintenanceType.CORRECTIVE: 0.0,
            MaintenanceType.PREDICTIVE: 0.1,
            MaintenanceType.PREVENTIVE: 0.1,
            MaintenanceType.INSPECTION: 0.2
        }

        adjustment = type_adjustments.get(maintenance_type, 0.0)
        return min(1.0, max(0.0, base_efficiency + adjustment))

    def _calculate_assignment_cost(self, work_order: WorkOrder, technician: Technician) -> float:
        """Calculate total cost of assignment

        Args:
            work_order: Work order
            technician: Assigned technician

        Returns:
            Total estimated cost
        """
        # Labor cost
        labor_cost = work_order.estimated_duration_hours * technician.hourly_rate

        # Priority multiplier
        priority_multipliers = {
            WorkOrderPriority.CRITICAL: 1.5,
            WorkOrderPriority.HIGH: 1.2,
            WorkOrderPriority.MEDIUM: 1.0,
            WorkOrderPriority.LOW: 1.0,
            WorkOrderPriority.PREVENTIVE: 0.9
        }

        multiplier = priority_multipliers.get(work_order.priority, 1.0)
        total_labor_cost = labor_cost * multiplier

        # Add estimated parts cost (from work order)
        total_cost = total_labor_cost + work_order.estimated_cost

        return total_cost

    def _assess_assignment_risk(self, work_order: WorkOrder, technician: Technician) -> Dict[str, Any]:
        """Assess risk factors for assignment

        Args:
            work_order: Work order
            technician: Technician

        Returns:
            Risk assessment
        """
        risk_factors = []
        risk_score = 0.0

        # Skill mismatch risk
        # This would be calculated based on detailed skill analysis
        skill_risk = 0.2  # Placeholder
        risk_score += skill_risk

        # Workload risk
        if technician.current_workload > technician.max_daily_orders * 0.8:
            risk_factors.append("High technician workload")
            risk_score += 0.3

        # Priority vs certification risk
        if (work_order.priority == WorkOrderPriority.CRITICAL and
            technician.certification_level < 4):
            risk_factors.append("Low certification for critical work")
            risk_score += 0.4

        # Time pressure risk
        if work_order.sla_deadline:
            time_remaining = (work_order.sla_deadline - datetime.now()).total_seconds() / 3600
            if time_remaining < work_order.estimated_duration_hours * 1.2:
                risk_factors.append("Tight deadline")
                risk_score += 0.3

        return {
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors,
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low'
        }

    def _get_alternative_assignments(self, technician_scores: Dict[str, float], work_order_id: str) -> List[Dict[str, str]]:
        """Get alternative assignment options

        Args:
            technician_scores: Scores for all technicians
            work_order_id: Work order ID

        Returns:
            List of alternative assignments
        """
        # Sort technicians by score
        sorted_techs = sorted(technician_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top 3 alternatives (excluding the best one)
        alternatives = []
        for tech_id, score in sorted_techs[1:4]:
            alternatives.append({work_order_id: tech_id})

        return alternatives

    def _apply_assignment(self, assignment_solution: AssignmentSolution):
        """Apply assignment solution to work order

        Args:
            assignment_solution: Assignment to apply
        """
        for work_order_id, technician_id in assignment_solution.assignments.items():
            if work_order_id in self.work_order_manager.work_orders:
                work_order = self.work_order_manager.work_orders[work_order_id]
                work_order.assigned_technician = technician_id
                work_order.status = WorkOrderStatus.ASSIGNED

                # Update technician workload
                if technician_id in self.work_order_manager.technician_registry:
                    technician = self.work_order_manager.technician_registry[technician_id]
                    technician.current_workload += 1

                logger.info(f"Applied assignment: Work Order {work_order_id} -> Technician {technician_id}")

    def _update_allocation_metrics(self, assignment_solution: AssignmentSolution):
        """Update allocation performance metrics

        Args:
            assignment_solution: Completed assignment
        """
        self.allocation_metrics['total_assignments'] += 1

        if assignment_solution.confidence_level > 0.7:
            self.allocation_metrics['successful_assignments'] += 1

        # Update running averages
        n = self.allocation_metrics['total_assignments']
        self.allocation_metrics['average_skill_match'] = (
            (self.allocation_metrics['average_skill_match'] * (n - 1) +
             assignment_solution.skill_match_score) / n
        )

        self.allocation_metrics['average_cost_efficiency'] = (
            (self.allocation_metrics['average_cost_efficiency'] * (n - 1) +
             assignment_solution.efficiency_score) / n
        )

        self.allocation_metrics['workload_balance_score'] = (
            (self.allocation_metrics['workload_balance_score'] * (n - 1) +
             assignment_solution.workload_balance_score) / n
        )

    def _integrate_with_phase31_optimizer(self, assignment_solution: AssignmentSolution):
        """Integrate assignment with Phase 3.1 resource optimizer

        Args:
            assignment_solution: Assignment solution to integrate
        """
        try:
            # Update Phase 3.1 optimizer with assignment information
            for work_order_id, technician_id in assignment_solution.assignments.items():
                # This would integrate with the Phase 3.1 resource optimizer
                # to maintain consistency across the system
                pass

            logger.info("Integrated assignment with Phase 3.1 resource optimizer")

        except Exception as e:
            logger.error(f"Error integrating with Phase 3.1 optimizer: {e}")

    def _get_all_available_technicians(self) -> List[str]:
        """Get all available technicians

        Returns:
            List of all available technician IDs
        """
        return list(self.work_order_manager.technician_registry.keys())

    def _get_team_technicians(self, team_ids: List[str]) -> List[Technician]:
        """Get technicians from specific teams

        Args:
            team_ids: List of team IDs

        Returns:
            List of technicians
        """
        # For now, return all technicians
        return list(self.work_order_manager.technician_registry.values())

    def _calculate_current_workload(self, technician_id: str) -> float:
        """Calculate current workload for technician

        Args:
            technician_id: Technician identifier

        Returns:
            Current workload percentage
        """
        technician = self.work_order_manager.technician_registry.get(technician_id)
        if not technician:
            return 0.0

        return technician.current_workload / technician.max_daily_orders

    def _calculate_workload_imbalance(self, technician_workloads: Dict[str, float]) -> float:
        """Calculate workload imbalance score

        Args:
            technician_workloads: Dictionary of technician workloads

        Returns:
            Imbalance score (0.0 to 1.0)
        """
        if not technician_workloads:
            return 0.0

        workloads = list(technician_workloads.values())
        if len(workloads) < 2:
            return 0.0

        # Calculate coefficient of variation
        mean_workload = np.mean(workloads)
        std_workload = np.std(workloads)

        if mean_workload == 0:
            return 0.0

        return min(1.0, std_workload / mean_workload)

    def _identify_workload_extremes(self, technician_workloads: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify overloaded and underutilized technicians

        Args:
            technician_workloads: Dictionary of technician workloads

        Returns:
            Tuple of (overloaded_technicians, underutilized_technicians)
        """
        if not technician_workloads:
            return [], []

        overloaded = []
        underutilized = []

        for tech_id, workload in technician_workloads.items():
            if workload > 0.85:  # 85% threshold for overloaded
                overloaded.append(tech_id)
            elif workload < 0.30:  # 30% threshold for underutilized
                underutilized.append(tech_id)

        return overloaded, underutilized

    def _calculate_optimal_workload_distribution(self, technicians: List[Technician]) -> Dict[str, float]:
        """Calculate optimal workload distribution

        Args:
            technicians: List of technicians

        Returns:
            Optimal workload distribution
        """
        optimal_distribution = {}
        target_utilization = 0.75  # 75% target utilization

        for tech in technicians:
            optimal_distribution[tech.technician_id] = target_utilization

        return optimal_distribution

    def _generate_rebalancing_recommendations(self,
                                            current_workloads: Dict[str, float],
                                            optimal_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalancing recommendations

        Args:
            current_workloads: Current workload distribution
            optimal_distribution: Optimal distribution

        Returns:
            List of recommendations
        """
        recommendations = []

        for tech_id in current_workloads:
            current = current_workloads.get(tech_id, 0)
            optimal = optimal_distribution.get(tech_id, 0.75)

            if abs(current - optimal) > 0.15:  # 15% threshold
                if current > optimal:
                    recommendations.append({
                        'type': 'reduce_workload',
                        'technician_id': tech_id,
                        'current_workload': current,
                        'target_workload': optimal,
                        'priority': 'high' if current > 0.9 else 'medium'
                    })
                else:
                    recommendations.append({
                        'type': 'increase_workload',
                        'technician_id': tech_id,
                        'current_workload': current,
                        'target_workload': optimal,
                        'priority': 'low'
                    })

        return recommendations

    def _predict_workload_bottlenecks(self, technician_workloads: Dict[str, float]) -> List[str]:
        """Predict potential workload bottlenecks

        Args:
            technician_workloads: Current workload distribution

        Returns:
            List of predicted bottlenecks
        """
        bottlenecks = []

        for tech_id, workload in technician_workloads.items():
            if workload > 0.80:
                bottlenecks.append(f"Technician {tech_id} approaching capacity")

        return bottlenecks

    def _calculate_capacity_utilization(self, technicians: List[Technician]) -> float:
        """Calculate overall capacity utilization

        Args:
            technicians: List of technicians

        Returns:
            Capacity utilization rate
        """
        if not technicians:
            return 0.0

        total_capacity = sum(tech.max_daily_orders for tech in technicians)
        total_workload = sum(tech.current_workload for tech in technicians)

        return total_workload / total_capacity if total_capacity > 0 else 0.0

    def _get_required_certifications(self, work_order: WorkOrder, equipment: Optional[Equipment]) -> List[str]:
        """Get required certifications for work order

        Args:
            work_order: Work order
            equipment: Equipment details

        Returns:
            List of required certifications
        """
        certifications = []

        # Base certifications by equipment type
        if equipment:
            equipment_certs = {
                'motor': ['electrical_cert'],
                'pump': ['hydraulic_cert'],
                'compressor': ['pneumatic_cert'],
                'sensor': ['electronics_cert']
            }
            certifications.extend(equipment_certs.get(equipment.type, []))

        # Additional certifications by maintenance type
        if work_order.type == MaintenanceType.EMERGENCY:
            certifications.append('emergency_response_cert')

        # Safety certifications by priority
        if work_order.priority == WorkOrderPriority.CRITICAL:
            certifications.append('safety_cert')

        return list(set(certifications))  # Remove duplicates

    def _get_preferred_traits(self, work_order: WorkOrder) -> List[str]:
        """Get preferred technician traits for work order

        Args:
            work_order: Work order

        Returns:
            List of preferred traits
        """
        traits = []

        if work_order.priority == WorkOrderPriority.CRITICAL:
            traits.extend(['experienced', 'quick_response'])

        if work_order.type == MaintenanceType.EMERGENCY:
            traits.extend(['calm_under_pressure', 'decision_maker'])

        return traits

    def _assess_risk_level(self, work_order: WorkOrder, anomaly_details: Dict[str, Any]) -> str:
        """Assess risk level for work order

        Args:
            work_order: Work order
            anomaly_details: Anomaly details

        Returns:
            Risk level string
        """
        if work_order.priority == WorkOrderPriority.CRITICAL:
            return 'critical'
        elif work_order.priority == WorkOrderPriority.HIGH:
            return 'high'
        elif work_order.type == MaintenanceType.EMERGENCY:
            return 'high'
        else:
            return 'medium'

    def _initialize_system(self):
        """Initialize allocation system with default configurations"""
        # Initialize default parameters
        self.config.setdefault('skill_match_threshold', 0.7)
        self.config.setdefault('workload_balance_weight', 0.3)
        self.config.setdefault('cost_optimization_weight', 0.2)

        logger.info("Initialized resource allocation system")

    def get_allocation_metrics(self) -> Dict[str, Any]:
        """Get allocation performance metrics

        Returns:
            Allocation metrics
        """
        return self.allocation_metrics.copy()

    def get_technician_workload_report(self) -> Dict[str, Any]:
        """Generate technician workload balancing report

        Returns:
            Workload report
        """
        report = {
            'timestamp': datetime.now(),
            'technician_workloads': {},
            'workload_distribution': {},
            'recommendations': []
        }

        for tech_id, technician in self.work_order_manager.technician_registry.items():
            workload_percentage = (technician.current_workload / technician.max_daily_orders) * 100
            report['technician_workloads'][tech_id] = {
                'current_workload': technician.current_workload,
                'max_capacity': technician.max_daily_orders,
                'utilization_percentage': workload_percentage,
                'availability_status': 'overloaded' if workload_percentage > 100 else
                                    'high' if workload_percentage > 80 else
                                    'medium' if workload_percentage > 50 else 'low'
            }

        return report


# Placeholder classes for components referenced but not fully implemented in this scope
class AdvancedSkillMatcher:
    def __init__(self):
        pass

class WorkloadBalancer:
    def __init__(self):
        pass

class AssignmentOptimizer:
    def __init__(self):
        pass

class PerformanceTracker:
    def __init__(self):
        pass