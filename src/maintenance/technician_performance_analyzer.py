"""
Technician Performance Analyzer Module for Phase 3.2
Advanced analytics for technician performance and workload balancing
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

# ML and analytics imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import scipy.stats as stats

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path
from src.maintenance.work_order_manager import (
    WorkOrder, WorkOrderPriority, WorkOrderStatus, MaintenanceType,
    Equipment, Technician, WorkOrderManager
)
from src.maintenance.work_order_lifecycle_tracker import WorkOrderLifecycleTracker

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TechnicianPerformanceMetrics:
    """Comprehensive technician performance metrics"""
    technician_id: str
    name: str
    evaluation_period: Tuple[datetime, datetime]

    # Productivity metrics
    total_work_orders: int
    completed_work_orders: int
    completion_rate: float
    average_completion_time: float
    work_orders_per_day: float
    efficiency_score: float

    # Quality metrics
    rework_rate: float
    customer_satisfaction: float
    first_time_fix_rate: float
    quality_score: float

    # Workload metrics
    total_hours_worked: float
    average_daily_hours: float
    overtime_hours: float
    workload_utilization: float
    stress_level_indicator: float

    # Skill metrics
    skill_utilization: Dict[str, float]
    skill_development_rate: Dict[str, float]
    cross_training_opportunities: List[str]

    # Cost metrics
    labor_cost_efficiency: float
    cost_per_work_order: float
    overtime_cost_ratio: float

    # Collaboration metrics
    team_collaboration_score: float
    knowledge_sharing_score: float
    mentoring_effectiveness: float

    # Trend indicators
    performance_trend: str  # 'improving', 'stable', 'declining'
    burnout_risk: str  # 'low', 'medium', 'high'
    promotion_readiness: float


@dataclass
class WorkloadAnalysis:
    """Workload analysis for technician balancing"""
    analysis_id: str
    analysis_date: datetime
    technician_workloads: Dict[str, float]  # technician_id -> current workload
    workload_imbalance_score: float
    overloaded_technicians: List[str]
    underutilized_technicians: List[str]
    optimal_workload_distribution: Dict[str, float]
    rebalancing_recommendations: List[Dict[str, Any]]
    predicted_bottlenecks: List[str]
    capacity_utilization_rate: float


@dataclass
class PerformanceComparison:
    """Performance comparison between technicians"""
    comparison_id: str
    comparison_date: datetime
    technicians_compared: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]  # metric -> {tech_id: value}
    performance_rankings: Dict[str, int]  # tech_id -> rank
    best_practices: List[str]
    improvement_opportunities: Dict[str, List[str]]  # tech_id -> opportunities
    skill_gaps: Dict[str, List[str]]  # tech_id -> missing skills


@dataclass
class TeamDynamics:
    """Team dynamics and collaboration analysis"""
    team_id: str
    team_members: List[str]
    collaboration_matrix: Dict[Tuple[str, str], float]  # (tech1, tech2) -> collaboration score
    communication_effectiveness: float
    knowledge_sharing_rate: float
    conflict_indicators: List[str]
    team_cohesion_score: float
    leadership_effectiveness: Dict[str, float]  # leader_id -> effectiveness


class TechnicianPerformanceAnalyzer:
    """Advanced technician performance analysis and workload balancing system"""

    def __init__(self,
                 work_order_manager: WorkOrderManager,
                 lifecycle_tracker: WorkOrderLifecycleTracker,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize Technician Performance Analyzer

        Args:
            work_order_manager: Work order management system
            lifecycle_tracker: Work order lifecycle tracker
            config: Configuration dictionary
        """
        self.work_order_manager = work_order_manager
        self.lifecycle_tracker = lifecycle_tracker
        self.config = config or {}

        # Performance data storage
        self.performance_history = defaultdict(list)  # tech_id -> [TechnicianPerformanceMetrics]
        self.workload_analyses = []  # [WorkloadAnalysis]
        self.performance_comparisons = []  # [PerformanceComparison]
        self.team_dynamics = {}  # team_id -> TeamDynamics

        # Analytics components
        self.performance_calculator = PerformanceCalculator()
        self.workload_balancer = WorkloadBalancer()
        self.trend_analyzer = TrendAnalyzer()
        self.skill_analyzer = SkillAnalyzer()
        self.collaboration_analyzer = CollaborationAnalyzer()

        # ML models for predictions
        self.burnout_predictor = BurnoutPredictor()
        self.performance_predictor = PerformancePredictor()
        self.workload_optimizer = WorkloadOptimizer()

        # Real-time monitoring
        self.monitoring_enabled = True
        self.analysis_queue = queue.Queue()
        self.analysis_executor = ThreadPoolExecutor(max_workers=2)

        # Benchmarks and thresholds
        self.performance_benchmarks = self._initialize_benchmarks()
        self.alert_thresholds = self._initialize_alert_thresholds()

        # Performance tracking
        self.analyzer_metrics = {
            'total_analyses_performed': 0,
            'average_analysis_time': 0.0,
            'workload_balance_improvements': 0,
            'performance_improvements_identified': 0
        }

        self._initialize_analyzer()
        logger.info("Initialized Technician Performance Analyzer")

    def analyze_technician_performance(self,
                                     technician_id: str,
                                     analysis_period: Optional[Tuple[datetime, datetime]] = None) -> TechnicianPerformanceMetrics:
        """Analyze comprehensive performance metrics for a technician

        Args:
            technician_id: Technician identifier
            analysis_period: Period to analyze (start, end), defaults to last 30 days

        Returns:
            Performance metrics
        """
        try:
            if not analysis_period:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                analysis_period = (start_date, end_date)

            # Get technician data
            technician = self.work_order_manager.technician_registry.get(technician_id)
            if not technician:
                logger.error(f"Technician {technician_id} not found")
                return None

            # Get work orders for the period
            work_orders = self._get_technician_work_orders(technician_id, analysis_period)

            # Calculate performance metrics
            performance_metrics = self._calculate_comprehensive_metrics(
                technician_id, work_orders, analysis_period
            )

            # Store performance history
            self.performance_history[technician_id].append(performance_metrics)

            # Perform trend analysis
            self._analyze_performance_trends(technician_id)

            # Check for alerts
            self._check_performance_alerts(performance_metrics)

            logger.info(f"Analyzed performance for technician {technician_id}")
            return performance_metrics

        except Exception as e:
            logger.error(f"Error analyzing technician performance: {e}")
            return None

    def analyze_workload_balance(self,
                               team_ids: Optional[List[str]] = None) -> WorkloadAnalysis:
        """Analyze workload balance across technicians

        Args:
            team_ids: Specific team IDs to analyze, None for all

        Returns:
            Workload analysis
        """
        try:
            # Get technicians to analyze
            if team_ids:
                technicians = self._get_team_technicians(team_ids)
            else:
                technicians = list(self.work_order_manager.technician_registry.values())

            # Calculate current workloads
            technician_workloads = {}
            for tech in technicians:
                workload = self._calculate_current_workload(tech.technician_id)
                technician_workloads[tech.technician_id] = workload

            # Calculate workload imbalance
            imbalance_score = self._calculate_workload_imbalance(technician_workloads)

            # Identify overloaded and underutilized technicians
            overloaded, underutilized = self._identify_workload_extremes(technician_workloads)

            # Calculate optimal distribution
            optimal_distribution = self._calculate_optimal_workload_distribution(technicians)

            # Generate rebalancing recommendations
            recommendations = self._generate_rebalancing_recommendations(
                technician_workloads, optimal_distribution
            )

            # Predict bottlenecks
            predicted_bottlenecks = self._predict_workload_bottlenecks(technician_workloads)

            # Calculate capacity utilization
            capacity_utilization = self._calculate_capacity_utilization(technicians)

            # Create workload analysis
            analysis = WorkloadAnalysis(
                analysis_id=f"WLA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                analysis_date=datetime.now(),
                technician_workloads=technician_workloads,
                workload_imbalance_score=imbalance_score,
                overloaded_technicians=overloaded,
                underutilized_technicians=underutilized,
                optimal_workload_distribution=optimal_distribution,
                rebalancing_recommendations=recommendations,
                predicted_bottlenecks=predicted_bottlenecks,
                capacity_utilization_rate=capacity_utilization
            )

            self.workload_analyses.append(analysis)
            logger.info("Completed workload balance analysis")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing workload balance: {e}")
            return None

    def compare_technician_performance(self,
                                     technician_ids: List[str],
                                     comparison_metrics: Optional[List[str]] = None) -> PerformanceComparison:
        """Compare performance between multiple technicians

        Args:
            technician_ids: List of technician IDs to compare
            comparison_metrics: Specific metrics to compare

        Returns:
            Performance comparison
        """
        try:
            if not comparison_metrics:
                comparison_metrics = [
                    'completion_rate', 'efficiency_score', 'quality_score',
                    'cost_per_work_order', 'workload_utilization'
                ]

            # Get recent performance metrics for each technician
            technician_metrics = {}
            for tech_id in technician_ids:
                recent_metrics = self._get_recent_performance_metrics(tech_id)
                if recent_metrics:
                    technician_metrics[tech_id] = recent_metrics

            if len(technician_metrics) < 2:
                logger.warning("Insufficient data for performance comparison")
                return None

            # Extract comparison data
            comparison_data = {}
            for metric in comparison_metrics:
                comparison_data[metric] = {}
                for tech_id, metrics in technician_metrics.items():
                    value = getattr(metrics, metric, 0)
                    comparison_data[metric][tech_id] = value

            # Calculate rankings
            rankings = self._calculate_performance_rankings(comparison_data)

            # Identify best practices
            best_practices = self._identify_best_practices(technician_metrics)

            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(
                technician_metrics, comparison_data
            )

            # Identify skill gaps
            skill_gaps = self._identify_skill_gaps(technician_ids)

            comparison = PerformanceComparison(
                comparison_id=f"PC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                comparison_date=datetime.now(),
                technicians_compared=technician_ids,
                comparison_metrics=comparison_data,
                performance_rankings=rankings,
                best_practices=best_practices,
                improvement_opportunities=improvement_opportunities,
                skill_gaps=skill_gaps
            )

            self.performance_comparisons.append(comparison)
            logger.info(f"Completed performance comparison for {len(technician_ids)} technicians")

            return comparison

        except Exception as e:
            logger.error(f"Error comparing technician performance: {e}")
            return None

    def analyze_team_dynamics(self, team_id: str) -> TeamDynamics:
        """Analyze team dynamics and collaboration patterns

        Args:
            team_id: Team identifier

        Returns:
            Team dynamics analysis
        """
        try:
            # Get team members
            team_members = self._get_team_members(team_id)
            if len(team_members) < 2:
                logger.warning(f"Team {team_id} has insufficient members for dynamics analysis")
                return None

            # Calculate collaboration matrix
            collaboration_matrix = self._calculate_collaboration_matrix(team_members)

            # Analyze communication effectiveness
            communication_effectiveness = self._analyze_communication_effectiveness(team_members)

            # Calculate knowledge sharing rate
            knowledge_sharing_rate = self._calculate_knowledge_sharing_rate(team_members)

            # Identify conflict indicators
            conflict_indicators = self._identify_conflict_indicators(team_members)

            # Calculate team cohesion
            team_cohesion = self._calculate_team_cohesion(collaboration_matrix)

            # Analyze leadership effectiveness
            leadership_effectiveness = self._analyze_leadership_effectiveness(team_members)

            dynamics = TeamDynamics(
                team_id=team_id,
                team_members=team_members,
                collaboration_matrix=collaboration_matrix,
                communication_effectiveness=communication_effectiveness,
                knowledge_sharing_rate=knowledge_sharing_rate,
                conflict_indicators=conflict_indicators,
                team_cohesion_score=team_cohesion,
                leadership_effectiveness=leadership_effectiveness
            )

            self.team_dynamics[team_id] = dynamics
            logger.info(f"Analyzed team dynamics for team {team_id}")

            return dynamics

        except Exception as e:
            logger.error(f"Error analyzing team dynamics: {e}")
            return None

    def generate_workload_recommendations(self,
                                        optimization_objective: str = 'balanced') -> List[Dict[str, Any]]:
        """Generate workload balancing recommendations

        Args:
            optimization_objective: 'balanced', 'efficiency', 'cost'

        Returns:
            List of recommendations
        """
        try:
            # Get current workload analysis
            current_analysis = self.analyze_workload_balance()
            if not current_analysis:
                return []

            recommendations = []

            # Immediate rebalancing recommendations
            if current_analysis.workload_imbalance_score > 0.3:
                recommendations.extend(current_analysis.rebalancing_recommendations)

            # Skill development recommendations
            skill_recommendations = self._generate_skill_development_recommendations()
            recommendations.extend(skill_recommendations)

            # Capacity planning recommendations
            capacity_recommendations = self._generate_capacity_planning_recommendations(current_analysis)
            recommendations.extend(capacity_recommendations)

            # Team optimization recommendations
            team_recommendations = self._generate_team_optimization_recommendations()
            recommendations.extend(team_recommendations)

            logger.info(f"Generated {len(recommendations)} workload recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating workload recommendations: {e}")
            return []

    def _calculate_comprehensive_metrics(self,
                                       technician_id: str,
                                       work_orders: List[WorkOrder],
                                       analysis_period: Tuple[datetime, datetime]) -> TechnicianPerformanceMetrics:
        """Calculate comprehensive performance metrics

        Args:
            technician_id: Technician identifier
            work_orders: Work orders for analysis
            analysis_period: Analysis period

        Returns:
            Performance metrics
        """
        technician = self.work_order_manager.technician_registry[technician_id]

        # Basic productivity metrics
        total_work_orders = len(work_orders)
        completed_work_orders = len([wo for wo in work_orders if wo.status == WorkOrderStatus.COMPLETED])
        completion_rate = completed_work_orders / total_work_orders if total_work_orders > 0 else 0

        # Calculate average completion time
        completed_orders = [wo for wo in work_orders if wo.actual_duration_hours]
        avg_completion_time = np.mean([wo.actual_duration_hours for wo in completed_orders]) if completed_orders else 0

        # Calculate work orders per day
        period_days = (analysis_period[1] - analysis_period[0]).days
        work_orders_per_day = total_work_orders / period_days if period_days > 0 else 0

        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(work_orders)

        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(work_orders)

        # Workload metrics
        workload_metrics = self._calculate_workload_metrics(technician_id, analysis_period)

        # Skill metrics
        skill_metrics = self._calculate_skill_metrics(technician_id, work_orders)

        # Cost metrics
        cost_metrics = self._calculate_cost_metrics(work_orders, technician)

        # Collaboration metrics
        collaboration_metrics = self._calculate_collaboration_metrics(technician_id)

        # Trend analysis
        performance_trend = self._analyze_individual_trend(technician_id)
        burnout_risk = self._assess_burnout_risk(technician_id, workload_metrics)
        promotion_readiness = self._assess_promotion_readiness(technician_id)

        return TechnicianPerformanceMetrics(
            technician_id=technician_id,
            name=technician.name,
            evaluation_period=analysis_period,
            total_work_orders=total_work_orders,
            completed_work_orders=completed_work_orders,
            completion_rate=completion_rate,
            average_completion_time=avg_completion_time,
            work_orders_per_day=work_orders_per_day,
            efficiency_score=efficiency_score,
            rework_rate=quality_metrics['rework_rate'],
            customer_satisfaction=quality_metrics['customer_satisfaction'],
            first_time_fix_rate=quality_metrics['first_time_fix_rate'],
            quality_score=quality_metrics['quality_score'],
            total_hours_worked=workload_metrics['total_hours'],
            average_daily_hours=workload_metrics['avg_daily_hours'],
            overtime_hours=workload_metrics['overtime_hours'],
            workload_utilization=workload_metrics['utilization'],
            stress_level_indicator=workload_metrics['stress_level'],
            skill_utilization=skill_metrics['skill_utilization'],
            skill_development_rate=skill_metrics['development_rate'],
            cross_training_opportunities=skill_metrics['cross_training_opportunities'],
            labor_cost_efficiency=cost_metrics['cost_efficiency'],
            cost_per_work_order=cost_metrics['cost_per_work_order'],
            overtime_cost_ratio=cost_metrics['overtime_cost_ratio'],
            team_collaboration_score=collaboration_metrics['collaboration_score'],
            knowledge_sharing_score=collaboration_metrics['knowledge_sharing'],
            mentoring_effectiveness=collaboration_metrics['mentoring_effectiveness'],
            performance_trend=performance_trend,
            burnout_risk=burnout_risk,
            promotion_readiness=promotion_readiness
        )

    def _get_technician_work_orders(self,
                                  technician_id: str,
                                  period: Tuple[datetime, datetime]) -> List[WorkOrder]:
        """Get work orders for technician in given period

        Args:
            technician_id: Technician identifier
            period: Time period (start, end)

        Returns:
            List of work orders
        """
        return [
            wo for wo in self.work_order_manager.work_orders.values()
            if (wo.assigned_technician == technician_id and
                period[0] <= wo.created_at <= period[1])
        ]

    def _calculate_efficiency_score(self, work_orders: List[WorkOrder]) -> float:
        """Calculate efficiency score based on work orders

        Args:
            work_orders: List of work orders

        Returns:
            Efficiency score (0.0 to 1.0)
        """
        if not work_orders:
            return 0.0

        efficiency_scores = []
        for wo in work_orders:
            if wo.actual_duration_hours and wo.estimated_duration_hours:
                # Efficiency = estimated / actual (capped at 1.0)
                efficiency = min(1.0, wo.estimated_duration_hours / wo.actual_duration_hours)
                efficiency_scores.append(efficiency)

        return np.mean(efficiency_scores) if efficiency_scores else 0.5

    def _calculate_quality_metrics(self, work_orders: List[WorkOrder]) -> Dict[str, float]:
        """Calculate quality-related metrics

        Args:
            work_orders: List of work orders

        Returns:
            Quality metrics
        """
        # Placeholder implementation
        return {
            'rework_rate': 0.05,  # 5% rework rate
            'customer_satisfaction': 0.85,  # 85% satisfaction
            'first_time_fix_rate': 0.90,  # 90% first-time fix
            'quality_score': 0.85  # Overall quality score
        }

    def _calculate_workload_metrics(self,
                                  technician_id: str,
                                  period: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate workload-related metrics

        Args:
            technician_id: Technician identifier
            period: Analysis period

        Returns:
            Workload metrics
        """
        technician = self.work_order_manager.technician_registry[technician_id]
        period_days = (period[1] - period[0]).days

        # Placeholder implementation
        total_hours = technician.current_workload * 8 * period_days  # Estimate
        avg_daily_hours = total_hours / period_days if period_days > 0 else 0
        overtime_hours = max(0, avg_daily_hours - 8) * period_days
        utilization = technician.current_workload / technician.max_daily_orders
        stress_level = min(1.0, utilization * 1.2)  # Stress increases with utilization

        return {
            'total_hours': total_hours,
            'avg_daily_hours': avg_daily_hours,
            'overtime_hours': overtime_hours,
            'utilization': utilization,
            'stress_level': stress_level
        }

    def _calculate_skill_metrics(self,
                               technician_id: str,
                               work_orders: List[WorkOrder]) -> Dict[str, Any]:
        """Calculate skill-related metrics

        Args:
            technician_id: Technician identifier
            work_orders: List of work orders

        Returns:
            Skill metrics
        """
        technician = self.work_order_manager.technician_registry[technician_id]

        # Calculate skill utilization
        skill_utilization = {}
        for skill in technician.skills:
            utilization = 0.7  # Placeholder - would calculate based on work orders
            skill_utilization[skill] = utilization

        # Placeholder implementation
        return {
            'skill_utilization': skill_utilization,
            'development_rate': {skill: 0.1 for skill in technician.skills},
            'cross_training_opportunities': ['electronics', 'programming']
        }

    def _calculate_cost_metrics(self,
                              work_orders: List[WorkOrder],
                              technician: Technician) -> Dict[str, float]:
        """Calculate cost-related metrics

        Args:
            work_orders: List of work orders
            technician: Technician details

        Returns:
            Cost metrics
        """
        if not work_orders:
            return {'cost_efficiency': 0.5, 'cost_per_work_order': 0, 'overtime_cost_ratio': 0}

        total_cost = sum(wo.actual_cost for wo in work_orders if wo.actual_cost)
        cost_per_work_order = total_cost / len(work_orders) if work_orders else 0

        # Placeholder calculations
        cost_efficiency = 0.8  # Would calculate based on cost vs. value
        overtime_cost_ratio = 0.1  # 10% of costs from overtime

        return {
            'cost_efficiency': cost_efficiency,
            'cost_per_work_order': cost_per_work_order,
            'overtime_cost_ratio': overtime_cost_ratio
        }

    def _calculate_collaboration_metrics(self, technician_id: str) -> Dict[str, float]:
        """Calculate collaboration metrics

        Args:
            technician_id: Technician identifier

        Returns:
            Collaboration metrics
        """
        # Placeholder implementation
        return {
            'collaboration_score': 0.8,
            'knowledge_sharing': 0.7,
            'mentoring_effectiveness': 0.6
        }

    def _initialize_benchmarks(self) -> Dict[str, float]:
        """Initialize performance benchmarks

        Returns:
            Performance benchmarks
        """
        return {
            'completion_rate': 0.90,
            'efficiency_score': 0.80,
            'quality_score': 0.85,
            'workload_utilization': 0.75,
            'cost_efficiency': 0.80
        }

    def _initialize_alert_thresholds(self) -> Dict[str, float]:
        """Initialize alert thresholds

        Returns:
            Alert thresholds
        """
        return {
            'burnout_risk_threshold': 0.7,
            'workload_imbalance_threshold': 0.3,
            'efficiency_decline_threshold': 0.1,
            'quality_decline_threshold': 0.1
        }

    def _initialize_analyzer(self):
        """Initialize the analyzer with default configurations"""
        logger.info("Initialized performance analyzer with default configurations")

    def get_analyzer_metrics(self) -> Dict[str, Any]:
        """Get analyzer performance metrics

        Returns:
            Analyzer metrics
        """
        return self.analyzer_metrics.copy()


# Placeholder classes for components referenced but not fully implemented
class PerformanceCalculator:
    def __init__(self):
        pass

class WorkloadBalancer:
    def __init__(self):
        pass

class TrendAnalyzer:
    def __init__(self):
        pass

class SkillAnalyzer:
    def __init__(self):
        pass

class CollaborationAnalyzer:
    def __init__(self):
        pass

class BurnoutPredictor:
    def __init__(self):
        pass

class PerformancePredictor:
    def __init__(self):
        pass

class WorkloadOptimizer:
    def __init__(self):
        pass