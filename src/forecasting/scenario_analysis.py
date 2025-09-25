"""
What-If Analysis and Scenario Modeling Framework
Comprehensive scenario simulation for maintenance strategy optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import json
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optimization imports
from scipy.optimize import minimize, differential_evolution
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.forecasting.failure_probability import FailureProbabilityEstimator, FailurePrediction
from src.forecasting.enhanced_forecaster import EnhancedForecaster, EnhancedForecastResult
from config.settings import settings, get_config

logger = logging.getLogger(__name__)


class MaintenanceStrategy(Enum):
    """Maintenance strategy types"""
    REACTIVE = "reactive"  # Fix after failure
    PREVENTIVE = "preventive"  # Scheduled maintenance
    PREDICTIVE = "predictive"  # Based on condition monitoring
    HYBRID = "hybrid"  # Combination of strategies


class ResourceType(Enum):
    """Resource types for maintenance"""
    TECHNICIAN = "technician"
    SPARE_PARTS = "spare_parts"
    EQUIPMENT = "equipment"
    BUDGET = "budget"


@dataclass
class MaintenanceAction:
    """Represents a maintenance action"""
    action_id: str
    equipment_id: str
    component_id: str
    action_type: str  # "inspection", "repair", "replacement", "calibration"
    scheduled_time: datetime
    duration: float  # Hours
    cost: float
    required_resources: Dict[ResourceType, int]
    effectiveness: float  # Reduction in failure probability (0-1)
    prerequisites: List[str] = field(default_factory=list)  # Required prior actions

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'equipment_id': self.equipment_id,
            'component_id': self.component_id,
            'action_type': self.action_type,
            'scheduled_time': self.scheduled_time.isoformat(),
            'duration': self.duration,
            'cost': self.cost,
            'required_resources': {k.value: v for k, v in self.required_resources.items()},
            'effectiveness': self.effectiveness,
            'prerequisites': self.prerequisites
        }


@dataclass
class Resource:
    """Resource constraints and availability"""
    resource_type: ResourceType
    total_available: int
    hourly_cost: float
    availability_schedule: Dict[datetime, int] = field(default_factory=dict)

    def get_available_at_time(self, time: datetime) -> int:
        """Get available resources at specific time"""
        return self.availability_schedule.get(time, self.total_available)


@dataclass
class Scenario:
    """Scenario definition for what-if analysis"""
    scenario_id: str
    name: str
    description: str
    maintenance_strategy: MaintenanceStrategy
    resource_constraints: Dict[ResourceType, Resource]
    operational_parameters: Dict[str, Any]
    time_horizon: int  # Hours
    budget_limit: Optional[float] = None
    performance_targets: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'name': self.name,
            'description': self.description,
            'maintenance_strategy': self.maintenance_strategy.value,
            'time_horizon': self.time_horizon,
            'budget_limit': self.budget_limit,
            'performance_targets': self.performance_targets,
            'operational_parameters': self.operational_parameters
        }


@dataclass
class ScenarioResult:
    """Results from scenario simulation"""
    scenario_id: str
    total_cost: float
    maintenance_cost: float
    downtime_cost: float
    failure_cost: float
    system_availability: float
    mean_time_between_failures: float
    total_failures: int
    unplanned_downtime: float  # Hours
    planned_downtime: float  # Hours
    resource_utilization: Dict[ResourceType, float]
    maintenance_actions_completed: int
    maintenance_actions_deferred: int
    performance_metrics: Dict[str, float]
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'total_cost': self.total_cost,
            'maintenance_cost': self.maintenance_cost,
            'downtime_cost': self.downtime_cost,
            'failure_cost': self.failure_cost,
            'system_availability': self.system_availability,
            'mean_time_between_failures': self.mean_time_between_failures,
            'total_failures': self.total_failures,
            'unplanned_downtime': self.unplanned_downtime,
            'planned_downtime': self.planned_downtime,
            'resource_utilization': {k.value: v for k, v in self.resource_utilization.items()},
            'maintenance_actions_completed': self.maintenance_actions_completed,
            'maintenance_actions_deferred': self.maintenance_actions_deferred,
            'performance_metrics': self.performance_metrics,
            'timeline_events': len(self.timeline)
        }


class MaintenanceScheduleOptimizer:
    """Optimization engine for maintenance scheduling"""

    def __init__(self):
        self.solver_timeout = 300  # 5 minutes

    def optimize_schedule(self,
                         maintenance_actions: List[MaintenanceAction],
                         resources: Dict[ResourceType, Resource],
                         constraints: Dict[str, Any],
                         objective: str = "minimize_cost") -> List[MaintenanceAction]:
        """
        Optimize maintenance schedule using linear programming

        Args:
            maintenance_actions: List of potential maintenance actions
            resources: Available resources
            constraints: Additional constraints
            objective: Optimization objective

        Returns:
            Optimized list of maintenance actions
        """

        # Create optimization problem
        prob = LpProblem("MaintenanceScheduling", LpMinimize)

        # Decision variables: whether to schedule each action
        action_vars = {}
        time_vars = {}

        for action in maintenance_actions:
            action_vars[action.action_id] = LpVariable(
                f"action_{action.action_id}", cat='Binary'
            )
            # Time slot variables (discretized)
            time_slots = range(0, constraints.get('time_horizon', 168), 1)  # Hourly slots
            time_vars[action.action_id] = {
                slot: LpVariable(f"time_{action.action_id}_{slot}", cat='Binary')
                for slot in time_slots
            }

        # Objective function
        if objective == "minimize_cost":
            prob += lpSum([
                action_vars[action.action_id] * action.cost
                for action in maintenance_actions
            ])
        elif objective == "minimize_downtime":
            prob += lpSum([
                action_vars[action.action_id] * action.duration
                for action in maintenance_actions
            ])

        # Constraints

        # 1. Resource constraints
        for resource_type, resource in resources.items():
            for hour in range(constraints.get('time_horizon', 168)):
                prob += lpSum([
                    lpSum([
                        time_vars[action.action_id][slot] * action.required_resources.get(resource_type, 0)
                        for slot in range(max(0, hour - int(action.duration)), hour + 1)
                    ])
                    for action in maintenance_actions
                    if hour < len(time_vars[action.action_id])
                ]) <= resource.get_available_at_time(datetime.now() + timedelta(hours=hour))

        # 2. Action scheduling constraints
        for action in maintenance_actions:
            # Each action can be scheduled at most once
            prob += lpSum([
                time_vars[action.action_id][slot]
                for slot in time_vars[action.action_id]
            ]) <= action_vars[action.action_id]

            # If action is selected, it must be scheduled at exactly one time
            prob += lpSum([
                time_vars[action.action_id][slot]
                for slot in time_vars[action.action_id]
            ]) == action_vars[action.action_id]

        # 3. Budget constraints
        if constraints.get('budget_limit'):
            prob += lpSum([
                action_vars[action.action_id] * action.cost
                for action in maintenance_actions
            ]) <= constraints['budget_limit']

        # Solve the problem
        prob.solve()

        # Extract solution
        optimized_actions = []
        for action in maintenance_actions:
            if action_vars[action.action_id].varValue == 1:
                # Find scheduled time
                for slot, var in time_vars[action.action_id].items():
                    if var.varValue == 1:
                        action.scheduled_time = datetime.now() + timedelta(hours=slot)
                        break
                optimized_actions.append(action)

        return optimized_actions


class ScenarioSimulator:
    """Monte Carlo simulation engine for scenario analysis"""

    def __init__(self,
                 failure_estimator: FailureProbabilityEstimator,
                 forecaster: EnhancedForecaster):
        """
        Initialize scenario simulator

        Args:
            failure_estimator: Failure probability estimation engine
            forecaster: Enhanced forecasting model
        """
        self.failure_estimator = failure_estimator
        self.forecaster = forecaster
        self.scheduler_optimizer = MaintenanceScheduleOptimizer()

    def simulate_scenario(self,
                         scenario: Scenario,
                         equipment_data: Dict[str, np.ndarray],
                         current_failures: Dict[str, List[FailurePrediction]],
                         n_simulations: int = 1000) -> ScenarioResult:
        """
        Simulate a maintenance scenario

        Args:
            scenario: Scenario definition
            equipment_data: Current equipment sensor data
            current_failures: Current failure predictions
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Scenario simulation results
        """
        logger.info(f"Simulating scenario: {scenario.name}")

        # Initialize result accumulators
        results = {
            'total_costs': [],
            'maintenance_costs': [],
            'failure_costs': [],
            'downtimes': [],
            'availabilities': [],
            'failure_counts': [],
            'completed_actions': [],
            'deferred_actions': []
        }

        # Run Monte Carlo simulations
        for sim in range(n_simulations):
            sim_result = self._run_single_simulation(
                scenario, equipment_data, current_failures
            )

            results['total_costs'].append(sim_result['total_cost'])
            results['maintenance_costs'].append(sim_result['maintenance_cost'])
            results['failure_costs'].append(sim_result['failure_cost'])
            results['downtimes'].append(sim_result['downtime'])
            results['availabilities'].append(sim_result['availability'])
            results['failure_counts'].append(sim_result['failure_count'])
            results['completed_actions'].append(sim_result['completed_actions'])
            results['deferred_actions'].append(sim_result['deferred_actions'])

        # Aggregate results
        return self._aggregate_simulation_results(scenario, results)

    def _run_single_simulation(self,
                              scenario: Scenario,
                              equipment_data: Dict[str, np.ndarray],
                              current_failures: Dict[str, List[FailurePrediction]]) -> Dict[str, Any]:
        """Run a single Monte Carlo simulation"""

        # Initialize simulation state
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=scenario.time_horizon)

        total_cost = 0.0
        maintenance_cost = 0.0
        failure_cost = 0.0
        downtime_hours = 0.0
        failure_count = 0
        completed_actions = 0
        deferred_actions = 0

        # Generate maintenance actions based on strategy
        maintenance_actions = self._generate_maintenance_actions(
            scenario, current_failures
        )

        # Optimize maintenance schedule
        if maintenance_actions:
            optimized_actions = self.scheduler_optimizer.optimize_schedule(
                maintenance_actions,
                scenario.resource_constraints,
                {'time_horizon': scenario.time_horizon, 'budget_limit': scenario.budget_limit}
            )
        else:
            optimized_actions = []

        # Simulate time progression
        simulation_time = current_time
        hourly_step = timedelta(hours=1)

        while simulation_time < end_time:
            # Check for scheduled maintenance
            for action in optimized_actions:
                if (action.scheduled_time <= simulation_time <
                    action.scheduled_time + timedelta(hours=action.duration)):

                    # Execute maintenance action
                    maintenance_cost += action.cost
                    total_cost += action.cost
                    downtime_hours += action.duration
                    completed_actions += 1

                    # Update failure probabilities (maintenance effect)
                    self._apply_maintenance_effect(current_failures, action)

            # Check for random failures
            failures_this_hour = self._simulate_failures(current_failures, simulation_time)

            for failure in failures_this_hour:
                failure_count += 1

                # Calculate failure costs
                component_cost = self._calculate_failure_cost(failure, scenario)
                failure_cost += component_cost
                total_cost += component_cost

                # Add unplanned downtime
                downtime_hours += self._calculate_failure_downtime(failure)

            simulation_time += hourly_step

        # Calculate availability
        total_hours = scenario.time_horizon
        availability = max(0, (total_hours - downtime_hours) / total_hours)

        return {
            'total_cost': total_cost,
            'maintenance_cost': maintenance_cost,
            'failure_cost': failure_cost,
            'downtime': downtime_hours,
            'availability': availability,
            'failure_count': failure_count,
            'completed_actions': completed_actions,
            'deferred_actions': deferred_actions
        }

    def _generate_maintenance_actions(self,
                                    scenario: Scenario,
                                    current_failures: Dict[str, List[FailurePrediction]]) -> List[MaintenanceAction]:
        """Generate maintenance actions based on strategy"""

        actions = []
        action_counter = 0

        for equipment_id, failures in current_failures.items():
            for failure in failures:
                if scenario.maintenance_strategy == MaintenanceStrategy.REACTIVE:
                    # Only repair after failure
                    if failure.failure_probability > 0.9:
                        actions.append(self._create_repair_action(
                            equipment_id, failure, action_counter
                        ))
                        action_counter += 1

                elif scenario.maintenance_strategy == MaintenanceStrategy.PREVENTIVE:
                    # Scheduled maintenance regardless of condition
                    actions.append(self._create_preventive_action(
                        equipment_id, failure, action_counter
                    ))
                    action_counter += 1

                elif scenario.maintenance_strategy == MaintenanceStrategy.PREDICTIVE:
                    # Condition-based maintenance
                    if failure.failure_probability > 0.3:
                        if failure.failure_probability > 0.7:
                            actions.append(self._create_repair_action(
                                equipment_id, failure, action_counter
                            ))
                        else:
                            actions.append(self._create_inspection_action(
                                equipment_id, failure, action_counter
                            ))
                        action_counter += 1

                elif scenario.maintenance_strategy == MaintenanceStrategy.HYBRID:
                    # Combination approach
                    if failure.severity.value in ['critical', 'high']:
                        if failure.failure_probability > 0.4:
                            actions.append(self._create_repair_action(
                                equipment_id, failure, action_counter
                            ))
                        else:
                            actions.append(self._create_preventive_action(
                                equipment_id, failure, action_counter
                            ))
                    else:
                        if failure.failure_probability > 0.6:
                            actions.append(self._create_inspection_action(
                                equipment_id, failure, action_counter
                            ))
                    action_counter += 1

        return actions

    def _create_repair_action(self,
                            equipment_id: str,
                            failure: FailurePrediction,
                            action_id: int) -> MaintenanceAction:
        """Create repair action"""

        base_cost = 1000  # Base repair cost
        duration = 4  # Hours

        # Adjust based on severity
        severity_multipliers = {
            'critical': 2.0,
            'high': 1.5,
            'medium': 1.0,
            'low': 0.7
        }

        multiplier = severity_multipliers.get(failure.severity.value, 1.0)

        return MaintenanceAction(
            action_id=f"repair_{action_id}",
            equipment_id=equipment_id,
            component_id=failure.component_id,
            action_type="repair",
            scheduled_time=datetime.now(),  # Will be optimized
            duration=duration * multiplier,
            cost=base_cost * multiplier,
            required_resources={
                ResourceType.TECHNICIAN: 2,
                ResourceType.SPARE_PARTS: 1
            },
            effectiveness=0.8  # Reduces failure probability by 80%
        )

    def _create_preventive_action(self,
                                equipment_id: str,
                                failure: FailurePrediction,
                                action_id: int) -> MaintenanceAction:
        """Create preventive maintenance action"""

        return MaintenanceAction(
            action_id=f"preventive_{action_id}",
            equipment_id=equipment_id,
            component_id=failure.component_id,
            action_type="preventive",
            scheduled_time=datetime.now(),
            duration=2,  # Hours
            cost=500,
            required_resources={
                ResourceType.TECHNICIAN: 1
            },
            effectiveness=0.5  # Reduces failure probability by 50%
        )

    def _create_inspection_action(self,
                                equipment_id: str,
                                failure: FailurePrediction,
                                action_id: int) -> MaintenanceAction:
        """Create inspection action"""

        return MaintenanceAction(
            action_id=f"inspection_{action_id}",
            equipment_id=equipment_id,
            component_id=failure.component_id,
            action_type="inspection",
            scheduled_time=datetime.now(),
            duration=1,  # Hours
            cost=200,
            required_resources={
                ResourceType.TECHNICIAN: 1
            },
            effectiveness=0.2  # Reduces failure probability by 20%
        )

    def _apply_maintenance_effect(self,
                                current_failures: Dict[str, List[FailurePrediction]],
                                action: MaintenanceAction):
        """Apply maintenance effect to failure probabilities"""

        for equipment_id, failures in current_failures.items():
            if equipment_id == action.equipment_id:
                for failure in failures:
                    if failure.component_id == action.component_id:
                        # Reduce failure probability
                        failure.failure_probability *= (1 - action.effectiveness)

                        # Update time to failure
                        if failure.time_to_failure:
                            failure.time_to_failure *= (1 + action.effectiveness)

    def _simulate_failures(self,
                          current_failures: Dict[str, List[FailurePrediction]],
                          current_time: datetime) -> List[FailurePrediction]:
        """Simulate random failures based on probabilities"""

        failures = []

        for equipment_id, failure_list in current_failures.items():
            for failure in failure_list:
                # Check if failure occurs this hour
                hourly_failure_prob = failure.failure_probability / (24 * 7)  # Weekly to hourly

                if np.random.random() < hourly_failure_prob:
                    failures.append(failure)

        return failures

    def _calculate_failure_cost(self,
                              failure: FailurePrediction,
                              scenario: Scenario) -> float:
        """Calculate cost of a failure"""

        base_costs = {
            'critical': 10000,
            'high': 5000,
            'medium': 2000,
            'low': 500
        }

        base_cost = base_costs.get(failure.severity.value, 2000)

        # Add downtime cost
        downtime_hours = self._calculate_failure_downtime(failure)
        downtime_cost_per_hour = scenario.operational_parameters.get('downtime_cost_per_hour', 500)

        total_cost = base_cost + (downtime_hours * downtime_cost_per_hour)

        return total_cost

    def _calculate_failure_downtime(self, failure: FailurePrediction) -> float:
        """Calculate downtime hours from a failure"""

        base_downtime = {
            'critical': 24,  # 24 hours
            'high': 8,
            'medium': 4,
            'low': 1
        }

        return base_downtime.get(failure.severity.value, 4)

    def _aggregate_simulation_results(self,
                                    scenario: Scenario,
                                    results: Dict[str, List]) -> ScenarioResult:
        """Aggregate Monte Carlo simulation results"""

        # Calculate means and confidence intervals
        total_cost = np.mean(results['total_costs'])
        maintenance_cost = np.mean(results['maintenance_costs'])
        failure_cost = np.mean(results['failure_costs'])
        downtime = np.mean(results['downtimes'])
        availability = np.mean(results['availabilities'])
        failure_count = np.mean(results['failure_counts'])

        # Calculate resource utilization (simplified)
        resource_utilization = {}
        for resource_type in scenario.resource_constraints:
            utilization = min(maintenance_cost / 10000, 1.0)  # Simplified calculation
            resource_utilization[resource_type] = utilization

        # Calculate MTBF
        mtbf = scenario.time_horizon / max(failure_count, 1)

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            total_cost=total_cost,
            maintenance_cost=maintenance_cost,
            downtime_cost=total_cost - maintenance_cost - failure_cost,
            failure_cost=failure_cost,
            system_availability=availability,
            mean_time_between_failures=mtbf,
            total_failures=int(failure_count),
            unplanned_downtime=downtime * 0.7,  # Assume 70% unplanned
            planned_downtime=downtime * 0.3,  # Assume 30% planned
            resource_utilization=resource_utilization,
            maintenance_actions_completed=int(np.mean(results['completed_actions'])),
            maintenance_actions_deferred=int(np.mean(results['deferred_actions'])),
            performance_metrics={
                'cost_per_hour': total_cost / scenario.time_horizon,
                'availability_target_met': availability >= scenario.performance_targets.get('availability', 0.95),
                'cost_efficiency': maintenance_cost / total_cost if total_cost > 0 else 0
            }
        )


class WhatIfAnalyzer:
    """Main what-if analysis engine"""

    def __init__(self,
                 failure_estimator: FailureProbabilityEstimator,
                 forecaster: EnhancedForecaster):
        """
        Initialize what-if analyzer

        Args:
            failure_estimator: Failure probability estimation engine
            forecaster: Enhanced forecasting model
        """
        self.failure_estimator = failure_estimator
        self.forecaster = forecaster
        self.simulator = ScenarioSimulator(failure_estimator, forecaster)
        self.scenarios = {}
        self.results = {}

    def create_scenario(self,
                       scenario_id: str,
                       name: str,
                       description: str,
                       maintenance_strategy: MaintenanceStrategy,
                       time_horizon: int = 168,
                       budget_limit: Optional[float] = None,
                       resource_config: Optional[Dict] = None) -> Scenario:
        """Create a new scenario for analysis"""

        # Default resource configuration
        if resource_config is None:
            resource_config = {
                ResourceType.TECHNICIAN: Resource(
                    resource_type=ResourceType.TECHNICIAN,
                    total_available=5,
                    hourly_cost=50
                ),
                ResourceType.SPARE_PARTS: Resource(
                    resource_type=ResourceType.SPARE_PARTS,
                    total_available=100,
                    hourly_cost=10
                ),
                ResourceType.BUDGET: Resource(
                    resource_type=ResourceType.BUDGET,
                    total_available=1000000,
                    hourly_cost=0
                )
            }

        scenario = Scenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            maintenance_strategy=maintenance_strategy,
            resource_constraints=resource_config,
            operational_parameters={
                'downtime_cost_per_hour': 500,
                'emergency_multiplier': 2.0,
                'preventive_discount': 0.7
            },
            time_horizon=time_horizon,
            budget_limit=budget_limit,
            performance_targets={
                'availability': 0.95,
                'max_cost': budget_limit if budget_limit else float('inf')
            }
        )

        self.scenarios[scenario_id] = scenario
        return scenario

    def run_scenario_analysis(self,
                            scenario_id: str,
                            equipment_data: Dict[str, np.ndarray],
                            current_failures: Dict[str, List[FailurePrediction]],
                            n_simulations: int = 1000) -> ScenarioResult:
        """Run what-if analysis for a specific scenario"""

        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")

        scenario = self.scenarios[scenario_id]
        result = self.simulator.simulate_scenario(
            scenario, equipment_data, current_failures, n_simulations
        )

        self.results[scenario_id] = result
        return result

    def compare_scenarios(self,
                         scenario_ids: List[str],
                         metrics: List[str] = None) -> pd.DataFrame:
        """Compare multiple scenarios"""

        if metrics is None:
            metrics = [
                'total_cost', 'system_availability', 'total_failures',
                'maintenance_cost', 'failure_cost', 'unplanned_downtime'
            ]

        comparison_data = []

        for scenario_id in scenario_ids:
            if scenario_id not in self.results:
                logger.warning(f"No results found for scenario {scenario_id}")
                continue

            result = self.results[scenario_id]
            scenario = self.scenarios[scenario_id]

            row = {
                'scenario_id': scenario_id,
                'name': scenario.name,
                'strategy': scenario.maintenance_strategy.value
            }

            for metric in metrics:
                if hasattr(result, metric):
                    row[metric] = getattr(result, metric)

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def optimize_maintenance_strategy(self,
                                    equipment_data: Dict[str, np.ndarray],
                                    current_failures: Dict[str, List[FailurePrediction]],
                                    constraints: Dict[str, Any]) -> Tuple[Scenario, ScenarioResult]:
        """Find optimal maintenance strategy using optimization"""

        # Define parameter space for optimization
        strategies = [MaintenanceStrategy.REACTIVE, MaintenanceStrategy.PREVENTIVE,
                     MaintenanceStrategy.PREDICTIVE, MaintenanceStrategy.HYBRID]

        budget_range = constraints.get('budget_range', (10000, 100000))
        technician_range = constraints.get('technician_range', (1, 10))

        best_scenario = None
        best_result = None
        best_score = float('inf')  # Assuming we want to minimize cost

        # Grid search optimization
        for strategy in strategies:
            for budget in np.linspace(budget_range[0], budget_range[1], 5):
                for n_technicians in range(technician_range[0], technician_range[1] + 1):

                    # Create scenario
                    scenario_id = f"opt_{strategy.value}_{int(budget)}_{n_technicians}"

                    resource_config = {
                        ResourceType.TECHNICIAN: Resource(
                            resource_type=ResourceType.TECHNICIAN,
                            total_available=n_technicians,
                            hourly_cost=50
                        ),
                        ResourceType.SPARE_PARTS: Resource(
                            resource_type=ResourceType.SPARE_PARTS,
                            total_available=100,
                            hourly_cost=10
                        )
                    }

                    scenario = self.create_scenario(
                        scenario_id=scenario_id,
                        name=f"Optimized {strategy.value}",
                        description=f"Auto-generated scenario for optimization",
                        maintenance_strategy=strategy,
                        budget_limit=budget,
                        resource_config=resource_config
                    )

                    # Run simulation
                    result = self.run_scenario_analysis(
                        scenario_id, equipment_data, current_failures, n_simulations=100
                    )

                    # Calculate optimization score
                    score = self._calculate_optimization_score(result, constraints)

                    if score < best_score:
                        best_score = score
                        best_scenario = scenario
                        best_result = result

        return best_scenario, best_result

    def _calculate_optimization_score(self,
                                    result: ScenarioResult,
                                    constraints: Dict[str, Any]) -> float:
        """Calculate optimization score for scenario ranking"""

        # Multi-objective optimization score
        weights = constraints.get('weights', {
            'cost': 0.4,
            'availability': 0.3,
            'failures': 0.2,
            'downtime': 0.1
        })

        # Normalize metrics (lower is better)
        cost_score = result.total_cost / 100000  # Normalize to ~1
        availability_score = (1 - result.system_availability) * 10  # Invert availability
        failure_score = result.total_failures / 10  # Normalize
        downtime_score = result.unplanned_downtime / 100  # Normalize

        total_score = (
            weights['cost'] * cost_score +
            weights['availability'] * availability_score +
            weights['failures'] * failure_score +
            weights['downtime'] * downtime_score
        )

        return total_score

    def generate_recommendations(self,
                               scenario_ids: List[str]) -> Dict[str, Any]:
        """Generate recommendations based on scenario analysis"""

        if not scenario_ids:
            return {'error': 'No scenarios provided'}

        comparison_df = self.compare_scenarios(scenario_ids)

        recommendations = {
            'best_overall': None,
            'most_cost_effective': None,
            'highest_availability': None,
            'lowest_risk': None,
            'insights': [],
            'summary': {}
        }

        if not comparison_df.empty:
            # Best overall (lowest total cost)
            best_idx = comparison_df['total_cost'].idxmin()
            recommendations['best_overall'] = comparison_df.loc[best_idx].to_dict()

            # Most cost effective (lowest cost per availability point)
            comparison_df['cost_effectiveness'] = (
                comparison_df['total_cost'] / comparison_df['system_availability']
            )
            cost_effective_idx = comparison_df['cost_effectiveness'].idxmin()
            recommendations['most_cost_effective'] = comparison_df.loc[cost_effective_idx].to_dict()

            # Highest availability
            availability_idx = comparison_df['system_availability'].idxmax()
            recommendations['highest_availability'] = comparison_df.loc[availability_idx].to_dict()

            # Lowest risk (fewest failures)
            risk_idx = comparison_df['total_failures'].idxmin()
            recommendations['lowest_risk'] = comparison_df.loc[risk_idx].to_dict()

            # Generate insights
            recommendations['insights'] = self._generate_insights(comparison_df)

            # Summary statistics
            recommendations['summary'] = {
                'scenarios_analyzed': len(scenario_ids),
                'avg_cost': comparison_df['total_cost'].mean(),
                'avg_availability': comparison_df['system_availability'].mean(),
                'cost_range': (comparison_df['total_cost'].min(), comparison_df['total_cost'].max()),
                'availability_range': (comparison_df['system_availability'].min(),
                                     comparison_df['system_availability'].max())
            }

        return recommendations

    def _generate_insights(self, comparison_df: pd.DataFrame) -> List[str]:
        """Generate textual insights from scenario comparison"""

        insights = []

        # Strategy performance insights
        strategy_performance = comparison_df.groupby('strategy').agg({
            'total_cost': 'mean',
            'system_availability': 'mean',
            'total_failures': 'mean'
        })

        best_strategy = strategy_performance['total_cost'].idxmin()
        insights.append(f"The {best_strategy} strategy shows the lowest average total cost.")

        highest_availability_strategy = strategy_performance['system_availability'].idxmax()
        insights.append(f"The {highest_availability_strategy} strategy achieves the highest average availability.")

        # Cost vs availability trade-offs
        correlation = comparison_df['total_cost'].corr(comparison_df['system_availability'])
        if correlation > 0.5:
            insights.append("Higher costs are associated with better availability - consider budget increases for critical systems.")
        elif correlation < -0.5:
            insights.append("Higher costs may not translate to better availability - review maintenance efficiency.")

        # Failure pattern insights
        if comparison_df['total_failures'].std() > comparison_df['total_failures'].mean() * 0.3:
            insights.append("High variability in failure rates across scenarios - strategy choice significantly impacts reliability.")

        return insights

    def save_analysis_results(self, filepath: Path):
        """Save all analysis results to file"""

        export_data = {
            'scenarios': {
                sid: scenario.to_dict()
                for sid, scenario in self.scenarios.items()
            },
            'results': {
                sid: result.to_dict()
                for sid, result in self.results.items()
            },
            'analysis_timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Analysis results saved to {filepath}")


if __name__ == "__main__":
    # Demo and testing
    print("\n" + "="*60)
    print("Testing What-If Analysis Framework")
    print("="*60)

    # Mock objects for testing
    from src.forecasting.failure_probability import FailureProbabilityEstimator, FailurePrediction, SeverityLevel
    from src.forecasting.enhanced_forecaster import EnhancedForecaster

    # Create mock components
    failure_estimator = FailureProbabilityEstimator()
    config = get_config()
    forecaster = EnhancedForecaster(base_model="transformer")

    # Create analyzer
    analyzer = WhatIfAnalyzer(failure_estimator, forecaster)

    # Create test scenarios
    print("\n1. Creating test scenarios...")

    scenarios = []
    for strategy in MaintenanceStrategy:
        scenario = analyzer.create_scenario(
            scenario_id=f"test_{strategy.value}",
            name=f"Test {strategy.value.title()} Strategy",
            description=f"Testing {strategy.value} maintenance approach",
            maintenance_strategy=strategy,
            time_horizon=168,  # 1 week
            budget_limit=50000
        )
        scenarios.append(scenario.scenario_id)

    print(f"Created {len(scenarios)} test scenarios")

    # Create mock equipment data and failures
    print("\n2. Preparing mock data...")

    equipment_data = {
        'SMAP': np.random.normal(0, 1, 100),
        'MSL': np.random.normal(0, 1, 100)
    }

    current_failures = {
        'SMAP': [
            FailurePrediction(
                equipment_id='SMAP',
                component_id='power_system',
                failure_probability=0.3,
                time_to_failure=72,
                severity=SeverityLevel.HIGH
            ),
            FailurePrediction(
                equipment_id='SMAP',
                component_id='communication',
                failure_probability=0.1,
                time_to_failure=200,
                severity=SeverityLevel.MEDIUM
            )
        ],
        'MSL': [
            FailurePrediction(
                equipment_id='MSL',
                component_id='mobility_front',
                failure_probability=0.5,
                time_to_failure=48,
                severity=SeverityLevel.CRITICAL
            )
        ]
    }

    # Run scenario analysis
    print("\n3. Running scenario simulations...")

    for scenario_id in scenarios[:2]:  # Test first 2 scenarios
        print(f"  Simulating {scenario_id}...")
        result = analyzer.run_scenario_analysis(
            scenario_id,
            equipment_data,
            current_failures,
            n_simulations=100  # Reduced for demo
        )

        print(f"    Total Cost: ${result.total_cost:,.2f}")
        print(f"    Availability: {result.system_availability:.3f}")
        print(f"    Total Failures: {result.total_failures}")

    # Compare scenarios
    print("\n4. Comparing scenarios...")
    comparison = analyzer.compare_scenarios(scenarios[:2])
    print(comparison[['scenario_id', 'name', 'total_cost', 'system_availability', 'total_failures']])

    # Generate recommendations
    print("\n5. Generating recommendations...")
    recommendations = analyzer.generate_recommendations(scenarios[:2])

    if recommendations['best_overall']:
        print(f"Best Overall Strategy: {recommendations['best_overall']['name']}")
        print(f"  Total Cost: ${recommendations['best_overall']['total_cost']:,.2f}")
        print(f"  Availability: {recommendations['best_overall']['system_availability']:.3f}")

    print(f"\nKey Insights:")
    for insight in recommendations['insights']:
        print(f"  - {insight}")

    print("\n" + "="*60)
    print("What-if analysis framework test complete")
    print("="*60)