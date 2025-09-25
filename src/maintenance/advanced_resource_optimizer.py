"""
Advanced Resource Optimization Module for Phase 3.1 IoT Predictive Maintenance
Intelligent technician-equipment matching, workload balancing, and resource allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pulp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SkillRequirement:
    """Skill requirement for maintenance tasks"""
    skill_name: str
    required_level: int  # 1-5 proficiency level
    importance: float  # Weight in assignment decision (0.0-1.0)
    alternatives: List[str] = field(default_factory=list)  # Alternative skills


@dataclass
class TechnicianProfile:
    """Enhanced technician profile with skills and performance metrics"""
    technician_id: str
    name: str
    skills: Dict[str, int]  # skill_name -> proficiency_level (1-5)
    certifications: List[str]
    experience_years: float
    hourly_rate: float
    efficiency_rating: float  # Historical performance multiplier
    availability_hours: Dict[str, float]  # date -> available_hours
    preferred_shift: str  # 'morning', 'afternoon', 'night'
    location: str
    travel_time_matrix: Dict[str, float] = field(default_factory=dict)  # location -> travel_time
    workload_preference: str = "balanced"  # 'light', 'balanced', 'heavy'
    stress_tolerance: float = 1.0  # 0.5-2.0, affects peak workload capacity
    team_compatibility: Dict[str, float] = field(default_factory=dict)  # tech_id -> compatibility_score


@dataclass
class EquipmentProfile:
    """Enhanced equipment profile with maintenance requirements"""
    equipment_id: str
    name: str
    type: str
    location: str
    criticality_level: int  # 1-5 (5 = most critical)
    required_skills: List[SkillRequirement]
    maintenance_complexity: int  # 1-5 complexity rating
    safety_requirements: List[str]
    access_restrictions: Dict[str, Any]  # time windows, permissions, etc.
    historical_task_durations: Dict[str, List[float]]  # tech_id -> [durations]
    failure_impact_cost: float
    downtime_cost_per_hour: float


@dataclass
class OptimizationConstraints:
    """Constraints for resource optimization"""
    max_daily_hours: float = 8.0
    max_weekly_hours: float = 40.0
    min_skill_level_match: int = 3  # Minimum skill level for assignment
    max_travel_time: float = 2.0  # Hours
    require_certifications: bool = True
    allow_overtime: bool = False
    overtime_cost_multiplier: float = 1.5
    min_team_size: int = 1
    max_team_size: int = 3
    priority_weight: float = 0.3
    cost_weight: float = 0.25
    efficiency_weight: float = 0.25
    travel_weight: float = 0.2


class AdvancedResourceOptimizer:
    """Advanced resource optimization with ML-based assignment"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Advanced Resource Optimizer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.technicians = {}
        self.equipment = {}
        self.constraints = OptimizationConstraints()

        # ML models for optimization
        self.skill_matcher = None
        self.workload_predictor = None
        self.efficiency_model = None

        # Historical data for learning
        self.assignment_history = []
        self.performance_history = []

        # Optimization results cache
        self.optimization_cache = {}

        logger.info("Initialized Advanced Resource Optimizer")

    def add_technician(self, technician: TechnicianProfile):
        """Add technician to optimization pool

        Args:
            technician: Technician profile
        """
        self.technicians[technician.technician_id] = technician
        logger.info(f"Added technician: {technician.name}")

    def add_equipment(self, equipment: EquipmentProfile):
        """Add equipment to optimization pool

        Args:
            equipment: Equipment profile
        """
        self.equipment[equipment.equipment_id] = equipment
        logger.info(f"Added equipment: {equipment.name}")

    def calculate_skill_match_score(self, technician: TechnicianProfile,
                                  required_skills: List[SkillRequirement]) -> float:
        """Calculate skill match score between technician and requirements

        Args:
            technician: Technician profile
            required_skills: Required skills list

        Returns:
            Skill match score (0.0-1.0)
        """
        if not required_skills:
            return 1.0

        total_score = 0.0
        total_weight = 0.0

        for skill_req in required_skills:
            tech_skill_level = technician.skills.get(skill_req.skill_name, 0)

            # Check for alternative skills if primary skill not available
            if tech_skill_level == 0:
                for alt_skill in skill_req.alternatives:
                    alt_level = technician.skills.get(alt_skill, 0)
                    if alt_level > tech_skill_level:
                        tech_skill_level = alt_level * 0.8  # Penalty for using alternative

            # Calculate match score for this skill
            if tech_skill_level >= skill_req.required_level:
                skill_score = min(tech_skill_level / skill_req.required_level, 1.5)  # Bonus for over-qualification
            else:
                skill_score = tech_skill_level / skill_req.required_level * 0.5  # Penalty for under-qualification

            total_score += skill_score * skill_req.importance
            total_weight += skill_req.importance

        return total_score / total_weight if total_weight > 0 else 0.0

    def calculate_efficiency_factor(self, technician: TechnicianProfile,
                                  equipment: EquipmentProfile) -> float:
        """Calculate efficiency factor based on historical performance

        Args:
            technician: Technician profile
            equipment: Equipment profile

        Returns:
            Efficiency factor (0.5-2.0)
        """
        base_efficiency = technician.efficiency_rating

        # Adjust based on equipment-specific experience
        if technician.technician_id in equipment.historical_task_durations:
            durations = equipment.historical_task_durations[technician.technician_id]
            if durations:
                # Compare to average duration across all technicians
                all_durations = []
                for tech_durations in equipment.historical_task_durations.values():
                    all_durations.extend(tech_durations)

                if all_durations:
                    avg_duration = np.mean(all_durations)
                    tech_avg = np.mean(durations)

                    # Lower duration = higher efficiency
                    experience_factor = avg_duration / tech_avg if tech_avg > 0 else 1.0
                    base_efficiency *= experience_factor

        # Adjust for equipment complexity vs technician experience
        complexity_factor = 1.0
        if equipment.maintenance_complexity > 3:
            if technician.experience_years < 2:
                complexity_factor = 0.8
            elif technician.experience_years > 5:
                complexity_factor = 1.2

        return max(0.5, min(2.0, base_efficiency * complexity_factor))

    def calculate_cost_factor(self, technician: TechnicianProfile,
                            task_duration: float, travel_time: float = 0.0) -> float:
        """Calculate total cost for technician assignment

        Args:
            technician: Technician profile
            task_duration: Estimated task duration in hours
            travel_time: Travel time in hours

        Returns:
            Total cost in currency units
        """
        base_cost = (task_duration + travel_time) * technician.hourly_rate

        # Apply efficiency factor to adjust actual time needed
        efficiency_factor = technician.efficiency_rating
        actual_duration = task_duration / efficiency_factor

        # Check for overtime
        daily_hours = self._get_daily_hours(technician, datetime.now().date())
        if daily_hours + actual_duration + travel_time > self.constraints.max_daily_hours:
            overtime_hours = (daily_hours + actual_duration + travel_time) - self.constraints.max_daily_hours
            overtime_cost = overtime_hours * technician.hourly_rate * (self.constraints.overtime_cost_multiplier - 1)
            base_cost += overtime_cost

        return base_cost

    def optimize_assignments(self, maintenance_tasks: List[Dict[str, Any]],
                           optimization_horizon: int = 7) -> Dict[str, Any]:
        """Optimize technician assignments for multiple maintenance tasks

        Args:
            maintenance_tasks: List of maintenance task dictionaries
            optimization_horizon: Optimization horizon in days

        Returns:
            Optimization results with assignments and metrics
        """
        if not maintenance_tasks:
            return {"assignments": [], "metrics": {}, "status": "no_tasks"}

        logger.info(f"Optimizing assignments for {len(maintenance_tasks)} tasks over {optimization_horizon} days")

        # Create optimization problem
        if ORTOOLS_AVAILABLE:
            return self._optimize_with_ortools(maintenance_tasks, optimization_horizon)
        else:
            return self._optimize_with_pulp(maintenance_tasks, optimization_horizon)

    def _optimize_with_ortools(self, maintenance_tasks: List[Dict[str, Any]],
                              horizon: int) -> Dict[str, Any]:
        """Optimize using Google OR-Tools CP-SAT solver

        Args:
            maintenance_tasks: List of maintenance tasks
            horizon: Optimization horizon in days

        Returns:
            Optimization results
        """
        model = cp_model.CpModel()

        # Decision variables: task_tech[task_id][tech_id] = 1 if tech assigned to task
        task_tech = {}
        tech_ids = list(self.technicians.keys())

        for task in maintenance_tasks:
            task_id = task['task_id']
            task_tech[task_id] = {}
            for tech_id in tech_ids:
                task_tech[task_id][tech_id] = model.NewBoolVar(f'assign_{task_id}_{tech_id}')

        # Constraint: Each task assigned to exactly one technician
        for task in maintenance_tasks:
            task_id = task['task_id']
            model.Add(sum(task_tech[task_id][tech_id] for tech_id in tech_ids) == 1)

        # Constraint: Technician capacity
        for tech_id in tech_ids:
            for day in range(horizon):
                day_date = datetime.now().date() + timedelta(days=day)
                daily_tasks = [task for task in maintenance_tasks
                             if task.get('scheduled_date', datetime.now().date()) == day_date]

                if daily_tasks:
                    daily_duration = sum(
                        task_tech[task['task_id']][tech_id] * task.get('duration_hours', 2.0)
                        for task in daily_tasks
                    )
                    model.Add(daily_duration <= self.constraints.max_daily_hours)

        # Objective: Minimize weighted cost + maximize efficiency
        objective_terms = []

        for task in maintenance_tasks:
            task_id = task['task_id']
            equipment_id = task.get('equipment_id')
            equipment = self.equipment.get(equipment_id) if equipment_id else None

            for tech_id in tech_ids:
                technician = self.technicians[tech_id]

                # Calculate assignment score
                if equipment:
                    skill_score = self.calculate_skill_match_score(technician, equipment.required_skills)
                    efficiency_factor = self.calculate_efficiency_factor(technician, equipment)
                    cost = self.calculate_cost_factor(technician, task.get('duration_hours', 2.0))
                else:
                    skill_score = 0.8
                    efficiency_factor = technician.efficiency_rating
                    cost = self.calculate_cost_factor(technician, task.get('duration_hours', 2.0))

                # Convert to integer objective (OR-Tools requirement)
                score = int(1000 * (
                    self.constraints.efficiency_weight * efficiency_factor +
                    self.constraints.priority_weight * skill_score -
                    self.constraints.cost_weight * (cost / 1000)  # Normalize cost
                ))

                objective_terms.append(task_tech[task_id][tech_id] * score)

        model.Maximize(sum(objective_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30  # 30-second time limit
        status = solver.Solve(model)

        # Extract results
        assignments = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for task in maintenance_tasks:
                task_id = task['task_id']
                for tech_id in tech_ids:
                    if solver.Value(task_tech[task_id][tech_id]) == 1:
                        assignments.append({
                            'task_id': task_id,
                            'technician_id': tech_id,
                            'technician_name': self.technicians[tech_id].name,
                            'skill_match_score': self.calculate_skill_match_score(
                                self.technicians[tech_id],
                                self.equipment.get(task.get('equipment_id'), EquipmentProfile('', '', '', '', 0, [], 0, [], {}, {}, 0, 0)).required_skills
                            ),
                            'estimated_cost': self.calculate_cost_factor(
                                self.technicians[tech_id],
                                task.get('duration_hours', 2.0)
                            )
                        })

        return {
            "assignments": assignments,
            "metrics": {
                "optimization_status": "optimal" if status == cp_model.OPTIMAL else "feasible" if status == cp_model.FEASIBLE else "failed",
                "total_cost": sum(a['estimated_cost'] for a in assignments),
                "average_skill_match": np.mean([a['skill_match_score'] for a in assignments]) if assignments else 0,
                "solver_time": solver.WallTime()
            },
            "status": "success" if assignments else "failed"
        }

    def _optimize_with_pulp(self, maintenance_tasks: List[Dict[str, Any]],
                           horizon: int) -> Dict[str, Any]:
        """Optimize using PuLP linear programming solver

        Args:
            maintenance_tasks: List of maintenance tasks
            horizon: Optimization horizon in days

        Returns:
            Optimization results
        """
        # Create optimization problem
        prob = pulp.LpProblem("MaintenanceOptimization", pulp.LpMaximize)

        # Decision variables
        task_tech_vars = {}
        tech_ids = list(self.technicians.keys())

        for task in maintenance_tasks:
            task_id = task['task_id']
            task_tech_vars[task_id] = {}
            for tech_id in tech_ids:
                task_tech_vars[task_id][tech_id] = pulp.LpVariable(
                    f"assign_{task_id}_{tech_id}", cat='Binary'
                )

        # Constraint: Each task assigned to exactly one technician
        for task in maintenance_tasks:
            task_id = task['task_id']
            prob += pulp.lpSum(task_tech_vars[task_id][tech_id] for tech_id in tech_ids) == 1

        # Constraint: Technician daily capacity
        for tech_id in tech_ids:
            for day in range(horizon):
                day_date = datetime.now().date() + timedelta(days=day)
                daily_tasks = [task for task in maintenance_tasks
                             if task.get('scheduled_date', datetime.now().date()) == day_date]

                if daily_tasks:
                    prob += pulp.lpSum(
                        task_tech_vars[task['task_id']][tech_id] * task.get('duration_hours', 2.0)
                        for task in daily_tasks
                    ) <= self.constraints.max_daily_hours

        # Objective function
        objective_terms = []

        for task in maintenance_tasks:
            task_id = task['task_id']
            equipment_id = task.get('equipment_id')
            equipment = self.equipment.get(equipment_id) if equipment_id else None

            for tech_id in tech_ids:
                technician = self.technicians[tech_id]

                # Calculate assignment score components
                if equipment:
                    skill_score = self.calculate_skill_match_score(technician, equipment.required_skills)
                    efficiency_factor = self.calculate_efficiency_factor(technician, equipment)
                    cost = self.calculate_cost_factor(technician, task.get('duration_hours', 2.0))
                else:
                    skill_score = 0.8
                    efficiency_factor = technician.efficiency_rating
                    cost = self.calculate_cost_factor(technician, task.get('duration_hours', 2.0))

                # Combine into single score
                score = (
                    self.constraints.efficiency_weight * efficiency_factor +
                    self.constraints.priority_weight * skill_score -
                    self.constraints.cost_weight * (cost / 1000)  # Normalize cost
                )

                objective_terms.append(task_tech_vars[task_id][tech_id] * score)

        prob += pulp.lpSum(objective_terms)

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract results
        assignments = []
        if prob.status == pulp.LpStatusOptimal:
            for task in maintenance_tasks:
                task_id = task['task_id']
                for tech_id in tech_ids:
                    if task_tech_vars[task_id][tech_id].varValue == 1:
                        equipment_id = task.get('equipment_id')
                        equipment = self.equipment.get(equipment_id) if equipment_id else None
                        technician = self.technicians[tech_id]

                        skill_match = self.calculate_skill_match_score(
                            technician,
                            equipment.required_skills if equipment else []
                        )

                        assignments.append({
                            'task_id': task_id,
                            'technician_id': tech_id,
                            'technician_name': technician.name,
                            'skill_match_score': skill_match,
                            'estimated_cost': self.calculate_cost_factor(
                                technician,
                                task.get('duration_hours', 2.0)
                            ),
                            'efficiency_factor': self.calculate_efficiency_factor(technician, equipment) if equipment else technician.efficiency_rating
                        })

        return {
            "assignments": assignments,
            "metrics": {
                "optimization_status": "optimal" if prob.status == pulp.LpStatusOptimal else "failed",
                "total_cost": sum(a['estimated_cost'] for a in assignments),
                "average_skill_match": np.mean([a['skill_match_score'] for a in assignments]) if assignments else 0,
                "average_efficiency": np.mean([a['efficiency_factor'] for a in assignments]) if assignments else 0,
                "optimization_score": prob.objective.value() if prob.objective else 0
            },
            "status": "success" if assignments else "failed"
        }

    def balance_workload(self, assignments: List[Dict[str, Any]],
                        time_window_days: int = 7) -> List[Dict[str, Any]]:
        """Balance workload across technicians

        Args:
            assignments: Current assignments
            time_window_days: Time window for balancing

        Returns:
            Balanced assignments
        """
        if not assignments:
            return assignments

        # Calculate current workload distribution
        tech_workloads = defaultdict(float)
        for assignment in assignments:
            tech_id = assignment['technician_id']
            task_duration = assignment.get('task_duration', 2.0)
            tech_workloads[tech_id] += task_duration

        # Calculate target workload per technician
        total_workload = sum(tech_workloads.values())
        num_technicians = len(self.technicians)
        target_workload = total_workload / num_technicians if num_technicians > 0 else 0

        # Identify over/under-loaded technicians
        overloaded = [(tech_id, workload) for tech_id, workload in tech_workloads.items()
                     if workload > target_workload * 1.2]
        underloaded = [(tech_id, workload) for tech_id, workload in tech_workloads.items()
                      if workload < target_workload * 0.8]

        # Rebalance assignments
        balanced_assignments = assignments.copy()

        for overloaded_tech, overload in overloaded:
            excess_workload = overload - target_workload

            # Find tasks that can be reassigned
            tech_tasks = [a for a in balanced_assignments if a['technician_id'] == overloaded_tech]
            tech_tasks.sort(key=lambda x: x.get('skill_match_score', 0.5))  # Reassign lower-skill matches first

            for task in tech_tasks:
                if excess_workload <= 0:
                    break

                # Try to reassign to underloaded technician
                for underloaded_tech, underload in underloaded:
                    if target_workload - underload >= task.get('task_duration', 2.0):
                        # Check if underloaded tech can handle the task
                        if self._can_handle_task(underloaded_tech, task):
                            # Reassign task
                            task['technician_id'] = underloaded_tech
                            task['technician_name'] = self.technicians[underloaded_tech].name

                            # Update workload tracking
                            excess_workload -= task.get('task_duration', 2.0)
                            underload += task.get('task_duration', 2.0)
                            break

        return balanced_assignments

    def _can_handle_task(self, technician_id: str, task: Dict[str, Any]) -> bool:
        """Check if technician can handle a specific task

        Args:
            technician_id: Technician ID
            task: Task dictionary

        Returns:
            True if technician can handle the task
        """
        technician = self.technicians.get(technician_id)
        if not technician:
            return False

        equipment_id = task.get('equipment_id')
        if not equipment_id:
            return True

        equipment = self.equipment.get(equipment_id)
        if not equipment:
            return True

        # Check skill requirements
        skill_score = self.calculate_skill_match_score(technician, equipment.required_skills)
        return skill_score >= 0.6  # Minimum acceptable skill match

    def _get_daily_hours(self, technician: TechnicianProfile, date: datetime.date) -> float:
        """Get current daily hours for technician

        Args:
            technician: Technician profile
            date: Date to check

        Returns:
            Current daily hours
        """
        date_str = date.isoformat()
        return technician.availability_hours.get(date_str, 0.0)

    def generate_optimization_suggestions(self, current_assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization suggestions

        Args:
            current_assignments: Current assignment list

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Analyze current assignments
        if not current_assignments:
            return suggestions

        # Suggestion 1: Skill mismatch detection
        low_skill_matches = [a for a in current_assignments if a.get('skill_match_score', 1.0) < 0.7]
        if low_skill_matches:
            suggestions.append({
                'type': 'skill_optimization',
                'priority': 'high',
                'title': 'Improve Skill Matching',
                'description': f'{len(low_skill_matches)} tasks have suboptimal skill matches',
                'potential_benefit': 'Reduce task duration by 15-25%',
                'affected_tasks': [a['task_id'] for a in low_skill_matches]
            })

        # Suggestion 2: Cost optimization
        high_cost_tasks = [a for a in current_assignments if a.get('estimated_cost', 0) > 500]
        if high_cost_tasks:
            suggestions.append({
                'type': 'cost_optimization',
                'priority': 'medium',
                'title': 'Reduce High-Cost Assignments',
                'description': f'{len(high_cost_tasks)} tasks have high estimated costs',
                'potential_benefit': f'Potential savings: ${sum(a.get("estimated_cost", 0) for a in high_cost_tasks) * 0.1:.0f}',
                'affected_tasks': [a['task_id'] for a in high_cost_tasks]
            })

        # Suggestion 3: Workload balancing
        tech_workloads = defaultdict(float)
        for assignment in current_assignments:
            tech_id = assignment['technician_id']
            tech_workloads[tech_id] += assignment.get('task_duration', 2.0)

        if tech_workloads:
            workload_variance = np.var(list(tech_workloads.values()))
            if workload_variance > 4.0:  # High variance threshold
                suggestions.append({
                    'type': 'workload_balancing',
                    'priority': 'medium',
                    'title': 'Balance Technician Workloads',
                    'description': 'Uneven workload distribution detected',
                    'potential_benefit': 'Improve team efficiency and satisfaction',
                    'affected_tasks': []
                })

        return suggestions

    def get_resource_utilization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource utilization metrics

        Returns:
            Resource utilization metrics
        """
        metrics = {
            'technician_utilization': {},
            'skill_distribution': {},
            'cost_efficiency': {},
            'overall_metrics': {}
        }

        # Technician utilization
        for tech_id, technician in self.technicians.items():
            daily_hours = sum(technician.availability_hours.values()) / len(technician.availability_hours) if technician.availability_hours else 0
            utilization = (daily_hours / self.constraints.max_daily_hours) * 100

            metrics['technician_utilization'][tech_id] = {
                'name': technician.name,
                'utilization_percent': min(100, utilization),
                'avg_daily_hours': daily_hours,
                'efficiency_rating': technician.efficiency_rating,
                'hourly_rate': technician.hourly_rate
            }

        # Skill distribution analysis
        all_skills = set()
        for tech in self.technicians.values():
            all_skills.update(tech.skills.keys())

        for skill in all_skills:
            skill_levels = [tech.skills.get(skill, 0) for tech in self.technicians.values() if skill in tech.skills]
            metrics['skill_distribution'][skill] = {
                'available_technicians': len(skill_levels),
                'average_level': np.mean(skill_levels) if skill_levels else 0,
                'max_level': max(skill_levels) if skill_levels else 0,
                'coverage_percent': (len(skill_levels) / len(self.technicians)) * 100
            }

        # Overall metrics
        metrics['overall_metrics'] = {
            'total_technicians': len(self.technicians),
            'average_utilization': np.mean([m['utilization_percent'] for m in metrics['technician_utilization'].values()]),
            'average_efficiency': np.mean([tech.efficiency_rating for tech in self.technicians.values()]),
            'skill_coverage': len(all_skills),
            'optimization_score': self._calculate_overall_optimization_score()
        }

        return metrics

    def _calculate_overall_optimization_score(self) -> float:
        """Calculate overall optimization score (0-100)

        Returns:
            Optimization score
        """
        if not self.technicians:
            return 0.0

        # Factors: utilization balance, skill coverage, efficiency
        utilizations = []
        for tech in self.technicians.values():
            daily_hours = sum(tech.availability_hours.values()) / len(tech.availability_hours) if tech.availability_hours else 0
            utilization = daily_hours / self.constraints.max_daily_hours
            utilizations.append(utilization)

        # Balance score (lower variance = better balance)
        balance_score = max(0, 100 - (np.var(utilizations) * 1000)) if utilizations else 0

        # Efficiency score
        efficiency_score = np.mean([tech.efficiency_rating for tech in self.technicians.values()]) * 100

        # Skill coverage score
        all_skills = set()
        for tech in self.technicians.values():
            all_skills.update(tech.skills.keys())

        skill_coverage_score = min(100, len(all_skills) * 10)  # 10 points per skill type

        # Weighted combination
        overall_score = (
            balance_score * 0.4 +
            efficiency_score * 0.4 +
            skill_coverage_score * 0.2
        )

        return min(100, max(0, overall_score))


# Initialize sample data for demonstration
def create_sample_resource_optimizer() -> AdvancedResourceOptimizer:
    """Create sample resource optimizer with demo data

    Returns:
        Configured resource optimizer
    """
    optimizer = AdvancedResourceOptimizer()

    # Sample technicians
    technicians = [
        TechnicianProfile(
            'tech_001', 'John Smith',
            {'electrical': 5, 'mechanical': 4, 'hydraulic': 3, 'programming': 2},
            ['electrical_cert', 'safety_cert'],
            8.5, 75.0, 1.2, {}, 'morning', 'Zone_A'
        ),
        TechnicianProfile(
            'tech_002', 'Jane Doe',
            {'electronics': 5, 'programming': 5, 'diagnostics': 4, 'mechanical': 3},
            ['electronics_cert', 'programming_cert'],
            6.2, 85.0, 1.1, {}, 'afternoon', 'Zone_A'
        ),
        TechnicianProfile(
            'tech_003', 'Bob Johnson',
            {'mechanical': 5, 'welding': 4, 'fabrication': 4, 'hydraulic': 3},
            ['mechanical_cert', 'welding_cert'],
            12.1, 70.0, 0.9, {}, 'morning', 'Zone_B'
        ),
        TechnicianProfile(
            'tech_004', 'Alice Brown',
            {'electrical': 4, 'automation': 5, 'sensors': 4, 'programming': 3},
            ['automation_cert', 'safety_cert'],
            4.8, 90.0, 1.3, {}, 'afternoon', 'Zone_B'
        )
    ]

    for tech in technicians:
        optimizer.add_technician(tech)

    # Sample equipment
    equipment_list = [
        EquipmentProfile(
            'EQ_001', 'Main Pump Assembly', 'pump', 'Zone_A', 5,
            [SkillRequirement('mechanical', 4, 0.8), SkillRequirement('hydraulic', 3, 0.6)],
            4, ['ppe_required'], {}, {}, 5000, 200
        ),
        EquipmentProfile(
            'EQ_002', 'Control System', 'control', 'Zone_A', 4,
            [SkillRequirement('electronics', 4, 0.9), SkillRequirement('programming', 3, 0.7)],
            3, ['clean_room'], {}, {}, 3000, 150
        ),
        EquipmentProfile(
            'EQ_003', 'Conveyor Belt', 'mechanical', 'Zone_B', 3,
            [SkillRequirement('mechanical', 3, 0.8), SkillRequirement('electrical', 2, 0.4)],
            2, [], {}, {}, 1500, 75
        )
    ]

    for equipment in equipment_list:
        optimizer.add_equipment(equipment)

    return optimizer