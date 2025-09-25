"""
Unit Tests for Maintenance Module
Tests for scheduler, optimizer, work order management, and priority calculation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import tempfile
import json
import os
from typing import List, Dict, Any, Tuple
import pulp

# Import modules to test
from src.maintenance.scheduler import MaintenanceScheduler
from src.maintenance.optimizer import MaintenanceOptimizer
from src.maintenance.work_order_manager import WorkOrderManager
from src.maintenance.priority_calculator import PriorityCalculator


class TestMaintenanceScheduler(unittest.TestCase):
    """Test cases for Maintenance Scheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = MaintenanceScheduler(
            max_technicians=5,
            work_hours=(8, 17),  # 8 AM to 5 PM
            planning_horizon_days=7
        )
        
        # Create sample work orders
        self.work_orders = pd.DataFrame({
            'work_order_id': ['WO-001', 'WO-002', 'WO-003', 'WO-004', 'WO-005'],
            'equipment_id': ['PUMP-001', 'VALVE-002', 'MOTOR-003', 'PUMP-001', 'VALVE-003'],
            'priority': ['HIGH', 'MEDIUM', 'CRITICAL', 'LOW', 'HIGH'],
            'estimated_duration': [120, 60, 180, 45, 90],  # minutes
            'required_skills': [['electrical'], ['mechanical'], ['electrical', 'mechanical'], 
                               ['mechanical'], ['electrical']],
            'due_date': [
                datetime.now() + timedelta(days=1),
                datetime.now() + timedelta(days=2),
                datetime.now() + timedelta(hours=6),
                datetime.now() + timedelta(days=3),
                datetime.now() + timedelta(days=1)
            ],
            'status': ['PENDING'] * 5
        })
        
        # Create sample technicians
        self.technicians = pd.DataFrame({
            'technician_id': ['TECH-001', 'TECH-002', 'TECH-003', 'TECH-004'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
            'skills': [['electrical', 'plumbing'], ['mechanical', 'electrical'], 
                      ['mechanical'], ['electrical', 'mechanical', 'plumbing']],
            'availability': [True, True, False, True],
            'current_load': [2, 1, 0, 3]  # Number of assigned tasks
        })
        
        # Create sample equipment
        self.equipment = pd.DataFrame({
            'equipment_id': ['PUMP-001', 'VALVE-002', 'MOTOR-003', 'VALVE-003'],
            'criticality': ['HIGH', 'MEDIUM', 'CRITICAL', 'LOW'],
            'location': ['Building A', 'Building B', 'Building A', 'Building C'],
            'downtime_cost': [1000, 500, 2000, 200]  # per hour
        })
    
    def test_initialization(self):
        """Test scheduler initialization"""
        self.assertEqual(self.scheduler.max_technicians, 5)
        self.assertEqual(self.scheduler.work_hours, (8, 17))
        self.assertEqual(self.scheduler.planning_horizon_days, 7)
        self.assertIsNotNone(self.scheduler.schedule)
    
    def test_schedule_creation(self):
        """Test creating maintenance schedule"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        self.assertIsInstance(schedule, pd.DataFrame)
        self.assertIn('work_order_id', schedule.columns)
        self.assertIn('assigned_technician', schedule.columns)
        self.assertIn('scheduled_start', schedule.columns)
        self.assertIn('scheduled_end', schedule.columns)
    
    def test_priority_based_scheduling(self):
        """Test that high priority work orders are scheduled first"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        # Critical work order should be scheduled early
        critical_wo = schedule[schedule['work_order_id'] == 'WO-003']
        if not critical_wo.empty:
            critical_start = pd.to_datetime(critical_wo['scheduled_start'].iloc[0])
            
            # Check it's scheduled before lower priority items
            low_wo = schedule[schedule['work_order_id'] == 'WO-004']
            if not low_wo.empty:
                low_start = pd.to_datetime(low_wo['scheduled_start'].iloc[0])
                self.assertLessEqual(critical_start, low_start)
    
    def test_technician_skill_matching(self):
        """Test that work orders are assigned to technicians with required skills"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        for _, assignment in schedule.iterrows():
            wo_id = assignment['work_order_id']
            tech_id = assignment['assigned_technician']
            
            if pd.notna(tech_id):
                # Get required skills for work order
                wo_skills = self.work_orders[
                    self.work_orders['work_order_id'] == wo_id
                ]['required_skills'].iloc[0]
                
                # Get technician skills
                tech_skills = self.technicians[
                    self.technicians['technician_id'] == tech_id
                ]['skills'].iloc[0]
                
                # Check skill match
                for skill in wo_skills:
                    self.assertIn(skill, tech_skills)
    
    def test_availability_constraints(self):
        """Test that unavailable technicians are not assigned"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        # TECH-003 is unavailable
        unavailable_assignments = schedule[
            schedule['assigned_technician'] == 'TECH-003'
        ]
        
        self.assertEqual(len(unavailable_assignments), 0)
    
    def test_work_hours_constraints(self):
        """Test that work is scheduled within work hours"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        for _, assignment in schedule.iterrows():
            if pd.notna(assignment['scheduled_start']):
                start_time = pd.to_datetime(assignment['scheduled_start'])
                end_time = pd.to_datetime(assignment['scheduled_end'])
                
                # Check within work hours
                self.assertGreaterEqual(start_time.hour, self.scheduler.work_hours[0])
                self.assertLessEqual(end_time.hour, self.scheduler.work_hours[1])
    
    def test_no_overlapping_assignments(self):
        """Test that technicians don't have overlapping assignments"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        # Group by technician
        for tech_id in schedule['assigned_technician'].dropna().unique():
            tech_assignments = schedule[
                schedule['assigned_technician'] == tech_id
            ].sort_values('scheduled_start')
            
            # Check for overlaps
            for i in range(len(tech_assignments) - 1):
                current_end = pd.to_datetime(tech_assignments.iloc[i]['scheduled_end'])
                next_start = pd.to_datetime(tech_assignments.iloc[i + 1]['scheduled_start'])
                
                self.assertLessEqual(current_end, next_start)
    
    def test_emergency_rescheduling(self):
        """Test emergency work order insertion"""
        # Create initial schedule
        initial_schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        # Add emergency work order
        emergency_wo = pd.DataFrame({
            'work_order_id': ['WO-EMERGENCY'],
            'equipment_id': ['PUMP-001'],
            'priority': ['EMERGENCY'],
            'estimated_duration': [60],
            'required_skills': [['electrical']],
            'due_date': [datetime.now() + timedelta(hours=2)],
            'status': ['PENDING']
        })
        
        # Reschedule with emergency
        new_schedule = self.scheduler.handle_emergency(
            emergency_wo,
            initial_schedule,
            self.technicians
        )
        
        # Emergency should be scheduled first
        emergency_assignment = new_schedule[
            new_schedule['work_order_id'] == 'WO-EMERGENCY'
        ]
        
        self.assertFalse(emergency_assignment.empty)
        
        if not emergency_assignment.empty:
            emergency_start = pd.to_datetime(emergency_assignment['scheduled_start'].iloc[0])
            
            # Should be scheduled very soon
            self.assertLess(
                (emergency_start - datetime.now()).total_seconds() / 3600,
                4  # Within 4 hours
            )
    
    def test_schedule_optimization_metrics(self):
        """Test schedule optimization metrics calculation"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        metrics = self.scheduler.calculate_schedule_metrics(
            schedule,
            self.work_orders,
            self.equipment
        )
        
        self.assertIn('total_completion_time', metrics)
        self.assertIn('technician_utilization', metrics)
        self.assertIn('priority_score', metrics)
        self.assertIn('estimated_downtime_cost', metrics)
        self.assertIn('on_time_completion_rate', metrics)
    
    def test_preventive_maintenance_scheduling(self):
        """Test scheduling of preventive maintenance"""
        # Create preventive maintenance tasks
        pm_tasks = pd.DataFrame({
            'work_order_id': ['PM-001', 'PM-002'],
            'equipment_id': ['PUMP-001', 'VALVE-002'],
            'priority': ['LOW', 'LOW'],
            'estimated_duration': [30, 45],
            'required_skills': [['mechanical'], ['mechanical']],
            'due_date': [
                datetime.now() + timedelta(days=7),
                datetime.now() + timedelta(days=14)
            ],
            'status': ['PENDING', 'PENDING'],
            'maintenance_type': ['PREVENTIVE', 'PREVENTIVE']
        })
        
        # Combine with corrective maintenance
        all_work_orders = pd.concat([self.work_orders, pm_tasks], ignore_index=True)
        
        schedule = self.scheduler.create_schedule(
            all_work_orders,
            self.technicians,
            self.equipment
        )
        
        # Preventive maintenance should be scheduled but with lower priority
        pm_scheduled = schedule[schedule['work_order_id'].isin(['PM-001', 'PM-002'])]
        
        self.assertFalse(pm_scheduled.empty)
    
    def test_schedule_export(self):
        """Test exporting schedule to different formats"""
        schedule = self.scheduler.create_schedule(
            self.work_orders,
            self.technicians,
            self.equipment
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to CSV
            csv_path = os.path.join(tmpdir, 'schedule.csv')
            self.scheduler.export_schedule(schedule, csv_path, format='csv')
            self.assertTrue(os.path.exists(csv_path))
            
            # Export to JSON
            json_path = os.path.join(tmpdir, 'schedule.json')
            self.scheduler.export_schedule(schedule, json_path, format='json')
            self.assertTrue(os.path.exists(json_path))
            
            # Export to iCal format for calendar integration
            ical_path = os.path.join(tmpdir, 'schedule.ics')
            self.scheduler.export_to_calendar(schedule, ical_path)
            self.assertTrue(os.path.exists(ical_path))


class TestMaintenanceOptimizer(unittest.TestCase):
    """Test cases for Maintenance Optimizer using PuLP"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = MaintenanceOptimizer()
        
        # Sample optimization problem data
        self.tasks = [
            {'id': 'T1', 'duration': 2, 'priority': 3, 'deadline': 5},
            {'id': 'T2', 'duration': 1, 'priority': 2, 'deadline': 3},
            {'id': 'T3', 'duration': 3, 'priority': 1, 'deadline': 8},
        ]
        
        self.resources = [
            {'id': 'R1', 'capacity': 8, 'cost': 100},
            {'id': 'R2', 'capacity': 8, 'cost': 120},
        ]
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNone(self.optimizer.problem)
    
    def test_problem_formulation(self):
        """Test linear programming problem formulation"""
        problem = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10
        )
        
        self.assertIsInstance(problem, pulp.LpProblem)
        self.assertEqual(problem.sense, pulp.LpMinimize)  # Minimization problem
        
        # Check variables exist
        variables = problem.variables()
        self.assertGreater(len(variables), 0)
    
    def test_constraint_creation(self):
        """Test constraint creation for optimization"""
        problem = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10
        )
        
        constraints = problem.constraints
        
        # Should have constraints for:
        # 1. Task completion
        # 2. Resource capacity
        # 3. Deadline constraints
        self.assertGreater(len(constraints), 0)
    
    def test_objective_function(self):
        """Test objective function formulation"""
        # Test different objective types
        
        # Minimize total cost
        problem_cost = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10,
            objective='minimize_cost'
        )
        
        # Maximize priority completion
        problem_priority = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10,
            objective='maximize_priority'
        )
        
        # Minimize makespan
        problem_makespan = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10,
            objective='minimize_makespan'
        )
        
        self.assertIsNotNone(problem_cost.objective)
        self.assertIsNotNone(problem_priority.objective)
        self.assertIsNotNone(problem_makespan.objective)
    
    def test_solve_optimization(self):
        """Test solving optimization problem"""
        problem = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10
        )
        
        # Solve problem
        solution = self.optimizer.solve(problem)
        
        self.assertIsNotNone(solution)
        self.assertIn('status', solution)
        self.assertIn('objective_value', solution)
        self.assertIn('schedule', solution)
        
        # Check if solution is optimal or feasible
        self.assertIn(solution['status'], ['Optimal', 'Feasible'])
    
    def test_resource_allocation(self):
        """Test resource allocation optimization"""
        allocation = self.optimizer.optimize_resource_allocation(
            self.tasks,
            self.resources,
            constraints={
                'max_overtime': 2,
                'min_utilization': 0.7
            }
        )
        
        self.assertIsNotNone(allocation)
        
        # Check each task is allocated
        for task in self.tasks:
            self.assertIn(task['id'], allocation)
            
            # Check resource assignment
            if allocation[task['id']]['assigned']:
                self.assertIn('resource', allocation[task['id']])
                self.assertIn('start_time', allocation[task['id']])
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization"""
        # Define multiple objectives
        objectives = [
            {'type': 'minimize', 'target': 'cost', 'weight': 0.4},
            {'type': 'maximize', 'target': 'priority', 'weight': 0.3},
            {'type': 'minimize', 'target': 'delays', 'weight': 0.3}
        ]
        
        solution = self.optimizer.multi_objective_optimization(
            self.tasks,
            self.resources,
            objectives,
            horizon=10
        )
        
        self.assertIsNotNone(solution)
        self.assertIn('pareto_front', solution)
        self.assertIn('selected_solution', solution)
    
    def test_constraint_violations(self):
        """Test handling of infeasible problems"""
        # Create an infeasible problem (impossible deadlines)
        infeasible_tasks = [
            {'id': 'T1', 'duration': 10, 'priority': 3, 'deadline': 2},  # Impossible
            {'id': 'T2', 'duration': 8, 'priority': 2, 'deadline': 3},   # Impossible
        ]
        
        problem = self.optimizer.formulate_scheduling_problem(
            infeasible_tasks,
            self.resources,
            horizon=10
        )
        
        solution = self.optimizer.solve(problem)
        
        # Should detect infeasibility
        self.assertEqual(solution['status'], 'Infeasible')
        
        # Should provide relaxed solution
        relaxed_solution = self.optimizer.solve_with_relaxation(problem)
        self.assertIsNotNone(relaxed_solution)
        self.assertIn('violated_constraints', relaxed_solution)
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis of optimization solution"""
        problem = self.optimizer.formulate_scheduling_problem(
            self.tasks,
            self.resources,
            horizon=10
        )
        
        solution = self.optimizer.solve(problem)
        
        # Perform sensitivity analysis
        sensitivity = self.optimizer.sensitivity_analysis(
            problem,
            solution,
            parameters=['resource_capacity', 'task_duration']
        )
        
        self.assertIn('resource_capacity', sensitivity)
        self.assertIn('task_duration', sensitivity)
        
        # Check shadow prices
        self.assertIn('shadow_prices', sensitivity)
    
    def test_optimization_with_uncertainty(self):
        """Test robust optimization with uncertainty"""
        # Add uncertainty to task durations
        uncertain_tasks = [
            {
                'id': 'T1',
                'duration_min': 1.5,
                'duration_expected': 2,
                'duration_max': 2.5,
                'priority': 3,
                'deadline': 5
            },
            {
                'id': 'T2',
                'duration_min': 0.8,
                'duration_expected': 1,
                'duration_max': 1.3,
                'priority': 2,
                'deadline': 3
            }
        ]
        
        robust_solution = self.optimizer.robust_optimization(
            uncertain_tasks,
            self.resources,
            horizon=10,
            confidence_level=0.95
        )
        
        self.assertIsNotNone(robust_solution)
        self.assertIn('worst_case_objective', robust_solution)
        self.assertIn('expected_objective', robust_solution)


class TestWorkOrderManager(unittest.TestCase):
    """Test cases for Work Order Manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = WorkOrderManager()
        
        # Sample work order data
        self.work_order_data = {
            'equipment_id': 'PUMP-001',
            'priority': 'HIGH',
            'description': 'Pump vibration detected',
            'estimated_duration': 120,
            'required_skills': ['mechanical', 'electrical'],
            'anomaly_id': 'ANOM-123'
        }
    
    def test_initialization(self):
        """Test work order manager initialization"""
        self.assertIsNotNone(self.manager)
        self.assertIsInstance(self.manager.work_orders, dict)
        self.assertEqual(self.manager.next_id, 1)
    
    def test_create_work_order(self):
        """Test creating a new work order"""
        wo_id = self.manager.create_work_order(**self.work_order_data)
        
        self.assertIsNotNone(wo_id)
        self.assertIn(wo_id, self.manager.work_orders)
        
        # Check work order attributes
        work_order = self.manager.get_work_order(wo_id)
        self.assertEqual(work_order['equipment_id'], 'PUMP-001')
        self.assertEqual(work_order['priority'], 'HIGH')
        self.assertEqual(work_order['status'], 'PENDING')
        self.assertIsNotNone(work_order['created_at'])
    
    def test_update_work_order(self):
        """Test updating work order"""
        wo_id = self.manager.create_work_order(**self.work_order_data)
        
        # Update work order
        updates = {
            'status': 'IN_PROGRESS',
            'assigned_technician': 'TECH-001',
            'actual_start_time': datetime.now()
        }
        
        success = self.manager.update_work_order(wo_id, **updates)
        
        self.assertTrue(success)
        
        # Verify updates
        work_order = self.manager.get_work_order(wo_id)
        self.assertEqual(work_order['status'], 'IN_PROGRESS')
        self.assertEqual(work_order['assigned_technician'], 'TECH-001')
        self.assertIsNotNone(work_order['actual_start_time'])
    
    def test_work_order_lifecycle(self):
        """Test complete work order lifecycle"""
        # Create
        wo_id = self.manager.create_work_order(**self.work_order_data)
        self.assertEqual(self.manager.get_work_order(wo_id)['status'], 'PENDING')
        
        # Assign
        self.manager.assign_work_order(wo_id, 'TECH-001')
        self.assertEqual(self.manager.get_work_order(wo_id)['status'], 'ASSIGNED')
        
        # Start
        self.manager.start_work_order(wo_id)
        self.assertEqual(self.manager.get_work_order(wo_id)['status'], 'IN_PROGRESS')
        
        # Complete
        self.manager.complete_work_order(wo_id, notes='Work completed successfully')
        self.assertEqual(self.manager.get_work_order(wo_id)['status'], 'COMPLETED')
        self.assertIsNotNone(self.manager.get_work_order(wo_id)['completed_at'])
    
    def test_work_order_validation(self):
        """Test work order validation"""
        # Invalid priority
        invalid_data = self.work_order_data.copy()
        invalid_data['priority'] = 'INVALID'
        
        with self.assertRaises(ValueError):
            self.manager.create_work_order(**invalid_data)
        
        # Missing required fields
        incomplete_data = {'equipment_id': 'PUMP-001'}
        
        with self.assertRaises(ValueError):
            self.manager.create_work_order(**incomplete_data)
    
    def test_query_work_orders(self):
        """Test querying work orders"""
        # Create multiple work orders
        wo1 = self.manager.create_work_order(**self.work_order_data)
        
        wo2_data = self.work_order_data.copy()
        wo2_data['priority'] = 'LOW'
        wo2 = self.manager.create_work_order(**wo2_data)
        
        wo3_data = self.work_order_data.copy()
        wo3_data['equipment_id'] = 'VALVE-002'
        wo3 = self.manager.create_work_order(**wo3_data)
        
        # Query by status
        pending_orders = self.manager.get_work_orders_by_status('PENDING')
        self.assertEqual(len(pending_orders), 3)
        
        # Query by priority
        high_priority = self.manager.get_work_orders_by_priority('HIGH')
        self.assertEqual(len(high_priority), 2)
        
        # Query by equipment
        pump_orders = self.manager.get_work_orders_by_equipment('PUMP-001')
        self.assertEqual(len(pump_orders), 2)
    
    def test_work_order_metrics(self):
        """Test work order metrics calculation"""
        # Create and complete some work orders
        wo1 = self.manager.create_work_order(**self.work_order_data)
        self.manager.assign_work_order(wo1, 'TECH-001')
        self.manager.start_work_order(wo1)
        self.manager.complete_work_order(wo1)
        
        wo2 = self.manager.create_work_order(**self.work_order_data)
        self.manager.assign_work_order(wo2, 'TECH-002')
        self.manager.start_work_order(wo2)
        
        # Calculate metrics
        metrics = self.manager.calculate_metrics()
        
        self.assertIn('total_work_orders', metrics)
        self.assertIn('completed_work_orders', metrics)
        self.assertIn('in_progress_work_orders', metrics)
        self.assertIn('completion_rate', metrics)
        self.assertIn('average_completion_time', metrics)
        
        self.assertEqual(metrics['total_work_orders'], 2)
        self.assertEqual(metrics['completed_work_orders'], 1)
        self.assertEqual(metrics['in_progress_work_orders'], 1)
    
    def test_work_order_history(self):
        """Test work order history tracking"""
        wo_id = self.manager.create_work_order(**self.work_order_data)
        
        # Perform various actions
        self.manager.assign_work_order(wo_id, 'TECH-001')
        self.manager.add_note(wo_id, 'Initial inspection completed')
        self.manager.update_work_order(wo_id, estimated_duration=150)
        self.manager.start_work_order(wo_id)
        
        # Get history
        history = self.manager.get_work_order_history(wo_id)
        
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        
        # Check history entries
        for entry in history:
            self.assertIn('timestamp', entry)
            self.assertIn('action', entry)
            self.assertIn('details', entry)
    
    def test_work_order_dependencies(self):
        """Test work order dependencies"""
        # Create parent work order
        parent_wo = self.manager.create_work_order(**self.work_order_data)
        
        # Create dependent work order
        dependent_data = self.work_order_data.copy()
        dependent_data['depends_on'] = parent_wo
        dependent_wo = self.manager.create_work_order(**dependent_data)
        
        # Try to start dependent before parent is complete
        with self.assertRaises(ValueError):
            self.manager.start_work_order(dependent_wo)
        
        # Complete parent
        self.manager.assign_work_order(parent_wo, 'TECH-001')
        self.manager.start_work_order(parent_wo)
        self.manager.complete_work_order(parent_wo)
        
        # Now dependent can be started
        self.manager.assign_work_order(dependent_wo, 'TECH-002')
        self.manager.start_work_order(dependent_wo)
        
        self.assertEqual(
            self.manager.get_work_order(dependent_wo)['status'],
            'IN_PROGRESS'
        )
    
    def test_work_order_export(self):
        """Test exporting work orders"""
        # Create some work orders
        for _ in range(3):
            self.manager.create_work_order(**self.work_order_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to JSON
            json_path = os.path.join(tmpdir, 'work_orders.json')
            self.manager.export_work_orders(json_path, format='json')
            
            self.assertTrue(os.path.exists(json_path))
            
            # Load and verify
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertEqual(len(exported_data), 3)


class TestPriorityCalculator(unittest.TestCase):
    """Test cases for Priority Calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = PriorityCalculator(
            weights={
                'severity': 0.3,
                'equipment_criticality': 0.25,
                'downtime_cost': 0.25,
                'time_since_last_maintenance': 0.2
            }
        )
        
        # Sample anomaly data
        self.anomaly = {
            'anomaly_id': 'ANOM-001',
            'equipment_id': 'PUMP-001',
            'anomaly_score': 0.85,
            'severity': 'HIGH',
            'detected_at': datetime.now()
        }
        
        # Sample equipment data
        self.equipment = {
            'equipment_id': 'PUMP-001',
            'criticality': 'HIGH',
            'downtime_cost_per_hour': 1000,
            'last_maintenance': datetime.now() - timedelta(days=90),
            'mtbf': 180  # days
        }
    
    def test_initialization(self):
        """Test priority calculator initialization"""
        self.assertIsNotNone(self.calculator)
        self.assertEqual(self.calculator.weights['severity'], 0.3)
        self.assertEqual(sum(self.calculator.weights.values()), 1.0)
    
    def test_calculate_priority_score(self):
        """Test priority score calculation"""
        priority_score = self.calculator.calculate_priority(
            self.anomaly,
            self.equipment
        )
        
        self.assertIsInstance(priority_score, float)
        self.assertGreaterEqual(priority_score, 0)
        self.assertLessEqual(priority_score, 100)
    
    def test_severity_factor(self):
        """Test severity factor calculation"""
        # Test different severity levels
        severities = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.5,
            'LOW': 0.25
        }
        
        for severity, expected_factor in severities.items():
            factor = self.calculator.get_severity_factor(severity)
            self.assertEqual(factor, expected_factor)
    
    def test_criticality_factor(self):
        """Test equipment criticality factor"""
        criticalities = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.5,
            'LOW': 0.25
        }
        
        for criticality, expected_factor in criticalities.items():
            factor = self.calculator.get_criticality_factor(criticality)
            self.assertEqual(factor, expected_factor)
    
    def test_downtime_cost_factor(self):
        """Test downtime cost factor calculation"""
        # Test different cost levels
        costs = [100, 500, 1000, 5000, 10000]
        
        for cost in costs:
            factor = self.calculator.get_downtime_cost_factor(cost)
            self.assertGreaterEqual(factor, 0)
            self.assertLessEqual(factor, 1)
        
        # Higher costs should have higher factors
        factor_low = self.calculator.get_downtime_cost_factor(100)
        factor_high = self.calculator.get_downtime_cost_factor(10000)
        self.assertLess(factor_low, factor_high)
    
    def test_time_based_factor(self):
        """Test time-based priority factor"""
        # Test different time periods
        recent_maintenance = datetime.now() - timedelta(days=30)
        old_maintenance = datetime.now() - timedelta(days=365)
        
        factor_recent = self.calculator.get_time_factor(recent_maintenance)
        factor_old = self.calculator.get_time_factor(old_maintenance)
        
        # Older maintenance should have higher priority
        self.assertLess(factor_recent, factor_old)
    
    def test_priority_classification(self):
        """Test priority classification into categories"""
        # Test different scores
        scores = [95, 75, 50, 25]
        expected_priorities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        
        for score, expected in zip(scores, expected_priorities):
            priority = self.calculator.classify_priority(score)
            self.assertEqual(priority, expected)
    
    def test_batch_priority_calculation(self):
        """Test batch priority calculation for multiple anomalies"""
        anomalies = [
            {
                'anomaly_id': f'ANOM-{i}',
                'equipment_id': f'EQUIP-{i}',
                'anomaly_score': 0.5 + i * 0.1,
                'severity': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][i % 4]
            }
            for i in range(10)
        ]
        
        equipment_list = [
            {
                'equipment_id': f'EQUIP-{i}',
                'criticality': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][i % 4],
                'downtime_cost_per_hour': 100 * (i + 1),
                'last_maintenance': datetime.now() - timedelta(days=30 * (i + 1))
            }
            for i in range(10)
        ]
        
        priorities = self.calculator.calculate_batch_priorities(
            anomalies,
            equipment_list
        )
        
        self.assertEqual(len(priorities), 10)
        
        for priority in priorities:
            self.assertIn('anomaly_id', priority)
            self.assertIn('priority_score', priority)
            self.assertIn('priority_class', priority)
    
    def test_dynamic_weight_adjustment(self):
        """Test dynamic adjustment of priority weights"""
        # Adjust weights based on historical performance
        historical_data = {
            'false_positive_rate': 0.2,
            'average_resolution_time': 4.5,
            'downtime_incidents': 3
        }
        
        adjusted_weights = self.calculator.adjust_weights(historical_data)
        
        self.assertIsInstance(adjusted_weights, dict)
        self.assertAlmostEqual(sum(adjusted_weights.values()), 1.0, places=5)
        
        # Weights should be adjusted based on performance
        if historical_data['false_positive_rate'] > 0.15:
            # Should reduce severity weight if too many false positives
            self.assertLess(
                adjusted_weights['severity'],
                self.calculator.weights['severity']
            )
    
    def test_priority_with_ml_prediction(self):
        """Test priority calculation with ML model predictions"""
        # Mock ML prediction
        ml_prediction = {
            'failure_probability': 0.75,
            'time_to_failure': 48,  # hours
            'confidence': 0.85
        }
        
        priority_score = self.calculator.calculate_priority_with_ml(
            self.anomaly,
            self.equipment,
            ml_prediction
        )
        
        # ML prediction should influence priority
        base_priority = self.calculator.calculate_priority(
            self.anomaly,
            self.equipment
        )
        
        # High failure probability should increase priority
        if ml_prediction['failure_probability'] > 0.7:
            self.assertGreater(priority_score, base_priority)


class TestMaintenanceIntegration(unittest.TestCase):
    """Integration tests for maintenance module"""
    
    @patch('src.maintenance.scheduler.MaintenanceOptimizer')
    def test_end_to_end_scheduling(self, mock_optimizer):
        """Test end-to-end maintenance scheduling pipeline"""
        # Setup
        scheduler = MaintenanceScheduler()
        manager = WorkOrderManager()
        calculator = PriorityCalculator()
        
        # Create anomaly
        anomaly = {
            'anomaly_id': 'ANOM-001',
            'equipment_id': 'PUMP-001',
            'anomaly_score': 0.9,
            'severity': 'HIGH'
        }
        
        equipment = {
            'equipment_id': 'PUMP-001',
            'criticality': 'HIGH',
            'downtime_cost_per_hour': 1000
        }
        
        # Calculate priority
        priority_score = calculator.calculate_priority(anomaly, equipment)
        priority_class = calculator.classify_priority(priority_score)
        
        # Create work order
        wo_id = manager.create_work_order(
            equipment_id='PUMP-001',
            priority=priority_class,
            description='High anomaly score detected',
            estimated_duration=120,
            required_skills=['mechanical']
        )
        
        # Create schedule
        work_orders = pd.DataFrame([manager.get_work_order(wo_id)])
        technicians = pd.DataFrame([{
            'technician_id': 'TECH-001',
            'skills': ['mechanical'],
            'availability': True
        }])
        
        equipment_df = pd.DataFrame([equipment])
        
        schedule = scheduler.create_schedule(
            work_orders,
            technicians,
            equipment_df
        )
        
        # Verify pipeline
        self.assertIsNotNone(schedule)
        self.assertFalse(schedule.empty)
        self.assertEqual(len(schedule), 1)
    
    def test_reactive_vs_preventive_balance(self):
        """Test balancing reactive and preventive maintenance"""
        scheduler = MaintenanceScheduler()
        
        # Create mix of reactive and preventive work orders
        reactive_orders = pd.DataFrame([
            {
                'work_order_id': f'RO-{i}',
                'priority': 'HIGH',
                'maintenance_type': 'REACTIVE',
                'estimated_duration': 120,
                'due_date': datetime.now() + timedelta(days=1)
            }
            for i in range(3)
        ])
        
        preventive_orders = pd.DataFrame([
            {
                'work_order_id': f'PO-{i}',
                'priority': 'LOW',
                'maintenance_type': 'PREVENTIVE',
                'estimated_duration': 60,
                'due_date': datetime.now() + timedelta(days=7)
            }
            for i in range(5)
        ])
        
        all_orders = pd.concat([reactive_orders, preventive_orders])
        
        # Calculate optimal balance
        balance = scheduler.calculate_maintenance_balance(all_orders)
        
        self.assertIn('reactive_percentage', balance)
        self.assertIn('preventive_percentage', balance)
        self.assertIn('recommended_ratio', balance)
        
        # Check if balance is reasonable (typically 20-40% reactive is good)
        self.assertLess(balance['reactive_percentage'], 50)


if __name__ == '__main__':
    unittest.main()