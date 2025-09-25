"""
Cost-Benefit Analysis System for Predictive Maintenance

This module provides comprehensive economic analysis for maintenance decisions,
including ROI calculations, Total Cost of Ownership (TCO) modeling, and
optimization of maintenance strategies based on financial impact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import math

# Local imports
from .failure_classification import FailureMode, Severity
from .equipment_health import SensorHealth, SubsystemHealth, HealthStatus
from .predictive_triggers import TriggerEvent, MaintenanceAction, MaintenanceType, Priority
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of maintenance costs"""
    DIRECT_LABOR = "direct_labor"                 # Technician time
    INDIRECT_LABOR = "indirect_labor"             # Management, support
    MATERIALS = "materials"                       # Parts, consumables
    TOOLS_EQUIPMENT = "tools_equipment"           # Special tools, equipment
    TRANSPORTATION = "transportation"             # Travel, shipping
    DOWNTIME = "downtime"                        # Production loss
    QUALITY_IMPACT = "quality_impact"            # Quality degradation
    SAFETY_IMPACT = "safety_impact"              # Safety incidents
    ENVIRONMENTAL = "environmental"               # Environmental compliance
    OPPORTUNITY_COST = "opportunity_cost"         # Missed opportunities


class BenefitCategory(Enum):
    """Categories of maintenance benefits"""
    FAILURE_PREVENTION = "failure_prevention"     # Avoided failures
    LIFE_EXTENSION = "life_extension"             # Extended equipment life
    PERFORMANCE_IMPROVEMENT = "performance_improvement"  # Better performance
    ENERGY_SAVINGS = "energy_savings"             # Reduced energy consumption
    QUALITY_IMPROVEMENT = "quality_improvement"   # Better output quality
    SAFETY_IMPROVEMENT = "safety_improvement"     # Reduced safety risks
    COMPLIANCE = "compliance"                     # Regulatory compliance
    REPUTATION = "reputation"                     # Brand/reputation value
    INSURANCE = "insurance"                       # Reduced insurance costs
    RESALE_VALUE = "resale_value"                # Equipment resale value


@dataclass
class CostElement:
    """Individual cost component"""
    category: CostCategory
    description: str
    amount: float
    confidence: float = 1.0                       # Confidence in estimate (0-1)
    is_recurring: bool = False                    # Recurring vs one-time cost
    recurrence_period_days: int = 365             # Days between recurrences

    # Cost breakdown
    labor_hours: float = 0.0
    hourly_rate: float = 0.0
    material_cost: float = 0.0
    overhead_rate: float = 0.15                   # Overhead as fraction of direct costs


@dataclass
class BenefitElement:
    """Individual benefit component"""
    category: BenefitCategory
    description: str
    amount: float
    confidence: float = 1.0                       # Confidence in estimate (0-1)
    is_recurring: bool = True                     # Benefits usually recurring
    recurrence_period_days: int = 365             # Days between benefits

    # Benefit realization timeline
    realization_delay_days: int = 0               # Days until benefit starts
    ramp_up_period_days: int = 0                  # Days to reach full benefit


@dataclass
class MaintenanceStrategy:
    """Maintenance strategy definition"""
    strategy_id: str
    strategy_name: str
    description: str

    # Strategy parameters
    maintenance_type: MaintenanceType
    intervention_threshold: float                 # Health score threshold
    maintenance_interval_days: int = 90           # Scheduled interval

    # Cost and benefit estimates
    costs: List[CostElement] = field(default_factory=list)
    benefits: List[BenefitElement] = field(default_factory=list)

    # Performance characteristics
    failure_reduction_factor: float = 0.8         # Factor by which failures are reduced
    life_extension_factor: float = 1.2            # Factor by which life is extended
    performance_improvement: float = 0.05         # Performance improvement fraction

    # Risk factors
    implementation_risk: float = 0.1              # Risk of implementation failure
    technology_risk: float = 0.05                 # Risk of technology issues


@dataclass
class FinancialAnalysis:
    """Financial analysis results"""
    analysis_id: str
    strategy: MaintenanceStrategy
    analysis_date: datetime

    # Cost analysis
    total_costs_npv: float                        # Net present value of costs
    annual_costs: float                           # Annualized costs
    cost_breakdown: Dict[CostCategory, float]     # Costs by category

    # Benefit analysis
    total_benefits_npv: float                     # Net present value of benefits
    annual_benefits: float                        # Annualized benefits
    benefit_breakdown: Dict[BenefitCategory, float]  # Benefits by category

    # Financial metrics
    net_present_value: float                      # NPV of strategy
    return_on_investment: float                   # ROI percentage
    payback_period_months: float                  # Payback period
    internal_rate_of_return: float               # IRR percentage

    # Sensitivity analysis
    best_case_npv: float                         # Best case scenario
    worst_case_npv: float                        # Worst case scenario
    break_even_threshold: float                  # Break-even point

    # Risk assessment
    value_at_risk_95: float                      # 95% VaR
    expected_shortfall: float                    # Expected shortfall

    # Analysis parameters
    discount_rate: float = 0.08                 # Discount rate used
    analysis_period_years: int = 5               # Analysis time horizon
    confidence_level: float = 0.8               # Overall analysis confidence


@dataclass
class EquipmentEconomics:
    """Economic model for equipment/subsystem"""
    equipment_id: str
    equipment_type: str
    subsystem: str

    # Asset information
    acquisition_cost: float
    current_book_value: float
    expected_remaining_life_years: float
    salvage_value: float

    # Operational economics
    annual_operating_cost: float
    annual_revenue_contribution: float
    downtime_cost_per_hour: float

    # Failure economics
    failure_probability_annual: float = 0.1
    failure_cost_distribution: Dict[str, float] = field(default_factory=dict)  # {failure_type: cost}

    # Maintenance history
    historical_maintenance_costs: List[Tuple[datetime, float]] = field(default_factory=list)
    reliability_trend: float = 0.0               # Annual reliability change

    # Performance economics
    performance_degradation_rate: float = 0.02   # Annual performance decline
    energy_consumption_kwh_per_hour: float = 0.0
    energy_cost_per_kwh: float = 0.12


class CostBenefitAnalyzer:
    """
    Comprehensive cost-benefit analysis system for predictive maintenance.

    Features:
    - ROI calculation and optimization
    - Total Cost of Ownership (TCO) modeling
    - Multi-strategy comparison
    - Risk-adjusted financial analysis
    - Sensitivity analysis and Monte Carlo simulation
    - Life-cycle cost analysis
    """

    def __init__(self):
        self.config = get_config()

        # Economic parameters
        self.discount_rate = self.config.get('financial', {}).get('discount_rate', 0.08)
        self.inflation_rate = self.config.get('financial', {}).get('inflation_rate', 0.03)
        self.corporate_tax_rate = self.config.get('financial', {}).get('tax_rate', 0.25)

        # Equipment economics database
        self.equipment_economics: Dict[str, EquipmentEconomics] = {}

        # Maintenance strategies
        self.strategies: Dict[str, MaintenanceStrategy] = {}

        # Cost databases
        self.labor_rates = self._load_labor_rates()
        self.material_costs = self._load_material_costs()
        self.overhead_rates = self._load_overhead_rates()

        # Initialize default strategies and economics
        self._initialize_default_strategies()
        self._initialize_equipment_economics()

        logger.info("Initialized CostBenefitAnalyzer")

    def _load_labor_rates(self) -> Dict[str, float]:
        """Load labor rates by skill category"""
        return {
            'systems_diagnosis': 85.0,           # $/hour
            'electrical_systems': 90.0,
            'mechanical_systems': 80.0,
            'thermal_systems': 82.0,
            'communication_systems': 88.0,
            'robotics': 95.0,
            'instrumentation': 87.0,
            'calibration': 92.0,
            'emergency_response': 120.0,
            'preventive_maintenance': 75.0,
            'management_oversight': 110.0
        }

    def _load_material_costs(self) -> Dict[str, float]:
        """Load material costs by category"""
        return {
            'thermal_sensors': 250.0,
            'coolant': 50.0,
            'lubricants': 35.0,
            'joint_lubricants': 45.0,
            'motor_brushes': 80.0,
            'calibration_standards': 500.0,
            'optical_cleaners': 25.0,
            'lint_free_cloths': 15.0,
            'electrical_components': 200.0,
            'mechanical_parts': 150.0,
            'communication_parts': 300.0,
            'software_licenses': 1000.0
        }

    def _load_overhead_rates(self) -> Dict[str, float]:
        """Load overhead rates by activity type"""
        return {
            'routine_maintenance': 0.15,
            'emergency_maintenance': 0.25,
            'equipment_replacement': 0.20,
            'system_upgrades': 0.30,
            'training': 0.10
        }

    def _initialize_default_strategies(self):
        """Initialize default maintenance strategies"""

        # Strategy 1: Reactive Maintenance (Baseline)
        reactive_strategy = MaintenanceStrategy(
            strategy_id="reactive",
            strategy_name="Reactive Maintenance",
            description="Fix equipment only after failure occurs",
            maintenance_type=MaintenanceType.REPAIR,
            intervention_threshold=0.0,  # No intervention until failure
            maintenance_interval_days=0,  # No scheduled maintenance
            failure_reduction_factor=1.0,  # No failure reduction
            life_extension_factor=1.0,     # No life extension
            performance_improvement=0.0,   # No performance improvement
            implementation_risk=0.0,       # Low implementation risk
            technology_risk=0.0
        )

        # Strategy 2: Preventive Maintenance
        preventive_strategy = MaintenanceStrategy(
            strategy_id="preventive",
            strategy_name="Preventive Maintenance",
            description="Scheduled maintenance at regular intervals",
            maintenance_type=MaintenanceType.INSPECTION,
            intervention_threshold=80.0,
            maintenance_interval_days=90,
            failure_reduction_factor=0.7,  # 30% failure reduction
            life_extension_factor=1.3,     # 30% life extension
            performance_improvement=0.10,  # 10% performance improvement
            implementation_risk=0.05,
            technology_risk=0.02
        )

        # Strategy 3: Predictive Maintenance
        predictive_strategy = MaintenanceStrategy(
            strategy_id="predictive",
            strategy_name="Predictive Maintenance",
            description="Condition-based maintenance using IoT and ML",
            maintenance_type=MaintenanceType.INSPECTION,
            intervention_threshold=70.0,
            maintenance_interval_days=120,  # Longer intervals due to prediction
            failure_reduction_factor=0.5,   # 50% failure reduction
            life_extension_factor=1.5,      # 50% life extension
            performance_improvement=0.15,   # 15% performance improvement
            implementation_risk=0.15,       # Higher implementation complexity
            technology_risk=0.10
        )

        # Strategy 4: Proactive Maintenance
        proactive_strategy = MaintenanceStrategy(
            strategy_id="proactive",
            strategy_name="Proactive Maintenance",
            description="Advanced predictive maintenance with optimization",
            maintenance_type=MaintenanceType.OVERHAUL,
            intervention_threshold=75.0,
            maintenance_interval_days=180,  # Optimized intervals
            failure_reduction_factor=0.3,   # 70% failure reduction
            life_extension_factor=1.8,      # 80% life extension
            performance_improvement=0.25,   # 25% performance improvement
            implementation_risk=0.20,       # Highest complexity
            technology_risk=0.15
        )

        strategies = [reactive_strategy, preventive_strategy, predictive_strategy, proactive_strategy]

        for strategy in strategies:
            self._populate_strategy_costs_benefits(strategy)
            self.strategies[strategy.strategy_id] = strategy

    def _populate_strategy_costs_benefits(self, strategy: MaintenanceStrategy):
        """Populate costs and benefits for a maintenance strategy"""

        # Cost elements based on strategy type
        if strategy.strategy_id == "reactive":
            # Reactive maintenance costs
            strategy.costs = [
                CostElement(
                    category=CostCategory.DIRECT_LABOR,
                    description="Emergency repair labor",
                    amount=0,  # Will be calculated based on failures
                    labor_hours=8.0,
                    hourly_rate=120.0,  # Emergency rate
                    confidence=0.6
                ),
                CostElement(
                    category=CostCategory.DOWNTIME,
                    description="Unplanned downtime costs",
                    amount=0,  # Will be calculated
                    confidence=0.7
                )
            ]

        elif strategy.strategy_id == "preventive":
            # Preventive maintenance costs
            strategy.costs = [
                CostElement(
                    category=CostCategory.DIRECT_LABOR,
                    description="Scheduled maintenance labor",
                    amount=1500.0,
                    labor_hours=20.0,
                    hourly_rate=75.0,
                    is_recurring=True,
                    recurrence_period_days=90,
                    confidence=0.9
                ),
                CostElement(
                    category=CostCategory.MATERIALS,
                    description="Maintenance materials and consumables",
                    amount=500.0,
                    is_recurring=True,
                    recurrence_period_days=90,
                    confidence=0.8
                )
            ]

            # Preventive maintenance benefits
            strategy.benefits = [
                BenefitElement(
                    category=BenefitCategory.FAILURE_PREVENTION,
                    description="Avoided failure costs",
                    amount=8000.0,
                    confidence=0.8,
                    realization_delay_days=30
                ),
                BenefitElement(
                    category=BenefitCategory.LIFE_EXTENSION,
                    description="Extended equipment life",
                    amount=15000.0,
                    confidence=0.7,
                    ramp_up_period_days=365
                )
            ]

        elif strategy.strategy_id == "predictive":
            # Predictive maintenance costs
            strategy.costs = [
                CostElement(
                    category=CostCategory.DIRECT_LABOR,
                    description="Condition-based maintenance labor",
                    amount=2000.0,
                    labor_hours=24.0,
                    hourly_rate=85.0,
                    is_recurring=True,
                    recurrence_period_days=120,
                    confidence=0.8
                ),
                CostElement(
                    category=CostCategory.TOOLS_EQUIPMENT,
                    description="IoT sensors and monitoring equipment",
                    amount=25000.0,
                    confidence=0.9
                ),
                CostElement(
                    category=CostCategory.INDIRECT_LABOR,
                    description="Data analysis and system management",
                    amount=5000.0,
                    is_recurring=True,
                    recurrence_period_days=365,
                    confidence=0.8
                )
            ]

            # Predictive maintenance benefits
            strategy.benefits = [
                BenefitElement(
                    category=BenefitCategory.FAILURE_PREVENTION,
                    description="Advanced failure prevention",
                    amount=20000.0,
                    confidence=0.85,
                    realization_delay_days=60
                ),
                BenefitElement(
                    category=BenefitCategory.PERFORMANCE_IMPROVEMENT,
                    description="Optimized performance",
                    amount=12000.0,
                    confidence=0.8,
                    ramp_up_period_days=180
                ),
                BenefitElement(
                    category=BenefitCategory.ENERGY_SAVINGS,
                    description="Reduced energy consumption",
                    amount=3000.0,
                    confidence=0.7,
                    realization_delay_days=90
                )
            ]

        elif strategy.strategy_id == "proactive":
            # Proactive maintenance costs
            strategy.costs = [
                CostElement(
                    category=CostCategory.DIRECT_LABOR,
                    description="Advanced predictive maintenance",
                    amount=3000.0,
                    labor_hours=32.0,
                    hourly_rate=95.0,
                    is_recurring=True,
                    recurrence_period_days=180,
                    confidence=0.7
                ),
                CostElement(
                    category=CostCategory.TOOLS_EQUIPMENT,
                    description="Advanced analytics and AI systems",
                    amount=50000.0,
                    confidence=0.8
                ),
                CostElement(
                    category=CostCategory.INDIRECT_LABOR,
                    description="Advanced analytics and optimization",
                    amount=12000.0,
                    is_recurring=True,
                    recurrence_period_days=365,
                    confidence=0.7
                )
            ]

            # Proactive maintenance benefits
            strategy.benefits = [
                BenefitElement(
                    category=BenefitCategory.FAILURE_PREVENTION,
                    description="Maximum failure prevention",
                    amount=35000.0,
                    confidence=0.8,
                    realization_delay_days=90
                ),
                BenefitElement(
                    category=BenefitCategory.LIFE_EXTENSION,
                    description="Maximum life extension",
                    amount=40000.0,
                    confidence=0.75,
                    ramp_up_period_days=365
                ),
                BenefitElement(
                    category=BenefitCategory.PERFORMANCE_IMPROVEMENT,
                    description="Maximum performance optimization",
                    amount=25000.0,
                    confidence=0.8,
                    ramp_up_period_days=270
                )
            ]

    def _initialize_equipment_economics(self):
        """Initialize economic models for equipment subsystems"""

        # SMAP Equipment Economics
        smap_economics = {
            'soil_monitoring': EquipmentEconomics(
                equipment_id="smap_soil_system",
                equipment_type="soil_monitoring_system",
                subsystem="soil_monitoring",
                acquisition_cost=150000.0,
                current_book_value=120000.0,
                expected_remaining_life_years=8.0,
                salvage_value=20000.0,
                annual_operating_cost=15000.0,
                annual_revenue_contribution=50000.0,
                downtime_cost_per_hour=2000.0,
                failure_probability_annual=0.08,
                performance_degradation_rate=0.03,
                energy_consumption_kwh_per_hour=5.0
            ),

            'radar_system': EquipmentEconomics(
                equipment_id="smap_radar_system",
                equipment_type="radar_system",
                subsystem="radar_system",
                acquisition_cost=500000.0,
                current_book_value=400000.0,
                expected_remaining_life_years=10.0,
                salvage_value=50000.0,
                annual_operating_cost=25000.0,
                annual_revenue_contribution=150000.0,
                downtime_cost_per_hour=5000.0,
                failure_probability_annual=0.05,
                performance_degradation_rate=0.02,
                energy_consumption_kwh_per_hour=20.0
            ),

            'power_system': EquipmentEconomics(
                equipment_id="smap_power_system",
                equipment_type="power_system",
                subsystem="power_system",
                acquisition_cost=200000.0,
                current_book_value=160000.0,
                expected_remaining_life_years=12.0,
                salvage_value=30000.0,
                annual_operating_cost=8000.0,
                annual_revenue_contribution=0.0,  # Enabling system
                downtime_cost_per_hour=10000.0,  # High impact if fails
                failure_probability_annual=0.03,
                performance_degradation_rate=0.01,
                energy_consumption_kwh_per_hour=2.0
            )
        }

        # MSL Equipment Economics
        msl_economics = {
            'mobility': EquipmentEconomics(
                equipment_id="msl_mobility_system",
                equipment_type="mobility_system",
                subsystem="mobility",
                acquisition_cost=800000.0,
                current_book_value=600000.0,
                expected_remaining_life_years=6.0,
                salvage_value=100000.0,
                annual_operating_cost=40000.0,
                annual_revenue_contribution=200000.0,
                downtime_cost_per_hour=8000.0,
                failure_probability_annual=0.12,
                performance_degradation_rate=0.04,
                energy_consumption_kwh_per_hour=15.0
            ),

            'robotic_arm': EquipmentEconomics(
                equipment_id="msl_robotic_arm",
                equipment_type="robotic_arm",
                subsystem="robotic_arm",
                acquisition_cost=1200000.0,
                current_book_value=900000.0,
                expected_remaining_life_years=8.0,
                salvage_value=150000.0,
                annual_operating_cost=50000.0,
                annual_revenue_contribution=300000.0,
                downtime_cost_per_hour=12000.0,
                failure_probability_annual=0.08,
                performance_degradation_rate=0.03,
                energy_consumption_kwh_per_hour=25.0
            ),

            'science_instruments': EquipmentEconomics(
                equipment_id="msl_science_instruments",
                equipment_type="science_instruments",
                subsystem="science_instruments",
                acquisition_cost=2000000.0,
                current_book_value=1500000.0,
                expected_remaining_life_years=10.0,
                salvage_value=200000.0,
                annual_operating_cost=80000.0,
                annual_revenue_contribution=500000.0,
                downtime_cost_per_hour=15000.0,
                failure_probability_annual=0.06,
                performance_degradation_rate=0.02,
                energy_consumption_kwh_per_hour=30.0
            )
        }

        # Combine all equipment economics
        self.equipment_economics.update(smap_economics)
        self.equipment_economics.update(msl_economics)

    def analyze_maintenance_strategy(self, strategy_id: str, equipment_id: str,
                                   analysis_period_years: int = 5) -> FinancialAnalysis:
        """Perform comprehensive financial analysis of maintenance strategy"""

        if strategy_id not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        if equipment_id not in self.equipment_economics:
            raise ValueError(f"Unknown equipment: {equipment_id}")

        strategy = self.strategies[strategy_id]
        equipment = self.equipment_economics[equipment_id]

        logger.info(f"Analyzing strategy '{strategy.strategy_name}' for equipment '{equipment_id}'")

        # Calculate costs over analysis period
        total_costs_npv, annual_costs, cost_breakdown = self._calculate_costs(
            strategy, equipment, analysis_period_years
        )

        # Calculate benefits over analysis period
        total_benefits_npv, annual_benefits, benefit_breakdown = self._calculate_benefits(
            strategy, equipment, analysis_period_years
        )

        # Calculate financial metrics
        net_present_value = total_benefits_npv - total_costs_npv
        roi = (net_present_value / total_costs_npv * 100) if total_costs_npv > 0 else 0
        payback_period = self._calculate_payback_period(strategy, equipment)
        irr = self._calculate_irr(strategy, equipment, analysis_period_years)

        # Sensitivity analysis
        best_case_npv, worst_case_npv = self._perform_sensitivity_analysis(
            strategy, equipment, analysis_period_years
        )

        # Risk analysis
        var_95, expected_shortfall = self._calculate_risk_metrics(
            strategy, equipment, analysis_period_years
        )

        # Break-even analysis
        break_even_threshold = self._calculate_break_even(strategy, equipment)

        analysis_id = f"analysis_{strategy_id}_{equipment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return FinancialAnalysis(
            analysis_id=analysis_id,
            strategy=strategy,
            analysis_date=datetime.now(),
            total_costs_npv=total_costs_npv,
            annual_costs=annual_costs,
            cost_breakdown=cost_breakdown,
            total_benefits_npv=total_benefits_npv,
            annual_benefits=annual_benefits,
            benefit_breakdown=benefit_breakdown,
            net_present_value=net_present_value,
            return_on_investment=roi,
            payback_period_months=payback_period,
            internal_rate_of_return=irr,
            best_case_npv=best_case_npv,
            worst_case_npv=worst_case_npv,
            break_even_threshold=break_even_threshold,
            value_at_risk_95=var_95,
            expected_shortfall=expected_shortfall,
            discount_rate=self.discount_rate,
            analysis_period_years=analysis_period_years,
            confidence_level=self._calculate_analysis_confidence(strategy)
        )

    def _calculate_costs(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics,
                        analysis_period_years: int) -> Tuple[float, float, Dict[CostCategory, float]]:
        """Calculate total costs for maintenance strategy"""

        total_costs_npv = 0.0
        cost_breakdown = {category: 0.0 for category in CostCategory}

        # Calculate costs year by year
        annual_cash_flows = []

        for year in range(analysis_period_years):
            annual_cost = 0.0

            # Direct maintenance costs
            for cost_element in strategy.costs:
                cost_in_year = self._calculate_cost_element_value(cost_element, year)
                annual_cost += cost_in_year
                cost_breakdown[cost_element.category] += cost_in_year

            # Failure-related costs (reduced by strategy effectiveness)
            baseline_failure_cost = equipment.failure_probability_annual * self._estimate_failure_cost(equipment)
            actual_failure_cost = baseline_failure_cost * strategy.failure_reduction_factor
            annual_cost += actual_failure_cost
            cost_breakdown[CostCategory.DOWNTIME] += actual_failure_cost

            # Discount to present value
            discount_factor = (1 + self.discount_rate) ** year
            annual_costs_npv = annual_cost / discount_factor
            total_costs_npv += annual_costs_npv
            annual_cash_flows.append(annual_cost)

        # Calculate average annual cost
        annual_costs = np.mean(annual_cash_flows)

        return total_costs_npv, annual_costs, cost_breakdown

    def _calculate_benefits(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics,
                           analysis_period_years: int) -> Tuple[float, float, Dict[BenefitCategory, float]]:
        """Calculate total benefits for maintenance strategy"""

        total_benefits_npv = 0.0
        benefit_breakdown = {category: 0.0 for category in BenefitCategory}

        # Calculate benefits year by year
        annual_cash_flows = []

        for year in range(analysis_period_years):
            annual_benefit = 0.0

            # Direct maintenance benefits
            for benefit_element in strategy.benefits:
                benefit_in_year = self._calculate_benefit_element_value(benefit_element, year)
                annual_benefit += benefit_in_year
                benefit_breakdown[benefit_element.category] += benefit_in_year

            # Performance improvement benefits
            performance_benefit = (equipment.annual_revenue_contribution *
                                 strategy.performance_improvement)
            annual_benefit += performance_benefit
            benefit_breakdown[BenefitCategory.PERFORMANCE_IMPROVEMENT] += performance_benefit

            # Energy savings
            energy_savings = (equipment.energy_consumption_kwh_per_hour * 8760 *
                            self.config.get('energy_cost_per_kwh', 0.12) *
                            strategy.performance_improvement * 0.1)  # 10% of performance improvement
            annual_benefit += energy_savings
            benefit_breakdown[BenefitCategory.ENERGY_SAVINGS] += energy_savings

            # Life extension benefits (realized over extended life)
            if year >= equipment.expected_remaining_life_years:
                life_extension_benefit = equipment.annual_revenue_contribution * 0.8  # Reduced value
                annual_benefit += life_extension_benefit
                benefit_breakdown[BenefitCategory.LIFE_EXTENSION] += life_extension_benefit

            # Discount to present value
            discount_factor = (1 + self.discount_rate) ** year
            annual_benefits_npv = annual_benefit / discount_factor
            total_benefits_npv += annual_benefits_npv
            annual_cash_flows.append(annual_benefit)

        # Calculate average annual benefit
        annual_benefits = np.mean(annual_cash_flows)

        return total_benefits_npv, annual_benefits, benefit_breakdown

    def _calculate_cost_element_value(self, cost_element: CostElement, year: int) -> float:
        """Calculate cost element value for specific year"""

        if not cost_element.is_recurring and year > 0:
            return 0.0  # One-time cost only in year 0

        if cost_element.is_recurring:
            # Check if cost occurs this year
            days_in_year = year * 365
            if days_in_year % cost_element.recurrence_period_days != 0:
                return 0.0

        # Base cost
        base_cost = cost_element.amount

        # If amount is zero, calculate from components
        if base_cost == 0.0:
            labor_cost = cost_element.labor_hours * cost_element.hourly_rate
            material_cost = cost_element.material_cost
            overhead_cost = (labor_cost + material_cost) * cost_element.overhead_rate
            base_cost = labor_cost + material_cost + overhead_cost

        # Apply inflation
        inflated_cost = base_cost * ((1 + self.inflation_rate) ** year)

        # Apply confidence factor (reduce cost if uncertain)
        adjusted_cost = inflated_cost * cost_element.confidence

        return adjusted_cost

    def _calculate_benefit_element_value(self, benefit_element: BenefitElement, year: int) -> float:
        """Calculate benefit element value for specific year"""

        # Check if benefit has started
        if year * 365 < benefit_element.realization_delay_days:
            return 0.0

        # Calculate ramp-up factor
        days_since_start = year * 365 - benefit_element.realization_delay_days
        if benefit_element.ramp_up_period_days > 0:
            ramp_up_factor = min(1.0, days_since_start / benefit_element.ramp_up_period_days)
        else:
            ramp_up_factor = 1.0

        # Base benefit with ramp-up
        base_benefit = benefit_element.amount * ramp_up_factor

        # Apply inflation (benefits also inflate)
        inflated_benefit = base_benefit * ((1 + self.inflation_rate) ** year)

        # Apply confidence factor (reduce benefit if uncertain)
        adjusted_benefit = inflated_benefit * benefit_element.confidence

        return adjusted_benefit

    def _estimate_failure_cost(self, equipment: EquipmentEconomics) -> float:
        """Estimate cost of equipment failure"""

        # Direct repair cost (estimated as 10% of replacement cost)
        repair_cost = equipment.acquisition_cost * 0.10

        # Downtime cost (estimated 48 hours average downtime)
        downtime_cost = equipment.downtime_cost_per_hour * 48

        # Opportunity cost (lost revenue)
        opportunity_cost = equipment.annual_revenue_contribution * (48 / 8760)  # 48 hours of year

        total_failure_cost = repair_cost + downtime_cost + opportunity_cost

        return total_failure_cost

    def _calculate_payback_period(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics) -> float:
        """Calculate payback period in months"""

        # Simplified payback calculation
        initial_investment = sum(cost.amount for cost in strategy.costs if not cost.is_recurring)
        annual_net_benefit = sum(benefit.amount for benefit in strategy.benefits) / len(strategy.benefits) if strategy.benefits else 0
        annual_cost_savings = equipment.failure_probability_annual * self._estimate_failure_cost(equipment) * (1 - strategy.failure_reduction_factor)

        total_annual_benefit = annual_net_benefit + annual_cost_savings

        if total_annual_benefit <= 0:
            return float('inf')  # Never pays back

        payback_years = initial_investment / total_annual_benefit
        return payback_years * 12  # Convert to months

    def _calculate_irr(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics,
                      analysis_period_years: int) -> float:
        """Calculate Internal Rate of Return"""

        # Simplified IRR calculation using Newton-Raphson method
        # This would normally use a financial library like numpy-financial

        # Generate cash flows
        cash_flows = []

        # Initial investment (negative)
        initial_cost = sum(cost.amount for cost in strategy.costs if not cost.is_recurring)
        cash_flows.append(-initial_cost)

        # Annual net benefits
        for year in range(1, analysis_period_years + 1):
            annual_benefit = sum(benefit.amount for benefit in strategy.benefits) / len(strategy.benefits) if strategy.benefits else 0
            annual_cost = sum(cost.amount for cost in strategy.costs if cost.is_recurring) / (365 / cost.recurrence_period_days if cost.recurrence_period_days > 0 else 1)
            net_annual_flow = annual_benefit - annual_cost
            cash_flows.append(net_annual_flow)

        # Simplified IRR approximation
        if len(cash_flows) < 2:
            return 0.0

        # Use discount rate as starting point
        irr_estimate = self.discount_rate

        # Basic iteration to find IRR (simplified)
        for _ in range(10):
            npv = sum(cf / ((1 + irr_estimate) ** i) for i, cf in enumerate(cash_flows))
            if abs(npv) < 1.0:
                break
            irr_estimate += 0.01 if npv > 0 else -0.01

        return irr_estimate * 100  # Return as percentage

    def _perform_sensitivity_analysis(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics,
                                     analysis_period_years: int) -> Tuple[float, float]:
        """Perform sensitivity analysis for best/worst case scenarios"""

        # Best case: 20% better costs, 20% better benefits, 10% better failure reduction
        best_case_strategy = self._create_strategy_variant(strategy, cost_factor=0.8, benefit_factor=1.2, effectiveness_factor=1.1)
        best_case_analysis = self.analyze_maintenance_strategy(best_case_strategy.strategy_id, equipment.equipment_id, analysis_period_years)

        # Worst case: 20% worse costs, 20% worse benefits, 10% worse failure reduction
        worst_case_strategy = self._create_strategy_variant(strategy, cost_factor=1.2, benefit_factor=0.8, effectiveness_factor=0.9)
        worst_case_analysis = self.analyze_maintenance_strategy(worst_case_strategy.strategy_id, equipment.equipment_id, analysis_period_years)

        return best_case_analysis.net_present_value, worst_case_analysis.net_present_value

    def _create_strategy_variant(self, base_strategy: MaintenanceStrategy, cost_factor: float,
                                benefit_factor: float, effectiveness_factor: float) -> MaintenanceStrategy:
        """Create a variant of strategy for sensitivity analysis"""

        variant = MaintenanceStrategy(
            strategy_id=f"{base_strategy.strategy_id}_variant",
            strategy_name=f"{base_strategy.strategy_name} Variant",
            description=f"Sensitivity variant of {base_strategy.strategy_name}",
            maintenance_type=base_strategy.maintenance_type,
            intervention_threshold=base_strategy.intervention_threshold,
            maintenance_interval_days=base_strategy.maintenance_interval_days,
            failure_reduction_factor=base_strategy.failure_reduction_factor * effectiveness_factor,
            life_extension_factor=base_strategy.life_extension_factor * effectiveness_factor,
            performance_improvement=base_strategy.performance_improvement * effectiveness_factor,
            implementation_risk=base_strategy.implementation_risk,
            technology_risk=base_strategy.technology_risk
        )

        # Adjust costs
        variant.costs = []
        for cost in base_strategy.costs:
            adjusted_cost = CostElement(
                category=cost.category,
                description=cost.description,
                amount=cost.amount * cost_factor,
                confidence=cost.confidence,
                is_recurring=cost.is_recurring,
                recurrence_period_days=cost.recurrence_period_days,
                labor_hours=cost.labor_hours,
                hourly_rate=cost.hourly_rate * cost_factor,
                material_cost=cost.material_cost * cost_factor,
                overhead_rate=cost.overhead_rate
            )
            variant.costs.append(adjusted_cost)

        # Adjust benefits
        variant.benefits = []
        for benefit in base_strategy.benefits:
            adjusted_benefit = BenefitElement(
                category=benefit.category,
                description=benefit.description,
                amount=benefit.amount * benefit_factor,
                confidence=benefit.confidence,
                is_recurring=benefit.is_recurring,
                recurrence_period_days=benefit.recurrence_period_days,
                realization_delay_days=benefit.realization_delay_days,
                ramp_up_period_days=benefit.ramp_up_period_days
            )
            variant.benefits.append(adjusted_benefit)

        return variant

    def _calculate_risk_metrics(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics,
                               analysis_period_years: int) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall"""

        # Monte Carlo simulation would be implemented here
        # For now, provide simplified risk metrics

        base_analysis = self.analyze_maintenance_strategy(strategy.strategy_id, equipment.equipment_id, analysis_period_years)
        base_npv = base_analysis.net_present_value

        # Estimate volatility based on strategy risk factors
        volatility = (strategy.implementation_risk + strategy.technology_risk) * base_npv

        # 95% VaR (simplified normal distribution assumption)
        var_95 = base_npv - 1.645 * volatility  # 95th percentile

        # Expected Shortfall (average loss beyond VaR)
        expected_shortfall = var_95 - 0.5 * volatility

        return var_95, expected_shortfall

    def _calculate_break_even(self, strategy: MaintenanceStrategy, equipment: EquipmentEconomics) -> float:
        """Calculate break-even threshold for strategy"""

        # Break-even occurs when NPV = 0
        # This would be solved iteratively by adjusting key parameters

        total_costs = sum(cost.amount for cost in strategy.costs)
        total_benefits = sum(benefit.amount for benefit in strategy.benefits)

        if total_benefits == 0:
            return float('inf')

        # Simplified break-even as cost/benefit ratio
        break_even_ratio = total_costs / total_benefits

        return break_even_ratio

    def _calculate_analysis_confidence(self, strategy: MaintenanceStrategy) -> float:
        """Calculate overall confidence in analysis"""

        cost_confidences = [cost.confidence for cost in strategy.costs]
        benefit_confidences = [benefit.confidence for benefit in strategy.benefits]

        all_confidences = cost_confidences + benefit_confidences

        if not all_confidences:
            return 0.5  # Default medium confidence

        # Weighted average confidence
        overall_confidence = np.mean(all_confidences)

        # Reduce confidence based on implementation risk
        risk_penalty = (strategy.implementation_risk + strategy.technology_risk) / 2
        adjusted_confidence = overall_confidence * (1 - risk_penalty)

        return max(0.1, min(1.0, adjusted_confidence))

    def compare_strategies(self, equipment_id: str, strategy_ids: List[str] = None,
                          analysis_period_years: int = 5) -> pd.DataFrame:
        """Compare multiple maintenance strategies"""

        if strategy_ids is None:
            strategy_ids = list(self.strategies.keys())

        comparison_results = []

        for strategy_id in strategy_ids:
            if strategy_id not in self.strategies:
                logger.warning(f"Unknown strategy: {strategy_id}")
                continue

            try:
                analysis = self.analyze_maintenance_strategy(strategy_id, equipment_id, analysis_period_years)

                comparison_results.append({
                    'Strategy': analysis.strategy.strategy_name,
                    'NPV ($)': analysis.net_present_value,
                    'ROI (%)': analysis.return_on_investment,
                    'Payback (months)': analysis.payback_period_months,
                    'IRR (%)': analysis.internal_rate_of_return,
                    'Annual Costs ($)': analysis.annual_costs,
                    'Annual Benefits ($)': analysis.annual_benefits,
                    'Risk (VaR 95%)': analysis.value_at_risk_95,
                    'Confidence': analysis.confidence_level
                })

            except Exception as e:
                logger.error(f"Error analyzing strategy {strategy_id}: {e}")
                continue

        if not comparison_results:
            return pd.DataFrame()

        df = pd.DataFrame(comparison_results)

        # Sort by NPV descending
        df = df.sort_values('NPV ($)', ascending=False)

        return df

    def optimize_maintenance_portfolio(self, equipment_ids: List[str] = None) -> Dict[str, str]:
        """Optimize maintenance strategy portfolio across multiple equipment"""

        if equipment_ids is None:
            equipment_ids = list(self.equipment_economics.keys())

        optimal_strategies = {}

        for equipment_id in equipment_ids:
            logger.info(f"Optimizing strategy for equipment: {equipment_id}")

            # Compare all strategies for this equipment
            comparison = self.compare_strategies(equipment_id)

            if not comparison.empty:
                # Select strategy with highest NPV
                best_strategy = comparison.iloc[0]['Strategy']
                optimal_strategies[equipment_id] = best_strategy

                logger.info(f"Optimal strategy for {equipment_id}: {best_strategy}")
            else:
                logger.warning(f"No viable strategies found for {equipment_id}")
                optimal_strategies[equipment_id] = "reactive"  # Default fallback

        return optimal_strategies

    def generate_business_case(self, strategy_id: str, equipment_id: str,
                              analysis_period_years: int = 5) -> Dict[str, any]:
        """Generate comprehensive business case for maintenance strategy"""

        analysis = self.analyze_maintenance_strategy(strategy_id, equipment_id, analysis_period_years)
        strategy = self.strategies[strategy_id]
        equipment = self.equipment_economics[equipment_id]

        # Executive summary
        executive_summary = {
            'strategy_name': strategy.strategy_name,
            'equipment': equipment.equipment_type,
            'investment_required': sum(cost.amount for cost in strategy.costs if not cost.is_recurring),
            'net_present_value': analysis.net_present_value,
            'return_on_investment': analysis.return_on_investment,
            'payback_period_months': analysis.payback_period_months,
            'confidence_level': analysis.confidence_level,
            'recommendation': 'APPROVE' if analysis.net_present_value > 0 and analysis.return_on_investment > 15 else 'REVIEW'
        }

        # Financial summary
        financial_summary = {
            'total_costs_npv': analysis.total_costs_npv,
            'total_benefits_npv': analysis.total_benefits_npv,
            'annual_costs': analysis.annual_costs,
            'annual_benefits': analysis.annual_benefits,
            'break_even_threshold': analysis.break_even_threshold,
            'best_case_npv': analysis.best_case_npv,
            'worst_case_npv': analysis.worst_case_npv
        }

        # Risk assessment
        risk_assessment = {
            'implementation_risk': strategy.implementation_risk,
            'technology_risk': strategy.technology_risk,
            'value_at_risk_95': analysis.value_at_risk_95,
            'expected_shortfall': analysis.expected_shortfall,
            'risk_mitigation_strategies': [
                'Phased implementation to reduce risk',
                'Pilot program before full deployment',
                'Regular performance monitoring',
                'Fallback to previous strategy if needed'
            ]
        }

        # Strategic benefits
        strategic_benefits = [
            f"Failure reduction: {(1-strategy.failure_reduction_factor)*100:.0f}%",
            f"Life extension: {(strategy.life_extension_factor-1)*100:.0f}%",
            f"Performance improvement: {strategy.performance_improvement*100:.0f}%",
            "Enhanced equipment reliability",
            "Improved operational efficiency",
            "Reduced safety risks"
        ]

        return {
            'executive_summary': executive_summary,
            'financial_summary': financial_summary,
            'risk_assessment': risk_assessment,
            'strategic_benefits': strategic_benefits,
            'detailed_analysis': analysis,
            'generated_date': datetime.now().isoformat()
        }