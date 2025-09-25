"""
Parts Inventory Management System for Phase 3.1 IoT Predictive Maintenance
Intelligent ordering, supply chain optimization, and maintenance integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
import uuid
import warnings
from enum import Enum

# ML imports for demand forecasting
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optimization
import pulp

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class PartCategory(Enum):
    """Part categories for classification"""
    CRITICAL = "critical"
    STANDARD = "standard"
    CONSUMABLE = "consumable"
    SAFETY = "safety"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    HYDRAULIC = "hydraulic"
    ELECTRONIC = "electronic"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    ORDERED = "ordered"
    SHIPPED = "shipped"
    RECEIVED = "received"
    CANCELLED = "cancelled"


class SupplierRating(Enum):
    """Supplier rating levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


@dataclass
class PartSpecification:
    """Detailed part specification"""
    part_id: str
    part_number: str
    name: str
    description: str
    category: PartCategory
    manufacturer: str
    model: str
    specifications: Dict[str, Any]
    compatible_equipment: List[str]
    installation_time_hours: float = 1.0
    required_skills: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    storage_requirements: Dict[str, str] = field(default_factory=dict)


@dataclass
class InventoryItem:
    """Enhanced inventory item with analytics"""
    part_id: str
    current_stock: int
    min_stock_level: int
    max_stock_level: int
    reorder_point: int
    economic_order_quantity: int
    unit_cost: float
    storage_cost_per_unit: float
    location: str
    bin_number: str
    last_updated: datetime
    reserved_quantity: int = 0
    on_order_quantity: int = 0
    last_used_date: Optional[datetime] = None
    usage_frequency: float = 0.0  # Uses per month
    lead_time_days: int = 7
    demand_variance: float = 0.0
    criticality_score: float = 1.0


@dataclass
class Supplier:
    """Supplier information and performance metrics"""
    supplier_id: str
    name: str
    contact_info: Dict[str, str]
    rating: SupplierRating
    delivery_performance: float  # 0-1 (percentage on-time)
    quality_score: float  # 0-1
    cost_competitiveness: float  # 0-1
    lead_time_reliability: float  # 0-1
    payment_terms: str
    minimum_order_value: float
    bulk_discount_tiers: Dict[int, float] = field(default_factory=dict)
    preferred_parts: List[str] = field(default_factory=list)
    geographical_location: str = ""
    certification_level: str = "standard"


@dataclass
class PurchaseOrder:
    """Purchase order with tracking"""
    order_id: str
    supplier_id: str
    order_date: datetime
    expected_delivery: datetime
    total_value: float
    status: OrderStatus
    items: List[Dict[str, Any]]
    priority: str = "normal"
    approval_required: bool = True
    approved_by: Optional[str] = None
    tracking_number: Optional[str] = None
    actual_delivery: Optional[datetime] = None
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class DemandForecast:
    """Demand forecast for parts"""
    part_id: str
    forecast_date: datetime
    forecast_period_days: int
    predicted_demand: float
    confidence_interval: Tuple[float, float]
    seasonal_factor: float
    trend_factor: float
    model_accuracy: float


class PartsInventoryManager:
    """Advanced parts inventory management with ML optimization"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Parts Inventory Manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Core data structures
        self.parts_catalog = {}  # part_id -> PartSpecification
        self.inventory = {}  # part_id -> InventoryItem
        self.suppliers = {}  # supplier_id -> Supplier
        self.purchase_orders = {}  # order_id -> PurchaseOrder

        # Historical data for analytics
        self.usage_history = pd.DataFrame()
        self.order_history = pd.DataFrame()
        self.supplier_performance = pd.DataFrame()

        # ML models
        self.demand_forecaster = None
        self.optimization_model = None

        # Configuration parameters
        self.service_level_target = 0.95  # 95% service level
        self.carrying_cost_rate = 0.25  # 25% annual carrying cost
        self.shortage_cost_multiplier = 5.0  # Cost of stockout

        # Initialize sample data
        self._initialize_sample_data()

        logger.info("Initialized Parts Inventory Manager")

    def _initialize_sample_data(self):
        """Initialize sample data for demonstration"""
        # Sample parts catalog
        sample_parts = [
            PartSpecification(
                "BRG001", "SKF-6308", "High-Performance Bearing",
                "Deep groove ball bearing for motor applications",
                PartCategory.CRITICAL, "SKF", "6308-2RS1",
                {"bore": "40mm", "outer_diameter": "90mm", "width": "23mm"},
                ["EQ_001", "EQ_003"], 2.0, ["mechanical"], ["ppe_required"],
                {"temperature": "dry_cool", "humidity": "low"}
            ),
            PartSpecification(
                "FLT001", "HYDAC-0160", "Hydraulic Filter",
                "High-efficiency hydraulic return filter",
                PartCategory.STANDARD, "HYDAC", "0160R010BN4HC",
                {"micron_rating": "10", "flow_rate": "160L/min"},
                ["EQ_001", "EQ_002"], 0.5, ["hydraulic"], [],
                {"temperature": "ambient"}
            ),
            PartSpecification(
                "SNS001", "TEMP-PT100", "Temperature Sensor",
                "Platinum RTD temperature sensor",
                PartCategory.ELECTRONIC, "Omega", "PR-13-2-100-1/8-6-E",
                {"range": "-200 to 850°C", "accuracy": "±0.3°C"},
                ["EQ_002", "EQ_004"], 1.5, ["electrical", "sensors"], ["esd_protection"],
                {"temperature": "controlled", "humidity": "low"}
            ),
            PartSpecification(
                "BLT001", "GATES-5VX800", "V-Belt",
                "High-performance V-belt for power transmission",
                PartCategory.CONSUMABLE, "Gates", "5VX800",
                {"length": "80 inches", "width": "5/8 inch"},
                ["EQ_003", "EQ_005"], 0.75, ["mechanical"], [],
                {"temperature": "ambient"}
            ),
            PartSpecification(
                "SFT001", "SAFETY-KIT", "Safety Kit",
                "Complete safety kit for maintenance work",
                PartCategory.SAFETY, "3M", "FALL-PROTECT-KIT",
                {"components": ["harness", "lanyard", "helmet", "gloves"]},
                ["ALL"], 0.25, [], ["safety_critical"],
                {"temperature": "ambient"}
            )
        ]

        for part in sample_parts:
            self.parts_catalog[part.part_id] = part

        # Sample inventory
        sample_inventory = [
            InventoryItem("BRG001", 25, 10, 50, 15, 20, 125.50, 2.50, "Warehouse-A", "A-1-05", datetime.now()),
            InventoryItem("FLT001", 8, 15, 40, 20, 25, 45.75, 1.25, "Warehouse-A", "A-2-03", datetime.now()),
            InventoryItem("SNS001", 12, 5, 30, 8, 15, 85.00, 1.75, "Warehouse-B", "B-1-12", datetime.now()),
            InventoryItem("BLT001", 35, 20, 100, 30, 40, 28.50, 0.50, "Warehouse-A", "A-3-08", datetime.now()),
            InventoryItem("SFT001", 5, 8, 20, 10, 12, 250.00, 5.00, "Safety-Storage", "S-1-01", datetime.now())
        ]

        for item in sample_inventory:
            self.inventory[item.part_id] = item

        # Sample suppliers
        sample_suppliers = [
            Supplier(
                "SUP001", "Industrial Parts Corp",
                {"email": "orders@ipc.com", "phone": "+1-555-0101"},
                SupplierRating.EXCELLENT, 0.95, 0.98, 0.85, 0.92,
                "Net 30", 500.0, {100: 0.05, 500: 0.10, 1000: 0.15},
                ["BRG001", "FLT001"], "Chicago, IL", "ISO9001"
            ),
            Supplier(
                "SUP002", "ElectroTech Solutions",
                {"email": "sales@electrotech.com", "phone": "+1-555-0202"},
                SupplierRating.GOOD, 0.88, 0.95, 0.90, 0.85,
                "Net 45", 250.0, {50: 0.03, 200: 0.08},
                ["SNS001"], "Austin, TX", "ISO9001"
            ),
            Supplier(
                "SUP003", "Safety First Inc",
                {"email": "orders@safetyfirst.com", "phone": "+1-555-0303"},
                SupplierRating.GOOD, 0.92, 0.97, 0.75, 0.90,
                "Net 15", 100.0, {25: 0.05, 100: 0.12},
                ["SFT001"], "Denver, CO", "safety_certified"
            )
        ]

        for supplier in sample_suppliers:
            self.suppliers[supplier.supplier_id] = supplier

    def add_part_to_catalog(self, part_spec: PartSpecification) -> bool:
        """Add new part to catalog

        Args:
            part_spec: Part specification

        Returns:
            True if added successfully
        """
        try:
            self.parts_catalog[part_spec.part_id] = part_spec

            # Initialize inventory item if not exists
            if part_spec.part_id not in self.inventory:
                self.inventory[part_spec.part_id] = InventoryItem(
                    part_spec.part_id, 0, 5, 50, 10, 20, 0.0, 0.0,
                    "Warehouse-A", "TBD", datetime.now()
                )

            logger.info(f"Added part to catalog: {part_spec.name}")
            return True

        except Exception as e:
            logger.error(f"Error adding part to catalog: {e}")
            return False

    def update_inventory_levels(self, part_id: str, quantity_change: int,
                              transaction_type: str = "usage") -> bool:
        """Update inventory levels with transaction tracking

        Args:
            part_id: Part ID
            quantity_change: Change in quantity (negative for usage)
            transaction_type: Type of transaction

        Returns:
            True if updated successfully
        """
        if part_id not in self.inventory:
            logger.error(f"Part {part_id} not found in inventory")
            return False

        try:
            item = self.inventory[part_id]

            # Update stock level
            new_stock = max(0, item.current_stock + quantity_change)
            item.current_stock = new_stock
            item.last_updated = datetime.now()

            # Update usage tracking
            if quantity_change < 0:  # Usage
                item.last_used_date = datetime.now()
                item.usage_frequency = self._calculate_usage_frequency(part_id)

            # Check for reorder point
            if new_stock <= item.reorder_point:
                self._trigger_reorder_alert(part_id)

            logger.info(f"Updated inventory for {part_id}: {item.current_stock} units")
            return True

        except Exception as e:
            logger.error(f"Error updating inventory: {e}")
            return False

    def forecast_demand(self, part_id: str, forecast_days: int = 90) -> Optional[DemandForecast]:
        """Forecast demand for a specific part using ML

        Args:
            part_id: Part ID
            forecast_days: Forecast horizon in days

        Returns:
            Demand forecast or None if unable to forecast
        """
        if part_id not in self.inventory:
            logger.error(f"Part {part_id} not found")
            return None

        try:
            # Generate synthetic historical usage data for demo
            historical_data = self._generate_usage_history(part_id, 365)

            if len(historical_data) < 30:
                logger.warning(f"Insufficient data for forecasting {part_id}")
                return self._create_simple_forecast(part_id, forecast_days)

            # Prepare features
            features = self._prepare_demand_features(historical_data)
            target = historical_data['daily_usage'].values

            # Train simple model
            if self.demand_forecaster is None:
                self.demand_forecaster = RandomForestRegressor(n_estimators=50, random_state=42)

            # Split data for validation
            if len(features) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                self.demand_forecaster.fit(X_train, y_train)

                # Calculate accuracy
                y_pred = self.demand_forecaster.predict(X_test)
                accuracy = 1.0 - mean_absolute_error(y_test, y_pred) / (np.mean(y_test) + 1e-6)
            else:
                self.demand_forecaster.fit(features, target)
                accuracy = 0.7  # Default accuracy

            # Generate forecast
            future_features = self._generate_future_features(part_id, forecast_days)
            predicted_demand = self.demand_forecaster.predict(future_features).sum()

            # Calculate confidence intervals (simplified)
            demand_std = np.std(target)
            confidence_lower = max(0, predicted_demand - 1.96 * demand_std)
            confidence_upper = predicted_demand + 1.96 * demand_std

            # Calculate seasonal and trend factors
            seasonal_factor = self._calculate_seasonal_factor(part_id, datetime.now())
            trend_factor = self._calculate_trend_factor(historical_data)

            forecast = DemandForecast(
                part_id=part_id,
                forecast_date=datetime.now(),
                forecast_period_days=forecast_days,
                predicted_demand=predicted_demand * seasonal_factor * trend_factor,
                confidence_interval=(confidence_lower, confidence_upper),
                seasonal_factor=seasonal_factor,
                trend_factor=trend_factor,
                model_accuracy=max(0.1, min(0.95, accuracy))
            )

            logger.info(f"Generated demand forecast for {part_id}: {predicted_demand:.1f} units")
            return forecast

        except Exception as e:
            logger.error(f"Error forecasting demand for {part_id}: {e}")
            return self._create_simple_forecast(part_id, forecast_days)

    def _generate_usage_history(self, part_id: str, days: int) -> pd.DataFrame:
        """Generate synthetic usage history for demonstration

        Args:
            part_id: Part ID
            days: Number of days of history

        Returns:
            Historical usage DataFrame
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Base usage patterns by part category
        item = self.inventory.get(part_id)
        part_spec = self.parts_catalog.get(part_id)

        if part_spec:
            if part_spec.category == PartCategory.CRITICAL:
                base_usage = 0.5  # Lower usage, high impact
                volatility = 0.3
            elif part_spec.category == PartCategory.CONSUMABLE:
                base_usage = 2.0  # Higher usage
                volatility = 0.5
            else:
                base_usage = 1.0
                volatility = 0.4
        else:
            base_usage = 1.0
            volatility = 0.4

        # Generate synthetic data with patterns
        usage_data = []
        for i, date in enumerate(dates):
            # Seasonal pattern
            seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25)

            # Weekly pattern (lower on weekends)
            weekly = 0.3 if date.weekday() >= 5 else 1.0

            # Trend (slight increase over time)
            trend = 1.0 + 0.1 * (i / days)

            # Random variation
            noise = np.random.lognormal(0, volatility)

            daily_usage = max(0, base_usage * seasonal * weekly * trend * noise)

            usage_data.append({
                'date': date,
                'daily_usage': daily_usage,
                'day_of_week': date.weekday(),
                'month': date.month,
                'is_weekend': date.weekday() >= 5
            })

        return pd.DataFrame(usage_data)

    def _prepare_demand_features(self, historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for demand forecasting

        Args:
            historical_data: Historical usage data

        Returns:
            Feature matrix
        """
        features = []

        for _, row in historical_data.iterrows():
            feature_row = [
                row['day_of_week'],
                row['month'],
                int(row['is_weekend']),
                np.sin(2 * np.pi * row['month'] / 12),  # Seasonal encoding
                np.cos(2 * np.pi * row['month'] / 12),
                np.sin(2 * np.pi * row['day_of_week'] / 7),  # Weekly encoding
                np.cos(2 * np.pi * row['day_of_week'] / 7)
            ]
            features.append(feature_row)

        return np.array(features)

    def _generate_future_features(self, part_id: str, forecast_days: int) -> np.ndarray:
        """Generate features for future demand forecasting

        Args:
            part_id: Part ID
            forecast_days: Number of days to forecast

        Returns:
            Future feature matrix
        """
        future_dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
        features = []

        for date in future_dates:
            feature_row = [
                date.weekday(),
                date.month,
                int(date.weekday() >= 5),
                np.sin(2 * np.pi * date.month / 12),
                np.cos(2 * np.pi * date.month / 12),
                np.sin(2 * np.pi * date.weekday() / 7),
                np.cos(2 * np.pi * date.weekday() / 7)
            ]
            features.append(feature_row)

        return np.array(features)

    def _create_simple_forecast(self, part_id: str, forecast_days: int) -> DemandForecast:
        """Create simple forecast when ML model fails

        Args:
            part_id: Part ID
            forecast_days: Forecast horizon

        Returns:
            Simple demand forecast
        """
        item = self.inventory.get(part_id)

        # Use current usage frequency or default
        if item and item.usage_frequency > 0:
            monthly_usage = item.usage_frequency
        else:
            monthly_usage = 5.0  # Default assumption

        daily_usage = monthly_usage / 30.0
        predicted_demand = daily_usage * forecast_days

        return DemandForecast(
            part_id=part_id,
            forecast_date=datetime.now(),
            forecast_period_days=forecast_days,
            predicted_demand=predicted_demand,
            confidence_interval=(predicted_demand * 0.7, predicted_demand * 1.3),
            seasonal_factor=1.0,
            trend_factor=1.0,
            model_accuracy=0.6
        )

    def _calculate_seasonal_factor(self, part_id: str, date: datetime) -> float:
        """Calculate seasonal adjustment factor

        Args:
            part_id: Part ID
            date: Date for calculation

        Returns:
            Seasonal factor
        """
        part_spec = self.parts_catalog.get(part_id)

        if not part_spec:
            return 1.0

        # Different seasonal patterns by category
        month = date.month

        if part_spec.category == PartCategory.CRITICAL:
            # Higher usage in winter due to equipment stress
            return 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
        elif part_spec.category == PartCategory.CONSUMABLE:
            # Higher usage in summer due to increased operations
            return 1.0 + 0.2 * np.sin(2 * np.pi * (month - 7) / 12)
        else:
            return 1.0

    def _calculate_trend_factor(self, historical_data: pd.DataFrame) -> float:
        """Calculate trend factor from historical data

        Args:
            historical_data: Historical usage data

        Returns:
            Trend factor
        """
        if len(historical_data) < 10:
            return 1.0

        # Simple linear trend calculation
        x = np.arange(len(historical_data))
        y = historical_data['daily_usage'].values

        try:
            slope = np.polyfit(x, y, 1)[0]
            # Convert slope to growth factor
            trend_factor = 1.0 + slope * 30  # 30-day projection
            return max(0.5, min(2.0, trend_factor))  # Bound the trend
        except:
            return 1.0

    def _calculate_usage_frequency(self, part_id: str) -> float:
        """Calculate usage frequency for a part

        Args:
            part_id: Part ID

        Returns:
            Usage frequency (uses per month)
        """
        # Simplified calculation - would use actual historical data
        part_spec = self.parts_catalog.get(part_id)

        if not part_spec:
            return 1.0

        # Estimate based on part category and equipment compatibility
        base_frequency = {
            PartCategory.CRITICAL: 2.0,
            PartCategory.STANDARD: 5.0,
            PartCategory.CONSUMABLE: 15.0,
            PartCategory.SAFETY: 1.0,
            PartCategory.ELECTRICAL: 3.0,
            PartCategory.MECHANICAL: 4.0,
            PartCategory.HYDRAULIC: 6.0,
            PartCategory.ELECTRONIC: 2.0
        }.get(part_spec.category, 3.0)

        # Adjust for number of compatible equipment
        equipment_factor = len(part_spec.compatible_equipment) * 0.5

        return base_frequency + equipment_factor

    def optimize_inventory_levels(self) -> Dict[str, Dict[str, float]]:
        """Optimize inventory levels using economic order quantity and safety stock

        Returns:
            Optimization recommendations
        """
        recommendations = {}

        for part_id, item in self.inventory.items():
            try:
                # Get demand forecast
                forecast = self.forecast_demand(part_id, 90)

                if not forecast:
                    continue

                # Calculate optimal parameters
                annual_demand = forecast.predicted_demand * (365 / forecast.forecast_period_days)

                # Economic Order Quantity (EOQ)
                ordering_cost = 50.0  # Assumed ordering cost
                holding_cost = item.unit_cost * self.carrying_cost_rate

                if holding_cost > 0:
                    eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
                else:
                    eoq = item.economic_order_quantity

                # Safety stock calculation
                service_factor = 1.65  # For 95% service level
                demand_std = (forecast.confidence_interval[1] - forecast.confidence_interval[0]) / 4
                lead_time_demand = annual_demand * (item.lead_time_days / 365)
                safety_stock = service_factor * demand_std * np.sqrt(item.lead_time_days / 365)

                # Reorder point
                reorder_point = lead_time_demand + safety_stock

                # Maximum stock level
                max_stock = reorder_point + eoq

                recommendations[part_id] = {
                    'current_stock': item.current_stock,
                    'recommended_eoq': max(1, int(eoq)),
                    'recommended_reorder_point': max(1, int(reorder_point)),
                    'recommended_max_stock': max(1, int(max_stock)),
                    'recommended_safety_stock': max(0, int(safety_stock)),
                    'annual_demand_forecast': annual_demand,
                    'optimization_potential': self._calculate_cost_savings(item, eoq, reorder_point, max_stock)
                }

            except Exception as e:
                logger.error(f"Error optimizing inventory for {part_id}: {e}")

        logger.info(f"Generated inventory optimization for {len(recommendations)} parts")
        return recommendations

    def _calculate_cost_savings(self, current_item: InventoryItem, eoq: float,
                               reorder_point: float, max_stock: float) -> float:
        """Calculate potential cost savings from optimization

        Args:
            current_item: Current inventory item
            eoq: Recommended EOQ
            reorder_point: Recommended reorder point
            max_stock: Recommended max stock

        Returns:
            Estimated annual cost savings
        """
        # Current holding cost
        current_holding = current_item.current_stock * current_item.storage_cost_per_unit * 12

        # Optimized holding cost (average inventory)
        optimized_avg_inventory = (reorder_point + max_stock) / 2
        optimized_holding = optimized_avg_inventory * current_item.storage_cost_per_unit * 12

        # Stockout cost reduction (simplified)
        stockout_reduction = max(0, (current_item.reorder_point - reorder_point) *
                               current_item.unit_cost * self.shortage_cost_multiplier)

        # Total savings
        holding_savings = max(0, current_holding - optimized_holding)
        total_savings = holding_savings + stockout_reduction

        return total_savings

    def create_purchase_order(self, part_requirements: List[Dict[str, Any]],
                            supplier_preference: Optional[str] = None) -> Optional[PurchaseOrder]:
        """Create optimized purchase order

        Args:
            part_requirements: List of parts and quantities needed
            supplier_preference: Preferred supplier ID

        Returns:
            Created purchase order
        """
        try:
            # Optimize supplier selection
            optimal_suppliers = self._optimize_supplier_selection(part_requirements, supplier_preference)

            if not optimal_suppliers:
                logger.error("No suitable suppliers found")
                return None

            # Create order for the best supplier combination
            best_supplier_id = list(optimal_suppliers.keys())[0]
            supplier = self.suppliers[best_supplier_id]

            # Generate order
            order_id = f"PO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            order_items = []
            total_value = 0.0

            for requirement in part_requirements:
                part_id = requirement['part_id']
                quantity = requirement['quantity']

                if part_id in self.inventory:
                    item = self.inventory[part_id]
                    line_total = quantity * item.unit_cost

                    order_items.append({
                        'part_id': part_id,
                        'part_name': self.parts_catalog.get(part_id, PartSpecification('', '', '', '', PartCategory.STANDARD, '', '', {}, [], 0)).name,
                        'quantity': quantity,
                        'unit_cost': item.unit_cost,
                        'line_total': line_total
                    })

                    total_value += line_total

            # Calculate delivery date
            expected_delivery = datetime.now() + timedelta(days=supplier.lead_time_reliability * 10)

            # Determine if approval is required
            approval_required = total_value > 1000.0  # Threshold for approval

            order = PurchaseOrder(
                order_id=order_id,
                supplier_id=best_supplier_id,
                order_date=datetime.now(),
                expected_delivery=expected_delivery,
                total_value=total_value,
                status=OrderStatus.PENDING,
                items=order_items,
                priority="normal",
                approval_required=approval_required
            )

            self.purchase_orders[order_id] = order

            # Update on-order quantities
            for item in order_items:
                if item['part_id'] in self.inventory:
                    self.inventory[item['part_id']].on_order_quantity += item['quantity']

            logger.info(f"Created purchase order {order_id} for ${total_value:.2f}")
            return order

        except Exception as e:
            logger.error(f"Error creating purchase order: {e}")
            return None

    def _optimize_supplier_selection(self, part_requirements: List[Dict[str, Any]],
                                   preferred_supplier: Optional[str] = None) -> Dict[str, float]:
        """Optimize supplier selection using multi-criteria optimization

        Args:
            part_requirements: Required parts and quantities
            preferred_supplier: Preferred supplier ID

        Returns:
            Optimized supplier scores
        """
        supplier_scores = {}

        for supplier_id, supplier in self.suppliers.items():
            # Skip if preferred supplier is specified and this isn't it
            if preferred_supplier and supplier_id != preferred_supplier:
                continue

            # Check if supplier can provide required parts
            available_parts = set(supplier.preferred_parts)
            required_parts = {req['part_id'] for req in part_requirements}

            if not available_parts.intersection(required_parts):
                continue  # Supplier can't provide any required parts

            # Calculate composite score
            # Factors: cost, quality, delivery, reliability
            cost_score = supplier.cost_competitiveness
            quality_score = supplier.quality_score
            delivery_score = supplier.delivery_performance
            reliability_score = supplier.lead_time_reliability

            # Weighted composite score
            composite_score = (
                cost_score * 0.3 +
                quality_score * 0.25 +
                delivery_score * 0.25 +
                reliability_score * 0.2
            )

            # Bonus for preferred parts coverage
            coverage = len(available_parts.intersection(required_parts)) / len(required_parts)
            composite_score *= (0.5 + 0.5 * coverage)

            supplier_scores[supplier_id] = composite_score

        # Sort by score (descending)
        return dict(sorted(supplier_scores.items(), key=lambda x: x[1], reverse=True))

    def _trigger_reorder_alert(self, part_id: str):
        """Trigger reorder alert for low stock

        Args:
            part_id: Part ID below reorder point
        """
        item = self.inventory.get(part_id)
        part_spec = self.parts_catalog.get(part_id)

        if not item or not part_spec:
            return

        # Generate automatic reorder suggestion
        forecast = self.forecast_demand(part_id, 30)

        if forecast:
            suggested_quantity = max(item.economic_order_quantity,
                                   int(forecast.predicted_demand * 1.2))  # 20% buffer
        else:
            suggested_quantity = item.economic_order_quantity

        alert = {
            'alert_id': f"REORDER_{part_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'part_id': part_id,
            'part_name': part_spec.name,
            'current_stock': item.current_stock,
            'reorder_point': item.reorder_point,
            'suggested_quantity': suggested_quantity,
            'urgency': 'high' if part_spec.category == PartCategory.CRITICAL else 'medium',
            'timestamp': datetime.now()
        }

        logger.warning(f"REORDER ALERT: {part_spec.name} (ID: {part_id}) - Stock: {item.current_stock}, Reorder Point: {item.reorder_point}")

    def get_inventory_analytics(self) -> Dict[str, Any]:
        """Get comprehensive inventory analytics

        Returns:
            Inventory analytics dictionary
        """
        analytics = {
            'summary': {},
            'parts_analysis': {},
            'supplier_performance': {},
            'cost_analysis': {},
            'optimization_opportunities': []
        }

        # Summary statistics
        total_parts = len(self.inventory)
        total_value = sum(item.current_stock * item.unit_cost for item in self.inventory.values())
        low_stock_count = sum(1 for item in self.inventory.values() if item.current_stock <= item.reorder_point)
        out_of_stock_count = sum(1 for item in self.inventory.values() if item.current_stock == 0)

        analytics['summary'] = {
            'total_parts': total_parts,
            'total_inventory_value': total_value,
            'low_stock_items': low_stock_count,
            'out_of_stock_items': out_of_stock_count,
            'average_turnover': self._calculate_average_turnover(),
            'service_level': self._calculate_service_level()
        }

        # Parts analysis by category
        category_analysis = defaultdict(lambda: {'count': 0, 'value': 0.0, 'low_stock': 0})

        for part_id, item in self.inventory.items():
            part_spec = self.parts_catalog.get(part_id)
            if part_spec:
                category = part_spec.category.value
                category_analysis[category]['count'] += 1
                category_analysis[category]['value'] += item.current_stock * item.unit_cost
                if item.current_stock <= item.reorder_point:
                    category_analysis[category]['low_stock'] += 1

        analytics['parts_analysis'] = dict(category_analysis)

        # Supplier performance
        supplier_perf = {}
        for supplier_id, supplier in self.suppliers.items():
            supplier_perf[supplier_id] = {
                'name': supplier.name,
                'rating': supplier.rating.value,
                'delivery_performance': supplier.delivery_performance,
                'quality_score': supplier.quality_score,
                'cost_competitiveness': supplier.cost_competitiveness
            }

        analytics['supplier_performance'] = supplier_perf

        # Cost analysis
        analytics['cost_analysis'] = {
            'total_carrying_cost': self._calculate_carrying_cost(),
            'total_ordering_cost': self._calculate_ordering_cost(),
            'potential_savings': self._calculate_potential_savings()
        }

        return analytics

    def _calculate_average_turnover(self) -> float:
        """Calculate average inventory turnover"""
        if not self.inventory:
            return 0.0

        # Simplified calculation
        turnover_rates = []
        for item in self.inventory.values():
            if item.current_stock > 0 and item.usage_frequency > 0:
                turnover = item.usage_frequency * 12 / item.current_stock  # Annual turnover
                turnover_rates.append(turnover)

        return np.mean(turnover_rates) if turnover_rates else 0.0

    def _calculate_service_level(self) -> float:
        """Calculate current service level"""
        if not self.inventory:
            return 0.0

        # Simplified calculation based on stock availability
        in_stock_items = sum(1 for item in self.inventory.values() if item.current_stock > 0)
        return (in_stock_items / len(self.inventory)) * 100

    def _calculate_carrying_cost(self) -> float:
        """Calculate total carrying cost"""
        return sum(item.current_stock * item.storage_cost_per_unit * 12
                  for item in self.inventory.values())

    def _calculate_ordering_cost(self) -> float:
        """Calculate estimated annual ordering cost"""
        # Simplified calculation
        total_orders = len(self.purchase_orders)
        average_ordering_cost = 50.0
        return total_orders * average_ordering_cost

    def _calculate_potential_savings(self) -> float:
        """Calculate potential savings from optimization"""
        optimization_results = self.optimize_inventory_levels()
        return sum(result.get('optimization_potential', 0)
                  for result in optimization_results.values())

    def generate_reorder_report(self) -> List[Dict[str, Any]]:
        """Generate comprehensive reorder report

        Returns:
            List of reorder recommendations
        """
        reorder_recommendations = []

        for part_id, item in self.inventory.items():
            # Check if reorder is needed
            available_stock = item.current_stock - item.reserved_quantity

            if available_stock <= item.reorder_point:
                part_spec = self.parts_catalog.get(part_id)
                forecast = self.forecast_demand(part_id, 60)

                # Calculate recommended order quantity
                if forecast:
                    # Order to cover forecast plus buffer
                    recommended_qty = max(
                        item.economic_order_quantity,
                        int(forecast.predicted_demand + forecast.confidence_interval[1] * 0.5)
                    )
                else:
                    recommended_qty = item.economic_order_quantity

                # Find best supplier
                suppliers = self._optimize_supplier_selection([{'part_id': part_id, 'quantity': recommended_qty}])
                best_supplier_id = list(suppliers.keys())[0] if suppliers else None

                recommendation = {
                    'part_id': part_id,
                    'part_name': part_spec.name if part_spec else 'Unknown',
                    'current_stock': item.current_stock,
                    'available_stock': available_stock,
                    'reorder_point': item.reorder_point,
                    'recommended_quantity': recommended_qty,
                    'urgency': self._calculate_urgency(item, part_spec),
                    'estimated_cost': recommended_qty * item.unit_cost,
                    'best_supplier': self.suppliers[best_supplier_id].name if best_supplier_id else 'None',
                    'expected_delivery_days': self.suppliers[best_supplier_id].lead_time_reliability * 10 if best_supplier_id else 14,
                    'forecast_accuracy': forecast.model_accuracy if forecast else 0.6
                }

                reorder_recommendations.append(recommendation)

        # Sort by urgency and cost impact
        reorder_recommendations.sort(key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['urgency']],
            -x['estimated_cost']
        ))

        logger.info(f"Generated reorder report with {len(reorder_recommendations)} recommendations")
        return reorder_recommendations

    def _calculate_urgency(self, item: InventoryItem, part_spec: Optional[PartSpecification]) -> str:
        """Calculate urgency level for reorder

        Args:
            item: Inventory item
            part_spec: Part specification

        Returns:
            Urgency level string
        """
        if item.current_stock == 0:
            return 'critical'

        if part_spec and part_spec.category == PartCategory.CRITICAL:
            if item.current_stock <= item.reorder_point * 0.5:
                return 'critical'
            elif item.current_stock <= item.reorder_point:
                return 'high'

        if part_spec and part_spec.category == PartCategory.SAFETY:
            return 'high'

        if item.current_stock <= item.reorder_point * 0.3:
            return 'high'
        elif item.current_stock <= item.reorder_point * 0.7:
            return 'medium'
        else:
            return 'low'


# Demo function
def create_demo_inventory_manager() -> PartsInventoryManager:
    """Create demo inventory manager with sample data

    Returns:
        Configured inventory manager
    """
    manager = PartsInventoryManager()

    # Simulate some usage to generate patterns
    for part_id in manager.inventory.keys():
        # Simulate random usage over time
        for _ in range(np.random.randint(1, 5)):
            usage = -np.random.randint(1, 3)
            manager.update_inventory_levels(part_id, usage, "maintenance_usage")

    logger.info("Created demo inventory manager with sample data")
    return manager