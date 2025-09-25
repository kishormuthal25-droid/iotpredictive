"""
Priority Calculator Module for IoT Anomaly Detection System
Intelligent priority calculation for maintenance tasks based on multiple factors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import math
from scipy.stats import norm, exponweib
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import fuzzy
import fuzzy.storage.fcl.Reader as FclReader

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """Risk categories for equipment failure"""
    SAFETY = "safety"           # Risk to human safety
    ENVIRONMENTAL = "environmental"  # Environmental impact
    PRODUCTION = "production"    # Production loss
    QUALITY = "quality"         # Product quality impact
    REGULATORY = "regulatory"   # Compliance/regulatory risk
    REPUTATION = "reputation"   # Brand/reputation damage


class BusinessImpact(Enum):
    """Business impact levels"""
    CATASTROPHIC = 5  # Complete shutdown, major safety risk
    SEVERE = 4        # Major production loss, significant cost
    MODERATE = 3      # Partial production loss, moderate cost
    MINOR = 2         # Small production impact, low cost
    NEGLIGIBLE = 1    # No significant impact


@dataclass
class PriorityFactors:
    """Factors influencing priority calculation"""
    # Anomaly factors
    anomaly_severity: float = 0.5  # 0-1 scale
    anomaly_confidence: float = 0.5  # 0-1 confidence level
    anomaly_frequency: int = 1      # Occurrence count
    time_since_last: float = 0      # Hours since last occurrence
    
    # Equipment factors
    equipment_criticality: str = 'medium'  # 'critical', 'high', 'medium', 'low'
    equipment_age: float = 0        # Years in service
    equipment_condition: float = 0.5  # 0-1 health score
    mtbf: float = 1000              # Mean time between failures (hours)
    mttr: float = 4                 # Mean time to repair (hours)
    
    # Business factors
    production_impact: float = 0    # Production loss per hour
    downtime_cost: float = 0        # Cost per hour of downtime
    safety_risk: bool = False        # Safety implications
    environmental_risk: bool = False  # Environmental implications
    regulatory_compliance: bool = False  # Regulatory requirements
    
    # Operational factors
    spare_parts_availability: float = 1.0  # 0-1 availability
    technician_availability: float = 1.0   # 0-1 availability
    weather_conditions: str = 'normal'     # Impact of weather
    shift_timing: str = 'day'             # Current shift
    
    # Historical factors
    failure_history: List[Dict] = field(default_factory=list)
    maintenance_history: List[Dict] = field(default_factory=list)
    false_positive_rate: float = 0.1  # Historical false positive rate
    
    # Dependencies
    upstream_equipment: List[str] = field(default_factory=list)
    downstream_equipment: List[str] = field(default_factory=list)
    redundancy_available: bool = False


@dataclass
class PriorityScore:
    """Priority calculation result"""
    final_score: float
    priority_level: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    risk_score: float
    urgency_score: float
    impact_score: float
    confidence_score: float
    component_scores: Dict[str, float]
    recommendations: List[str]
    estimated_time_to_failure: Optional[float] = None
    cost_of_delay: Optional[float] = None


class PriorityCalculator:
    """Main priority calculation engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Priority Calculator
        
        Args:
            config: Configuration parameters
        """
        self.config = config or self._get_default_config()
        self.weight_optimizer = WeightOptimizer()
        self.risk_calculator = RiskCalculator()
        self.fuzzy_engine = FuzzyPriorityEngine()
        self.ml_predictor = MLPriorityPredictor()
        
        # Historical data for learning
        self.priority_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=500)
        
        # Scalers for normalization
        self.scaler = MinMaxScaler()
        
        logger.info("Initialized Priority Calculator")
        
    def calculate_priority(self, 
                          factors: PriorityFactors,
                          use_ml: bool = True,
                          use_fuzzy: bool = True) -> PriorityScore:
        """Calculate priority score based on multiple factors
        
        Args:
            factors: Priority factors
            use_ml: Use machine learning prediction
            use_fuzzy: Use fuzzy logic
            
        Returns:
            Priority score and details
        """
        component_scores = {}
        
        # 1. Calculate risk score
        risk_score = self.risk_calculator.calculate_risk(factors)
        component_scores['risk'] = risk_score
        
        # 2. Calculate urgency score
        urgency_score = self._calculate_urgency(factors)
        component_scores['urgency'] = urgency_score
        
        # 3. Calculate impact score
        impact_score = self._calculate_impact(factors)
        component_scores['impact'] = impact_score
        
        # 4. Calculate confidence score
        confidence_score = self._calculate_confidence(factors)
        component_scores['confidence'] = confidence_score
        
        # 5. Calculate equipment score
        equipment_score = self._calculate_equipment_score(factors)
        component_scores['equipment'] = equipment_score
        
        # 6. Calculate operational score
        operational_score = self._calculate_operational_score(factors)
        component_scores['operational'] = operational_score
        
        # 7. Apply fuzzy logic if enabled
        if use_fuzzy:
            fuzzy_score = self.fuzzy_engine.calculate_priority(factors)
            component_scores['fuzzy'] = fuzzy_score
        else:
            fuzzy_score = 0
            
        # 8. Apply ML prediction if enabled
        if use_ml and len(self.priority_history) > 100:
            ml_score = self.ml_predictor.predict_priority(factors, self.priority_history)
            component_scores['ml_prediction'] = ml_score
        else:
            ml_score = 0
            
        # 9. Calculate weighted final score
        weights = self.config['weights']
        final_score = (
            weights['risk'] * risk_score +
            weights['urgency'] * urgency_score +
            weights['impact'] * impact_score +
            weights['confidence'] * confidence_score +
            weights['equipment'] * equipment_score +
            weights['operational'] * operational_score
        )
        
        # Include fuzzy and ML scores if available
        if use_fuzzy:
            final_score = 0.7 * final_score + 0.3 * fuzzy_score
        if use_ml and ml_score > 0:
            final_score = 0.8 * final_score + 0.2 * ml_score
            
        # Normalize final score to 0-100
        final_score = min(100, max(0, final_score * 100))
        
        # Determine priority level
        priority_level = self._determine_priority_level(final_score)
        
        # Calculate additional metrics
        ttf = self._estimate_time_to_failure(factors)
        cod = self._calculate_cost_of_delay(factors, ttf)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(factors, component_scores)
        
        # Create result
        result = PriorityScore(
            final_score=final_score,
            priority_level=priority_level,
            risk_score=risk_score,
            urgency_score=urgency_score,
            impact_score=impact_score,
            confidence_score=confidence_score,
            component_scores=component_scores,
            recommendations=recommendations,
            estimated_time_to_failure=ttf,
            cost_of_delay=cod
        )
        
        # Store in history for learning
        self.priority_history.append({
            'factors': factors,
            'result': result,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Calculated priority: {priority_level} (score: {final_score:.2f})")
        
        return result
        
    def _calculate_urgency(self, factors: PriorityFactors) -> float:
        """Calculate urgency score based on time-sensitive factors"""
        urgency = 0.0
        
        # Time since last occurrence (more recent = more urgent)
        if factors.time_since_last < 1:  # Within last hour
            urgency += 0.3
        elif factors.time_since_last < 24:  # Within last day
            urgency += 0.2
        elif factors.time_since_last < 168:  # Within last week
            urgency += 0.1
            
        # Frequency of occurrence
        if factors.anomaly_frequency > 10:
            urgency += 0.3
        elif factors.anomaly_frequency > 5:
            urgency += 0.2
        elif factors.anomaly_frequency > 2:
            urgency += 0.1
            
        # Equipment condition deterioration
        if factors.equipment_condition < 0.3:
            urgency += 0.2
        elif factors.equipment_condition < 0.5:
            urgency += 0.1
            
        # Safety or environmental risk
        if factors.safety_risk:
            urgency += 0.3
        if factors.environmental_risk:
            urgency += 0.2
            
        # Normalize to 0-1
        return min(1.0, urgency)
        
    def _calculate_impact(self, factors: PriorityFactors) -> float:
        """Calculate business impact score"""
        impact = 0.0
        
        # Production impact
        if factors.production_impact > 0:
            # Normalize production impact (assume max 100% loss)
            impact += min(1.0, factors.production_impact / 100) * 0.3
            
        # Downtime cost
        if factors.downtime_cost > 0:
            # Normalize cost (assume max $10000/hour as catastrophic)
            impact += min(1.0, factors.downtime_cost / 10000) * 0.3
            
        # Equipment criticality
        criticality_scores = {
            'critical': 0.3,
            'high': 0.2,
            'medium': 0.1,
            'low': 0.05
        }
        impact += criticality_scores.get(factors.equipment_criticality, 0.1)
        
        # Dependencies
        if factors.upstream_equipment:
            impact += 0.05 * min(3, len(factors.upstream_equipment))
        if factors.downstream_equipment:
            impact += 0.05 * min(3, len(factors.downstream_equipment))
            
        # No redundancy increases impact
        if not factors.redundancy_available:
            impact *= 1.2
            
        return min(1.0, impact)
        
    def _calculate_confidence(self, factors: PriorityFactors) -> float:
        """Calculate confidence in the priority assessment"""
        confidence = factors.anomaly_confidence
        
        # Adjust for false positive rate
        confidence *= (1 - factors.false_positive_rate)
        
        # Boost confidence if pattern is recurring
        if factors.anomaly_frequency > 3:
            confidence = min(1.0, confidence * 1.2)
            
        # Reduce confidence for new equipment
        if factors.equipment_age < 0.5:  # Less than 6 months
            confidence *= 0.8
            
        return confidence
        
    def _calculate_equipment_score(self, factors: PriorityFactors) -> float:
        """Calculate equipment-related priority score"""
        score = 0.0
        
        # Equipment age factor (older equipment more prone to failure)
        if factors.equipment_age > 10:
            score += 0.2
        elif factors.equipment_age > 5:
            score += 0.15
        elif factors.equipment_age > 2:
            score += 0.1
            
        # MTBF consideration
        if factors.mtbf < 100:  # Frequent failures
            score += 0.2
        elif factors.mtbf < 500:
            score += 0.1
            
        # MTTR consideration
        if factors.mttr > 8:  # Long repair time
            score += 0.2
        elif factors.mttr > 4:
            score += 0.1
            
        # Equipment condition
        score += (1 - factors.equipment_condition) * 0.3
        
        # Maintenance history
        recent_maintenance = self._check_recent_maintenance(factors.maintenance_history)
        if not recent_maintenance:
            score += 0.1
            
        return min(1.0, score)
        
    def _calculate_operational_score(self, factors: PriorityFactors) -> float:
        """Calculate operational factors score"""
        score = 0.0
        
        # Spare parts availability (low availability = higher priority)
        score += (1 - factors.spare_parts_availability) * 0.2
        
        # Technician availability
        score += (1 - factors.technician_availability) * 0.2
        
        # Weather conditions
        if factors.weather_conditions == 'severe':
            score += 0.2
        elif factors.weather_conditions == 'poor':
            score += 0.1
            
        # Shift timing (night/weekend = higher priority for scheduling)
        if factors.shift_timing in ['night', 'weekend']:
            score += 0.1
            
        return min(1.0, score)
        
    def _determine_priority_level(self, score: float) -> str:
        """Determine priority level from score"""
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def _estimate_time_to_failure(self, factors: PriorityFactors) -> Optional[float]:
        """Estimate time to failure in hours"""
        if not factors.failure_history:
            return None
            
        # Simple estimation based on degradation rate
        if factors.equipment_condition <= 0:
            return 0
            
        # Estimate based on condition degradation
        degradation_rate = self._calculate_degradation_rate(factors)
        if degradation_rate > 0:
            ttf = factors.equipment_condition / degradation_rate
            return max(0, ttf)
            
        return None
        
    def _calculate_degradation_rate(self, factors: PriorityFactors) -> float:
        """Calculate equipment degradation rate"""
        # Simplified degradation model
        base_rate = 0.001  # Base degradation per hour
        
        # Adjust for age
        age_factor = 1 + (factors.equipment_age / 10)
        
        # Adjust for anomaly severity
        severity_factor = 1 + factors.anomaly_severity
        
        # Adjust for frequency
        frequency_factor = 1 + (factors.anomaly_frequency / 10)
        
        return base_rate * age_factor * severity_factor * frequency_factor
        
    def _calculate_cost_of_delay(self, 
                                factors: PriorityFactors,
                                ttf: Optional[float]) -> Optional[float]:
        """Calculate cost of delaying maintenance"""
        if factors.downtime_cost <= 0:
            return None
            
        # Base cost is hourly downtime cost
        base_cost = factors.downtime_cost
        
        # Add production impact cost
        production_cost = factors.production_impact * 100  # Convert to $/hour
        
        # Calculate total hourly cost
        hourly_cost = base_cost + production_cost
        
        # If we have TTF, calculate expected cost
        if ttf:
            # Probability of failure increases over time
            failure_probability = 1 - math.exp(-ttf / factors.mtbf)
            expected_cost = hourly_cost * factors.mttr * failure_probability
            return expected_cost
            
        return hourly_cost * factors.mttr
        
    def _check_recent_maintenance(self, 
                                 maintenance_history: List[Dict],
                                 days: int = 30) -> bool:
        """Check if maintenance was performed recently"""
        if not maintenance_history:
            return False
            
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for maintenance in maintenance_history:
            if 'date' in maintenance:
                maintenance_date = maintenance['date']
                if isinstance(maintenance_date, str):
                    maintenance_date = datetime.fromisoformat(maintenance_date)
                if maintenance_date > cutoff_date:
                    return True
                    
        return False
        
    def _generate_recommendations(self, 
                                 factors: PriorityFactors,
                                 scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High risk recommendations
        if scores.get('risk', 0) > 0.7:
            recommendations.append("⚠️ High risk detected - immediate inspection recommended")
            
        # Urgency-based recommendations
        if scores.get('urgency', 0) > 0.7:
            recommendations.append("🔴 Urgent action required - schedule maintenance immediately")
        elif scores.get('urgency', 0) > 0.5:
            recommendations.append("🟡 Schedule maintenance within 24 hours")
            
        # Equipment-based recommendations
        if factors.equipment_condition < 0.3:
            recommendations.append("🔧 Equipment in poor condition - comprehensive overhaul recommended")
            
        # Spare parts recommendations
        if factors.spare_parts_availability < 0.5:
            recommendations.append("📦 Order spare parts immediately to avoid delays")
            
        # Safety recommendations
        if factors.safety_risk:
            recommendations.append("⛑️ Safety risk identified - follow safety protocols")
            
        # Cost optimization
        if factors.downtime_cost > 5000:
            recommendations.append("💰 High downtime cost - consider expedited service")
            
        return recommendations
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'weights': {
                'risk': 0.25,
                'urgency': 0.20,
                'impact': 0.20,
                'confidence': 0.15,
                'equipment': 0.10,
                'operational': 0.10
            },
            'thresholds': {
                'critical': 80,
                'high': 60,
                'medium': 40,
                'low': 0
            }
        }
        
    def update_weights_from_feedback(self, feedback: Dict[str, Any]):
        """Update weights based on feedback
        
        Args:
            feedback: Feedback on priority accuracy
        """
        self.feedback_history.append(feedback)
        
        # Optimize weights if enough feedback
        if len(self.feedback_history) >= 50:
            optimized_weights = self.weight_optimizer.optimize(
                list(self.feedback_history)
            )
            self.config['weights'].update(optimized_weights)
            logger.info("Updated priority weights based on feedback")


class RiskCalculator:
    """Calculate risk scores for equipment failures"""
    
    def __init__(self):
        """Initialize Risk Calculator"""
        self.risk_matrix = self._create_risk_matrix()
        
    def calculate_risk(self, factors: PriorityFactors) -> float:
        """Calculate overall risk score
        
        Args:
            factors: Priority factors
            
        Returns:
            Risk score (0-1)
        """
        # Calculate probability of failure
        probability = self._calculate_failure_probability(factors)
        
        # Calculate consequence severity
        consequence = self._calculate_consequence_severity(factors)
        
        # Use risk matrix to get risk score
        risk_score = self._lookup_risk_matrix(probability, consequence)
        
        # Apply modifiers
        if factors.safety_risk:
            risk_score *= 1.5
        if factors.environmental_risk:
            risk_score *= 1.3
        if factors.regulatory_compliance:
            risk_score *= 1.2
            
        return min(1.0, risk_score)
        
    def _calculate_failure_probability(self, factors: PriorityFactors) -> float:
        """Calculate probability of failure"""
        # Base probability from anomaly
        base_prob = factors.anomaly_severity * factors.anomaly_confidence
        
        # Adjust for equipment condition
        condition_factor = 1 - factors.equipment_condition
        
        # Adjust for age (Weibull distribution)
        age_factor = self._weibull_failure_rate(
            factors.equipment_age,
            scale=10,  # Characteristic life in years
            shape=2    # Shape parameter
        )
        
        # Combine factors
        probability = base_prob * (1 + condition_factor) * (1 + age_factor)
        
        return min(1.0, probability / 2)  # Normalize
        
    def _calculate_consequence_severity(self, factors: PriorityFactors) -> float:
        """Calculate consequence severity if failure occurs"""
        severity = 0.0
        
        # Safety consequences
        if factors.safety_risk:
            severity += 0.4
            
        # Environmental consequences
        if factors.environmental_risk:
            severity += 0.3
            
        # Production consequences
        if factors.production_impact > 50:
            severity += 0.3
        elif factors.production_impact > 20:
            severity += 0.2
        elif factors.production_impact > 0:
            severity += 0.1
            
        # Financial consequences
        if factors.downtime_cost > 5000:
            severity += 0.3
        elif factors.downtime_cost > 1000:
            severity += 0.2
        elif factors.downtime_cost > 100:
            severity += 0.1
            
        # Equipment criticality
        criticality_impact = {
            'critical': 0.3,
            'high': 0.2,
            'medium': 0.1,
            'low': 0.05
        }
        severity += criticality_impact.get(factors.equipment_criticality, 0.1)
        
        return min(1.0, severity)
        
    def _weibull_failure_rate(self, age: float, scale: float, shape: float) -> float:
        """Calculate failure rate using Weibull distribution"""
        if age <= 0:
            return 0
        return (shape / scale) * (age / scale) ** (shape - 1)
        
    def _create_risk_matrix(self) -> np.ndarray:
        """Create 5x5 risk matrix"""
        # Risk matrix: rows = probability, cols = consequence
        matrix = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Very Low probability
            [0.2, 0.3, 0.4, 0.5, 0.6],  # Low probability
            [0.3, 0.4, 0.5, 0.6, 0.7],  # Medium probability
            [0.4, 0.5, 0.6, 0.7, 0.8],  # High probability
            [0.5, 0.6, 0.7, 0.8, 0.9]   # Very High probability
        ])
        return matrix
        
    def _lookup_risk_matrix(self, probability: float, consequence: float) -> float:
        """Lookup risk score from risk matrix"""
        # Convert to matrix indices (0-4)
        prob_idx = min(4, int(probability * 5))
        cons_idx = min(4, int(consequence * 5))
        
        return self.risk_matrix[prob_idx, cons_idx]


class FuzzyPriorityEngine:
    """Fuzzy logic engine for priority calculation"""
    
    def __init__(self):
        """Initialize Fuzzy Priority Engine"""
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        """Setup fuzzy logic system"""
        # This is a simplified implementation
        # In production, use python-fuzzy or skfuzzy
        pass
        
    def calculate_priority(self, factors: PriorityFactors) -> float:
        """Calculate priority using fuzzy logic
        
        Args:
            factors: Priority factors
            
        Returns:
            Fuzzy priority score (0-1)
        """
        # Simplified fuzzy logic implementation
        # Define membership functions
        severity_high = self._triangular_membership(
            factors.anomaly_severity, 0.6, 0.8, 1.0
        )
        urgency_high = self._triangular_membership(
            factors.time_since_last, 0, 1, 24
        )
        impact_high = factors.production_impact > 50
        
        # Apply fuzzy rules
        rules = []
        
        # Rule 1: IF severity is high AND urgency is high THEN priority is critical
        rules.append(min(severity_high, urgency_high) * 1.0)
        
        # Rule 2: IF impact is high THEN priority is high
        rules.append((1.0 if impact_high else 0) * 0.8)
        
        # Rule 3: IF equipment is critical THEN priority is high
        rules.append((1.0 if factors.equipment_criticality == 'critical' else 0) * 0.7)
        
        # Defuzzification (max-min composition)
        fuzzy_score = max(rules) if rules else 0.5
        
        return fuzzy_score
        
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)


class MLPriorityPredictor:
    """Machine learning-based priority prediction"""
    
    def __init__(self):
        """Initialize ML Priority Predictor"""
        self.model = None
        self.feature_names = []
        
    def predict_priority(self, 
                        factors: PriorityFactors,
                        history: deque) -> float:
        """Predict priority using ML model
        
        Args:
            factors: Priority factors
            history: Historical priority data
            
        Returns:
            ML-predicted priority score (0-1)
        """
        # Extract features
        features = self._extract_features(factors)
        
        # Simple prediction based on historical patterns
        if len(history) < 10:
            return 0.5  # Default if not enough history
            
        # Find similar historical cases
        similar_cases = self._find_similar_cases(factors, history)
        
        if similar_cases:
            # Average priority of similar cases
            avg_priority = np.mean([
                case['result'].final_score / 100 
                for case in similar_cases
            ])
            return avg_priority
            
        return 0.5
        
    def _extract_features(self, factors: PriorityFactors) -> np.ndarray:
        """Extract features from factors"""
        features = [
            factors.anomaly_severity,
            factors.anomaly_confidence,
            factors.anomaly_frequency,
            factors.time_since_last / 168,  # Normalize to weeks
            1.0 if factors.equipment_criticality == 'critical' else 0.5,
            factors.equipment_age / 10,  # Normalize to decades
            factors.equipment_condition,
            factors.production_impact / 100,
            factors.downtime_cost / 10000,
            1.0 if factors.safety_risk else 0,
            1.0 if factors.environmental_risk else 0
        ]
        return np.array(features)
        
    def _find_similar_cases(self, 
                           factors: PriorityFactors,
                           history: deque,
                           n_similar: int = 5) -> List[Dict]:
        """Find similar historical cases"""
        if not history:
            return []
            
        # Calculate similarity scores
        similarities = []
        current_features = self._extract_features(factors)
        
        for case in history:
            if 'factors' in case:
                case_features = self._extract_features(case['factors'])
                # Cosine similarity
                similarity = np.dot(current_features, case_features) / (
                    np.linalg.norm(current_features) * np.linalg.norm(case_features) + 1e-10
                )
                similarities.append((similarity, case))
                
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [case for _, case in similarities[:n_similar]]


class WeightOptimizer:
    """Optimize weights for priority calculation"""
    
    def __init__(self):
        """Initialize Weight Optimizer"""
        self.optimization_history = []
        
    def optimize(self, feedback_data: List[Dict]) -> Dict[str, float]:
        """Optimize weights based on feedback
        
        Args:
            feedback_data: Historical feedback on priorities
            
        Returns:
            Optimized weights
        """
        # Extract training data from feedback
        X, y = self._prepare_training_data(feedback_data)
        
        if len(X) < 10:
            # Not enough data for optimization
            return {}
            
        # Define objective function
        def objective(weights):
            # Calculate predictions with these weights
            predictions = X @ weights
            # MSE loss
            loss = np.mean((predictions - y) ** 2)
            # Add regularization
            regularization = 0.01 * np.sum(weights ** 2)
            return loss + regularization
            
        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        # Initial weights
        n_features = X.shape[1]
        initial_weights = np.ones(n_features) / n_features
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints
        )
        
        if result.success:
            # Map back to weight dictionary
            weight_names = ['risk', 'urgency', 'impact', 'confidence', 'equipment', 'operational']
            optimized = {}
            for i, name in enumerate(weight_names[:len(result.x)]):
                optimized[name] = float(result.x[i])
            return optimized
            
        return {}
        
    def _prepare_training_data(self, feedback_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from feedback"""
        X = []
        y = []
        
        for feedback in feedback_data:
            if 'component_scores' in feedback and 'actual_priority' in feedback:
                scores = feedback['component_scores']
                features = [
                    scores.get('risk', 0),
                    scores.get('urgency', 0),
                    scores.get('impact', 0),
                    scores.get('confidence', 0),
                    scores.get('equipment', 0),
                    scores.get('operational', 0)
                ]
                X.append(features)
                y.append(feedback['actual_priority'] / 100)  # Normalize
                
        return np.array(X), np.array(y)