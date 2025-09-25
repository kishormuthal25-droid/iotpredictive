"""
Predictive Caching and Adaptive Optimization System
ML-powered caching predictions for 80-sensor dashboard optimization
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import asyncio

# Import existing cache system
from src.utils.advanced_cache import advanced_cache, DashboardCacheHelper

logger = logging.getLogger(__name__)


@dataclass
class CacheAccessPattern:
    """User cache access pattern"""
    cache_key: str
    access_time: datetime
    user_agent: str = ""
    page_context: str = ""
    time_of_day: int = 0  # Hour of day
    day_of_week: int = 0  # Day of week
    session_id: str = ""


@dataclass
class PredictionMetrics:
    """Predictive caching metrics"""
    total_predictions: int = 0
    correct_predictions: int = 0
    cache_hits_from_predictions: int = 0
    preload_operations: int = 0
    accuracy: float = 0.0
    last_model_update: datetime = field(default_factory=datetime.now)


class PredictiveCacheManager:
    """
    ML-powered predictive caching for 80-sensor dashboard
    Learns user patterns to preload likely-to-be-accessed data
    """

    def __init__(self):
        """Initialize predictive cache manager"""
        self.cache_helper = DashboardCacheHelper()
        self.metrics = PredictionMetrics()

        # Access pattern tracking
        self.access_patterns: deque = deque(maxlen=10000)
        self.user_sessions: Dict[str, List[CacheAccessPattern]] = defaultdict(list)

        # ML model for prediction
        self.prediction_model = None
        self.feature_scaler = StandardScaler()
        self.is_model_trained = False

        # Pattern analysis
        self.pattern_lock = threading.RLock()
        self.common_sequences: Dict[str, List[str]] = {}
        self.time_based_patterns: Dict[int, Set[str]] = defaultdict(set)

        # Preloading management
        self.preload_queue = asyncio.Queue(maxsize=100)
        self.preload_active = False
        self.preload_thread = None

        # Adaptive optimization
        self.optimization_rules: Dict[str, Dict[str, Any]] = {}
        self.performance_feedback: deque = deque(maxlen=1000)

        # Configuration
        self.min_pattern_frequency = 3
        self.prediction_confidence_threshold = 0.7
        self.preload_window_minutes = 5

        logger.info("Predictive Cache Manager initialized")

    def record_cache_access(self, cache_key: str, user_agent: str = "",
                          page_context: str = "", session_id: str = ""):
        """Record cache access for pattern learning"""
        now = datetime.now()

        pattern = CacheAccessPattern(
            cache_key=cache_key,
            access_time=now,
            user_agent=user_agent,
            page_context=page_context,
            time_of_day=now.hour,
            day_of_week=now.weekday(),
            session_id=session_id
        )

        with self.pattern_lock:
            self.access_patterns.append(pattern)
            if session_id:
                self.user_sessions[session_id].append(pattern)

        # Trigger pattern analysis periodically
        if len(self.access_patterns) % 100 == 0:
            self._analyze_patterns()

    def predict_next_cache_keys(self, current_context: Dict[str, Any],
                               num_predictions: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next likely cache keys based on current context

        Args:
            current_context: Current user/system context
            num_predictions: Number of predictions to return

        Returns:
            List of (cache_key, confidence) tuples
        """
        try:
            if not self.is_model_trained:
                return self._fallback_predictions(current_context, num_predictions)

            # Extract features from current context
            features = self._extract_prediction_features(current_context)

            if len(features) == 0:
                return self._fallback_predictions(current_context, num_predictions)

            # Get ML predictions
            features_scaled = self.feature_scaler.transform([features])
            prediction_probs = self.prediction_model.predict_proba(features_scaled)[0]

            # Get top predictions
            cache_keys = self.prediction_model.classes_
            predictions = list(zip(cache_keys, prediction_probs))
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Filter by confidence threshold
            confident_predictions = [
                (key, conf) for key, conf in predictions[:num_predictions]
                if conf >= self.prediction_confidence_threshold
            ]

            return confident_predictions

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_predictions(current_context, num_predictions)

    def _fallback_predictions(self, context: Dict[str, Any], num_predictions: int) -> List[Tuple[str, float]]:
        """Fallback predictions based on simple patterns"""
        current_hour = datetime.now().hour
        current_page = context.get('page_context', 'overview')

        # Use time-based patterns
        with self.pattern_lock:
            if current_hour in self.time_based_patterns:
                common_keys = list(self.time_based_patterns[current_hour])[:num_predictions]
                return [(key, 0.6) for key in common_keys]

            # Fallback to most common recent patterns
            if self.access_patterns:
                recent_keys = [p.cache_key for p in list(self.access_patterns)[-50:]]
                key_counts = defaultdict(int)
                for key in recent_keys:
                    key_counts[key] += 1

                sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
                return [(key, 0.5) for key, _ in sorted_keys[:num_predictions]]

        return []

    def preload_predicted_data(self, context: Dict[str, Any]):
        """Preload data based on predictions"""
        if not self.preload_active:
            return

        predictions = self.predict_next_cache_keys(context)

        for cache_key, confidence in predictions:
            if confidence >= self.prediction_confidence_threshold:
                asyncio.create_task(self._preload_cache_key(cache_key))

    async def _preload_cache_key(self, cache_key: str):
        """Preload specific cache key"""
        try:
            # Check if already cached
            cached_data = advanced_cache.get(cache_key)
            if cached_data is not None:
                return

            # Attempt to generate and cache data
            # This would be specific to each cache key pattern
            generated_data = await self._generate_cache_data(cache_key)
            if generated_data:
                advanced_cache.set(cache_key, generated_data, 300)  # 5-minute TTL
                self.metrics.preload_operations += 1

        except Exception as e:
            logger.debug(f"Preload failed for {cache_key}: {e}")

    async def _generate_cache_data(self, cache_key: str) -> Any:
        """Generate data for cache key (placeholder)"""
        # This would be implemented based on cache key patterns
        # For now, return None to indicate no preload possible
        return None

    def _analyze_patterns(self):
        """Analyze access patterns for ML training"""
        with self.pattern_lock:
            if len(self.access_patterns) < 100:
                return

            # Analyze time-based patterns
            self._analyze_time_patterns()

            # Analyze sequential patterns
            self._analyze_sequence_patterns()

            # Train/update ML model
            if len(self.access_patterns) >= 500:
                self._train_prediction_model()

    def _analyze_time_patterns(self):
        """Analyze time-based access patterns"""
        hour_patterns = defaultdict(set)

        for pattern in list(self.access_patterns)[-1000:]:  # Last 1000 accesses
            hour_patterns[pattern.time_of_day].add(pattern.cache_key)

        # Update time-based patterns
        for hour, keys in hour_patterns.items():
            if len(keys) >= self.min_pattern_frequency:
                self.time_based_patterns[hour] = keys

    def _analyze_sequence_patterns(self):
        """Analyze sequential access patterns"""
        sequences = defaultdict(list)

        # Group by session
        for session_id, session_patterns in self.user_sessions.items():
            if len(session_patterns) >= 3:
                cache_keys = [p.cache_key for p in session_patterns[-10:]]  # Last 10 in session

                # Find common sequences
                for i in range(len(cache_keys) - 1):
                    current_key = cache_keys[i]
                    next_key = cache_keys[i + 1]
                    sequences[current_key].append(next_key)

        # Update common sequences
        for key, next_keys in sequences.items():
            if len(next_keys) >= self.min_pattern_frequency:
                key_counts = defaultdict(int)
                for next_key in next_keys:
                    key_counts[next_key] += 1

                # Store most common next keys
                sorted_next = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
                self.common_sequences[key] = [k for k, c in sorted_next[:5]]

    def _train_prediction_model(self):
        """Train ML model for cache key prediction"""
        try:
            # Prepare training data
            features, labels = self._prepare_training_data()

            if len(features) < 50:  # Minimum data requirement
                return

            # Train Random Forest model
            self.prediction_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features)

            # Train model
            self.prediction_model.fit(features_scaled, labels)
            self.is_model_trained = True
            self.metrics.last_model_update = datetime.now()

            logger.info(f"Prediction model trained with {len(features)} samples")

        except Exception as e:
            logger.error(f"Model training failed: {e}")

    def _prepare_training_data(self) -> Tuple[List[List[float]], List[str]]:
        """Prepare training data from access patterns"""
        features = []
        labels = []

        patterns_list = list(self.access_patterns)

        for i in range(len(patterns_list) - 1):
            current = patterns_list[i]
            next_pattern = patterns_list[i + 1]

            # Only use patterns from same session or close in time
            time_diff = (next_pattern.access_time - current.access_time).total_seconds()
            if time_diff > 600:  # 10 minutes max gap
                continue

            # Extract features
            feature_vector = self._extract_pattern_features(current)
            features.append(feature_vector)
            labels.append(next_pattern.cache_key)

        return features, labels

    def _extract_pattern_features(self, pattern: CacheAccessPattern) -> List[float]:
        """Extract features from access pattern"""
        return [
            pattern.time_of_day,
            pattern.day_of_week,
            hash(pattern.page_context) % 1000,  # Simple hash of page context
            hash(pattern.cache_key) % 1000,     # Simple hash of cache key
            len(pattern.cache_key),
            1.0 if 'sensor' in pattern.cache_key else 0.0,
            1.0 if 'chart' in pattern.cache_key else 0.0,
            1.0 if 'overview' in pattern.cache_key else 0.0
        ]

    def _extract_prediction_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features for prediction from current context"""
        now = datetime.now()

        return [
            now.hour,
            now.weekday(),
            hash(context.get('page_context', '')) % 1000,
            hash(context.get('last_cache_key', '')) % 1000,
            len(context.get('last_cache_key', '')),
            1.0 if 'sensor' in str(context) else 0.0,
            1.0 if 'chart' in str(context) else 0.0,
            1.0 if 'overview' in str(context) else 0.0
        ]

    def start_adaptive_optimization(self):
        """Start adaptive optimization based on performance feedback"""
        self.preload_active = True
        logger.info("Adaptive optimization started")

    def stop_adaptive_optimization(self):
        """Stop adaptive optimization"""
        self.preload_active = False
        logger.info("Adaptive optimization stopped")

    def record_performance_feedback(self, cache_key: str, response_time: float,
                                  cache_hit: bool, user_satisfaction: float = 0.5):
        """Record performance feedback for adaptive optimization"""
        feedback = {
            'cache_key': cache_key,
            'response_time': response_time,
            'cache_hit': cache_hit,
            'user_satisfaction': user_satisfaction,
            'timestamp': datetime.now()
        }

        self.performance_feedback.append(feedback)

        # Update optimization rules
        self._update_optimization_rules(feedback)

    def _update_optimization_rules(self, feedback: Dict[str, Any]):
        """Update optimization rules based on feedback"""
        cache_key = feedback['cache_key']

        if cache_key not in self.optimization_rules:
            self.optimization_rules[cache_key] = {
                'avg_response_time': feedback['response_time'],
                'cache_hit_rate': 1.0 if feedback['cache_hit'] else 0.0,
                'access_count': 1,
                'priority': 'medium'
            }
        else:
            rules = self.optimization_rules[cache_key]
            rules['access_count'] += 1

            # Update average response time
            alpha = 0.1
            rules['avg_response_time'] = (
                alpha * feedback['response_time'] +
                (1 - alpha) * rules['avg_response_time']
            )

            # Update cache hit rate
            rules['cache_hit_rate'] = (
                alpha * (1.0 if feedback['cache_hit'] else 0.0) +
                (1 - alpha) * rules['cache_hit_rate']
            )

            # Adjust priority based on performance
            if rules['avg_response_time'] > 500:  # > 500ms
                rules['priority'] = 'high'
            elif rules['cache_hit_rate'] < 0.5:
                rules['priority'] = 'high'
            elif rules['access_count'] > 100:
                rules['priority'] = 'high'
            else:
                rules['priority'] = 'medium'

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on learned patterns"""
        recommendations = []

        # Analyze high-priority cache keys
        high_priority_keys = [
            key for key, rules in self.optimization_rules.items()
            if rules['priority'] == 'high'
        ]

        for cache_key in high_priority_keys:
            rules = self.optimization_rules[cache_key]
            recommendations.append({
                'cache_key': cache_key,
                'issue': 'Poor performance detected',
                'avg_response_time': rules['avg_response_time'],
                'cache_hit_rate': rules['cache_hit_rate'],
                'recommendation': 'Increase cache TTL and implement preloading',
                'priority': rules['priority']
            })

        # Recommend based on time patterns
        current_hour = datetime.now().hour
        if current_hour in self.time_based_patterns:
            common_keys = self.time_based_patterns[current_hour]
            recommendations.append({
                'type': 'time_based',
                'recommendation': f'Preload {len(common_keys)} commonly accessed keys for hour {current_hour}',
                'cache_keys': list(common_keys)[:10],
                'priority': 'medium'
            })

        return recommendations

    def get_predictive_metrics(self) -> Dict[str, Any]:
        """Get predictive caching metrics"""
        return {
            'total_patterns': len(self.access_patterns),
            'trained_model': self.is_model_trained,
            'prediction_accuracy': self.metrics.accuracy,
            'preload_operations': self.metrics.preload_operations,
            'cache_hits_from_predictions': self.metrics.cache_hits_from_predictions,
            'time_patterns': len(self.time_based_patterns),
            'sequence_patterns': len(self.common_sequences),
            'optimization_rules': len(self.optimization_rules),
            'last_model_update': self.metrics.last_model_update.isoformat() if self.metrics.last_model_update else None
        }

    def export_model(self, filename: str = None):
        """Export trained model for persistence"""
        if not self.is_model_trained:
            logger.warning("No trained model to export")
            return

        if not filename:
            filename = f"predictive_cache_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"

        model_data = {
            'prediction_model': self.prediction_model,
            'feature_scaler': self.feature_scaler,
            'time_patterns': dict(self.time_based_patterns),
            'sequence_patterns': self.common_sequences,
            'optimization_rules': self.optimization_rules
        }

        joblib.dump(model_data, filename)
        logger.info(f"Predictive cache model exported to {filename}")

    def load_model(self, filename: str):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filename)

            self.prediction_model = model_data['prediction_model']
            self.feature_scaler = model_data['feature_scaler']
            self.time_based_patterns = defaultdict(set, model_data['time_patterns'])
            self.common_sequences = model_data['sequence_patterns']
            self.optimization_rules = model_data['optimization_rules']

            self.is_model_trained = True
            logger.info(f"Predictive cache model loaded from {filename}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")


# Global predictive cache manager
predictive_cache = PredictiveCacheManager()


# Convenience functions
def record_cache_access(cache_key: str, context: Dict[str, Any] = None):
    """Record cache access for pattern learning"""
    context = context or {}
    predictive_cache.record_cache_access(
        cache_key,
        user_agent=context.get('user_agent', ''),
        page_context=context.get('page_context', ''),
        session_id=context.get('session_id', '')
    )


def get_cache_predictions(context: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Get cache predictions for current context"""
    return predictive_cache.predict_next_cache_keys(context)


def start_predictive_optimization():
    """Start predictive optimization"""
    predictive_cache.start_adaptive_optimization()