"""
Phase 2 API Extensions
API endpoints for advanced analytics, alert management, and threshold configuration
"""

from flask import Flask, request, jsonify, Blueprint
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import uuid

# Import Phase 2 components
from src.dashboard.components.subsystem_failure_analyzer import nasa_subsystem_analyzer
from src.dashboard.components.detection_details_panel import detection_details_panel
from src.dashboard.components.alert_action_manager import alert_action_manager
from src.dashboard.components.threshold_manager import threshold_manager
from src.database.phase2_enhancements import phase2_db_manager

logger = logging.getLogger(__name__)

# Create Blueprint for Phase 2 API endpoints
phase2_api = Blueprint('phase2_api', __name__, url_prefix='/api/v2')


# =============================================================================
# Subsystem Analysis Endpoints
# =============================================================================

@phase2_api.route('/subsystem/analysis/<subsystem>', methods=['GET'])
def get_subsystem_analysis(subsystem: str):
    """Get comprehensive analysis for a specific subsystem"""
    try:
        subsystem = subsystem.upper()

        if subsystem == 'POWER':
            analysis = nasa_subsystem_analyzer.analyze_power_system_patterns()
        elif subsystem == 'MOBILITY':
            analysis = nasa_subsystem_analyzer.analyze_mobility_system_patterns()
        elif subsystem == 'COMMUNICATION':
            analysis = nasa_subsystem_analyzer.analyze_communication_system_patterns()
        else:
            return jsonify({'error': f'Unsupported subsystem: {subsystem}'}), 400

        return jsonify({
            'success': True,
            'subsystem': subsystem,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting subsystem analysis for {subsystem}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/subsystem/health-summary', methods=['GET'])
def get_subsystem_health_summary():
    """Get health summary for all subsystems"""
    try:
        summary = nasa_subsystem_analyzer.get_subsystem_health_summary()

        return jsonify({
            'success': True,
            'health_summary': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting subsystem health summary: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/subsystem/failure-patterns/<subsystem>', methods=['GET'])
def get_failure_patterns(subsystem: str):
    """Get failure pattern visualization data for subsystem"""
    try:
        subsystem = subsystem.upper()

        # Get visualization data (would normally return plot data)
        figure = nasa_subsystem_analyzer.create_failure_pattern_visualization(subsystem)

        # Convert plotly figure to JSON for API response
        figure_json = figure.to_dict() if figure else {}

        return jsonify({
            'success': True,
            'subsystem': subsystem,
            'visualization_data': figure_json,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting failure patterns for {subsystem}: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Detection Details Endpoints
# =============================================================================

@phase2_api.route('/detection/details/<equipment_id>', methods=['GET'])
def get_detection_details(equipment_id: str):
    """Get detailed detection analysis for specific equipment"""
    try:
        # Get detection summary
        summary = detection_details_panel.analyze_equipment_detection(equipment_id)

        # Convert to serializable format
        summary_dict = {
            'equipment_id': summary.equipment_id,
            'equipment_name': summary.equipment_name,
            'equipment_type': summary.equipment_type,
            'subsystem': summary.subsystem,
            'overall_anomaly_score': summary.overall_anomaly_score,
            'overall_confidence': summary.overall_confidence,
            'is_anomaly': summary.is_anomaly,
            'severity_level': summary.severity_level,
            'sensor_count': summary.sensor_count,
            'anomalous_sensor_count': summary.anomalous_sensor_count,
            'reconstruction_quality': summary.reconstruction_quality,
            'recommendations': summary.recommendations,
            'sensor_details': [
                {
                    'sensor_name': sensor.sensor_name,
                    'current_value': sensor.current_value,
                    'expected_value': sensor.expected_value,
                    'anomaly_score': sensor.anomaly_score,
                    'confidence': sensor.confidence,
                    'severity_level': sensor.severity_level,
                    'threshold_exceeded': sensor.threshold_exceeded,
                    'z_score': sensor.z_score,
                    'trend_direction': sensor.trend_direction
                }
                for sensor in summary.sensor_details
            ]
        }

        return jsonify({
            'success': True,
            'detection_summary': summary_dict,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting detection details for {equipment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/detection/sensor-breakdown/<equipment_id>/<sensor_name>', methods=['GET'])
def get_sensor_breakdown(equipment_id: str, sensor_name: str):
    """Get detailed breakdown for a specific sensor"""
    try:
        # This would get detailed sensor analysis
        # For now, return mock data structure
        breakdown = {
            'sensor_name': sensor_name,
            'equipment_id': equipment_id,
            'current_value': 25.6,
            'expected_value': 28.5,
            'anomaly_score': 0.82,
            'confidence': 0.91,
            'historical_data': [],  # Would contain historical values
            'threshold_levels': {
                'critical': 0.90,
                'high': 0.75,
                'medium': 0.60,
                'warning': 0.45
            },
            'sensor_metadata': {
                'unit': 'V',
                'sensor_type': 'voltage',
                'calibration_date': '2024-01-15',
                'expected_range': {'min': 25.0, 'max': 35.0}
            }
        }

        return jsonify({
            'success': True,
            'sensor_breakdown': breakdown,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting sensor breakdown for {equipment_id}/{sensor_name}: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Alert Management Endpoints
# =============================================================================

@phase2_api.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'unknown')
        notes = data.get('notes', '')

        result = alert_action_manager.acknowledge_alert(alert_id, user_id, notes)

        # Record in database
        if result.success:
            from src.database.phase2_enhancements import AlertActionRecord
            action_record = AlertActionRecord(
                action_id=result.action_id,
                alert_id=alert_id,
                action_type='acknowledge',
                user_id=user_id,
                timestamp=datetime.now(),
                notes=notes
            )
            phase2_db_manager.record_alert_action(action_record)

        return jsonify({
            'success': result.success,
            'message': result.message,
            'action_id': result.action_id,
            'new_status': result.new_alert_status
        })

    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/alerts/<alert_id>/dismiss', methods=['POST'])
def dismiss_alert(alert_id: str):
    """Dismiss an alert"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'unknown')
        reason = data.get('reason', 'No reason provided')
        notes = data.get('notes', '')

        result = alert_action_manager.dismiss_alert(alert_id, user_id, reason, notes)

        # Record in database
        if result.success:
            from src.database.phase2_enhancements import AlertActionRecord
            action_record = AlertActionRecord(
                action_id=result.action_id,
                alert_id=alert_id,
                action_type='dismiss',
                user_id=user_id,
                timestamp=datetime.now(),
                reason=reason,
                notes=notes
            )
            phase2_db_manager.record_alert_action(action_record)

        return jsonify({
            'success': result.success,
            'message': result.message,
            'action_id': result.action_id,
            'new_status': result.new_alert_status
        })

    except Exception as e:
        logger.error(f"Error dismissing alert {alert_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/alerts/<alert_id>/create-work-order', methods=['POST'])
def create_work_order_from_alert(alert_id: str):
    """Create work order from alert"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'unknown')
        work_order_params = data.get('work_order_params', {})

        result = alert_action_manager.create_work_order_from_alert(
            alert_id, user_id, work_order_params
        )

        # Record in database
        if result.success:
            from src.database.phase2_enhancements import AlertActionRecord
            action_record = AlertActionRecord(
                action_id=result.action_id,
                alert_id=alert_id,
                action_type='create_work_order',
                user_id=user_id,
                timestamp=datetime.now(),
                work_order_id=result.work_order_id,
                metadata=work_order_params
            )
            phase2_db_manager.record_alert_action(action_record)

        return jsonify({
            'success': result.success,
            'message': result.message,
            'action_id': result.action_id,
            'work_order_id': result.work_order_id,
            'new_status': result.new_alert_status
        })

    except Exception as e:
        logger.error(f"Error creating work order from alert {alert_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/alerts/<alert_id>/escalate', methods=['POST'])
def escalate_alert(alert_id: str):
    """Escalate an alert"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'unknown')
        reason = data.get('reason', 'Manual escalation')
        notes = data.get('notes', '')

        result = alert_action_manager.escalate_alert(alert_id, user_id, reason, notes)

        # Record in database
        if result.success:
            from src.database.phase2_enhancements import AlertActionRecord
            action_record = AlertActionRecord(
                action_id=result.action_id,
                alert_id=alert_id,
                action_type='escalate',
                user_id=user_id,
                timestamp=datetime.now(),
                reason=reason,
                notes=notes
            )
            phase2_db_manager.record_alert_action(action_record)

        return jsonify({
            'success': result.success,
            'message': result.message,
            'action_id': result.action_id,
            'new_status': result.new_alert_status
        })

    except Exception as e:
        logger.error(f"Error escalating alert {alert_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/alerts/<alert_id>/actions', methods=['GET'])
def get_alert_actions(alert_id: str):
    """Get action history for an alert"""
    try:
        actions = phase2_db_manager.get_alert_actions(alert_id)

        actions_data = [
            {
                'action_id': action.action_id,
                'action_type': action.action_type,
                'user_id': action.user_id,
                'timestamp': action.timestamp.isoformat(),
                'reason': action.reason,
                'notes': action.notes,
                'work_order_id': action.work_order_id,
                'escalation_level': action.escalation_level
            }
            for action in actions
        ]

        return jsonify({
            'success': True,
            'alert_id': alert_id,
            'actions': actions_data,
            'total_actions': len(actions_data)
        })

    except Exception as e:
        logger.error(f"Error getting actions for alert {alert_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """Get alert action statistics"""
    try:
        days = request.args.get('days', 30, type=int)
        stats = phase2_db_manager.get_alert_action_statistics(days)

        return jsonify({
            'success': True,
            'statistics': stats,
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Threshold Management Endpoints
# =============================================================================

@phase2_api.route('/thresholds/<equipment_id>', methods=['GET'])
def get_equipment_thresholds(equipment_id: str):
    """Get current thresholds for equipment"""
    try:
        config = threshold_manager.threshold_configs.get(equipment_id)

        if not config:
            return jsonify({'error': f'No threshold configuration found for {equipment_id}'}), 404

        threshold_data = {
            'equipment_id': config.equipment_id,
            'equipment_type': config.equipment_type,
            'subsystem': config.subsystem,
            'criticality': config.criticality,
            'thresholds': {
                'critical': config.critical_threshold,
                'high': config.high_threshold,
                'medium': config.medium_threshold,
                'warning': config.warning_threshold
            },
            'adaptation_rate': config.adaptation_rate,
            'sensitivity_multiplier': config.sensitivity_multiplier,
            'last_updated': config.last_updated.isoformat(),
            'update_count': config.update_count,
            'performance_metrics': {
                'false_positive_rate': config.false_positive_rate,
                'false_negative_rate': config.false_negative_rate,
                'accuracy_score': config.accuracy_score
            }
        }

        return jsonify({
            'success': True,
            'threshold_configuration': threshold_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting thresholds for {equipment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/thresholds/<equipment_id>/optimize', methods=['POST'])
def optimize_equipment_thresholds(equipment_id: str):
    """Optimize thresholds for equipment"""
    try:
        data = request.get_json() or {}
        optimization_type = data.get('optimization_type', 'accuracy')
        user_id = data.get('user_id', 'system')

        result = threshold_manager.optimize_thresholds_for_equipment(
            equipment_id, optimization_type
        )

        # Record optimization result in database
        if result.improvement_score > 0:
            from src.database.phase2_enhancements import ThresholdHistoryRecord
            history_record = ThresholdHistoryRecord(
                change_id=str(uuid.uuid4()),
                equipment_id=equipment_id,
                equipment_type=result.optimization_type,  # Temporary use
                subsystem='',  # Would get from equipment info
                criticality='',  # Would get from equipment info
                old_thresholds=result.old_thresholds,
                new_thresholds=result.new_thresholds,
                optimization_type=optimization_type,
                improvement_score=result.improvement_score,
                confidence=result.confidence,
                justification=result.justification,
                user_id=user_id,
                timestamp=datetime.now(),
                applied=False
            )
            phase2_db_manager.record_threshold_change(history_record)

        return jsonify({
            'success': True,
            'optimization_result': {
                'equipment_id': result.equipment_id,
                'optimization_type': result.optimization_type,
                'old_thresholds': result.old_thresholds,
                'new_thresholds': result.new_thresholds,
                'improvement_score': result.improvement_score,
                'confidence': result.confidence,
                'justification': result.justification
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error optimizing thresholds for {equipment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/thresholds/<equipment_id>/apply', methods=['POST'])
def apply_threshold_optimization(equipment_id: str):
    """Apply optimized thresholds"""
    try:
        data = request.get_json() or {}
        new_thresholds = data.get('new_thresholds', {})
        user_id = data.get('user_id', 'unknown')

        # Create optimization result object for application
        from src.dashboard.components.threshold_manager import ThresholdOptimizationResult
        optimization_result = ThresholdOptimizationResult(
            equipment_id=equipment_id,
            optimization_type='manual',
            old_thresholds={},  # Would get current thresholds
            new_thresholds=new_thresholds,
            improvement_score=0.0,
            confidence=1.0,
            justification='Manual threshold adjustment'
        )

        success = threshold_manager.apply_threshold_optimization(
            equipment_id, optimization_result
        )

        return jsonify({
            'success': success,
            'message': 'Thresholds applied successfully' if success else 'Failed to apply thresholds',
            'equipment_id': equipment_id,
            'applied_thresholds': new_thresholds,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error applying thresholds for {equipment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/thresholds/recommendations', methods=['GET'])
def get_threshold_recommendations():
    """Get threshold optimization recommendations"""
    try:
        # Get filter parameters
        spacecraft = request.args.get('spacecraft')
        subsystem = request.args.get('subsystem')
        criticality = request.args.get('criticality')

        equipment_filter = {}
        if spacecraft:
            equipment_filter['spacecraft'] = spacecraft
        if subsystem:
            equipment_filter['subsystem'] = subsystem
        if criticality:
            equipment_filter['criticality'] = criticality

        recommendations = threshold_manager.get_threshold_recommendations(equipment_filter)

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'filter_applied': equipment_filter,
            'total_recommendations': len(recommendations),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting threshold recommendations: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/thresholds/<equipment_id>/history', methods=['GET'])
def get_threshold_history(equipment_id: str):
    """Get threshold change history for equipment"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = phase2_db_manager.get_threshold_history(equipment_id, limit)

        history_data = [
            {
                'change_id': record.change_id,
                'optimization_type': record.optimization_type,
                'old_thresholds': record.old_thresholds,
                'new_thresholds': record.new_thresholds,
                'improvement_score': record.improvement_score,
                'confidence': record.confidence,
                'justification': record.justification,
                'user_id': record.user_id,
                'timestamp': record.timestamp.isoformat(),
                'applied': record.applied
            }
            for record in history
        ]

        return jsonify({
            'success': True,
            'equipment_id': equipment_id,
            'threshold_history': history_data,
            'total_changes': len(history_data),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting threshold history for {equipment_id}: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# System Health and Statistics Endpoints
# =============================================================================

@phase2_api.route('/system/health', methods=['GET'])
def get_system_health():
    """Get overall system health for Phase 2 components"""
    try:
        # Get database statistics
        db_stats = phase2_db_manager.get_database_statistics()

        # Get subsystem health summary
        subsystem_health = nasa_subsystem_analyzer.get_subsystem_health_summary()

        # Compile system health
        system_health = {
            'database_health': {
                'status': 'healthy',
                'statistics': db_stats
            },
            'subsystem_health': subsystem_health,
            'component_status': {
                'subsystem_analyzer': 'operational',
                'detection_panel': 'operational',
                'alert_manager': 'operational',
                'threshold_manager': 'operational'
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'success': True,
            'system_health': system_health
        })

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return jsonify({'error': str(e)}), 500


@phase2_api.route('/system/statistics', methods=['GET'])
def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        days = request.args.get('days', 7, type=int)

        # Get various statistics
        alert_stats = phase2_db_manager.get_alert_action_statistics(days)
        db_stats = phase2_db_manager.get_database_statistics()

        statistics = {
            'alert_statistics': alert_stats,
            'database_statistics': db_stats,
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'success': True,
            'statistics': statistics
        })

    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Utility Functions
# =============================================================================

def register_phase2_api(app: Flask):
    """Register Phase 2 API blueprint with Flask app"""
    app.register_blueprint(phase2_api)
    logger.info("Phase 2 API endpoints registered")


# Error handlers for the blueprint
@phase2_api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@phase2_api.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500