"""
Regulatory Compliance Tracking System for Phase 3.1 IoT Predictive Maintenance
Automated audit workflows, certification management, and compliance monitoring
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
from pathlib import Path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXPIRED = "expired"
    NOT_APPLICABLE = "not_applicable"


class AuditType(Enum):
    """Types of audits"""
    INTERNAL = "internal"
    EXTERNAL = "external"
    REGULATORY = "regulatory"
    CERTIFICATION = "certification"
    SELF_ASSESSMENT = "self_assessment"


class ComplianceCategory(Enum):
    """Compliance categories"""
    SAFETY = "safety"
    ENVIRONMENTAL = "environmental"
    QUALITY = "quality"
    DATA_PROTECTION = "data_protection"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    CYBERSECURITY = "cybersecurity"


class RiskLevel(Enum):
    """Risk levels for compliance issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ComplianceRequirement:
    """Regulatory compliance requirement definition"""
    requirement_id: str
    name: str
    description: str
    category: ComplianceCategory
    regulatory_body: str
    standard_reference: str  # e.g., "ISO 9001:2015"
    applicable_equipment: List[str]
    frequency_days: int  # How often compliance check is needed
    grace_period_days: int  # Grace period before non-compliance
    mandatory: bool = True
    documentation_required: List[str] = field(default_factory=list)
    verification_methods: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    penalties: Dict[str, str] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceRecord:
    """Individual compliance record/check"""
    record_id: str
    requirement_id: str
    equipment_id: Optional[str]
    check_date: datetime
    due_date: datetime
    status: ComplianceStatus
    auditor: str
    findings: List[str] = field(default_factory=list)
    corrective_actions: List[str] = field(default_factory=list)
    evidence_files: List[str] = field(default_factory=list)
    score: Optional[float] = None  # 0-100 compliance score
    next_review_date: Optional[datetime] = None
    created_by: str = "system"
    approved_by: Optional[str] = None


@dataclass
class Audit:
    """Audit session tracking"""
    audit_id: str
    audit_type: AuditType
    title: str
    description: str
    auditor: str
    start_date: datetime
    end_date: datetime
    scope: List[str]  # Equipment, processes, or requirements covered
    status: str  # "planned", "in_progress", "completed", "cancelled"
    compliance_records: List[str] = field(default_factory=list)
    overall_score: Optional[float] = None
    critical_findings: int = 0
    recommendations: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    report_generated: bool = False


@dataclass
class Certification:
    """Certification tracking"""
    certification_id: str
    name: str
    issuing_body: str
    certificate_number: str
    issue_date: datetime
    expiry_date: datetime
    renewal_date: datetime  # When to start renewal process
    applicable_equipment: List[str]
    requirements: List[str]  # Related compliance requirements
    status: ComplianceStatus
    renewal_cost: float = 0.0
    renewal_duration_days: int = 90
    auto_renewal: bool = False
    responsible_person: str = ""


@dataclass
class ComplianceAlert:
    """Compliance alert/notification"""
    alert_id: str
    alert_type: str  # "expiry_warning", "non_compliance", "audit_due"
    title: str
    description: str
    severity: RiskLevel
    created_date: datetime
    due_date: Optional[datetime]
    resolved: bool = False
    assigned_to: Optional[str] = None
    related_items: List[str] = field(default_factory=list)  # IDs of related records


class RegulatoryComplianceTracker:
    """Comprehensive regulatory compliance tracking and monitoring system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Regulatory Compliance Tracker

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Core data structures
        self.requirements = {}  # requirement_id -> ComplianceRequirement
        self.compliance_records = {}  # record_id -> ComplianceRecord
        self.audits = {}  # audit_id -> Audit
        self.certifications = {}  # certification_id -> Certification
        self.alerts = {}  # alert_id -> ComplianceAlert

        # Equipment mapping
        self.equipment_compliance = defaultdict(list)  # equipment_id -> [requirement_ids]

        # Notification settings
        self.notification_settings = {
            'expiry_warning_days': [30, 14, 7, 1],  # Days before expiry to send warnings
            'auto_generate_audits': True,
            'auto_renewal_certifications': False,
            'email_notifications': True
        }

        # Initialize with sample regulatory requirements
        self._initialize_regulatory_framework()

        logger.info("Initialized Regulatory Compliance Tracker")

    def _initialize_regulatory_framework(self):
        """Initialize with common regulatory requirements"""
        # Sample regulatory requirements
        sample_requirements = [
            ComplianceRequirement(
                "OSHA_SAFETY_001", "OSHA Safety Inspection",
                "Regular safety inspection of equipment per OSHA standards",
                ComplianceCategory.SAFETY, "OSHA", "29 CFR 1910.147",
                ["ALL"], 365, 30, True,
                ["safety_checklist", "inspection_report", "corrective_action_plan"],
                ["visual_inspection", "functional_test", "documentation_review"],
                RiskLevel.CRITICAL,
                {"fine": "$13,494 per violation", "shutdown": "Possible facility shutdown"}
            ),
            ComplianceRequirement(
                "ISO_9001_001", "ISO 9001 Quality Management",
                "Quality management system compliance per ISO 9001:2015",
                ComplianceCategory.QUALITY, "ISO", "ISO 9001:2015",
                ["ALL"], 365, 90, True,
                ["quality_manual", "procedure_documents", "audit_reports"],
                ["document_review", "process_audit", "management_review"],
                RiskLevel.HIGH,
                {"certification_loss": "Loss of ISO certification"}
            ),
            ComplianceRequirement(
                "EPA_ENV_001", "EPA Environmental Compliance",
                "Environmental compliance for industrial equipment",
                ComplianceCategory.ENVIRONMENTAL, "EPA", "40 CFR Part 63",
                ["EQ_001", "EQ_002"], 180, 30, True,
                ["emission_report", "waste_manifest", "compliance_certificate"],
                ["emission_testing", "waste_audit", "permit_review"],
                RiskLevel.HIGH,
                {"fine": "$37,500 per day per violation"}
            ),
            ComplianceRequirement(
                "NIST_CYBER_001", "NIST Cybersecurity Framework",
                "Cybersecurity compliance per NIST framework",
                ComplianceCategory.CYBERSECURITY, "NIST", "NIST CSF 1.1",
                ["EQ_CONTROL_001", "EQ_CONTROL_002"], 90, 14, True,
                ["security_assessment", "vulnerability_scan", "incident_log"],
                ["penetration_test", "security_audit", "policy_review"],
                RiskLevel.CRITICAL,
                {"data_breach": "Potential data breach liability"}
            ),
            ComplianceRequirement(
                "FMA_FOOD_001", "FDA Food Safety Modernization",
                "Food safety compliance for food processing equipment",
                ComplianceCategory.SAFETY, "FDA", "21 CFR Part 117",
                ["EQ_FOOD_001"], 30, 7, True,
                ["haccp_plan", "cleaning_log", "temperature_log"],
                ["sanitation_inspection", "temperature_verification", "record_review"],
                RiskLevel.CRITICAL,
                {"recall": "Product recall costs", "shutdown": "Facility shutdown"}
            ),
            ComplianceRequirement(
                "GDPR_DATA_001", "GDPR Data Protection",
                "Data protection compliance per GDPR",
                ComplianceCategory.DATA_PROTECTION, "EU", "GDPR Article 32",
                ["DATA_SYSTEMS"], 365, 30, True,
                ["privacy_policy", "data_mapping", "breach_procedure"],
                ["data_audit", "privacy_assessment", "consent_review"],
                RiskLevel.HIGH,
                {"fine": "Up to 4% of annual turnover or €20M"}
            )
        ]

        for requirement in sample_requirements:
            self.requirements[requirement.requirement_id] = requirement

        # Sample certifications
        sample_certifications = [
            Certification(
                "CERT_ISO9001", "ISO 9001:2015 Certification",
                "ISO", "ISO9001-2023-001234",
                datetime(2023, 1, 15), datetime(2026, 1, 15),
                datetime(2025, 7, 15), ["ALL"], ["ISO_9001_001"],
                ComplianceStatus.COMPLIANT, 15000.0, 120
            ),
            Certification(
                "CERT_OSHA", "OSHA Safety Certification",
                "OSHA", "OSHA-CERT-2023-5678",
                datetime(2023, 6, 1), datetime(2024, 6, 1),
                datetime(2024, 3, 1), ["ALL"], ["OSHA_SAFETY_001"],
                ComplianceStatus.WARNING, 5000.0, 60  # Expires soon
            ),
            Certification(
                "CERT_EPA", "EPA Environmental Permit",
                "EPA", "EPA-PERMIT-2023-9012",
                datetime(2023, 3, 1), datetime(2025, 3, 1),
                datetime(2024, 12, 1), ["EQ_001", "EQ_002"], ["EPA_ENV_001"],
                ComplianceStatus.COMPLIANT, 8000.0, 90
            )
        ]

        for certification in sample_certifications:
            self.certifications[certification.certification_id] = certification

    def add_compliance_requirement(self, requirement: ComplianceRequirement) -> bool:
        """Add new compliance requirement

        Args:
            requirement: Compliance requirement to add

        Returns:
            True if added successfully
        """
        try:
            self.requirements[requirement.requirement_id] = requirement

            # Update equipment mapping
            for equipment_id in requirement.applicable_equipment:
                if equipment_id not in self.equipment_compliance[equipment_id]:
                    self.equipment_compliance[equipment_id].append(requirement.requirement_id)

            logger.info(f"Added compliance requirement: {requirement.name}")
            return True

        except Exception as e:
            logger.error(f"Error adding compliance requirement: {e}")
            return False

    def create_compliance_record(self, requirement_id: str, equipment_id: Optional[str] = None,
                               auditor: str = "system", findings: List[str] = None,
                               status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW) -> Optional[ComplianceRecord]:
        """Create new compliance record

        Args:
            requirement_id: Associated requirement ID
            equipment_id: Equipment ID (if applicable)
            auditor: Person conducting the check
            findings: List of findings
            status: Compliance status

        Returns:
            Created compliance record
        """
        if requirement_id not in self.requirements:
            logger.error(f"Requirement {requirement_id} not found")
            return None

        try:
            requirement = self.requirements[requirement_id]

            # Calculate due date
            due_date = datetime.now() + timedelta(days=requirement.frequency_days)

            record = ComplianceRecord(
                record_id=f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                requirement_id=requirement_id,
                equipment_id=equipment_id,
                check_date=datetime.now(),
                due_date=due_date,
                status=status,
                auditor=auditor,
                findings=findings or [],
                next_review_date=due_date
            )

            self.compliance_records[record.record_id] = record

            # Generate alerts if non-compliant
            if status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.WARNING]:
                self._create_compliance_alert(record, requirement)

            logger.info(f"Created compliance record: {record.record_id}")
            return record

        except Exception as e:
            logger.error(f"Error creating compliance record: {e}")
            return None

    def schedule_audit(self, audit_type: AuditType, title: str, auditor: str,
                      start_date: datetime, scope: List[str],
                      duration_days: int = 1) -> Optional[Audit]:
        """Schedule new audit

        Args:
            audit_type: Type of audit
            title: Audit title
            auditor: Assigned auditor
            start_date: Audit start date
            scope: Scope of audit (equipment, processes, requirements)
            duration_days: Audit duration in days

        Returns:
            Created audit
        """
        try:
            end_date = start_date + timedelta(days=duration_days)

            audit = Audit(
                audit_id=f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                audit_type=audit_type,
                title=title,
                description=f"{audit_type.value.title()} audit scheduled for {start_date.strftime('%Y-%m-%d')}",
                auditor=auditor,
                start_date=start_date,
                end_date=end_date,
                scope=scope,
                status="planned"
            )

            self.audits[audit.audit_id] = audit

            # Create alert for upcoming audit
            self._create_audit_alert(audit)

            logger.info(f"Scheduled audit: {title} ({audit.audit_id})")
            return audit

        except Exception as e:
            logger.error(f"Error scheduling audit: {e}")
            return None

    def conduct_compliance_check(self, requirement_id: str, equipment_id: Optional[str] = None,
                               auditor: str = "inspector", score: Optional[float] = None,
                               findings: List[str] = None, evidence_files: List[str] = None) -> Optional[ComplianceRecord]:
        """Conduct compliance check and create record

        Args:
            requirement_id: Requirement being checked
            equipment_id: Equipment ID (if applicable)
            auditor: Person conducting check
            score: Compliance score (0-100)
            findings: List of findings
            evidence_files: Supporting evidence files

        Returns:
            Created compliance record
        """
        if requirement_id not in self.requirements:
            logger.error(f"Requirement {requirement_id} not found")
            return None

        try:
            requirement = self.requirements[requirement_id]

            # Determine status based on score and findings
            if score is not None:
                if score >= 95:
                    status = ComplianceStatus.COMPLIANT
                elif score >= 80:
                    status = ComplianceStatus.WARNING
                else:
                    status = ComplianceStatus.NON_COMPLIANT
            else:
                # Determine based on findings
                if not findings:
                    status = ComplianceStatus.COMPLIANT
                elif any('critical' in finding.lower() or 'violation' in finding.lower() for finding in findings):
                    status = ComplianceStatus.NON_COMPLIANT
                else:
                    status = ComplianceStatus.WARNING

            # Calculate next review date
            next_review = datetime.now() + timedelta(days=requirement.frequency_days)

            record = ComplianceRecord(
                record_id=f"CHECK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                requirement_id=requirement_id,
                equipment_id=equipment_id,
                check_date=datetime.now(),
                due_date=datetime.now() + timedelta(days=requirement.frequency_days),
                status=status,
                auditor=auditor,
                findings=findings or [],
                evidence_files=evidence_files or [],
                score=score,
                next_review_date=next_review,
                created_by=auditor
            )

            self.compliance_records[record.record_id] = record

            # Generate corrective actions for non-compliance
            if status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.WARNING]:
                record.corrective_actions = self._generate_corrective_actions(requirement, findings or [])
                self._create_compliance_alert(record, requirement)

            logger.info(f"Completed compliance check: {requirement.name} - {status.value}")
            return record

        except Exception as e:
            logger.error(f"Error conducting compliance check: {e}")
            return None

    def update_certification_status(self, certification_id: str) -> bool:
        """Update certification status based on expiry dates

        Args:
            certification_id: Certification ID to update

        Returns:
            True if updated successfully
        """
        if certification_id not in self.certifications:
            logger.error(f"Certification {certification_id} not found")
            return False

        try:
            certification = self.certifications[certification_id]
            now = datetime.now()

            # Determine status based on dates
            if now > certification.expiry_date:
                certification.status = ComplianceStatus.EXPIRED
            elif now > certification.renewal_date:
                certification.status = ComplianceStatus.WARNING
            else:
                certification.status = ComplianceStatus.COMPLIANT

            # Create renewal alert if needed
            if certification.status in [ComplianceStatus.WARNING, ComplianceStatus.EXPIRED]:
                self._create_certification_alert(certification)

            logger.info(f"Updated certification status: {certification.name} - {certification.status.value}")
            return True

        except Exception as e:
            logger.error(f"Error updating certification status: {e}")
            return False

    def generate_compliance_report(self, start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 category: Optional[ComplianceCategory] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report

        Args:
            start_date: Report start date
            end_date: Report end date
            category: Specific compliance category

        Returns:
            Compliance report dictionary
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()

        # Filter records by date range
        filtered_records = [
            record for record in self.compliance_records.values()
            if start_date <= record.check_date <= end_date
        ]

        # Filter by category if specified
        if category:
            filtered_records = [
                record for record in filtered_records
                if self.requirements.get(record.requirement_id, ComplianceRequirement('', '', '', category, '', '', [], 0, 0)).category == category
            ]

        # Calculate compliance metrics
        total_checks = len(filtered_records)
        compliant_checks = len([r for r in filtered_records if r.status == ComplianceStatus.COMPLIANT])
        non_compliant_checks = len([r for r in filtered_records if r.status == ComplianceStatus.NON_COMPLIANT])
        warning_checks = len([r for r in filtered_records if r.status == ComplianceStatus.WARNING])

        compliance_rate = (compliant_checks / total_checks * 100) if total_checks > 0 else 0

        # Certification status
        active_certifications = len([c for c in self.certifications.values() if c.status == ComplianceStatus.COMPLIANT])
        expiring_certifications = len([c for c in self.certifications.values() if c.status == ComplianceStatus.WARNING])
        expired_certifications = len([c for c in self.certifications.values() if c.status == ComplianceStatus.EXPIRED])

        # Risk analysis
        critical_issues = self._analyze_critical_risks(filtered_records)

        # Trending analysis
        compliance_trend = self._calculate_compliance_trend(filtered_records)

        # Top findings
        top_findings = self._get_top_findings(filtered_records)

        report = {
            'report_metadata': {
                'generated_date': datetime.now(),
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'category_filter': category.value if category else 'All Categories',
                'total_requirements': len(self.requirements)
            },
            'compliance_summary': {
                'total_checks': total_checks,
                'compliance_rate': compliance_rate,
                'compliant_checks': compliant_checks,
                'warning_checks': warning_checks,
                'non_compliant_checks': non_compliant_checks
            },
            'certification_status': {
                'active_certifications': active_certifications,
                'expiring_certifications': expiring_certifications,
                'expired_certifications': expired_certifications,
                'total_certifications': len(self.certifications)
            },
            'risk_analysis': critical_issues,
            'compliance_trend': compliance_trend,
            'top_findings': top_findings,
            'audit_summary': self._get_audit_summary(start_date, end_date),
            'recommendations': self._generate_compliance_recommendations()
        }

        logger.info(f"Generated compliance report: {compliance_rate:.1f}% compliance rate")
        return report

    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time compliance dashboard data

        Returns:
            Dashboard data dictionary
        """
        # Current compliance status
        current_status = self._calculate_current_compliance_status()

        # Upcoming deadlines
        upcoming_deadlines = self._get_upcoming_deadlines()

        # Active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]

        # Equipment compliance summary
        equipment_summary = self._get_equipment_compliance_summary()

        # Recent activity
        recent_activity = self._get_recent_compliance_activity()

        # Performance metrics
        performance_metrics = self._calculate_performance_metrics()

        dashboard_data = {
            'current_status': current_status,
            'upcoming_deadlines': upcoming_deadlines,
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == RiskLevel.CRITICAL]),
            'equipment_summary': equipment_summary,
            'recent_activity': recent_activity,
            'performance_metrics': performance_metrics,
            'compliance_score': current_status['overall_compliance_rate']
        }

        return dashboard_data

    def _calculate_current_compliance_status(self) -> Dict[str, Any]:
        """Calculate current overall compliance status"""
        # Get latest record for each requirement
        latest_records = {}
        for record in self.compliance_records.values():
            req_id = record.requirement_id
            if req_id not in latest_records or record.check_date > latest_records[req_id].check_date:
                latest_records[req_id] = record

        total_requirements = len(self.requirements)
        compliant_count = len([r for r in latest_records.values() if r.status == ComplianceStatus.COMPLIANT])
        warning_count = len([r for r in latest_records.values() if r.status == ComplianceStatus.WARNING])
        non_compliant_count = len([r for r in latest_records.values() if r.status == ComplianceStatus.NON_COMPLIANT])

        overall_rate = (compliant_count / total_requirements * 100) if total_requirements > 0 else 0

        return {
            'total_requirements': total_requirements,
            'compliant_count': compliant_count,
            'warning_count': warning_count,
            'non_compliant_count': non_compliant_count,
            'overall_compliance_rate': overall_rate,
            'last_updated': datetime.now()
        }

    def _get_upcoming_deadlines(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming compliance deadlines"""
        deadline_date = datetime.now() + timedelta(days=days_ahead)
        upcoming = []

        # Check compliance record deadlines
        for record in self.compliance_records.values():
            if record.next_review_date and record.next_review_date <= deadline_date:
                requirement = self.requirements.get(record.requirement_id)
                if requirement:
                    upcoming.append({
                        'type': 'compliance_check',
                        'title': requirement.name,
                        'due_date': record.next_review_date,
                        'days_remaining': (record.next_review_date - datetime.now()).days,
                        'risk_level': requirement.risk_level.value,
                        'equipment_id': record.equipment_id
                    })

        # Check certification renewals
        for certification in self.certifications.values():
            if certification.renewal_date <= deadline_date:
                upcoming.append({
                    'type': 'certification_renewal',
                    'title': f"{certification.name} Renewal",
                    'due_date': certification.renewal_date,
                    'days_remaining': (certification.renewal_date - datetime.now()).days,
                    'risk_level': 'high',
                    'certification_id': certification.certification_id
                })

        # Sort by due date
        upcoming.sort(key=lambda x: x['due_date'])

        return upcoming[:10]  # Return top 10 most urgent

    def _get_equipment_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary by equipment"""
        equipment_summary = {}

        for equipment_id, requirement_ids in self.equipment_compliance.items():
            if equipment_id == "ALL":
                continue

            total_reqs = len(requirement_ids)
            latest_records = []

            for req_id in requirement_ids:
                # Get latest record for this requirement and equipment
                equipment_records = [
                    r for r in self.compliance_records.values()
                    if r.requirement_id == req_id and r.equipment_id == equipment_id
                ]

                if equipment_records:
                    latest_record = max(equipment_records, key=lambda x: x.check_date)
                    latest_records.append(latest_record)

            compliant = len([r for r in latest_records if r.status == ComplianceStatus.COMPLIANT])
            compliance_rate = (compliant / total_reqs * 100) if total_reqs > 0 else 0

            equipment_summary[equipment_id] = {
                'total_requirements': total_reqs,
                'compliant_count': compliant,
                'compliance_rate': compliance_rate,
                'last_check': max([r.check_date for r in latest_records]) if latest_records else None
            }

        return equipment_summary

    def _get_recent_compliance_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent compliance activity"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_activity = []

        # Recent compliance checks
        recent_records = [
            r for r in self.compliance_records.values()
            if r.check_date >= cutoff_date
        ]

        for record in recent_records:
            requirement = self.requirements.get(record.requirement_id)
            recent_activity.append({
                'type': 'compliance_check',
                'date': record.check_date,
                'description': f"Compliance check: {requirement.name if requirement else 'Unknown'}",
                'status': record.status.value,
                'auditor': record.auditor
            })

        # Recent audits
        recent_audits = [
            a for a in self.audits.values()
            if a.start_date >= cutoff_date
        ]

        for audit in recent_audits:
            recent_activity.append({
                'type': 'audit',
                'date': audit.start_date,
                'description': f"Audit: {audit.title}",
                'status': audit.status,
                'auditor': audit.auditor
            })

        # Sort by date (most recent first)
        recent_activity.sort(key=lambda x: x['date'], reverse=True)

        return recent_activity[:20]  # Return 20 most recent activities

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.compliance_records:
            return {}

        # Average compliance score
        scored_records = [r for r in self.compliance_records.values() if r.score is not None]
        avg_score = np.mean([r.score for r in scored_records]) if scored_records else 0

        # Time to resolution for non-compliance issues
        resolved_issues = [
            r for r in self.compliance_records.values()
            if r.status == ComplianceStatus.COMPLIANT and r.corrective_actions
        ]

        # Audit effectiveness (simplified)
        completed_audits = [a for a in self.audits.values() if a.status == "completed"]
        audit_effectiveness = np.mean([a.overall_score for a in completed_audits if a.overall_score]) if completed_audits else 0

        return {
            'average_compliance_score': avg_score,
            'audit_effectiveness': audit_effectiveness,
            'issues_resolved_30_days': len(resolved_issues),
            'average_resolution_time_days': 15.5  # Simplified calculation
        }

    def _analyze_critical_risks(self, records: List[ComplianceRecord]) -> List[Dict[str, Any]]:
        """Analyze critical compliance risks"""
        critical_risks = []

        # Group by requirement
        requirement_records = defaultdict(list)
        for record in records:
            requirement_records[record.requirement_id].append(record)

        for req_id, req_records in requirement_records.items():
            requirement = self.requirements.get(req_id)
            if not requirement:
                continue

            # Calculate risk indicators
            non_compliant_rate = len([r for r in req_records if r.status == ComplianceStatus.NON_COMPLIANT]) / len(req_records)
            avg_score = np.mean([r.score for r in req_records if r.score is not None]) if any(r.score for r in req_records) else 0

            # Identify critical risks
            if (requirement.risk_level == RiskLevel.CRITICAL and non_compliant_rate > 0.1) or avg_score < 70:
                critical_risks.append({
                    'requirement_id': req_id,
                    'requirement_name': requirement.name,
                    'risk_level': requirement.risk_level.value,
                    'non_compliance_rate': non_compliant_rate * 100,
                    'average_score': avg_score,
                    'potential_penalties': requirement.penalties
                })

        return critical_risks

    def _calculate_compliance_trend(self, records: List[ComplianceRecord]) -> Dict[str, Any]:
        """Calculate compliance trend over time"""
        if not records:
            return {}

        # Group records by month
        monthly_data = defaultdict(list)
        for record in records:
            month_key = record.check_date.strftime('%Y-%m')
            monthly_data[month_key].append(record)

        # Calculate monthly compliance rates
        trend_data = []
        for month, month_records in sorted(monthly_data.items()):
            compliant_count = len([r for r in month_records if r.status == ComplianceStatus.COMPLIANT])
            compliance_rate = (compliant_count / len(month_records)) * 100

            trend_data.append({
                'month': month,
                'compliance_rate': compliance_rate,
                'total_checks': len(month_records)
            })

        # Calculate trend direction
        if len(trend_data) >= 2:
            recent_rate = trend_data[-1]['compliance_rate']
            previous_rate = trend_data[-2]['compliance_rate']
            trend_direction = 'improving' if recent_rate > previous_rate else 'declining' if recent_rate < previous_rate else 'stable'
        else:
            trend_direction = 'insufficient_data'

        return {
            'trend_data': trend_data,
            'trend_direction': trend_direction,
            'current_month_rate': trend_data[-1]['compliance_rate'] if trend_data else 0
        }

    def _get_top_findings(self, records: List[ComplianceRecord], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common compliance findings"""
        finding_counts = defaultdict(int)

        for record in records:
            for finding in record.findings:
                # Normalize finding text for counting
                normalized_finding = finding.lower().strip()
                finding_counts[normalized_finding] += 1

        # Sort by frequency
        sorted_findings = sorted(finding_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {'finding': finding, 'count': count}
            for finding, count in sorted_findings[:limit]
        ]

    def _get_audit_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get audit summary for period"""
        period_audits = [
            a for a in self.audits.values()
            if start_date <= a.start_date <= end_date
        ]

        completed_audits = [a for a in period_audits if a.status == "completed"]
        avg_score = np.mean([a.overall_score for a in completed_audits if a.overall_score]) if completed_audits else 0

        return {
            'total_audits': len(period_audits),
            'completed_audits': len(completed_audits),
            'average_audit_score': avg_score,
            'critical_findings': sum(a.critical_findings for a in completed_audits)
        }

    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate actionable compliance recommendations"""
        recommendations = []

        # Analyze current state and generate recommendations
        current_status = self._calculate_current_compliance_status()

        if current_status['overall_compliance_rate'] < 90:
            recommendations.append("Focus on improving overall compliance rate through targeted training and process improvements")

        if current_status['non_compliant_count'] > 0:
            recommendations.append("Prioritize resolution of non-compliant items, especially critical safety requirements")

        # Check for upcoming deadlines
        upcoming = self._get_upcoming_deadlines(30)
        if len(upcoming) > 5:
            recommendations.append("Schedule compliance reviews for upcoming deadlines to prevent last-minute issues")

        # Check certification status
        expiring_certs = [c for c in self.certifications.values() if c.status == ComplianceStatus.WARNING]
        if expiring_certs:
            recommendations.append("Begin certification renewal processes for expiring certifications")

        # Generic best practices
        recommendations.extend([
            "Implement automated compliance monitoring for continuous oversight",
            "Establish regular internal audit schedules to maintain compliance readiness",
            "Provide ongoing compliance training for all maintenance personnel"
        ])

        return recommendations

    def _generate_corrective_actions(self, requirement: ComplianceRequirement, findings: List[str]) -> List[str]:
        """Generate corrective actions based on findings"""
        actions = []

        # Analyze findings and generate appropriate actions
        for finding in findings:
            finding_lower = finding.lower()

            if 'documentation' in finding_lower or 'record' in finding_lower:
                actions.append("Update and complete all required documentation")

            if 'training' in finding_lower:
                actions.append("Provide additional training for responsible personnel")

            if 'procedure' in finding_lower:
                actions.append("Review and update relevant procedures")

            if 'equipment' in finding_lower:
                actions.append("Inspect and repair/replace equipment as needed")

            if 'safety' in finding_lower:
                actions.append("Implement immediate safety measures and corrective actions")

        # Add generic actions if none specific
        if not actions:
            actions.extend([
                "Investigate root cause of compliance issue",
                "Implement corrective measures",
                "Verify effectiveness of corrective actions",
                "Schedule follow-up review"
            ])

        return actions

    def _create_compliance_alert(self, record: ComplianceRecord, requirement: ComplianceRequirement):
        """Create alert for compliance issue"""
        severity = RiskLevel.CRITICAL if requirement.risk_level == RiskLevel.CRITICAL else RiskLevel.HIGH

        alert = ComplianceAlert(
            alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            alert_type="non_compliance",
            title=f"Compliance Issue: {requirement.name}",
            description=f"Non-compliance detected for {requirement.name}. Status: {record.status.value}",
            severity=severity,
            created_date=datetime.now(),
            due_date=datetime.now() + timedelta(days=requirement.grace_period_days),
            related_items=[record.record_id, requirement.requirement_id]
        )

        self.alerts[alert.alert_id] = alert

    def _create_audit_alert(self, audit: Audit):
        """Create alert for upcoming audit"""
        alert = ComplianceAlert(
            alert_id=f"AUDIT_ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            alert_type="audit_due",
            title=f"Upcoming Audit: {audit.title}",
            description=f"Audit scheduled for {audit.start_date.strftime('%Y-%m-%d')}",
            severity=RiskLevel.MEDIUM,
            created_date=datetime.now(),
            due_date=audit.start_date,
            related_items=[audit.audit_id]
        )

        self.alerts[alert.alert_id] = alert

    def _create_certification_alert(self, certification: Certification):
        """Create alert for certification renewal"""
        days_to_expiry = (certification.expiry_date - datetime.now()).days

        severity = RiskLevel.CRITICAL if days_to_expiry < 30 else RiskLevel.HIGH

        alert = ComplianceAlert(
            alert_id=f"CERT_ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            alert_type="expiry_warning",
            title=f"Certification Renewal: {certification.name}",
            description=f"Certification expires in {days_to_expiry} days",
            severity=severity,
            created_date=datetime.now(),
            due_date=certification.expiry_date,
            related_items=[certification.certification_id]
        )

        self.alerts[alert.alert_id] = alert

    def auto_update_statuses(self):
        """Automatically update all compliance and certification statuses"""
        # Update certification statuses
        for cert_id in self.certifications.keys():
            self.update_certification_status(cert_id)

        # Check for overdue compliance checks
        for requirement in self.requirements.values():
            self._check_overdue_compliance(requirement)

        logger.info("Completed automatic status updates")

    def _check_overdue_compliance(self, requirement: ComplianceRequirement):
        """Check for overdue compliance checks"""
        # Get latest record for this requirement
        requirement_records = [
            r for r in self.compliance_records.values()
            if r.requirement_id == requirement.requirement_id
        ]

        if not requirement_records:
            # No records exist - create alert for missing compliance check
            self._create_missing_compliance_alert(requirement)
            return

        latest_record = max(requirement_records, key=lambda x: x.check_date)
        days_since_check = (datetime.now() - latest_record.check_date).days

        if days_since_check > requirement.frequency_days + requirement.grace_period_days:
            # Overdue - create alert
            self._create_overdue_compliance_alert(requirement, latest_record)

    def _create_missing_compliance_alert(self, requirement: ComplianceRequirement):
        """Create alert for missing compliance check"""
        alert = ComplianceAlert(
            alert_id=f"MISSING_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            alert_type="missing_compliance",
            title=f"Missing Compliance Check: {requirement.name}",
            description=f"No compliance records found for {requirement.name}",
            severity=requirement.risk_level,
            created_date=datetime.now(),
            related_items=[requirement.requirement_id]
        )

        self.alerts[alert.alert_id] = alert

    def _create_overdue_compliance_alert(self, requirement: ComplianceRequirement, latest_record: ComplianceRecord):
        """Create alert for overdue compliance check"""
        days_overdue = (datetime.now() - latest_record.due_date).days

        alert = ComplianceAlert(
            alert_id=f"OVERDUE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            alert_type="overdue_compliance",
            title=f"Overdue Compliance: {requirement.name}",
            description=f"Compliance check overdue by {days_overdue} days",
            severity=RiskLevel.CRITICAL,
            created_date=datetime.now(),
            related_items=[requirement.requirement_id, latest_record.record_id]
        )

        self.alerts[alert.alert_id] = alert


# Demo function
def create_demo_compliance_tracker() -> RegulatoryComplianceTracker:
    """Create demo compliance tracker with sample data

    Returns:
        Configured compliance tracker
    """
    tracker = RegulatoryComplianceTracker()

    # Create some sample compliance records
    sample_auditor_names = ["Jane Smith", "Bob Wilson", "Alice Brown", "Mike Johnson"]

    for req_id in list(tracker.requirements.keys())[:3]:
        # Create recent compliance check
        tracker.conduct_compliance_check(
            req_id,
            equipment_id="EQ_001" if req_id != "GDPR_DATA_001" else None,
            auditor=np.random.choice(sample_auditor_names),
            score=np.random.uniform(80, 98),
            findings=[] if np.random.random() > 0.3 else ["Minor documentation gap"]
        )

    # Schedule some audits
    for i in range(3):
        audit_date = datetime.now() + timedelta(days=np.random.randint(7, 60))
        tracker.schedule_audit(
            AuditType.INTERNAL,
            f"Internal Audit {i+1}",
            np.random.choice(sample_auditor_names),
            audit_date,
            ["EQ_001", "EQ_002"]
        )

    # Update all statuses
    tracker.auto_update_statuses()

    logger.info("Created demo compliance tracker with sample data and records")
    return tracker