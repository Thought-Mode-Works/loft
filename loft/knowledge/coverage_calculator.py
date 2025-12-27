"""
Metrics calculation engine for knowledge coverage.

Computes comprehensive coverage, quality, and performance metrics.

Issue #274: Knowledge Coverage Metrics
"""

import logging
from typing import Dict, List

from sqlalchemy import func

from loft.knowledge.coverage_schemas import (
    CoverageGap,
    CoverageMetrics,
    CoverageReport,
    DoctrineMetrics,
    DomainMetrics,
    JurisdictionMetrics,
    QualityMetrics,
)
from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.models import LegalQuestion, LegalRule

logger = logging.getLogger(__name__)


class CoverageCalculator:
    """
    Calculate comprehensive coverage metrics for knowledge base.

    Computes metrics for domains, doctrines, jurisdictions, and overall quality.
    """

    def __init__(self, knowledge_db: KnowledgeDatabase):
        """
        Initialize calculator with knowledge database.

        Args:
            knowledge_db: Knowledge database instance
        """
        self.db = knowledge_db

    def calculate_metrics(self) -> CoverageMetrics:
        """
        Calculate comprehensive coverage metrics.

        Returns:
            CoverageMetrics with all metrics populated
        """
        logger.info("Calculating knowledge coverage metrics")

        metrics = CoverageMetrics()

        # Get database session
        with self.db.SessionLocal() as session:
            # Overall counts
            metrics.total_rules = (
                session.query(func.count(LegalRule.rule_id)).scalar() or 0
            )
            metrics.active_rules = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(is_active=True)
                .scalar()
                or 0
            )
            metrics.archived_rules = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(is_archived=True)
                .scalar()
                or 0
            )
            metrics.total_questions = (
                session.query(func.count(LegalQuestion.question_id)).scalar() or 0
            )
            metrics.answered_questions = (
                session.query(func.count(LegalQuestion.question_id))
                .filter(LegalQuestion.answer.isnot(None))
                .scalar()
                or 0
            )

            # Calculate domain metrics
            metrics.domains = self._calculate_domain_metrics(session)

            # Calculate doctrine metrics
            metrics.doctrines = self._calculate_doctrine_metrics(session)

            # Calculate jurisdiction metrics
            metrics.jurisdictions = self._calculate_jurisdiction_metrics(session)

            # Calculate quality metrics
            metrics.quality = self._calculate_quality_metrics(session)

        logger.info(
            f"Calculated metrics: {metrics.total_rules} rules across "
            f"{metrics.domain_count} domains"
        )

        return metrics

    def _calculate_domain_metrics(self, session) -> Dict[str, DomainMetrics]:
        """
        Calculate metrics for each domain.

        Args:
            session: Database session

        Returns:
            Dict mapping domain names to DomainMetrics
        """
        domain_metrics = {}

        # Get all domains
        domains = (
            session.query(LegalRule.domain)
            .filter(LegalRule.domain.isnot(None))
            .distinct()
            .all()
        )

        for (domain,) in domains:
            metrics = DomainMetrics(domain=domain)

            # Rule counts
            metrics.rule_count = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(domain=domain)
                .scalar()
                or 0
            )
            metrics.active_rule_count = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(domain=domain, is_active=True)
                .scalar()
                or 0
            )
            metrics.archived_rule_count = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(domain=domain, is_archived=True)
                .scalar()
                or 0
            )

            # Average confidence
            avg_conf = (
                session.query(func.avg(LegalRule.confidence))
                .filter_by(domain=domain, is_active=True)
                .filter(LegalRule.confidence.isnot(None))
                .scalar()
            )
            metrics.avg_confidence = float(avg_conf) if avg_conf else 0.0

            # Question metrics
            metrics.question_count = (
                session.query(func.count(LegalQuestion.question_id))
                .filter_by(domain=domain)
                .scalar()
                or 0
            )
            metrics.answered_question_count = (
                session.query(func.count(LegalQuestion.question_id))
                .filter_by(domain=domain)
                .filter(LegalQuestion.answer.isnot(None))
                .scalar()
                or 0
            )

            # Accuracy (from correct answers)
            if metrics.answered_question_count > 0:
                correct_count = (
                    session.query(func.count(LegalQuestion.question_id))
                    .filter_by(domain=domain, correct=True)
                    .scalar()
                    or 0
                )
                metrics.accuracy = correct_count / metrics.answered_question_count
            else:
                metrics.accuracy = None

            # Doctrines in this domain
            doctrines = (
                session.query(LegalRule.doctrine)
                .filter_by(domain=domain)
                .filter(LegalRule.doctrine.isnot(None))
                .distinct()
                .all()
            )
            metrics.doctrines = [d[0] for d in doctrines]

            # Jurisdictions in this domain
            jurisdictions = (
                session.query(LegalRule.jurisdiction)
                .filter_by(domain=domain)
                .filter(LegalRule.jurisdiction.isnot(None))
                .distinct()
                .all()
            )
            metrics.jurisdictions = [j[0] for j in jurisdictions]

            domain_metrics[domain] = metrics

        return domain_metrics

    def _calculate_doctrine_metrics(self, session) -> Dict[str, DoctrineMetrics]:
        """
        Calculate metrics for each doctrine.

        Args:
            session: Database session

        Returns:
            Dict mapping doctrine names to DoctrineMetrics
        """
        doctrine_metrics = {}

        # Get all doctrines
        doctrines = (
            session.query(LegalRule.doctrine, LegalRule.domain)
            .filter(LegalRule.doctrine.isnot(None))
            .distinct()
            .all()
        )

        for doctrine, domain in doctrines:
            key = f"{domain}:{doctrine}" if domain else doctrine

            metrics = DoctrineMetrics(doctrine=doctrine, domain=domain or "unknown")

            # Rule count
            query = session.query(func.count(LegalRule.rule_id)).filter_by(
                doctrine=doctrine, is_active=True
            )
            if domain:
                query = query.filter_by(domain=domain)

            metrics.rule_count = query.scalar() or 0

            # Average confidence
            query = (
                session.query(func.avg(LegalRule.confidence))
                .filter_by(doctrine=doctrine, is_active=True)
                .filter(LegalRule.confidence.isnot(None))
            )
            if domain:
                query = query.filter_by(domain=domain)

            avg_conf = query.scalar()
            metrics.avg_confidence = float(avg_conf) if avg_conf else 0.0

            # Question count (if domain available)
            if domain:
                # Find questions that use rules from this doctrine
                # This is approximate - we'd need to track which rules were used
                metrics.question_count = (
                    session.query(func.count(LegalQuestion.question_id))
                    .filter_by(domain=domain)
                    .scalar()
                    or 0
                )

            doctrine_metrics[key] = metrics

        return doctrine_metrics

    def _calculate_jurisdiction_metrics(
        self, session
    ) -> Dict[str, JurisdictionMetrics]:
        """
        Calculate metrics for each jurisdiction.

        Args:
            session: Database session

        Returns:
            Dict mapping jurisdiction names to JurisdictionMetrics
        """
        jurisdiction_metrics = {}

        # Get all jurisdictions
        jurisdictions = (
            session.query(LegalRule.jurisdiction)
            .filter(LegalRule.jurisdiction.isnot(None))
            .distinct()
            .all()
        )

        for (jurisdiction,) in jurisdictions:
            metrics = JurisdictionMetrics(jurisdiction=jurisdiction)

            # Rule count
            metrics.rule_count = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(jurisdiction=jurisdiction, is_active=True)
                .scalar()
                or 0
            )

            # Domains in this jurisdiction
            domains = (
                session.query(LegalRule.domain)
                .filter_by(jurisdiction=jurisdiction)
                .filter(LegalRule.domain.isnot(None))
                .distinct()
                .all()
            )
            metrics.domains = [d[0] for d in domains]

            # Average confidence
            avg_conf = (
                session.query(func.avg(LegalRule.confidence))
                .filter_by(jurisdiction=jurisdiction, is_active=True)
                .filter(LegalRule.confidence.isnot(None))
                .scalar()
            )
            metrics.avg_confidence = float(avg_conf) if avg_conf else 0.0

            jurisdiction_metrics[jurisdiction] = metrics

        return jurisdiction_metrics

    def _calculate_quality_metrics(self, session) -> QualityMetrics:
        """
        Calculate overall quality metrics.

        Args:
            session: Database session

        Returns:
            QualityMetrics
        """
        metrics = QualityMetrics()

        # Total rules
        metrics.total_rules = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .scalar()
            or 0
        )

        if metrics.total_rules == 0:
            return metrics

        # Average confidence
        avg_conf = (
            session.query(func.avg(LegalRule.confidence))
            .filter_by(is_active=True)
            .filter(LegalRule.confidence.isnot(None))
            .scalar()
        )
        metrics.avg_confidence = float(avg_conf) if avg_conf else 0.0

        # Confidence distribution
        metrics.high_confidence_rules = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .filter(LegalRule.confidence >= 0.9)
            .scalar()
            or 0
        )
        metrics.medium_confidence_rules = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .filter(LegalRule.confidence >= 0.7, LegalRule.confidence < 0.9)
            .scalar()
            or 0
        )
        metrics.low_confidence_rules = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .filter(LegalRule.confidence < 0.7)
            .scalar()
            or 0
        )

        # Rules with metadata
        metrics.rules_with_reasoning = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .filter(LegalRule.reasoning.isnot(None))
            .filter(LegalRule.reasoning != "")
            .scalar()
            or 0
        )

        metrics.rules_with_sources = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .filter(LegalRule.source_cases.isnot(None))
            .scalar()
            or 0
        )

        # Validated rules
        metrics.validated_rules = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(is_active=True)
            .filter(LegalRule.validation_count > 0)
            .scalar()
            or 0
        )

        return metrics

    def identify_gaps(self, metrics: CoverageMetrics) -> List[CoverageGap]:
        """
        Identify coverage gaps based on metrics.

        Args:
            metrics: Coverage metrics

        Returns:
            List of identified gaps
        """
        gaps = []

        # Check for domains with low rule count
        for domain_name, domain in metrics.domains.items():
            if domain.rule_count < 10:
                gaps.append(
                    CoverageGap(
                        area=domain_name,
                        gap_type="missing_rules",
                        severity=min(1.0 - (domain.rule_count / 10.0), 1.0),
                        description=f"Only {domain.rule_count} rules in domain",
                        suggested_action="Add more rules from case law or legal principles",
                    )
                )

            # Check for low accuracy
            if domain.accuracy is not None and domain.accuracy < 0.7:
                gaps.append(
                    CoverageGap(
                        area=domain_name,
                        gap_type="low_accuracy",
                        severity=1.0 - domain.accuracy,
                        description=f"Accuracy is only {domain.accuracy:.1%}",
                        suggested_action="Review and improve existing rules, add missing rules",
                    )
                )

            # Check for low confidence
            if domain.avg_confidence < 0.7:
                gaps.append(
                    CoverageGap(
                        area=domain_name,
                        gap_type="low_confidence",
                        severity=1.0 - domain.avg_confidence,
                        description=f"Average confidence is only {domain.avg_confidence:.1%}",
                        suggested_action="Validate rules against more cases, refine rule extraction",
                    )
                )

        # Check overall quality
        if metrics.quality.quality_score < 0.7:
            gaps.append(
                CoverageGap(
                    area="overall",
                    gap_type="low_quality",
                    severity=1.0 - metrics.quality.quality_score,
                    description=f"Overall quality score is {metrics.quality.quality_score:.1%}",
                    suggested_action="Improve rule metadata, increase confidence through validation",
                )
            )

        return gaps

    def generate_recommendations(
        self, metrics: CoverageMetrics, gaps: List[CoverageGap]
    ) -> List[str]:
        """
        Generate actionable recommendations based on metrics and gaps.

        Args:
            metrics: Coverage metrics
            gaps: Identified gaps

        Returns:
            List of recommendations
        """
        recommendations = []

        # Prioritize high-severity gaps
        high_severity_gaps = [g for g in gaps if g.severity >= 0.7]

        if high_severity_gaps:
            recommendations.append(
                f"Address {len(high_severity_gaps)} high-severity gaps identified in coverage"
            )

        # Check for domains needing attention
        low_coverage_domains = [
            name
            for name, domain in metrics.domains.items()
            if domain.coverage_score < 0.5
        ]

        if low_coverage_domains:
            recommendations.append(
                f"Focus on improving coverage in: {', '.join(low_coverage_domains)}"
            )

        # Quality improvements
        if metrics.quality.low_confidence_rules > metrics.quality.total_rules * 0.3:
            recommendations.append(
                "Over 30% of rules have low confidence - validate against more cases"
            )

        # Metadata completeness
        if metrics.quality.rules_with_reasoning < metrics.quality.total_rules * 0.5:
            recommendations.append("Add reasoning to rules to improve explainability")

        # Validation
        if metrics.quality.validated_rules < metrics.quality.total_rules * 0.5:
            recommendations.append(
                "Validate more rules against test cases to improve confidence"
            )

        # Balanced coverage
        if metrics.domain_count > 0:
            avg_rules_per_domain = metrics.total_rules / metrics.domain_count
            unbalanced_domains = [
                name
                for name, domain in metrics.domains.items()
                if domain.rule_count < avg_rules_per_domain * 0.5
            ]

            if unbalanced_domains:
                recommendations.append(
                    f"Balance rule distribution - these domains are underrepresented: "
                    f"{', '.join(unbalanced_domains)}"
                )

        return recommendations

    def generate_report(self) -> CoverageReport:
        """
        Generate comprehensive coverage report.

        Returns:
            CoverageReport with metrics, gaps, and recommendations
        """
        logger.info("Generating coverage report")

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Identify gaps
        gaps = self.identify_gaps(metrics)

        # Generate recommendations
        recommendations = self.generate_recommendations(metrics, gaps)

        report = CoverageReport(
            metrics=metrics, gaps=gaps, recommendations=recommendations
        )

        logger.info(
            f"Generated report: {metrics.domain_count} domains, "
            f"{len(gaps)} gaps, {len(recommendations)} recommendations"
        )

        return report
