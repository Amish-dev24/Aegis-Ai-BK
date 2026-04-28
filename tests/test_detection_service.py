"""
Unit tests for DetectionService business logic.

These tests do NOT load AI model weights — they test the pure-Python
decision logic (classify_threat_level, threat thresholds, etc.) which
works without any model files.

The DetectionService singleton is imported from the already-initialised
module-level instance.  Because model files point to non-existent paths
(set in conftest), all model references are None — no inference occurs.
"""

import pytest

from app.models.detection import DetectionType, ThreatLevel

# ── Fixture: reuse the module-level singleton (models are all None) ──────────


@pytest.fixture(scope="module")
def svc():
    """Return the already-initialised DetectionService singleton."""
    from app.services.detection_service import detection_service

    return detection_service


# ── classify_threat_level — Weapon ───────────────────────────────────────────


class TestWeaponThreatLevels:
    def test_weapon_critical(self, svc):
        lvl = svc.classify_threat_level(DetectionType.WEAPON, confidence=0.95)
        assert lvl == ThreatLevel.CRITICAL

    def test_weapon_high(self, svc):
        lvl = svc.classify_threat_level(DetectionType.WEAPON, confidence=0.75)
        assert lvl == ThreatLevel.HIGH

    def test_weapon_medium(self, svc):
        lvl = svc.classify_threat_level(DetectionType.WEAPON, confidence=0.55)
        assert lvl == ThreatLevel.MEDIUM

    def test_weapon_low(self, svc):
        lvl = svc.classify_threat_level(DetectionType.WEAPON, confidence=0.3)
        assert lvl == ThreatLevel.LOW

    def test_weapon_boundary_critical(self, svc):
        """Exactly at the critical threshold (0.9) should be CRITICAL."""
        lvl = svc.classify_threat_level(DetectionType.WEAPON, confidence=0.9)
        assert lvl == ThreatLevel.CRITICAL


# ── classify_threat_level — Violence ─────────────────────────────────────────


class TestViolenceThreatLevels:
    def test_violence_critical(self, svc):
        lvl = svc.classify_threat_level(DetectionType.VIOLENCE, confidence=0.85)
        assert lvl == ThreatLevel.CRITICAL

    def test_violence_high(self, svc):
        lvl = svc.classify_threat_level(DetectionType.VIOLENCE, confidence=0.70)
        assert lvl == ThreatLevel.HIGH

    def test_violence_medium(self, svc):
        lvl = svc.classify_threat_level(DetectionType.VIOLENCE, confidence=0.55)
        assert lvl == ThreatLevel.MEDIUM

    def test_violence_low(self, svc):
        lvl = svc.classify_threat_level(DetectionType.VIOLENCE, confidence=0.20)
        assert lvl == ThreatLevel.LOW


# ── classify_threat_level — Abandoned Object ─────────────────────────────────


class TestAbandonedObjectThreatLevels:
    def test_no_critical_for_abandoned(self, svc):
        """Abandoned objects have no CRITICAL level per design."""
        lvl = svc.classify_threat_level(DetectionType.ABANDONED_OBJECT, confidence=0.99)
        assert lvl != ThreatLevel.CRITICAL

    def test_abandoned_high(self, svc):
        lvl = svc.classify_threat_level(DetectionType.ABANDONED_OBJECT, confidence=0.85)
        assert lvl == ThreatLevel.HIGH

    def test_abandoned_medium(self, svc):
        lvl = svc.classify_threat_level(DetectionType.ABANDONED_OBJECT, confidence=0.6)
        assert lvl == ThreatLevel.MEDIUM

    def test_abandoned_low(self, svc):
        lvl = svc.classify_threat_level(DetectionType.ABANDONED_OBJECT, confidence=0.1)
        assert lvl == ThreatLevel.LOW


# ── classify_threat_level — Mask / Face ──────────────────────────────────────


class TestMaskFaceThreatLevels:
    def test_max_level_is_medium(self, svc):
        """Mask/face detections are capped at MEDIUM, never HIGH or CRITICAL."""
        lvl = svc.classify_threat_level(DetectionType.MASK_FACE, confidence=1.0)
        assert lvl == ThreatLevel.MEDIUM

    def test_mask_face_medium(self, svc):
        lvl = svc.classify_threat_level(DetectionType.MASK_FACE, confidence=0.95)
        assert lvl == ThreatLevel.MEDIUM

    def test_mask_face_low(self, svc):
        lvl = svc.classify_threat_level(DetectionType.MASK_FACE, confidence=0.5)
        assert lvl == ThreatLevel.LOW


# ── classify_threat_level — Crowd Density ────────────────────────────────────


class TestCrowdDensityThreatLevels:
    def test_crowd_low_count(self, svc):
        lvl = svc.classify_threat_level(
            DetectionType.CROWD_DENSITY,
            confidence=0.5,
            metadata={"count": 5, "density": 0.05},
        )
        assert lvl == ThreatLevel.LOW

    def test_crowd_high_count(self, svc):
        """count=95 out of scale=100 should be HIGH."""
        lvl = svc.classify_threat_level(
            DetectionType.CROWD_DENSITY,
            confidence=0.95,
            metadata={"count": 95, "density": 0.95},
        )
        assert lvl in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_crowd_zero_count_is_low(self, svc):
        lvl = svc.classify_threat_level(
            DetectionType.CROWD_DENSITY,
            confidence=0.0,
            metadata={"count": 0, "density": 0.0},
        )
        assert lvl == ThreatLevel.LOW


# ── classify_threat_level — Company threshold overrides ──────────────────────


class TestCompanyThresholdOverrides:
    def test_company_lower_critical_threshold(self, svc):
        """If company sets critical_threshold=0.6 for weapons, 0.65 must be CRITICAL."""
        company_thresholds = {
            "critical_threshold": 0.6,
            "high_threshold": 0.4,
            "medium_threshold": 0.2,
        }
        lvl = svc.classify_threat_level(
            DetectionType.WEAPON,
            confidence=0.65,
            company_thresholds=company_thresholds,
        )
        assert lvl == ThreatLevel.CRITICAL

    def test_company_all_none_falls_back_to_defaults(self, svc):
        """When all company thresholds are None, system defaults are used."""
        company_thresholds = {
            "critical_threshold": None,
            "high_threshold": None,
            "medium_threshold": None,
        }
        lvl_override = svc.classify_threat_level(
            DetectionType.WEAPON,
            confidence=0.95,
            company_thresholds=company_thresholds,
        )
        lvl_default = svc.classify_threat_level(
            DetectionType.WEAPON,
            confidence=0.95,
        )
        assert lvl_override == lvl_default


# ── DEFAULT_THRESHOLDS sanity check ──────────────────────────────────────────


class TestDefaultThresholds:
    def test_all_detection_types_have_entry(self, svc):
        for dt in DetectionType:
            assert dt in svc.DEFAULT_THRESHOLDS, f"Missing default threshold for {dt}"

    def test_threshold_tuples_are_length_3(self, svc):
        for dt, thresholds in svc.DEFAULT_THRESHOLDS.items():
            assert len(thresholds) == 3, f"{dt} should have (critical, high, medium)"
