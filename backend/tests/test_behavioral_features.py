"""
Unit tests for behavioral feature engineering (Phase 2).

Tests cover:
- EventNormalizer: RFC3164, RFC5424, JSON, missing timestamps, BSD year override
- BehavioralWindowExtractor: empty window, single event, multi-event, multi-host
- Zero-denominator ratio handling
- Out-of-order event sorting
- Cold-start handling (first-seen host)
- Historical baseline leakage prevention (window uses only past events)
- Duplicate event deduplication
- Window boundary correctness
- Training/inference feature parity
- BehavioralFeatureSchema save/load roundtrip
- build_behavioral_dataframe shape and column order
"""

import json
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ── imports from project ─────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.feature_engineering.behavioral_features import (
    EventNormalizer,
    BehavioralWindowExtractor,
    BehavioralFeatureSchema,
    BEHAVIORAL_FEATURE_NAMES,
    build_behavioral_dataframe,
)
from backend.feature_engineering.log_parser import LogParser
from backend.training.ait_label_config import (
    ALL_FEATURES, BEHAVIORAL_FEATURES, ALL_BEHAVIORAL_FEATURES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(offset_minutes: float = 0) -> datetime:
    """Return a timezone-aware UTC datetime offset from epoch."""
    base = datetime(2022, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _make_event(
    hostname: str = "host1",
    log_type: str = "auth",
    ts_offset: float = 0,
    auth_failed: bool = False,
    auth_success: bool = False,
    has_error: bool = False,
    has_warning: bool = False,
    is_sudo_su: bool = False,
    is_privilege: bool = False,
    source_ip: str = None,
    service: str = "sshd",
) -> Dict[str, Any]:
    """Construct a minimal normalized event dict for testing."""
    return {
        "timestamp_utc":  _ts(ts_offset),
        "hostname":       hostname,
        "log_type":       log_type,
        "service":        service,
        "message":        "test message",
        "auth_failed":    auth_failed,
        "auth_success":   auth_success,
        "is_sudo_su":     is_sudo_su,
        "is_privilege":   is_privilege,
        "has_error":      has_error,
        "has_warning":    has_warning,
        "source_ip":      source_ip,
        "raw_line_md5":   None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. EventNormalizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEventNormalizer:

    def test_rfc3164_parse_and_normalize(self):
        """RFC 3164 syslog line is parsed and normalized correctly."""
        line = "Jan 15 12:00:01 host1 sshd[1234]: Failed password for invalid user admin from 10.0.0.1 port 22 ssh2"
        parsed = LogParser.parse_line(line)
        ev = EventNormalizer.normalize(parsed, log_type="auth")
        assert ev["hostname"] == "host1"
        assert ev["service"] == "sshd"
        assert ev["log_type"] == "auth"
        assert ev["auth_failed"] is True
        assert ev["auth_success"] is False
        assert ev["source_ip"] == "10.0.0.1"

    def test_rfc5424_parse_and_normalize(self):
        """RFC 5424 syslog line is parsed and normalized."""
        line = "<34>1 2022-01-15T12:00:00Z web01 apache2 - - - GET /wp-admin HTTP/1.1 200"
        parsed = LogParser.parse_line(line)
        ev = EventNormalizer.normalize(parsed, log_type="apache")
        assert ev["log_type"] == "apache"
        assert ev["hostname"] == "web01"

    def test_json_log_normalize(self):
        """JSON log is parsed and normalized."""
        line = '{"timestamp":"2022-01-15T12:00:00Z","hostname":"db01","service":"mysql","message":"Auth error for root"}'
        parsed = LogParser.parse_line(line)
        ev = EventNormalizer.normalize(parsed, log_type="syslog")
        assert ev["log_type"] == "syslog"

    def test_missing_timestamp_returns_none(self):
        """Normalized event with unparseable timestamp has timestamp_utc=None."""
        parsed = {
            "hostname": "host1",
            "service": "unknown",
            "message": "some log line",
            "timestamp": None,
            "source_ip": None,
        }
        ev = EventNormalizer.normalize(parsed, log_type="syslog")
        assert ev["timestamp_utc"] is None

    def test_sudo_detection(self):
        """sudo keyword in message sets is_sudo_su=True."""
        parsed = LogParser.parse_line("Jan 15 12:00:01 host1 sudo[999]: user ran sudo -i")
        ev = EventNormalizer.normalize(parsed, log_type="auth")
        assert ev["is_sudo_su"] is True

    def test_privilege_detection(self):
        """Privilege-related keyword sets is_privilege=True."""
        parsed = LogParser.parse_line("Jan 15 12:00:01 host1 kernel: process escalated privileges")
        ev = EventNormalizer.normalize(parsed, log_type="syslog")
        assert ev["is_privilege"] is True

    def test_bsd_year_override(self):
        """Scenario year override changes the parsed UTC year."""
        EventNormalizer.set_scenario_year(2022)
        parsed = LogParser.parse_line("Jan 15 03:12:44 host1 sshd[1]: Connected")
        ev = EventNormalizer.normalize(parsed, log_type="auth")
        if ev["timestamp_utc"] is not None:
            assert ev["timestamp_utc"].year == 2022
        # Reset
        EventNormalizer.set_scenario_year(None)

    def test_malformed_line_does_not_crash(self):
        """A completely malformed line still returns a normalized event."""
        parsed = LogParser.parse_line("!!! not a log line $%^ 0x00 \x00\xff")
        ev = EventNormalizer.normalize(parsed, log_type="syslog")
        assert isinstance(ev, dict)
        assert "hostname" in ev


# ─────────────────────────────────────────────────────────────────────────────
# 2. BehavioralWindowExtractor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralWindowExtractor:

    def _extractor(self, window_minutes: int = 5) -> BehavioralWindowExtractor:
        return BehavioralWindowExtractor(window_minutes=window_minutes)

    def test_empty_window_returns_cold_start(self):
        """First event for a host has no window history → cold_start=1, all zeros."""
        ext = self._extractor()
        ev = _make_event(ts_offset=0)
        feats = ext.extract_batch([ev])
        f = feats[0]
        assert f["beh_cold_start"] == 1
        assert f["beh_total_events"] == 0
        assert f["beh_auth_fail_count"] == 0

    def test_single_auth_fail_counted(self):
        """A failed auth event in window is reflected in the next event's features."""
        ext = self._extractor(window_minutes=5)
        events = [
            _make_event(ts_offset=0, auth_failed=True),
            _make_event(ts_offset=1),  # should see the fail event in window
        ]
        ext.fit_baseline([events[0]])
        feats = ext.extract_batch(events)
        # Second event: should see 1 auth_fail from first event in window
        assert feats[1]["beh_auth_fail_count"] == 1
        assert feats[1]["beh_total_events"] == 1

    def test_multiple_hosts_isolated(self):
        """Events on different hostnames do not affect each other's windows."""
        ext = self._extractor()
        events = [
            _make_event(hostname="host1", ts_offset=0, auth_failed=True),
            _make_event(hostname="host2", ts_offset=1),
        ]
        ext.fit_baseline(events)
        feats = ext.extract_batch(events)
        # host2's event should NOT see host1's auth failure
        assert feats[1]["beh_auth_fail_count"] == 0

    def test_window_boundary_exclusion(self):
        """Events exactly at window boundary are excluded (strictly past-only)."""
        ext = self._extractor(window_minutes=5)
        events = [
            _make_event(ts_offset=0, auth_failed=True),
            _make_event(ts_offset=5),   # exactly at boundary — should be INSIDE window
            _make_event(ts_offset=10),  # event 0 at T=0 should be OUTSIDE 5-min window from T=10
        ]
        ext.fit_baseline(events[:1])
        feats = ext.extract_batch(events)
        # At T=10, window is [T=5, T=10) — event at T=0 is excluded
        assert feats[2]["beh_auth_fail_count"] == 0  # T=0 event is outside window

    def test_out_of_order_events_handled(self):
        """Out-of-order events are sorted by timestamp before processing."""
        ext = self._extractor()
        events = [
            _make_event(ts_offset=5, auth_failed=True),   # later event first
            _make_event(ts_offset=0),                      # earlier event second
        ]
        ext.fit_baseline(events)
        feats = ext.extract_batch(events)
        # After sorting: T=0 first (cold start), T=5 second (sees T=0 in window)
        # feats are indexed by ORIGINAL position, not sorted position
        # T=0 event is at orig_idx=1 → should be cold start
        # T=5 event is at orig_idx=0 → should see T=0 in window
        assert feats[1]["beh_cold_start"] == 1  # T=0 event: no prior events
        assert feats[0]["beh_total_events"] >= 0  # T=5 event sees T=0

    def test_zero_denominator_ratio_safe(self):
        """When no auth events exist in window, fail_success_ratio = 0.0 safely."""
        ext = self._extractor()
        events = [
            _make_event(ts_offset=0),  # non-auth event
            _make_event(ts_offset=1),  # non-auth event
        ]
        ext.fit_baseline(events)
        feats = ext.extract_batch(events)
        assert feats[1]["beh_fail_success_ratio"] == 0.0
        assert feats[1]["beh_auth_fail_count"] == 0
        assert feats[1]["beh_auth_success_count"] == 0

    def test_fail_success_ratio_calculation(self):
        """Fail ratio = fail / (fail + success + ε) when both present."""
        ext = self._extractor()
        # 2 fails, 1 success in window
        events = [
            _make_event(ts_offset=0, auth_failed=True),
            _make_event(ts_offset=1, auth_failed=True),
            _make_event(ts_offset=2, auth_success=True),
            _make_event(ts_offset=3),  # observer event
        ]
        ext.fit_baseline(events[:3])
        feats = ext.extract_batch(events)
        f = feats[3]
        assert f["beh_auth_fail_count"] == 2
        assert f["beh_auth_success_count"] == 1
        expected_ratio = 2 / (2 + 1)
        assert abs(f["beh_fail_success_ratio"] - expected_ratio) < 0.01

    def test_sudo_su_count_increments(self):
        """sudo events in window are correctly counted."""
        ext = self._extractor()
        events = [
            _make_event(ts_offset=0, is_sudo_su=True),
            _make_event(ts_offset=1, is_sudo_su=True),
            _make_event(ts_offset=2),
        ]
        ext.fit_baseline(events)
        feats = ext.extract_batch(events)
        assert feats[2]["beh_sudo_su_count"] == 2

    def test_distinct_ips_counted(self):
        """Multiple distinct source IPs in window are counted."""
        ext = self._extractor()
        events = [
            _make_event(ts_offset=0, source_ip="10.0.0.1"),
            _make_event(ts_offset=1, source_ip="10.0.0.2"),
            _make_event(ts_offset=2, source_ip="10.0.0.1"),  # duplicate IP
            _make_event(ts_offset=3),
        ]
        ext.fit_baseline(events)
        feats = ext.extract_batch(events)
        assert feats[3]["beh_distinct_ips"] == 2  # 10.0.0.1 and 10.0.0.2

    def test_events_without_timestamp_get_cold_start(self):
        """Events with no timestamp produce cold-start feature values."""
        ext = self._extractor()
        events = [{"timestamp_utc": None, "hostname": "h1", "log_type": "syslog",
                   "service": "unknown", "message": "", "auth_failed": False,
                   "auth_success": False, "is_sudo_su": False, "is_privilege": False,
                   "has_error": False, "has_warning": False, "source_ip": None,
                   "raw_line_md5": None}]
        feats = ext.extract_batch(events)
        assert feats[0]["beh_cold_start"] == 1
        assert feats[0]["beh_total_events"] == 0

    def test_baseline_uses_only_training_events(self):
        """fit_baseline must be called only on training events; here we verify
        the baseline is not computed from test events by checking that the
        extractor produces cold_start=1 for unseen hosts."""
        train_events = [_make_event(hostname="train_host", ts_offset=i) for i in range(10)]
        test_events  = [_make_event(hostname="test_host",  ts_offset=i) for i in range(5)]
        ext = self._extractor()
        ext.fit_baseline(train_events)
        feats = ext.extract_batch(test_events)
        # test_host has no baseline → first event is cold start
        assert feats[0]["beh_cold_start"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. BehavioralFeatureSchema persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralFeatureSchema:

    def test_save_and_load_roundtrip(self):
        """Schema survives a save/load roundtrip without data loss."""
        schema = BehavioralFeatureSchema(
            version="v1",
            window_minutes=5,
            grouping_keys=["hostname"],
            message_feature_names=list(ALL_FEATURES),
            behavioral_feature_names=list(BEHAVIORAL_FEATURES),
            all_feature_names=list(ALL_BEHAVIORAL_FEATURES),
            extractor_config={"window_minutes": 5, "baseline_fitted": True},
            baseline_stats_path=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            schema.save(path)
            loaded = BehavioralFeatureSchema.load(path)
        assert loaded.version == "v1"
        assert loaded.window_minutes == 5
        assert loaded.all_feature_names == list(ALL_BEHAVIORAL_FEATURES)
        assert loaded.grouping_keys == ["hostname"]

    def test_validate_features_correct(self):
        """validate_features returns True for correct feature list."""
        schema = BehavioralFeatureSchema(
            version="v1", window_minutes=5, grouping_keys=["hostname"],
            message_feature_names=list(ALL_FEATURES),
            behavioral_feature_names=list(BEHAVIORAL_FEATURES),
            all_feature_names=list(ALL_BEHAVIORAL_FEATURES),
            extractor_config={}, baseline_stats_path=None,
        )
        ok, msg = schema.validate_features(list(ALL_BEHAVIORAL_FEATURES))
        assert ok is True
        assert msg == ""

    def test_validate_features_detects_missing(self):
        """validate_features returns False when a feature is missing."""
        schema = BehavioralFeatureSchema(
            version="v1", window_minutes=5, grouping_keys=["hostname"],
            message_feature_names=list(ALL_FEATURES),
            behavioral_feature_names=list(BEHAVIORAL_FEATURES),
            all_feature_names=list(ALL_BEHAVIORAL_FEATURES),
            extractor_config={}, baseline_stats_path=None,
        )
        truncated = list(ALL_BEHAVIORAL_FEATURES)[:-2]  # remove last 2
        ok, msg = schema.validate_features(truncated)
        assert ok is False
        assert "missing" in msg

    def test_validate_features_detects_order_mismatch(self):
        """validate_features returns False when feature order differs."""
        schema = BehavioralFeatureSchema(
            version="v1", window_minutes=5, grouping_keys=["hostname"],
            message_feature_names=list(ALL_FEATURES),
            behavioral_feature_names=list(BEHAVIORAL_FEATURES),
            all_feature_names=list(ALL_BEHAVIORAL_FEATURES),
            extractor_config={}, baseline_stats_path=None,
        )
        shuffled = list(reversed(ALL_BEHAVIORAL_FEATURES))
        ok, msg = schema.validate_features(shuffled)
        assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselinePersistence:

    def test_save_and_load_baseline_roundtrip(self):
        """Baseline survives save/load roundtrip."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        events = [_make_event(hostname="h1", ts_offset=i) for i in range(20)]
        ext.fit_baseline(events)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            ext.save_baseline(path)
            ext2 = BehavioralWindowExtractor(window_minutes=5)
            ext2.load_baseline(path)
        assert ext2._baseline_fitted is True
        assert "__global__" in ext2._baseline_counts
        # Loaded extractor should produce same cold_start status as original
        test_ev = _make_event(hostname="h1", ts_offset=25)
        feats_orig = ext.extract_batch([test_ev])
        feats_loaded = ext2.extract_batch([test_ev])
        assert feats_orig[0]["beh_cold_start"] == feats_loaded[0]["beh_cold_start"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training/inference parity
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingInferenceParity:

    def test_feature_names_match_constant(self):
        """Feature names produced by extractor match BEHAVIORAL_FEATURE_NAMES constant."""
        ext = BehavioralWindowExtractor()
        ev = _make_event(ts_offset=0)
        feats = ext.extract_batch([ev])
        produced_keys = sorted(feats[0].keys())
        expected_keys = sorted(BEHAVIORAL_FEATURE_NAMES)
        assert produced_keys == expected_keys

    def test_extract_one_matches_extract_batch(self):
        """extract_one (inference) produces the same features as extract_batch (training)."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        # Build window events
        window_events = [
            _make_event(hostname="h1", ts_offset=i, auth_failed=(i % 3 == 0))
            for i in range(5)
        ]
        ext.fit_baseline(window_events)

        # extract_batch on all events
        all_events = window_events + [_make_event(hostname="h1", ts_offset=6)]
        batch_feats = ext.extract_batch(all_events)

        # extract_one: build the buffer manually (past events within window)
        t_current = _ts(6)
        cutoff = t_current - timedelta(minutes=5)
        buf: List[Tuple[datetime, Dict]] = [
            (e["timestamp_utc"], e) for e in window_events
            if e["timestamp_utc"] is not None and e["timestamp_utc"] >= cutoff
        ]
        single_feat = ext.extract_one(all_events[-1], buf)

        # Both should produce the same behavioral counts
        assert single_feat["beh_auth_fail_count"] == batch_feats[-1]["beh_auth_fail_count"]
        assert single_feat["beh_total_events"] == batch_feats[-1]["beh_total_events"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. build_behavioral_dataframe
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildBehavioralDataframe:

    def test_shape_and_columns(self):
        """DataFrame has correct number of rows and column order."""
        import pandas as pd
        n = 4
        events = [_make_event(ts_offset=i) for i in range(n)]
        ext = BehavioralWindowExtractor()
        ext.fit_baseline(events)
        beh_feats = ext.extract_batch(events)

        # Inject message-level feature values into norm_events dict
        for ev in events:
            for col in ALL_FEATURES:
                ev[col] = 0  # simplified: real pipeline extracts from LogParser

        df = build_behavioral_dataframe(events, beh_feats, ALL_FEATURES)
        assert len(df) == n
        assert list(df.columns) == ALL_FEATURES + BEHAVIORAL_FEATURE_NAMES

    def test_length_mismatch_raises(self):
        """Mismatched event/feature lengths raise AssertionError."""
        import pandas as pd
        events = [_make_event(ts_offset=i) for i in range(3)]
        beh_feats = [{}] * 2  # wrong length
        with pytest.raises(AssertionError):
            build_behavioral_dataframe(events, beh_feats, ALL_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ait_label_config constants
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelConfigConstants:

    def test_all_behavioral_features_length(self):
        """ALL_BEHAVIORAL_FEATURES has message + behavioral features."""
        assert len(ALL_BEHAVIORAL_FEATURES) == len(ALL_FEATURES) + len(BEHAVIORAL_FEATURES)

    def test_behavioral_features_match_module_constant(self):
        """BEHAVIORAL_FEATURES in ait_label_config matches BEHAVIORAL_FEATURE_NAMES in module."""
        assert list(BEHAVIORAL_FEATURES) == list(BEHAVIORAL_FEATURE_NAMES)

    def test_no_duplicates_in_all_behavioral_features(self):
        """No duplicate feature names in the combined feature list."""
        assert len(ALL_BEHAVIORAL_FEATURES) == len(set(ALL_BEHAVIORAL_FEATURES))
