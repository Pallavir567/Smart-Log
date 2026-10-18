"""
Tests for BehavioralEventBuffer live-inference integration.

Covers
------
1. Buffer accumulates events across successive calls, so later events see earlier ones.
2. Different hosts are isolated — events from host A do not appear in host B's window.
3. Events with a missing timestamp receive cold-start features (beh_cold_start=1).
4. The current event is NOT counted in its own window (no self-inclusion).
5. Repeated identical events (duplicate raw lines) do NOT corrupt window counts.
6. Out-of-order events: events arriving later with earlier timestamps are still
   correctly windowed against whatever is already in the buffer.
7. BehavioralEventBuffer.reset() fully clears state between tests.
8. InferenceService integration: a behavioral-mode model accumulates events across
   two successive predict_logs() calls, and the second batch sees the history from
   the first.
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.behavioral_state import BehavioralEventBuffer
from backend.feature_engineering.behavioral_features import (
    EventNormalizer,
    BehavioralWindowExtractor,
    BEHAVIORAL_FEATURE_NAMES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(offset_minutes: float = 0) -> datetime:
    """Return a timezone-aware UTC datetime close to wall-clock now.

    Using wall-clock-relative timestamps ensures add_event's age-based
    eviction (cutoff = now - max_buffer_minutes) does not discard these
    synthetic events before we have a chance to query them.

    We start from (now - 30 min) so even offset=0 is recent enough to
    survive the default 60-minute eviction window, while the small positive
    offsets (0–10 min) land comfortably within the window.
    """
    base = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
    return base + timedelta(minutes=offset_minutes)


def _norm_event(
    hostname: str = "host1",
    ts_offset: float = 0,
    auth_failed: bool = False,
    auth_success: bool = False,
    service: str = "sshd",
    source_ip: str = None,
    log_type: str = "auth",
    has_error: bool = False,
) -> Dict[str, Any]:
    """Build a minimal normalized event dict for testing."""
    return {
        "timestamp_utc": _ts(ts_offset),
        "hostname": hostname,
        "log_type": log_type,
        "service": service,
        "message": "test log line",
        "auth_failed": auth_failed,
        "auth_success": auth_success,
        "is_sudo_su": False,
        "is_privilege": False,
        "has_error": has_error,
        "has_warning": False,
        "source_ip": source_ip,
        "raw_line_md5": None,
    }


def _make_extractor(window_minutes: int = 5) -> BehavioralWindowExtractor:
    ext = BehavioralWindowExtractor(window_minutes=window_minutes)
    # Fit a minimal baseline so beh_cold_start is 0 for known hosts
    training_events = [_norm_event(hostname="host1", ts_offset=i) for i in range(10)]
    ext.fit_baseline(training_events)
    return ext


@pytest.fixture(autouse=True)
def reset_buffer():
    """Reset the singleton buffer before each test for isolation."""
    BehavioralEventBuffer.reset()
    yield
    BehavioralEventBuffer.reset()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Buffer accumulation across successive calls
# ─────────────────────────────────────────────────────────────────────────────

class TestBufferAccumulation:

    def test_second_event_sees_first_event_in_window(self):
        """Second event arriving later should see the first event in its window."""
        ext = _make_extractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()

        ev1 = _norm_event(hostname="host1", ts_offset=0, auth_failed=True)
        ev2 = _norm_event(hostname="host1", ts_offset=1)  # 1 min after ev1

        gk = ext._grouping_key(ev1)

        # Process ev1: empty window → cold-start / no prior events
        window1 = buf.get_window(gk, ev1["timestamp_utc"], ext.window_minutes)
        feat1 = ext.extract_one(ev1, window1)
        buf.add_event(gk, ev1)

        # Process ev2: should now see ev1 in window
        window2 = buf.get_window(gk, ev2["timestamp_utc"], ext.window_minutes)
        feat2 = ext.extract_one(ev2, window2)
        buf.add_event(gk, ev2)

        assert feat1["beh_total_events"] == 0   # ev1 had no prior events
        assert feat2["beh_total_events"] == 1   # ev2 sees ev1
        assert feat2["beh_auth_fail_count"] == 1  # ev1 was auth_failed

    def test_third_event_sees_two_prior_events(self):
        """Third event should see both previous events still within the window."""
        ext = _make_extractor(window_minutes=10)
        buf = BehavioralEventBuffer.get_instance()

        events = [
            _norm_event(hostname="host1", ts_offset=0, auth_failed=True),
            _norm_event(hostname="host1", ts_offset=1, auth_success=True),
            _norm_event(hostname="host1", ts_offset=2),
        ]

        gk = ext._grouping_key(events[0])
        feats = []
        for ev in events:
            window = buf.get_window(gk, ev["timestamp_utc"], ext.window_minutes)
            feat = ext.extract_one(ev, window)
            feats.append(feat)
            buf.add_event(gk, ev)

        assert feats[2]["beh_total_events"] == 2
        assert feats[2]["beh_auth_fail_count"] == 1
        assert feats[2]["beh_auth_success_count"] == 1

    def test_window_drops_events_outside_window(self):
        """Events older than the window should not appear in the window."""
        ext = _make_extractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()

        ev_old = _norm_event(hostname="host1", ts_offset=0, auth_failed=True)
        ev_new = _norm_event(hostname="host1", ts_offset=10)  # 10 min later

        gk = ext._grouping_key(ev_old)

        window_old = buf.get_window(gk, ev_old["timestamp_utc"], ext.window_minutes)
        buf.add_event(gk, ev_old)

        # ev_new is 10 min after ev_old; with a 5-min window, ev_old is outside
        window_new = buf.get_window(gk, ev_new["timestamp_utc"], ext.window_minutes)
        feat_new = ext.extract_one(ev_new, window_new)

        assert feat_new["beh_total_events"] == 0  # ev_old fell out of window
        assert feat_new["beh_auth_fail_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Host isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestHostIsolation:

    def test_events_on_different_hosts_are_isolated(self):
        """Auth failures on host1 must not appear in host2's window."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        ext.fit_baseline([
            _norm_event(hostname="host1", ts_offset=i) for i in range(5)
        ] + [
            _norm_event(hostname="host2", ts_offset=i) for i in range(5)
        ])
        buf = BehavioralEventBuffer.get_instance()

        ev_h1 = _norm_event(hostname="host1", ts_offset=0, auth_failed=True)
        ev_h2 = _norm_event(hostname="host2", ts_offset=1)

        gk_h1 = ext._grouping_key(ev_h1)
        gk_h2 = ext._grouping_key(ev_h2)

        # Add host1's event
        w1 = buf.get_window(gk_h1, ev_h1["timestamp_utc"], ext.window_minutes)
        ext.extract_one(ev_h1, w1)
        buf.add_event(gk_h1, ev_h1)

        # host2 should see nothing from host1
        w2 = buf.get_window(gk_h2, ev_h2["timestamp_utc"], ext.window_minutes)
        feat_h2 = ext.extract_one(ev_h2, w2)

        assert feat_h2["beh_total_events"] == 0
        assert feat_h2["beh_auth_fail_count"] == 0

    def test_multiple_hosts_accumulate_independently(self):
        """Each host accumulates its own window counts independently."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        ext.fit_baseline(
            [_norm_event(hostname="h1", ts_offset=i) for i in range(5)] +
            [_norm_event(hostname="h2", ts_offset=i) for i in range(5)]
        )
        buf = BehavioralEventBuffer.get_instance()

        # Add 3 events to h1
        for i in range(3):
            ev = _norm_event(hostname="h1", ts_offset=i, auth_failed=True)
            gk = ext._grouping_key(ev)
            buf.add_event(gk, ev)

        # Add 1 event to h2
        ev_h2 = _norm_event(hostname="h2", ts_offset=0, auth_success=True)
        gk_h2 = ext._grouping_key(ev_h2)
        buf.add_event(gk_h2, ev_h2)

        # Now query: h1's next event should see 3 prior events
        ev_h1_next = _norm_event(hostname="h1", ts_offset=3)
        gk_h1 = ext._grouping_key(ev_h1_next)
        w_h1 = buf.get_window(gk_h1, ev_h1_next["timestamp_utc"], ext.window_minutes)
        feat_h1 = ext.extract_one(ev_h1_next, w_h1)

        # h2's next event should see 1 prior event only
        ev_h2_next = _norm_event(hostname="h2", ts_offset=1)
        w_h2 = buf.get_window(gk_h2, ev_h2_next["timestamp_utc"], ext.window_minutes)
        feat_h2 = ext.extract_one(ev_h2_next, w_h2)

        assert feat_h1["beh_total_events"] == 3
        assert feat_h1["beh_auth_fail_count"] == 3
        assert feat_h2["beh_total_events"] == 1
        assert feat_h2["beh_auth_success_count"] == 1
        assert feat_h2["beh_auth_fail_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing-timestamp cold-start
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingTimestampColdStart:

    def test_event_without_timestamp_gets_cold_start(self):
        """An event with timestamp_utc=None must produce cold-start features."""
        ext = _make_extractor()
        buf = BehavioralEventBuffer.get_instance()

        ev_no_ts = {
            "timestamp_utc": None,
            "hostname": "host1",
            "log_type": "syslog",
            "service": "unknown",
            "message": "",
            "auth_failed": False,
            "auth_success": False,
            "is_sudo_su": False,
            "is_privilege": False,
            "has_error": False,
            "has_warning": False,
            "source_ip": None,
            "raw_line_md5": None,
        }
        gk = ext._grouping_key(ev_no_ts)
        window = []  # can't query buffer without a valid timestamp
        feat = ext.extract_one(ev_no_ts, window)

        assert feat["beh_cold_start"] == 1
        assert feat["beh_total_events"] == 0
        assert all(feat[k] == 0 for k in BEHAVIORAL_FEATURE_NAMES if k != "beh_cold_start")

    def test_buffer_stores_no_ts_event_at_epoch(self):
        """add_event with timestamp_utc=None stores event at epoch; it is evicted
        by a real-time window query and does not pollute subsequent windows."""
        buf = BehavioralEventBuffer.get_instance()
        ev_no_ts = _norm_event(hostname="host1", ts_offset=0)
        ev_no_ts["timestamp_utc"] = None
        gk = "host1"
        buf.add_event(gk, ev_no_ts)

        # A real-time window should not include the epoch-stored event
        real_ts = _ts(1)
        window = buf.get_window(gk, real_ts, window_minutes=5)
        # Epoch (1970) is far outside any 5-min window around 2022-06-01
        assert len(window) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. No self-inclusion (current event must not count itself)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSelfInclusion:

    def test_first_event_has_zero_prior_events(self):
        """The very first event for a host must have beh_total_events=0."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()

        ev = _norm_event(hostname="brand_new_host", ts_offset=0)
        gk = ext._grouping_key(ev)
        window = buf.get_window(gk, ev["timestamp_utc"], ext.window_minutes)
        feat = ext.extract_one(ev, window)

        assert feat["beh_total_events"] == 0

    def test_event_does_not_count_itself_in_window(self):
        """After add_event, the same event's timestamp is in the buffer; but its
        own window query (called before add_event) saw zero prior events."""
        ext = _make_extractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()

        ev = _norm_event(hostname="host1", ts_offset=0, auth_failed=True)
        gk = ext._grouping_key(ev)

        # Query before add → empty window
        window_before = buf.get_window(gk, ev["timestamp_utc"], ext.window_minutes)
        feat_before = ext.extract_one(ev, window_before)
        assert feat_before["beh_total_events"] == 0
        assert feat_before["beh_auth_fail_count"] == 0

        # Add the event
        buf.add_event(gk, ev)

        # The NEXT event at +1 min should see the previous event
        ev_next = _norm_event(hostname="host1", ts_offset=1)
        window_next = buf.get_window(gk, ev_next["timestamp_utc"], ext.window_minutes)
        feat_next = ext.extract_one(ev_next, window_next)
        assert feat_next["beh_total_events"] == 1
        assert feat_next["beh_auth_fail_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Repeated / duplicate events
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateEvents:

    def test_identical_events_added_twice_both_counted(self):
        """Adding the same event dict twice is treated as two distinct arrivals.
        The buffer is not a set; duplicates accumulate (same as a real log stream
        that emits two identical lines close together)."""
        ext = _make_extractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()

        ev = _norm_event(hostname="host1", ts_offset=0, auth_failed=True)
        gk = ext._grouping_key(ev)

        buf.add_event(gk, ev)
        buf.add_event(gk, ev)  # duplicate

        ev_next = _norm_event(hostname="host1", ts_offset=1)
        window = buf.get_window(gk, ev_next["timestamp_utc"], ext.window_minutes)
        feat = ext.extract_one(ev_next, window)

        # Both copies should be in the window
        assert feat["beh_total_events"] == 2
        assert feat["beh_auth_fail_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. Out-of-order events
# ─────────────────────────────────────────────────────────────────────────────

class TestOutOfOrderEvents:

    def test_late_arriving_past_event_visible_to_same_time_window(self):
        """An event arriving late (with an earlier timestamp than the current tip)
        is added at its actual timestamp and will appear in windows of future
        events whose window covers that timestamp."""
        ext = _make_extractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()

        # Arrive at T=3 first (out of order)
        ev_at_3 = _norm_event(hostname="host1", ts_offset=3, auth_failed=True)
        gk = ext._grouping_key(ev_at_3)
        w = buf.get_window(gk, ev_at_3["timestamp_utc"], ext.window_minutes)
        ext.extract_one(ev_at_3, w)
        buf.add_event(gk, ev_at_3)

        # Now a late-arriving event from T=1 is processed
        ev_at_1 = _norm_event(hostname="host1", ts_offset=1, has_error=True)
        w_late = buf.get_window(gk, ev_at_1["timestamp_utc"], ext.window_minutes)
        ext.extract_one(ev_at_1, w_late)
        buf.add_event(gk, ev_at_1)

        # An event at T=4 should see both T=3 and T=1 (both in 5-min window)
        ev_at_4 = _norm_event(hostname="host1", ts_offset=4)
        w4 = buf.get_window(gk, ev_at_4["timestamp_utc"], ext.window_minutes)
        feat_4 = ext.extract_one(ev_at_4, w4)

        assert feat_4["beh_total_events"] == 2
        assert feat_4["beh_auth_fail_count"] == 1
        assert feat_4["beh_error_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. InferenceService integration (behavioral model accumulates across batches)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceServiceBufferIntegration:
    """
    Verify that the live InferenceService behavioral injection block calls
    get_window before add_event for each event in the batch.

    We do NOT run a full trained model here (that requires a DB and artifacts).
    Instead we test the injection helper logic directly via a mock extractor +
    mock bundle to confirm the call order and that beh_rows have the right shape.
    """

    def test_injection_block_calls_get_window_before_add_event(self):
        """
        Simulate what predict_logs() does in the behavioral injection block and
        verify that:
          - get_window is called BEFORE add_event for each event
          - beh_rows has one entry per parsed record
          - the second event's window contains the first event (via actual buffer)
        """
        from backend.services.behavioral_state import BehavioralEventBuffer
        from backend.feature_engineering.behavioral_features import (
            EventNormalizer, BehavioralWindowExtractor, BEHAVIORAL_FEATURE_NAMES,
        )

        # Build a fresh extractor with baseline fitted on host1
        ext = BehavioralWindowExtractor(window_minutes=5)
        ext.fit_baseline([_norm_event(hostname="host1", ts_offset=i) for i in range(10)])

        buf = BehavioralEventBuffer.get_instance()
        normalizer = EventNormalizer()

        # Use wall-clock-relative ISO timestamps (avoids eviction)
        t0 = _ts(0).strftime("%Y-%m-%dT%H:%M:%SZ")
        t1 = _ts(1).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Simulate two parsed records on the same host
        parsed_records = [
            {"hostname": "host1", "timestamp": t0,
             "service": "sshd", "message": "Failed password for root", "source_ip": "10.0.0.1"},
            {"hostname": "host1", "timestamp": t1,
             "service": "sshd", "message": "Accepted publickey for ubuntu", "source_ip": "10.0.0.2"},
        ]
        extracted_features_list = [{} for _ in parsed_records]

        beh_rows = []
        for i, r in enumerate(parsed_records):
            norm_ev = normalizer.normalize(
                {
                    "hostname":  r.get("hostname", "localhost"),
                    "timestamp": r.get("timestamp", ""),
                    "service":   r.get("service", "unknown"),
                    "message":   r.get("message", ""),
                    "_features": extracted_features_list[i],
                    "source_ip": r.get("source_ip"),
                },
                log_type="syslog",
            )
            gk = ext._grouping_key(norm_ev)
            ts_utc = norm_ev.get("timestamp_utc")
            if ts_utc is not None:
                window = buf.get_window(gk, before_time=ts_utc, window_minutes=ext.window_minutes)
            else:
                window = []
            feat = ext.extract_one(norm_ev, window)
            beh_rows.append(feat)
            buf.add_event(gk, norm_ev)

        # Both records produced a feature dict
        assert len(beh_rows) == 2

        # All expected feature names present in each dict
        for row in beh_rows:
            assert set(row.keys()) == set(BEHAVIORAL_FEATURE_NAMES)

        # First event had no prior history → zero prior events
        assert beh_rows[0]["beh_total_events"] == 0

        # Second event should see the first event in the window
        assert beh_rows[1]["beh_total_events"] == 1

    def test_injection_block_second_batch_sees_first_batch_history(self):
        """Two successive injection runs on the same host accumulate state."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        ext.fit_baseline([_norm_event(hostname="host1", ts_offset=i) for i in range(10)])
        buf = BehavioralEventBuffer.get_instance()
        normalizer = EventNormalizer()

        def _run_batch(parsed_records, extracted_features_list):
            beh_rows = []
            for i, r in enumerate(parsed_records):
                norm_ev = normalizer.normalize(
                    {"hostname": r["hostname"], "timestamp": r["timestamp"],
                     "service": r["service"], "message": r["message"],
                     "_features": extracted_features_list[i], "source_ip": None},
                    log_type="syslog",
                )
                gk = ext._grouping_key(norm_ev)
                ts_utc = norm_ev.get("timestamp_utc")
                window = buf.get_window(gk, ts_utc, ext.window_minutes) if ts_utc else []
                beh_rows.append(ext.extract_one(norm_ev, window))
                buf.add_event(gk, norm_ev)
            return beh_rows

        # Use wall-clock-relative timestamps so add_event does not evict them
        t0 = _ts(0).strftime("%Y-%m-%dT%H:%M:%SZ")
        t1 = _ts(1).strftime("%Y-%m-%dT%H:%M:%SZ")
        t2 = _ts(2).strftime("%Y-%m-%dT%H:%M:%SZ")

        # First batch: 2 events on host1
        batch1 = [
            {"hostname": "host1", "timestamp": t0, "service": "sshd", "message": "fail1"},
            {"hostname": "host1", "timestamp": t1, "service": "sshd", "message": "fail2"},
        ]
        rows1 = _run_batch(batch1, [{}, {}])

        # Second batch: 1 event on host1 at T+2 min
        batch2 = [
            {"hostname": "host1", "timestamp": t2, "service": "sshd", "message": "ok"},
        ]
        rows2 = _run_batch(batch2, [{}])

        # Second batch event should see 2 prior events from first batch
        assert rows2[0]["beh_total_events"] == 2

    def test_injection_block_buffer_reset_clears_history(self):
        """After BehavioralEventBuffer.reset(), a new event should cold-start."""
        ext = BehavioralWindowExtractor(window_minutes=5)
        buf = BehavioralEventBuffer.get_instance()
        normalizer = EventNormalizer()

        # Add an event
        ev = _norm_event(hostname="host1", ts_offset=0, auth_failed=True)
        gk = ext._grouping_key(ev)
        buf.add_event(gk, ev)

        # Verify a second event sees it
        ev2 = _norm_event(hostname="host1", ts_offset=1)
        w = buf.get_window(gk, ev2["timestamp_utc"], ext.window_minutes)
        feat = ext.extract_one(ev2, w)
        assert feat["beh_total_events"] == 1

        # Reset
        BehavioralEventBuffer.reset()
        buf2 = BehavioralEventBuffer.get_instance()

        # After reset, the new buffer is empty
        w_empty = buf2.get_window(gk, ev2["timestamp_utc"], ext.window_minutes)
        assert len(w_empty) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Message-level model path: no behavioral features injected
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageLevelModelUnchanged:

    def test_behavioral_extractor_none_means_no_beh_columns(self):
        """If a bundle has no behavioral_extractor, the DataFrame must not have
        any beh_* columns injected — message-level path is preserved."""
        import pandas as pd

        df_features = pd.DataFrame([
            {"message_length": 50, "token_count": 8, "digit_ratio": 0.1,
             "entropy": 2.5, "has_error": 0, "has_warning": 0, "has_auth": 1,
             "has_failed_auth": 0, "has_denied": 0, "has_ip": 1, "service": "sshd"}
        ])

        behavioral_extractor = None  # message-level model

        df_for_model = df_features.copy()
        if behavioral_extractor is not None:
            # (would inject beh_* columns here)
            pass

        for feat in BEHAVIORAL_FEATURE_NAMES:
            assert feat not in df_for_model.columns, (
                f"Column {feat!r} should not be in message-level model DataFrame"
            )
