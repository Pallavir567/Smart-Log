"""
Behavioral Feature Engineering for Smart-Log
==============================================
Implements time-window based behavioral feature extraction on top of existing
message-level features from LogParser.

Architecture
------------
EventNormalizer          — Converts LogParser output into a normalized event dict
BehavioralWindowExtractor — Computes per-event behavioral features using a
                            configurable sliding window of PAST events only
BehavioralFeatureSchema  — Dataclass persisted alongside each model artifact;
                           guarantees training/inference feature parity

Design constraints
------------------
- Uses only fields available in AIT-LDS logs (timestamps, hostname, service,
  log_type, auth flags, sudo/privilege keywords)
- No future events are used to compute features for a current event
- Historical baseline is computed on training data only and stored as a
  JSON artifact; loaded at inference time
- Cold-start (first-seen host) is handled explicitly and flagged
- Username features are omitted by default due to high heuristic false-positive
  rate in the AIT-LDS data

Usage (training)
----------------
    normalizer = EventNormalizer()
    extractor = BehavioralWindowExtractor(window_minutes=5)
    norm_events = [normalizer.normalize(parsed, log_type) for parsed in records]
    extractor.fit_baseline(norm_events)              # train split only
    features = extractor.extract_batch(norm_events)  # returns list[dict]

Usage (inference — streaming)
------------------------------
    extractor.load_baseline(path)          # from saved artifact
    norm_event = normalizer.normalize(parsed, log_type)
    beh_features = extractor.extract_one(norm_event, event_buffer)
"""

from __future__ import annotations

import json
import math
import re
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("smartlog.behavioral")

# ---------------------------------------------------------------------------
# Canonical behavioral feature names (order is significant — must match model)
# ---------------------------------------------------------------------------
BEHAVIORAL_FEATURE_NAMES: List[str] = [
    "beh_total_events",          # total events in window for this grouping key
    "beh_auth_fail_count",       # auth failure events in window
    "beh_auth_success_count",    # auth success events (auth but not failed)
    "beh_fail_success_ratio",    # fail / (fail + success + ε)
    "beh_error_count",           # has_error events in window
    "beh_warning_count",         # has_warning events in window
    "beh_distinct_services",     # distinct service names in window
    "beh_sudo_su_count",         # sudo/su keyword events in window
    "beh_privilege_count",       # privilege-change keyword events in window
    "beh_distinct_ips",          # distinct source IPs in window (non-null)
    "beh_rare_event_ratio",      # proportion of rare log_type×service combos
    "beh_event_freq_delta",      # z-score of window size vs historical baseline
    "beh_cold_start",            # 1 if no historical baseline for this key
]

# Regex to detect sudo/su events in message text
_SUDO_RE = re.compile(
    r"\b(sudo|su\b|sudoers|NOPASSWD|runuser|su -)", re.IGNORECASE
)
_PRIV_RE = re.compile(
    r"\b(escalat\w*|privileg\w*|setuid|setgid|chown|chmod 777|passwd|shadow|"
    r"visudo|pkexec|polkit|doas)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# EventNormalizer
# ---------------------------------------------------------------------------

class EventNormalizer:
    """
    Converts a LogParser-parsed record into a normalized event dict suitable
    for behavioral window extraction.

    Fields in normalized event
    --------------------------
    timestamp_utc   : datetime | None  (timezone-aware UTC; None if unparseable)
    hostname        : str
    log_type        : str  (auth / syslog / audit / apache / openvpn / dnsmasq)
    service         : str
    message         : str
    auth_failed     : bool   (has_failed_auth)
    auth_success    : bool   (has_auth AND NOT auth_failed)
    is_sudo_su      : bool   (sudo/su keywords in message)
    is_privilege    : bool   (privilege-change keywords in message)
    has_error       : bool
    has_warning     : bool
    source_ip       : str | None
    raw_line_md5    : str | None  (deduplication key, if provided)
    """

    # Scenario year override: when processing historical AIT-LDS datasets the
    # BSD syslog parser assigns the current calendar year. Override via
    # set_scenario_year() to match the dataset's actual year.
    _scenario_year: Optional[int] = None

    @classmethod
    def set_scenario_year(cls, year: int) -> None:
        """Set the year used when normalizing BSD-format timestamps."""
        cls._scenario_year = year

    @classmethod
    def normalize(
        cls,
        parsed: Dict[str, Any],
        log_type: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Normalize one LogParser output dict into a behavioral event dict.

        Args:
            parsed:   Output of LogParser.parse_line()
            log_type: Log source type string (auth/syslog/audit/apache/openvpn/dnsmasq)

        Returns:
            Normalized event dict. timestamp_utc may be None for unparseable timestamps.
        """
        ts_utc = cls._parse_timestamp(parsed.get("timestamp"))
        msg = str(parsed.get("message") or "")

        # Auth outcome: failed_auth takes priority; success = auth & not failed
        features = parsed.get("_features", {})
        # Support both pre-extracted features dict or re-derive from message
        has_failed = cls._check_failed_auth(msg, features)
        has_auth   = cls._check_auth(msg, features)

        auth_failed  = has_failed
        auth_success = has_auth and not has_failed

        is_sudo_su  = bool(_SUDO_RE.search(msg))
        is_privilege = bool(_PRIV_RE.search(msg))

        return {
            "timestamp_utc":  ts_utc,
            "hostname":       str(parsed.get("hostname") or "unknown"),
            "log_type":       str(log_type),
            "service":        str(parsed.get("service") or "unknown"),
            "message":        msg,
            "auth_failed":    auth_failed,
            "auth_success":   auth_success,
            "is_sudo_su":     is_sudo_su,
            "is_privilege":   is_privilege,
            "has_error":      cls._check_kw(msg, features, "has_error",
                                ["error","fail","failed","failure","fatal","exception","crash"]),
            "has_warning":    cls._check_kw(msg, features, "has_warning",
                                ["warn","warning","caution","alert"]),
            "source_ip":      parsed.get("source_ip"),
            "raw_line_md5":   parsed.get("raw_line_md5"),
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @classmethod
    def _parse_timestamp(cls, raw_ts: Any) -> Optional[datetime]:
        """
        Parse a timestamp string into a timezone-aware UTC datetime.
        Returns None if unparseable. Handles ISO-8601, BSD (Mmm dd HH:MM:SS),
        and epoch integers.

        Year-override note
        ------------------
        LogParser normalises BSD syslog timestamps to ISO strings using the
        current calendar year *before* EventNormalizer is called.  When
        set_scenario_year() has been used to declare the dataset's real year,
        we post-correct the parsed year so that downstream tests and training
        pipelines see the intended year regardless of when they are run.
        """
        if raw_ts is None:
            return None
        if isinstance(raw_ts, datetime):
            dt = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
            if cls._scenario_year and dt.year != cls._scenario_year:
                dt = dt.replace(year=cls._scenario_year)
            return dt

        s = str(raw_ts).strip()
        if not s:
            return None

        # Try ISO-8601 variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                # Apply scenario year override (LogParser may have baked in
                # the current calendar year from a BSD-format source line).
                if cls._scenario_year and dt.year != cls._scenario_year:
                    dt = dt.replace(year=cls._scenario_year)
                return dt
            except ValueError:
                pass

        # BSD: "Jan 15 03:12:44" — attach scenario year or current year
        parts = s.split()
        if len(parts) == 3:
            year = cls._scenario_year or datetime.utcnow().year
            try:
                dt = datetime.strptime(f"{year} {parts[0]} {parts[1]:>2} {parts[2]}",
                                       "%Y %b %d %H:%M:%S")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Epoch integer/float
        try:
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass

        return None

    @staticmethod
    def _check_failed_auth(msg: str, features: Dict) -> bool:
        if "has_failed_auth" in features:
            return bool(features["has_failed_auth"])
        ml = msg.lower()
        return any(k in ml for k in [
            "failed password", "authentication failure",
            "invalid user", "failed login", "auth failure",
        ])

    @staticmethod
    def _check_auth(msg: str, features: Dict) -> bool:
        if "has_auth" in features:
            return bool(features["has_auth"])
        ml = msg.lower()
        return any(k in ml for k in [
            "auth", "login", "session", "password", "sshd", "sudo", "pam",
        ])

    @staticmethod
    def _check_kw(msg: str, features: Dict, key: str, keywords: List[str]) -> bool:
        if key in features:
            return bool(features[key])
        ml = msg.lower()
        return any(k in ml for k in keywords)


# ---------------------------------------------------------------------------
# BehavioralWindowExtractor
# ---------------------------------------------------------------------------

class BehavioralWindowExtractor:
    """
    Computes behavioral features for each event using a sliding window of
    PAST events only (strictly before the current event's timestamp).

    Grouping key: hostname (configurable to hostname+log_type).
    Window: configurable (default 5 minutes).

    Leakage prevention
    ------------------
    For training, events are sorted by timestamp and features for event E
    are derived only from events with timestamp < E.timestamp.
    For inference, a rolling event buffer per grouping key is maintained
    externally (see behavioral_state.py).

    Historical baseline
    -------------------
    fit_baseline() computes per-key mean/std of beh_total_events from training
    data. Used to compute beh_event_freq_delta (z-score).
    Call save_baseline() / load_baseline() to persist alongside model artifact.
    """

    def __init__(
        self,
        window_minutes: int = 5,
        grouping_keys: Tuple[str, ...] = ("hostname",),
        min_events_for_ratio: int = 1,
        rare_type_threshold: int = 5,
    ):
        """
        Args:
            window_minutes:       Size of the look-back window in minutes.
            grouping_keys:        Fields used to form the grouping key.
                                  Valid options: ("hostname",) or
                                  ("hostname", "log_type").
            min_events_for_ratio: Denominator minimum for ratio features;
                                  prevents division by zero.
            rare_type_threshold:  A (log_type, service) combo seen fewer than
                                  this many times in training is considered rare.
        """
        self.window_minutes = window_minutes
        self.window_delta = timedelta(minutes=window_minutes)
        self.grouping_keys = tuple(grouping_keys)
        self.min_events_for_ratio = min_events_for_ratio
        self.rare_type_threshold = rare_type_threshold

        # Populated by fit_baseline()
        self._baseline_counts: Dict[str, Dict[str, float]] = {}
        # {group_key → {"mean": float, "std": float}}

        # Common (log_type, service) pairs seen >= threshold times in training
        self._common_type_service: set = set()

        self._baseline_fitted = False

    # ── public interface ──────────────────────────────────────────────────────

    def fit_baseline(self, norm_events: List[Dict[str, Any]]) -> None:
        """
        Compute historical baseline statistics from training events.
        Must be called ONLY on training data — never on validation or test events.

        Computes:
        - Per-group mean and std of window event counts
        - Common (log_type, service) pairs for rare-event detection
        """
        if not norm_events:
            logger.warning("fit_baseline called with empty event list.")
            return

        # Sort by timestamp; events without timestamps go to the end
        events = sorted(
            norm_events,
            key=lambda e: e["timestamp_utc"] or datetime.max.replace(tzinfo=timezone.utc),
        )

        # Compute window counts per group to build baseline
        group_window_counts: Dict[str, List[int]] = defaultdict(list)
        group_buffers: Dict[str, List[datetime]] = defaultdict(list)

        for ev in events:
            if ev["timestamp_utc"] is None:
                continue
            gk = self._grouping_key(ev)
            t = ev["timestamp_utc"]
            buf = group_buffers[gk]

            # Evict old events
            cutoff = t - self.window_delta
            buf = [ts for ts in buf if ts >= cutoff]
            group_window_counts[gk].append(len(buf))
            buf.append(t)
            group_buffers[gk] = buf

        # Compute per-group mean/std
        self._baseline_counts = {}
        for gk, counts in group_window_counts.items():
            arr = np.array(counts, dtype=float)
            self._baseline_counts[gk] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        # Compute global fallback
        all_counts = [c for counts in group_window_counts.values() for c in counts]
        if all_counts:
            arr = np.array(all_counts, dtype=float)
            self._baseline_counts["__global__"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        # Common (log_type, service) combos
        type_service_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for ev in events:
            type_service_counts[(ev["log_type"], ev["service"])] += 1
        self._common_type_service = {
            k for k, cnt in type_service_counts.items()
            if cnt >= self.rare_type_threshold
        }

        self._baseline_fitted = True
        logger.info(
            f"Baseline fitted: {len(self._baseline_counts) - 1} groups, "
            f"{len(self._common_type_service)} common type-service pairs, "
            f"{len(events):,} training events"
        )

    def extract_batch(
        self,
        norm_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compute behavioral features for each event in sorted order.
        Uses only events BEFORE the current event (past-only window).

        For events without a timestamp, all behavioral features are set to
        0 / cold-start = 1.

        Returns a list of feature dicts (same length as norm_events).
        """
        # Sort by timestamp for correct past-only window computation
        indexed = list(enumerate(norm_events))
        valid   = [(i, e) for i, e in indexed if e["timestamp_utc"] is not None]
        invalid = [(i, e) for i, e in indexed if e["timestamp_utc"] is None]

        # Per-group rolling buffer: list of (timestamp, event_dict)
        group_buffers: Dict[str, List[Tuple[datetime, Dict]]] = defaultdict(list)

        results: List[Optional[Dict[str, Any]]] = [None] * len(norm_events)

        # Sort valid events by time
        valid.sort(key=lambda x: x[1]["timestamp_utc"])

        for orig_idx, ev in valid:
            gk = self._grouping_key(ev)
            t  = ev["timestamp_utc"]
            buf = group_buffers[gk]

            # Evict events outside the window
            cutoff = t - self.window_delta
            buf = [(ts, e) for ts, e in buf if ts >= cutoff]

            # Compute features on past-only buffer (before appending current)
            feat = self._compute_features(ev, buf)
            results[orig_idx] = feat

            # Append current event to buffer
            buf.append((t, ev))
            group_buffers[gk] = buf

        # Fill in cold-start features for events without timestamps
        for orig_idx, ev in invalid:
            results[orig_idx] = self._cold_start_features()

        return results  # type: ignore[return-value]

    def extract_one(
        self,
        norm_event: Dict[str, Any],
        event_buffer: List[Tuple[datetime, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Compute behavioral features for a single event using a pre-filtered
        event buffer (from behavioral_state.py at inference time).

        Args:
            norm_event:   Normalized event dict (current event, NOT in buffer).
            event_buffer: List of (timestamp, event_dict) tuples for the same
                          grouping key, already filtered to the window period.
                          Must contain only PAST events.

        Returns:
            Behavioral feature dict.
        """
        if norm_event["timestamp_utc"] is None:
            return self._cold_start_features()
        return self._compute_features(norm_event, event_buffer)

    # ── baseline persistence ──────────────────────────────────────────────────

    def save_baseline(self, path: Path) -> None:
        """Serialize baseline statistics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window_minutes":        self.window_minutes,
            "grouping_keys":         list(self.grouping_keys),
            "rare_type_threshold":   self.rare_type_threshold,
            "baseline_counts":       self._baseline_counts,
            "common_type_service":   [list(k) for k in self._common_type_service],
            "baseline_fitted":       self._baseline_fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Baseline saved to {path}")

    def load_baseline(self, path: Path) -> None:
        """Load baseline statistics from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.window_minutes       = payload["window_minutes"]
        self.window_delta         = timedelta(minutes=self.window_minutes)
        self.grouping_keys        = tuple(payload["grouping_keys"])
        self.rare_type_threshold  = payload["rare_type_threshold"]
        self._baseline_counts     = payload["baseline_counts"]
        self._common_type_service = {tuple(k) for k in payload["common_type_service"]}
        self._baseline_fitted     = payload["baseline_fitted"]
        logger.info(f"Baseline loaded from {path}")

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for serialization into model metadata."""
        return {
            "window_minutes":       self.window_minutes,
            "grouping_keys":        list(self.grouping_keys),
            "min_events_for_ratio": self.min_events_for_ratio,
            "rare_type_threshold":  self.rare_type_threshold,
            "baseline_fitted":      self._baseline_fitted,
            "n_baseline_groups":    len(self._baseline_counts),
            "n_common_type_service": len(self._common_type_service),
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _grouping_key(self, ev: Dict[str, Any]) -> str:
        return "__".join(str(ev.get(k, "unknown")) for k in self.grouping_keys)

    def _compute_features(
        self,
        ev: Dict[str, Any],
        buf: List[Tuple[datetime, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Compute all behavioral features from a window buffer (past events only).
        The current event (ev) is NOT included in the buffer during computation.
        """
        n = len(buf)
        gk = self._grouping_key(ev)

        if n == 0:
            return self._cold_start_features()

        events_in_window = [e for _, e in buf]

        # Auth counts
        fail_count    = sum(1 for e in events_in_window if e["auth_failed"])
        success_count = sum(1 for e in events_in_window if e["auth_success"])
        denom = max(fail_count + success_count, self.min_events_for_ratio)
        fail_ratio = round(fail_count / denom, 4)

        # Error / warning
        error_count   = sum(1 for e in events_in_window if e["has_error"])
        warning_count = sum(1 for e in events_in_window if e["has_warning"])

        # Distinct services
        distinct_services = len({e["service"] for e in events_in_window})

        # Sudo / privilege
        sudo_count  = sum(1 for e in events_in_window if e["is_sudo_su"])
        priv_count  = sum(1 for e in events_in_window if e["is_privilege"])

        # Distinct source IPs (non-null)
        ips = {e["source_ip"] for e in events_in_window if e["source_ip"]}
        distinct_ips = len(ips)

        # Rare-event ratio
        if self._baseline_fitted and self._common_type_service:
            rare = sum(
                1 for e in events_in_window
                if (e["log_type"], e["service"]) not in self._common_type_service
            )
            rare_ratio = round(rare / max(n, 1), 4)
        else:
            rare_ratio = 0.0

        # Event-frequency z-score vs historical baseline
        baseline = self._baseline_counts.get(gk) or self._baseline_counts.get("__global__")
        if baseline and baseline["std"] > 0:
            freq_delta = round((n - baseline["mean"]) / baseline["std"], 4)
            cold_start = 0
        elif baseline:
            # Std = 0 (constant activity) — delta is 0
            freq_delta = 0.0
            cold_start = 0
        else:
            # No baseline for this group
            freq_delta = 0.0
            cold_start = 1

        return {
            "beh_total_events":      n,
            "beh_auth_fail_count":   fail_count,
            "beh_auth_success_count": success_count,
            "beh_fail_success_ratio": fail_ratio,
            "beh_error_count":       error_count,
            "beh_warning_count":     warning_count,
            "beh_distinct_services": distinct_services,
            "beh_sudo_su_count":     sudo_count,
            "beh_privilege_count":   priv_count,
            "beh_distinct_ips":      distinct_ips,
            "beh_rare_event_ratio":  rare_ratio,
            "beh_event_freq_delta":  freq_delta,
            "beh_cold_start":        cold_start,
        }

    def _cold_start_features(self) -> Dict[str, Any]:
        """Return zero-valued features with cold_start flag set."""
        return {
            "beh_total_events":       0,
            "beh_auth_fail_count":    0,
            "beh_auth_success_count": 0,
            "beh_fail_success_ratio": 0.0,
            "beh_error_count":        0,
            "beh_warning_count":      0,
            "beh_distinct_services":  0,
            "beh_sudo_su_count":      0,
            "beh_privilege_count":    0,
            "beh_distinct_ips":       0,
            "beh_rare_event_ratio":   0.0,
            "beh_event_freq_delta":   0.0,
            "beh_cold_start":         1,
        }


# ---------------------------------------------------------------------------
# BehavioralFeatureSchema
# ---------------------------------------------------------------------------

@dataclass
class BehavioralFeatureSchema:
    """
    Persisted alongside each model artifact (.joblib) to guarantee that the
    same feature names, ordering, and window configuration are used at both
    training time and inference time.

    Saved as a companion JSON file: <artifact_path>.behavioral_schema.json
    """
    version: str                        # schema version, e.g. "v1"
    window_minutes: int
    grouping_keys: List[str]
    message_feature_names: List[str]    # from ait_label_config.ALL_FEATURES
    behavioral_feature_names: List[str] # BEHAVIORAL_FEATURE_NAMES
    all_feature_names: List[str]        # message + behavioral (in order)
    extractor_config: Dict[str, Any]    # BehavioralWindowExtractor.get_config()
    baseline_stats_path: Optional[str]  # path to saved baseline JSON
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    def save(self, path: Path) -> None:
        """Save schema to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"BehavioralFeatureSchema saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BehavioralFeatureSchema":
        """Load schema from JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def validate_features(self, feature_names: List[str]) -> Tuple[bool, str]:
        """
        Check that the provided feature name list matches this schema exactly.
        Returns (ok: bool, error_message: str).
        """
        if feature_names == self.all_feature_names:
            return True, ""
        # Detailed diff
        missing = [f for f in self.all_feature_names if f not in feature_names]
        extra   = [f for f in feature_names if f not in self.all_feature_names]
        order_ok = (
            [f for f in feature_names if f in self.all_feature_names]
            == [f for f in self.all_feature_names if f in feature_names]
        )
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        if not order_ok:
            parts.append("ordering mismatch")
        return False, "; ".join(parts)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_behavioral_dataframe(
    norm_events: List[Dict[str, Any]],
    beh_features: List[Dict[str, Any]],
    message_feature_cols: List[str],
    all_behavioral_feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Combine message-level features and behavioral features into a single
    DataFrame suitable for preprocessing and model training.

    Args:
        norm_events:              List of normalized event dicts.
        beh_features:             List of behavioral feature dicts (same length).
        message_feature_cols:     Ordered list of message-level feature names.
        all_behavioral_feature_names: Ordered list of behavioral feature names.
                                  Defaults to BEHAVIORAL_FEATURE_NAMES.

    Returns:
        pd.DataFrame with columns: message_feature_cols + behavioral_feature_names
    """
    if all_behavioral_feature_names is None:
        all_behavioral_feature_names = BEHAVIORAL_FEATURE_NAMES

    assert len(norm_events) == len(beh_features), (
        f"Length mismatch: norm_events={len(norm_events)}, beh_features={len(beh_features)}"
    )

    rows = []
    for ev, beh in zip(norm_events, beh_features):
        row = {}
        # Message-level features are stored under their canonical names in the
        # normalized event; fall back to 0 / "unknown" for missing fields.
        for col in message_feature_cols:
            row[col] = ev.get(col, 0)
        # Behavioral features
        for col in all_behavioral_feature_names:
            row[col] = beh.get(col, 0)
        rows.append(row)

    return pd.DataFrame(rows, columns=message_feature_cols + all_behavioral_feature_names)
