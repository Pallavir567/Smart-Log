"""
Behavioral State Buffer for Smart-Log Inference
================================================
Thread-safe, in-memory sliding-window event buffer for live inference.

At inference time, the BehavioralWindowExtractor needs a buffer of recent
past events per grouping key to compute window-based behavioral features.
This module maintains that buffer per-host as events arrive.

Design decisions
----------------
- Events are stored as (timestamp_utc, normalized_event_dict) tuples.
- The buffer is capped at MAX_EVENTS_PER_KEY entries per grouping key to
  bound memory usage.
- State is NOT persisted between server restarts. On restart, all hosts
  start in cold-start mode (beh_cold_start=1) and features degrade
  gracefully until enough history accumulates.
- Expired events (older than MAX_BUFFER_MINUTES) are evicted on each
  insertion.
- Thread-safety: a threading.Lock is held per key during read+write.

Usage
-----
    from backend.services.behavioral_state import BehavioralEventBuffer

    buffer = BehavioralEventBuffer.get_instance()

    # On each incoming event:
    norm_event = EventNormalizer.normalize(parsed, log_type)
    gk = extractor._grouping_key(norm_event)
    window = buffer.get_window(gk, norm_event["timestamp_utc"], window_minutes=5)
    beh_features = extractor.extract_one(norm_event, window)
    buffer.add_event(gk, norm_event)
"""

from __future__ import annotations

import threading
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("smartlog.behavioral_state")

# Default limits
MAX_EVENTS_PER_KEY: int = 2000       # max events stored per grouping key
MAX_BUFFER_MINUTES: int = 60         # events older than this are evicted


class BehavioralEventBuffer:
    """
    Singleton thread-safe event buffer for maintaining per-host behavioral
    history during live inference.

    Each entry in the buffer is a tuple:
        (timestamp_utc: datetime, event: Dict[str, Any])

    Events without a valid timestamp_utc are stored with timestamp=epoch
    and will not contribute to time-windowed features.
    """

    _instance: Optional[BehavioralEventBuffer] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        max_events_per_key: int = MAX_EVENTS_PER_KEY,
        max_buffer_minutes: int = MAX_BUFFER_MINUTES,
    ):
        self._max_events = max_events_per_key
        self._max_age = timedelta(minutes=max_buffer_minutes)
        # {grouping_key: deque of (timestamp_utc, event_dict)}
        self._buffers: Dict[str, Deque[Tuple[datetime, Dict[str, Any]]]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        self._per_key_lock: Dict[str, threading.Lock] = defaultdict(threading.Lock)

    @classmethod
    def get_instance(cls) -> BehavioralEventBuffer:
        """Return the process-wide singleton buffer."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ── public interface ──────────────────────────────────────────────────────

    def add_event(
        self,
        grouping_key: str,
        norm_event: Dict[str, Any],
    ) -> None:
        """
        Add a normalized event to the buffer for the given grouping key.
        Evicts events older than max_buffer_minutes before inserting.

        Args:
            grouping_key: String key (e.g. "host1" or "host1__auth")
            norm_event:   Normalized event dict from EventNormalizer.normalize()
        """
        ts = norm_event.get("timestamp_utc")
        if ts is None:
            # Store with epoch so it's always evicted in future get_window calls
            ts = datetime.fromtimestamp(0, tz=timezone.utc)
        elif not isinstance(ts, datetime):
            ts = datetime.fromtimestamp(0, tz=timezone.utc)

        entry: Tuple[datetime, Dict[str, Any]] = (ts, norm_event)

        with self._per_key_lock[grouping_key]:
            buf = self._buffers[grouping_key]
            # Evict expired events
            now = datetime.now(tz=timezone.utc)
            cutoff = now - self._max_age
            while buf and buf[0][0] < cutoff:
                buf.popleft()
            buf.append(entry)

    def get_window(
        self,
        grouping_key: str,
        before_time: datetime,
        window_minutes: int = 5,
    ) -> List[Tuple[datetime, Dict[str, Any]]]:
        """
        Return past events in the window [before_time - window_minutes, before_time)
        for the given grouping key.

        The current event itself should NOT be in the buffer yet when this is called.
        Events are returned in chronological order.

        Args:
            grouping_key:   Grouping key (hostname or hostname+log_type)
            before_time:    The current event's timestamp (exclusive upper bound)
            window_minutes: How many minutes to look back

        Returns:
            List of (timestamp_utc, event_dict) tuples, sorted ascending by time.
            Returns empty list if no history or cold-start.
        """
        if before_time is None:
            return []

        # Ensure timezone-aware
        if before_time.tzinfo is None:
            before_time = before_time.replace(tzinfo=timezone.utc)

        cutoff = before_time - timedelta(minutes=window_minutes)

        with self._per_key_lock[grouping_key]:
            buf = self._buffers[grouping_key]
            window = [
                (ts, ev) for ts, ev in buf
                if cutoff <= ts < before_time
            ]

        return sorted(window, key=lambda x: x[0])

    def get_stats(self) -> Dict[str, Any]:
        """
        Return diagnostic statistics about current buffer state.
        Useful for the /api/system/status endpoint.
        """
        total_events = sum(len(buf) for buf in self._buffers.values())
        return {
            "tracked_groups":    len(self._buffers),
            "total_buffered":    total_events,
            "max_events_per_key": self._max_events,
            "max_buffer_minutes": self._max_age.seconds // 60,
        }

    def clear(self, grouping_key: Optional[str] = None) -> None:
        """
        Clear the buffer for a specific grouping key, or all keys if None.
        Useful for testing and manual resets.
        """
        if grouping_key is not None:
            with self._per_key_lock[grouping_key]:
                if grouping_key in self._buffers:
                    self._buffers[grouping_key].clear()
        else:
            # Clear all — take all per-key locks
            for key in list(self._buffers.keys()):
                with self._per_key_lock[key]:
                    self._buffers[key].clear()
            logger.info("BehavioralEventBuffer fully cleared.")
