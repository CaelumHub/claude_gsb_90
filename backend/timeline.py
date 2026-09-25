"""
timeline.py
-----------
Pure temporal aggregation for graph evolution.

The module deliberately contains no HTTP or DOM code.  Its input is a stable
event snapshot collected from the current graph data:

* ``node_events`` -- ``(id, first_seen_ms, source, name, registered_at_ms)``
* ``edge_events`` -- ``(u, v, created_at_ms)``

and its output is a JSON-serialisable, reusable aggregation result.  Presentation
code (a frontend chart or another API client) can consume the same result
without re-implementing time semantics.

Time rules
~~~~~~~~~~
* Every timestamp is Unix epoch time in milliseconds.
* Buckets are half-open: ``[bucket_start, next_bucket_start)``.
* Selected intervals are closed: ``start <= event_time <= end``.
* ``tz_offset_minutes`` follows JavaScript's ``Date.getTimezoneOffset()``
  convention reversed in sign: China is ``480``, UTC is ``0``.
* An undirected relationship is counted once using its normalised endpoint key
  ``(min(u, v), max(u, v))``; duplicate records keep the earliest timestamp.
* Graph node birth time is the earliest of user registration time and the
  earliest incident relationship time.  This keeps isolated registered users
  and edge-only nodes in the same consistent node universe.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

GRANULARITIES = ("auto", "hour", "day", "week", "month", "year")
BUCKET_ORDER = ("hour", "day", "week", "month", "year")

HOUR_MS = 3_600_000
DAY_MS = 86_400_000

NodeEvent = Tuple[int, int, str, str, Optional[int]]
EdgeEvent = Tuple[int, int, int]


def valid_ms(value) -> bool:
    """Return true for a positive integer-compatible millisecond timestamp."""
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return False
    # Reject zero/negative values rather than silently placing them at epoch.
    return ts > 0


def normalise_node_events(events: Iterable) -> List[NodeEvent]:
    """Keep the earliest valid event for each node, with stable tie-breaking."""
    by_id: Dict[int, NodeEvent] = {}
    for raw in events:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            node_id = int(raw[0])
            first_seen = int(raw[1])
        except (TypeError, ValueError):
            continue
        if not valid_ms(first_seen):
            continue
        source = str(raw[2]) if len(raw) > 2 and raw[2] else "unknown"
        name = str(raw[3]) if len(raw) > 3 and raw[3] is not None else str(node_id)
        registered_at: Optional[int] = None
        if len(raw) > 4 and valid_ms(raw[4]):
            registered_at = int(raw[4])
        event = (node_id, first_seen, source, name, registered_at)
        old = by_id.get(node_id)
        if old is None or (first_seen, source) < (old[1], old[2]):
            by_id[node_id] = event
    return [by_id[k] for k in sorted(by_id)]


def normalise_edge_events(events: Iterable) -> Tuple[List[EdgeEvent], int]:
    """Normalise undirected edges and retain the earliest creation time."""
    by_key: Dict[Tuple[int, int], int] = {}
    duplicates = 0
    invalid = 0
    for raw in events:
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            invalid += 1
            continue
        try:
            u, v, ts = int(raw[0]), int(raw[1]), int(raw[2])
        except (TypeError, ValueError):
            invalid += 1
            continue
        if u == v or not valid_ms(ts):
            invalid += 1
            continue
        if u > v:
            u, v = v, u
        key = (u, v)
        if key in by_key:
            duplicates += 1
            if ts < by_key[key]:
                by_key[key] = ts
        else:
            by_key[key] = ts
    edges = [(u, v, ts) for (u, v), ts in by_key.items()]
    edges.sort(key=lambda e: (e[2], e[0], e[1]))
    return edges, duplicates + invalid


def choose_granularity(min_ts: int, max_ts: int) -> str:
    """Choose a bucket size that keeps the full curve compact and readable."""
    span = max(0, max_ts - min_ts)
    if span <= 2 * DAY_MS:
        return "hour"
    if span <= 180 * DAY_MS:
        return "day"
    if span <= 2 * 365 * DAY_MS:
        return "week"
    if span <= 8 * 365 * DAY_MS:
        return "month"
    return "year"


def floor_bucket(ts: int, granularity: str, tz_offset_minutes: int) -> int:
    """Floor an epoch millisecond timestamp to a local calendar bucket."""
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    dt = datetime.fromtimestamp(ts / 1000.0, tz=tz)
    if granularity == "hour":
        dt = dt.replace(minute=0, second=0, microsecond=0)
    elif granularity == "day":
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif granularity == "week":
        day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day = day - timedelta(days=dt.weekday())  # Python's Monday is 0.
        dt = day
    elif granularity == "month":
        dt = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif granularity == "year":
        dt = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"unsupported granularity: {granularity}")
    return int(dt.timestamp() * 1000)


def next_bucket_start(start: int, granularity: str, tz_offset_minutes: int) -> int:
    if granularity == "hour":
        return start + HOUR_MS
    if granularity == "day":
        return start + DAY_MS
    if granularity == "week":
        return start + 7 * DAY_MS
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    dt = datetime.fromtimestamp(start / 1000.0, tz=tz)
    if granularity == "month":
        year = dt.year + (1 if dt.month == 12 else 0)
        month = 1 if dt.month == 12 else dt.month + 1
        dt = dt.replace(year=year, month=month, day=1)
    elif granularity == "year":
        dt = dt.replace(year=dt.year + 1, month=1, day=1)
    else:
        raise ValueError(f"unsupported granularity: {granularity}")
    return int(dt.timestamp() * 1000)


def bucket_label(start: int, granularity: str, tz_offset_minutes: int) -> str:
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    dt = datetime.fromtimestamp(start / 1000.0, tz=tz)
    if granularity == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    if granularity == "day":
        return dt.strftime("%Y-%m-%d")
    if granularity == "week":
        return dt.strftime("%Y-%m-%d") + " 周起"
    if granularity == "month":
        return dt.strftime("%Y-%m")
    return str(dt.year)


def _in_range(ts: int, start: Optional[int], end: Optional[int]) -> bool:
    return (start is None or ts >= start) and (end is None or ts <= end)


def _group_events(
    node_events: Sequence[NodeEvent],
    registration_events: Sequence[NodeEvent],
    edge_events: Sequence[EdgeEvent],
    granularity: str,
    tz_offset_minutes: int,
) -> Dict[int, Dict[str, int]]:
    groups: Dict[int, Dict[str, int]] = {}

    def add(ts: int, key: str) -> None:
        b = floor_bucket(ts, granularity, tz_offset_minutes)
        bucket = groups.setdefault(
            b, {"new_nodes": 0, "new_registered_users": 0, "new_edges": 0}
        )
        bucket[key] += 1

    for _id, ts, _source, _name, _registered_at in node_events:
        add(ts, "new_nodes")
    for _id, _ts, _source, _name, registered_at in registration_events:
        add(registered_at, "new_registered_users")
    for _u, _v, ts in edge_events:
        add(ts, "new_edges")
    return groups


def _make_series(
    node_events: Sequence[NodeEvent],
    registration_events: Sequence[NodeEvent],
    edge_events: Sequence[EdgeEvent],
    granularity: str,
    tz_offset_minutes: int,
    before_nodes: int,
    before_edges: int,
) -> List[dict]:
    timed_nodes = list(node_events)
    timed_registrations = list(registration_events)
    timed_edges = list(edge_events)
    if not timed_nodes and not timed_registrations and not timed_edges:
        return []

    all_ts = (
        [e[1] for e in timed_nodes]
        + [e[4] for e in timed_registrations if e[4]]
        + [e[2] for e in timed_edges]
    )
    min_ts = min(all_ts)
    max_ts = max(all_ts)
    groups = _group_events(
        timed_nodes,
        timed_registrations,
        timed_edges,
        granularity,
        tz_offset_minutes,
    )

    series = []
    cursor = floor_bucket(min_ts, granularity, tz_offset_minutes)
    last = floor_bucket(max_ts, granularity, tz_offset_minutes)
    nodes = before_nodes
    edges = before_edges
    # Bound the loop defensively; calendar calculations are otherwise monotonic.
    for _ in range(10_000):
        nxt = next_bucket_start(cursor, granularity, tz_offset_minutes)
        delta = groups.get(cursor, {"new_nodes": 0, "new_registered_users": 0, "new_edges": 0})
        nodes += delta["new_nodes"]
        edges += delta["new_edges"]
        series.append(
            {
                "bucket_start": cursor,
                "bucket_end": nxt,
                "label": bucket_label(cursor, granularity, tz_offset_minutes),
                "new_nodes": delta["new_nodes"],
                "new_registered_users": delta["new_registered_users"],
                "new_edges": delta["new_edges"],
                "nodes": nodes,
                "edges": edges,
            }
        )
        if cursor >= last:
            break
        cursor = nxt
    return series


def _markers(
    node_events: Sequence[NodeEvent],
    edge_events: Sequence[EdgeEvent],
    series: Sequence[dict],
    start: Optional[int],
    end: Optional[int],
) -> List[dict]:
    markers: List[dict] = []
    registrations = sorted(
        [(e[4], e[0], e[3]) for e in node_events if e[4]],
        key=lambda x: (x[0], x[1]),
    )
    range_registrations = sorted(
        [(e[4], e[0], e[3]) for e in node_events if e[4] and _in_range(e[4], start, end)],
        key=lambda x: (x[0], x[1]),
    )
    if range_registrations:
        ts, uid, name = range_registrations[0]
        is_first = bool(registrations and range_registrations[0] == registrations[0])
        nodes_at_time = sum(1 for e in node_events if e[1] <= ts)
        markers.append(
            {
                "type": "first_user" if is_first else "range_first_user",
                "time": ts,
                "node_count": nodes_at_time,
                "title": ("首位新增用户" if is_first else "区间首位新增用户") + f"：{name} (#{uid})",
            }
        )
    range_edges = [e for e in edge_events if _in_range(e[2], start, end)]
    if range_edges:
        u, v, ts = range_edges[0]
        is_first = edge_events[0][2] == ts
        markers.append(
            {
                "type": "first_edge" if is_first else "range_first_edge",
                "time": ts,
                "edge_count": sum(1 for e in edge_events if e[2] <= ts),
                "title": ("首条新增关系" if is_first else "区间首条新增关系") + f"：{u} ↔ {v}",
            }
        )
    if series:
        peak_users = max(series, key=lambda b: (b["new_registered_users"], -b["bucket_start"]))
        peak_nodes = max(series, key=lambda b: (b["new_nodes"], -b["bucket_start"]))
        peak_edges = max(series, key=lambda b: (b["new_edges"], -b["bucket_start"]))
        if peak_users["new_registered_users"] > 0:
            markers.append(
                {
                    "type": "peak_users",
                    "time": peak_users["bucket_start"],
                    "value": peak_users["new_registered_users"],
                    "node_count": peak_users["nodes"],
                    "title": f"新增用户峰值：{peak_users['new_registered_users']} 人",
                }
            )
        if peak_nodes["new_nodes"] > 0:
            markers.append(
                {
                    "type": "peak_nodes",
                    "time": peak_nodes["bucket_start"],
                    "value": peak_nodes["new_nodes"],
                    "node_count": peak_nodes["nodes"],
                    "title": f"新增节点峰值：{peak_nodes['new_nodes']} 个",
                }
            )
        if peak_edges["new_edges"] > 0:
            markers.append(
                {
                    "type": "peak_edges",
                    "time": peak_edges["bucket_start"],
                    "value": peak_edges["new_edges"],
                    "edge_count": peak_edges["edges"],
                    "title": f"新增关系峰值：{peak_edges['new_edges']} 条",
                }
            )
    markers.sort(key=lambda m: (m["time"], m["type"]))
    return markers


def build_timeline(
    raw_node_events: Iterable,
    raw_edge_events: Iterable,
    start: Optional[int] = None,
    end: Optional[int] = None,
    granularity: str = "auto",
    tz_offset_minutes: int = 0,
    detail_limit: int = 100,
) -> dict:
    """Build a complete, deterministic timeline aggregation.

    The full entity universe is always used to calculate ``before_*`` and final
    counts; the series and detail lists only contain events in the selected
    closed interval.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"unsupported granularity: {granularity}")
    detail_limit = max(0, min(int(detail_limit), 1000))

    all_nodes = normalise_node_events(raw_node_events)
    all_edges, edge_records_skipped = normalise_edge_events(raw_edge_events)

    if start is not None and end is not None and start > end:
        raise ValueError("start must not be later than end")

    timed_node_ts = [e[1] for e in all_nodes]
    timed_edge_ts = [e[2] for e in all_edges]
    all_ts = timed_node_ts + timed_edge_ts
    data_start = min(all_ts) if all_ts else None
    data_end = max(all_ts) if all_ts else None

    selected_nodes = [e for e in all_nodes if _in_range(e[1], start, end)]
    selected_registrations = [
        e for e in all_nodes if e[4] and _in_range(e[4], start, end)
    ]
    selected_edges = [e for e in all_edges if _in_range(e[2], start, end)]

    if granularity == "auto":
        auto_ts = (
            [e[1] for e in selected_nodes]
            + [e[4] for e in selected_registrations if e[4]]
            + [e[2] for e in selected_edges]
        )
        if auto_ts:
            resolved = choose_granularity(min(auto_ts), max(auto_ts))
        else:
            resolved = "day"
    else:
        resolved = granularity

    before_nodes = sum(1 for e in all_nodes if start is not None and e[1] < start)
    before_edges = sum(1 for e in all_edges if start is not None and e[2] < start)
    before_users = sum(
        1 for e in all_nodes if e[4] and start is not None and e[4] < start
    )

    series = _make_series(
        selected_nodes,
        selected_registrations,
        selected_edges,
        resolved,
        tz_offset_minutes,
        before_nodes,
        before_edges,
    )

    new_user_events = sorted(selected_registrations, key=lambda e: (e[4], e[0]))
    # Graph node births; registration-only users and first-seen-via-edge nodes
    # are both represented here.
    new_node_details = [
        {
            "id": e[0],
            "time": e[1],
            "source": e[2],
            "name": e[3],
            "registered_at": e[4],
        }
        for e in sorted(selected_nodes, key=lambda e: (e[1], e[0]))
    ]
    new_user_details = [
        {
            "id": e[0],
            "time": e[4],
            "name": e[3],
            "first_seen_at": e[1],
            "source": e[2],
        }
        for e in new_user_events
    ]
    new_edge_details = [
        {"from": u, "to": v, "time": ts}
        for u, v, ts in selected_edges[:detail_limit]
    ]

    registered_users = sum(1 for e in all_nodes if e[4])
    result = {
        "time_unit": "epoch_ms",
        "granularity": resolved,
        "requested_granularity": granularity,
        "timezone_offset_minutes": tz_offset_minutes,
        "data_start": data_start,
        "data_end": data_end,
        "interval": {
            "start": start,
            "end": end,
            "before_node_count": before_nodes,
            "before_edge_count": before_edges,
            "before_registered_user_count": before_users,
            "new_node_count": len(selected_nodes),
            "new_registered_user_count": len(new_user_events),
            "new_edge_count": len(selected_edges),
            "end_node_count": before_nodes + len(selected_nodes),
            "end_edge_count": before_edges + len(selected_edges),
        },
        "summary": {
            "total_nodes": len(all_nodes),
            "total_registered_users": registered_users,
            "total_edge_only_nodes": sum(1 for e in all_nodes if not e[4]),
            "total_edges": len(all_edges),
            "edge_records_skipped": edge_records_skipped,
        },
        "series": series,
        "markers": _markers(all_nodes, all_edges, series, start, end),
        "details": {
            "limit": detail_limit,
            "new_nodes": new_node_details[:detail_limit],
            "new_registered_users": new_user_details[:detail_limit],
            "new_edges": new_edge_details,
        },
    }
    return result
