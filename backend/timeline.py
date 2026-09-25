"""
timeline.py
-----------
图演化时间线 —— 事件采集与时间聚合。

设计要点
--------
* **聚合与展示分离**：本模块只做「事件流 → 聚合结果」的纯计算，不感知
  HTTP / HTML。service 层缓存事件流，并把它喂给多个展示入口（累计曲线、
  区间增量、关键时间点），同一份聚合结果被反复复用。
* **时间口径统一**：所有时间戳都是 Unix 毫秒（int）；桶边界按 UTC 对齐
  （day = UTC 00:00，week = ISO 周一 00:00 UTC，month = UTC 每月 1 日）；
  区间一律左闭右开 ``[start, end)``。前端直接使用这里生成的桶标签，
  避免各自格式化造成的口径漂移。
* **增量准确、曲线与真实数据一致**：

  - 边按无向对 ``(min, max)`` 去重并取最早时间戳，因此曲线终点边数
    恒等于 ``load_full_graph().edge_count``；
  - 节点取 users.json 与边端点的并集，首次出现时间取两者中较早者；
  - 区间增量用 ``bisect`` 在有序事件流上精确计数，与曲线同源同口径，
    二者必然吻合。

* **结果稳定可复现**：事件排序键为 ``(ts, id)``，桶边界只依赖 UTC 日历，
  相同输入必然产生相同输出（``generated_at`` 除外）。
"""

from __future__ import annotations

import bisect
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from . import config
except ImportError:  # pragma: no cover
    import config

DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
# 1970-01-05 00:00:00 UTC 是周一，作为 ISO 周桶的对齐锚点。
WEEK_ANCHOR_MS = 4 * DAY_MS

GRANULARITIES = ("day", "week", "month")


# ---------------------------------------------------------------------------
# 时间桶（UTC 对齐）
# ---------------------------------------------------------------------------
def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _month_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def bucket_start(ts: int, granularity: str) -> int:
    """返回 ``ts`` 所在桶的起始毫秒（UTC 对齐）。"""
    if granularity == "day":
        return (int(ts) // DAY_MS) * DAY_MS
    if granularity == "week":
        return WEEK_ANCHOR_MS + ((int(ts) - WEEK_ANCHOR_MS) // WEEK_MS) * WEEK_MS
    if granularity == "month":
        d = _utc(int(ts))
        return _month_ms(d.year, d.month)
    raise ValueError(f"未知粒度: {granularity}")


def bucket_next(start: int, granularity: str) -> int:
    """返回当前桶的下一个桶起始毫秒（开区间端点）。"""
    if granularity == "day":
        return start + DAY_MS
    if granularity == "week":
        return start + WEEK_MS
    if granularity == "month":
        d = _utc(start)
        year, month = d.year, d.month + 1
        if month > 12:
            year, month = year + 1, 1
        return _month_ms(year, month)
    raise ValueError(f"未知粒度: {granularity}")


def bucket_label(start: int, granularity: str) -> str:
    """桶的展示标签（UTC 口径，前后端共用）。"""
    d = _utc(start)
    if granularity == "day":
        return d.strftime("%Y-%m-%d")
    if granularity == "week":
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    if granularity == "month":
        return d.strftime("%Y-%m")
    return str(start)


def fmt_dt(ms: int) -> str:
    """事件时刻的展示格式（UTC，精确到分）。"""
    return _utc(int(ms)).strftime("%Y-%m-%d %H:%M")


def choose_granularity(span_ms: int) -> str:
    """``auto`` 粒度：按时间跨度选择，保证桶数量在可展示的范围内。"""
    if span_ms <= 62 * DAY_MS:
        return "day"
    if span_ms <= 3 * 365 * DAY_MS:
        return "week"
    return "month"


def _check_granularity(granularity: str) -> str:
    if granularity not in GRANULARITIES:
        raise ValueError(f"未知粒度: {granularity}（可选: auto/day/week/month）")
    return granularity


# ---------------------------------------------------------------------------
# 事件流构建（纯函数，便于测试与复用）
# ---------------------------------------------------------------------------
def events_from(users: Dict[int, int], edges: List[Tuple[int, int, int]]) -> dict:
    """从原始输入构建事件流。

    参数
    ----
    users : ``{uid: created_ms}``，``0``/``None`` 表示时间未知。
    edges : ``[(u, v, ts), ...]``，允许重复与反向；按无向对去重并取最早
            有效时间戳；``ts <= 0`` 视为时间未知；自环忽略（与 Graph 一致）。

    返回的事件流是后续所有聚合的唯一数据源，排序确定、可复现。
    """
    user_ts: Dict[int, int] = {}
    users_without_time = 0
    for uid, ts in users.items():
        ts = int(ts or 0)
        if ts > 0:
            user_ts[int(uid)] = ts
        else:
            users_without_time += 1

    edge_first: Dict[Tuple[int, int], int] = {}
    endpoint_first: Dict[int, int] = {}
    endpoints_all = set()
    for u, v, ts in edges:
        u, v, ts = int(u), int(v), int(ts or 0)
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        endpoints_all.add(u)
        endpoints_all.add(v)
        if ts > 0:
            prev = edge_first.get(key, 0)
            if prev <= 0 or ts < prev:
                edge_first[key] = ts
            for n in (u, v):
                p = endpoint_first.get(n)
                if p is None or ts < p:
                    endpoint_first[n] = ts
        else:
            edge_first.setdefault(key, 0)

    edges_without_time = sum(1 for t in edge_first.values() if t <= 0)

    # 节点首次出现 = min(用户创建时间, 首次作为边端点的时间)。
    node_first: Dict[int, int] = dict(endpoint_first)
    for uid, ts in user_ts.items():
        p = node_first.get(uid)
        if p is None or ts < p:
            node_first[uid] = ts

    nodes_total = len(set(users) | endpoints_all)
    edge_events = sorted(
        (ts, a, b) for (a, b), ts in edge_first.items() if ts > 0
    )
    return {
        "user_ts": user_ts,               # {uid: ms} 用户创建事件
        "edge_events": edge_events,       # [(ms, a, b)] 去重后的边事件，按 (ts,a,b) 排序
        "node_first": node_first,         # {uid: ms} 节点首次出现
        "users_total": len(users),
        "users_without_time": users_without_time,
        "edges_total": len(edge_first),
        "edges_without_time": edges_without_time,
        "nodes_total": nodes_total,
        "nodes_without_time": nodes_total - len(node_first),
    }


def collect_events(store) -> dict:
    """从存储层采集事件流（磁盘 IO 集中在这里，结果由 service 缓存复用）。"""
    users = {}
    for uid, rec in store.load_users().items():
        ts = rec.get("created_at_ms") or rec.get("created_at") or 0
        users[int(uid)] = int(ts or 0)
    edges = [(u, v, ts) for u, v, _w, ts in store.iter_all_edges_with_ts()]
    return events_from(users, edges)


# ---------------------------------------------------------------------------
# 聚合（纯函数）
# ---------------------------------------------------------------------------
def _sorted_series(events: dict) -> Tuple[List[int], List[int], List[int]]:
    user_ts = sorted(events["user_ts"].values())
    edge_ts = [t for t, _a, _b in events["edge_events"]]  # 已有序
    node_ts = sorted(events["node_first"].values())
    return user_ts, edge_ts, node_ts


def _count_between(series: List[int], lo: int, hi: int) -> int:
    """有序事件流中落在 ``[lo, hi)`` 内的事件数（精确，不依赖分桶）。"""
    return bisect.bisect_left(series, hi) - bisect.bisect_left(series, lo)


def _count_before(series: List[int], ts: int) -> int:
    return bisect.bisect_left(series, ts)


def _empty_result(granularity: str, start: Optional[int], end: Optional[int]) -> dict:
    return {
        "granularity": granularity,
        "timezone": "UTC",
        "range": {"start": start, "end": end},
        "buckets": [],
        "milestones": [],
        "totals": {
            "users": 0, "nodes": 0, "edges": 0,
            "users_without_time": 0, "nodes_without_time": 0,
            "edges_without_time": 0,
            "first_event": None, "last_event": None,
        },
        "generated_at": config.now_ms(),
    }


def build_curve(
    events: dict,
    granularity: str = "auto",
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> dict:
    """累计增长曲线 + 关键时间点。

    每个桶同时携带**区间增量**（``new_*``）与**截至桶末的绝对累计值**
    （``users``/``edges``/``nodes``）；累计值从时间零点算起，与展示窗口
    无关，因此任意窗口截取都与全量曲线严格一致。
    """
    user_ts, edge_ts, node_ts = _sorted_series(events)
    all_ts = user_ts + edge_ts
    if not all_ts:
        g = "day" if granularity == "auto" else _check_granularity(granularity)
        return _empty_result(g, start, end)

    lo = min(all_ts) if start is None else int(start)
    hi = max(all_ts) if end is None else int(end)
    if hi < lo:
        lo, hi = hi, lo
    if granularity == "auto":
        granularity = choose_granularity(hi - lo)
    _check_granularity(granularity)

    buckets = []
    b = bucket_start(lo, granularity)
    while b <= hi:
        bn = bucket_next(b, granularity)
        buckets.append({
            "start": b,
            "end": bn,
            "label": bucket_label(b, granularity),
            "new_users": _count_between(user_ts, b, bn),
            "new_edges": _count_between(edge_ts, b, bn),
            "new_nodes": _count_between(node_ts, b, bn),
            "users": _count_before(user_ts, bn),
            "edges": _count_before(edge_ts, bn),
            "nodes": _count_before(node_ts, bn),
        })
        b = bn

    return {
        "granularity": granularity,
        "timezone": "UTC",
        "range": {"start": lo, "end": hi},
        "buckets": buckets,
        "milestones": _milestones(events, buckets, user_ts, edge_ts),
        "totals": {
            "users": events["users_total"],
            "nodes": events["nodes_total"],
            "edges": events["edges_total"],
            "users_without_time": events["users_without_time"],
            "nodes_without_time": events["nodes_without_time"],
            "edges_without_time": events["edges_without_time"],
            "first_event": min(all_ts),
            "last_event": max(all_ts),
        },
        "generated_at": config.now_ms(),
    }


def _milestones(events: dict, buckets: List[dict],
                user_ts: List[int], edge_ts: List[int]) -> List[dict]:
    """关键时间点：首个用户 / 第一条关系 / 新增峰值桶 / 最近活动。

    峰值并列时取最早的桶（``max`` 返回首个最大元素），保证可复现。
    """
    out = []
    if user_ts:
        first = user_ts[0]
        uid = min(uid for uid, t in events["user_ts"].items() if t == first)
        out.append({
            "type": "first_user", "label": "首个用户",
            "ts": first, "time": fmt_dt(first), "user": uid,
        })
    if events["edge_events"]:
        ts, a, b = events["edge_events"][0]
        out.append({
            "type": "first_edge", "label": "第一条关系",
            "ts": ts, "time": fmt_dt(ts), "edge": [a, b],
        })
    if buckets:
        peak_u = max(buckets, key=lambda x: x["new_users"])
        if peak_u["new_users"] > 0:
            out.append({
                "type": "peak_new_users", "label": "新增用户峰值",
                "ts": peak_u["start"], "time": peak_u["label"],
                "value": peak_u["new_users"],
            })
        peak_e = max(buckets, key=lambda x: x["new_edges"])
        if peak_e["new_edges"] > 0:
            out.append({
                "type": "peak_new_edges", "label": "新增关系峰值",
                "ts": peak_e["start"], "time": peak_e["label"],
                "value": peak_e["new_edges"],
            })
    if user_ts or edge_ts:
        last = max(user_ts + edge_ts)
        out.append({
            "type": "latest_event", "label": "最近活动",
            "ts": last, "time": fmt_dt(last),
        })
    return out


def delta(events: dict, start: int, end: int, granularity: str = "auto") -> dict:
    """区间 ``[start, end)`` 的增量统计。

    计数直接在有序事件流上用 bisect 完成（精确到毫秒，不依赖分桶取整），
    桶明细复用 :func:`build_curve` 的同一套分桶逻辑，因此与曲线同源同口径。
    """
    start, end = int(start), int(end)
    if end <= start:
        raise ValueError("end 必须大于 start")
    user_ts, edge_ts, node_ts = _sorted_series(events)
    if granularity == "auto":
        granularity = choose_granularity(end - start)
    _check_granularity(granularity)

    # 桶明细：覆盖到区间内最后一个毫秒，避免 end 恰在桶边界时多出空桶。
    curve = build_curve(events, granularity, start, end - 1)

    new_user_ids = sorted(
        ((ts, uid) for uid, ts in events["user_ts"].items() if start <= ts < end)
    )
    new_edge_list = [
        [a, b, ts] for ts, a, b in events["edge_events"] if start <= ts < end
    ]
    return {
        "granularity": granularity,
        "timezone": "UTC",
        "range": {"start": start, "end": end},
        "new_users": _count_between(user_ts, start, end),
        "new_edges": _count_between(edge_ts, start, end),
        "new_nodes": _count_between(node_ts, start, end),
        "at_start": {
            "users": _count_before(user_ts, start),
            "edges": _count_before(edge_ts, start),
            "nodes": _count_before(node_ts, start),
        },
        "at_end": {
            "users": _count_before(user_ts, end),
            "edges": _count_before(edge_ts, end),
            "nodes": _count_before(node_ts, end),
        },
        "buckets": curve["buckets"],
        # 明细列表（(ts, uid) / [a, b, ts]，均按时间排序，确定可复现）。
        "new_user_list": [[uid, ts] for ts, uid in new_user_ids],
        "new_edge_list": new_edge_list,
        "generated_at": config.now_ms(),
    }
