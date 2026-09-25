"""
run.py
------
Entry point.  Start the server:

    python -m backend.run            # serve on 127.0.0.1:8080
    GSB_PORT=9000 python -m backend.run
    python backend/run.py --seed    # seed demo data on startup if empty

Also supports a ``--seed`` flag to auto-populate demo data when the dataset is
empty, and a ``--check`` flag to run a quick self-test of the algorithm stack
and exit (useful for CI / validation).
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running both as a package (`python -m backend.run`) and as a script
# (`python backend/run.py`).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import api, config, seed
    from backend.service import SocialGraphService
else:
    from . import api, config, seed
    from .service import SocialGraphService


def _check() -> int:
    """Run a smoke test over the algorithm stack; exit 0 on success."""
    from backend import algorithms
    from backend.graph import Graph

    g = Graph(directed=False)
    # A small barbell: two 3-cliques joined by a bridge.
    edges = [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (5, 6), (4, 6)]
    for u, v in edges:
        g.add_edge(u, v)
    g.freeze()

    assert g.node_count == 6, g.node_count
    assert g.edge_count == 7, g.edge_count

    path, dist, alg = algorithms.shortest_path(g, 1, 6)
    assert path == [1, 2, 3, 4, 5, 6] or path == [1, 3, 4, 6, 5] or len(path) - 1 == dist, path

    common = algorithms.common_friends(g, 1, 6)
    pr = algorithms.pagerank(g)
    assert abs(sum(pr.values()) - 1.0) < 1e-6, sum(pr.values())

    lv = algorithms.louvain(g)
    assert lv["num_communities"] >= 2, lv  # cliques should separate

    rec = algorithms.hybrid_recommend(g, 1, k=3)
    assert "items" in rec

    _check_timeline()

    print("[check] OK: graph, bfs, pagerank, louvain, recommend, timeline all pass")
    return 0


def _check_timeline() -> None:
    """Pure-function checks for the evolution timeline (no disk involved)."""
    from backend import timeline

    # users: uid -> created ms (uid 4 has unknown time);
    # edges: (u, v, ts) with a reversed duplicate and a self-loop.
    events = timeline.events_from(
        users={1: 1_000, 2: 2_000, 3: 5_000, 4: 0},
        edges=[
            (1, 2, 1_500),
            (2, 1, 900),      # duplicate of (1,2) with earlier ts -> wins
            (2, 3, 4_000),
            (1, 3, 4_000),
            (3, 3, 100),      # self-loop -> ignored
        ],
    )
    # Dedup + earliest-ts semantics.
    assert events["edges_total"] == 3, events["edges_total"]
    assert [t for t, _a, _b in events["edge_events"]] == [900, 4_000, 4_000]
    # Node first-seen = min(user created, first edge appearance).
    assert events["node_first"][2] == 900  # edge (2,1,900) predates user ts 2000
    assert events["nodes_total"] == 4 and events["nodes_without_time"] == 1

    curve = timeline.build_curve(events, "day")
    final = curve["buckets"][-1]
    # Curve endpoint must equal the true (deduped) counts.
    assert final["edges"] == 3 and final["users"] == 3 and final["nodes"] == 3
    assert curve["totals"]["edges"] == 3 and curve["totals"]["users"] == 4

    # Delta over the full span must match the curve increments exactly.
    d = timeline.delta(events, 0, timeline.DAY_MS, "day")
    assert d["new_edges"] == 3 and d["new_users"] == 3 and d["new_nodes"] == 3
    assert sum(b["new_edges"] for b in curve["buckets"]) == d["new_edges"]
    assert sum(b["new_users"] for b in curve["buckets"]) == d["new_users"]
    # Mid-range delta is exact (not bucket-rounded): only events < 1000.
    # Edge (2,1,900) makes both endpoints first-appear at t=900.
    d2 = timeline.delta(events, 0, 1_000, "day")
    assert d2["new_edges"] == 1 and d2["new_users"] == 0 and d2["new_nodes"] == 2

    # Bucket alignment: week starts Monday 00:00 UTC, month starts on the 1st.
    some_ts = 1_790_000_000_000
    w = timeline._utc(timeline.bucket_start(some_ts, "week"))
    assert w.weekday() == 0 and (w.hour, w.minute) == (0, 0)
    m = timeline._utc(timeline.bucket_start(some_ts, "month"))
    assert m.day == 1 and (m.hour, m.minute) == (0, 0)

    # Reproducibility: identical input -> identical output (modulo generated_at).
    c2 = timeline.build_curve(events, "day")
    assert c2["buckets"] == curve["buckets"]
    assert c2["milestones"] == curve["milestones"]
    # Invalid inputs are rejected.
    for bad in lambda: timeline.build_curve(events, "hourly"), \
            lambda: timeline.delta(events, 100, 100):
        try:
            bad()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def main() -> int:
    parser = argparse.ArgumentParser(description="Social graph analysis server")
    parser.add_argument("--seed", action="store_true", help="seed demo data if empty")
    parser.add_argument("--check", action="store_true", help="run self-test and exit")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.check:
        return _check()

    if args.host:
        config.HOST = args.host
    if args.port:
        config.PORT = args.port

    config.ensure_dirs()
    service = SocialGraphService()

    if args.seed:
        users = service.store.load_users()
        if not users:
            print("[social-graph] empty dataset -- seeding demo data ...")
            result = seed.generate_demo(service)
            print(f"[social-graph] seeded {result['users']} users, {result['edges']} edges")

    api.run(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
