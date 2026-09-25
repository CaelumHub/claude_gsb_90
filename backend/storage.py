"""
storage.py
----------
Sharded JSON storage for the graph and all derived data.

Graph storage model
~~~~~~~~~~~~~~~~~~~
The adjacency list is split across ``SHARD_COUNT`` JSON files.  Each shard holds
the *outgoing* edge lists for a subset of users (keyed by ``user % SHARD_COUNT``
so a user always lives in exactly one shard).  This gives us three properties
that matter at scale:

1. **Fast, lazy load** -- algorithms never parse the whole graph off disk at
   once; they can load one shard at a time or stream edges.
2. **Bounded incremental append** -- importing new edges touches only the
   shards of the affected users; no global rewrite.
3. **Simple merge** -- shards can be compacted/merged independently.

Each shard file looks like::

    {
      "shard": 3,
      "version": 2,
      "updated_at": 1720000000000,
      "users": { "42": {"name": "alice", "tags": ["tech"]} },      # optional denorm
      "edges": [ [42, 7, 1.0, 1720000000000], ... ]                  # (u, v, w, ts)
    }

Derived data (users, profiles, tags, recommendations, community, pagerank) is
stored in separate top-level JSON files.  The **index** maps every user id to
its shard for O(1) adjacency lookup without scanning files.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

try:
    from . import config
    from .graph import Graph
except ImportError:  # pragma: no cover
    import config
    from graph import Graph


# ---------------------------------------------------------------------------
# Low-level shard file helpers
# ---------------------------------------------------------------------------
def _shard_path(shard_id: int) -> str:
    return os.path.join(config.GRAPH_DIR, f"shard_{shard_id:04x}.json")


def _load_shard(shard_id: int) -> dict:
    path = _shard_path(shard_id)
    if not os.path.exists(path):
        return {"shard": shard_id, "version": 1, "updated_at": 0, "users": {}, "edges": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("users", {})
        data.setdefault("edges", [])
        return data
    except (OSError, ValueError):
        # Corrupt shard -> start fresh (logged by caller).
        return {"shard": shard_id, "version": 1, "updated_at": 0, "users": {}, "edges": []}


def _write_shard(shard_id: int, data: dict) -> None:
    data["shard"] = shard_id
    data["version"] = data.get("version", 1) + 1
    data["updated_at"] = config.now_ms()
    config.atomic_write_json(_shard_path(shard_id), data)


def _user_shard(user_id: int) -> int:
    # Use a stable hash so ids don't all land on the same shard when clustered.
    return (user_id * 2654435761) % config.SHARD_COUNT


def valid_timestamp(value) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
class GraphIndex:
    """O(1) user -> shard lookup plus aggregate metadata."""

    def __init__(self) -> None:
        self.user_shard: Dict[int, int] = {}
        self.meta: Dict[str, object] = {
            "version": config.INDEX_VERSION,
            "node_count": 0,
            "edge_count": 0,
            "shard_count": config.SHARD_COUNT,
            "built_at": 0,
        }

    def load(self) -> "GraphIndex":
        data = config.read_json(config.INDEX_FILE, {})
        self.user_shard = {int(k): v for k, v in data.get("user_shard", {}).items()}
        self.meta.update(data.get("meta", {}))
        return self

    def save(self) -> None:
        config.atomic_write_json(
            config.INDEX_FILE,
            {"user_shard": {str(k): v for k, v in self.user_shard.items()}, "meta": self.meta},
        )

    def rebuild(self, shard_stats: List[Tuple[int, int, int]]) -> None:
        """Recompute the index from a list of ``(shard_id, node_count, edge_count)``."""
        self.user_shard = {}
        total_nodes = 0
        total_edges = 0
        for shard_id, node_count, edge_count in shard_stats:
            # We don't know individual users here unless we scan; see
            # ``rebuild_index_from_shards`` in the service for the full pass.
            total_nodes += node_count
            total_edges += edge_count
        self.meta["node_count"] = total_nodes
        self.meta["edge_count"] = total_edges
        self.meta["built_at"] = config.now_ms()


def rebuild_index_from_shards() -> GraphIndex:
    """Full index rebuild: scan every shard, map each user to its shard.

    This is the expensive-but-rare operation performed after large incremental
    imports or merges, so that day-to-day lookups stay O(1).
    """
    index = GraphIndex()
    edge_count = 0
    for shard_id in range(config.SHARD_COUNT):
        data = _load_shard(shard_id)
        for user in data["users"]:
            index.user_shard[int(user)] = shard_id
        # Nodes not present in ``users`` (edge-only) still need a shard mapping.
        for edge in data["edges"]:
            u, v = int(edge[0]), int(edge[1])
            index.user_shard.setdefault(u, shard_id)
            index.user_shard.setdefault(v, shard_id)
        edge_count += len(data["edges"])
    # ``user_shard`` is keyed by unique user id, so its length is the true node
    # count (the per-shard ``users`` maps are denormalised and may repeat ids).
    index.meta.update(
        {
            "version": config.INDEX_VERSION,
            "node_count": len(index.user_shard),
            "edge_count": edge_count,
            "shard_count": config.SHARD_COUNT,
            "built_at": config.now_ms(),
        }
    )
    index.save()
    return index


# ---------------------------------------------------------------------------
# GraphStore -- the main facade
# ---------------------------------------------------------------------------
class GraphStore:
    """Owns sharded graph persistence, users, profiles, tags and derived data."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self.index = GraphIndex().load()

    # ---- users ------------------------------------------------------------
    def load_users(self) -> Dict[int, dict]:
        data = config.read_json(config.USERS_FILE, {"users": {}})
        users = data.get("users", {})
        return {int(k): v for k, v in users.items()}

    def save_users(self, users: Dict[int, dict]) -> None:
        serialised = {}
        for k, v in users.items():
            rec = dict(v)
            if "created_at" in rec:
                rec["created_at_ms"] = rec.pop("created_at")
            serialised[str(k)] = rec
        config.atomic_write_json(config.USERS_FILE, {"users": serialised})

    def load_profiles(self) -> Dict[int, dict]:
        data = config.read_json(config.PROFILES_FILE, {"profiles": {}})
        return {int(k): v for k, v in data.get("profiles", {}).items()}

    def save_profiles(self, profiles: Dict[int, dict]) -> None:
        config.atomic_write_json(
            config.PROFILES_FILE, {"profiles": {str(k): v for k, v in profiles.items()}}
        )

    def load_tags(self) -> Dict[str, dict]:
        data = config.read_json(config.TAGS_FILE, {"tags": {}})
        tags = data.get("tags", {})
        default_tag = {"name": "默认", "color": None, "created_at": 0}
        if "默认" not in tags:
            tags["默认"] = dict(default_tag)
        else:
            tags["默认"].setdefault("color", None)
            tags["默认"].setdefault("created_at", 0)
        return tags

    def save_tags(self, tags: Dict[str, dict]) -> None:
        config.atomic_write_json(config.TAGS_FILE, {"tags": tags})

    # ---- graph build ------------------------------------------------------
    def load_full_graph(self) -> Graph:
        """Load every shard into one frozen Graph.

        Memory-bounded in the sense that we stream shard by shard rather than
        materialising a giant JSON blob, but the in-memory CSR still scales
        with |V|+|E| -- which is exactly what the compact representation is for.
        """
        g = Graph(directed=False)
        for shard_id in range(config.SHARD_COUNT):
            data = _load_shard(shard_id)
            for u, v, w, _ts in data["edges"]:
                g.add_edge(int(u), int(v), float(w))
        g.freeze()
        return g

    def load_shard_graph(self, shard_id: int) -> Graph:
        data = _load_shard(shard_id)
        g = Graph(directed=False)
        for u, v, w, _ts in data["edges"]:
            g.add_edge(int(u), int(v), float(w))
        g.freeze()
        return g

    def iter_all_edges(self) -> Iterator[Tuple[int, int, float]]:
        for u, v, w, _ts in self.iter_all_edge_records():
            yield u, v, w

    def iter_all_edge_records(self) -> Iterator[Tuple[int, int, float, int]]:
        """Stream every persisted edge with its relationship timestamp."""
        for shard_id in range(config.SHARD_COUNT):
            data = _load_shard(shard_id)
            for record in data["edges"]:
                # Older shard files may contain three-field records.
                if len(record) >= 4 and valid_timestamp(record[3]):
                    u, v, w, ts = record[0], record[1], record[2], int(record[3])
                else:
                    u, v, w = record[0], record[1], record[2]
                    ts = config.now_ms()
                yield int(u), int(v), float(w), ts

    def neighborhood_graph(self, root: int, depth: int, limit: int) -> Graph:
        """Extract the induced subgraph around ``root`` up to ``depth`` hops."""
        g = Graph(directed=False)
        frontier = {root}
        visited = {root}
        for _ in range(depth):
            nxt: Set[int] = set()
            for node in frontier:
                shard = _user_shard(node)
                data = _load_shard(shard)
                for u, v, w, _ts in data["edges"]:
                    u, v = int(u), int(v)
                    if u == node:
                        g.add_edge(u, v, float(w))
                        if v not in visited and len(visited) < limit:
                            nxt.add(v)
                    elif v == node:
                        g.add_edge(u, v, float(w))
                        if u not in visited and len(visited) < limit:
                            nxt.add(u)
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        g.freeze()
        return g

    def neighbors_of(self, user: int) -> List[Tuple[int, float]]:
        shard = self.index.user_shard.get(user, _user_shard(user))
        data = _load_shard(shard)
        out = []
        for u, v, w, _ts in data["edges"]:
            u, v = int(u), int(v)
            if u == user:
                out.append((v, float(w)))
            elif v == user:
                out.append((u, float(w)))
        return out

    # ---- incremental import ----------------------------------------------
    def import_edges(self, edges: Iterable[Tuple]) -> dict:
        """Append edges to shards and refresh the index incrementally.

        Accepted edge tuples are ``(u, v, w)`` or ``(u, v, w, timestamp_ms)``.
        The timestamp is the relationship's event time and must therefore be
        preserved independently of file update time.  Relationships are
        canonicalised/deduplicated before writing so timeline counts match the
        true undirected graph.
        """
        pending: Dict[int, List[Tuple[int, int, float, int]]] = defaultdict(list)
        canonical: Dict[Tuple[int, int], Tuple[float, int]] = {}
        skipped = 0
        self_loops = 0
        now = config.now_ms()
        submitted = 0
        invalid = 0
        for edge in edges:
            submitted += 1
            try:
                u, v = int(edge[0]), int(edge[1])
                w = float(edge[2]) if len(edge) > 2 and edge[2] is not None else 1.0
            except (TypeError, ValueError, IndexError):
                invalid += 1
                continue
            if u == v:
                self_loops += 1
                continue
            if u > v:
                u, v = v, u
            ts = int(edge[3]) if len(edge) > 3 and valid_timestamp(edge[3]) else now
            key = (u, v)
            old = canonical.get(key)
            if old is None:
                canonical[key] = (w, ts)
            else:
                skipped += 1
                # Re-importing history must not overwrite the actual first time.
                canonical[key] = (w, min(ts, old[1]))

        for (u, v), (w, ts) in canonical.items():
            pending[_user_shard(u)].append((u, v, w, ts))

        touched_shards: List[Tuple[int, int, int]] = []
        imported = 0
        for shard_id, new_edges in pending.items():
            data = _load_shard(shard_id)
            existing: Dict[Tuple[int, int], Tuple[float, int]] = {}
            for record in data["edges"]:
                try:
                    u, v = int(record[0]), int(record[1])
                    w = float(record[2])
                    old_ts = int(record[3]) if len(record) > 3 and valid_timestamp(record[3]) else now
                except (TypeError, ValueError, IndexError):
                    skipped += 1
                    continue
                if u > v:
                    u, v = v, u
                if u == v:
                    self_loops += 1
                    continue
                prev = existing.get((u, v))
                if prev is None:
                    existing[(u, v)] = (w, old_ts)
                else:
                    skipped += 1
                    existing[(u, v)] = (w, min(old_ts, prev[1]))
            for u, v, w, ts in new_edges:
                if (u, v) in existing:
                    skipped += 1
                else:
                    imported += 1
                prev = existing.get((u, v))
                existing[(u, v)] = (w, min(ts, prev[1]) if prev else ts)
            data["edges"] = [
                [u, v, w, ts]
                for (u, v), (w, ts) in sorted(existing.items(), key=lambda item: (item[0][0], item[0][1], item[1][1]))
            ]
            for u, v, _w, _ts in data["edges"]:
                data["users"].setdefault(str(u), {"name": str(u)})
                data["users"].setdefault(str(v), {"name": str(v)})
            _write_shard(shard_id, data)
            touched_shards.append((shard_id, len(data["users"]), len(data["edges"])))

        # Canonical imports can move an edge to its owner shard; rebuild from
        # disk so mappings and counts cannot retain a stale duplicate.
        fresh_index = rebuild_index_from_shards()
        self.index = fresh_index
        canonical_edges = len(canonical)
        return {
            "submitted": submitted,
            "imported": imported,
            "duplicates": max(0, canonical_edges - imported),
            "invalid": invalid,
            "skipped": skipped + invalid,
            "self_loops": self_loops,
            "touched_shards": len(touched_shards),
        }

    def _refresh_index(self, shard_stats: List[Tuple[int, int, int]]) -> None:
        for shard_id, node_count, edge_count in shard_stats:
            data = _load_shard(shard_id)
            for user in data["users"]:
                self.index.user_shard[int(user)] = shard_id
            for u, v, w, _ts in data["edges"]:
                self.index.user_shard.setdefault(int(u), shard_id)
                self.index.user_shard.setdefault(int(v), shard_id)
        total_edges = 0
        total_nodes = len(self.index.user_shard)
        for shard_id in range(config.SHARD_COUNT):
            path = _shard_path(shard_id)
            if not os.path.exists(path):
                continue
            data = _load_shard(shard_id)
            edge_count = len(data["edges"])
            total_edges += edge_count
        self.index.meta["node_count"] = total_nodes
        self.index.meta["edge_count"] = total_edges
        self.index.meta["built_at"] = config.now_ms()
        self.index.save()

    # ---- merge / rebuild --------------------------------------------------
    def merge_shards(self) -> dict:
        """Compact all shards: deduplicate, sort, and drop empty shards.

        This is the "file merge + index rebuild" step.  After a long period of
        incremental imports, shards accumulate duplicate/redundant entries and
        the index drifts; a merge rewrites everything in canonical form.
        """
        global_edges: Dict[Tuple[int, int], Tuple[float, int]] = {}
        removed = 0
        for shard_id in range(config.SHARD_COUNT):
            data = _load_shard(shard_id)
            for record in data["edges"]:
                try:
                    raw_u, raw_v = int(record[0]), int(record[1])
                    w = float(record[2])
                    ts = int(record[3]) if len(record) > 3 and valid_timestamp(record[3]) else config.now_ms()
                except (TypeError, ValueError, IndexError):
                    removed += 1
                    continue
                if raw_u == raw_v:
                    removed += 1
                    continue
                key = (min(raw_u, raw_v), max(raw_u, raw_v))
                if key in global_edges:
                    removed += 1
                    old_w, old_ts = global_edges[key]
                    global_edges[key] = (w, min(old_ts, ts))
                else:
                    global_edges[key] = (w, ts)

        merged_edges: Dict[int, Dict[Tuple[int, int], Tuple[float, int]]] = defaultdict(dict)
        for (u, v), value in global_edges.items():
            merged_edges[_user_shard(u)][(u, v)] = value
        # Edges are assigned to the canonical owner shard by minimum endpoint;
        # clear old files so an edge moved from a non-canonical shard cannot
        # survive as a duplicate.
        for shard_id in range(config.SHARD_COUNT):
            path = _shard_path(shard_id)
            if os.path.exists(path):
                os.remove(path)
        # Rewrite every touched shard into the canonical owner shard.
        for shard_id, edge_map in merged_edges.items():
            data = {
                "shard": shard_id,
                "version": 1,
                "updated_at": config.now_ms(),
                "users": {},
                "edges": [
                    [u, v, w, ts]
                    for (u, v), (w, ts) in sorted(edge_map.items())
                ],
            }
            # Rebuild denormalised users.
            for u, v, w, ts in data["edges"]:
                data["users"].setdefault(str(u), {"name": str(u)})
                data["users"].setdefault(str(v), {"name": str(v)})
            config.atomic_write_json(_shard_path(shard_id), data)
        # Full index rebuild.
        rebuild_index_from_shards()
        return {"removed_duplicates": removed, "shards": len(merged_edges)}

    def shard_usage(self) -> List[dict]:
        out = []
        for shard_id in range(config.SHARD_COUNT):
            path = _shard_path(shard_id)
            if os.path.exists(path):
                size = os.path.getsize(path)
                data = _load_shard(shard_id)
                out.append(
                    {
                        "shard": shard_id,
                        "users": len(data["users"]),
                        "edges": len(data["edges"]),
                        "size_bytes": size,
                    }
                )
        return out


# ---------------------------------------------------------------------------
# Derived data stores
# ---------------------------------------------------------------------------
class DerivedStore:
    """Recommendations, community, pagerank -- each a single JSON file."""

    def __init__(self) -> None:
        config.ensure_dirs()

    def load_recommendations(self) -> Dict[int, dict]:
        data = config.read_json(config.RECOMMENDATIONS_FILE, {"recs": {}})
        recs = {int(k): v for k, v in data.get("recs", {}).items()}
        for v in recs.values():
            if isinstance(v, dict) and isinstance(v.get("items"), list):
                items = v.get("items", [])
                v["items"] = items[: config.RECOMMEND_CLAMP_MAX]
        return recs

    def save_recommendations(self, recs: Dict[int, dict]) -> None:
        config.atomic_write_json(
            config.RECOMMENDATIONS_FILE, {"recs": {str(k): v for k, v in recs.items()}}
        )

    def load_community(self) -> dict:
        data = config.read_json(config.COMMUNITY_FILE, {})
        data.setdefault("communities", {})
        data.setdefault("num_communities", 0)
        data.setdefault("modularity", 0.0)
        data.setdefault("computed_at", 0)
        data.setdefault("community_sizes", [])
        data.setdefault("resolution", config.LOUVAIN_RESOLUTION)
        data.setdefault("iterations", 0)
        return data

    def save_community(self, community: dict) -> None:
        config.atomic_write_json(config.COMMUNITY_FILE, community)

    def load_pagerank(self) -> dict:
        return config.read_json(config.PAGERANK_FILE, {})

    def save_pagerank(self, ranks: Dict[int, float]) -> None:
        config.atomic_write_json(
            config.PAGERANK_FILE, {"ranks": {str(k): v for k, v in ranks.items()}}
        )


def log_import(entry: dict) -> None:
    """Append a one-line JSON record to the import log (append-only, cheap)."""
    try:
        with open(config.IMPORT_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
