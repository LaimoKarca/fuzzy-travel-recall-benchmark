"""Build the deterministic heterogeneous travel-memory graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


MAX_ADJACENT_GAP = pd.Timedelta(hours=8)

REQUIRED_EVENT_COLUMNS = (
    "user_id",
    "event_id",
    "trail_id",
    "venue_id",
    "name",
    "venue_category",
    "venue_city",
    "address",
    "latitude",
    "longitude",
    "timestamp",
    "previous_event_id",
    "next_event_id",
    "is_high_memory_load",
)

NODE_COLUMNS = (
    "node_id",
    "node_type",
    "user_id",
    "event_id",
    "venue_id",
    "trail_id",
    "name",
    "category",
    "timestamp",
    "city",
    "address",
    "latitude",
    "longitude",
    "is_high_memory_load",
)

EDGE_COLUMNS = (
    "source_node_id",
    "target_node_id",
    "relation",
    "user_id",
    "trail_id",
)

TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}


class GraphBuildError(RuntimeError):
    """Raised when canonical events cannot form a trustworthy graph."""


@dataclass(frozen=True)
class BuiltTravelGraph:
    graph: Any
    nodes: pd.DataFrame
    edges: pd.DataFrame
    events: pd.DataFrame


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        examples = sorted(set(series.loc[invalid].astype(str)))[:3]
        raise GraphBuildError(
            f"{label} contains invalid boolean value(s): {', '.join(examples)}"
        )
    return normalized.isin(TRUE_VALUES)


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in events]
    if missing:
        raise GraphBuildError(
            "events CSV is missing required column(s): " + ", ".join(missing)
        )

    events = events.loc[:, REQUIRED_EVENT_COLUMNS].copy()
    for column in ("user_id", "event_id", "trail_id", "venue_id", "venue_category", "timestamp"):
        if events[column].astype(str).str.strip().eq("").any():
            raise GraphBuildError(f"events CSV contains blank {column} value(s)")
    if events["event_id"].duplicated().any():
        raise GraphBuildError("events CSV contains duplicate event_id values")

    try:
        events["_timestamp"] = pd.to_datetime(
            events["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise GraphBuildError(f"events CSV contains invalid timestamp value(s): {exc}") from exc
    events["_is_high_memory_load"] = _parse_boolean(
        events["is_high_memory_load"], "events.is_high_memory_load"
    )

    if events.groupby("user_id")["_is_high_memory_load"].nunique().gt(1).any():
        raise GraphBuildError("is_high_memory_load is inconsistent within a user")
    if events.groupby("trail_id")["user_id"].nunique().gt(1).any():
        raise GraphBuildError("trail_id is shared by multiple users")

    poi_fields = (
        "name",
        "venue_category",
        "venue_city",
        "address",
        "latitude",
        "longitude",
    )
    inconsistent_poi = [
        column
        for column in poi_fields
        if events.groupby("venue_id")[column].nunique(dropna=False).gt(1).any()
    ]
    if inconsistent_poi:
        raise GraphBuildError(
            "venue_id has inconsistent metadata in: " + ", ".join(inconsistent_poi)
        )

    for coordinate in ("latitude", "longitude"):
        nonblank = events[coordinate].astype(str).str.strip().ne("")
        try:
            numeric = pd.to_numeric(events.loc[nonblank, coordinate], errors="raise")
        except (TypeError, ValueError) as exc:
            raise GraphBuildError(
                f"events CSV contains invalid {coordinate} value(s): {exc}"
            ) from exc
        if not numeric.map(math.isfinite).all():
            raise GraphBuildError(f"events CSV contains non-finite {coordinate} value(s)")

    events = events.sort_values(["user_id", "event_id"], kind="stable").reset_index(drop=True)
    lookup = events.set_index("event_id", drop=False)
    event_ids = set(events["event_id"])
    referenced = set(events.loc[events["previous_event_id"].ne(""), "previous_event_id"])
    referenced |= set(events.loc[events["next_event_id"].ne(""), "next_event_id"])
    missing_references = sorted(referenced - event_ids)
    if missing_references:
        raise GraphBuildError(
            "events CSV contains missing previous/next event reference(s): "
            + ", ".join(missing_references[:5])
        )

    next_targets = events.loc[events["next_event_id"].ne(""), "next_event_id"]
    if next_targets.duplicated().any():
        raise GraphBuildError("NEXT relation branches into the same event more than once")

    for _, event in events.iterrows():
        for relation, reference_column, reciprocal_column in (
            ("PREVIOUS", "previous_event_id", "next_event_id"),
            ("NEXT", "next_event_id", "previous_event_id"),
        ):
            reference = event[reference_column]
            if not reference:
                continue
            related = lookup.loc[reference]
            if related["user_id"] != event["user_id"] or related["trail_id"] != event["trail_id"]:
                raise GraphBuildError(f"{relation} relation crosses a user or trail boundary")
            if related[reciprocal_column] != event["event_id"]:
                raise GraphBuildError(f"{relation} relation is not reciprocal")
            if relation == "NEXT":
                gap = related["_timestamp"] - event["_timestamp"]
                if gap < pd.Timedelta(0) or gap > MAX_ADJACENT_GAP:
                    raise GraphBuildError("NEXT relation has an invalid timestamp gap")
    return events


def _node_row(node_id: str, node_type: str, **values: object) -> dict[str, object]:
    row = {column: "" for column in NODE_COLUMNS}
    row.update({"node_id": node_id, "node_type": node_type})
    row.update(values)
    return row


def _build_nodes(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    user_high = events.groupby("user_id", sort=True)["_is_high_memory_load"].first()
    for user_id, high_load in user_high.items():
        rows.append(
            _node_row(
                f"user::{user_id}",
                "user",
                user_id=user_id,
                is_high_memory_load=bool(high_load),
            )
        )

    for _, event in events.sort_values("event_id", kind="stable").iterrows():
        rows.append(
            _node_row(
                f"event::{event['event_id']}",
                "event",
                user_id=event["user_id"],
                event_id=event["event_id"],
                venue_id=event["venue_id"],
                trail_id=event["trail_id"],
                name=event["name"],
                category=event["venue_category"],
                timestamp=event["timestamp"],
                city=event["venue_city"],
                address=event["address"],
                latitude=event["latitude"],
                longitude=event["longitude"],
                is_high_memory_load=bool(event["_is_high_memory_load"]),
            )
        )

    pois = events.sort_values("venue_id", kind="stable").drop_duplicates("venue_id")
    for _, poi in pois.iterrows():
        rows.append(
            _node_row(
                f"poi::{poi['venue_id']}",
                "poi",
                venue_id=poi["venue_id"],
                name=poi["name"],
                category=poi["venue_category"],
                city=poi["venue_city"],
                address=poi["address"],
                latitude=poi["latitude"],
                longitude=poi["longitude"],
            )
        )

    for category in sorted(events["venue_category"].unique()):
        rows.append(
            _node_row(f"category::{category}", "category", category=category)
        )

    trails = events.sort_values("trail_id", kind="stable").drop_duplicates("trail_id")
    for _, trail in trails.iterrows():
        rows.append(
            _node_row(
                f"trail::{trail['trail_id']}",
                "trail",
                user_id=trail["user_id"],
                trail_id=trail["trail_id"],
                is_high_memory_load=bool(trail["_is_high_memory_load"]),
            )
        )
    return pd.DataFrame(rows, columns=NODE_COLUMNS)


def _build_edges(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        common = {"user_id": event["user_id"], "trail_id": event["trail_id"]}
        event_node = f"event::{event['event_id']}"
        rows.extend(
            [
                {
                    "source_node_id": f"user::{event['user_id']}",
                    "target_node_id": event_node,
                    "relation": "HAS",
                    **common,
                },
                {
                    "source_node_id": event_node,
                    "target_node_id": f"poi::{event['venue_id']}",
                    "relation": "AT",
                    **common,
                },
                {
                    "source_node_id": event_node,
                    "target_node_id": f"trail::{event['trail_id']}",
                    "relation": "IN_TRAIL",
                    **common,
                },
            ]
        )
        if event["next_event_id"]:
            rows.append(
                {
                    "source_node_id": event_node,
                    "target_node_id": f"event::{event['next_event_id']}",
                    "relation": "NEXT",
                    **common,
                }
            )

    pois = events.sort_values("venue_id", kind="stable").drop_duplicates("venue_id")
    for _, poi in pois.iterrows():
        rows.append(
            {
                "source_node_id": f"poi::{poi['venue_id']}",
                "target_node_id": f"category::{poi['venue_category']}",
                "relation": "CATEGORY",
                "user_id": "",
                "trail_id": "",
            }
        )

    relation_order = {name: index for index, name in enumerate(("HAS", "AT", "CATEGORY", "IN_TRAIL", "NEXT"))}
    edges = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    edges["_relation_order"] = edges["relation"].map(relation_order)
    return edges.sort_values(
        ["_relation_order", "source_node_id", "target_node_id"], kind="stable"
    ).drop(columns="_relation_order").reset_index(drop=True)


def build_travel_graph(events: pd.DataFrame) -> BuiltTravelGraph:
    """Validate events and build NetworkX plus portable node/edge tables."""

    try:
        import networkx as nx
    except (ImportError, OSError) as exc:
        raise GraphBuildError(
            "networkx is not installed; run 'python -m pip install -r requirements.txt'"
        ) from exc

    prepared = _validate_events(events)
    nodes = _build_nodes(prepared)
    edges = _build_edges(prepared)
    if nodes["node_id"].duplicated().any():
        raise GraphBuildError("generated graph contains duplicate node IDs")
    node_ids = set(nodes["node_id"])
    endpoints = set(edges["source_node_id"]) | set(edges["target_node_id"])
    if not endpoints.issubset(node_ids):
        raise GraphBuildError("generated edge references a missing node")

    graph = nx.DiGraph()
    for row in nodes.to_dict("records"):
        graph.add_node(row.pop("node_id"), **row)
    for row in edges.to_dict("records"):
        source = row.pop("source_node_id")
        target = row.pop("target_node_id")
        graph.add_edge(source, target, **row)
    if graph.number_of_nodes() != len(nodes) or graph.number_of_edges() != len(edges):
        raise GraphBuildError("NetworkX graph size does not match node/edge tables")
    return BuiltTravelGraph(graph=graph, nodes=nodes, edges=edges, events=prepared)
