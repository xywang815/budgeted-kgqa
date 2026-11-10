#!/usr/bin/env python3
"""Prospectively frozen selector components for the week-2 KGQA study."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from kgqa_core import GraphPath, evidence_tokens, pack_in_order, rank_by_scores


def directional_relation_key(path: GraphPath) -> tuple[str, ...]:
    """Return the ordered directed relation chain without entity names."""
    key: list[str] = []
    for index, edge in enumerate(path.edges):
        source = path.nodes[index]
        target = path.nodes[index + 1]
        if edge.head == source and edge.tail == target:
            direction = "forward"
        elif edge.tail == source and edge.head == target:
            direction = "inverse"
        else:
            raise ValueError(f"Edge {edge} does not connect {source!r} and {target!r}")
        key.append(f"{direction}:{edge.relation}")
    return tuple(key)


def relation_only(
    paths: Sequence[GraphPath],
    scores: Sequence[float],
    budget: int,
    tokenizer,
    standalone_costs: Sequence[int],
) -> list[GraphPath]:
    """Direction-aware relation BM25 ranking without cost or diversity utility."""
    ranked = rank_by_scores(paths, scores)
    cost_by_path = dict(zip(paths, standalone_costs, strict=True))
    ranked_costs = [cost_by_path[path] for path in ranked]
    return pack_in_order(ranked, budget, tokenizer, ranked_costs)


def shortest_first(
    paths: Sequence[GraphPath],
    budget: int,
    tokenizer,
    standalone_costs: Sequence[int],
) -> list[GraphPath]:
    """Pack paths by ascending standalone token cost with deterministic ties."""
    if len(paths) != len(standalone_costs):
        raise ValueError("paths and standalone_costs have different lengths")
    ordered_indices = sorted(
        range(len(paths)),
        key=lambda index: (standalone_costs[index], paths[index].hops, index),
    )
    ordered = [paths[index] for index in ordered_indices]
    ordered_costs = [standalone_costs[index] for index in ordered_indices]
    return pack_in_order(ordered, budget, tokenizer, ordered_costs)


def unique_key_then_fill(
    paths: Sequence[GraphPath],
    scores: Sequence[float],
    keys: Sequence[Hashable],
    budget: int,
    tokenizer,
    standalone_costs: Sequence[int],
) -> list[GraphPath]:
    """Select one feasible path per key by relevance, then fill by relevance."""
    if not (
        len(paths) == len(scores) == len(keys) == len(standalone_costs)
    ):
        raise ValueError("paths, scores, keys, and standalone_costs must align")
    if budget < 1:
        raise ValueError("budget must be positive")

    ordered_indices = sorted(
        range(len(paths)),
        key=lambda index: (-scores[index], paths[index].hops, index),
    )
    selected_indices: list[int] = []
    selected_paths: list[GraphPath] = []
    selected_keys: set[Hashable] = set()
    current_tokens = 0

    def try_add(index: int) -> bool:
        nonlocal current_tokens
        if standalone_costs[index] > (budget - current_tokens) + 2:
            return False
        trial = selected_paths + [paths[index]]
        trial_tokens = evidence_tokens(trial, tokenizer)
        if trial_tokens > budget:
            return False
        selected_indices.append(index)
        selected_paths.append(paths[index])
        current_tokens = trial_tokens
        return True

    for index in ordered_indices:
        if keys[index] in selected_keys:
            continue
        if try_add(index):
            selected_keys.add(keys[index])

    selected_index_set = set(selected_indices)
    for index in ordered_indices:
        if index in selected_index_set:
            continue
        if try_add(index):
            selected_index_set.add(index)

    return selected_paths


def unit_endpoint(
    paths: Sequence[GraphPath],
    scores: Sequence[float],
    budget: int,
    tokenizer,
    standalone_costs: Sequence[int],
) -> list[GraphPath]:
    return unique_key_then_fill(
        paths,
        scores,
        [path.endpoint for path in paths],
        budget,
        tokenizer,
        standalone_costs,
    )


def unique_relation_chain(
    paths: Sequence[GraphPath],
    scores: Sequence[float],
    budget: int,
    tokenizer,
    standalone_costs: Sequence[int],
) -> list[GraphPath]:
    return unique_key_then_fill(
        paths,
        scores,
        [directional_relation_key(path) for path in paths],
        budget,
        tokenizer,
        standalone_costs,
    )
