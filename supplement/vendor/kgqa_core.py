#!/usr/bin/env python3
"""Core graph-path utilities for the budget-constrained MetaQA study."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi


RELATION_TEXT = {
    "directed_by": "directed by director",
    "written_by": "written by writer",
    "starred_actors": "stars starred actor acted in",
    "release_year": "release year released in",
    "in_language": "language spoken in",
    "has_genre": "genre type",
    "has_tags": "tag described by",
    "has_imdb_rating": "IMDb rating",
    "has_imdb_votes": "IMDb votes",
}
FORWARD_RELATION_TEXT = {
    "directed_by": "directed by director",
    "written_by": "written by writer",
    "starred_actors": "stars actors cast",
    "release_year": "release year released in",
    "in_language": "in language spoken language",
    "has_genre": "has genre type",
    "has_tags": "has tag described by",
    "has_imdb_rating": "has IMDb rating",
    "has_imdb_votes": "has IMDb votes",
}
INVERSE_RELATION_TEXT = {
    "directed_by": "director directed movies films",
    "written_by": "writer wrote movies films",
    "starred_actors": "actor acted in starred in movies films",
    "release_year": "year movies films released in",
    "in_language": "language movies films spoken in",
    "has_genre": "genre movies films type",
    "has_tags": "tag described movies films",
    "has_imdb_rating": "IMDb rating movies films",
    "has_imdb_votes": "IMDb votes movies films",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class Edge:
    head: str
    relation: str
    tail: str


@dataclass(frozen=True, slots=True)
class Traversal:
    next_entity: str
    edge: Edge


@dataclass(frozen=True, slots=True)
class GraphPath:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]

    @property
    def endpoint(self) -> str:
        return self.nodes[-1]

    @property
    def hops(self) -> int:
        return len(self.edges)

    def text(self) -> str:
        pieces = [self.nodes[0]]
        for index, edge in enumerate(self.edges):
            source = self.nodes[index]
            target = self.nodes[index + 1]
            relation = RELATION_TEXT.get(edge.relation, edge.relation.replace("_", " "))
            if edge.head == source and edge.tail == target:
                pieces.append(f" --[{relation}]--> {target}")
            elif edge.tail == source and edge.head == target:
                pieces.append(f" <--[{relation}]-- {target}")
            else:
                raise ValueError(f"Edge {edge} does not connect {source!r} and {target!r}")
        return "".join(pieces)

    def raw_text(self) -> str:
        """Render the stored relation identifiers without manual verbalization."""
        pieces = [self.nodes[0]]
        for index, edge in enumerate(self.edges):
            source = self.nodes[index]
            target = self.nodes[index + 1]
            if edge.head == source and edge.tail == target:
                pieces.append(f" --[{edge.relation}]--> {target}")
            elif edge.tail == source and edge.head == target:
                pieces.append(f" <--[{edge.relation}]-- {target}")
            else:
                raise ValueError(f"Edge {edge} does not connect {source!r} and {target!r}")
        return "".join(pieces)

    def relation_text(self) -> str:
        """Direction-aware, entity-free verbalization of the relation chain."""
        phrases = []
        for index, edge in enumerate(self.edges):
            source = self.nodes[index]
            target = self.nodes[index + 1]
            if edge.head == source and edge.tail == target:
                phrase = FORWARD_RELATION_TEXT.get(
                    edge.relation, edge.relation.replace("_", " ")
                )
            elif edge.tail == source and edge.head == target:
                phrase = INVERSE_RELATION_TEXT.get(
                    edge.relation, edge.relation.replace("_", " ")
                )
            else:
                raise ValueError(
                    f"Edge {edge} does not connect {source!r} and {target!r}"
                )
            phrases.append(phrase)
        return " then ".join(phrases)

    def raw_relation_text(self) -> str:
        """Direction-aware chain made only from raw KG relation identifiers."""
        phrases = []
        for index, edge in enumerate(self.edges):
            source = self.nodes[index]
            target = self.nodes[index + 1]
            if edge.head == source and edge.tail == target:
                direction = "forward"
            elif edge.tail == source and edge.head == target:
                direction = "inverse"
            else:
                raise ValueError(
                    f"Edge {edge} does not connect {source!r} and {target!r}"
                )
            phrases.append(f"{direction} {edge.relation}")
        return " then ".join(phrases)

    def signature(self) -> frozenset[str]:
        values = []
        for node in self.nodes[1:]:
            values.extend(lexical_tokens(node))
        for edge in self.edges:
            values.extend(lexical_tokens(RELATION_TEXT.get(edge.relation, edge.relation)))
        return frozenset(values)

    def directional_signature(self) -> frozenset[str]:
        values = lexical_tokens(self.relation_text())
        for node in self.nodes[1:]:
            values.extend(lexical_tokens(node))
        return frozenset(values)

    def raw_directional_signature(self) -> frozenset[str]:
        values = lexical_tokens(self.raw_relation_text())
        for node in self.nodes[1:]:
            values.extend(lexical_tokens(node))
        return frozenset(values)


def lexical_tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def load_kg(path: Path) -> dict[str, tuple[Traversal, ...]]:
    adjacency: dict[str, list[Traversal]] = defaultdict(list)
    seen: set[Edge] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed processed KG row at {path}:{line_number}")
            edge = Edge(*fields)
            if edge in seen:
                raise ValueError(f"Processed KG still contains duplicate edge: {edge}")
            seen.add(edge)
            adjacency[edge.head].append(Traversal(edge.tail, edge))
            adjacency[edge.tail].append(Traversal(edge.head, edge))

    return {
        entity: tuple(
            sorted(
                traversals,
                key=lambda item: (
                    item.next_entity,
                    item.edge.relation,
                    item.edge.head,
                    item.edge.tail,
                ),
            )
        )
        for entity, traversals in adjacency.items()
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            rows.append(row)
    return rows


def enumerate_simple_paths(
    adjacency: dict[str, tuple[Traversal, ...]],
    topic: str,
    max_hops: int,
    max_paths: int,
    exact_hops_only: bool = False,
) -> tuple[list[GraphPath], bool]:
    """Enumerate deterministic simple paths by BFS with a candidate cap."""
    if max_hops < 1:
        raise ValueError("max_hops must be positive")
    if max_paths < 1:
        raise ValueError("max_paths must be positive")

    paths: list[GraphPath] = []
    frontier: list[GraphPath] = [GraphPath((topic,), ())]

    for depth in range(1, max_hops + 1):
        next_frontier: list[GraphPath] = []
        for prefix in frontier:
            current = prefix.nodes[-1]
            for traversal in adjacency.get(current, ()):
                if traversal.next_entity in prefix.nodes:
                    continue
                path = GraphPath(
                    prefix.nodes + (traversal.next_entity,),
                    prefix.edges + (traversal.edge,),
                )
                is_candidate = not exact_hops_only or depth == max_hops
                if is_candidate:
                    paths.append(path)
                    if len(paths) >= max_paths:
                        return paths, True
                if depth < max_hops:
                    next_frontier.append(path)
        frontier = next_frontier
        if not frontier:
            break

    return paths, False


def render_evidence(paths: Sequence[GraphPath], raw_labels: bool = False) -> str:
    renderer = GraphPath.raw_text if raw_labels else GraphPath.text
    return "\n".join(f"- {renderer(path)}" for path in paths)


def evidence_tokens(
    paths: Sequence[GraphPath], tokenizer, raw_labels: bool = False
) -> int:
    if not paths:
        return 0
    return len(
        tokenizer.encode(
            render_evidence(paths, raw_labels=raw_labels),
            add_special_tokens=False,
        )
    )


def standalone_evidence_tokens(
    paths: Sequence[GraphPath],
    tokenizer,
    batch_size: int = 256,
    raw_labels: bool = False,
) -> list[int]:
    """Tokenize individual evidence lines in batches for selection costs."""
    costs: list[int] = []
    for start in range(0, len(paths), batch_size):
        batch = [
            f"- {path.raw_text() if raw_labels else path.text()}"
            for path in paths[start : start + batch_size]
        ]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        costs.extend(max(1, len(token_ids)) for token_ids in encoded)
    return costs


def answer_recall(paths: Sequence[GraphPath], gold_answers: Iterable[str]) -> float:
    gold = set(gold_answers)
    if not gold:
        return 0.0
    endpoints = {path.endpoint for path in paths}
    return len(endpoints & gold) / len(gold)


def bm25_scores(question: str, paths: Sequence[GraphPath]) -> list[float]:
    if not paths:
        return []
    corpus = [lexical_tokens(path.text()) for path in paths]
    query = lexical_tokens(question)
    ranker = BM25Okapi(corpus)
    return [float(score) for score in ranker.get_scores(query)]


def relation_bm25_scores(
    question: str, paths: Sequence[GraphPath], raw_labels: bool = False
) -> list[float]:
    """BM25 over direction-aware relation chains, excluding entity-name noise."""
    if not paths:
        return []
    corpus = [
        lexical_tokens(
            path.raw_relation_text() if raw_labels else path.relation_text()
        )
        for path in paths
    ]
    query = lexical_tokens(question)
    ranker = BM25Okapi(corpus)
    return [float(score) for score in ranker.get_scores(query)]


def rank_by_scores(paths: Sequence[GraphPath], scores: Sequence[float]) -> list[GraphPath]:
    if len(paths) != len(scores):
        raise ValueError("paths and scores have different lengths")
    indices = sorted(
        range(len(paths)),
        key=lambda index: (-scores[index], paths[index].hops, index),
    )
    return [paths[index] for index in indices]


def pack_in_order(
    paths: Sequence[GraphPath],
    budget: int,
    tokenizer,
    standalone_costs: Sequence[int] | None = None,
    raw_labels: bool = False,
) -> list[GraphPath]:
    if standalone_costs is not None and len(standalone_costs) != len(paths):
        raise ValueError("standalone_costs and paths have different lengths")
    selected: list[GraphPath] = []
    current_tokens = 0
    for index, path in enumerate(paths):
        if standalone_costs is not None:
            # Token merges across a newline can change the isolated cost by a
            # token or two.  The small tolerance avoids false rejection while
            # eliminating expensive tokenization of clearly infeasible paths.
            if standalone_costs[index] > (budget - current_tokens) + 2:
                continue
        trial = selected + [path]
        trial_tokens = evidence_tokens(trial, tokenizer, raw_labels=raw_labels)
        if trial_tokens <= budget:
            selected.append(path)
            current_tokens = trial_tokens
    return selected


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def budget_aware_mmr(
    paths: Sequence[GraphPath],
    scores: Sequence[float],
    budget: int,
    tokenizer,
    relevance_weight: float = 0.70,
    cost_exponent: float = 0.50,
    standalone_costs: Sequence[int] | None = None,
    signatures: Sequence[frozenset[str]] | None = None,
    raw_labels: bool = False,
) -> list[GraphPath]:
    """Greedy relevance-diversity selection with an exact final token cap."""
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be in [0, 1]")
    if cost_exponent < 0.0:
        raise ValueError("cost_exponent must be non-negative")
    if len(paths) != len(scores):
        raise ValueError("paths and scores have different lengths")
    if not paths:
        return []

    maximum_score = max(scores)
    if maximum_score > 0.0:
        relevance = [max(0.0, score) / maximum_score for score in scores]
    else:
        relevance = [0.0 for _ in scores]
    if signatures is None:
        signatures = [path.signature() for path in paths]
    elif len(signatures) != len(paths):
        raise ValueError("signatures and paths have different lengths")
    if standalone_costs is None:
        standalone_costs = standalone_evidence_tokens(
            paths, tokenizer, raw_labels=raw_labels
        )
    elif len(standalone_costs) != len(paths):
        raise ValueError("standalone_costs and paths have different lengths")

    remaining = set(range(len(paths)))
    selected_indices: list[int] = []
    selected_paths: list[GraphPath] = []

    current_tokens = 0
    while remaining:
        remaining_budget = budget - current_tokens
        scored_candidates = []
        for index in remaining:
            if standalone_costs[index] > remaining_budget + 2:
                continue
            if selected_indices:
                redundancy = max(
                    _jaccard(signatures[index], signatures[chosen])
                    for chosen in selected_indices
                )
                diversity = 1.0 - redundancy
            else:
                diversity = 1.0
            gain = (
                relevance_weight * relevance[index]
                + (1.0 - relevance_weight) * diversity
            )
            utility = gain / math.pow(standalone_costs[index], cost_exponent)
            key = (utility, relevance[index], -standalone_costs[index], -index)
            scored_candidates.append((key, index))

        if not scored_candidates:
            break
        scored_candidates.sort(reverse=True)

        accepted_index = None
        for _key, index in scored_candidates:
            trial = selected_paths + [paths[index]]
            trial_tokens = evidence_tokens(
                trial, tokenizer, raw_labels=raw_labels
            )
            if trial_tokens <= budget:
                accepted_index = index
                current_tokens = trial_tokens
                break
            # The selected set only grows, so this path can never fit later.
            remaining.remove(index)

        if accepted_index is None:
            break
        remaining.remove(accepted_index)
        selected_indices.append(accepted_index)
        selected_paths.append(paths[accepted_index])

    return selected_paths
