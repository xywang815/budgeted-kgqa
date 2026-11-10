#!/usr/bin/env python3
"""Dependency-light integrity tests for the frozen week-2 selectors."""

from kgqa_core import Edge, GraphPath, evidence_tokens
from week2_selectors import (
    directional_relation_key,
    relation_only,
    shortest_first,
    unique_relation_chain,
    unit_endpoint,
)


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def path(topic, relation, endpoint):
    return GraphPath((topic, endpoint), (Edge(topic, relation, endpoint),))


def main() -> int:
    tokenizer = WhitespaceTokenizer()
    paths = [
        path("T", "has_genre", "A"),
        path("T", "has_tags", "A"),
        path("T", "has_genre", "B"),
        path("T", "written_by", "C"),
    ]
    scores = [4.0, 3.0, 2.0, 1.0]
    costs = [evidence_tokens([item], tokenizer) for item in paths]
    budget = evidence_tokens(paths[:2], tokenizer)

    relation_selected = relation_only(paths, scores, budget, tokenizer, costs)
    assert relation_selected == paths[:2]

    shortest_selected = shortest_first(paths, budget, tokenizer, costs)
    assert evidence_tokens(shortest_selected, tokenizer) <= budget
    assert len(shortest_selected) == 2

    endpoint_selected = unit_endpoint(paths, scores, budget, tokenizer, costs)
    assert [item.endpoint for item in endpoint_selected] == ["A", "B"]

    chain_selected = unique_relation_chain(paths, scores, budget, tokenizer, costs)
    assert [item.endpoint for item in chain_selected] == ["A", "A"]
    assert len({directional_relation_key(item) for item in chain_selected}) == 2

    for selected in (
        relation_selected,
        shortest_selected,
        endpoint_selected,
        chain_selected,
    ):
        assert evidence_tokens(selected, tokenizer) <= budget

    print("status=PASS tests=selector_order,unique_endpoint,unique_chain,budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
