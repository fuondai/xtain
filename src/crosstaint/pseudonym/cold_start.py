"""Cold-start fallback for addresses with insufficient graph history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PseudonymCandidate:
    address: str
    chain: str
    matcher_confidence: float
    temporal_distance_blocks: int
    tag_overlap_score: float


class ColdStartFallback:
    """Fallback pseudonym resolver for addresses with insufficient graph history."""

    def __init__(
        self,
        default_confidence: float = 0.5,
        min_tag_overlap_for_match: float = 0.3,
    ) -> None:
        self._default_confidence = default_confidence
        self._min_tag_overlap = min_tag_overlap_for_match
        self._address_cache: dict[tuple[str, str], PseudonymCandidate] = {}

    def resolve(
        self,
        address: str,
        chain: str,
        candidates: list[dict[str, Any]],
        matcher_scores: dict[str, float],
        query_tags: list[str] | None = None,
        query_first_block: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates and not matcher_scores:
            return []

        self_tags = set(query_tags or [])
        reference = {"first_block": query_first_block} if query_first_block is not None else {}
        scored_candidates: list[tuple[float, dict[str, Any]]] = []
        for cand in candidates:
            cand_address = str(cand.get("address", ""))
            cand_tags = set(cand.get("tags", []))

            matcher_confidence = float(matcher_scores.get(cand_address, self._default_confidence))
            tag_overlap = self._compute_tag_overlap(self_tags, cand_tags)
            temporal_penalty = self._temporal_penalty(cand, reference)
            temporal_penalty = max(0.0, min(1.0, temporal_penalty))

            combined_score = (
                0.50 * matcher_confidence
                + 0.35 * tag_overlap
                + 0.15 * (1.0 - temporal_penalty)
            )
            scored_candidates.append((combined_score, cand))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, cand in scored_candidates:
            results.append({
                **cand,
                "resolution_score": float(score),
                "is_cold_start": True,
                "matcher_confidence": float(matcher_scores.get(cand.get("address", ""), self._default_confidence)),
            })
        return results

    def _compute_tag_overlap(self, tags_a: set[str], tags_b: set[str]) -> float:
        if not tags_a or not tags_b:
            return 0.0
        intersection = len(tags_a & tags_b)
        union = len(tags_a | tags_b)
        return float(intersection) / float(union) if union > 0 else 0.0

    def _temporal_penalty(
        self,
        candidate: dict[str, Any],
        reference: dict[str, Any],
    ) -> float:
        cand_block = int(candidate.get("first_block", 0))
        ref_block = int(reference.get("first_block", cand_block))
        block_diff = abs(cand_block - ref_block)
        return float(min(block_diff, 10000)) / 10000.0

    def batch_resolve(
        self,
        queries: list[tuple[str, str, list[dict[str, Any]], dict[str, float]]],
    ) -> list[list[dict[str, Any]]]:
        return [
            self.resolve(address, chain, candidates, scores)
            for address, chain, candidates, scores in queries
        ]
