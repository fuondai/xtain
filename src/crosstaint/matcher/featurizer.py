"""Bridge event featurizer: converts DecodedEvent into fixed-dimensional feature vectors."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from crosstaint.config import Config
from crosstaint.types import DecodedEvent


class EventFeaturizer:
    FEAT_DIM = 64

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        selector_matrix = self._rng.standard_normal((256, 32)).astype(np.float32)
        selector_matrix /= np.linalg.norm(selector_matrix, axis=1, keepdims=True) + 1e-8
        self._selector_proj = selector_matrix
        topic_matrix = self._rng.standard_normal((256, 32)).astype(np.float32)
        topic_matrix /= np.linalg.norm(topic_matrix, axis=1, keepdims=True) + 1e-8
        self._topic_proj = topic_matrix
        chain_matrix = self._rng.standard_normal((32, 16)).astype(np.float32)
        chain_matrix /= np.linalg.norm(chain_matrix, axis=1, keepdims=True) + 1e-8
        self._chain_proj = chain_matrix
        bridge_matrix = self._rng.standard_normal((16, 16)).astype(np.float32)
        bridge_matrix /= np.linalg.norm(bridge_matrix, axis=1, keepdims=True) + 1e-8
        self._bridge_proj = bridge_matrix

    def featurize(self, event: DecodedEvent) -> np.ndarray:
        selector_emb = self._embed_selector(event.selector)
        topic_emb = self._embed_topics(event.topics)
        chain_emb = self._embed_chain(event.chain)
        bridge_emb = self._embed_bridge(event.bridge)
        param_emb = self._embed_params(event.params)
        positional_emb = self._embed_positional(event.block_number)
        combined = np.concatenate([
            selector_emb,
            topic_emb,
            chain_emb,
            bridge_emb,
            param_emb,
            positional_emb,
        ])
        if combined.shape[0] > self.FEAT_DIM:
            combined = combined[:self.FEAT_DIM]
        elif combined.shape[0] < self.FEAT_DIM:
            pad = np.zeros(self.FEAT_DIM - combined.shape[0], dtype=np.float32)
            combined = np.concatenate([combined, pad])
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        return combined.astype(np.float32)

    def featurize_batch(self, events: list[DecodedEvent]) -> np.ndarray:
        if not events:
            return np.zeros((0, self.FEAT_DIM), dtype=np.float32)
        return np.stack([self.featurize(e) for e in events], axis=0)

    def _embed_selector(self, selector: str | None) -> np.ndarray:
        if not selector:
            return np.zeros(32, dtype=np.float32)
        cleaned = selector.replace("0x", "").lower()
        if len(cleaned) % 2 != 0:
            cleaned = "0" + cleaned
        try:
            bytes_data = bytes.fromhex(cleaned[:64])
        except ValueError:
            bytes_data = cleaned.encode("utf-8")[:32]
        indices = list(bytes_data)
        indices = [min(255, max(0, i)) for i in indices]
        while len(indices) < 32:
            indices.append(0)
        return self._selector_proj[np.array(indices, dtype=np.intp)].sum(axis=0)

    def _embed_topics(self, topics: list[str]) -> np.ndarray:
        if not topics:
            return np.zeros(32, dtype=np.float32)
        vectors: list[np.ndarray] = []
        for topic in topics[:4]:
            vectors.append(self._topic_hash(topic))
        while len(vectors) < 4:
            vectors.append(np.zeros(32, dtype=np.float32))
        stacked = np.stack(vectors[:4], axis=0)
        return np.mean(stacked, axis=0)

    def _topic_hash(self, topic: str) -> np.ndarray:
        cleaned = topic.replace("0x", "").lower()
        if len(cleaned) % 2 != 0:
            cleaned = "0" + cleaned
        try:
            raw_bytes = bytes.fromhex(cleaned[:64])
        except ValueError:
            raw_bytes = cleaned.encode("utf-8")[:32]
        indices = [min(255, max(0, b)) for b in raw_bytes]
        while len(indices) < 32:
            indices.append(0)
        return self._selector_proj[np.array(indices[:32], dtype=np.intp)].sum(axis=0)

    def _embed_chain(self, chain: str) -> np.ndarray:
        chain_code = self._string_code(chain)
        indices = [(chain_code >> (i * 8)) & 0xFF for i in range(4)]
        while len(indices) < 4:
            indices.append(0)
        emb = self._chain_proj[np.array(indices[:4], dtype=np.intp)].sum(axis=0)
        return emb

    def _embed_bridge(self, bridge: str | None) -> np.ndarray:
        if not bridge:
            return np.zeros(16, dtype=np.float32)
        bridge_code = self._string_code(bridge)
        indices = [(bridge_code >> (i * 8)) & 0xFF for i in range(4)]
        while len(indices) < 4:
            indices.append(0)
        emb = self._bridge_proj[np.array(indices[:4], dtype=np.intp)].sum(axis=0)
        return emb

    def _embed_params(self, params: dict[str, Any]) -> np.ndarray:
        if not params:
            return np.zeros(16, dtype=np.float32)
        keys = sorted(params.keys())[:8]
        hashes: list[float] = []
        for key in keys:
            h = self._string_code(key)
            val = params[key]
            if isinstance(val, (int, float)):
                hashes.append(float(val) % 1.0)
            else:
                hashes.append(float(h % 1000) / 1000.0)
        while len(hashes) < 8:
            hashes.append(0.0)
        arr = np.array(hashes[:8], dtype=np.float32)
        mean = arr.mean()
        std = arr.std() + 1e-8
        return ((arr - mean) / std).astype(np.float32) * 2.0

    def _embed_positional(self, block_number: int) -> np.ndarray:
        freq = np.array([2.0 ** (-i / 16.0) for i in range(8)], dtype=np.float32)
        phase = np.array([float((block_number >> i) & 0xFF) / 128.0 * 3.14159 for i in range(8)], dtype=np.float32)
        return (np.sin(freq + phase) * 0.5 + 0.5).astype(np.float32)

    def _string_code(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], base=16) & 0x7FFFFFFF

    def get_projection_matrices(self) -> dict[str, np.ndarray]:
        return {
            "selector_proj": self._selector_proj,
            "topic_proj": self._topic_proj,
            "chain_proj": self._chain_proj,
            "bridge_proj": self._bridge_proj,
        }
