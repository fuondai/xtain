from __future__ import annotations

import logging
from typing import AsyncIterator

from crosstaint.types import RawEvent, DecodedEvent, ChainClientProtocol


logger = logging.getLogger(__name__)


class EventStream:
    def __init__(self, chain_clients: dict[str, ChainClientProtocol]) -> None:
        self._clients = chain_clients

    async def stream(
        self,
        start_block: int,
        end_block: int,
        bridge: str,
    ) -> AsyncIterator[RawEvent]:
        for chain, client in self._clients.items():
            async for event in self._stream_chain(
                client, chain, start_block, end_block, bridge
            ):
                yield event

    async def _stream_chain(
        self,
        client: ChainClientProtocol,
        chain: str,
        start_block: int,
        end_block: int,
        bridge: str,
    ) -> AsyncIterator[RawEvent]:
        try:
            events = await client.get_block_range(start_block, end_block)
            for event in events:
                if bridge and event.bridge != bridge:
                    continue
                yield event
        except Exception:
            logger.exception("failed to stream %s events from %s", bridge or "all", chain)
            raise

    async def stream_range(
        self,
        chain: str,
        start: int,
        end: int,
    ) -> list[DecodedEvent]:
        client = self._clients.get(chain)
        if client is None:
            return []

        try:
            raw_events = await client.get_block_range(start, end)
        except Exception:
            logger.exception("failed to stream range %s:%s-%s", chain, start, end)
            raise

        decoded = []
        for raw_event in raw_events:
            decoded_event = self._decode_raw_event(raw_event)
            if decoded_event is not None:
                decoded.append(decoded_event)

        return decoded

    def _decode_raw_event(self, event: RawEvent) -> DecodedEvent | None:
        return DecodedEvent(
            event_id=event.event_id,
            chain=event.chain,
            block_number=event.block_number,
            timestamp=event.timestamp,
            tx_hash=event.tx_hash,
            event_name=event.event_name,
            emitter_address=event.emitter_address,
            topics=event.topics,
            params=event.params,
            bridge=event.bridge,
            selector=str(event.params.get("selector", "")),
            value=self._extract_value(event.params),
            bridge_family=event.bridge or "",
        )

    def _extract_value(self, params: dict[str, object]) -> int:
        for key in ("amount", "value", "qty", "tokensAmount"):
            value = params.get(key)
            if value is None:
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value, 16) if value.startswith("0x") else int(value)
                except ValueError:
                    continue
        return 0

    async def stream_all_chains(
        self,
        start_block: int,
        end_block: int,
    ) -> AsyncIterator[DecodedEvent]:
        for event_chain in self._clients:
            async for event in self._stream_chain(
                self._clients[event_chain],
                event_chain,
                start_block,
                end_block,
                bridge="",
            ):
                decoded = self._decode_raw_event(event)
                if decoded is not None:
                    yield decoded

    def get_supported_chains(self) -> list[str]:
        return list(self._clients.keys())

    def add_client(self, chain: str, client: ChainClientProtocol) -> None:
        self._clients[chain] = client

    def remove_client(self, chain: str) -> bool:
        if chain in self._clients:
            del self._clients[chain]
            return True
        return False
