from __future__ import annotations

from typing import Any

from crosstaint.types import (
    DecodedEvent,
    IREdge,
    EdgeType,
    TaintOperator,
)


class BridgeIRTranslator:
    def _extract_asset(self, event: DecodedEvent) -> str | None:
        params = event.params
        for key in ("token", "asset", "erc20", "symbol", "currency"):
            if key in params:
                return str(params[key])
        return None

    def _extract_value(self, event: DecodedEvent) -> int:
        params = event.params
        for key in ("amount", "value", "qty", "tokensAmount"):
            if key in params:
                val = params[key]
                if isinstance(val, int):
                    return val
                if isinstance(val, float):
                    return int(val)
                if isinstance(val, str):
                    try:
                        return int(val, 16) if val.startswith("0x") else int(val)
                    except ValueError:
                        continue
        return 0

    def translate(
        self,
        source_event: DecodedEvent,
        dest_event: DecodedEvent,
        bridge_id: str,
        bridge_mode: str,
        source_node_id: str,
        dest_node_id: str,
    ) -> IREdge | None:
        operator = TaintOperator.LOCK
        if bridge_mode == "burn_mint":
            operator = TaintOperator.BURN
        elif bridge_mode == "pool":
            operator = TaintOperator.RELEASE
        elif bridge_mode == "intent":
            operator = TaintOperator.INTENT_FILL

        value = self._extract_value(source_event)
        if value == 0:
            value = self._extract_value(dest_event)

        asset = self._extract_asset(source_event) or self._extract_asset(dest_event)

        edge_id = f"{bridge_id}_{source_event.event_id}_{dest_event.event_id}"

        return IREdge(
            edge_id=edge_id,
            source_node=source_node_id,
            target_node=dest_node_id,
            edge_type=EdgeType.CROSS_CHAIN_BRIDGE,
            operator=operator,
            bridge=bridge_id,
            source_event=source_event.event_id,
            target_event=dest_event.event_id,
            value=value,
            asset=asset,
            timestamp=dest_event.timestamp,
            block_number=dest_event.block_number,
            fee_bound_pct=0.01,
        )
