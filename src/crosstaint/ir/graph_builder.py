from __future__ import annotations

from typing import Any

from crosstaint.types import (
    IRNode,
    IREdge,
    DecodedEvent,
    NodeType,
    EdgeType,
    BridgeAdapterProtocol,
    EventId,
)
from crosstaint.ir.operators import BridgeIRTranslator


class IRGraphBuilder:
    def __init__(
        self, bridge_adapters: dict[str, BridgeAdapterProtocol] | None = None
    ) -> None:
        self._adapters: dict[str, BridgeAdapterProtocol] = bridge_adapters or {}
        self._translator = BridgeIRTranslator()
        self._nodes: dict[str, IRNode] = {}
        self._edges: dict[str, IREdge] = {}
        self._edge_counter: int = 0

    def _node_key(self, chain: str, address: str) -> str:
        return f"{chain}:{address.lower()}"

    def _ensure_node(
        self,
        chain: str,
        address: str,
        node_type: str = NodeType.EOA,
        timestamp: Any | None = None,
    ) -> str:
        node_key = self._node_key(chain, address)
        if node_key not in self._nodes:
            first_seen = timestamp if timestamp is not None else None
            node = IRNode(
                node_id=node_key,
                address=address.lower(),
                chain=chain,
                node_type=node_type,
                tags=frozenset(),
                first_seen=first_seen,
                degree_at_first_seen=0,
                in_value_total=0,
                out_value_total=0,
                bridge_count=0,
                dex_swap_count=0,
            )
            self._nodes[node_key] = node
        return node_key

    def _add_node(
        self,
        chain: str,
        address: str,
        node_type: str = NodeType.EOA,
        tags: frozenset[str] | None = None,
        timestamp: Any | None = None,
    ) -> IRNode:
        node_id = self._ensure_node(chain, address, node_type, timestamp)
        return self._nodes[node_id]

    def _update_node_stats(
        self,
        node_id: str,
        in_value: int = 0,
        out_value: int = 0,
        bridge_increment: int = 0,
        dex_increment: int = 0,
    ) -> None:
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        self._nodes[node_id] = IRNode(
            node_id=node.node_id,
            address=node.address,
            chain=node.chain,
            node_type=node.node_type,
            tags=node.tags,
            first_seen=node.first_seen,
            degree_at_first_seen=node.degree_at_first_seen,
            in_value_total=node.in_value_total + in_value,
            out_value_total=node.out_value_total + out_value,
            bridge_count=node.bridge_count + bridge_increment,
            dex_swap_count=node.dex_swap_count + dex_increment,
        )

    def _add_edge(self, edge: IREdge) -> bool:
        if edge.edge_id in self._edges:
            return False
        self._edges[edge.edge_id] = edge
        if edge.source_node in self._nodes:
            self._update_node_stats(
                edge.source_node,
                out_value=edge.value,
                bridge_increment=1 if edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE else 0,
                dex_increment=1 if edge.edge_type == EdgeType.DEX_SWAP else 0,
            )
        if edge.target_node in self._nodes:
            self._update_node_stats(
                edge.target_node,
                in_value=edge.value,
            )
        return True

    def _generate_edge_id(self, prefix: str = "edge") -> str:
        self._edge_counter += 1
        return f"{prefix}_{self._edge_counter:08d}"

    def _extract_address_from_event(
        self, event: DecodedEvent, field: str = "sender"
    ) -> str:
        if field in event.params:
            return str(event.params[field])
        topic_address = self._extract_topic_address(event, 1)
        if topic_address:
            return topic_address
        return event.emitter_address

    def _extract_topic_address(self, event: DecodedEvent, index: int) -> str:
        if index < len(event.topics):
            topic = event.topics[index]
            if len(topic) == 32:
                return "0x" + topic[12:].hex()
        return ""

    def _get_adapter_for_event(
        self, event: DecodedEvent
    ) -> BridgeAdapterProtocol | None:
        if event.bridge and event.bridge in self._adapters:
            return self._adapters[event.bridge]
        for adapter in self._adapters.values():
            if event.event_name in adapter.source_event_names() or \
               event.event_name in adapter.dest_event_names():
                return adapter
        return None

    def _classify_node_type(self, event: DecodedEvent) -> str:
        if bool(event.params.get("emitter_is_contract", False)):
            return NodeType.CONTRACT
        if any(k in event.params for k in ("router", "pool", "vault", "bridge")):
            return NodeType.CONTRACT
        return NodeType.EOA

    def _infer_node_type_from_params(self, params: dict[str, object]) -> str:
        if "is_contract" in params and params["is_contract"]:
            return NodeType.CONTRACT
        if any(k in params for k in ("router", "pool", "vault", "bridge")):
            return NodeType.CONTRACT
        return NodeType.EOA

    def _build_edge_from_events(
        self,
        source_event: DecodedEvent,
        dest_event: DecodedEvent,
        bridge_id: str,
        bridge_mode: str,
    ) -> IREdge | None:
        source_address = self._extract_address_from_event(source_event, "sender")
        dest_address = self._extract_address_from_event(dest_event, "recipient")
        if not dest_address:
            dest_address = self._extract_topic_address(dest_event, 3)
        if not dest_address:
            dest_address = dest_event.emitter_address

        source_node_id = self._node_key(source_event.chain, source_address)
        dest_node_id = self._node_key(dest_event.chain, dest_address)

        self._ensure_node(
            source_event.chain,
            source_address,
            NodeType.EOA,
            source_event.timestamp,
        )
        self._ensure_node(
            dest_event.chain,
            dest_address,
            NodeType.EOA,
            dest_event.timestamp,
        )

        edge = self._translator.translate(
            source_event=source_event,
            dest_event=dest_event,
            bridge_id=bridge_id,
            bridge_mode=bridge_mode,
            source_node_id=source_node_id,
            dest_node_id=dest_node_id,
        )
        return edge

    def build(
        self,
        ir_events: list[IREdge],
        nodes: list[IRNode],
    ) -> tuple[dict[str, IRNode], dict[str, IREdge]]:
        for node in nodes:
            self._nodes[node.node_id] = node
        for edge in ir_events:
            self._add_edge(edge)
        return dict(self._nodes), dict(self._edges)

    def build_from_events(
        self,
        raw_events: list[DecodedEvent],
        pairs: list[tuple[DecodedEvent, DecodedEvent]],
    ) -> tuple[dict[str, IRNode], dict[str, IREdge]]:
        for event in raw_events:
            node_type = self._infer_node_type_from_params(event.params)
            self._ensure_node(
                event.chain,
                event.emitter_address,
                node_type,
                event.timestamp,
            )
            for adapter in self._adapters.values():
                edge: IREdge | None = None
                if event.event_name in adapter.source_event_names():
                    if adapter.bridge_mode() in ("lock_mint", "pool"):
                        edge = adapter.decode_lock(event)
                    elif adapter.bridge_mode() == "burn_mint":
                        edge = adapter.decode_burn(event)
                elif event.event_name in adapter.dest_event_names():
                    if adapter.bridge_mode() == "intent":
                        edge = adapter.decode_intent_fill(event)
                    else:
                        edge = adapter.decode_mint(event)
                        if edge is None:
                            edge = adapter.decode_release(event)
                if edge is not None and edge.edge_id:
                    self._add_edge(edge)

        for source_event, dest_event in pairs:
            adapter = self._get_adapter_for_event(source_event)
            if adapter is None:
                for a in self._adapters.values():
                    if a.is_pair(source_event, dest_event):
                        adapter = a
                        break
            if adapter is None:
                continue
            bridge_id = adapter.bridge_id()
            bridge_mode = adapter.bridge_mode()

            source_address = self._extract_address_from_event(source_event, "sender")
            dest_address = self._extract_address_from_event(dest_event, "recipient")
            if not dest_address:
                dest_address = self._extract_topic_address(dest_event, 3)
            if not dest_address:
                dest_address = dest_event.emitter_address

            source_node_id = self._node_key(source_event.chain, source_address)
            dest_node_id = self._node_key(dest_event.chain, dest_address)

            self._ensure_node(
                source_event.chain,
                source_address,
                NodeType.EOA,
                source_event.timestamp,
            )
            self._ensure_node(
                dest_event.chain,
                dest_address,
                NodeType.EOA,
                dest_event.timestamp,
            )

            edge = self._translator.translate(
                source_event=source_event,
                dest_event=dest_event,
                bridge_id=bridge_id,
                bridge_mode=bridge_mode,
                source_node_id=source_node_id,
                dest_node_id=dest_node_id,
            )
            if edge is not None:
                self._add_edge(edge)

        return dict(self._nodes), dict(self._edges)

    def get_node(self, node_key: str) -> IRNode | None:
        return self._nodes.get(node_key)

    def get_edge(self, edge_key: str) -> IREdge | None:
        return self._edges.get(edge_key)

    def get_nodes_by_chain(self, chain: str) -> list[IRNode]:
        return [
            node
            for node in self._nodes.values()
            if node.chain == chain
        ]

    def get_edges_by_type(self, edge_type: str) -> list[IREdge]:
        return [
            edge
            for edge in self._edges.values()
            if edge.edge_type == edge_type
        ]

    def get_node_degree(self, node_key: str) -> tuple[int, int]:
        in_degree = sum(
            1 for e in self._edges.values() if e.target_node == node_key
        )
        out_degree = sum(
            1 for e in self._edges.values() if e.source_node == node_key
        )
        return in_degree, out_degree

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._edge_counter = 0
