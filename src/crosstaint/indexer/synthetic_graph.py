from __future__ import annotations

import datetime
import hashlib
from typing import Any

from crosstaint.types import (
    IRNode,
    IREdge,
    SyntheticTrajectory,
    SyntheticHop,
    EdgeType,
    NodeType,
    AddressTag,
    BridgeMode,
)
from crosstaint.synth import BRIDGE_MODES


TORNADO_CASH_MIXERS: dict[str, tuple[str, float]] = {
    "0xd90e2f925DA726b50C4Ed8D0Fb90Ad053324F31b": ("ethereum", 0.1),
    "0x12D66f87A04A9E220743712cE6d9bB1B5616B8Fc": ("ethereum", 1.0),
    "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936": ("ethereum", 10.0),
    "0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF": ("ethereum", 100.0),
}


class SyntheticGraphBuilder:
    def __init__(
        self,
        trajectories: list[SyntheticTrajectory],
        exploit_addresses: list[str] | None = None,
    ) -> None:
        self._trajectories = trajectories
        self._exploit_addresses = set(exploit_addresses or [])
        self._nodes: dict[str, IRNode] = {}
        self._edges: dict[str, IREdge] = {}
        self._edge_counter = 0
        self._timestamp_base = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

    def build(self) -> tuple[dict[str, IRNode], dict[str, IREdge]]:
        self._nodes.clear()
        self._edges.clear()
        self._edge_counter = 0

        for trajectory in self._trajectories:
            self._process_trajectory(trajectory)

        self._inject_tornado_mixers()

        return dict(self._nodes), dict(self._edges)

    def _process_trajectory(self, trajectory: SyntheticTrajectory) -> None:
        addresses = list(trajectory.addresses)
        if not addresses:
            return

        origin_addr = addresses[0]
        first_chain = trajectory.hops[0].from_chain if trajectory.hops else "ethereum"
        self._ensure_node(
            address=origin_addr,
            chain=first_chain,
            tags=self._get_address_tags(origin_addr),
        )
        cumulative_time = 0.0
        for hop_idx, hop in enumerate(trajectory.hops):
            if hop_idx + 1 >= len(addresses):
                break

            source_addr = addresses[hop_idx]
            dest_addr = addresses[hop_idx + 1]
            source_chain = hop.from_chain
            dest_chain = hop.to_chain

            self._ensure_node(
                address=source_addr,
                chain=source_chain,
                tags=self._get_address_tags(source_addr),
            )

            self._ensure_node(
                address=dest_addr,
                chain=dest_chain,
                tags=self._get_address_tags(dest_addr),
            )

            edge_type = (
                EdgeType.CROSS_CHAIN_BRIDGE
                if hop.from_chain != hop.to_chain
                else EdgeType.INTRA_CHAIN_TRANSFER
            )

            self._create_edge(
                source=source_addr,
                target=dest_addr,
                source_chain=source_chain,
                target_chain=dest_chain,
                edge_type=edge_type,
                bridge=hop.bridge,
                operator=hop.operator,
                value=hop.value,
                asset=hop.asset,
                timestamp=self._timestamp_base + datetime.timedelta(seconds=cumulative_time),
                hop=hop,
            )

            cumulative_time += hop.inter_arrival_seconds

    def _node_key(self, chain: str, address: str) -> str:
        return f"{chain}:{address.lower()}"

    def _ensure_node(
        self,
        address: str,
        chain: str,
        tags: frozenset[str] | None = None,
    ) -> None:
        node_key = self._node_key(chain, address)
        if node_key in self._nodes:
            existing = self._nodes[node_key]
            new_tags = existing.tags | (tags or frozenset())
            updated = IRNode(
                node_id=existing.node_id,
                address=existing.address,
                chain=existing.chain,
                node_type=existing.node_type,
                tags=new_tags,
                first_seen=existing.first_seen,
                degree_at_first_seen=existing.degree_at_first_seen,
                in_value_total=existing.in_value_total,
                out_value_total=existing.out_value_total,
                bridge_count=existing.bridge_count,
                dex_swap_count=existing.dex_swap_count,
            )
            self._nodes[node_key] = updated
            return

        node_id = node_key
        node_type = self._infer_node_type(address, tags)
        node_tags = tags or frozenset()

        if address in self._exploit_addresses:
            node_tags = node_tags | frozenset([AddressTag.KNOWN_ATTACKER])

        node = IRNode(
            node_id=node_id,
            address=address,
            chain=chain,
            node_type=node_type,
            tags=node_tags,
            first_seen=self._timestamp_base,
            degree_at_first_seen=0,
            in_value_total=0,
            out_value_total=0,
            bridge_count=0,
            dex_swap_count=0,
        )
        self._nodes[node_key] = node

    def _infer_node_type(self, address: str, tags: frozenset[str] | None = None) -> str:
        if tags and AddressTag.MIXER_DEPOSIT in tags:
            return NodeType.MIXER

        if address.lower() in [m.lower() for m in TORNADO_CASH_MIXERS]:
            return NodeType.MIXER

        return NodeType.EOA

    def _get_address_tags(self, address: str) -> frozenset[str]:
        tags: list[str] = []

        if address in self._exploit_addresses:
            tags.append(AddressTag.KNOWN_ATTACKER)

        if address.lower() in [m.lower() for m in TORNADO_CASH_MIXERS]:
            tags.append(AddressTag.MIXER_DEPOSIT)

        return frozenset(tags)

    def _create_edge(
        self,
        source: str,
        target: str,
        source_chain: str,
        target_chain: str,
        edge_type: str,
        bridge: str | None,
        operator: str,
        value: int,
        asset: str,
        timestamp: datetime.datetime,
        hop: SyntheticHop | None = None,
    ) -> None:
        edge_id = f"edge_{self._edge_counter}"
        self._edge_counter += 1

        source_key = self._node_key(source_chain, source.lower())
        target_key = self._node_key(target_chain, target.lower())

        source_node = self._nodes.get(source_key)
        target_node = self._nodes.get(target_key)

        if source_node:
            updated_source = IRNode(
                node_id=source_node.node_id,
                address=source_node.address,
                chain=source_node.chain,
                node_type=source_node.node_type,
                tags=source_node.tags,
                first_seen=source_node.first_seen,
                degree_at_first_seen=source_node.degree_at_first_seen,
                in_value_total=source_node.in_value_total,
                out_value_total=source_node.out_value_total + value,
                bridge_count=(
                    source_node.bridge_count + (1 if bridge else 0)
                ),
                dex_swap_count=source_node.dex_swap_count,
            )
            self._nodes[source_key] = updated_source

        if target_node:
            updated_target = IRNode(
                node_id=target_node.node_id,
                address=target_node.address,
                chain=target_node.chain,
                node_type=target_node.node_type,
                tags=target_node.tags,
                first_seen=target_node.first_seen,
                degree_at_first_seen=target_node.degree_at_first_seen,
                in_value_total=target_node.in_value_total + value,
                out_value_total=target_node.out_value_total,
                bridge_count=(
                    target_node.bridge_count + (1 if bridge else 0)
                ),
                dex_swap_count=target_node.dex_swap_count,
            )
            self._nodes[target_key] = updated_target

        edge = IREdge(
            edge_id=edge_id,
            source_node=source_key,
            target_node=target_key,
            edge_type=edge_type,
            operator=operator,
            bridge=bridge or "",
            source_event=None,
            target_event=None,
            value=value,
            asset=asset,
            timestamp=timestamp,
            block_number=int(timestamp.timestamp() // 12),
            fee_bound_pct=self._fee_bound_pct(bridge),
        )
        self._edges[edge_id] = edge

    def _fee_bound_pct(self, bridge: str | None) -> float:
        mode = BRIDGE_MODES.get(bridge or "", BridgeMode.LOCK_MINT)
        if mode in (BridgeMode.LOCK_MINT, BridgeMode.BURN_MINT):
            return 0.01
        if mode == BridgeMode.POOL:
            return 0.05
        if mode == BridgeMode.INTENT:
            return 0.08
        return 0.01

    def _inject_tornado_mixers(self) -> None:
        for mixer_addr, (chain, _) in TORNADO_CASH_MIXERS.items():
            node_key = self._node_key(chain, mixer_addr.lower())
            if node_key not in self._nodes:
                mixer_node = IRNode(
                    node_id=node_key,
                    address=mixer_addr.lower(),
                    chain=chain,
                    node_type=NodeType.MIXER,
                    tags=frozenset([AddressTag.MIXER_DEPOSIT]),
                    first_seen=self._timestamp_base - datetime.timedelta(days=30),
                    degree_at_first_seen=0,
                    in_value_total=0,
                    out_value_total=0,
                    bridge_count=0,
                    dex_swap_count=0,
                )
                self._nodes[node_key] = mixer_node

    def create_exploit_case(
        self,
        origin: str,
        chain: str,
        value: int,
        hops: list[tuple[str, str, str]],
    ) -> tuple[list[IRNode], list[IREdge]]:
        nodes = []
        edges = []

        origin_key = self._node_key(chain, origin.lower())
        origin_node = self._create_exploit_node(origin_key, chain)
        nodes.append(origin_node)
        self._nodes[origin_key] = origin_node

        cumulative_time = 0.0
        current_addr = origin.lower()
        current_chain = chain

        for hop_idx, (_, dst_chain, bridge) in enumerate(hops):
            source_chain = current_chain

            next_key = self._generate_exploit_address_key(
                origin=origin,
                source_chain=source_chain,
                dest_chain=dst_chain,
                bridge=bridge,
                index=hop_idx,
            )
            next_addr = next_key.split(":", 1)[1]

            next_node = self._create_exploit_node(next_key, dst_chain)
            nodes.append(next_node)
            self._nodes[next_key] = next_node

            edge_type = (
                EdgeType.CROSS_CHAIN_BRIDGE
                if source_chain != dst_chain
                else EdgeType.INTRA_CHAIN_TRANSFER
            )

            edge = IREdge(
                edge_id=self._exploit_edge_id(origin, source_chain, dst_chain, bridge, hop_idx),
                source_node=self._node_key(source_chain, current_addr),
                target_node=next_key,
                edge_type=edge_type,
                operator="Lock",
                bridge=bridge,
                source_event=None,
                target_event=None,
                value=value,
                asset="ETH",
                timestamp=self._timestamp_base + datetime.timedelta(seconds=cumulative_time),
                block_number=hop_idx + 1,
                fee_bound_pct=self._fee_bound_pct(bridge),
            )
            edges.append(edge)
            self._edges[edge.edge_id] = edge
            self._update_node_values(edge)

            current_addr = next_addr
            current_chain = dst_chain
            cumulative_time += 3600.0

        return nodes, edges

    def _exploit_edge_id(
        self,
        origin: str,
        source_chain: str,
        dest_chain: str,
        bridge: str,
        index: int,
    ) -> str:
        digest = hashlib.sha256(
            f"exploit-edge:{origin.lower()}:{source_chain}:{dest_chain}:{bridge}:{index}".encode()
        ).hexdigest()
        return f"exploit_edge_{digest[:16]}"

    def _create_exploit_node(self, node_key: str, chain: str) -> IRNode:
        return IRNode(
            node_id=node_key,
            address=node_key.split(":", 1)[1] if ":" in node_key else node_key,
            chain=chain,
            node_type=NodeType.EOA,
            tags=frozenset([AddressTag.KNOWN_ATTACKER]),
            first_seen=self._timestamp_base,
            degree_at_first_seen=0,
            in_value_total=0,
            out_value_total=0,
            bridge_count=0,
            dex_swap_count=0,
        )

    def _generate_exploit_address_key(
        self,
        origin: str,
        source_chain: str,
        dest_chain: str,
        bridge: str,
        index: int,
    ) -> str:
        data = f"exploit:{origin.lower()}:{source_chain}:{dest_chain}:{bridge}:{index}".encode()
        hash_digest = hashlib.sha256(data).hexdigest()
        return f"{dest_chain}:0x{hash_digest[:40]}"

    def _update_node_values(self, edge: IREdge) -> None:
        if edge.source_node in self._nodes:
            source = self._nodes[edge.source_node]
            self._nodes[edge.source_node] = IRNode(
                node_id=source.node_id,
                address=source.address,
                chain=source.chain,
                node_type=source.node_type,
                tags=source.tags,
                first_seen=source.first_seen,
                degree_at_first_seen=source.degree_at_first_seen,
                in_value_total=source.in_value_total,
                out_value_total=source.out_value_total + edge.value,
                bridge_count=source.bridge_count + (1 if edge.bridge else 0),
                dex_swap_count=source.dex_swap_count,
            )
        if edge.target_node in self._nodes:
            target = self._nodes[edge.target_node]
            self._nodes[edge.target_node] = IRNode(
                node_id=target.node_id,
                address=target.address,
                chain=target.chain,
                node_type=target.node_type,
                tags=target.tags,
                first_seen=target.first_seen,
                degree_at_first_seen=target.degree_at_first_seen,
                in_value_total=target.in_value_total + edge.value,
                out_value_total=target.out_value_total,
                bridge_count=target.bridge_count + (1 if edge.bridge else 0),
                dex_swap_count=target.dex_swap_count,
            )

    def add_noise_trajectories(
        self,
        trajectories: list[SyntheticTrajectory],
        noise_ratio: float = 1.0,
    ) -> None:
        num_noise = int(len(trajectories) * noise_ratio)
        selected_trajectories = trajectories[:num_noise]
        for trajectory in selected_trajectories:
            self._process_trajectory(trajectory)

    def add_mixer_node(self, mixer_addr: str, chain: str) -> IRNode:
        node_key = self._node_key(chain, mixer_addr.lower())
        if node_key in self._nodes:
            return self._nodes[node_key]

        mixer_node = IRNode(
            node_id=node_key,
            address=mixer_addr.lower(),
            chain=chain,
            node_type=NodeType.MIXER,
            tags=frozenset([AddressTag.MIXER_DEPOSIT]),
            first_seen=self._timestamp_base - datetime.timedelta(days=30),
            degree_at_first_seen=0,
            in_value_total=0,
            out_value_total=0,
            bridge_count=0,
            dex_swap_count=0,
        )
        self._nodes[node_key] = mixer_node
        return mixer_node

    def export_graph(self) -> tuple[dict[str, Any], dict[str, Any]]:
        nodes_export = {}
        for addr, node in self._nodes.items():
            nodes_export[addr] = {
                "node_id": node.node_id,
                "address": node.address,
                "chain": node.chain,
                "node_type": node.node_type,
                "tags": list(node.tags),
                "first_seen": (
                    node.first_seen.isoformat() if node.first_seen else None
                ),
                "degree_at_first_seen": node.degree_at_first_seen,
                "in_value_total": str(node.in_value_total),
                "out_value_total": str(node.out_value_total),
                "bridge_count": node.bridge_count,
                "dex_swap_count": node.dex_swap_count,
            }

        edges_export = {}
        for edge_id, edge in self._edges.items():
            edges_export[edge_id] = {
                "edge_id": edge.edge_id,
                "source_node": edge.source_node,
                "target_node": edge.target_node,
                "edge_type": edge.edge_type,
                "operator": edge.operator,
                "bridge": edge.bridge,
                "source_event": edge.source_event,
                "target_event": edge.target_event,
                "value": str(edge.value),
                "asset": edge.asset,
                "timestamp": edge.timestamp.isoformat(),
                "block_number": edge.block_number,
                "fee_bound_pct": edge.fee_bound_pct,
            }

        return nodes_export, edges_export

    def snapshot(self) -> tuple[dict[str, IRNode], dict[str, IREdge]]:
        return dict(self._nodes), dict(self._edges)

    def get_statistics(self) -> dict[str, Any]:
        chain_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        edge_type_counts: dict[str, int] = {}
        bridge_counts: dict[str, int] = {}

        for node in self._nodes.values():
            chain_counts[node.chain] = chain_counts.get(node.chain, 0) + 1
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        for edge in self._edges.values():
            edge_type_counts[edge.edge_type] = (
                edge_type_counts.get(edge.edge_type, 0) + 1
            )
            if edge.bridge:
                bridge_counts[edge.bridge] = (
                    bridge_counts.get(edge.bridge, 0) + 1
                )

        total_value = sum(e.value for e in self._edges.values())

        return {
            "num_nodes": len(self._nodes),
            "num_edges": len(self._edges),
            "chain_distribution": chain_counts,
            "node_type_distribution": type_counts,
            "edge_type_distribution": edge_type_counts,
            "bridge_distribution": bridge_counts,
            "total_value": total_value,
        }
