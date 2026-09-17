from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from crosstaint.types import (
    DecodedEvent,
    IREdge,
    BridgeAdapterProtocol,
    EdgeType,
    TaintOperator,
)

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class WormholeEventParams:
    amount: int
    sender_address: str
    recipient_address: str
    source_chain: int
    dest_chain: int


@dataclass(frozen=True, slots=True)
class LayerZeroEventParams:
    destination_chain: int
    destination_address: str
    amount: int
    sender_address: str


@dataclass(frozen=True, slots=True)
class MultichainEventParams:
    token: str
    amount: int
    sender: str
    recipient: str
    dest_chain: int


@dataclass(frozen=True, slots=True)
class StargateEventParams:
    amount: int
    src_chain: int
    dst_chain: int
    sender: str
    to: str


@dataclass(frozen=True, slots=True)
class AcrossEventParams:
    amount: int
    origin_token: str
    recipient: str
    depositor: str
    destination_chain_id: int


@dataclass(frozen=True, slots=True)
class HopEventParams:
    amount: int
    recipient: str
    sender: str
    chain_id: int


@dataclass(frozen=True, slots=True)
class CBridgeEventParams:
    amount: int
    sender: str
    receiver: str
    destination_chain_id: int
    destination_id: str


@dataclass(frozen=True, slots=True)
class SynapseEventParams:
    token: str
    amount: int
    to: str
    from_: str
    chain_id: int


@dataclass(frozen=True, slots=True)
class HyperlaneEventParams:
    sender: str
    recipient: str
    amount: int
    dest_domain: int
    origin_domain: int


class BaseBridgeAdapter(ABC):
    @abstractmethod
    def bridge_id(self) -> str: ...

    @abstractmethod
    def bridge_mode(self) -> str: ...

    def source_event_names(self) -> tuple[str, ...]:
        return ()

    def dest_event_names(self) -> tuple[str, ...]:
        return ()

    def is_source_event(self, event: DecodedEvent) -> bool:
        return event.event_name in self.source_event_names()

    def is_dest_event(self, event: DecodedEvent) -> bool:
        return event.event_name in self.dest_event_names()

    def _get_int(self, params: dict[str, object], key: str) -> int:
        val = params.get(key)
        if val is None:
            return 0
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            try:
                return int(val, 16) if val.startswith("0x") else int(val)
            except ValueError:
                return 0
        return 0

    def _get_str(self, params: dict[str, object], key: str) -> str:
        val = params.get(key)
        return str(val) if val is not None else ""

    def _get_topic_address(self, topics: tuple[bytes, ...], index: int) -> str:
        if index < len(topics):
            topic = topics[index]
            if len(topic) == 32:
                return "0x" + topic[12:].hex()
        return ""

    def _node_id(self, chain: str, address: str) -> str:
        return f"{chain}:{address.lower()}"

    def _create_bridge_edge(
        self,
        edge_id: str,
        source_node: str,
        target_node: str,
        operator: str,
        event: DecodedEvent,
        amount: int,
    ) -> IREdge:
        from crosstaint.ir.operators import BridgeIRTranslator
        translator = BridgeIRTranslator()
        asset = translator._extract_asset(event)
        return IREdge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            edge_type=EdgeType.CROSS_CHAIN_BRIDGE,
            operator=operator,
            bridge=self.bridge_id(),
            source_event=event.event_id,
            target_event=None,
            value=amount,
            asset=asset,
            timestamp=event.timestamp,
            block_number=event.block_number,
            fee_bound_pct=0.01,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        return None

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        return None

    def decode_burn(self, event: DecodedEvent) -> IREdge | None:
        return None

    def decode_release(self, event: DecodedEvent) -> IREdge | None:
        return None

    def decode_intent_fill(self, event: DecodedEvent) -> IREdge | None:
        return None

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if source.chain == dest.chain:
            return False
        if not self.is_source_event(source):
            return False
        if not self.is_dest_event(dest):
            return False
        return True


class WormholeAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("LogMessagePublished", "Send")
    DEST_EVENTS = ("TransferRedeemed", "Receive")

    def bridge_id(self) -> str:
        return "wormhole"

    def bridge_mode(self) -> str:
        return "lock_mint"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_wormhole_params(
        self, event: DecodedEvent
    ) -> WormholeEventParams | None:
        params = event.params
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "value")
        topics = event.topics
        sender = self._get_topic_address(topics, 2) if len(topics) > 2 else ""
        recipient = self._get_topic_address(topics, 3) if len(topics) > 3 else ""
        if not sender:
            sender = self._get_str(params, "sender")
        if not recipient:
            recipient = self._get_str(params, "recipient")
        source_chain = self._get_int(params, "sourceChain")
        if source_chain == 0:
            source_chain = self._get_int(params, "emitterChainId")
        dest_chain = self._get_int(params, "destChain")
        if dest_chain == 0:
            dest_chain = self._get_int(params, "nonce")
        return WormholeEventParams(
            amount=amount,
            sender_address=sender,
            recipient_address=recipient,
            source_chain=source_chain,
            dest_chain=dest_chain,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_wormhole_params(event)
        if parsed is None:
            return None
        if not parsed.amount or not parsed.sender_address:
            return None
        source_node = self._node_id(event.chain, parsed.sender_address)
        target_node = self._node_id(event.chain, parsed.recipient_address)
        edge_id = f"wormhole_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_wormhole_params(event)
        if parsed is None:
            return None
        if not parsed.amount:
            return None
        recipient = parsed.recipient_address or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender_address)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"wormhole_mint_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.MINT,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_wormhole_params(source)
        dest_parsed = self._extract_wormhole_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.recipient_address.lower() != dest_parsed.sender_address.lower():
            return False
        return source_parsed.amount == dest_parsed.amount or (
            abs(source_parsed.amount - dest_parsed.amount) < source_parsed.amount * 0.01
        )


class LayerZeroAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("PacketSent",)
    DEST_EVENTS = ("PacketReceived",)

    def bridge_id(self) -> str:
        return "layerzero"

    def bridge_mode(self) -> str:
        return "lock_mint"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_layerzero_params(
        self, event: DecodedEvent
    ) -> LayerZeroEventParams | None:
        params = event.params
        destination_chain = self._get_int(params, "dstChainId")
        if destination_chain == 0:
            destination_chain = self._get_int(params, "destinationChainId")
        if destination_chain == 0:
            destination_chain = self._get_int(params, "chainId")
        destination_address = self._get_str(params, "to")
        if not destination_address:
            destination_address = self._get_str(params, "destinationAddress")
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "value")
        if amount == 0:
            amount = self._get_int(params, "qty")
        sender_address = self._get_str(params, "sender")
        if not sender_address:
            sender_address = self._get_topic_address(event.topics, 1)
        return LayerZeroEventParams(
            destination_chain=destination_chain,
            destination_address=destination_address,
            amount=amount,
            sender_address=sender_address,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_layerzero_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.sender_address)
        target_node = self._node_id(event.chain, parsed.destination_address)
        edge_id = f"layerzero_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_layerzero_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.destination_address or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender_address)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"layerzero_mint_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.MINT,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_layerzero_params(source)
        dest_parsed = self._extract_layerzero_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.destination_address.lower() != dest_parsed.sender_address.lower():
            return False
        if source_parsed.destination_chain != dest_parsed.destination_chain:
            return False
        return True


class MultichainAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("LogAnySwapIn", "SwapRemote")
    DEST_EVENTS = ("LogAnySwapOut", "SwapRemote")

    def bridge_id(self) -> str:
        return "multichain"

    def bridge_mode(self) -> str:
        return "lock_mint"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_multichain_params(
        self, event: DecodedEvent
    ) -> MultichainEventParams | None:
        params = event.params
        token = self._get_str(params, "token")
        if not token:
            token = self._get_str(params, "localToken")
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "tokenAmount")
        sender = self._get_str(params, "sender")
        if not sender:
            sender = self._get_str(params, "from")
        recipient = self._get_str(params, "recipient")
        if not recipient:
            recipient = self._get_str(params, "to")
        dest_chain = self._get_int(params, "destChainId")
        if dest_chain == 0:
            dest_chain = self._get_int(params, "chainId")
        return MultichainEventParams(
            token=token,
            amount=amount,
            sender=sender,
            recipient=recipient,
            dest_chain=dest_chain,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_multichain_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, parsed.recipient)
        edge_id = f"multichain_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_multichain_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.recipient or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"multichain_mint_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.MINT,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_multichain_params(source)
        dest_parsed = self._extract_multichain_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.recipient.lower() != dest_parsed.sender.lower():
            return False
        return True


class StargateAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("TokenMintAndSwap", "Send")
    DEST_EVENTS = ("TokenRedeem", "Receive")

    def bridge_id(self) -> str:
        return "stargate"

    def bridge_mode(self) -> str:
        return "pool"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_stargate_params(
        self, event: DecodedEvent
    ) -> StargateEventParams | None:
        params = event.params
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "amountSD")
        if amount == 0:
            amount = self._get_int(params, "mintAmount")
        src_chain = self._get_int(params, "srcChainId")
        if src_chain == 0:
            src_chain = self._get_int(params, "chainId")
        dst_chain = self._get_int(params, "dstChainId")
        if dst_chain == 0:
            dst_chain = self._get_int(params, "toChainId")
        sender = self._get_str(params, "sender")
        if not sender:
            sender = self._get_str(params, "from")
        to = self._get_str(params, "to")
        if not to:
            to = self._get_str(params, "recipient")
        return StargateEventParams(
            amount=amount,
            src_chain=src_chain,
            dst_chain=dst_chain,
            sender=sender,
            to=to,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_stargate_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, parsed.to)
        edge_id = f"stargate_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_release(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_stargate_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.to or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"stargate_release_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.RELEASE,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_stargate_params(source)
        dest_parsed = self._extract_stargate_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.to.lower() != dest_parsed.sender.lower():
            return False
        return True


class AcrossAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("FundsDeposited",)
    DEST_EVENTS = ("FilledRelay",)

    def bridge_id(self) -> str:
        return "across"

    def bridge_mode(self) -> str:
        return "intent"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_across_params(
        self, event: DecodedEvent
    ) -> AcrossEventParams | None:
        params = event.params
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "depositAmount")
        if amount == 0:
            amount = self._get_int(params, "inputAmount")
        origin_token = self._get_str(params, "originToken")
        if not origin_token:
            origin_token = self._get_str(params, "token")
        recipient = self._get_str(params, "recipient")
        if not recipient:
            recipient = self._get_str(params, "destinationRecipient")
        depositor = self._get_str(params, "depositor")
        if not depositor:
            depositor = self._get_str(params, "sender")
        destination_chain_id = self._get_int(params, "destinationChainId")
        if destination_chain_id == 0:
            destination_chain_id = self._get_int(params, "destinationChain")
        return AcrossEventParams(
            amount=amount,
            origin_token=origin_token,
            recipient=recipient,
            depositor=depositor,
            destination_chain_id=destination_chain_id,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_across_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.depositor)
        target_node = self._node_id(event.chain, parsed.recipient)
        edge_id = f"across_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_intent_fill(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_across_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.recipient or event.emitter_address
        source_node = self._node_id(event.chain, parsed.depositor)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"across_intent_fill_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.INTENT_FILL,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_across_params(source)
        dest_parsed = self._extract_across_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.recipient.lower() != dest_parsed.recipient.lower():
            return False
        return True


class HopAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("TransferSent",)
    DEST_EVENTS = ("TransferReceived",)

    def bridge_id(self) -> str:
        return "hop"

    def bridge_mode(self) -> str:
        return "pool"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_hop_params(self, event: DecodedEvent) -> HopEventParams | None:
        params = event.params
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "bonderFee")
        recipient = self._get_str(params, "recipient")
        if not recipient:
            recipient = self._get_str(params, "to")
        sender = self._get_str(params, "sender")
        if not sender:
            sender = self._get_str(params, "from")
        chain_id = self._get_int(params, "chainId")
        if chain_id == 0:
            chain_id = self._get_int(params, "destinationChainId")
        return HopEventParams(
            amount=amount,
            recipient=recipient,
            sender=sender,
            chain_id=chain_id,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_hop_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, parsed.recipient)
        edge_id = f"hop_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_release(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_hop_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.recipient or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"hop_release_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.RELEASE,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_hop_params(source)
        dest_parsed = self._extract_hop_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.recipient.lower() != dest_parsed.sender.lower():
            return False
        return True


class CBridgeAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("Send",)
    DEST_EVENTS = ("Receive",)

    def bridge_id(self) -> str:
        return "cbridge"

    def bridge_mode(self) -> str:
        return "pool"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_cbridge_params(
        self, event: DecodedEvent
    ) -> CBridgeEventParams | None:
        params = event.params
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "amt")
        sender = self._get_str(params, "sender")
        if not sender:
            sender = self._get_str(params, "from")
        receiver = self._get_str(params, "receiver")
        if not receiver:
            receiver = self._get_str(params, "to")
        destination_chain_id = self._get_int(params, "destination_chain_id")
        if destination_chain_id == 0:
            destination_chain_id = self._get_int(params, "destChainId")
        if destination_chain_id == 0:
            destination_chain_id = self._get_int(params, "chainId")
        destination_id = self._get_str(params, "destination_id")
        if not destination_id:
            destination_id = self._get_str(params, "destId")
        return CBridgeEventParams(
            amount=amount,
            sender=sender,
            receiver=receiver,
            destination_chain_id=destination_chain_id,
            destination_id=destination_id,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_cbridge_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, parsed.receiver)
        edge_id = f"cbridge_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_release(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_cbridge_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.receiver or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"cbridge_release_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.RELEASE,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_cbridge_params(source)
        dest_parsed = self._extract_cbridge_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.receiver.lower() != dest_parsed.receiver.lower():
            return False
        return True


class SynapseAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("TokenSend", "Deposit")
    DEST_EVENTS = ("TokenReceive", "Withdraw")

    def bridge_id(self) -> str:
        return "synapse"

    def bridge_mode(self) -> str:
        return "burn_mint"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_synapse_params(
        self, event: DecodedEvent
    ) -> SynapseEventParams | None:
        params = event.params
        token = self._get_str(params, "token")
        if not token:
            token = self._get_str(params, "erc20")
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "tokensAmount")
        to = self._get_str(params, "to")
        if not to:
            to = self._get_str(params, "recipient")
        from_ = self._get_str(params, "from")
        if not from_:
            from_ = self._get_str(params, "sender")
        chain_id = self._get_int(params, "chainId")
        if chain_id == 0:
            chain_id = self._get_int(params, "fromChainId")
        return SynapseEventParams(
            token=token,
            amount=amount,
            to=to,
            from_=from_,
            chain_id=chain_id,
        )

    def decode_burn(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_synapse_params(event)
        if parsed is None or not parsed.amount:
            return None
        source_node = self._node_id(event.chain, parsed.from_)
        target_node = self._node_id(event.chain, parsed.to)
        edge_id = f"synapse_burn_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.BURN,
            event=event,
            amount=parsed.amount,
        )

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_synapse_params(event)
        if parsed is None or not parsed.amount:
            return None
        recipient = parsed.to or event.emitter_address
        source_node = self._node_id(event.chain, parsed.from_)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"synapse_mint_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.MINT,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_synapse_params(source)
        dest_parsed = self._extract_synapse_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.to.lower() != dest_parsed.from_.lower():
            return False
        return True


class HyperlaneAdapter(BaseBridgeAdapter):
    SOURCE_EVENTS = ("Dispatch", "Process")
    DEST_EVENTS = ("Process", "Dispatch")

    def bridge_id(self) -> str:
        return "hyperlane"

    def bridge_mode(self) -> str:
        return "lock_mint"

    def source_event_names(self) -> tuple[str, ...]:
        return self.SOURCE_EVENTS

    def dest_event_names(self) -> tuple[str, ...]:
        return self.DEST_EVENTS

    def _extract_hyperlane_params(
        self, event: DecodedEvent
    ) -> HyperlaneEventParams | None:
        params = event.params
        sender = self._get_str(params, "sender")
        if not sender:
            sender = self._get_topic_address(event.topics, 1)
        recipient = self._get_str(params, "recipient")
        if not recipient:
            recipient = self._get_str(params, "destRecipient")
        amount = self._get_int(params, "amount")
        if amount == 0:
            amount = self._get_int(params, "value")
        dest_domain = self._get_int(params, "destination")
        if dest_domain == 0:
            dest_domain = self._get_int(params, "destDomain")
        if dest_domain == 0:
            dest_domain = self._get_int(params, "domainId")
        origin_domain = self._get_int(params, "origin")
        if origin_domain == 0:
            origin_domain = self._get_int(params, "originDomain")
        if origin_domain == 0:
            origin_domain = self._get_int(params, "senderChainId")
        return HyperlaneEventParams(
            sender=sender,
            recipient=recipient,
            amount=amount,
            dest_domain=dest_domain,
            origin_domain=origin_domain,
        )

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.SOURCE_EVENTS:
            return None
        parsed = self._extract_hyperlane_params(event)
        if parsed is None or not parsed.sender:
            return None
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, parsed.recipient)
        edge_id = f"hyperlane_lock_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.LOCK,
            event=event,
            amount=parsed.amount,
        )

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        if event.event_name not in self.DEST_EVENTS:
            return None
        parsed = self._extract_hyperlane_params(event)
        if parsed is None:
            return None
        recipient = parsed.recipient or event.emitter_address
        source_node = self._node_id(event.chain, parsed.sender)
        target_node = self._node_id(event.chain, recipient)
        edge_id = f"hyperlane_mint_{event.event_id}"
        return self._create_bridge_edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            operator=TaintOperator.MINT,
            event=event,
            amount=parsed.amount,
        )

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        if not super().is_pair(source, dest):
            return False
        source_parsed = self._extract_hyperlane_params(source)
        dest_parsed = self._extract_hyperlane_params(dest)
        if source_parsed is None or dest_parsed is None:
            return False
        if source_parsed.recipient.lower() != dest_parsed.sender.lower():
            return False
        if source_parsed.dest_domain != dest_parsed.origin_domain:
            return False
        return True


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BridgeAdapterProtocol] = {}
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        default_adapters: list[BaseBridgeAdapter] = [
            WormholeAdapter(),
            LayerZeroAdapter(),
            MultichainAdapter(),
            StargateAdapter(),
            AcrossAdapter(),
            HopAdapter(),
            CBridgeAdapter(),
            SynapseAdapter(),
            HyperlaneAdapter(),
        ]
        for adapter in default_adapters:
            self.register(adapter)

    def register(self, adapter: BridgeAdapterProtocol) -> None:
        bridge_id = adapter.bridge_id()
        self._adapters[bridge_id] = adapter

    def get(self, bridge_id: str) -> BridgeAdapterProtocol | None:
        return self._adapters.get(bridge_id)

    @property
    def supported_bridges(self) -> list[str]:
        return list(self._adapters.keys())

    def get_adapter_for_event(self, event: DecodedEvent) -> BridgeAdapterProtocol | None:
        for adapter in self._adapters.values():
            if adapter.bridge_id() == event.bridge:
                return adapter
            if event.event_name in adapter.source_event_names() or \
               event.event_name in adapter.dest_event_names():
                return adapter
        return None


ALL_BRIDGE_ADAPTERS: tuple[BaseBridgeAdapter, ...] = (
    WormholeAdapter(),
    LayerZeroAdapter(),
    MultichainAdapter(),
    StargateAdapter(),
    AcrossAdapter(),
    HopAdapter(),
    CBridgeAdapter(),
    SynapseAdapter(),
    HyperlaneAdapter(),
)


def all_bridge_adapters() -> tuple[BaseBridgeAdapter, ...]:
    """Return the default adapter tuple used by the IR graph builder."""
    return ALL_BRIDGE_ADAPTERS
