from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import structlog

from datatypes.fee_rate import FeeRate
from datatypes.utxo import UTxO
from utils import sat_vB_to_sat_kvB

LOGGER = structlog.stdlib.get_logger(__name__)

VARINT_S: int = 0xFC
VARINT_M: int = 0xFFFF
VARINT_L: int = 0xFFFFFFFF
VARINT_XL: int = 0xFFFFFFFFFFFFFFFF


def get_var_int_weight(length: int) -> int:
    weight: int = 0

    if length <= VARINT_S:
        weight = 1
    elif length <= VARINT_M:
        weight = 3
    elif length <= VARINT_L:
        weight = 5
    elif length <= VARINT_XL:
        weight = 9

    return weight


@dataclass
class TxDescriptor:
    TX_VERSION_SIZE_VBYTES: ClassVar[int] = 4
    TX_MARKER_SIZE_VBYTES: ClassVar[int] = 1
    TX_FLAG_SIZE_VBYTES: ClassVar[int] = 4
    TX_LOCKTIME_SIZE_VBYTES: ClassVar[int] = 4

    inputs: list[UTxO]
    payments: list[UTxO]
    change: list[UTxO] = field(default_factory=list)
    excess: int = 0

    def fee(self, fee_rate: FeeRate) -> int:
        # https://btc.stackexchange.com/questions/92689/how-is-the-size-of-a-btc-transaction-calculated
        return fee_rate.fee(self.weight)

    def fix_rounding_errors(self, fee_rate: FeeRate) -> None:
        input_amount = sum(utxo.amount for utxo in self.inputs)
        payment_amount = sum(utxo.amount for utxo in self.payments)
        change_amount = sum(utxo.amount for utxo in self.change)

        extra_sats: int = (
            input_amount
            - payment_amount
            - change_amount
            - self.excess
            - self.fee(fee_rate)
        )

        LOGGER.debug(f"Tx rounding error: {extra_sats}.")
        if self.change:
            self.change[0].amount += extra_sats
        else:
            self.excess += extra_sats

    def valid(self, fee_rate: FeeRate) -> bool:
        total_input: int = sum(utxo.amount for utxo in self.inputs)
        total_output: int = sum(utxo.amount for utxo in self.payments)
        total_output += sum(utxo.amount for utxo in self.change)
        total_output += self.excess
        total_output += self.fee(fee_rate)
        return total_input == total_output

    @property
    def weight(self) -> int:
        legacy_header_weight: int = (
            self.TX_VERSION_SIZE_VBYTES + self.TX_LOCKTIME_SIZE_VBYTES
        )
        segwit_header_weight: int = (
            self.TX_MARKER_SIZE_VBYTES + self.TX_FLAG_SIZE_VBYTES
        )
        outputs: list[UTxO] = self.payments + self.change
        utxo_len_weight: int = get_var_int_weight(
            len(self.inputs)
        ) + get_var_int_weight(len(outputs))
        header_weight: int = (
            legacy_header_weight + segwit_header_weight + utxo_len_weight
        )
        output_vector_weight: int = sum(
            utxo.output_type.value.output_weight for utxo in outputs
        )
        input_vector_weight: int = sum(
            utxo.output_type.value.input_weight for utxo in self.inputs
        )
        return header_weight + output_vector_weight + input_vector_weight

    @property
    def real_fee_rate(self) -> FeeRate:
        total_input: int = sum(utxo.amount for utxo in self.inputs)
        payload: int = sum(utxo.amount for utxo in self.payments)
        payload += sum(utxo.amount for utxo in self.change)
        actual_fee = total_input - payload + self.excess
        LOGGER.debug(f"Tx actual fee {actual_fee}.")
        sats_vB = actual_fee / self.weight
        sats_kvB: int = sat_vB_to_sat_kvB(sats_vB)
        return FeeRate(sats_kvB)

    @property
    def digest(self) -> dict:
        return {
            "#inputs": len(self.inputs),
            "#payments": len(self.payments),
            "#change": len(self.change),
            "excess": self.excess,
        }
