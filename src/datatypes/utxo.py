from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import structlog

from datatypes.fee_rate import FeeRate

LOGGER = structlog.stdlib.get_logger(__name__)


class TypeMetadata(NamedTuple):
    input_weight: int
    output_weight: int


class OutputType(Enum):
    P2WPKH = TypeMetadata(input_weight=68, output_weight=31)


@dataclass
class UTxO:
    output_type: OutputType
    amount: int
    wallet_id: int = field(init=False, default=-1)

    def input_fee(self, fee_rate: FeeRate) -> int:
        "Returns fee associated to the expenditure of this utxo"
        input_weight: int = self.output_type.value.input_weight
        return fee_rate.fee(input_weight)

    def output_fee(self, fee_rate: FeeRate) -> int:
        "Returns fee associated to the creation of this utxo"
        output_weight: int = self.output_type.value.output_weight
        return fee_rate.fee(output_weight)
