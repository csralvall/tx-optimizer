"""A module with the needed data to create a simplified UTxO model.

This module includes:
    - OutputType (class): the type of the script inside the UTxO.
    - TypeMetadata (class): metadata associated with the OutputType.
    - UTxO (class): a simplified Unspent Transaction Output model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import structlog

from datatypes.fee_rate import FeeRate

LOGGER = structlog.stdlib.get_logger(__name__)


class TypeMetadata(NamedTuple):
    """Weight data associated to each UTxO to compute its fees.

    Attributes:
        input_weight: the weight of the UTxO when included as a transaction
            input (mainly witness data).
        output_weight: the weight of the UTxO when produced by a transaction.
    """

    input_weight: int
    output_weight: int


class OutputType(Enum):
    """Representation of the types of ScriptPubKeys available."""

    #: Pay to Witness Public Key Hash
    P2WPKH = TypeMetadata(input_weight=68, output_weight=31)


@dataclass
class UTxO:
    """A simplification of an Unspent Transaction Output.

    Attributes:
        output_type: the type of the ScriptPubKey of the UTxO.
        amount: the amount of satoshis this UTxO has blocked in it.
        wallet_id: a unique field to identify the UTxO inside the simulation
            wallet.
    """

    output_type: OutputType
    amount: int
    wallet_id: int = field(init=False, default=-1)

    def input_fee(self, fee_rate: FeeRate) -> int:
        """The fee to spend the UTxO as input at the specified fee rate.

        Args:
            fee_rate: the fee rate at which the UTxO is going to be spend.
        """
        input_weight: int = self.output_type.value.input_weight
        return fee_rate.fee(input_weight)

    def output_fee(self, fee_rate: FeeRate) -> int:
        """The fee to spend the UTxO as output at the specified fee rate.

        Args:
            fee_rate: the fee rate at which the UTxO is going to be created.
        """
        output_weight: int = self.output_type.value.output_weight
        return fee_rate.fee(output_weight)
