"""Module containing the Bitcoin transaction representation and utilities.

The following components are included:
    - VARINT_S (constant): smallest category a VarInt may fall.
    - VARINT_M (constant)
    - VARINT_L (constant)
    - VARINT_XL (constant): the largest category a VarInt may fall.
    - PRESERVED_EFFECTIVE_VALUE_KEY (constant): abbreviation string.
    - FINAL_FEE_RATE_KEY (constant): abbreviation string.
    - get_var_int_weight (function): function to get the weight taken by the
        varint size field.
    - TxDescriptor (class): the Bitcoin transaction representation used by the
        application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import structlog

from datatypes.fee_rate import FeeRate
from datatypes.utxo import UTxO
from utils.bitcoin import sat_vB_to_sat_kvB

LOGGER = structlog.stdlib.get_logger(__name__)

#: the different size categories a VarInt may fall in
VARINT_S: int = 0xFC
VARINT_M: int = 0xFFFF
VARINT_L: int = 0xFFFFFFFF
VARINT_XL: int = 0xFFFFFFFFFFFFFFFF

#: abbreviations used in transaction summary
PRESERVED_EFFECTIVE_VALUE_KEY: str = "PEV"
FINAL_FEE_RATE_KEY: str = "FFR"


def get_var_int_weight(length: int) -> int:
    """Get the weight of a transaction VarInt size field.

    Args:
        length: the length or count of items the VarInt will use.

    Returns:
        The weigth taken by the VarInt to express its size.
    """
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
    """Dataclass used to describe the changes produced by a transaction.

    Transaction are Bitcoin structures that allow the transference of value
    between parties. Per se, Bitcoin account model is different from the
    traditional system, so transactions also differ from what one may imagine
    when talking about transactions. Transactions can be thought as
    functions with inputs and ouputs, where the inputs are provided by the
    "paying" party and output by the "receiver" party. Transactions should
    include fields to allow the accountability of them by the rest of the
    protocol participating parties, and are also data structures that should be
    serialized and transmited over the network. All those things add complexity
    to its representation.
    Here we tried to be as close as possible to what a transaction really is in
    Bitcoin, but abstracting as much of the complex parts as possible.

    A TxDescriptor includes the four principal things to consider when making a
    transaction, and avoid the other complementary items transactions have.
    These minimal components are:
    - A collection of inputs: although inputs are references to UTxOs together
      with the witness of the ownership of those, we decided to avoid the
      signature and dereference complexity by considering inputs as the
      collection of UTxOs that will be spend in a transaction.

    - A collection of payment outputs: these are the UTxOs that will be created
      for the claimers of the payment obligations fulfilled by this transaction.

    - A collection of change outputs: for our own knowledge, currently all
      implementations of coin selection strategies just consider transactions
      with one or zero change outputs. Some strategies that haven't been
      developed yet may consider the creation of more change outputs based on
      other criteria, probably not fee reduction. We didn't want to cut this
      possibility so we consider the change outputs as a whole collection of
      UTxOs. We expect this possibility to be studied in the future by new coin
      selection strategies.

    - The amount of bitcoin excess produced: commonly, the amount provided by
      inputs don't match exactly the amount required by the transaction. The
      extra amount can be returned in change outputs, but sometimes this isn't
      economically viable. In those cases, the extra amount is donated as part
      of the fees for the miner.

    Other complexities associated with transactions arise of the different
    types of UTxOs they may reference or create. To avoid them we have only
    used P2WPKH UTxOs, but this can change in the future.

    Attributes:
        TX_VERSION_SIZE_VBYTES: the size in virtual bytes of the transaction
            version field.
        TX_MARKER_SIZE_VBYTES: the size in virtual bytes of the marker field.
            The size of this field has been set considering the use of only
            P2WPKH UTxOs.
        TX_FLAG_SIZE_VBYTES: the size in virtual bytes of the witness flag
            field, used to signal the inclusion of witness data as part of a
            transaction.
        TX_LOCKTIME_SIZE_VBYTES: the size in virtual bytes of the locktime
            field.

        inputs: the list of UTxOs providing the liquidity to the transaction.
        payments: the UTxOs that will be created by this transaction and which
            will receive part of the bitcoin units provided by the inputs.
        change: a list of UTxOs with the created change outputs for the
            transaction. It can be empty.
        excess: the amount of extra bitcoin units released to miners as part of
            the transaction fee. It should be zero or higher.
    """

    TX_VERSION_SIZE_VBYTES: ClassVar[int] = 4
    TX_MARKER_SIZE_VBYTES: ClassVar[int] = 1
    TX_FLAG_SIZE_VBYTES: ClassVar[int] = 4
    TX_LOCKTIME_SIZE_VBYTES: ClassVar[int] = 4

    inputs: list[UTxO]
    payments: list[UTxO]
    change: list[UTxO] = field(default_factory=list)
    excess: int = 0

    def fee(self, fee_rate: FeeRate) -> int:
        """Get the transaction fee derived from the passed fee rate.

        Args:
            fee_rate: the fee rate to compute the transaction fee.

        Returns:
            The transaction fee expressed in satoshis.
        """
        # https://btc.stackexchange.com/questions/92689/how-is-the-size-of-a-btc-transaction-calculated
        return fee_rate.fee(self.weight)

    def fix_rounding_errors(self, fee_rate: FeeRate) -> None:
        """Fix machine rounding errors.

        Floating point arithmetic, used to convert units from satoshis to
        bitcoin and vice versa, can produce, usually off-by-one, errors while
        rounding. As those units represent value, we must be sure there are not
        lost units in the coin selection process.
        Here we recover those units with change outputs or release them in
        transaction excess.

        Args:
            fee_rate: the fee rate to compute transaction fees, which
                are part of transaction costs.
        """
        extra_sats: int = (
            self.input_amount
            - self.output_amount
            - self.excess
            - self.fee(fee_rate)
        )

        LOGGER.debug(f"Tx rounding error: {extra_sats}.")
        if self.change:
            self.change[0].amount += extra_sats
        else:
            self.excess += extra_sats

    def valid(self, fee_rate: FeeRate) -> bool:
        """Check protocol transaction validity conditions.

        This method checks minimal conditions for the transaction to be relayed
        and eventually added to a block in the Bitcoin protocol. These minimal
        conditions are:
            - no negative change
            - no negative excess
            - no negative payments
            - not missing bitcoin units

        Args:
            fee_rate: the fee rate to compute the transaction fee.
        """
        total_outgoing: int = self.output_amount
        total_outgoing += self.fee(fee_rate) + self.excess
        no_negative_change: bool = all(
            utxo.amount >= 0 for utxo in self.change
        )
        no_negative_excess: bool = self.excess >= 0
        no_negative_payments: bool = all(
            utxo.amount >= 0 for utxo in self.payments
        )
        no_missing_quantities: bool = self.input_amount == total_outgoing
        return (
            no_negative_change
            and no_negative_excess
            and no_negative_payments
            and no_missing_quantities
        )

    @property
    def input_amount(self) -> int:
        """The sum of all the input amounts."""
        return sum(utxo.amount for utxo in self.inputs)

    @property
    def payment_amount(self) -> int:
        """The sum of all payment output amounts."""
        return sum(utxo.amount for utxo in self.payments)

    @property
    def change_amount(self) -> int:
        """The sum of all change output amounts."""
        return sum(utxo.amount for utxo in self.change)

    @property
    def output_amount(self) -> int:
        """The sum of all the payment and change outputs amounts combined."""
        return self.payment_amount + self.change_amount

    @property
    def weight(self) -> int:
        """The weight of the transaction expressed in virtual bytes."""
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
    def final_fee_rate(self) -> FeeRate:
        """The real fee rate of the transaction produced."""
        actual_fee = self.input_amount - self.output_amount
        if actual_fee < 0:
            return FeeRate(0)
        LOGGER.debug(f"Tx actual fee {actual_fee}.")
        sats_vB = actual_fee / self.weight
        sats_kvB: int = sat_vB_to_sat_kvB(sats_vB)
        return FeeRate(sats_kvB)

    @property
    def digest(self) -> dict:
        """The transaction summary."""
        return {
            "#inputs": len(self.inputs),
            "#payments": len(self.payments),
            "#change": len(self.change),
            "excess": self.excess,
            PRESERVED_EFFECTIVE_VALUE_KEY: self.change_amount,
            FINAL_FEE_RATE_KEY: str(self.final_fee_rate),
        }
