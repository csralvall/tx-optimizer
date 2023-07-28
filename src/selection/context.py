from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from itertools import accumulate
from statistics import fmean, pstdev
from typing import ClassVar, Literal, NamedTuple

import structlog

from datatypes.fee_rate import CONSOLIDATION_FEE_RATE, FeeRate
from datatypes.transaction import TxDescriptor
from datatypes.utxo import OutputType, UTxO
from datatypes.wallet import Wallet
from selection.metrics import waste

LOGGER = structlog.stdlib.get_logger(__name__)

TARGET_FEE_RATE_KEY: str = "TFR"


class UTxOKey(NamedTuple):
    amount: float
    id: int


@dataclass
class FeeRatedUTxO:
    utxo_key: UTxOKey
    effective_amount: int

    @classmethod
    def new(cls, utxo: UTxO, fee_rate: FeeRate) -> FeeRatedUTxO:
        return FeeRatedUTxO(
            utxo_key=UTxOKey(amount=utxo.amount, id=utxo.wallet_id),
            effective_amount=utxo.amount - utxo.input_fee(fee_rate),
        )


class InvalidTransaction(Exception):
    """Raise when the input minus output minus fees doesn't equal zero."""


class DustUTxO(Exception):
    """Raise when the UTxO amount cannot cover fees at consolidation fee rate."""


class NotEnoughFunds(Exception):
    """Raise when payments at current fee rate can't be covered by wallet."""

    def __str__(self) -> str:
        return "Can't cover payments at current fee rate."


@dataclass
class SelectionContext:
    CSV_DATA_HEADER: ClassVar[tuple] = (
        "algorithm",
        "balance",
        "#wallet",
        "#inputs",
        "#payments",
        "#change",
        "excess",
        "preserved_effective_value",
        "waste",
        "fee",
        "final_fee_rate",
        "target_fee_rate",
    )

    wallet: Wallet
    payments: list[UTxO]
    fee_rate: FeeRate
    tx: TxDescriptor = field(init=False)
    change_type: OutputType = OutputType.P2WPKH
    status: Literal["ongoing", "success", "failed"] = "ongoing"

    def __post_init__(self) -> None:
        min_tx = TxDescriptor(
            inputs=[], payments=self.payments, change=[], excess=0
        )
        self._minimal_tx_fees: int = min_tx.fee(self.fee_rate)
        self.tx = min_tx
        self._target = self.payment_amount + self._minimal_tx_fees
        self._change_template = UTxO(output_type=self.change_type, amount=0)

    @property
    def fee_rate_delta(self) -> FeeRate:
        return self.fee_rate - CONSOLIDATION_FEE_RATE

    @property
    def fee_rated_utxos(self) -> list[FeeRatedUTxO]:
        return [FeeRatedUTxO.new(utxo, self.fee_rate) for utxo in self.wallet]

    @property
    def payment_amount(self) -> int:
        return sum(utxo.amount for utxo in self.payments)

    @property
    def payments_mean(self) -> float:
        return fmean(utxo.amount for utxo in self.payments)

    @property
    def payments_stdev(self) -> float:
        return pstdev(utxo.amount for utxo in self.payments)

    @property
    def base_fee(self) -> int:
        return self._minimal_tx_fees

    @property
    def target(self) -> int:
        return self._target

    @property
    def minimal_number_of_inputs(self) -> int:
        utxos_subsums: accumulate[int] = accumulate(
            fr_utxo.effective_amount for fr_utxo in self.fee_rated_utxos
        )

        minimal_number_of_inputs: int = 0
        for index, utxo_sum_amount in enumerate(utxos_subsums, 1):
            if (utxo_sum_amount - self.target) > 0:
                minimal_number_of_inputs = index
                break

        return minimal_number_of_inputs

    @property
    def change_cost(self) -> int:
        change_cost = self._change_template.output_fee(
            self.fee_rate
        ) + self._change_template.input_fee(self.fee_rate)
        return change_cost

    @property
    def digest(self) -> dict:
        base_data: dict = {
            "status": self.status,
            "balance": self.wallet.balance,
            "#wallet": len(self.wallet),
        }
        if self.status == "success" and self.tx:
            base_data.pop("status")
            base_data = {
                **base_data,
                "waste": waste(self.tx, self.fee_rate),
                "fee": self.tx.fee(self.fee_rate),
                **self.tx.digest,
            }
        base_data[TARGET_FEE_RATE_KEY] = self.fee_rate
        return base_data

    def to_csv(self) -> tuple:
        tx_data: tuple = (0, 0, len(self.payments), 0, 0, 0, 0)
        outcome: str = self.status
        if self.status == "success" and self.tx:
            outcome = self.algorithm
            tx_data = (
                len(self.tx.inputs),
                len(self.tx.payments),
                len(self.tx.change),
                self.tx.excess,
                self.tx.change_amount,
                waste(self.tx, self.fee_rate),
                self.tx.fee(self.fee_rate),
                self.tx.final_fee_rate,
            )
        return (
            outcome,
            self.wallet.balance,
            len(self.wallet),
            *tx_data,
            str(self.fee_rate),
        )

    def funds_are_enough(self) -> None:
        if not self.minimal_number_of_inputs:
            self.status = "failed"
            raise NotEnoughFunds

    def get_change_utxo(self, overpayment: int) -> UTxO:
        change_utxo = copy(self._change_template)
        amount_with_fee_discount: int = overpayment - change_utxo.output_fee(
            self.fee_rate
        )
        if amount_with_fee_discount < change_utxo.input_fee(self.fee_rate):
            raise DustUTxO

        change_utxo.amount = amount_with_fee_discount
        return change_utxo

    def get_tx(self, utxo_ids: list[int]) -> TxDescriptor:
        tx = deepcopy(self.tx)
        tx.inputs = [self.wallet.get(id) for id in utxo_ids]
        return tx

    def settle_tx(self, selector: str, tx: TxDescriptor) -> None:
        if not tx.valid(self.fee_rate):
            raise InvalidTransaction
        self.status = "success"
        self.algorithm = selector
        self.tx = tx
