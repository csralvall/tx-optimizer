import csv
import time
from pathlib import Path

import click
import pandas
import structlog
from pandas import DataFrame

from datatypes.fee_rate import FeeRate
from datatypes.transaction import TxDescriptor
from datatypes.utxo import OutputType, UTxO
from datatypes.wallet import Wallet
from log_config import configure_loggers
from selection.context import NotEnoughFunds, SelectionContext
from selection.models import (
    CoinSelectionAlgorithm,
    UTxOSelectionFailed,
    greatest_first,
    minimize_inputs_without_change,
)
from utils.bitcoin import btc_to_sat

LOGGER = structlog.stdlib.get_logger("simulation")


def simulate(
    scenario: DataFrame,
    main_algorithm: CoinSelectionAlgorithm,
    fallback_algorithm: CoinSelectionAlgorithm,
    csv_writer,
) -> None:
    wallet = Wallet()
    total_payments: int = (
        scenario[scenario["amount"] < 0]["block_id"].unique().shape[0]
    )
    realized_payments: int = 0
    blocks = scenario.groupby("block_id")
    for _, block in blocks:
        pending_payments: list = []
        fee_rate = FeeRate(btc_to_sat(block.fee_rate.values[0]))
        for _, tx in block.iterrows():
            if tx.amount > 0:
                utxo: UTxO = UTxO(
                    output_type=OutputType.P2WPKH, amount=btc_to_sat(tx.amount)
                )
                wallet.add(utxo)
            else:
                pending_payments.append(
                    UTxO(
                        output_type=OutputType.P2WPKH,
                        amount=btc_to_sat(abs(tx.amount)),
                    )
                )

        if not pending_payments:
            continue

        new_tx: TxDescriptor
        selector: str
        selection_context = SelectionContext(
            wallet=wallet, payments=pending_payments, fee_rate=fee_rate
        )
        try:
            selection_context.funds_are_enough()
        except NotEnoughFunds as e:
            LOGGER.warn(str(e), **selection_context.digest)
            continue

        selection_start_time: float = time.time()
        try:
            new_tx = main_algorithm(selection_context=selection_context)
            selector = f"{main_algorithm.__name__}"
        except UTxOSelectionFailed:
            new_tx = fallback_algorithm(selection_context=selection_context)
            selector = f"{fallback_algorithm.__name__}"
        finally:
            selection_end_time: float = time.time()

        selection_context.settle_tx(new_tx)

        pending_payments = [
            payment
            for payment in pending_payments
            if payment not in new_tx.payments
        ]
        csv_writer.writerow(selection_context.to_csv())

        for utxo in new_tx.inputs:
            wallet.pop(utxo)

        for utxo in new_tx.change:
            wallet.add(utxo)

        formatted_processing_time: str = (
            f"{selection_end_time - selection_start_time:.4f}"
        )
        realized_payments += 1
        info_str: str = f"{selector} - {realized_payments}/{total_payments}"
        LOGGER.info(
            info_str,
            processing_time=formatted_processing_time,
            **selection_context.digest,
        )


@click.command()
@click.option(
    "--all", is_flag=True, help="Run all stored simulation scenarios."
)
def main(all: bool = False):
    configure_loggers(log_level="INFO")

    results_dir = Path.cwd() / "../results"

    if all:
        data_dir = Path.cwd() / "../data"

        for path in data_dir.glob("bustabit-2019-2020-tiny.csv"):
            results_file = results_dir / path.name
            with path.open(mode="r") as csv_input, path, results_file.open(
                mode="w"
            ) as csv_output:
                coin_selection_scenario: DataFrame = pandas.read_csv(
                    csv_input, names=["block_id", "amount", "fee"]
                )
                writer = csv.writer(csv_output)
                simulate(
                    scenario=coin_selection_scenario,
                    main_algorithm=minimize_inputs_without_change,
                    fallback_algorithm=greatest_first,
                    csv_writer=writer,
                )


if __name__ == "__main__":
    main()
