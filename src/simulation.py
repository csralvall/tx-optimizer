import csv
import time
from pathlib import Path
from typing import TextIO

import click
import pandas
import structlog
from pandas import DataFrame

from datatypes.fee_rate import FeeRate
from datatypes.transaction import TxDescriptor
from datatypes.utxo import OutputType, UTxO
from datatypes.wallet import Wallet
from selection.context import NotEnoughFunds, SelectionContext
from selection.models import (
    CoinSelectionAlgorithm,
    UTxOSelectionFailed,
    avoid_change,
    greatest_first,
    maximize_effective_value,
    minimize_waste,
    single_random_draw,
)
from utils.bitcoin import btc_to_sat

LOGGER = structlog.stdlib.get_logger("simulation")

MODELS: dict[str, CoinSelectionAlgorithm] = {
    "greatest_first": greatest_first,
    "single_random_draw": single_random_draw,
    "minimize_waste": minimize_waste,
    "maximize_effective_value": maximize_effective_value,
    "avoid_change": avoid_change,
}


def run_simulation(
    scenario: DataFrame,
    main_algorithm: CoinSelectionAlgorithm,
    fallback_algorithm: CoinSelectionAlgorithm,
    output_file: TextIO,
) -> None:
    txs_writer = csv.writer(output_file)
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
        txs_writer.writerow(selection_context.to_csv())

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


@click.command(help="Run bitcoin coin selection simulations.")
@click.option(
    "--scenario",
    default="*",
    help="Provide a scenario name to run from scenarios directory.",
)
@click.option(
    "--model", default="", help="Select a model to run the scenarios."
)
@click.pass_obj
def simulate(ctx, scenario: str, model: str) -> None:
    data_root: Path = ctx.get("data_path")
    simulation_dir: Path = data_root / "simulations"
    scenarios_dir: Path = data_root / "scenarios"
    for scenario_path in scenarios_dir.glob(scenario):
        simulation_scenario: Path = simulation_dir / scenario_path.stem
        simulation_scenario.mkdir(parents=True, exist_ok=True)
        transactions_log_path: Path = simulation_scenario / "transactions.csv"
        with (
            scenario_path.open(mode="r") as csv_input,
            transactions_log_path.open(mode="w") as transactions_log_output,
        ):
            coin_selection_scenario: DataFrame = pandas.read_csv(
                csv_input, names=["block_id", "amount", "fee_rate"]
            )
            if model:
                return run_simulation(
                    scenario=coin_selection_scenario,
                    main_algorithm=MODELS[model],
                    fallback_algorithm=greatest_first,
                    output_file=transactions_log_output,
                )

            for algorithm in MODELS.values():
                run_simulation(
                    scenario=coin_selection_scenario,
                    main_algorithm=algorithm,
                    fallback_algorithm=greatest_first,
                    output_file=transactions_log_output,
                )
