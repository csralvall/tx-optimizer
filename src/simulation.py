from __future__ import annotations

import csv
import json
import time
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

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
    InternalSolverError,
    UTxOSelectionFailed,
    avoid_change,
    greatest_first,
    maximize_effective_value,
    minimize_waste,
    single_random_draw,
)
from utils.bitcoin import btc_to_sat
from utils.hardware import get_hardware_spec
from utils.time import human_readable_elapsed_time

LOGGER = structlog.stdlib.get_logger(__name__)


class Simulation:
    MODELS: ClassVar[dict[str, CoinSelectionAlgorithm]] = {
        "greatest-first": greatest_first,
        "single-random-draw": single_random_draw,
        "minimize-waste": minimize_waste,
        "maximize-effective_value": maximize_effective_value,
        "avoid-change": avoid_change,
    }
    UTXOS_ACTIVITY_CSV_HEADER: ClassVar[tuple[str, str, str, str]] = (
        "block_id",
        "wallet_id",
        "condition",
        "amount",
    )
    PAYMENT_REQUEST: ClassVar[str] = "PAYMENT_REQUEST"
    INCOME: ClassVar[str] = "INCOME"
    TX_PAYMENT: ClassVar[str] = "PAYMENT"
    TX_INPUT: ClassVar[str] = "INPUT"
    TX_CHANGE: ClassVar[str] = "CHANGE"

    def __init__(
        self,
        path: Path,
        scenario: str = "*",
        model: str = "",
        excluded: frozenset[tuple[str, str]] = frozenset(),
    ):
        self.path: Path = path
        self.scenario: str = scenario
        self.model: str = model
        self.excluded: frozenset[tuple[str, str]] = excluded

    @property
    def _selectors(
        self,
    ) -> Generator[tuple[str, CoinSelectionAlgorithm], None, None]:
        selectors: dict[str, CoinSelectionAlgorithm] = self.MODELS
        if self.model:
            selectors = {self.model: self.MODELS[self.model]}
        yield from selectors.items()

    @property
    def _scenarios(self) -> Generator[Path, None, None]:
        scenarios_dir: Path = self.path / "scenarios"
        yield from scenarios_dir.glob(self.scenario)

    def is_excluded(self, scenario_name: str, selector_name: str) -> bool:
        partially_excluded: bool = (
            scenario_name,
            selector_name,
        ) in self.excluded
        totally_excluded: bool = ("*", selector_name) in self.excluded
        return partially_excluded or totally_excluded

    def __enter__(self) -> Simulation:
        self.simulation_summary: dict = get_hardware_spec()
        self.simulation_dir: Path = (
            self.path / "simulations" / f"{time.strftime('%Y_%m_%d-%H_%M_%S')}"
        )
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_):
        self.simulation_end_time: float = time.time()
        elapsed_time: float = (
            self.simulation_end_time - self.simulation_start_time
        )
        self.simulation_summary[
            "simulation_elapsed_time"
        ] = human_readable_elapsed_time(elapsed_time)
        simulation_summary_path: Path = (
            self.simulation_dir / "simulation_summary.json"
        )
        with simulation_summary_path.open(mode="w") as simulation_summary_file:
            json.dump(self.simulation_summary, simulation_summary_file)

    def run(self) -> None:
        self.simulation_start_time: float = time.time()
        for scenario_path in self._scenarios:
            coin_selection_scenario: DataFrame = pandas.read_csv(
                scenario_path, names=["block_id", "amount", "fee_rate"]
            )
            for selector_name, selector in self._selectors:
                if self.is_excluded(scenario_path.stem, selector_name):
                    continue

                simulation_scenario: Path = (
                    self.simulation_dir / scenario_path.stem / selector_name
                )
                simulation_scenario.mkdir(parents=True, exist_ok=True)
                self._sim(
                    scenario=coin_selection_scenario,
                    main_algorithm=selector,
                    fallback_algorithm=single_random_draw,
                    output_dir=simulation_scenario,
                )

    def _sim(
        self,
        scenario: DataFrame,
        main_algorithm: CoinSelectionAlgorithm,
        fallback_algorithm: CoinSelectionAlgorithm,
        output_dir: Path,
    ) -> None:
        failed_txs_path: Path = output_dir / "failed_txs"
        failed_txs_path.mkdir(parents=True, exist_ok=True)
        with (
            (output_dir / "transactions_summary.csv").open(
                mode="w"
            ) as transactions_output,
            (output_dir / "utxo_activity.csv").open(mode="w") as utxos_output,
        ):
            summary_writer = csv.writer(transactions_output)
            # write csv header
            summary_writer.writerow(SelectionContext.CSV_DATA_HEADER)
            wallet = Wallet()
            utxos_writer = csv.writer(utxos_output)
            utxos_writer.writerow(self.UTXOS_ACTIVITY_CSV_HEADER)
            total_payments: int = (
                scenario[scenario["amount"] < 0]["block_id"].unique().shape[0]
            )
            processed_payments: int = 0
            for block_id, block in scenario.groupby("block_id"):
                pending_payments: list = []
                for _, tx in block.iterrows():
                    utxo = UTxO(
                        output_type=OutputType.P2WPKH,
                        amount=btc_to_sat(abs(tx.amount)),
                    )
                    if tx.amount > 0:
                        wallet.add(utxo)
                        utxos_writer.writerow(
                            (
                                block_id,
                                utxo.wallet_id,
                                self.INCOME,
                                utxo.amount,
                            )
                        )
                    else:
                        pending_payments.append(utxo)
                        utxos_writer.writerow(
                            (block_id, -1, self.PAYMENT_REQUEST, utxo.amount)
                        )

                if not pending_payments:
                    continue

                new_tx: TxDescriptor
                selector: str
                selection_context = SelectionContext(
                    wallet=wallet,
                    payments=pending_payments,
                    fee_rate=FeeRate(btc_to_sat(block.fee_rate.values[0])),
                )
                try:
                    selection_context.funds_are_enough()
                except NotEnoughFunds as e:
                    LOGGER.warn(str(e), **selection_context.digest)
                    processed_payments += 1
                    continue

                selection_start_time: float = time.time()
                try:
                    new_tx = main_algorithm(
                        selection_context=selection_context
                    )
                    selector = f"{main_algorithm.__name__}"
                except (UTxOSelectionFailed, InternalSolverError) as e:
                    if isinstance(e, InternalSolverError):
                        e.model.to_json(
                            failed_txs_path / f"txs_{block_id}.json"
                        )
                    new_tx = fallback_algorithm(
                        selection_context=selection_context
                    )
                    selector = f"{fallback_algorithm.__name__}"
                finally:
                    selection_end_time: float = time.time()

                selection_context.settle_tx(selector, new_tx)

                summary_writer.writerow(selection_context.to_csv())

                pending_payments = [
                    payment
                    for payment in pending_payments
                    if payment not in new_tx.payments
                ]

                for utxo in new_tx.payments:
                    utxos_writer.writerow(
                        (block_id, -1, self.TX_PAYMENT, utxo.amount)
                    )

                for utxo in new_tx.inputs:
                    wallet.pop(utxo)
                    utxos_writer.writerow(
                        (block_id, utxo.wallet_id, self.TX_INPUT, utxo.amount)
                    )

                for utxo in new_tx.change:
                    wallet.add(utxo)
                    utxos_writer.writerow(
                        (block_id, utxo.wallet_id, self.TX_CHANGE, utxo.amount)
                    )

                processed_payments += 1
                LOGGER.info(
                    f"{selector} - {processed_payments}/{total_payments}",
                    processing_time=f"{selection_end_time - selection_start_time:.4f}",
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
@click.option(
    "--exclude",
    type=str,
    default=[""],
    multiple=True,
    help="""
    Specify combinations of scenarios and models to avoid.
    Use * in scenarios to disable the model completely.
    Format: <scenario>,<model>.
    """,
)
@click.pass_obj
def simulate(ctx, scenario: str, model: str, exclude: list[str]) -> None:
    excluded: set = set()
    for excluded_combination in exclude:
        excluded.add(tuple(excluded_combination.split(",")))

    data_root: Path = ctx.get("data_path")
    with Simulation(
        path=data_root,
        scenario=scenario,
        model=model,
        excluded=frozenset(excluded),
    ) as simulation:
        simulation.run()
