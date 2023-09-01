from __future__ import annotations

import csv
import json
import time
from collections.abc import Generator, Hashable
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


class SimulationBench:
    MODELS: ClassVar[dict[str, CoinSelectionAlgorithm]] = {
        "greatest-first": greatest_first,
        "single-random-draw": single_random_draw,
        "minimize-waste": minimize_waste,
        "maximize-effective-value": maximize_effective_value,
        "avoid-change": avoid_change,
    }

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

    def __enter__(self) -> SimulationBench:
        self.simulation_summary: dict = get_hardware_spec()
        self.simulation_dir: Path = (
            self.path
            / "simulations"
            / f"{time.strftime('%d_%m_%Y__%H_%M_%S')}"
        )
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_):
        self.simulation_end_time: float = time.process_time()
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

    def __iter__(self) -> Generator[Simulation, None, None]:
        self.simulation_start_time: float = time.process_time()
        for scenario_path in self._scenarios:
            coin_selection_scenario: DataFrame = pandas.read_csv(
                scenario_path, names=["block_id", "amount", "fee_rate"]
            )
            coin_selection_scenario.name = scenario_path.stem
            for selector_name, selector in self._selectors:
                if self.is_excluded(scenario_path.stem, selector_name):
                    continue

                simulation_path: Path = (
                    self.simulation_dir / scenario_path.stem / selector_name
                )
                simulation_path.mkdir(parents=True, exist_ok=True)
                yield Simulation(
                    output_dir=simulation_path,
                    scenario=coin_selection_scenario,
                    main_algorithm=selector,
                    fallback_algorithm=single_random_draw,
                )


class Simulation:
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
        output_dir: Path,
        scenario: DataFrame,
        main_algorithm: CoinSelectionAlgorithm,
        fallback_algorithm: CoinSelectionAlgorithm,
    ) -> None:
        self.scenario: DataFrame = scenario
        self.scenario_name = getattr(self.scenario, "name", "")
        self.main_algorithm: CoinSelectionAlgorithm = main_algorithm
        self.fallback_algorithm: CoinSelectionAlgorithm = fallback_algorithm
        self.processed_payments: int = 0
        self.total_payments: int = (
            scenario[scenario["amount"] < 0]["block_id"].unique().shape[0]
        )
        self.wallet = Wallet()
        self.pending_payments = []
        self.failed_txs_path: Path = output_dir / "failed_txs"
        self.txs_path: Path = output_dir / "transactions_summary.csv"
        self.utxos_path: Path = output_dir / "utxo_activity.csv"

    def process_block(self, block_id: Hashable, block: DataFrame) -> None:
        for _, tx in block.iterrows():
            utxo = UTxO(
                output_type=OutputType.P2WPKH,
                amount=btc_to_sat(abs(tx.amount)),
            )
            if tx.amount > 0:
                self.wallet.add(utxo)
                self.utxos_writer.writerow(
                    (
                        block_id,
                        utxo.wallet_id,
                        self.INCOME,
                        utxo.amount,
                    )
                )
            else:
                self.pending_payments.append(utxo)
                self.utxos_writer.writerow(
                    (block_id, -1, self.PAYMENT_REQUEST, utxo.amount)
                )

    def select(
        self, block_id: Hashable, context: SelectionContext
    ) -> TxDescriptor:
        selected_tx: TxDescriptor
        start_time: float = time.process_time()
        try:
            selected_tx = self.main_algorithm(selection_context=context)
            if context.status != "success":
                context.settle_tx(self.main_algorithm.__name__, selected_tx)
        except (UTxOSelectionFailed, InternalSolverError) as e:
            if isinstance(e, InternalSolverError):
                e.model.to_json(self.failed_txs_path / f"txs_{block_id}.json")
            selected_tx = self.fallback_algorithm(selection_context=context)
            context.settle_tx(self.fallback_algorithm.__name__, selected_tx)
        finally:
            end_time: float = time.process_time()

        context.cpu_time = f"{end_time - start_time:.4f}"
        return selected_tx

    def update(self, block_id: Hashable, tx: TxDescriptor) -> None:
        self.processed_payments += 1
        remaining_payments: list[UTxO] = []
        for utxo in self.pending_payments:
            if utxo not in tx.payments:
                remaining_payments.append(utxo)
                continue
            self.utxos_writer.writerow(
                (block_id, -1, self.TX_PAYMENT, utxo.amount)
            )
        self.pending_payments = remaining_payments

        for utxo in tx.inputs:
            self.wallet.pop(utxo)
            self.utxos_writer.writerow(
                (block_id, utxo.wallet_id, self.TX_INPUT, utxo.amount)
            )

        for utxo in tx.change:
            self.wallet.add(utxo)
            self.utxos_writer.writerow(
                (block_id, utxo.wallet_id, self.TX_CHANGE, utxo.amount)
            )

    def _sim(self) -> None:
        # write csv header
        self.txs_writer.writerow(SelectionContext.CSV_DATA_HEADER)
        self.utxos_writer.writerow(self.UTXOS_ACTIVITY_CSV_HEADER)
        for block_id, block in self.scenario.groupby("block_id"):
            self.process_block(block_id, block)

            if not self.pending_payments:
                continue

            selection_context = SelectionContext(
                id=block_id,
                wallet=self.wallet,
                payments=self.pending_payments,
                fee_rate=FeeRate(btc_to_sat(block.fee_rate.values[0])),
            )
            try:
                selection_context.funds_are_enough()
            except NotEnoughFunds as e:
                LOGGER.warn(
                    str(e),
                    # Drop key-value pairs where value is zero
                    **{k: v for k, v in selection_context.digest.items() if v},
                )
                self.txs_writer.writerow(selection_context.to_csv())
                self.processed_payments += 1
                continue

            new_tx = self.select(block_id, selection_context)

            self.txs_writer.writerow(selection_context.to_csv())

            self.update(block_id, new_tx)

            LOGGER.info(
                f"{selection_context.policy} - {self.scenario_name} - {self.processed_payments}/{self.total_payments}",
                # Drop key-value pairs where value is zero
                **{k: v for k, v in selection_context.digest.items() if v},
            )

    def run(self) -> None:
        self.failed_txs_path.mkdir(parents=True, exist_ok=True)
        with (
            self.txs_path.open(mode="w") as txs_output,
            self.utxos_path.open(mode="w") as utxos_output,
        ):
            self.txs_writer = csv.writer(txs_output)
            self.utxos_writer = csv.writer(utxos_output)
            self._sim()


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
    with SimulationBench(
        path=data_root,
        scenario=scenario,
        model=model,
        excluded=frozenset(excluded),
    ) as simulation_bench:
        for simulation in simulation_bench:
            simulation.run()
