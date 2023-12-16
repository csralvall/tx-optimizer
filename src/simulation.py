"""Module wrapping up the model and algorithms with the scenarios to simulate.

Includes the following:
    - SimulationBench (class): the orchestrator of a whole simulation case, with
          multiple models and scenarios.
    - Simulation (class): a single coin selection simulation case combining a
        single model and a single scenario.
    - simulate (function): a function providing the interface to the cli utility
        to instantiate a SimulationBench to run simulations.
"""
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
    maximize_effective_value_with_fallback,
    minimize_waste,
    single_random_draw,
)
from utils.bitcoin import btc_to_sat
from utils.hardware import get_hardware_spec
from utils.time import human_readable_elapsed_time

LOGGER = structlog.stdlib.get_logger(__name__)


class SimulationBench:
    """Context manager to orchestrate simulations.

    This class consolidates the parametrization of the simulation. It manages
    the combinations of model-scenario to run, creates the simulation
    directories, and produces metadata about the hardware where the simulation
    was run and the whole cpu time it took.

    Args:
        path: the path of a directory containing the scenarios to simulate, and
            where the simulation data produced is going to be saved.
        scenario: the name of the particular scenario to simulate. If the
            parameter is not set and the default '*' is left as it is, all
            scenarios will be simulated.
        model: if set and it is a valid model name, run only this model in all
            the specified scenarios.
        excluded: a set of tuples specifying combinations of scenario-model to
            not simulate. If the scenario is '*', avoid all scenarios combined
            with the model. If the model is '*', the other way around. If the
            pair is ('*', '*'), no simulation is run.

    Attributes:
        MODELS: a mapping between the exposed model names and the functions
            that execute each one of them.
    """

    MODELS: ClassVar[dict[str, CoinSelectionAlgorithm]] = {
        "greatest-first": greatest_first,
        "single-random-draw": single_random_draw,
        "minimize-waste": minimize_waste,
        "maximize-effective-value": maximize_effective_value,
        "maximize-effective-value-with-fallback": maximize_effective_value_with_fallback,
        "avoid-change": avoid_change,
    }

    def __init__(
        self,
        path: Path,
        scenario: str = "*",
        model: str = "",
        excluded: frozenset[tuple[str, str]] = frozenset(),
    ):
        #: the root path of the directory to read and write simulation data
        self.path: Path = path

        #: the pattern matching the scenarios to run
        self.scenario: str = scenario

        #: if set, the only model to simulate the scenarios with
        self.model: str = model

        #: a set of scenario and model combinations to not simulate
        self.excluded: frozenset[tuple[str, str]] = excluded

    @property
    def _selectors(
        self,
    ) -> Generator[tuple[str, CoinSelectionAlgorithm], None, None]:
        """Filter selection models to simulate.

        Yields:
           An iterator of tuples where the first element is the name of the
           model to run and the second one is the functions that executes the
           model. If a model was specified in the intialization, just return
           the tuple belonging to that model.
        """
        selectors: dict[str, CoinSelectionAlgorithm] = self.MODELS
        if self.model:
            selectors = {self.model: self.MODELS[self.model]}
        yield from selectors.items()

    @property
    def _scenarios(self) -> Generator[Path, None, None]:
        """Filter scenarios to simulate.

        Yields:
            An iterator over all files in the 'scenarios' subdirectory matching
            the pattern specified as "scenario" during the initialization. If
            the pattern is '*', all files in the subdirectory match.
        """
        scenarios_dir: Path = self.path / "scenarios"
        yield from scenarios_dir.glob(self.scenario)

    def is_excluded(self, scenario_name: str, selector_name: str) -> bool:
        """Check if the scenario and selector are excluded from simulation.

        Args:
            scenario_name: the name of the scenario to simulate.
            selector_name: the name of the model to simulate.

        Returns:
            True if the combination of scenario and model has been excluded or
            False if not.
        """
        partially_excluded: bool = (
            scenario_name,
            selector_name,
        ) in self.excluded
        totally_excluded: bool = ("*", selector_name) in self.excluded
        return partially_excluded or totally_excluded

    def __enter__(self) -> SimulationBench:
        """Enter and create the simulation context.

        This method belongs to the `contextmanager` interface. Here we get the
        data about the hardware where the simulations is going to run and setup
        the directory to store the simulations under the path set at
        initialization, with the following format:
        <day>_<month>_<year>__<hour>_<minute>_<second>

        Returns:
            The current SimulationBench context instance.
        """
        self.simulation_summary: dict = get_hardware_spec()
        self.simulation_dir: Path = (
            self.path
            / "simulations"
            / f"{time.strftime('%d_%m_%Y__%H_%M_%S')}"
        )
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_):
        """Clean up the context and finish simulation.

        This method belongs to the `contextmanager` interface. Here we compute
        the total cpu elapsed time the whole simulation took and add it to the
        hardware specification obtained in the `__enter__` method. We attach
        this data to the simulation directory and get out of the context.
        """
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
        """Instanciate and yield scenario-model pairs ready to simulate.

        This method initiates the simulation timer, gets the scenario-mode pairs
        to run, provides them with their own subdirectory under the simulation
        root path and yields the Simulation instance which will run the coin
        selection.

        Yields:
            An instance of Simulation ready to run.
        """
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
    """The model combined with a scenario ready to start the coin selection.

    Args:
        output_dir: the path of the directory where the simulation data should
            be stored.
        scenario: the name of the scenario where the coin selection is going to
            be performed.
        main_algorithm: the name of the algorithm to simulate. This is going to
            be the main subject of study of the simulation.
        fallback_algorithm: an emergency algorithm to use in the cases the main
            algorithm couldn't produce a solution.

    Attributes:
        UTXOS_ACTIVITY_CSV_HEADER: a tuple to place at the beginning of the
        `utxo_activity.csv` file.
        PAYMENT_REQUEST: the string to identify payment requests.
        INCOME: the string to identify incoming UTxOs.
        TX_PAYMENT: the string to identify payments effectively made.
        TX_INPUT: the string to identify UTxOs used as transaction inputs.
        TX_CHANGE: the string to identify change UTxOs produced by transactions.
    """

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
        #: the scenario selected for the simulation
        self.scenario: DataFrame = scenario

        #: the name of the scenario
        self.scenario_name = getattr(self.scenario, "name", "")

        #: the main algorithm to run the scenario
        self.main_algorithm: CoinSelectionAlgorithm = main_algorithm

        #: the algorithm to use in case of failure of the main algorithm
        self.fallback_algorithm: CoinSelectionAlgorithm = fallback_algorithm

        #: the number of processed payments from the scenario
        self.processed_payments: int = 0

        #: the total amount of payments in the scenario
        self.total_payments: int = (
            scenario[scenario["amount"] < 0]["block_id"].unique().shape[0]
        )

        #: the simulation wallet
        self.wallet = Wallet()

        #: the payments that should be paid in the next coin selection
        self.pending_payments = []

        #: the file path to store models in case of failure of the solver
        self.failed_txs_path: Path = output_dir / "failed_txs"

        #: the file path to store a digest of each coin selection
        self.txs_path: Path = output_dir / "transactions_summary.csv"

        #: the file path to store the utxo activity of the wallet
        self.utxos_path: Path = output_dir / "utxo_activity.csv"

    def process_block(self, block_id: Hashable, block: DataFrame) -> None:
        """Extract the incoming and outgoing amounts of a scenario block.

        Args:
            block_id: the identifier assigned to the incoming or outgoing
                amount.
            block: a DataFrame containing the incoming or outgoing amounts
                produced in that block of the scenario.
        """
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
        """Coin select with the main algorithm or the fallback one if it fails.

        Args:
            block_id: the identifier assigned to the produced transaction.
            context: the selection context of the particular transaction. It
                includes data needed during selection like fee rate, cost of
                change or payment amount.

        Returns:
            A TxDescriptor representing the transaction.
        """
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
        """Update the simulation state with the latest transaction made.

        This function updates the simulation state, which depeneds of the
        pendign payments, the UTxOs used as inputs for the transaction, the
        change new UTxOs produces and the simulation wallet.

        Args:
            block_id: the identifier assigned to the produced transaction.
            tx: a descriptor of the transaction made.
        """
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
        """Read scenario, produce selection context and coin select.

        This is the main place where the simulation takes place. The function
        setups the csv file headers and starts reading the scenario data
        grouping by `block_id`. Process the `block` and checks for pending
        payments. If any, it creates a SelectionContext with the fee rate of the
        last block procesed as the target fee rate of the coin selection. Then
        it tries to produce a valid coin selection, produces statistics about
        the transaction produced if any and stores the data in the respective
        csv files.
        """
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
        """Setup the simulation files to store the data and run the simulation."""
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
    """Parse the cli command options and execute the simulation.

    Args:
        ctx: dictionary containing the option and flags of the parent command.
        scenario: a pattern matching the scenarios to simulate.
        model: if set, the name of the model to simulate the scenarios with.
        exclude: if set, a list of strings of the format
            '<scenario-pattern>,<model>' where `scenario-pattern` matches the
            scenario to exclude and `model` is the name of the model to exclude.
            Each strings represent a combination of scenarios and models to not
            simulate.
    """
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
