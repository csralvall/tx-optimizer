import random
from collections.abc import Callable
from functools import partial
from multiprocessing import cpu_count
from typing import Protocol, cast

import structlog
from pulp import (
    PULP_CBC_CMD,
    LpAffineExpression,
    LpMinimize,
    LpProblem,
    LpSolver,
    LpVariable,
    lpSum,
)

from datatypes.fee_rate import FeeRate
from datatypes.transaction import TxDescriptor
from selection.context import DustUTxO, SelectionContext
from selection.metrics import waste

LOGGER = structlog.stdlib.get_logger(__name__)

DEFAULT_SOLVER = PULP_CBC_CMD(msg=False, timeLimit=15, threads=cpu_count() - 1)


class UTxOSelectionFailed(Exception):
    "Raised when the solution is infeasible with the restrictions given."
    pass


class CoinSelectionAlgorithm(Protocol):
    def __call__(self, selection_context: SelectionContext) -> TxDescriptor:
        ...

    def __name__(self) -> str:
        ...


def greatest_first(selection_context: SelectionContext) -> TxDescriptor:
    fee_rate = selection_context.fee_rate
    selection_input_ids = [
        fee_rated_utxo.utxo_key.id
        for fee_rated_utxo in selection_context.fee_rated_utxos[
            : selection_context.minimal_number_of_inputs
        ]
    ]
    tx = selection_context.get_tx(selection_input_ids)
    overpayment = (
        sum(utxo.amount for utxo in tx.inputs) - selection_context.target
    )

    try:
        tx.change.append(selection_context.get_change_utxo(overpayment))
    except DustUTxO:
        tx.excess = int(overpayment)

    tx.fix_rounding_errors(fee_rate)

    return tx


def single_random_draw(selection_context: SelectionContext) -> TxDescriptor:
    random.shuffle(selection_context.fee_rated_utxos)
    selected_amount: int = 0
    selected_input_ids: list[int] = []
    for fee_rated_utxo in selection_context.fee_rated_utxos:
        selected_amount += fee_rated_utxo.effective_amount
        selected_input_ids.append(fee_rated_utxo.utxo_key.id)
        if selected_amount > selection_context.target:
            break

    tx = selection_context.get_tx(selected_input_ids)

    overpayment: int = selected_amount - selection_context.target

    try:
        tx.change.append(selection_context.get_change_utxo(overpayment))
    except DustUTxO:
        tx.excess = int(overpayment)

    tx.fix_rounding_errors(selection_context.fee_rate)

    return tx


def minimize_inputs_without_change(
    selection_context: SelectionContext,
    solver: LpSolver = DEFAULT_SOLVER,
) -> TxDescriptor:
    # Equation Terms
    equation_terms: list[tuple[LpVariable, float]] = [
        (
            LpVariable(f"x_{fee_rated_utxo.utxo_key.id}", cat="Binary"),
            fee_rated_utxo.effective_amount,
        )
        for fee_rated_utxo in selection_context.fee_rated_utxos
    ]

    # Desicion variable
    desicion_variables = [
        list(terms) for terms in zip(*equation_terms, strict=True)
    ][0]

    total_input = LpAffineExpression(equation_terms)
    overpayment = total_input - selection_context.target

    # Define model
    model = LpProblem("minimize_inputs_avoiding_change", LpMinimize)

    # Objective function
    model += overpayment

    # Constraints
    model += (
        lpSum(desicion_variables)
        == selection_context.minimal_number_of_inputs,
        "use_the_minimal_number_of_inputs_constraint",
    )

    model += (overpayment >= 0, "pay_requested_amount_constraint")

    model += (
        overpayment <= selection_context.change_cost,
        "avoid_change_constraint",
    )

    # Run solver
    model.solve(solver)

    if model.status < 0:
        raise UTxOSelectionFailed

    selected_input_ids = [
        int(variable.name.split("_")[1])
        for variable in model.variables()
        if variable.value()
    ]
    tx = selection_context.get_tx(selected_input_ids)

    tx.excess = int(cast(int, overpayment.value()))

    tx.fix_rounding_errors(selection_context.fee_rate)

    return tx


def avoid_change(
    selection_context: SelectionContext,
    solver: LpSolver = DEFAULT_SOLVER,
) -> TxDescriptor:
    # Equation Terms
    desicion_terms: list[tuple[LpVariable, float]] = [
        (
            LpVariable(f"x_{fee_rated_utxo.utxo_key.id}", cat="Binary"),
            fee_rated_utxo.effective_amount,
        )
        for fee_rated_utxo in selection_context.fee_rated_utxos
    ]

    total_input = LpAffineExpression(desicion_terms)
    overpayment = total_input - selection_context.target

    # Define model
    model = LpProblem("avoid_change", LpMinimize)

    # Objective function
    model += overpayment

    # Constraints
    model += (
        overpayment >= 0,
        "input_amount_exeeds_payments_plus_fees_constraint",
    )

    model += (
        overpayment <= selection_context.change_cost,
        "avoid_change_constraint",
    )

    # Run solver
    model.solve(solver)

    if model.status < 0:
        raise UTxOSelectionFailed

    selected_input_ids = [
        int(variable.name.split("_")[1])
        for variable in model.variables()
        if variable.value()
    ]
    tx = selection_context.get_tx(selected_input_ids)

    tx.excess = int(cast(int, overpayment.value()))

    tx.fix_rounding_errors(selection_context.fee_rate)

    return tx


def minimize_waste_without_change(
    selection_context: SelectionContext,
    solver: LpSolver = DEFAULT_SOLVER,
) -> TxDescriptor:
    # Overpayment Terms
    desicion_terms: list[tuple[LpVariable, float]] = [
        (
            LpVariable(f"x_{fee_rated_utxo.utxo_key.id}", cat="Binary"),
            fee_rated_utxo.effective_amount,
        )
        for fee_rated_utxo in selection_context.fee_rated_utxos
    ]

    desicion_variables = [
        list(terms) for terms in zip(*desicion_terms, strict=True)
    ][0]

    total_input = LpAffineExpression(desicion_terms)

    overpayment = total_input - selection_context.target

    # Waste Terms
    timing_terms: list[tuple[LpVariable, int]] = [
        (
            LpVariable(f"y_{utxo.wallet_id}", cat="Binary"),
            utxo.input_fee(selection_context.fee_rate_delta),
        )
        for utxo in selection_context.wallet
    ]

    timing_variables = [
        list(terms) for terms in zip(*timing_terms, strict=True)
    ][0]

    timing_cost = LpAffineExpression(timing_terms)

    waste = timing_cost

    # Define model
    model = LpProblem("minimize_waste_avoiding_change", LpMinimize)

    # Objective function
    model += waste + overpayment

    # Constraints
    model += (overpayment >= 0, "pay_requested_amount_constraint")

    model += (
        overpayment <= selection_context.change_cost,
        "avoid_change_constraint",
    )

    for idx, (desicion_var, timing_var) in enumerate(
        zip(desicion_variables, timing_variables, strict=True)
    ):
        consolidation_constraint = LpAffineExpression(
            [(desicion_var, 1), (timing_var, -1)]
        )
        model += (
            lpSum(consolidation_constraint) == 0,
            f"consolidate_variables_constraint_{idx}",
        )

    # Run solver
    model.solve(solver)

    if model.status < 0:
        raise UTxOSelectionFailed

    selected_input_ids: list[int] = []
    for variable in model.variables():
        if "y" in variable.name:
            break

        if variable.value():
            selected_input_ids.append(int(variable.name.split("_")[1]))

    tx = selection_context.get_tx(selected_input_ids)

    overpayment_amount: int = cast(int, overpayment.value())

    tx.excess = int(overpayment_amount)

    # recover extra sats given away by rounding errors
    tx.fix_rounding_errors(selection_context.fee_rate)

    return tx


def minimize_waste_with_change(
    selection_context: SelectionContext,
    solver: LpSolver = DEFAULT_SOLVER,
) -> TxDescriptor:
    # Overpayment Terms
    desicion_terms: list[tuple[LpVariable, int]] = [
        (
            LpVariable(f"x_{fee_rated_utxo.utxo_key.id}", cat="Binary"),
            fee_rated_utxo.effective_amount,
        )
        for fee_rated_utxo in selection_context.fee_rated_utxos
    ]

    desicion_variables = [
        list(terms) for terms in zip(*desicion_terms, strict=True)
    ][0]

    total_input = LpAffineExpression(desicion_terms)

    overpayment = total_input - selection_context.target

    # Waste Terms

    # Difference between current fee and consolidation fee to get timing cost

    timing_terms: list[tuple[LpVariable, int]] = [
        (
            LpVariable(f"y_{utxo.wallet_id}", cat="Binary"),
            utxo.input_fee(selection_context.fee_rate_delta),
        )
        for utxo in selection_context.wallet
    ]

    timing_variables = [
        list(terms) for terms in zip(*timing_terms, strict=True)
    ][0]

    timing_cost = LpAffineExpression(timing_terms)

    waste = timing_cost + selection_context.change_cost

    # Define model
    model = LpProblem("minimize_waste_including_change", LpMinimize)

    # Objective function
    model += waste + overpayment

    # Constraints
    model += (
        overpayment >= selection_context.change_cost,
        "pay_above_change_threshold_constraint",
    )

    for idx, (desicion_var, timing_var) in enumerate(
        zip(desicion_variables, timing_variables, strict=True)
    ):
        consolidation_constraint = LpAffineExpression(
            [(desicion_var, 1), (timing_var, -1)]
        )
        model += (
            lpSum(consolidation_constraint) == 0,
            f"consolidate_variables_constraint_{idx}",
        )

    # Run solver
    model.solve(solver)

    if model.status < 0:
        raise UTxOSelectionFailed

    selected_input_ids: list[int] = []
    for variable in model.variables():
        if "y" in variable.name:
            break

        if variable.value():
            selected_input_ids.append(int(variable.name.split("_")[1]))

    tx = selection_context.get_tx(selected_input_ids)

    overpayment_amount: int = cast(int, overpayment.value())

    # overpayment_amount should be above change output costs
    change_utxo = selection_context.get_change_utxo(overpayment_amount)
    tx.change.append(change_utxo)

    # recover extra sats given away by rounding errors
    tx.fix_rounding_errors(selection_context.fee_rate)

    return tx


def minimize_waste_pairing_change_effective_value_with_payments(
    selection_context: SelectionContext,
    solver: LpSolver = DEFAULT_SOLVER,
) -> TxDescriptor:
    # Overpayment Terms
    desicion_terms: list[tuple[LpVariable, float]] = [
        (
            LpVariable(f"x_{fee_rated_utxo.utxo_key.id}", cat="Binary"),
            fee_rated_utxo.effective_amount,
        )
        for fee_rated_utxo in selection_context.fee_rated_utxos
    ]

    desicion_variables = [
        list(terms) for terms in zip(*desicion_terms, strict=True)
    ][0]

    total_input = LpAffineExpression(desicion_terms)

    low_limit: float = (
        selection_context.payments_mean - selection_context.payments_stdev
    )
    top_limit: float = (
        selection_context.payments_mean + selection_context.payments_stdev
    )
    desired_change: int = 0
    for payment in selection_context.payments:
        if payment.amount >= low_limit and payment.amount <= top_limit:
            desired_change += payment.amount

    target: int = (
        selection_context.base_fee
        + selection_context.payment_amount
        + desired_change
    )

    # aim for a change equal to the accumulated amount of most common payments
    excess = total_input - target

    # Waste Terms
    # Difference between current fee and consolidation fee to get timing cost
    fee_delta: FeeRate = selection_context.fee_rate_delta

    timing_terms: list[tuple[LpVariable, int]] = [
        (
            LpVariable(f"y_{utxo.wallet_id}", cat="Binary"),
            utxo.input_fee(fee_delta),
        )
        for utxo in selection_context.wallet
    ]

    timing_variables = [
        list(terms) for terms in zip(*timing_terms, strict=True)
    ][0]

    timing_cost = LpAffineExpression(timing_terms)

    waste = timing_cost + selection_context.change_cost

    # Define model
    model = LpProblem(
        "minimize_waste_pairing_change_effective_value_with_payments",
        LpMinimize,
    )

    # Objective function
    model += waste + excess

    # Constraints
    model += (excess >= 0, "pay_requested_amount_constraint")

    for idx, (desicion_var, timing_var) in enumerate(
        zip(desicion_variables, timing_variables, strict=True)
    ):
        consolidation_constraint = LpAffineExpression(
            [(desicion_var, 1), (timing_var, -1)]
        )
        model += (
            lpSum(consolidation_constraint) == 0,
            f"consolidate_variables_constraint_{idx}",
        )

    # Run solver
    model.solve(solver)

    if model.status < 0:
        raise UTxOSelectionFailed

    selected_input_ids: list[int] = []
    for variable in model.variables():
        if "y" in variable.name:
            break

        if variable.value():
            selected_input_ids.append(int(variable.name.split("_")[1]))

    tx = selection_context.get_tx(selected_input_ids)

    overpayment_amount: int = cast(int, excess.value())

    change_utxo = selection_context.get_change_utxo(overpayment_amount)
    tx.change.append(change_utxo)

    try:
        tx.change.append(selection_context.get_change_utxo(overpayment_amount))
    except DustUTxO:
        tx.excess = overpayment_amount
        pass

    tx.fix_rounding_errors(selection_context.fee_rate)

    return tx


def minimize_waste(selection_context: SelectionContext) -> TxDescriptor:
    txs: list[TxDescriptor] = []
    try:
        txs.append(minimize_waste_without_change(selection_context))
    except UTxOSelectionFailed:
        pass
    finally:
        txs.append(minimize_waste_with_change(selection_context))

    current_waste: Callable = partial(
        waste, fee_rate=selection_context.fee_rate
    )
    return min(txs, key=lambda x: current_waste(x))


def maximize_effective_value(
    selection_context: SelectionContext,
) -> TxDescriptor:
    txs: list[TxDescriptor] = []
    try:
        txs.append(minimize_waste_without_change(selection_context))
    except UTxOSelectionFailed:
        pass
    finally:
        txs.append(
            minimize_waste_pairing_change_effective_value_with_payments(
                selection_context
            )
        )

    current_waste: Callable = partial(
        waste, fee_rate=selection_context.fee_rate
    )
    return min(txs, key=lambda x: current_waste(x))
