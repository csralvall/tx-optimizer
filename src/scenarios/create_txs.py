import csv
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from random import random

import click
import numpy as np
import structlog
from numpy import ndarray
from pandas import DataFrame

from utils.bitcoin import btc_round

LOGGER = structlog.stdlib.get_logger(__name__)


def flip(ratio: float) -> int:
    return 1 if random() < ratio else -1


def get_btc_sample(dist: Callable, size: int) -> list[float]:
    sample: list[float] = []
    while len(sample) != size:
        amount: float = dist()
        if amount > 0:
            sample.append(btc_round(amount))

    return sample


def group_txs(
    txs: list[float], grouping_freq: int = 1
) -> list[tuple[int, float]]:
    outgoing_tx_count: int = sum(1 for tx in txs if tx < 0)
    block_sizes: ndarray = np.random.poisson(grouping_freq, outgoing_tx_count)
    grouped_txs: list[tuple[int, float]] = []
    txs_iter = iter(txs)
    block_id: int
    block_size: int
    for block_id, block_size in enumerate(block_sizes):
        if not block_size:
            continue
        block: list[tuple[int, float]] = []
        for tx in txs_iter:
            block.append((block_id, tx))
            if len(block) >= block_size and any(v < 0 for (_, v) in block):
                break

        grouped_txs.extend(block)
        if len(txs) == len(grouped_txs):
            break

    return grouped_txs


def generate_txs(
    incoming_dist: Callable,
    outgoing_dist: Callable,
    incoming_outgoing_ratio: str,
    outgoing_tx_count: int = 5000,
) -> list[float]:
    _in: int
    _out: int
    _in, _out = list(map(int, incoming_outgoing_ratio.split(":")))

    ratio: float = _in / (_in + _out)
    # flip until outgoing_tx_count is satisfied
    txs: list[int] = []
    while outgoing_tx_count:
        tx_type: int = flip(ratio)
        txs.append(flip(ratio))
        if tx_type < 0:
            outgoing_tx_count -= 1

    type_counts = Counter(txs)
    incoming_amounts: list[float] = get_btc_sample(
        incoming_dist, size=type_counts[1]
    )
    outgoing_amounts: list[float] = get_btc_sample(
        outgoing_dist, size=type_counts[-1]
    )

    final_txs: list[float] = []
    for tx_type in txs:
        if tx_type > 0:
            final_txs.append(tx_type * incoming_amounts.pop())
        else:
            final_txs.append(tx_type * outgoing_amounts.pop())

    return final_txs


def resample_fee_rates(
    fee_rates: DataFrame, start: int, end: int, size: int
) -> DataFrame:
    filtered_fee_rates: DataFrame = fee_rates[
        (fee_rates["block_id"] > start) & (fee_rates["block_id"] < end)
    ]
    if len(filtered_fee_rates) < size:
        return filtered_fee_rates.sample(n=size, random_state=1, replace=True)
    return filtered_fee_rates.sample(n=size, random_state=1).sort_index()


def txs_to_csv(
    data_path: Path, filename: str, txs: list[tuple[int, float]]
) -> None:
    transactions_dir: Path = data_path / "transactions"
    transactions_dir.mkdir(parents=True, exist_ok=True)
    path: Path = transactions_dir / f"{filename}.csv"
    with path.open(mode="w") as txs_file:
        writer = csv.writer(txs_file)
        for data in txs:
            writer.writerow(data)


def default_txs(data_path: Path) -> None:
    binance_exchange_txs: list[float] = generate_txs(
        incoming_dist=partial(np.random.normal, loc=3, scale=1),
        outgoing_dist=partial(np.random.normal, loc=5, scale=1),
        incoming_outgoing_ratio="2:1",
    )
    binance_blocks: list[tuple[int, float]] = group_txs(
        binance_exchange_txs, grouping_freq=12
    )
    coinbase_exchange_txs: list[float] = generate_txs(
        incoming_dist=partial(np.random.normal, loc=0.006, scale=1),
        outgoing_dist=partial(np.random.normal, loc=0.008, scale=1),
        incoming_outgoing_ratio="5:1",
    )
    coinbase_blocks: list[tuple[int, float]] = group_txs(
        coinbase_exchange_txs, grouping_freq=12
    )
    bitpay_payment_processor_txs: list[float] = generate_txs(
        incoming_dist=partial(np.random.normal, loc=0.25, scale=1),
        outgoing_dist=partial(np.random.normal, loc=5, scale=1),
        incoming_outgoing_ratio="10:1",
    )
    bitpay_blocks: list[tuple[int, float]] = group_txs(
        bitpay_payment_processor_txs, grouping_freq=8
    )
    poolin_mining_pool_txs: list[float] = generate_txs(
        incoming_dist=partial(np.random.normal, loc=3, scale=1),
        outgoing_dist=partial(np.random.normal, loc=5, scale=1),
        incoming_outgoing_ratio="2:1",
    )
    poolin_blocks: list[tuple[int, float]] = group_txs(
        poolin_mining_pool_txs, grouping_freq=6
    )
    xapo_custodial_service_txs: list[float] = generate_txs(
        incoming_dist=partial(np.random.normal, loc=0.05, scale=1),
        outgoing_dist=partial(np.random.normal, loc=0.5, scale=1),
        incoming_outgoing_ratio="3:1",
    )
    xapo_blocks: list[tuple[int, float]] = group_txs(
        xapo_custodial_service_txs, grouping_freq=6
    )
    scenarios: dict[str, list[tuple[int, float]]] = {
        "binance_exchange_profile": binance_blocks,
        "coinbase_exchange_profile": coinbase_blocks,
        "bitpay_payment_processor_profile": bitpay_blocks,
        "poolin_mining_pool_profile": poolin_blocks,
        "xapo_custodial_service_profile": xapo_blocks,
    }

    for name, scenario in scenarios.items():
        txs_to_csv(data_path, name, scenario)


@click.command(help="Create custom or default transaction profiles.")
@click.option(
    "--incoming", type=dict, help="Distribution of incoming transactions"
)
@click.option(
    "--outgoing", type=dict, help="Distribution of outgoing transactions"
)
@click.option(
    "--ratio",
    type=str,
    default="1:1",
    help="Relation between incoming and outgoing transactions.",
)
@click.option(
    "--size", type=int, default=0, help="Amount of outgoing transactions"
)
@click.option(
    "--freq",
    type=int,
    default=0,
    help="Mean amount of transactions per block.",
)
@click.argument("filename", default="")
@click.pass_obj
def create_txs(
    ctx,
    incoming: dict,
    outgoing: dict,
    ratio: str = "1:1",
    size: int = 0,
    freq: int = 0,
    filename: str = "",
) -> None:
    data_path = ctx.get("data_path")
    if not filename:
        default_txs(data_path)
        return
    incoming_dist = getattr(np.random, incoming.pop("name"))
    outgoing_dist = getattr(np.random, outgoing.pop("name"))
    incoming_txs = partial(incoming_dist, **incoming)
    outgoing_txs = partial(outgoing_dist, **outgoing)
    txs: list[float] = generate_txs(incoming_txs, outgoing_txs, ratio, size)
    blocks: list[tuple[int, float]] = group_txs(txs, freq)
    txs_to_csv(data_path, filename, blocks)
