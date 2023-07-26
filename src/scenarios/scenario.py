import csv
from pathlib import Path

import click
import pandas
from pandas import DataFrame

from scenarios.create_txs import create_txs
from scenarios.extract_fees import extract_fees
from utils.bitcoin import btc_to_str


def get_fee_rate_per_block(fee_rates: DataFrame, block_ids: list[int]) -> dict:
    fee_rate_dict: dict = {}
    only_fee_rates: list[float] = [
        serie.fee for _, serie in fee_rates.iterrows()
    ]
    only_fee_rates = only_fee_rates[: len(block_ids)]
    for block_id, fee in zip(block_ids, only_fee_rates, strict=True):
        fee_rate_dict.setdefault(block_id, fee)

    return fee_rate_dict


def join_fee_rate_with_txs(
    data_path: Path, fee_rates: DataFrame, txs: DataFrame, scenario_name: str
) -> None:
    scenarios_dir = data_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = scenarios_dir / f"{scenario_name}.csv"
    block_ids: list[int] = [tx.block_id for _, tx in txs]
    fee_rate_dict: dict[int, float] = get_fee_rate_per_block(
        fee_rates, block_ids
    )
    with scenario_file.open(mode="w") as csv_output:
        writer = csv.writer(csv_output)
        for _, tx in txs:
            formatted_amount = btc_to_str(tx.amount)
            formatted_fee_rate = btc_to_str(fee_rate_dict.get(tx.block_id, 0))
            writer.writerow(
                (tx.block_id, formatted_amount, formatted_fee_rate)
            )


def generate_scenarios_from_data(data_path: Path) -> None:
    fee_rates_dir: Path = data_path / "feerates"
    fee_rates_dir.mkdir(parents=True, exist_ok=True)
    txs_dir: Path = data_path / "transactions"
    txs_dir.mkdir(parents=True, exist_ok=True)
    for fee_path in fee_rates_dir.glob("*.csv"):
        for txs_path in txs_dir.glob("*.csv"):
            with fee_path.open(mode="r") as fee_rate_input, txs_path.open(
                mode="r"
            ) as txs_input:
                fee_rate_scenario: DataFrame = pandas.read_csv(
                    fee_rate_input, names=["block_id", "fee"]
                )
                txs_scenario: DataFrame = pandas.read_csv(
                    txs_input, names=["block_id", "amount"]
                )
                join_fee_rate_with_txs(
                    data_path,
                    fee_rate_scenario,
                    txs_scenario,
                    f"{txs_path.name}_with_{fee_path.name}",
                )


@click.command(
    help="Produce scenarios from /transactions and /feerate, and save them in /scenarios."
)
@click.pass_obj
def generate(ctx) -> None:
    data_path = ctx.obj.get("data_path")
    generate_scenarios_from_data(data_path)


@click.group(help="Produce scenarios to feed coin selection simulation.")
def scenario() -> None:
    return None


scenario.add_command(create_txs)
scenario.add_command(extract_fees)
scenario.add_command(generate)
