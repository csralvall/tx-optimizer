from getpass import getpass
from pathlib import Path

import click
from sshtunnel import SSHTunnelForwarder

from utils.authproxy import AuthServiceProxy
from utils.bitcoin import btc_to_str, sat_to_btc

SSH_DEFAULT_PORT = 22
BITCOIN_CORE_RPC_MAINET_PORT = 8332
LOCALHOST = "127.0.0.1"


@click.command()
@click.option(
    "-w",
    "--wallet",
    type=str,
    default="",
    help="The name of the wallet to generate from.",
)
@click.option(
    "-f",
    "--filename",
    type=str,
    default="",
    help="File to output the fees to.",
)
@click.option(
    "-u",
    "--node-user",
    type=str,
    help="User for the bitcoin core host.",
    required=True,
)
@click.option(
    "--rpc-user",
    "rpc_user",
    type=str,
    help="User for the RPC interface.",
    required=True,
)
@click.option(
    "-a",
    "--node-address",
    "node_address",
    type=str,
    help="IP address of the bitcoin node.",
    required=True,
)
@click.option(
    "-k",
    "--key",
    type=str,
    help="Path to the key to connect to the bitcoin node.",
    required=True,
)
@click.option(
    "--from",
    "start",
    default=1,
    type=int,
    help="Initial block to read from bitcoin mainet blockchain.",
)
@click.option(
    "--to",
    "end",
    default=0,
    type=int,
    help="Final block to read from bitcoin mainet blockchain.",
)
def extract_fees(
    wallet: str = "",
    filename: str = "",
    node_user: str = "",
    rpc_user: str = "",
    node_address: str = "",
    key: str = "",
    start: int = 0,
    end: int = 0,
):
    """Script to extract the average fee rate from bitcoin blocks."""
    bitcoin_server = SSHTunnelForwarder(
        (node_address, SSH_DEFAULT_PORT),
        ssh_pkey=key,
        ssh_username=node_user,
        ssh_password=getpass(prompt="Bitcoin core host password: "),
        allow_agent=False,
        remote_bind_address=(LOCALHOST, BITCOIN_CORE_RPC_MAINET_PORT),
    )

    bitcoin_server.start()

    rpc_port: str = bitcoin_server.local_bind_port
    rpc_password: str = getpass(prompt="Bitcoin RPC password: ")
    rpc = AuthServiceProxy(
        f"http://{rpc_user}:{rpc_password}@{LOCALHOST}:{rpc_port}/wallet/{wallet}"
    )

    fee_rate_dir = Path.cwd() / "../data/feerates"
    fee_rate_dir.mkdir(parents=True, exist_ok=True)

    if not end:
        end = rpc.getblockcount()

    if not filename:
        filename = f"blocks_{start}_to_{end}.csv"

    filepath = fee_rate_dir / filename

    # Fetch all of the transactions in the wallet
    with filepath.open("a") as f:
        for block_number in range(start, end):
            block_data: dict = rpc.getblockstats(block_number, ["avgfeerate"])
            avg_fee_rate: int = block_data["avgfeerate"]  # sat/vB
            if avg_fee_rate > 0:
                f.write(
                    f"{block_number},{btc_to_str(sat_to_btc(avg_fee_rate))}\n"
                )
