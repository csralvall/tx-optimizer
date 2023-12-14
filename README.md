# Bitcoin transaction optimizer

A command line application written in Python to simulate bitcoin transactions
using *Binary Integer Programming* models to test selection policies and
compare different coin selection algorithms.

This application uses [**Pulp**](https://github.com/coin-or/pulp) for modeling
the selection function and restrictions and it can compute the models in any of
the solvers for which **Pulp** interface is implemented, although for the
stored simulations, the [**Branch and Bound**](https://github.com/coin-or/Cbc)
solver of coin-OR was used.

The currently implemented algorithms/models are five:
- `greatest-first`: a greedy implementation that always select the biggest
`UTxO`.
- `single-random-draw`: randomly select `UTxOs` until meeting the criteria.
- `avoid-change`: try to produce changeless transaction always. Inspired by one
of the Binary Linear Programming models used by Daniel J. Diroff in its work
[*Bitcoin Coin Selection with Leverage*](https://arxiv.org/abs/1911.01330).
- `minimize-waste`: the most similar to the actual implementation of Bitcoin
Core's `branch-and-bound` algorithm. It selects `UTxOs` trying to minimize the
`Waste` metric, proposed by Mark Erhardt.
- `maximize-effective-value`: a tweak of the `minimize-waste` policy blended
with the `double-target` strategy, mentioned in the [Master
Thesis](https://murch.one/wp-content/uploads/2016/11/erhardt2016coinselection.pdf)
of Mark Erhardt and in the same spirit of the [*Random Improve* selection
policy](https://iohk.io/en/blog/posts/2018/07/03/self-organisation-in-coin-selection/)
proposed by Edsko De Vries for Cardano. It tries to minimize waste but
producing a change output closer to the median amount of the payments.

In case of selection failure of any of the above algorithms, the emergency
selection algorithm, `single-random-draw` will try to solve the selection. If
it also fails, the payment obligation will stay in queue while the simulator
continues processing incoming transaction, until the next payment obligation is
added to the queue, and the simulated selection algorithms comes to action
again.

The data in the scenarios cames from the [simulator](https://github.com/achow101/coin-selection-simulation)
implemented by Ava Chow and Mark Erhardt. It has been tweaked to process multi
payment transactions, although the stored simulations didn't use the feature.

The application also implements a tool to generate scenarios. It allows the
extraction of fee estimations from a personal node's blockchain, the generation
of custom transaction profiles using mathematical distributions and the
combination of those transactions and fee estimations in new scenarios.

You can also produce new transaction profiles, fee rate market fluctuations or
even complete scenarios with external tools and load them usign a different
data path.

## Setup

1. Install [pdm](https://pdm-project.org/).
2. Create virtual environment with any python `>=3.10`:
    ```python
    pdm venv create 3.10
    ```
3. Check the virtual environment is activated:
    ```bash
    eval $(pdm venv activate -v)
    ```
3. Install dependencies:
    ```python
    pdm install
    ```
4. Execute simulation:
    ```python
    python simulate.py --all
    ```

## Data

The `data/` directory is a sample of how the data consumed by the tool should
be structured to run the simulations without hassle. To replace this default
directory by another use the option `--data-path PATH` like:
```bash
./btc_selection --data-path <new-data-path> <command>
```

### Fee rates

In `data/fee_rates` you will find the fee rates extracted from [coin-selection-simulation](https://github.com/achow101/coin-selection-simulation/blob/main/README.md).
The files has been adapted to include a `block_id`.

All files should be *csv* files, where each line follows the format:
`<block_id>,<fee-rate-per-vKb>`.

The `block_id` introduces temporality to these fee rate corpus, allowing the
generation of derived fee rate scenarios focusing on particular periods, with
high fees or rapid changes in the fee rate, for example.

The command to extract the fee rate estimation of each block from a node's
blockchain is:
```bash
./btc_selection scenario extract-fees --help
```

### Transactions

In `data/transactions` you will find the transactions extracted from [coin-selection-simulation](https://github.com/achow101/coin-selection-simulation/blob/main/README.md).
The files has been adapted to also include a `block_id`.

All files should be *csv* files, where each line follows the format:
`<block_id>,<positive-or-negative-transaction-amount-in-bitcoin>`.

The sign of the amount determines if the transaction is an income for the
wallet or a payment obligation.

For transaction, the idea of `block_id`, is to batch incoming amounts or
outgoing payments in a single transaction. For incoming amounts it doesn't have
any use yet, but for outgoing amounts it allows the creation of multi payment
transactions, i.e., with more than one output besides the possible change
output.

The command to generate transaction profiles based on mathematical
distributions is:
```bash
./btc_selection scenario create-txs --help
```

### Scenarios

In `data/scenarios` you will find the transactions extracted from
[coin-selection-simulation](https://github.com/achow101/coin-selection-simulation/blob/main/README.md),
also with the extra `block_id` added.

All files should be *csv* files, where each line follows the format:
`<block_id>,<signed-transaction-amount-in-bitcoin>,<fee-rate-per-vKb>`.

These are the files consumed by the simulator.

You can create multiple scenarios combining the data stored in the
`data/fee_rates` and the `data/transactions` directories with the command:
```bash
./btc_selection scenario generate --help
```

It combines fee rates with transactions based on the `block_id` mentioned
before, so to work properly the range of the ids in the transaction file should
include the range of the ids in the fee rate file.

### Simulations

In the directory `data/simulations` are stored the resulting simulations. Its
format is the following:
```
/data/simulations/
├── <day>_<month>_<year>__<hour>_<minute>_<second>/
│   ├── <scenario-name>/
│   │   ├── <model-name-1>/
│   │   │   ├── failed_txs/
│   │   │   ├── transactions_summary.csv
│   │   │   └── utxo_activity.csv
│   │   └── <model-name-2>/
│   │       ├── failed_txs/
│   │       ├── transactions_summary.csv
│   │       └── utxo_activity.csv
│   └── simulation_summary.json
└── <day>_<month>_<year>__<hour>_<minute>_<second>/
```

- **transactions_summary.csv**: a digest of each transaction produced by a model
during the simulation. It records the following fields from first to last on
each transaction:
    - `id`: a transaction identifier
    - `policy`: the name of the algorithm/model applied.
    - `balance`: the remaining balance in the wallet after the selection.
    - `#wallet`: the remaining amount of `UTxOs` in wallet after the selection.
    - `#inputs`: the number of inputs.
    - `#payments`: the number of payment outputs.
    - `#change`: the number of change ouputs.
    - `#negative_effective_valued_utxos`: the number of inputs with negative effective value.
    - `excess`: the extra amount of bitcoin released to the miners.
    - `change_effective_value`: the effective amount of the change output created.
    - `waste`: the value of the Waste metric.
    - `fee`: the final fee
    - `final_fee_rate`: the final fee rate
    - `target_fee_rate`: the fee rate the transaction was trying to achieve
    - `cpu_time`: the time consumed by the solver/algorithm to produce the
    transaction without considering other processes.
- **utxo_activity.csv**:
    - `block_id`: the identifier of the transaction.
    - `wallet_id`: an internal unique identifier for each `UTxO` in the wallet.
    Payments use the same identifier as they don't belong to the wallet (-1).
    - `condition`: the endorsed condition in which the `UTxO` participated of the
    wallet operation.
    - `amount`: the amount carried by the `UTxO`.

- **failed_txs**: a directory to store the failed models when there is an error
with the `coin-or/Cbc` solver.

## Recipes

- Run all implemented models over all scenarios:
    ```bash
    ./btc_selection simulate
    ```
- Run bitcoin random coin selection for one model over all scenarios:
    ```bash
    ./btc_selection simulate --model single-random-draw
    ```
- Run bitcoin changeless coin selection for the scenario `random_blocks`.
    ```bash
    ./btc_selection simulate --model single-random-draw --scenario random_blocks.csv
    ```
- Run all model and scenario combinations except `maximize-effective-value` for
  all scenarios and the combination of the `random-blocks` scenario with the
  `minimize-waste` model:
    ```bash
    ./btc_selection simulate --exclude "*,maximize-effective-value" --exclude "random-blocks,minimize-waste"
    ```
