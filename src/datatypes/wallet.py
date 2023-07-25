from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import structlog
from sortedcontainers import SortedDict

from datatypes.utxo import UTxO

LOGGER = structlog.stdlib.get_logger(__name__)


class UTxONotInWallet(Exception):
    """Raise when the UTxO used in the wallet operation doesn't belong to it."""

    pass


class InconsistentWallet(Exception):
    """Raise when wallet indexes have differences in their pointed utxos."""

    pass


@dataclass
class Wallet:
    balance: int = 0
    _index: int = 0
    _lookup_pool: dict = field(default_factory=dict)
    _utxo_pool: SortedDict = field(default_factory=SortedDict)

    def add(self, utxo: UTxO) -> None:
        utxo.wallet_id = self._index
        self._utxo_pool[(utxo.amount, self._index)] = utxo
        self._lookup_pool[self._index] = utxo
        self._index += 1
        self.balance += utxo.amount

    def pop(self, utxo: UTxO) -> UTxO:
        if utxo.wallet_id < 0:
            raise UTxONotInWallet
        sorted_utxo: UTxO = cast(
            UTxO, self._utxo_pool.pop((utxo.amount, utxo.wallet_id))
        )
        lookup_utxo: UTxO = cast(UTxO, self._lookup_pool.pop(utxo.wallet_id))

        if sorted_utxo is not lookup_utxo:
            raise InconsistentWallet

        self.balance -= sorted_utxo.amount

        return sorted_utxo

    def get(self, utxo_id: int) -> UTxO:
        utxo: UTxO = cast(UTxO, self._lookup_pool.get(utxo_id))
        return utxo

    def __len__(self):
        return len(self.utxo_pool)
