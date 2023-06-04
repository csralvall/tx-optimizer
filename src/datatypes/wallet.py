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
    index: int = 0
    utxo_pool: SortedDict = field(default_factory=SortedDict)
    lookup_pool: SortedDict = field(default_factory=SortedDict)

    def add(self, utxo: UTxO) -> None:
        utxo.wallet_id = self.index
        self.utxo_pool[(utxo.amount, self.index)] = utxo
        self.lookup_pool[self.index] = utxo
        self.index += 1

    def pop(self, utxo: UTxO) -> UTxO:
        if utxo.wallet_id < 0:
            raise UTxONotInWallet
        sorted_utxo: UTxO = cast(
            UTxO, self.utxo_pool.pop((utxo.amount, utxo.wallet_id))
        )
        lookup_utxo: UTxO = cast(UTxO, self.lookup_pool.pop(utxo.wallet_id))

        if sorted_utxo is not lookup_utxo:
            raise InconsistentWallet

        return sorted_utxo

    def get(self, utxo_id: int) -> UTxO:
        utxo: UTxO = cast(UTxO, self.lookup_pool.get(utxo_id))
        return utxo

    def __len__(self):
        return len(self.utxo_pool)
