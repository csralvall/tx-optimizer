"""A module including a cryptographic-less model of a Bitcoin wallet.

It includes:
    - UTxONotInWallet (exception): raised when the UTxO doesn't belongs to the
        wallet.
    - InconsistentWallet (exception): raised when the indexes of the wallet
        differ.
    - Wallet (class): a storage for the simulation UTxOs.
"""

from __future__ import annotations

from collections.abc import Generator
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
    """A class to store and track a collection of UTxOs.

    This class lefts aside any cryptographic consideration about UTxO property.
    The wallet implemented here considers a UTxO as part of its property by
    just including it in one of their "pools". We don't handle private keys
    here.

    Attributes:
        balance: the total amount of bitcoin which the wallet can unlock with
            its keys.
    """

    balance: int = 0

    #: internal counter for the UTxOs that pass through the wallet
    _index: int = 0

    #: light dictionary to get UTxOs in the pool by id
    _lookup_pool: dict = field(default_factory=dict)

    #: the pool of UTxOs, indexed by the amount and id, to keep them sorted in descending order
    _utxo_pool: SortedDict = field(default_factory=SortedDict)

    def add(self, utxo: UTxO) -> None:
        """Add a new UTxO to the wallet.

        Args:
            utxo: the new UTxO to include in the wallet.
        """
        utxo.wallet_id = self._index
        self._utxo_pool[(utxo.amount, self._index)] = utxo
        self._lookup_pool[self._index] = utxo
        self._index += 1
        self.balance += utxo.amount

    def pop(self, utxo: UTxO) -> UTxO:
        """Remove the UTxO from the wallet.

        Args:
            utxo: the UTxO to remove from the wallet.

        Raises:
            UTxONotInWallet: raised when the UTxO to be removed doesn't belong
                to the wallet.
            InconsistentWallet: raised when for some reason the references from
                the lookup pool and the main pool differ.
        """
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
        """Retrieve a UTxO by its id.

        Args:
            utxo_id: the short identifier given to the UTxO upon creation.

        Returns:
            The UTxO associated with the utxo_id provided in the wallet.
        """
        utxo: UTxO = cast(UTxO, self._lookup_pool.get(utxo_id))
        return utxo

    def __len__(self) -> int:
        """Retrieve the amount of UTxOs currently stored in the wallet.

        Returns:
            The length of the UTxO pool.
        """
        return len(self._utxo_pool)

    def __iter__(self) -> Generator[UTxO, None, None]:
        """Traverse the wallet pool from greatest to smallest bitcoin amount.

        Yields:
            The UTxOs from the one with the greatest bitcoin amount to the one
            with the smallest bitcoin amount.
        """
        yield from reversed(self._utxo_pool.values())
