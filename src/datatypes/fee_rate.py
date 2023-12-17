"""This module contains constants and classes related to bitcoin fee rates.

Includes the following:
    - FeeRate (class): a class to represent Bitcoin fee rates.
    - DUST_RELAY_FEE_RATE: the minimum fee rate at which UTxOs dust condition
        must be evaluated .
    - CONSOLIDATION_FEE_RATE: a fee rate reference threshold to considerate
        UTxO consolidation or not.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from utils.bitcoin import sat_kvB_to_sat_vB

LOGGER = structlog.stdlib.get_logger(__name__)


@dataclass
class FeeRate:
    """Class to represent bitcoin fee rates and compute fees based on it.

    Attributes:
        sats_vkB: the fee rate expresed in sats per virtual kilo byte.
    """

    sats_vkB: int

    def __sub__(self, other: FeeRate) -> FeeRate:
        return FeeRate(self.sats_vkB - other.sats_vkB)

    def __repr__(self) -> str:
        return str(self.sats_vkB)

    def fee(self, weight: int) -> int:
        fee: int = sat_kvB_to_sat_vB(self.sats_vkB * weight)
        return fee


#: the fee rate at which check if a UTxO is dust
DUST_RELAY_FEE_RATE = FeeRate(3000)  # 3000 sats/kvB

#: fee rate threshold of reference to adopt a consolidation or conservative coin selection strategy
CONSOLIDATION_FEE_RATE = FeeRate(3000)  # 3000 sats/kvB
