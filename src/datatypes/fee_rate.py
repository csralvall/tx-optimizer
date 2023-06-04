from __future__ import annotations

from dataclasses import dataclass

import structlog

from utils import sat_kvB_to_sat_vB

LOGGER = structlog.stdlib.get_logger(__name__)


@dataclass
class FeeRate:
    sats_vkB: int

    def __sub__(self, other: FeeRate) -> FeeRate:
        return FeeRate(self.sats_vkB - other.sats_vkB)

    def __repr__(self) -> str:
        return str(self.sats_vkB)

    def fee(self, weight: int) -> int:
        fee: int = sat_kvB_to_sat_vB(self.sats_vkB * weight)
        return fee


DUST_RELAY_FEE_RATE = FeeRate(3000)  # 3000 sats/kvB
CONSOLIDATION_FEE_RATE = FeeRate(3000)  # 3000 sats/kvB
