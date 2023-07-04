from datatypes.fee_rate import CONSOLIDATION_FEE_RATE, FeeRate
from datatypes.transaction import TxDescriptor


def waste(tx: TxDescriptor, fee_rate: FeeRate) -> float:
    fee_delta: FeeRate = fee_rate - CONSOLIDATION_FEE_RATE
    timing_cost: int = sum((utxo.input_fee(fee_delta)) for utxo in tx.inputs)
    change_cost: int = sum(
        (utxo.output_fee(fee_rate) + utxo.input_fee(CONSOLIDATION_FEE_RATE))
        for utxo in tx.change
    )
    creation_cost: int = change_cost if tx.change else tx.excess
    return timing_cost + creation_cost
