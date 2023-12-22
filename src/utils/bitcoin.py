from decimal import ROUND_DOWN, Decimal

COIN = 1e8
SATOSHI = Decimal(1e-8)
KILO = 1e3


def btc_round(amount: float) -> float:
    return float(
        Decimal(amount).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    )


def sat_round(amount: float) -> int:
    return int(Decimal(amount).quantize(Decimal("1."), rounding=ROUND_DOWN))


def btc_to_sat(amount: float) -> int:
    return sat_round(btc_round(amount) * COIN)


def sat_to_btc(amount: int) -> float:
    return btc_round(sat_round(amount) / COIN)


def btc_to_str(amount: float) -> str:
    return f"{btc_round(amount):.8f}"


def sat_kvB_to_sat_vB(amount: int) -> int:
    return int(sat_round(amount / KILO))


def sat_vB_to_sat_kvB(amount: float) -> int:
    return int(sat_round(amount * KILO))
