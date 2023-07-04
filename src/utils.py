from decimal import ROUND_UP, Decimal

COIN = 100000000
SATOSHI = Decimal(0.00000001)
KILO = 1000


def btc_round(amount: float) -> Decimal:
    return Decimal(amount).quantize(Decimal("0.00000001"), rounding=ROUND_UP)


def sat_round(amount: float) -> Decimal:
    return Decimal(amount).quantize(Decimal("1."), rounding=ROUND_UP)


def btc_to_sat(amount: float) -> int:
    sats: Decimal = (btc_round(amount) * COIN).quantize(
        Decimal("1."), rounding=ROUND_UP
    )
    return int(sats)


def sat_kvB_to_sat_vB(amount: int) -> int:
    return int(sat_round(amount / KILO))


def sat_vB_to_sat_kvB(amount: float) -> int:
    return int(sat_round(amount * KILO))
