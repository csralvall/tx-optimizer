from decimal import ROUND_DOWN, Decimal

# sats = satoshis

#: the amount of satoshis in one bitcoin
COIN = 1e8

#: the fraction of satoshis in one bitcoin
SATOSHI = Decimal(1e-8)

#: factor to convert virtual bytes to virtual kilo bytes and vice versa
KILO = 1e3


def btc_round(amount: float) -> float:
    """Round down bitcoin quantities up to the eighth decimal.

    Args:
        amount: the original bitcoin amount to round down.

    Returns:
        The amount of bitcoin expressed as a float rounded down to the eighth
        decimal.
    """
    return float(
        Decimal(amount).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    )


def sat_round(amount: float) -> int:
    """Round down all decimals and transform the float amount into an integer.

    Args:
        amount: the original bitcoin amount to round down expressed in
            satoshis.

    Returns:
        The amount of satoshis without decimals.
    """
    return int(Decimal(amount).quantize(Decimal("1."), rounding=ROUND_DOWN))


def btc_to_sat(amount: float) -> int:
    """Conversor from bitcoin to satoshis.

    Args:
        amount: the bitcoin amount expressed in bitcoin units.

    Returns:
        The same amount of bitcoin expressed in satoshis.
    """
    return sat_round(btc_round(amount) * COIN)


def sat_to_btc(amount: int) -> float:
    """Conversor from satoshis to bitcoin.

    Args:
        amount: the bitcoin amount expressed in satoshis.

    Returns:
        The bitcoin amount expressed in bitcoin units.
    """
    return btc_round(sat_round(amount) / COIN)


def btc_to_str(amount: float) -> str:
    """A string representation of the bitcoin amount.

    Args:
        amount: the bitcoin amount expressed in bitcoin units.

    Returns:
        A string representing the bitcoin amount with eight decimal points.
    """
    return f"{btc_round(amount):.8f}"


def sat_kvB_to_sat_vB(amount: int) -> int:
    """Convert from sats per virtual kilo byte to sats per virtual byte.

    Args:
        amount: a sats per virtual kilo byte rate.

    Returns:
        The amount expressed as a sats per virtual byte rate.
    """
    return int(sat_round(amount / KILO))


def sat_vB_to_sat_kvB(amount: float) -> int:
    """Convert from sats per virtual byte to sats per virtual kilo byte.

    Args:
        amount: a sats per virtual byte rate.

    Returns:
        The amount expressed as a sats per virtual kilo byte rate.
    """
    return int(sat_round(amount * KILO))
