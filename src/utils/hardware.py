from typing import Literal

import psutil


def mhz_to_ghz(mhz: float) -> float:
    ghz: float = mhz / 1000
    return ghz


def scale_bytes(
    bytes_: float, unit: Literal["bytes", "Kb", "Mb", "Gb"]
) -> str:
    scaling_table = {"bytes": 0, "Kb": 1, "Mb": 2, "Gb": 3}
    if unit not in scaling_table:
        raise ValueError("Must select from ['bytes', 'Kb', 'Mb', 'Gb']")

    scaled_bytes = bytes_ / 1024 ** scaling_table[unit]
    return f"{scaled_bytes:.3f} {unit}"


def get_hardware_spec() -> dict:
    spec: dict = {}
    spec["physical_cores"] = psutil.cpu_count(logical=False)
    spec["logical_cores"] = psutil.cpu_count(logical=True)
    cpufreq: tuple = psutil.cpu_freq()
    spec["max_cpu_freq"] = f"{mhz_to_ghz(cpufreq.max):.2f}Ghz"
    sys_virtual_mem = psutil.virtual_memory()
    spec["RAM"] = {
        "total": scale_bytes(sys_virtual_mem.total, unit="Gb"),
        "available_at_execution": scale_bytes(
            sys_virtual_mem.available, unit="Gb"
        ),
    }
    return spec
