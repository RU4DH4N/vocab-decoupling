import math


def require_positive(**values: int | float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive, {name}={value}")
