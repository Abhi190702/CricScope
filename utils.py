import numpy as np

EPSILON = 1e-9
MAX_CRICKET_RATE = 36.0

def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """
    Safe division with floating-point tolerance
    and cricket-specific bounds checking.
    """
    if abs(denominator) < EPSILON:
        return default
    result = numerator / denominator
    if not np.isfinite(result):
        return default
    return float(np.clip(result, 0.0, MAX_CRICKET_RATE))