from enum import Enum

import numpy as np


class KernelEnum(Enum):
    TRIANGULAR = "triangular"
    RECTANGULAR = "rectangular"
    GAUSSIAN = "gaussian"


def compute_kernel_weights(distances: np.ndarray, kernel: KernelEnum, bandwidth: float = None) -> np.ndarray:
    if bandwidth is None:
        bandwidth = np.max(distances)
    u = distances / bandwidth
    match kernel:
        case KernelEnum.TRIANGULAR:
            return np.clip(1 - u, 0, 1)
        case KernelEnum.RECTANGULAR:
            return (u <= 1).astype(float)
        case KernelEnum.GAUSSIAN:
            return np.exp(-0.5 * (u ** 2))
        case _:
            raise ValueError(f"Unsupported kernel: {kernel}")
