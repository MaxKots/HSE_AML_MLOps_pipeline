import random

import numpy as np

from config.settings import settings


def set_random_seed(seed: int | None = None) -> int:
    actual_seed = seed if seed is not None else settings.random_seed

    random.seed(actual_seed)
    np.random.seed(actual_seed)

    return actual_seed
