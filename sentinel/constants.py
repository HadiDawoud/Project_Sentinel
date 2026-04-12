from enum import IntEnum
from typing import Dict, Tuple


class ClassificationLabel(IntEnum):
    NON_RADICAL = 0
    MILDLY_RADICAL = 1
    MODERATELY_RADICAL = 2
    HIGHLY_RADICAL = 3


LABEL_MAP: Dict[int, str] = {
    0: "Non-Radical",
    1: "Mildly Radical",
    2: "Moderately Radical",
    3: "Highly Radical"
}

LABEL_TUPLES: Tuple[str, ...] = (
    "Non-Radical",
    "Mildly Radical",
    "Moderately Radical",
    "Highly Radical"
)

RISK_WEIGHTS: Dict[str, float] = {
    "Non-Radical": 0.0,
    "Mildly Radical": 0.33,
    "Moderately Radical": 0.66,
    "Highly Radical": 1.0
}

RISK_THRESHOLDS: Dict[str, int] = {
    "Non-Radical": 25,
    "Mildly Radical": 50,
    "Moderately Radical": 75,
    "Highly Radical": 100
}

MAX_INPUT_LENGTH = 10000
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
