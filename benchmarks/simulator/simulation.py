import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Request:
    id: int
    arrival_time: float
    