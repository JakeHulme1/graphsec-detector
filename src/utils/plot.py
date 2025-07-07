import os
from typing import Dict, List
import matplotlib.pyplot as plt

def plot_training(histories: dict[str, list[float]], out_path: str) -> None:
    """
    Plots training curves and writes to out_path
    """
    assert isinstance(histories, dict), "histories must be a dict"
    for name, series in histories.items():
        assert isinstance(series, list), f"{name} must be a list"
        assert all(isinstance(x, (int, float)) for x in series), f"all values in {name} must be numeric"

    plt.figure()
    for name, series in histories.items():
        plt.plot(range(1, len(series)+1), series, label=name)