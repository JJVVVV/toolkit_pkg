from typing import Tuple

import seaborn as sns

from .chartbase import ChartBase


class Histogram(ChartBase):
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 100, is_ch=False) -> None:
        super().__init__(nrows, ncols, figsize, dpi, is_ch)

    def draw(self, data, bins, title="", xlabel="", ylabel="Count", **kwargs) -> None:
        colors = sns.color_palette("Greens")
        self.axes[0][0].hist(data, bins=bins, edgecolor="black", color=colors[1], **kwargs)
        for _, ax in enumerate(self.axes.flat):
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="y", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        self.fig.tight_layout()
