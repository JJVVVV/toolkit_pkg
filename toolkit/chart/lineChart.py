from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .chartbase import DOT_LINE_STYLE, LINE_STYLE, ChartBase

Data = Tuple[float]


class LineChart(ChartBase):
    def __init__(
        self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 100, is_ch=False, usetex=False, font_size: int = 8
    ) -> None:
        super().__init__(nrows, ncols, figsize, dpi, is_ch, usetex)
        plt.rcParams["font.size"] = font_size
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.width"] = 0.5
        plt.rcParams["ytick.major.width"] = 0.5
        plt.rcParams["lines.linewidth"] = 1.25
        plt.rcParams["xtick.labelsize"] = font_size
        plt.rcParams["ytick.labelsize"] = font_size
        self.fontsize = font_size

    def draw(
        self,
        x: List[Data] | List[List[List[Data]]],
        y: List[Data] | List[List[List[Data]]],
        line_label: List[str] | List[List[List[str]]],
        xlabel: str = "x",
        ylabel: str = "y",
        colors: List[str] | None = None,
        dot: bool = False,
        color_palette: Literal["husl", "Greens", "Spectral"] = "husl",
        bottom: int = None,
        top: int = None,
    ) -> None:
        x = np.array(x)
        y = np.array(y)
        line_label = np.array(line_label)
        # only one plot
        if self.nrows == self.ncols == 1 and len(x.shape) == 2:
            x = x[None, None, :, :]
            y = y[None, None, :, :]
            line_label = line_label[None, None, :]
        if len(x.shape) == 3:
            x = x[:, :, None, :]
            y = y[:, :, None, :]
            if len(line_label.shape) == 2:
                line_label = line_label[:, :, None]

        # print(x)
        # print(y)
        # print(line_label)
        assert x.shape[0] == self.nrows and x.shape[1] == self.ncols
        assert all(x[i][j].shape == y[i][j].shape for i in range(self.nrows) for j in range(self.ncols))
        lines_pre_ax = x.shape[2]
        # print(lines_pre_ax)
        # Use a pastel color scheme
        if colors is None:
            colors = sns.color_palette(color_palette, lines_pre_ax)
        # colors = [colors[1], colors2[1]]
        # colors = ["#37B971", "#ADD71B"]
        for i in range(self.nrows):
            for j in range(self.ncols):
                # print(i, j)
                ax = self.axes[i][j]
                for k in range(lines_pre_ax):
                    # if self.nrows == 1 and self.ncols == 1:
                    #     ax.plot(x[k], y[k], DOT_LINE_STYLE[k], color=colors[k], label=line_label[k])
                    # else:
                    # print(i, j, k)
                    if len(x[i][j][k]) == 1 and x[i][j][k].item() is None:
                        continue
                    ax.plot(
                        x[i][j][k],
                        y[i][j][k],
                        list(LINE_STYLE.keys())[k] if not dot else DOT_LINE_STYLE[k],
                        color=colors[k],
                        label=line_label[i][j][k].item(),
                    )
                    # ax.axhline(y=81.875, linestyle="--", color="red", label="ChatGPT Zeroshot")

                ax.set_xlabel(xlabel, fontsize=self.fontsize)
                ax.set_ylabel(ylabel, fontsize=self.fontsize)
                ax.grid(axis="y", linestyle="-", alpha=0.8, linewidth=0.5)
                # ax.legend(loc="lower right", bbox_to_anchor=(1, 0.05))
                ax.legend(loc="best")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if bottom is not None:
                    ax.set_ylim(bottom=bottom)
                if top is not None:
                    ax.set_ylim(top=top)
        # 调整子图间距和布局
        self.fig.subplots_adjust(wspace=0.3)
