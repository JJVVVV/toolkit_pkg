from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .chartbase import DOT_LINE_STYLE, LINE_STYLE, ChartBase

Data = Tuple[float]


class BarChart(ChartBase):
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 100, is_ch=False, font_size: int = 8) -> None:
        super().__init__(nrows, ncols, figsize, dpi, is_ch)
        plt.rcParams["font.size"] = font_size
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.width"] = 0.5
        plt.rcParams["ytick.major.width"] = 0.5
        plt.rcParams["lines.linewidth"] = 1.25
        self.fontsize = font_size
        plt.rcParams["xtick.labelsize"] = font_size
        plt.rcParams["ytick.labelsize"] = font_size

    def draw(
        self,
        groups: List[Data] | List[List[List[Data]]],
        groups_name: List[str] | List[List[List[str]]],
        col_names: List[str] | List[List[List[str]]],
        xlabel: str = "x",
        ylabel: str = "y",
        colors: List[str] | None = None,
        color_palette: Literal["husl", "Greens", "Spectral"] = "husl",
    ) -> None:
        """
        groups: shape=(m, n)
        m行表示有m个group, 即`len(groups_name)=m`, 同时也代表每个col有m个bar.
        n列表示有n个col, 即`len(col_names)=n`.
        """
        groups = np.array(groups)
        col_names = np.array(col_names)
        groups_name = np.array(groups_name)
        # only one plot
        if self.nrows == self.ncols == 1 and len(groups.shape) == 2:
            groups = groups[None, None, :, :]
            col_names = col_names[None, None, :]
            groups_name = groups_name[None, None, :]
        num_bar_per_col = [[groups_subplot.shape[0] for groups_subplot in row] for row in groups]
        num_col = [[groups_subplot.shape[1] for groups_subplot in row] for row in groups]
        col_names = np.array(col_names)

        assert groups.shape[0] == self.nrows and groups.shape[1] == self.ncols
        # Use a pastel color scheme
        if colors is None:
            colors = sns.color_palette(color_palette, num_bar_per_col[0][0])
        # colors = [colors[1], colors2[1]]
        # colors = ["#37B971", "#ADD71B"]
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axes[i][j]
                ax_num_bar_per_col = num_bar_per_col[i][j]
                ax_num_col = num_col[i][j]
                col_width = 0.7
                single_width = col_width / ax_num_bar_per_col
                bar_width = single_width * 0.9
                x = np.arange(ax_num_col)
                for k in range(ax_num_bar_per_col):
                    ax.bar(
                        x - col_width / 2 + (k + 0.5) * single_width, groups[i][j][k], width=bar_width, label=groups_name[i][j][k], color=colors[k]
                    )

                ax.set_xlabel(xlabel, fontsize=self.fontsize)
                ax.set_ylabel(ylabel, fontsize=self.fontsize)
                ax.grid(axis="y", linestyle="-", alpha=0.8, linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(col_names[i][j], fontsize=self.fontsize)
                # ax.legend(loc="lower right", bbox_to_anchor=(1, 0.05))
                ax.legend(loc="best")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        # 调整子图间距和布局
        self.fig.subplots_adjust(wspace=0.3)
