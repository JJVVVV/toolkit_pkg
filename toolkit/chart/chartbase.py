from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Data = Tuple[float]

DOT_STYLE = {
    "o": "大圆点",
    ".": "小圆点",
    ",": "像素点",
    "v": "向下三角形",
    "^": "向上三角形",
    "<": "向左三角形",
    ">": "向右三角形",
    "s": "正方形",
    "p": "五边形",
    "*": "星号",
    "h": "六边形1",
    "H": "六边形2",
    "+": "加号",
    "x": "叉号",
    "D": "菱形",
    "d": "小菱形",
    "1": "向下箭头",
    "2": "向上箭头",
    "3": "向左箭头",
    "4": "向右箭头",
}

LINE_STYLE = {"-": "实线", "--": "虚线", "-.": "点划线", ":": "点线"}

DOT_LINE_STYLE = [dot + line for dot in DOT_STYLE for line in LINE_STYLE]


class ChartBase:
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 100, is_ch=False) -> None:
        if is_ch:
            # sns.set(font_scale=0.9)
            plt.rc("font", family="HYZhengYuan")
        else:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)

        if nrows == 1 and ncols == 1:
            self.axes = np.array([[self.axes]])

    def __getattr__(self, name):
        return getattr(self.fig, name)

    def draw(self) -> None:
        raise NotImplementedError("`draw` function must be implimented in subclass.")

    def show(self):
        self.fig.show()

    def save(self, fig_name: str = "default_name", form: str = "eps", save_dir: Path | str = Path("figures"), bbox_inches="tight"):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        full_name = fig_name + "." + form
        self.fig.savefig(save_dir / full_name, format=form, bbox_inches=bbox_inches, dpi=self.dpi)


class LineChart(ChartBase):
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 100, is_ch=False, font_size: int = 8) -> None:
        super().__init__(nrows, ncols, figsize, dpi, is_ch)
        plt.rcParams["font.size"] = font_size
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.width"] = 0.5
        plt.rcParams["ytick.major.width"] = 0.5
        plt.rcParams["lines.linewidth"] = 1.25

    def draw(
        self,
        x: List[List[List[Data]]],
        y: List[List[List[Data]]],
        line_label: List[List[List[str]]],
        xlabel: str = "x",
        ylabel: str = "y",
        colors: List[str] | None = None,
        no_dot: bool = True,
    ) -> None:
        x = np.array(x)
        y = np.array(y)
        assert x.shape[0] == self.nrows and x.shape[1] == self.ncols
        # Use a pastel color scheme
        # line_label = (line_label,) if isinstance(line_label, str) else line_label
        lines_pre_ax = x.shape[2]
        colors = sns.color_palette("Spectral", lines_pre_ax)
        colors = sns.color_palette("husl", lines_pre_ax)
        colors = sns.color_palette("Greens", lines_pre_ax)
        # colors = [colors[1], colors2[1]]
        # colors = ["#37B971", "#ADD71B"]
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axes[i][j]
                for k in range(lines_pre_ax):
                    # if self.nrows == 1 and self.ncols == 1:
                    #     ax.plot(x[k], y[k], DOT_LINE_STYLE[k], color=colors[k], label=line_label[k])
                    # else:
                    ax.plot(
                        x[i][j][k],
                        y[i][j][k],
                        list(LINE_STYLE.keys())[k] if no_dot else DOT_LINE_STYLE[k],
                        color=colors[k],
                        label=line_label[i][j][k].item(),
                    )
                    # ax.axhline(y=81.875, linestyle="--", color="red", label="ChatGPT Zeroshot")

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", linestyle="-", alpha=0.8, linewidth=0.5)
                # ax.legend(loc="lower right", bbox_to_anchor=(1, 0.05))
                ax.legend(loc="lower right")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        # 调整子图间距和布局
        self.fig.subplots_adjust(wspace=0.3)


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
