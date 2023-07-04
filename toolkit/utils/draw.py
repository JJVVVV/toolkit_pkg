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
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 300, is_ch=False) -> None:
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
            self.axes = np.array[[self.axes]]

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
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 300, is_ch=False, font_size: int = 8) -> None:
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
        xlabel: str,
        ylabel: str,
        line_label: List[List[List[str]]],
        colors: List[str] | None = None,
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
                    if self.nrows == 1 and self.ncols == 1:
                        ax.plot(x[k], y[k], DOT_LINE_STYLE[k], color=colors[k], label=line_label[k])
                    else:
                        ax.plot(x[i][j][k], y[i][j][k], DOT_LINE_STYLE[k], color=colors[k], label=line_label[i][j][k])
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


def attention_plot(
    attention,
    attention_o=None,
    x_texts="auto",
    y_texts="auto",
    colors="Greys",
    figsize=(7, 3),
    annot=False,
    figure_path="./figures",
    figure_name="attention_weight.png",
    rotation=0,
    **kargs,
):
    # 绘制注意力分数的热力图
    plt.rc("font", family="HYZhengYuan")
    # sns.set(font_scale=0.8)
    fig, axes = plt.subplots(nrows=2, figsize=figsize, dpi=200, gridspec_kw={"height_ratios": [1, 1]})
    plt.subplots_adjust(wspace=0.3)
    ax1 = sns.heatmap(
        attention,
        cbar=True,
        cmap=colors,
        annot=annot,
        square=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        yticklabels=y_texts,
        xticklabels=x_texts,
        ax=axes[0],
        **kargs,
    )
    ax1.set_title("Weights of normal attention", fontsize=10)
    # ax1.tick_params(axis='y', labelrotation=45)
    if attention_o is not None:
        ax2 = sns.heatmap(
            attention_o,
            cbar=True,
            cmap=colors,
            annot=annot,
            square=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            yticklabels=y_texts,
            xticklabels=x_texts,
            ax=axes[1],
            **kargs,
        )
        ax2.set_title("Weights of reversed attention", fontsize=10)

    # for ax in axes.flat:
    #     ax.set_xlabel("Input Sequence")
    #     ax.set_ylabel("Output Sequence")
    # 旋转刻度值
    for ax in axes.flat:
        # ax.tick_params(axis="x", labeltop=True, labelbottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # plt.show()
    return fig


def attention_plot_list(
    attention_list: list,
    attention_o_list: list = None,
    x_texts_list="auto",
    y_texts_list="auto",
    colors="Greys",
    figsize=(9, 4),
    annot=False,
    figure_path="./figures",
    figure_name="attention_weight.png",
    rotation=0,
    **kargs,
):
    # sns.set(font_family='HYZhengYuan', font_scale=0.8)
    sns.set(font_scale=0.9)
    plt.rc("font", family="HYZhengYuan")
    n = len(attention_list)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=figsize, dpi=300, gridspec_kw={"height_ratios": [1, 1]})
    # plt.subplots_adjust(left=0, right=0, bottom=0, top=0, wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=-0.85)
    # plt.subplots_adjust(wspace=0.3)
    for i in range(n):
        ax1 = sns.heatmap(
            attention_list[i],
            # cbar=True,
            cmap=colors,
            annot=annot,
            square=True,
            fmt=".2f",
            annot_kws={"fontsize": 5},
            yticklabels=y_texts_list[i],
            xticklabels=x_texts_list[i],
            # ax=axes[i, 0],
            ax=axes[0, i],
            **kargs,
        )
        ax1.set_title("Weights of normal attention", fontsize=8)
        if attention_o_list is not None:
            ax2 = sns.heatmap(
                attention_o_list[i],
                # cbar=True,
                cmap=colors,
                annot=annot,
                square=True,
                fmt=".2f",
                annot_kws={"fontsize": 5},
                yticklabels=y_texts_list[i],
                xticklabels=x_texts_list[i],
                # ax=axes[i, 1],
                ax=axes[1, i],
                **kargs,
            )
            ax2.set_title("Weights of reversed attention", fontsize=8)
    # 旋转刻度值
    for ax in axes.flat:
        # ax.tick_params(axis="x", labeltop=True, labelbottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(axis="both", which="both", pad=-4)
    fig.tight_layout()
    return fig, axes
