from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple = (8, 4.5), dpi: int = 100, is_ch=False, usetex=False) -> None:
        if is_ch:
            # sns.set(font_scale=0.9)
            plt.rc("font", family="HYZhengYuan")
        else:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        # plt.rcParams['text.usetex'] = usetex
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)

        if nrows == 1 and ncols == 1:
            self.axes = np.array([[self.axes]])
        elif nrows==1:
            self.axes = self.axes[None, :]

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
