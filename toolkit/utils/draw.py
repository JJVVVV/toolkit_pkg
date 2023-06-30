import matplotlib.pyplot as plt
import seaborn as sns

# 绘制注意力分数的热力图
plt.rc("font", family="HYZhengYuan")


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


