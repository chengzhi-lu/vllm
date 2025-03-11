import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    import numpy as np
    return np, pd, plt, sns


@app.cell
def __():
    base_dir = "/root/vllm/examples/analysis/data"
    return base_dir,


@app.cell
def __(base_dir, pd):
    data = pd.read_csv(
        f"{base_dir}/request_level/request_result_new_unpredict_seq_length_policy.csv"
    )
    return data,


@app.cell
def __(data):
    data["total_length"] = data["prompt_length"] + data["decode_length"]
    data[data["decode_length"] > 1].sort_values(by="total_length", ascending=False)
    return


@app.cell
def __(data, np, pd):
    def get_cov(data):
        if (
            len(data) < 5
            or max(data["decode_length"]) < 3
            or max(data["total_length"]) >= 4096
        ):
            return (-1, -1)
        else:
            return (
                np.std(data["decode_length"]) / np.mean(data["decode_length"]),
                (max(data["decode_length"]) - min(data["decode_length"]))
                / min(data["decode_length"])
                / 1000,
            )


    result = (
        data[["prompt_length", "decode_length", "total_length"]]
        .groupby(["prompt_length"])
        .apply(lambda x: get_cov(x))
        .reset_index()
    )
    result[["cov", "relative_diff"]] = result[0].apply(pd.Series)
    result.drop(columns=0, inplace=True)

    result_for_line_plot = data[
        (data["prompt_length"] > 3) & (data["total_length"] < 4096)
    ][["prompt_length", "decode_length"]]
    result_for_line_plot["total_length"] = (
        result_for_line_plot["prompt_length"]
        + result_for_line_plot["decode_length"]
    ) / 1000
    return get_cov, result, result_for_line_plot


@app.cell
def __(base_dir, plt, result, result_for_line_plot, sns):
    plt.figure(figsize=(8, 2.5), dpi=120)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 14
    color = sns.color_palette("deep").as_hex()[:3]
    ax = plt.subplot(121)
    sns.ecdfplot(data=result[result["cov"] != -1], y="cov", ax=ax, color=color[0])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(color[0])
    ax.set_ylabel("CoV", color=color[0])
    ax.tick_params(axis="y", colors=color[0])
    over_one_portion = result[
        (result["cov"] != -1) & (result["cov"] < 1)
    ].count() / len(result[result["cov"] != -1])
    # ax.axvline(x=over_one_portion[0],ymin=0,ymax=0.5)
    ax.scatter(x=over_one_portion[0], y=1, color=color[2])
    ax.annotate(
        xy=(0.18, 1.2),
        text=f"({round(over_one_portion[0],2)},1)",
        fontsize=12,
        color=color[2],
    )
    # ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Proportion")
    ax1 = ax.twinx()
    sns.ecdfplot(
        data=result[result["relative_diff"] != -1],
        y="relative_diff",
        ax=ax1,
        color=color[1],
        label="CoV",
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_color(color[1])
    ax1.spines["left"].set_color(color[0])
    ax1.tick_params(axis="y", colors=color[1])
    # ax1.spines["right"].set_visible(False)
    ax1.set_ylabel(r"RR ($\times10^3$)", color=color[1])
    # ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlabel("(a) Proportion")
    ax2 = plt.subplot(122)
    sns.lineplot(
        data=result_for_line_plot,
        x="prompt_length",
        y="total_length",
        ax=ax2,
        color="gray",
    )
    ax2.set_xlabel("Prompt.Length")
    ax2.set_ylabel(r"Seq.Len ($\times10^3$)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.45)
    plt.savefig(f"{base_dir}/fig/seq_unpredictable.pdf")
    plt.show()
    return ax, ax1, ax2, color, over_one_portion


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
