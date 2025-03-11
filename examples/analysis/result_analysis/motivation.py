import marimo

__generated_with = "0.8.15"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    BASEDIR = "./examples/analysis/data"
    return BASEDIR, mo, np, pd, plt, sns


@app.cell
def __(BASEDIR, pd):
    df = pd.read_csv(f"{BASEDIR}/request_level/exec_time.csv")
    df = df[1:]
    df = df[df["prompt_len"] < 4096]
    df["prompt_len"] = df["prompt_len"] // 100 * 100
    df = df.groupby(["prompt_len"]).mean().reset_index()
    df["speed"] = df["prompt_len"] / df["exec_time"]
    return df,


@app.cell
def __(df, plt, sns):
    plt.figure(figsize=(4, 2.5), dpi=120)
    ax = sns.lineplot(data=df, x="prompt_len", y="exec_time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.xlim(0,256)
    # plt.ylim(0,0.1)
    plt.grid(alpha=0.5, linestyle="--")
    plt.xlabel("Prompt Length")
    plt.ylabel("Exec Time")
    plt.tight_layout()
    # plt.savefig(f"{BASEDIR}/fig/prompt_length_exec_time.pdf")
    plt.show()
    return ax,


@app.cell
def __(df):
    from scipy.stats import linregress

    tmp_df = df[df["prompt_len"] < 300]
    tmp_df["normalized_exec_time"] = (
        tmp_df["exec_time"] / tmp_df["exec_time"].min()
    )
    slope, _, _, _, _ = linregress(
        tmp_df["prompt_len"], tmp_df["normalized_exec_time"]
    )
    return linregress, slope, tmp_df


@app.cell
def __(df, plt, sns):
    h = 2
    df["derivative"] = (df["exec_time"].shift(-2) - df["exec_time"].shift(2)) / (
        2 * h
    )

    # 使用 Seaborn 绘制导数的变化趋势
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="prompt_len", y="derivative", data=df, marker="o")

    plt.title("Derivative of exec with respect to prompt_len")
    plt.xlabel("prompt_len")
    plt.ylabel("Derivative")
    plt.show()
    return h,


if __name__ == "__main__":
    app.run()
