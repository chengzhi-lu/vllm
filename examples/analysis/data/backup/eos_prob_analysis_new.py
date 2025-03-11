import marimo

__generated_with = "0.9.10"
app = marimo.App(width="full")


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from pandarallel import pandarallel
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    from matplotlib.ticker import MultipleLocator
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    pandarallel.initialize(progress_bar=False)
    return MultipleLocator, mo, mpl, np, pandarallel, pd, plt, sns


@app.cell
def __(pd):
    model_names = ["llama", "mistral"]
    dataset_names = ["alpaca", "sharegpt"]
    dataset_name_map = {"alpaca": "Alpaca", "sharegpt": "ShareGPT"}
    model_name_map = {"llama": "Llama", "mistral": "Mistral"}

    eos_prob_rank_result_df = pd.DataFrame()
    for model_name in model_names:
        for dataset_name in dataset_names:
            tmp_df = pd.read_csv(
                f"/root/vllm/examples/analysis/data/eos_result/{model_name}_{dataset_name}_eos_prob_result.csv"
            )
            tmp_df["model_dataset"] = (
                model_name_map[model_name] + " " + dataset_name_map[dataset_name]
            )
            eos_prob_rank_result_df = pd.concat([eos_prob_rank_result_df, tmp_df])
    eos_prob_rank_result_df = eos_prob_rank_result_df[
        eos_prob_rank_result_df["eos_prob"] != 0
    ]
    return (
        dataset_name,
        dataset_name_map,
        dataset_names,
        eos_prob_rank_result_df,
        model_name,
        model_name_map,
        model_names,
        tmp_df,
    )


@app.cell
def __(eos_prob_rank_result_df):
    print(min(eos_prob_rank_result_df["eos_token_rank"]))
    return


@app.cell
def __(eos_prob_rank_result_df, np):
    from scipy.stats import spearmanr, pearsonr


    def calc_eos_prob_output_len_corr(row):
        if min(row["eos_token_rank"]) != 1:
            return -1

        token_nums = max(row["token_num"]) - row["token_num"]
        token_nums = np.array(token_nums.tolist())
        eos_token_ranks = row["eos_token_rank"].tolist()
        max_eos_token_rank = np.array(
            [min(eos_token_ranks[:i]) for i in range(1, len(eos_token_ranks) + 1)]
        )

        return spearmanr(token_nums, max_eos_token_rank).statistic


    eos_prob_seq_len_corrs = (
        eos_prob_rank_result_df.groupby(["request_id", "model_dataset"])
        .parallel_apply(lambda row: calc_eos_prob_output_len_corr(row))
        .reset_index()
    )
    eos_prob_seq_len_corrs["corr"] = eos_prob_seq_len_corrs[0]
    return (
        calc_eos_prob_output_len_corr,
        eos_prob_seq_len_corrs,
        pearsonr,
        spearmanr,
    )


@app.cell
def __(MultipleLocator, eos_prob_seq_len_corrs, plt, sns):
    plt.figure(figsize=(6, 1.8), dpi=120)
    ax = sns.ecdfplot(
        eos_prob_seq_len_corrs[eos_prob_seq_len_corrs["corr"] > -1],
        x="corr",
        hue="model_dataset",
    )
    ax_legend = ax.get_legend()
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Corr", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    ax_legend.set(ncols=4, frame_on=False, title="")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout(h_pad=0, pad=0.1)
    # plt.savefig("/root/vllm/examples/analysis/data/fig/seq_len_eos_rank_corr.pdf")
    plt.show()
    return ax, ax_legend


@app.cell(disabled=True)
def __(mo):
    mo.md("""## test warm up window size""")
    return


@app.cell(disabled=True, hide_code=True)
def __(eos_prob_rank_result_df, np, pd):
    def max_eos_prob_left_seq_len(row, i):
        if len(row) <= i:
            return pd.Series([-1, -1], index=["max_eos_prob", "left_seq_len"])
        # max_eos_prob = np.std(row["eos_prob"][:i]) / np.mean(
        #     row["eos_prob"][:i]
        # )
        max_eos_prob = np.max(row["eos_token_rank"][:i])
        left_seq_len = max(row["token_num"]) - min(row["token_num"][i:])
        tmp_df = pd.Series(
            [max_eos_prob, left_seq_len], index=["max_eos_prob", "left_seq_len"]
        )


    corrs = []
    for i in range(1, 50):
        eos_prob_left_seq_len = (
            eos_prob_rank_result_df.groupby(["request_id"])
            .parallel_apply(lambda row: max_eos_prob_left_seq_len(row, i))
            .reset_index()
            .drop(columns=["request_id"])
        )
        corrs.append(
            eos_prob_left_seq_len[eos_prob_left_seq_len["max_eos_prob"] >= 0]
            .corr(method="spearman")
            .loc["max_eos_prob", "left_seq_len"]
        )
    return corrs, eos_prob_left_seq_len, i, max_eos_prob_left_seq_len


@app.cell(hide_code=True)
def __(corrs, plt, sns):
    plt.figure(figsize=(4, 2.5), dpi=150)
    print(corrs)
    sns.lineplot(corrs, hue="model_dataset")
    plt.xlabel("# of Iters")
    plt.ylabel("Corr. Seq Len vs. Prob")
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()
    print(corrs)
    return


@app.cell(disabled=True, hide_code=True)
def __(eos_prob_rank_result_df, np, pd):
    def eos_prob_predict_len(row, i):
        if len(row) <= i:
            return pd.Series([-1, -1], index=["pred_len", "left_seq_len"])
        # max_eos_prob = np.std(row["eos_token_rank"][:i]) / np.mean(
        #     row["eos_prob"][:i]
        # )

        max_eos_prob = 1 - np.max(row["eos_token_rank"][:i]) / 32000
        _n = int(-np.log(np.max(row["eos_token_rank"][:i]))) + 23
        predict_len = (
            max_eos_prob
            * (1 + _n * max_eos_prob ** (_n + 1) - (_n + 1) * max_eos_prob**_n)
            / ((1 - max_eos_prob) ** 2)
        )

        left_seq_len = max(row["token_num"]) - min(row["token_num"][i:])
        return pd.Series(
            [predict_len, left_seq_len], index=["pred_len", "left_seq_len"]
        )


    eos_pred_len_df = (
        eos_prob_rank_result_df.groupby(["request_id"])
        .apply(
            lambda row: eos_prob_predict_len(row, 3),
            include_groups=False,
        )
        .reset_index()
        .drop(columns=["request_id"])
    )
    eos_pred_len_df
    return eos_pred_len_df, eos_prob_predict_len


@app.cell(disabled=True, hide_code=True)
def __(eos_prob_rank_result_df):
    _tmp_eos_prob_rank_result_df = eos_prob_rank_result_df[
        eos_prob_rank_result_df["request_id"] == 0
    ]
    _tmp_eos_prob_rank_result_df["left_seq_len"] = (
        max(_tmp_eos_prob_rank_result_df["token_num"])
        - _tmp_eos_prob_rank_result_df["token_num"]
    )
    _tmp_eos_prob_rank_result_df
    return


@app.cell(disabled=True, hide_code=True)
def __(eos_prob_rank_result_df):
    seq_output_len = (
        eos_prob_rank_result_df.groupby(["prompt_len"])
        .agg({"token_num": "mean"})
        .reset_index()
    )
    return (seq_output_len,)


if __name__ == "__main__":
    app.run()
