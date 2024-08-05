import marimo

__generated_with = "0.7.12"
app = marimo.App(width="full")


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    return mo, np, pd, plt, sns


@app.cell
def __(pd):
    eos_prob_rank_result_df = pd.read_csv(
        "/root/vllm/examples/analysis/data/eos_result/eos_prob_result_2.csv"
    )
    eos_prob_rank_result_df = eos_prob_rank_result_df[
        eos_prob_rank_result_df["eos_prob"] != 0
    ]
    return eos_prob_rank_result_df,


@app.cell
def __(eos_prob_rank_result_df):
    eos_prob_rank_result_df.columns
    return


@app.cell
def __(eos_prob_rank_result_df, sns):
    from scipy.stats import spearmanr, pearsonr


    def calc_eos_prob_output_len_corr(row):
        return spearmanr(
            max(row["token_num"]) - row["token_num"], row["eos_token_rank"]
        )[0]


    eos_prob_seq_len_corrs = (
        eos_prob_rank_result_df.groupby(["request_id"])
        .apply(
            lambda row: calc_eos_prob_output_len_corr(row), include_groups=False
        )
        .reset_index()
    )
    sns.ecdfplot(eos_prob_seq_len_corrs[0])
    return (
        calc_eos_prob_output_len_corr,
        eos_prob_seq_len_corrs,
        pearsonr,
        spearmanr,
    )


@app.cell
def __(eos_prob_rank_result_df, np, pd, plt, sns):
    def max_eos_prob_left_seq_len(row, i):
        if len(row) <= i:
            return pd.Series([-1, -1], index=["max_eos_prob", "left_seq_len"])
        # max_eos_prob = np.std(row["eos_prob"][:i]) / np.mean(
        #     row["eos_prob"][:i]
        # )
        max_eos_prob = np.max(row["eos_token_rank"][:i])
        left_seq_len = max(row["token_num"]) - min(row["token_num"][i:])
        return pd.Series(
            [max_eos_prob, left_seq_len], index=["max_eos_prob", "left_seq_len"]
        )


    corrs = []
    for i in range(1, 50):
        eos_prob_left_seq_len = (
            eos_prob_rank_result_df.groupby(["request_id"])
            .apply(
                lambda row: max_eos_prob_left_seq_len(row, i), include_groups=False
            )
            .reset_index()
            .drop(columns=["request_id"])
        )
        corrs.append(
            eos_prob_left_seq_len[eos_prob_left_seq_len["max_eos_prob"] >= 0]
            .corr(method="spearman")
            .loc["max_eos_prob", "left_seq_len"]
        )
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.lineplot(corrs)
    plt.xlabel("# of Iters")
    plt.ylabel("Corr. Seq Len vs. Prob")
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()
    print(corrs)
    return corrs, eos_prob_left_seq_len, i, max_eos_prob_left_seq_len


@app.cell
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


@app.cell
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


@app.cell
def __(eos_prob_rank_result_df):
    seq_output_len = (
        eos_prob_rank_result_df.groupby(["prompt_len"])
        .agg({"token_num": "mean"})
        .reset_index()
    )
    return seq_output_len,


@app.cell
def __(seq_output_len):
    seq_output_len
    return


@app.cell
def __(seq_output_len):
    seq_output_len.corr()
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
