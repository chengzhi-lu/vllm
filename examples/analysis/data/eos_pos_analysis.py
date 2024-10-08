import marimo

__generated_with = "0.7.12"
app = marimo.App(width="full")


@app.cell
def __():
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    return json, np, pd, plt, sns


@app.cell
def __(json, pd):
    with open(
        "/root/vllm/examples/analysis/data/eos_result/tmp_result_13b.json", "r"
    ) as f:
        result = json.load(f)
    eos_postions = result["eos_poss"]
    eos_probabilities = result["eos_probabilities"]
    df = pd.DataFrame(eos_postions).T
    df_prob = pd.DataFrame(eos_probabilities).T
    return df, df_prob, eos_postions, eos_probabilities, f, result


@app.cell
def __(df):
    df
    return


@app.cell
def __(df, df_prob, plt, sns):
    _corrs = []
    ratio = []
    _count = df.describe().T["count"]
    for _i in range(1, 30):
        _df_eos_prob_count = df_prob[:_i].describe().T[["count", "mean"]]
        _df_eos_prob_count["count"] = _count - _i
        _df_eos_prob_count = _df_eos_prob_count[_df_eos_prob_count["count"] > 0]
        ratio.append(_df_eos_prob_count["count"] / _df_eos_prob_count["mean"])
        _corrs.append(
            _df_eos_prob_count.corr(method="pearson").loc["mean", "count"]
        )
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.lineplot(_corrs)
    plt.xlabel("# of Iters")
    plt.ylabel("Corr. Seq Len vs. Prob")
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()
    return ratio,


@app.cell
def __(df_prob, np, pd, plt, sns):
    _output_count = df_prob.describe().T[["count"]]
    _prob_mean = df_prob[:15].describe().T[["max"]]
    _ns = (-np.log(_prob_mean["max"]) + 28).astype(int)
    values = 1 - _prob_mean
    predicted_seq_lens = []
    for _i in range(len(_prob_mean)):
        value = values.values[_i]
        _n = _ns.values[_i]
        predicted_seq_lens.append(
            value
            * (
                (1 + _n * value ** (_n + 1) - (_n + 1) * value**_n)
                / (1 - value) ** 2
            )
        )
    predicted_seq_len = (
        pd.DataFrame(predicted_seq_lens, columns=["max"]).astype(int) + 1
    )
    real_seq_len = pd.DataFrame(_output_count, columns=["count"])
    predicted_seq_len = pd.concat([predicted_seq_len, real_seq_len], axis=1)
    compared_result = predicted_seq_len["count"] - predicted_seq_len["max"]
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.ecdfplot(compared_result)
    plt.grid(True, alpha=0.5, linestyle="--")
    plt.show()
    return (
        compared_result,
        predicted_seq_len,
        predicted_seq_lens,
        real_seq_len,
        value,
        values,
    )


@app.cell
def __(predicted_seq_len):
    predicted_seq_len
    return


@app.cell
def __(predicted_seq_len):
    predicted_seq_len[predicted_seq_len["max"] == predicted_seq_len["count"]]
    return


@app.cell
def __(df_prob, pd, plt, sns):
    output_count = df_prob.describe().T[["count"]]
    prob_mean = df_prob[:15].describe().T[["max"]]
    prob_std = df_prob[:15].describe().T[["std"]]
    output_count = output_count // 10 * 10
    count_prob_mean = pd.DataFrame(
        {
            "mean": prob_mean.values.flatten(),
            "count": output_count.values.flatten(),
            "std": prob_std.values.flatten(),
        }
    )
    count_prob_mean["cov"] = count_prob_mean["mean"]
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.scatterplot(data=count_prob_mean, x="mean", y="count")
    plt.xlim(0, 0.1)
    plt.xlabel("Avg. Prob")
    plt.ylabel("Sequence Length")
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()
    return count_prob_mean, output_count, prob_mean, prob_std


@app.cell
def __(df_prob, pd, plt, sns):
    def get_left_count_corr(col):
        col = col[col > 0]
        left_ratio = []
        avg_probs = []
        estimation_length = []
        for _i in range(15, len(col)):
            probs = col.iloc[:_i].max()
            left_ratio.append(len(col) - _i)
            avg_probs.append(probs)
            estimation_length.append(_i / probs)
        tmp_df = pd.DataFrame(
            {
                "left_ratio": left_ratio,
                "mean": avg_probs,
                "estimation_length": estimation_length,
            }
        )
        _corr = tmp_df.corr(method="pearson").loc["mean", "left_ratio"]
        return (col.name, _corr)


    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.ecdfplot(df_prob.apply(lambda col: get_left_count_corr(col), axis=0).T[1])
    plt.xlabel("Left Length vs. Max Prob")
    plt.grid(alpha=0.3, linestyle="--")
    return get_left_count_corr,


@app.cell
def __(df_prob, pd):
    from scipy.stats import spearmanr


    def get_left_count(col, start):
        col = col[col > 0]
        left_ratio = []
        avg_probs = []
        estimation_length = []
        for _i in range(_start, len(col)):
            probs = col.iloc[:_i].mean()
            left_ratio.append(len(col) - _i)
            avg_probs.append(probs)
            estimation_length.append(_i / probs)
        tmp_df = pd.DataFrame(
            {
                "left_ratio": left_ratio,
                "mean": avg_probs,
                "estimation_length": estimation_length,
            }
        )
        return tmp_df


    _start_values = range(1, 21)
    correlations = {}
    for _start in _start_values:
        tmp_dfs = [get_left_count(df_prob[col], _start) for col in df_prob]
        _true_pred = [
            (tmp_df.loc[:, "left_ratio"][0], tmp_df.loc[:, "estimation_length"][0])
            for tmp_df in tmp_dfs
            if len(tmp_df) > 0
        ]
        _true_pred = [x for x in _true_pred if x[0] > 0 and x[1] > 0]
        (_true_length, _pred_length) = (
            [x[0] for x in _true_pred],
            [x[1] for x in _true_pred],
        )
        if len(_true_length) > 1 and len(_pred_length) > 1:
            (_corr, _) = spearmanr(_true_length, _pred_length)
            correlations[_start] = _corr
        else:
            correlations[_start] = None
    for _start, _corr in correlations.items():
        print(f"Start: {_start}, Correlation: {_corr}")
    return correlations, get_left_count, spearmanr, tmp_dfs


@app.cell
def __(df_prob, np, pd):
    from scipy.stats import spearmanr


    def get_left_count(col, start, n):
        col = col[col > 0]
        left_ratio = []
        gittins_index = []
        for _i in range(_start, len(col)):
            probs = col.iloc[:_i].max()
            left_ratio.append(len(col) - _i)
            gittins_index.append(
                (1 - (1 - probs) ** _n)
                / np.sum(np.arange(1, _n) * (1 - probs) ** np.arange(1, _n))
            )
        tmp_df = pd.DataFrame(
            {"left_ratio": left_ratio, "estimation_length": gittins_index}
        )
        return tmp_df


    _start_values = range(15, 16)
    n_values = range(3, 15)
    correlations = {}
    for _n in n_values:
        for _start in _start_values:
            correlations[_n] = []
            tmp_dfs = [
                get_left_count(df_prob[col], _start, _n) for col in df_prob
            ]
            for _i in range(0, len(df_prob)):
                _true_pred = [
                    (
                        tmp_df.loc[:, "left_ratio"][_i],
                        tmp_df.loc[:, "estimation_length"][_i],
                    )
                    for tmp_df in tmp_dfs
                    if len(tmp_df) > _i
                ]
                _true_pred = [x for x in _true_pred if x[0] > 0 and x[1] > 0]
                (_true_length, _pred_length) = (
                    [x[0] for x in _true_pred],
                    [x[1] for x in _true_pred],
                )
                if len(_true_length) > 1 and len(_pred_length) > 1:
                    (_corr, _) = spearmanr(_true_length, _pred_length)
                    correlations[_n].append(_corr)
                else:
                    correlations[_n].append(None)
    return correlations, get_left_count, n_values, spearmanr, tmp_dfs


@app.cell
def __(tmp_dfs):
    tmp_dfs
    return


@app.cell
def __(correlations, np):
    for _n in correlations:
        _corrs = [_i for _i in correlations[_n] if _i is not None]
        print(_corrs)
        print(_n, np.mean(_corrs))
    return


@app.cell
def __(df_prob, pd):
    from scipy.stats import spearmanr


    def get_left_count(col, start):
        col = col[col > 0]
        left_ratio = []
        estimation_length = []
        for _i in range(_start, _start + 2):
            mean_probs = col.iloc[:_i].mean()
            max_probs = col.iloc[:_i].max()
            left_ratio.append(len(col) - _i)
            estimation_length.append(mean_probs / max_probs)
        tmp_df = pd.DataFrame(
            {"left_ratio": left_ratio, "estimation_length": estimation_length}
        )
        return tmp_df


    _start_values = range(2, 40)
    probs_correlations = {}
    for _start in _start_values:
        tmp_dfs = [get_left_count(df_prob[col], _start) for col in df_prob]
        _true_pred = [
            (tmp_df.loc[:, "left_ratio"][0], tmp_df.loc[:, "estimation_length"][0])
            for tmp_df in tmp_dfs
            if len(tmp_df) > 0
        ]
        _true_pred = [x for x in _true_pred if x[0] > 0 and x[1] > 0]
        (_true_length, _pred_length) = (
            [x[0] for x in _true_pred],
            [x[1] for x in _true_pred],
        )
        if len(_true_length) > 1 and len(_pred_length) > 1:
            (_corr, _) = spearmanr(_true_length, _pred_length)
            probs_correlations[_start] = _corr
        else:
            probs_correlations[_start] = None
    for _start, _corr in probs_correlations.items():
        print(f"Start: {_start}, Correlation: {_corr}")
    return get_left_count, probs_correlations, spearmanr, tmp_dfs


@app.cell
def __(df, df_prob, get_left_count_corr, pd, plt):
    left_count_corr = df_prob.apply(lambda col: get_left_count_corr(col), axis=0).T
    left_count_corr["index"] = left_count_corr[0].apply(lambda x: int(x))
    left_count_corr["left_count_corr"] = left_count_corr[1]
    _count = df.describe().T["count"]
    _corrs = []
    for _i in range(1, 30):
        df_eos_prob_count = df_prob[:_i].describe().T[["count", "max"]]
        df_eos_prob_count["count"] = _count - _i
        df_eos_prob_count = df_eos_prob_count[
            df_eos_prob_count["count"] > 0
        ].reset_index()
        df_eos_prob_count = pd.merge(
            df_eos_prob_count, left_count_corr, on="index"
        )
        _corrs.append(
            df_eos_prob_count[["max", "left_count_corr"]]
            .corr(method="pearson")
            .loc["max", "left_count_corr"]
        )
    print(_corrs)
    plt.figure(figsize=(4, 2.5), dpi=150)
    plt.plot(range(1, 30), _corrs)
    plt.xlabel("# of Iters")
    plt.ylabel("Corr")
    plt.grid(alpha=0.3, linestyle="--")
    return df_eos_prob_count, left_count_corr


@app.cell
def __(df_eos_prob_count):
    df_eos_prob_count[["mean", "left_count_corr"]]
    return


@app.cell
def __(df_eos_prob_count, plt, sns):
    sns.scatterplot(df_eos_prob_count, x="left_count_corr", y="mean")
    plt.yscale("log")
    return


@app.cell
def __(df_prob, plt, sns):
    plt.figure(figsize=(4, 2.5), dpi=150)
    rolling_mean_df_prob = df_prob.rolling(window=10, center=True).mean()
    data = rolling_mean_df_prob.loc[:, [12, 15, 36]]
    sns.lineplot(data=data, dashes=False)
    plt.yscale("log")
    return data, rolling_mean_df_prob


@app.cell
def __(mo):
    mo.md("""\n    # EoS Position Analysis\n""")
    return


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df, pd, plt, sns):
    output = df.describe().T[["count"]]
    pos_mean = df[:20].describe().T[["std"]]
    count_pos_mean = pd.DataFrame(
        {"mean": pos_mean.values.flatten(), "count": output.values.flatten()}
    )
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.scatterplot(data=count_pos_mean, x="mean", y="count")
    plt.xlabel("Avg. Pos")
    plt.ylabel("Sequence Length")
    plt.grid(alpha=0.3, linestyle="--")
    return count_pos_mean, output, pos_mean


@app.cell
def __(df, np, plt):
    mov_avg = np.convolve(df[3], np.ones(20) / 20, mode="valid")
    plt.plot(mov_avg)
    return mov_avg,


@app.cell
def __():
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
