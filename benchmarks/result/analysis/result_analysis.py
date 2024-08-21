import marimo

__generated_with = "0.7.12"
app = marimo.App(width="full")


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import json
    import re
    import marimo as mo
    return json, mo, np, os, pd, plt, re, sns


@app.cell
def __():
    base_dir = "/root/vllm/benchmarks/result"
    replace_name = {
        "fcfs": "FCFS",
        "infer": "Infer",
        "full": "Full",
        "half": "Half",
        "sjf": "SJF",
        "tfittradeoff": "TFITTradeoff",
    }
    return base_dir, replace_name


@app.cell
def __(mo):
    mo.md("""\n    # E2E result\n""")
    return


@app.cell
def __(base_dir, os):
    _date = "20240816"
    _counters = [0]
    e2e_result_dir_names = [
        os.path.join(base_dir, _date, str(counter)) for counter in _counters
    ]
    return e2e_result_dir_names,


@app.cell
def __(e2e_result_dir_names, json, os, pd, replace_name):
    e2e_result_dfs = {}
    for dir_name in e2e_result_dir_names:
        for file in os.listdir(dir_name):
            if file.endswith(".json"):
                with open(os.path.join(dir_name, file), "r") as f:
                    data = json.load(f)
                e2e_result_df = pd.DataFrame(data)
                e2e_result_df.drop(
                    columns=[
                        "date",
                        "backend",
                        "tokenizer_id",
                        "best_of",
                        "use_beam_search",
                    ],
                    inplace=True,
                )
                e2e_result_df.replace(
                    replace_name,
                    inplace=True,
                )
                e2e_result_dfs[file] = e2e_result_df
    return data, dir_name, e2e_result_df, e2e_result_dfs, f, file


@app.cell
def __():
    def add_num_annotation(ax, rotation=0):
        for _p in ax.patches:
            if _p.get_height() == 0:
                continue
            ax.annotate(
                str(round(_p.get_height(), 2)),
                (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 1),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                rotation=rotation,
            )
    return add_num_annotation,


@app.cell
def __():
    def get_tp_ratio(df):
        print(df)
        min_result = df["output_throughput"].min()
        df["output_throughput"] = df["output_throughput"] / min_result
        return df
    return get_tp_ratio,


@app.cell
def e2e_result(
    add_num_annotation,
    e2e_result_dfs,
    get_tp_ratio,
    pd,
    plt,
    sns,
):
    e2e_result = {
        "scheduling_policies": [],
        "swap_policies": [],
        "request_throughput": [],
        "output_throughput": [],
        "request_rates": [],
    }
    for _df_name in e2e_result_dfs:
        _tmp_df = e2e_result_dfs[_df_name]
        e2e_result["scheduling_policies"].append(
            _tmp_df["scheduler_policy"].iloc[0]
        )
        e2e_result["swap_policies"].append(_tmp_df["swap_policy"].iloc[0])
        e2e_result["request_throughput"].append(
            _tmp_df["request_throughput"].mean()
        )
        e2e_result["request_rates"].append(_tmp_df["request_rate"].iloc[0])
        e2e_result["output_throughput"].append(_tmp_df["output_throughput"].mean())

    _result_df = pd.DataFrame(e2e_result)

    _result_df = (
        _result_df.groupby(
            ["scheduling_policies", "swap_policies", "request_rates"]
        )
        .mean()
        .reset_index()
    )
    _result_df = (
        _result_df.groupby(
            ["swap_policies", "request_rates"],
            group_keys=False,
        )
        .apply(lambda row: get_tp_ratio(row))
        .reset_index()
    )
    sns.set_style(style="whitegrid")
    sns.set_palette("deep")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 4),
        dpi=150,
    )
    sns.barplot(
        data=_result_df,
        x="request_rates",
        y="request_throughput",
        hue="scheduling_policies",
        ax=axes[0],
        width=0.7,
    )
    add_num_annotation(axes[0], rotation=90)
    axes[0].legend(
        title="",
        frameon=False,
        ncol=3,
        loc=(0, 1),
        handlelength=1.0,
        columnspacing=0.5,
    )
    axes[0].set_ylim(0, 1.7)
    axes[0].set_ylabel("Throughput (requests/s)")
    axes[0].set_xlabel("Request Rate (r/s)")
    axes[0].grid(linestyle="--", alpha=0.5, axis="y")
    sns.barplot(
        data=_result_df,
        x="request_rates",
        y="output_throughput",
        hue="scheduling_policies",
        legend=False,
        ax=axes[1],
        width=0.7,
    )
    add_num_annotation(axes[1], rotation=90)
    # axes[1].set_ylim(0, 600\
    axes[1].set_ylabel("Throughput (Token/s)")
    axes[1].set_xlabel("Request Rate (r/s)")
    axes[1].grid(linestyle="--", alpha=0.5, axis="y")
    plt.show()
    return axes, e2e_result, fig


@app.cell
def __(e2e_result_dfs):
    selected_result = {
        "scheduler_policy": [],
        "swap_policy": [],
        "request_rate": [],
    }
    selected_columns = [
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "mean_lat_ms",
        "median_lat_ms",
        "p99_lat_ms",
    ]
    for _column in selected_columns:
        selected_result[_column] = []
    for _df_name in e2e_result_dfs:
        tmp_df = e2e_result_dfs[_df_name]
        for _column in selected_result:
            if _column in tmp_df.columns:
                if isinstance(tmp_df[_column][0], float):
                    selected_result[_column].append(tmp_df[_column].mean())
                else:
                    selected_result[_column].append(tmp_df[_column][0])
    return selected_columns, selected_result, tmp_df


@app.cell(hide_code=True)
def __(add_num_annotation, plt, sns):
    def line_plot(_long_df):
        (_fig, _axes) = plt.subplots(
            figsize=(4 * 2, 2 * 2), dpi=150, nrows=2, ncols=2
        )
        _long_df = _long_df[_long_df["metric_name"] != "Median"]
        metric_types = _long_df["metric_type"].unique().tolist()
        metric_names = _long_df["metric_name"].unique().tolist()
        scheduler_policies = _long_df["scheduler_policy"].unique().tolist()
        _long_df["line_type"] = _long_df[
            ["scheduler_policy", "metric_name"]
        ].apply(
            lambda row: row["scheduler_policy"] + " " + row["metric_name"], axis=1
        )
        line_styles = ["-", "--", "-.", ":"]
        mark_styles = ["d", "o", "v"]
        colors = ["r", "g", "b", "y"]
        show_legend = True
        for _i, metric_type in enumerate(metric_types):
            _ax = _axes[_i // 2][_i % 2]
            if _i > 0:
                show_legend = False
            data = _long_df[(_long_df["metric_type"] == metric_type)]

            # 按 line_type 分组
            grouped = data.groupby(["scheduler_policy", "metric_name"])
            # # 遍历每个 line_type 并绘制线条
            for name, group in grouped:
                policy, metric = name
                line_style = line_styles[scheduler_policies.index(policy)]
                mark_style = mark_styles[metric_names.index(metric)]
                color = colors[scheduler_policies.index(policy)]
                _ax.plot(
                    group["request_rate"],
                    group["Ratio"],
                    label=name,
                    linestyle=line_style,
                    marker=mark_style,
                    color=color,
                )

            _ax.set_xlabel(metric_type)
            _ax.set_ylabel("")
            add_num_annotation(_ax, rotation=0)
            _ax.grid(linestyle="--", alpha=0.5, axis="y")
        _axes[0][0].legend(
            title="",
            frameon=False,
            ncol=3,
            loc=(0, 1),
            handlelength=1.0,
            columnspacing=0.5,
        )


    def barplot(_long_df, request_rate):
        (_fig, _ax) = plt.subplots(
            figsize=(6 * 2, 1.5 * 2), dpi=150, nrows=2, ncols=2
        )
        metric_types = _long_df["metric_type"].unique().tolist()
        metric_names = _long_df["metric_name"].unique().tolist()

        _long_df = _long_df[_long_df["request_rate"] == request_rate]

        show_legend = True
        for metric_type in metric_types:
            _i = metric_types.index(metric_type)
            if _i > 0:
                show_legend = False
            sns.barplot(
                hue="scheduler_policy",
                y="Ratio",
                x="metric_name",
                data=_long_df[_long_df["metric_type"] == metric_type],
                ax=_ax[_i // 2][_i % 2],
                legend=show_legend,
            )
            _ax[_i // 2][_i % 2].set_xlabel(metric_type)
            _ax[_i // 2][_i % 2].set_ylabel("")
            _ax[_i // 2][_i % 2].set_ylim(
                0,
                _long_df[_long_df["metric_type"] == metric_type]["Ratio"].max()
                * 1.5,
            )
            add_num_annotation(_ax[_i // 2][_i % 2], rotation=90)
            _ax[_i // 2][_i % 2].grid(linestyle="--", alpha=0.5, axis="y")

            _ax[0][0].legend(
                title="",
                frameon=False,
                ncol=3,
                loc=(0, 1),
                handlelength=1.0,
                columnspacing=0.5,
            )
    return barplot, line_plot


@app.cell
def __(barplot, fig, pd, plt, selected_columns, selected_result):
    def get_metric_ratio(df):
        min_result = df["Value"].min()
        df["Ratio"] = df["Value"] / 1
        return df


    _result_df = pd.DataFrame(selected_result)
    _result_df = (
        _result_df.groupby(["scheduler_policy", "swap_policy", "request_rate"])
        .mean()
        .reset_index()
    )
    _long_df = _result_df[
        ["scheduler_policy", "swap_policy", "request_rate"] + selected_columns
    ]
    _long_df = _long_df.melt(
        id_vars=["scheduler_policy", "swap_policy", "request_rate"],
        value_vars=selected_columns,
        var_name="Metric",
        value_name="Value",
    )
    _long_df = (
        _long_df.groupby(
            ["Metric", "request_rate"],
            group_keys=False,
        )
        .apply(lambda row: get_metric_ratio(row))
        .reset_index()
    )
    _long_df[["metric_name", "metric_type"]] = _long_df["Metric"].apply(
        lambda row: pd.Series(
            [row.split("_", 2)[0].capitalize(), row.split("_", 2)[1].upper()]
        )
    )
    # _long_df = _long_df[_long_df["metric_name"] == "P99"]
    show_legend = True

    line_plot(_long_df)
    # barplot(_long_df, 2)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    plt.show()
    return get_metric_ratio, show_legend


@app.cell
def __(mo):
    mo.md(r"""# Request Level Analysis""")
    return


@app.cell
def __(e2e_result_dfs, np, pd):
    request_level_result = pd.DataFrame()
    for _df_name in e2e_result_dfs:
        _tmp_df = e2e_result_dfs[_df_name].copy()
        _tmp_df["request_level_p99_itls"] = _tmp_df["itls"].apply(
            lambda row: 0 if len(row) == 0 else np.percentile(row, 99)
        )
        _tmp_df.drop(
            columns=[
                "model_id",
                "swap_space",
                "preemption_mode",
                "max_num_seqs",
                "swap_policy",
                "iter_theshold",
                "swap_out_partial_rate",
                "waiting_iter_base",
                "duration",
                "completed",
                "total_input_tokens",
                "total_output_tokens",
                "median_tpot_ms",
                "median_itl_ms",
                "median_lat_ms",
                "input_lens",
                "output_lens",
                "itls",
            ],
            inplace=True,
        )
        request_level_result = pd.concat([request_level_result, _tmp_df], axis=0)
    return request_level_result,


@app.cell
def __(plt, request_level_result, sns):
    def get_p99_ratio(df):
        min_result = df["itls_p99"].min()
        df["itls_p99"] = df["itls_p99"] / min_result
        return df


    _request_level_itls_p99_max = (
        request_level_result.groupby(["scheduler_policy", "request_rate"])
        .apply(lambda _df: _df["request_level_p99_itls"].quantile(0.99))
        .reset_index()
    )
    _request_level_itls_p99_max.columns = [
        "scheduler_policy",
        "request_rate",
        "itls_p99",
    ]
    _request_level_itls_p99_max = (
        _request_level_itls_p99_max.groupby(
            ["request_rate"],
            group_keys=False,
        )
        .apply(lambda row: get_p99_ratio(row))
        .reset_index()
    )
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.barplot(
        data=_request_level_itls_p99_max,
        x="request_rate",
        y="itls_p99",
        hue="scheduler_policy",
    )
    plt.legend(title="")
    plt.grid(alpha=0.3, linestyle="--")
    plt.ylabel("P99 ITL")
    return get_p99_ratio,


@app.cell
def __(plt, request_level_result, sns):
    def get_max_mean_ttft_ratio(df):
        min_result = df["median_ttft_ms"].min()
        df["median_ttft_ms"] = df["median_ttft_ms"] / min_result
        return df


    _request_level_mean_ttfts_max = (
        request_level_result.groupby(["scheduler_policy", "request_rate"])
        .agg({"median_ttft_ms": "median"})
        .reset_index()
    )
    _request_level_mean_ttfts_max.columns = [
        "scheduler_policy",
        "request_rate",
        "median_ttft_ms",
    ]
    _request_level_mean_ttfts_max = (
        _request_level_mean_ttfts_max.groupby(
            ["request_rate"],
            group_keys=False,
        )
        .apply(lambda row: get_max_mean_ttft_ratio(row))
        .reset_index()
    )
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.barplot(
        data=_request_level_mean_ttfts_max,
        x="request_rate",
        y="median_ttft_ms",
        hue="scheduler_policy",
    )
    plt.legend(title="")
    plt.grid(alpha=0.3, linestyle="--")
    plt.ylabel("Median TTFT")
    return get_max_mean_ttft_ratio,


@app.cell
def __(mo):
    mo.md("""# GPU KV Cache Util""")
    return


@app.cell
def __(base_dir, os):
    _date = "20240816"
    _counters = [0]
    execute_result_dir_names = [
        os.path.join(base_dir, _date, str(counter)) for counter in _counters
    ]
    return execute_result_dir_names,


@app.cell
def __(execute_result_dir_names, os, pd):
    execute_result_dfs = {
        "SJF": pd.DataFrame(),
        "FCFS": pd.DataFrame(),
        "TFITTradeoff": pd.DataFrame(),
    }
    for _dir_name in execute_result_dir_names:
        for _file in os.listdir(_dir_name):
            if (
                _file.endswith(".csv")
                and "_detailed" not in _file
                and "10.0qps" in _file
            ):
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                _detailed_result_df["Cache Efficiency"] = (
                    _detailed_result_df["Running"]
                    / _detailed_result_df["GPU KV cache usage"]
                )
                if "sjf" in _file:
                    execute_result_dfs["SJF"] = _detailed_result_df
                elif "fcfs" in _file:
                    execute_result_dfs["FCFS"] = _detailed_result_df
                elif "tfittradeoff" in _file:
                    execute_result_dfs["TFITTradeoff"] = _detailed_result_df
    return execute_result_dfs,


@app.cell
def __(execute_result_dfs, plt, sns):
    plt.figure(figsize=(16, 6), dpi=150)

    # Subplot 1: Avg generation throughput
    plt.subplot(2, 3, 1)
    sns.lineplot(
        data=execute_result_dfs["TFITTradeoff"],
        x=execute_result_dfs["TFITTradeoff"].index,
        y="Avg generation throughput",
        label="TFIT",
    )
    sns.lineplot(
        data=execute_result_dfs["FCFS"],
        x=execute_result_dfs["FCFS"].index,
        y="Avg generation throughput",
        label="FCFS",
    )
    sns.lineplot(
        data=execute_result_dfs["SJF"],
        x=execute_result_dfs["SJF"].index,
        y="Avg generation throughput",
        label="SJF",
    )
    plt.title("Avg generation throughput")
    plt.grid(alpha=0.5, linestyle="--")
    # Subplot 2: Running
    plt.subplot(2, 3, 2)
    sns.lineplot(
        data=execute_result_dfs["TFITTradeoff"],
        x=execute_result_dfs["TFITTradeoff"].index,
        y="Running",
        label="TFIT",
    )
    sns.lineplot(
        data=execute_result_dfs["FCFS"],
        x=execute_result_dfs["FCFS"].index,
        y="Running",
        label="FCFS",
    )
    sns.lineplot(
        data=execute_result_dfs["SJF"],
        x=execute_result_dfs["SJF"].index,
        y="Running",
        label="SJF",
    )
    plt.title("Running")
    plt.grid(alpha=0.5, linestyle="--")
    # Subplot 3: Pending
    plt.subplot(2, 3, 3)
    sns.lineplot(
        data=execute_result_dfs["TFITTradeoff"],
        x=execute_result_dfs["TFITTradeoff"].index,
        y="Pending",
        label="TFIT",
    )
    sns.lineplot(
        data=execute_result_dfs["FCFS"],
        x=execute_result_dfs["FCFS"].index,
        y="Pending",
        label="FCFS",
    )
    sns.lineplot(
        data=execute_result_dfs["SJF"],
        x=execute_result_dfs["SJF"].index,
        y="Pending",
        label="SJF",
    )
    plt.title("Pending")
    plt.grid(alpha=0.5, linestyle="--")
    # Suplt.grid(alpha=0.5, linestyle='--')bplot 4: Swapped
    plt.subplot(2, 3, 4)
    sns.lineplot(
        data=execute_result_dfs["TFITTradeoff"],
        x=execute_result_dfs["TFITTradeoff"].index,
        y="Swapped",
        label="TFIT",
    )
    sns.lineplot(
        data=execute_result_dfs["FCFS"],
        x=execute_result_dfs["FCFS"].index,
        y="Swapped",
        label="FCFS",
    )
    sns.lineplot(
        data=execute_result_dfs["SJF"],
        x=execute_result_dfs["SJF"].index,
        y="Swapped",
        label="SJF",
    )
    plt.title("Swapped")
    plt.grid(alpha=0.5, linestyle="--")
    # Subplot 5: GPU KV cache usage
    plt.subplot(2, 3, 5)
    sns.lineplot(
        data=execute_result_dfs["TFITTradeoff"],
        x=execute_result_dfs["TFITTradeoff"].index,
        y="GPU KV cache usage",
        label="TFIT",
    )
    sns.lineplot(
        data=execute_result_dfs["FCFS"],
        x=execute_result_dfs["FCFS"].index,
        y="GPU KV cache usage",
        label="FCFS",
    )
    sns.lineplot(
        data=execute_result_dfs["SJF"],
        x=execute_result_dfs["SJF"].index,
        y="GPU KV cache usage",
        label="SJF",
    )
    plt.grid(alpha=0.5, linestyle="--")
    # Subplot 6: KV Cache Efficiency
    start_index = 4
    plt.subplot(2, 3, 6)
    sns.lineplot(
        data=execute_result_dfs["TFITTradeoff"][start_index:],
        x=execute_result_dfs["TFITTradeoff"].index[start_index:],
        y="Cache Efficiency",
        label="TFIT",
    )
    sns.lineplot(
        data=execute_result_dfs["FCFS"][start_index:],
        x=execute_result_dfs["FCFS"].index[start_index:],
        y="Cache Efficiency",
        label="FCFS",
    )
    sns.lineplot(
        data=execute_result_dfs["SJF"][start_index:],
        x=execute_result_dfs["SJF"].index[start_index:],
        y="Cache Efficiency",
        label="SJF",
    )
    print(execute_result_dfs["SJF"]["Swapped"].mean())
    print(execute_result_dfs["TFITTradeoff"]["Swapped"].mean())
    print(execute_result_dfs["FCFS"]["Swapped"].mean())

    plt.title("GPU KV cache usage")
    plt.grid(alpha=0.5, linestyle="--")
    plt.tight_layout()
    plt.gca()
    return start_index,


@app.cell
def __(mo):
    mo.md(r"""# Detailed Analysis""")
    return


@app.cell
def __(base_dir, os):
    _date = "20240816"
    _counters = [0]
    detailed_result_dir_names = [
        os.path.join(base_dir, _date, str(counter)) for counter in _counters
    ]
    return detailed_result_dir_names,


@app.cell(hide_code=True)
def __(detailed_result_dir_names, os, pd):
    detailed_result_dfs = {
        "SJF": pd.DataFrame(),
        "FCFS": pd.DataFrame(),
        "TFITTradeoff": pd.DataFrame(),
    }
    for _dir_name in detailed_result_dir_names:
        for _file in os.listdir(_dir_name):
            if _file.endswith("_detailed.csv"):
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                last_row = _detailed_result_df.iloc[-1:, :]
                if "sjf" in _file:
                    detailed_result_dfs["SJF"] = pd.concat(
                        [
                            detailed_result_dfs["SJF"],
                            last_row,
                        ],
                        axis=0,
                    )
                elif "fcfs" in _file:
                    detailed_result_dfs["FCFS"] = pd.concat(
                        [
                            detailed_result_dfs["FCFS"],
                            last_row,
                        ],
                        axis=0,
                    )
                elif "tfitt" in _file:
                    detailed_result_dfs["TFITTradeoff"] = pd.concat(
                        [
                            detailed_result_dfs["TFITTradeoff"],
                            last_row,
                        ],
                        axis=0,
                    )
    return detailed_result_dfs, last_row


@app.cell
def __(detailed_result_dfs):
    sjf_mean = detailed_result_dfs["SJF"].mean()
    fcfs_mean = detailed_result_dfs["FCFS"].mean()
    tfittradeoff_mean = detailed_result_dfs["TFITTradeoff"].mean()
    return fcfs_mean, sjf_mean, tfittradeoff_mean


@app.cell
def __(fcfs_mean, pd, sjf_mean, tfittradeoff_mean):
    detailed_mean_result = pd.concat(
        [sjf_mean, fcfs_mean, tfittradeoff_mean], axis=1
    )
    detailed_mean_result.columns = ["SJF", "FCFS", "TFITTradeoff"]
    detailed_mean_result = detailed_mean_result.reset_index().melt(
        id_vars="index", var_name="policy", value_name="value"
    )
    detailed_mean_result.rename(columns={"index": "metric"}, inplace=True)
    return detailed_mean_result,


@app.cell
def __(detailed_mean_result, plt, sns):
    plt.figure(figsize=(12, 2.5), dpi=150)
    sns.barplot(
        data=detailed_mean_result[
            detailed_mean_result["metric"].isin(
                [
                    "Total schedule time",
                    "execution time",
                    "handle output time",
                    "swap time",
                ]
            )
        ],
        hue="policy",
        y="value",
        x="metric",
    )
    # plt.yscale("log")
    plt.legend(title="")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
