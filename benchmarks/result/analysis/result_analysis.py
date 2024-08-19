import marimo

__generated_with = "0.7.20"
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
    base_dir = "/root/v1/vllm/benchmarks/result"
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
def e2e_result(add_num_annotation, e2e_result_dfs, pd, plt, sns):
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
    axes[1].set_ylim(0, 600)
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


@app.cell
def __():
    def get_metric_ratio(df):
        min_result = df["Value"].min()
        df["Ratio"] = df["Value"] / min_result
        return df
    return get_metric_ratio,


@app.cell
def __(add_num_annotation, plt, sns):
    def line_plot(_long_df):
        (_fig, _axes) = plt.subplots(
            figsize=(4 * 2, 2.5 * 2), dpi=150, nrows=2, ncols=2
        )
        _long_df = _long_df[_long_df['metric_name'] != "Median"] 
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
            figsize=(6 * 2, 2 * 2), dpi=150, nrows=2, ncols=2
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
def __(
    fig,
    get_metric_ratio,
    line_plot,
    pd,
    plt,
    selected_columns,
    selected_result,
):
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
    return show_legend,


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


@app.cell
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
        data=detailed_mean_result,
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
def __(detailed_result_dfs, plt, sns):
    swap_times = []
    swap_block_nums = []

    for df in detailed_result_dfs.values():
        swap_times.extend(df["swap time"])
        swap_block_nums.extend(df["swap out block num"] + df["swap in block num"])

    plt.figure(figsize=(4, 2.5), dpi=150)

    sns.lineplot(x=swap_block_nums, y=swap_times, label="Swap Time")
    # plt.title("Swap Time")
    plt.xlabel("Swap Block Num")
    plt.ylabel("Time")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()


    plt.gca()
    return df, swap_block_nums, swap_times


@app.cell
def __(np, swap_block_nums, swap_times):
    block_swap_out_time = []
    for i in range(len(swap_times)):
        block_swap_out_time.append(swap_times[i] / swap_block_nums[i])
    print(np.mean(block_swap_out_time))
    return block_swap_out_time, i


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
