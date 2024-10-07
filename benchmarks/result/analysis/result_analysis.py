import marimo

__generated_with = "0.8.14"
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
    _date = "results"
    _counters = [512]
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
                print(len(file))
                e2e_result_df.replace(
                    replace_name,
                    inplace=True,
                )
                e2e_result_dfs[file] = e2e_result_df
    return data, dir_name, e2e_result_df, e2e_result_dfs, f, file


@app.cell
def __(e2e_result_dir_names):
    e2e_result_dir_names
    return


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
        df["output_throughput"] = df["output_throughput"] / 1
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
        .drop(columns=["swap_policies", "request_rates"])
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
    axes[0].set_ylim(0, 2.0)
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


@app.cell
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
        .drop(columns=["Metric", "request_rate"])
        .reset_index()
    )
    print(_long_df)
    _long_df[["metric_name", "metric_type"]] = _long_df["Metric"].apply(
        lambda row: pd.Series(
            [row.split("_", 2)[0].capitalize(), row.split("_", 2)[1].upper()]
        )
    )
    # _long_df = _long_df[_long_df["metric_name"] == "P99"]
    show_legend = True

    # line_plot(_long_df)
    # print(_long_df)
    barplot(_long_df, 2)  # Need to change
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


@app.cell(hide_code=True)
def __(mo):
    mo.md("""# GPU KV Cache Util""")
    return


@app.cell
def __(base_dir, os):
    _date = "results"
    _counters = [512]
    execute_result_dir_names = [
        os.path.join(base_dir, _date, str(counter)) for counter in _counters
    ]
    return execute_result_dir_names,


@app.cell
def __(execute_result_dir_names, os, pd, plt, sns):
    # For Motivation
    execute_result_dfs_moti = {}
    for _dir_name in execute_result_dir_names:
        for _file in os.listdir(_dir_name):
            if (
                _file.endswith(".csv")
                and "_detailed" not in _file
                and "50.0qps" in _file  # Need to change
            ):
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                # _detailed_result_df["gpu memory iter"] = _detailed_result_df[
                #     "gpu memory iter"
                # ]
                # print((_detailed_result_df["gpu memory iter"]).mean())
                # print((_detailed_result_df["gpu computation iter"]).mean() / 6900)
                # _detailed_result_df["gpu computation iter"] = (
                #     _detailed_result_df["gpu computation iter"] / 6900
                # )
                if "sjf" in _file:
                    execute_result_dfs_moti["SJF"] = _detailed_result_df
                elif "fcfs" in _file:
                    execute_result_dfs_moti["FCFS"] = _detailed_result_df
                elif "tfittradeoff" in _file:
                    execute_result_dfs_moti["TFITTradeoff"] = _detailed_result_df
    print(execute_result_dfs_moti)


    # For Motivation:
    plt.figure(figsize=(16, 6), dpi=150)
    # metric_labels_moti = ["gpu memory iter", "gpu computation iter"]
    # print(execute_result_dfs_moti[""])
    metric_labels_moti = [
        "Avg generation throughput",
        "Running",
        "Pending",
        "Swapped",
        "GPU KV cache usage",
        # "Cache Efficiency",
    ]
    policies_moti = list(execute_result_dfs_moti.keys())

    for metric_label_moti in metric_labels_moti:
        plt.subplot(2, 3, metric_labels_moti.index(metric_label_moti) + 1)
        plt.title(metric_label_moti)
        plt.grid(alpha=0.5, linestyle="--")
        for policy_moti in policies_moti:
            sns.lineplot(
                data=execute_result_dfs_moti[policy_moti],
                x=execute_result_dfs_moti[policy_moti].index,
                y=metric_label_moti,
                label=policy_moti,
            )

    plt.subplot(2, 3, len(metric_labels_moti) + 1)
    plt.title("Motivation")
    plt.grid(alpha=0.5, linestyle="dashdot")
    plt.show()
    # Option 1: Draw all motivations
    # policy_moti = "SJF"  # TFITTradeoff, SJF, FCFS
    # sns.jointplot(
    #     data=execute_result_dfs_moti[policy_moti],
    #     x=metric_labels_moti[1],
    #     y=metric_labels_moti[0],
    #     label=policy_moti,
    # )

    # policy_moti = "FCFS"  # TFITTradeoff, SJF, FCFS
    # sns.jointplot(
    #     data=execute_result_dfs_moti[policy_moti],
    #     x=metric_labels_moti[1],
    #     y=metric_labels_moti[0],
    #     label=policy_moti,
    # )

    # # Option 2: Draw single policy motivation
    # for policy_moti in policies_moti:
    #     # policy_moti = "TFITTradeoff" # TFITTradeoff, SJF, FCFS
    #     sns.jointplot(
    #         data=execute_result_dfs_moti[policy_moti],
    #         x=metric_labels_moti[1],
    #         y=metric_labels_moti[0],
    #         label=policy_moti,
    #     )

    # plt.tight_layout()
    # plt.gca()
    return (
        execute_result_dfs_moti,
        metric_label_moti,
        metric_labels_moti,
        policies_moti,
        policy_moti,
    )


@app.cell
def __(execute_result_dir_names, os, pd):
    execute_result_dfs = {
        # "SJF": pd.DataFrame(),
        # "FCFS": pd.DataFrame(),
        # "TFITTradeoff": pd.DataFrame(),
    }
    for _dir_name in execute_result_dir_names:
        for _file in os.listdir(_dir_name):
            if (
                _file.endswith(".csv")
                and "_detailed" not in _file
                and "2.0qps" in _file  # Need to change
            ):
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                _detailed_result_df["Cache Efficiency"] = (
                    _detailed_result_df["Running"]
                    / _detailed_result_df["GPU KV cache usage"]
                )
                if "sjf" in _file:
                    continue
                    execute_result_dfs["SJF"] = _detailed_result_df
                elif "fcfs" in _file:
                    execute_result_dfs["FCFS"] = _detailed_result_df
                elif "tfittradeoff" in _file:
                    execute_result_dfs["TFITTradeoff"] = _detailed_result_df
    return execute_result_dfs,


@app.cell
def __(mo):
    mo.md(r"""# Detailed Analysis""")
    return


@app.cell
def __(base_dir, os):
    _date = "results"
    _counters = ["motivation"]
    detailed_result_dir_names = [
        os.path.join(base_dir, _date, str(counter)) for counter in _counters
    ]
    return detailed_result_dir_names,


@app.cell
def __(detailed_result_dir_names, os, pd):
    detailed_result_dfs = pd.DataFrame()
    for _dir_name in detailed_result_dir_names:
        for _file in os.listdir(_dir_name):
            if (
                _file.endswith("_detailed.csv") and "20.0qps" in _file
            ):  # need to change
                qps = _file.split("qps")[0]
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                _detailed_result_df["qps"] = float(qps)
                if "sjf" in _file:
                    _detailed_result_df["schedule_policy"] = "SJF"

                elif "fcfs" in _file:
                    _detailed_result_df["schedule_policy"] = "FCFS"
                elif "tfitt" in _file:
                    _detailed_result_df["schedule_policy"] = "TFITTradeoff"
                _detailed_result_df["gpu computation iter"] = (
                    _detailed_result_df["gpu computation iter"] / 5000
                )
                # _detailed_result_df["gpu memory iter"] = (
                #     _detailed_result_df["gpu memory iter"] / 0.5
                # )
                _detailed_result_df["computation memory rate"] = (
                    _detailed_result_df["gpu computation iter"]
                    / _detailed_result_df["gpu memory iter"]
                )
                _detailed_result_df.reset_index(inplace=True)
                detailed_result_dfs = pd.concat(
                    [
                        detailed_result_dfs,
                        _detailed_result_df,
                    ],
                    axis=0,
                    ignore_index=True,
                )
    return detailed_result_dfs, qps


@app.cell(disabled=True, hide_code=True)
def __(detailed_result_dfs, plt, sns):
    jointplot_ax = sns.jointplot(
        detailed_result_dfs[detailed_result_dfs["schedule_policy"] == "FCFS"],
        x="gpu computation iter",
        y="gpu memory iter",
        hue="schedule_policy",
        height=6,
        ratio=5,
        legend=False,
    )
    jointplot_ax.fig.set_size_inches((4, 2.5))
    jointplot_ax.fig.set_dpi(120)
    # plt.xlim(0, 1)
    plt.legend(frameon=False, fontsize=10)
    # plt.yscale("log")
    plt.grid(alpha=0.5, linestyle="-.")
    plt.xlabel("GPU Computaion Rate", fontsize=10)
    plt.ylabel("GPU Memory Rate", fontsize=10)
    plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
    plt.savefig("100_qps.pdf")
    plt.show()
    return jointplot_ax,


@app.cell
def __(detailed_result_dfs):
    def get_iteration_time(df):
        df = df.sort_values(by="total iteration number")
        df["iteration_time"] = df["execution time"].diff().shift(-1)

        return df[
            [
                "schedule_policy",
                "iteration_time",
                "gpu memory iter",
                "gpu computation iter",
            ]
        ]


    iteration_time_gpu_resource = (
        detailed_result_dfs.groupby(["schedule_policy"])
        .apply(lambda df: get_iteration_time(df))
        .drop(columns=["schedule_policy"])
        .reset_index()
        .rename(columns={"level_1": "iter"})
    )
    request_rate_gpu_resource = (
        detailed_result_dfs[
            [
                "schedule_policy",
                "gpu memory iter",
                "gpu computation iter",
                "throughput iter",
            ]
        ]
        .reset_index()
        .rename(columns={"index": "iter"})
    )
    return (
        get_iteration_time,
        iteration_time_gpu_resource,
        request_rate_gpu_resource,
    )


@app.cell
def __(plt, sns):
    def plot_perf_gpu_resource_line(source_df, perf_name, window_size=5):
        window_size = 5
        update_perf_gpu_resource = source_df[
            ["iter", "gpu memory iter", "gpu computation iter", perf_name]
        ]
        update_perf_gpu_resource["gpu computation iter"] = round(
            update_perf_gpu_resource["gpu computation iter"], 6
        )
        update_perf_gpu_resource = (
            update_perf_gpu_resource.groupby(["gpu memory iter"])
            .mean()
            .reset_index()
        )
        for col in ["gpu memory iter", "gpu computation iter", perf_name]:
            update_perf_gpu_resource[col] = (
                update_perf_gpu_resource[col]
                .rolling(window=window_size, min_periods=1)
                .mean()
            )

        plt.figure(figsize=(4, 2.5), dpi=150)
        colors = sns.color_palette("deep", 3)
        gpu_resource_per_iter = plt.subplot(111)
        sns.lineplot(
            data=update_perf_gpu_resource,
            y="gpu memory iter",
            x="iter",
            color=colors[0],
            ax=gpu_resource_per_iter,
            label="mem",
        )
        sns.lineplot(
            data=update_perf_gpu_resource,
            y="gpu computation iter",
            x="iter",
            color=colors[1],
            ax=gpu_resource_per_iter,
            label="comp",
        )
        perf_per_tier = gpu_resource_per_iter.twinx()
        sns.lineplot(
            data=update_perf_gpu_resource,
            y=perf_name,
            x="iter",
            color=colors[2],
            ax=perf_per_tier,
            label=perf_name,
        )
        perf_per_tier.grid(False)
        perf_per_tier.set_yscale("log")
        gpu_resource_per_iter.set_yscale("log")
        plt.show()
    return plot_perf_gpu_resource_line,


@app.cell
def __(plot_perf_gpu_resource_line, request_rate_gpu_resource):
    plot_perf_gpu_resource_line(request_rate_gpu_resource, "throughput iter")
    return


@app.cell
def __(plt, request_rate_gpu_resource, sns):
    def plot_perf_gpu_resource_heatmap(source_df, perf_name, bucket_size=0.05):
        update_perf_gpu_resource_bucket = source_df.copy()
        update_perf_gpu_resource_bucket["gpu memory iter"] = round(
            update_perf_gpu_resource_bucket["gpu memory iter"]
            // bucket_size
            * bucket_size,
            2,
        )
        update_perf_gpu_resource_bucket["gpu computation iter"] = round(
            update_perf_gpu_resource_bucket["gpu computation iter"]
            // bucket_size
            * bucket_size,
            2,
        )
        update_perf_gpu_resource_bucket = (
            update_perf_gpu_resource_bucket.groupby(
                ["gpu memory iter", "gpu computation iter"]
            )
            .agg({perf_name: "mean"})
            .reset_index()
        )
        update_perf_gpu_resource_bucket_avg_time = (
            update_perf_gpu_resource_bucket.sort_values(
                by=["gpu memory iter"], ascending=False
            )
        )
        update_iteration_time_gpu_resource_bucket_avg_time_pivot = (
            update_perf_gpu_resource_bucket_avg_time.pivot(
                index="gpu memory iter",
                columns="gpu computation iter",
                values=perf_name,
            )
        )
        update_iteration_time_gpu_resource_bucket_avg_time_pivot = (
            update_iteration_time_gpu_resource_bucket_avg_time_pivot.sort_values(
                by="gpu memory iter", ascending=False
            )
        )
        sns.heatmap(
            update_iteration_time_gpu_resource_bucket_avg_time_pivot, cmap="crest"
        )
        plt.show()


    plot_perf_gpu_resource_heatmap(
        request_rate_gpu_resource, "throughput iter", 0.1
    )
    return plot_perf_gpu_resource_heatmap,


@app.cell(disabled=True, hide_code=True)
def __(detailed_result_dfs, plt, sns):
    plt.cla()
    _tmp_result = detailed_result_dfs[
        ["index", "computation memory rate", "schedule_policy"]
    ]
    sns.histplot(
        data=_tmp_result,
        x="computation memory rate",
        hue="schedule_policy",
        legend=True,
    )
    plt.xscale("log")
    plt.grid(alpha=0.5, linestyle="-.")
    plt.xlabel("Computation/Memory", fontsize=10)
    # plt.ylabel("GPU Memory Rate", fontsize=10)
    plt.show()
    return


@app.cell
def __():
    return


@app.cell
def __(add_num_annotation, detailed_result_dfs, plt, sns):
    detailed_mean_result = (
        detailed_result_dfs.groupby(["schedule_policy"])
        .max()
        .reset_index()
        .melt(id_vars="schedule_policy", var_name="metric", value_name="value")
    )
    plt.figure(figsize=(12, 2.5), dpi=150)
    ax = sns.barplot(
        data=detailed_mean_result[
            detailed_mean_result["metric"].isin(
                [
                    "Total schedule time",
                    "execution time",
                    "handle output time",
                    "swap time",
                    "total iteration number",
                ]
            )
        ],
        hue="schedule_policy",
        y="value",
        x="metric",
    )
    # plt.yscale("log")
    plt.legend(title="")
    add_num_annotation(ax)
    plt.xticks(rotation=45)
    plt.show()
    return ax, detailed_mean_result


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
