import marimo

__generated_with = "0.9.14"
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
        "las": "LAS",
        "infer": "Infer",
        "full": "Full",
        "half": "Half",
        "srjf": "SRJF",
        "sjf": "SJF",
        "tfittradeoff": "TFITTradeoff",
    }
    return base_dir, replace_name


@app.cell
def __(base_dir, os):
    # _date = "20240918"
    date = "20241030"
    counters = [0,118,119,123,125,126]
    selected_qps = 2
    e2e_result_dir_names = [
        os.path.join(base_dir, date, str(counter)) for counter in counters
    ]
    return counters, date, e2e_result_dir_names, selected_qps


@app.cell(hide_code=True)
def __(e2e_result_dir_names, json, os, pd, replace_name):
    e2e_result_dfs = {}
    for dir_name in e2e_result_dir_names:
        for file in os.listdir(dir_name):
            if file.endswith(".json"):
                # print(file)
                with open(os.path.join(dir_name, file), "r") as f:
                    data = json.load(f)
                # print(data)
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


@app.cell(hide_code=True)
def __():
    def add_num_annotation(ax, rotation=0, fontsize=10):
        for _p in ax.patches:
            if _p.get_height() == 0:
                continue
            ax.annotate(
                str(round(_p.get_height(), 2)),
                (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 1.05),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                rotation=rotation,
                fontsize=fontsize,
            )
    return (add_num_annotation,)


@app.cell(hide_code=True)
def __():
    def get_tp_ratio(df):
        print(df)
        min_result = df["output_throughput"].min()
        df["output_throughput"] = df["output_throughput"] / 1
        return df
    return (get_tp_ratio,)


@app.cell(hide_code=True)
def __():
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
            e2e_result["output_throughput"].append(
                _tmp_df["output_throughput"].mean()
            )

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
    return (e2e_result,)


@app.cell(hide_code=True)
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
        save_legend = True
        for _i, metric_type in enumerate(metric_types):
            _ax = _axes[_i // 2][_i % 2]
            if _i > 0:
                save_legend = False
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

        save_legend = True
        for metric_type in metric_types:
            _i = metric_types.index(metric_type)
            if _i > 0:
                save_legend = False
            ax = sns.barplot(
                hue="scheduler_policy",
                y="Ratio",
                x="metric_name",
                data=_long_df[_long_df["metric_type"] == metric_type],
                ax=_ax[_i // 2][_i % 2],
                legend=save_legend,
                width=0.6,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            _ax[_i // 2][_i % 2].set_xlabel(metric_type)
            _ax[_i // 2][_i % 2].set_ylabel("")
            _ax[_i // 2][_i % 2].set_ylim(
                0,
                _long_df[_long_df["metric_type"] == metric_type]["Ratio"].max()
                * 1.5,
            )
            add_num_annotation(_ax[_i // 2][_i % 2], rotation=0)
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


@app.cell(hide_code=True)
def __(line_plot, pd, plt, selected_columns, selected_result):
    def get_metric_ratio(df):
        min_result = df["Value"].min()
        df["Ratio"] = df["Value"] / min_result
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
    long_df = (
        _long_df.groupby(
            ["Metric", "request_rate"],
            group_keys=False,
        )
        .apply(lambda row: get_metric_ratio(row))
        .reset_index()
    )

    long_df[["metric_name", "metric_type"]] = _long_df["Metric"].apply(
        lambda row: pd.Series(
            [row.split("_", 2)[0].capitalize(), row.split("_", 2)[1].upper()]
        )
    )
    # _long_df = _long_df[_long_df["metric_name"] == "P99"]
    save_legend = True

    print()

    line_plot(long_df)
    # print(_long_df)
    # barplot(_long_df, 2)  # Need to change
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.show()
    return get_metric_ratio, long_df, save_legend


@app.cell(hide_code=True)
def __(add_num_annotation, plt, sns):
    def bar_plot_1(_long_df, required_metric_name):
        (_fig, _ax) = plt.subplots(
            figsize=(4 * 2, 2 * 2), dpi=150, nrows=2, ncols=2
        )

        for metric_name in _long_df["metric_name"].unique().tolist():
            if required_metric_name != metric_name:
                _long_df = _long_df[_long_df["metric_name"] != metric_name]
        metric_types = _long_df["metric_type"].unique().tolist()
        metric_names = _long_df["metric_name"].unique().tolist()
        scheduler_policies = _long_df["scheduler_policy"].unique().tolist()

        save_legend = True
        for _i, metric_type in enumerate(metric_types):
            if _i > 0:
                save_legend = False

            data = _long_df[(_long_df["metric_type"] == metric_type)]

            ax = sns.barplot(
                x="request_rate",
                y="Ratio",
                hue="scheduler_policy",
                data=data,
                ax=_ax[_i // 2][_i % 2],
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            _ax[_i // 2][_i % 2].set_xlabel("Request Rate")
            # _ax[_i // 2][_i % 2].set_ylabel(_long_df["metric_name"].unique().tolist()[0] + " " + metric_type+" Ratio")

            add_num_annotation(_ax[_i // 2][_i % 2], rotation=60, fontsize=7)

            _ax[_i // 2][_i % 2].grid(linestyle="--", alpha=0.5, axis="y")

            if _i == 0:
                _ax[_i // 2][_i % 2].legend(
                    title="",
                    frameon=False,
                    ncol=3,
                    loc=(0, 1.1),
                    handlelength=1.0,
                    columnspacing=0.7,
                )
            else:
                _ax[_i // 2][_i % 2].legend().remove()


    def bar_plot_2(_long_df, required_metric_name_list, required_metric_type_list):
        (_fig, _ax) = plt.subplots(
            figsize=(4 * 2, 2 * 2), dpi=150, nrows=2, ncols=2
        )
        _long_df = _long_df[
            (
                (_long_df["metric_name"] == required_metric_name_list[0])
                & (_long_df["metric_type"] == required_metric_type_list[0])
            )
            | (
                (_long_df["metric_name"] == required_metric_name_list[0])
                & (_long_df["metric_type"] == required_metric_type_list[1])
            )
            | (
                (_long_df["metric_name"] == required_metric_name_list[1])
                & (_long_df["metric_type"] == required_metric_type_list[0])
            )
            | (
                (_long_df["metric_name"] == required_metric_name_list[1])
                & (_long_df["metric_type"] == required_metric_type_list[1])
            )
        ]
        metric_types = _long_df["metric_type"].unique().tolist()
        metric_names = _long_df["metric_name"].unique().tolist()
        scheduler_policies = _long_df["scheduler_policy"].unique().tolist()
        metrics = [
            (required_metric_name_list[0], required_metric_type_list[0]),
            (required_metric_name_list[0], required_metric_type_list[1]),
            (required_metric_name_list[1], required_metric_type_list[0]),
            (required_metric_name_list[1], required_metric_type_list[1]),
        ]
        save_legend = True
        for _i, (metric_name, metric_type) in enumerate(metrics):
            if _i > 0:
                save_legend = False

            data = _long_df[
                (_long_df["metric_type"] == metric_type)
                & (_long_df["metric_name"] == metric_name)
            ]
            ax = sns.barplot(
                x="request_rate",
                y="Ratio",
                hue="scheduler_policy",
                data=data,
                ax=_ax[_i // 2][_i % 2],
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            _ax[_i // 2][_i % 2].set_xlabel("Request Rate")
            _ax[_i // 2][_i % 2].set_ylabel(
                metrics[_i][0] + " " + metrics[_i][1] + " Ratio"
            )

            add_num_annotation(_ax[_i // 2][_i % 2], rotation=60, fontsize=7)

            _ax[_i // 2][_i % 2].grid(linestyle="--", alpha=0.5, axis="y")

            if _i == 0:
                _ax[_i // 2][_i % 2].legend(
                    title="",
                    frameon=False,
                    ncol=3,
                    loc=(0, 1.1),
                    handlelength=1.0,
                    columnspacing=0.7,
                )
            else:
                _ax[_i // 2][_i % 2].legend().remove()
    return bar_plot_1, bar_plot_2


@app.cell
def __(bar_plot_2, long_df, plt):
    required_metric_name = "P99"
    # bar_plot_1(_long_df, required_metric_name=required_metric_name)
    required_metric_name_list = ["Mean", "P99"]
    required_metric_type_list = ["TTFT", "TPOT"]
    bar_plot_2(long_df, required_metric_name_list, required_metric_type_list)
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    # plt.savefig(f"{required_metric_name}.pdf", format='pdf')
    plt.show()
    return (
        required_metric_name,
        required_metric_name_list,
        required_metric_type_list,
    )


@app.cell(hide_code=True)
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
    return (request_level_result,)


@app.cell(hide_code=True)
def __(request_level_result):
    def get_p99_ratio(df):
        min_result = df["itls_p99"].min()
        df["itls_p99"] = df["itls_p99"] / min_result
        return df


    request_level_itls_p99_max = (
        request_level_result.groupby(["scheduler_policy", "request_rate"])
        .apply(
            lambda _df: _df["request_level_p99_itls"].quantile(0.99),
            include_groups=False,
        )
        .reset_index()
    )
    request_level_itls_p99_max.columns = [
        "scheduler_policy",
        "request_rate",
        "itls_p99",
    ]
    request_level_itls_p99_max = (
        request_level_itls_p99_max.groupby(
            ["request_rate"],
        )
        .apply(lambda row: get_p99_ratio(row), include_groups=False)
        .reset_index()
    )
    return get_p99_ratio, request_level_itls_p99_max


@app.cell
def __(plt, request_level_itls_p99_max, sns):
    plt.figure(figsize=(4, 2.5), dpi=150)
    sns.barplot(
        data=request_level_itls_p99_max,
        x="request_rate",
        y="itls_p99",
        hue="scheduler_policy",
    )
    plt.legend(title="")
    plt.grid(alpha=0.3, linestyle="--")
    plt.ylabel("P99 ITL")
    return


@app.cell(hide_code=True)
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
    ax = sns.barplot(
        data=_request_level_mean_ttfts_max,
        x="request_rate",
        y="median_ttft_ms",
        hue="scheduler_policy",
    )

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            format(height, ".2f"),  # Format the number to two decimal places
            (p.get_x() + p.get_width() / 2.0, height),  # Position of the label
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            xytext=(0, 5),  # Offset of the label
            textcoords="offset points",
            rotation=30,
            fontsize=6,
        )

    plt.legend(title="")
    plt.grid(alpha=0.3, linestyle="--")
    plt.ylabel("Median TTFT")
    return ax, get_max_mean_ttft_ratio, height, p


@app.cell(hide_code=True)
def __(base_dir, counters, date, os):
    execute_result_dir_names = [
        os.path.join(base_dir, date, str(counter)) for counter in counters
    ]
    return (execute_result_dir_names,)


@app.cell(hide_code=True)
def __(execute_result_dir_names, os, pd, selected_qps):
    # For Motivation
    execute_result_dfs_moti = {}
    for _dir_name in execute_result_dir_names:
        for _file in os.listdir(_dir_name):
            if (
                _file.endswith(".csv")
                and "_detailed" in _file
                and f"{selected_qps}.0qps" in _file  # Need to change
            ):
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                _detailed_result_df["gpu memory iter"] = _detailed_result_df[
                    "gpu memory iter"
                ]
                _detailed_result_df["gpu computation iter"] = (
                    _detailed_result_df["gpu computation iter"] / 6900
                )
                if "sjf" in _file:
                    execute_result_dfs_moti["SJF"] = _detailed_result_df
                if "srjf" in _file:
                    execute_result_dfs_moti["SRJF"] = _detailed_result_df
                elif "las" in _file:
                    execute_result_dfs_moti["LAS"] = _detailed_result_df
                elif "fcfs" in _file:
                    execute_result_dfs_moti["FCFS"] = _detailed_result_df
                elif "tfittradeoff" in _file:
                    execute_result_dfs_moti["TFITTradeoff"] = _detailed_result_df
    return (execute_result_dfs_moti,)


@app.cell(hide_code=True)
def __():
    # # For Motivation:
    # plt.figure(figsize=(16, 6), dpi=150)
    # metric_labels_moti = ["gpu memory iter", "gpu computation iter"]
    # # print(execute_result_dfs_moti[""])
    # # metric_labels = ["Avg generation throughput", "Running", "Pending", "Swapped", "GPU KV cache usage", "Cache Efficiency"]
    # policies_moti = list(execute_result_dfs_moti.keys())

    # for metric_label_moti in metric_labels_moti:
    #     plt.subplot(2, 3, metric_labels_moti.index(metric_label_moti)+1)
    #     plt.title(metric_label_moti)
    #     plt.grid(alpha=0.5, linestyle="--")
    #     for policy_moti in policies_moti:
    #         sns.lineplot(
    #             data=execute_result_dfs_moti[policy_moti],
    #             x=execute_result_dfs_moti[policy_moti].index,
    #             y=metric_label_moti,
    #             label=policy_moti,
    #         )

    # plt.subplot(2, 3, len(metric_labels_moti)+1)
    # plt.title("Motivation")
    # plt.grid(alpha=0.5, linestyle="dashdot")

    # # Option 1: Draw all motivations
    # # policy_moti = "SJF" # TFITTradeoff, SJF, FCFS
    # # sns.jointplot(
    # #     data=execute_result_dfs_moti[policy_moti],
    # #     x=metric_labels_moti[1],
    # #     y=metric_labels_moti[0],
    # #     label=policy_moti,
    # # )

    # policy_moti = "FCFS" # TFITTradeoff, SJF, FCFS
    # sns.jointplot(
    #     data=execute_result_dfs_moti[policy_moti],
    #     x=metric_labels_moti[1],
    #     y=metric_labels_moti[0],
    #     label=policy_moti,
    # )

    # # # Option 2: Draw single policy motivation
    # # for policy_moti in policies_moti:
    # # # policy_moti = "TFITTradeoff" # TFITTradeoff, SJF, FCFS
    # #     sns.jointplot(
    # #         data=execute_result_dfs_moti[policy_moti],
    # #         x=metric_labels_moti[1],
    # #         y=metric_labels_moti[0],
    # #         label=policy_moti,
    # #     )

    # plt.tight_layout()
    # plt.gca()
    return


@app.cell
def __(execute_result_dir_names, os, pd, selected_qps):
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
                and f"{selected_qps}.0qps" in _file  # Need to change
            ):
                _detailed_result_df = pd.read_csv(os.path.join(_dir_name, _file))
                _detailed_result_df["Cache Efficiency"] = (
                    _detailed_result_df["Running"]
                    / _detailed_result_df["GPU KV cache usage"]
                )
                _detailed_result_df["Total Throuhgput"] = (
                    _detailed_result_df["Avg prompt throughput"]
                    + _detailed_result_df["Avg generation throughput"]
                )
                if "sjf" in _file:
                    # continue
                    execute_result_dfs["SJF"] = _detailed_result_df
                elif "las" in _file:
                    execute_result_dfs["LAS"] = _detailed_result_df
                elif "srjf" in _file:
                    execute_result_dfs["SRJF"] = _detailed_result_df
                elif "fcfs" in _file:
                    execute_result_dfs["FCFS"] = _detailed_result_df
                elif "tfittradeoff" in _file:
                    execute_result_dfs["TFITTradeoff"] = _detailed_result_df
    return (execute_result_dfs,)


@app.cell(hide_code=True)
def __(execute_result_dfs, plt, sns):
    plt.figure(figsize=(16, 6), dpi=150)
    # Subplot 1: Avg generation throughput
    metric_labels = [
        "Total Throuhgput",
        "Avg generation throughput",
        "Running",
        "Pending",
        "Swapped",
        "GPU KV cache usage",
        # "Cache Efficiency",
    ]
    policies = list(execute_result_dfs.keys())

    for metric_label in metric_labels:
        plt.subplot(2, 3, metric_labels.index(metric_label) + 1)
        plt.title(metric_label)
        plt.grid(alpha=0.5, linestyle="--")
        for policy in policies:
            sns.lineplot(
                data=execute_result_dfs[policy],
                x=execute_result_dfs[policy].index,
                y=metric_label,
                label=policy,
            )
    plt.tight_layout()
    plt.gca()
    return metric_label, metric_labels, policies, policy


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
