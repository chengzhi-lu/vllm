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
    import marimo as mo
    return json, mo, np, os, pd, plt, sns


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
    _date = "20240730"
    _counters = [760]
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
def e2e_result(e2e_result_dfs, pd, plt, sns):
    e2e_result = {
        "scheduling_policies": [],
        "swap_policies": [],
        "throughput": []
    }
    for _df_name in e2e_result_dfs:
        _tmp_df = e2e_result_dfs[_df_name]
        e2e_result["scheduling_policies"].append(
            _tmp_df["scheduler_policy"].iloc[0])
        e2e_result["swap_policies"].append(_tmp_df["swap_policy"].iloc[0])
        e2e_result["throughput"].append(_tmp_df["request_throughput"].mean())
    _result_df = pd.DataFrame(e2e_result)
    _result_df = (_result_df.groupby(["scheduling_policies",
                                      "swap_policies"]).mean().reset_index())
    sns.set_style(style="whitegrid")
    sns.set_palette("deep")
    plt.figure(figsize=(4, 2.5), dpi=150)
    _ax = sns.barplot(
        x="scheduling_policies",
        y="throughput",
        hue="swap_policies",
        data=_result_df,
    )
    for _p in _ax.patches:
        if _p.get_height() == 0:
            continue
        _ax.annotate(
            str(round(_p.get_height(), 2)),
            (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 1),
            ha="center",
            va="center",
            xytext=(0, 6),
            textcoords="offset points",
        )
    plt.xlabel("Scheduling Policies")
    plt.ylabel("Throughput (requests/s)")
    plt.legend(
        loc=(0.02, 0.85),
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=0.5,
    )
    plt.grid(linestyle="--", alpha=0.5)
    plt.ylim(0, 1.4)
    plt.show()
    return e2e_result,


@app.cell
def __(e2e_result_dfs, pd, plt, sns):
    token_throughput_result = {
        "scheduling_policies": [],
        "swap_policies": [],
        "throughput": [],
    }
    for _df_name in e2e_result_dfs:
        _tmp_df = e2e_result_dfs[_df_name]
        token_throughput_result["scheduling_policies"].append(
            _tmp_df["scheduler_policy"].iloc[0])
        token_throughput_result["swap_policies"].append(
            _tmp_df["swap_policy"].iloc[0])
        token_throughput_result["throughput"].append(
            _tmp_df["output_throughput"].mean())
    _result_df = pd.DataFrame(token_throughput_result)
    _result_df = (_result_df.groupby(["scheduling_policies",
                                      "swap_policies"]).mean().reset_index())
    sns.set_style(style="whitegrid")
    sns.set_palette("deep")
    plt.figure(figsize=(4, 2.5), dpi=150)
    _ax = sns.barplot(
        x="scheduling_policies",
        y="throughput",
        hue="swap_policies",
        data=_result_df,
    )
    for _p in _ax.patches:
        if _p.get_height() == 0:
            continue
        _ax.annotate(
            str(round(_p.get_height(), 2)),
            (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 1),
            ha="center",
            va="center",
            xytext=(0, 6),
            textcoords="offset points",
        )
    plt.xlabel("Scheduling Policies")
    plt.ylabel("Throughput (requests/s)")
    plt.legend(
        loc=(0.02, 0.85),
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=0.5,
    )
    plt.grid(linestyle="--", alpha=0.5)
    plt.ylim(0, 800)
    plt.show()
    return token_throughput_result,


@app.cell
def __(e2e_result_dfs):
    selected_result = {"scheduler_policy": [], "swap_policy": []}
    _selected_columns = [
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
    for _column in _selected_columns:
        selected_result[_column] = []
    for _df_name in e2e_result_dfs:
        tmp_df = e2e_result_dfs[_df_name]
        for _column in selected_result:
            if _column in tmp_df.columns:
                if isinstance(tmp_df[_column][0], float):
                    selected_result[_column].append(tmp_df[_column].mean())
                else:
                    selected_result[_column].append(tmp_df[_column][0])
    return selected_result, tmp_df


@app.cell
def __(pd, plt, selected_result, sns):
    _result_df = pd.DataFrame(selected_result)
    _result_df = (_result_df.groupby(["scheduler_policy",
                                      "swap_policy"]).mean().reset_index())
    ttft_columns = ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms"]
    _lat_df = _result_df[["scheduler_policy", "swap_policy"] + ttft_columns]
    _tpot_long_df = _lat_df.melt(
        id_vars=["scheduler_policy", "swap_policy"],
        value_vars=ttft_columns,
        var_name="Metric",
        value_name="Value",
    )
    (_fig, _ax) = plt.subplots(figsize=(12, 2.5), dpi=150, nrows=1, ncols=3)
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=_tpot_long_df[_tpot_long_df["Metric"] == "mean_ttft_ms"],
        ax=_ax[0],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=_tpot_long_df[_tpot_long_df["Metric"] == "median_ttft_ms"],
        ax=_ax[1],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=_tpot_long_df[_tpot_long_df["Metric"] == "p99_ttft_ms"],
        ax=_ax[2],
    )
    for _i in range(3):
        for _p in _ax[_i].patches:
            if _p.get_height() == 0:
                continue
            _ax[_i].annotate(
                format(_p.get_height(), ".2f"),
                (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 0.8),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    _ax[0].set_title("Mean TTFT (ms)")
    _ax[1].set_title("Median TTFT (ms)")
    _ax[2].set_title("99th Percentile TTFT (ms)")
    _ax[0].set_xlabel("")
    _ax[1].set_xlabel("")
    _ax[2].set_xlabel("")
    _ax[0].set_ylabel("Value")
    _ax[1].set_ylabel("")
    _ax[2].set_ylabel("")
    plt.legend(bbox_to_anchor=(0.7, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return ttft_columns,


@app.cell
def __(pd, plt, selected_result, sns):
    _result_df = pd.DataFrame(selected_result)
    _result_df = (_result_df.groupby(["scheduler_policy",
                                      "swap_policy"]).mean().reset_index())
    tpot_columns = ["mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"]
    tpot_df = _result_df[["scheduler_policy", "swap_policy"] + tpot_columns]
    _tpot_long_df = tpot_df.melt(
        id_vars=["scheduler_policy", "swap_policy"],
        value_vars=tpot_columns,
        var_name="Metric",
        value_name="Value",
    )
    (_fig, _ax) = plt.subplots(figsize=(12, 2.5), dpi=150, nrows=1, ncols=3)
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=_tpot_long_df[_tpot_long_df["Metric"] == "mean_tpot_ms"],
        ax=_ax[0],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=_tpot_long_df[_tpot_long_df["Metric"] == "median_tpot_ms"],
        ax=_ax[1],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=_tpot_long_df[_tpot_long_df["Metric"] == "p99_tpot_ms"],
        ax=_ax[2],
    )
    for _i in range(3):
        for _p in _ax[_i].patches:
            if _p.get_height() == 0:
                continue
            _ax[_i].annotate(
                format(_p.get_height(), ".2f"),
                (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 0.8),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    _ax[0].set_title("Mean TPOT (ms)")
    _ax[1].set_title("Median TPOT (ms)")
    _ax[2].set_title("99th Percentile TPOT (ms)")
    _ax[0].set_xlabel("")
    _ax[1].set_xlabel("")
    _ax[2].set_xlabel("")
    _ax[0].set_ylabel("Value")
    _ax[1].set_ylabel("")
    _ax[2].set_ylabel("")
    plt.legend(bbox_to_anchor=(0.7, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return tpot_columns, tpot_df


@app.cell
def __(pd, plt, selected_result, sns):
    _result_df = pd.DataFrame(selected_result)
    itl_columns = ["mean_itl_ms", "median_itl_ms", "p99_itl_ms"]
    itl_df = _result_df[["scheduler_policy", "swap_policy"] + itl_columns]
    itl_df = (itl_df.groupby(["scheduler_policy",
                              "swap_policy"]).mean().reset_index())
    itl_long_df = itl_df.melt(
        id_vars=["scheduler_policy", "swap_policy"],
        value_vars=itl_columns,
        var_name="Metric",
        value_name="Value",
    )
    (_fig, _ax) = plt.subplots(figsize=(12, 2.5), dpi=150, nrows=1, ncols=3)
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=itl_long_df[itl_long_df["Metric"] == "mean_itl_ms"],
        ax=_ax[0],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=itl_long_df[itl_long_df["Metric"] == "median_itl_ms"],
        ax=_ax[1],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=itl_long_df[itl_long_df["Metric"] == "p99_itl_ms"],
        ax=_ax[2],
    )
    for _i in range(3):
        for _p in _ax[_i].patches:
            if _p.get_height() == 0:
                continue
            _ax[_i].annotate(
                format(_p.get_height(), ".2f"),
                (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 0.8),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    _ax[0].set_title("Mean ITL (ms)")
    _ax[1].set_title("Median ITL (ms)")
    _ax[2].set_title("99th Percentile ITL (ms)")
    _ax[0].set_xlabel("")
    _ax[1].set_xlabel("")
    _ax[2].set_xlabel("")
    _ax[0].set_ylabel("Value")
    _ax[1].set_ylabel("")
    _ax[2].set_ylabel("")
    plt.legend(bbox_to_anchor=(0.7, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return itl_columns, itl_df, itl_long_df


@app.cell
def __(pd, plt, selected_result, sns):
    _result_df = pd.DataFrame(selected_result)
    lat_columns = ["mean_lat_ms", "median_lat_ms", "p99_lat_ms"]
    _lat_df = _result_df[["scheduler_policy", "swap_policy"] + lat_columns]
    _lat_df = (_lat_df.groupby(["scheduler_policy",
                                "swap_policy"]).mean().reset_index())
    lat_long_df = _lat_df.melt(
        id_vars=["scheduler_policy", "swap_policy"],
        value_vars=lat_columns,
        var_name="Metric",
        value_name="Value",
    )
    (_fig, _ax) = plt.subplots(figsize=(12, 2.5), dpi=150, nrows=1, ncols=3)
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=lat_long_df[lat_long_df["Metric"] == "mean_lat_ms"],
        ax=_ax[0],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=lat_long_df[lat_long_df["Metric"] == "median_lat_ms"],
        ax=_ax[1],
        legend=False,
    )
    sns.barplot(
        x="scheduler_policy",
        y="Value",
        hue="swap_policy",
        data=lat_long_df[lat_long_df["Metric"] == "p99_lat_ms"],
        ax=_ax[2],
    )
    for _i in range(3):
        for _p in _ax[_i].patches:
            if _p.get_height() == 0:
                continue
            _ax[_i].annotate(
                format(_p.get_height(), ".2f"),
                (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 0.8),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    _ax[0].set_title("Mean LAT (ms)")
    _ax[1].set_title("Median LAT (ms)")
    _ax[2].set_title("99th Percentile LAT (ms)")
    _ax[0].set_xlabel("")
    _ax[1].set_xlabel("")
    _ax[2].set_xlabel("")
    _ax[0].set_ylabel("Value")
    _ax[1].set_ylabel("")
    _ax[2].set_ylabel("")
    plt.tight_layout()
    plt.show()
    return lat_columns, lat_long_df


@app.cell
def __(e2e_result_dfs):
    selected_result_for_ecdf = {"scheduler_policy": [], "swap_policy": []}
    _selected_columns = ["ttfts", "itls", "latencies"]
    for _column in _selected_columns:
        selected_result_for_ecdf[_column] = []
    for _df_name in e2e_result_dfs:
        _tmp_df = e2e_result_dfs[_df_name]
        for _column in selected_result_for_ecdf:
            if _column in _tmp_df.columns:
                if isinstance(_tmp_df[_column][0], float):
                    selected_result_for_ecdf[_column].append(_tmp_df[_column])
                else:
                    selected_result_for_ecdf[_column].append(
                        _tmp_df[_column][0])
    return selected_result_for_ecdf,


@app.cell
def __(pd, selected_result_for_ecdf):

    def explode_row(row):
        ttft = row["ttfts"]
        itl = row["itls"]
        latencies = row["latencies"]
        total_length = len(ttft) + len(itl) + len(latencies)
        new_df = pd.DataFrame({
            "scheduler_policy": [row["scheduler_policy"]] * total_length,
            "swap_policy": [row["swap_policy"]] * total_length,
            "type": ["ttft"] * len(ttft) + ["itl"] * len(itl) +
            ["latency"] * len(latencies),
            "value":
            ttft + itl + latencies,
        })
        return new_df

    _result_df = pd.DataFrame(selected_result_for_ecdf)
    _result_df["ttfts"] = _result_df["ttfts"].apply(lambda x: x.tolist())
    _result_df["latencies"] = _result_df["latencies"].apply(
        lambda x: x.tolist())
    long_df = _result_df.apply(lambda x: explode_row(x),
                               axis=1,
                               result_type="reduce").reset_index(drop=True)
    long_df = pd.concat([long_df[_i] for _i in range(len(long_df))])
    long_df["hue"] = long_df["swap_policy"] + " " + long_df["scheduler_policy"]
    return explode_row, long_df


@app.cell
def __(long_df, plt, sns):
    colors = sns.color_palette("deep", n_colors=4)
    count = 0
    (_fig, _ax) = plt.subplots(figsize=(12, 2.5), dpi=150, ncols=3, nrows=1)
    sns.ecdfplot(
        data=long_df[long_df["type"] == "ttft"],
        x="value",
        hue="hue",
        palette=colors,
        ax=_ax[0],
        legend=False,
    )
    sns.ecdfplot(
        data=long_df[long_df["type"] == "itl"],
        x="value",
        hue="hue",
        palette=colors,
        ax=_ax[1],
        legend=False,
    )
    sns.ecdfplot(
        data=long_df[long_df["type"] == "latency"],
        x="value",
        hue="hue",
        palette=colors,
        ax=_ax[2],
    )
    _ax[0].set_title("TTFT")
    _ax[0].set_xlabel("Time")
    _ax[1].set_xlabel("Time")
    _ax[1].set_ylabel("")
    _ax[1].set_title("ITL")
    _ax[1].set_xlim(0, 0.4)
    _ax[2].set_title("Latency")
    _ax[2].get_legend().set_title("")
    plt.show()
    return colors, count


@app.cell
def __(mo):
    mo.md("""\n    # Generation TP\n""")
    return


@app.cell
def __(base_dir, os):
    _date = "20240717"
    _counters = [439]
    tp_dir_names = [
        os.path.join(base_dir, _date, str(counter)) for counter in _counters
    ]
    return tp_dir_names,


@app.cell
def __(os, pd, tp_dir_names):
    infer_result = pd.read_csv(
        os.path.join(tp_dir_names[0],
                     "1000.0qps-Llama-2-13b-chat-hf-132949-infer.csv"))
    fcfs_result = pd.read_csv(
        os.path.join(tp_dir_names[0],
                     "1000.0qps-Llama-2-13b-chat-hf-074220-fcfs.csv"))
    return fcfs_result, infer_result


@app.cell
def __(infer_result):
    infer_result[infer_result["CPU KV cache usage"] > 0].describe()
    return


@app.cell
def __(fcfs_result):
    fcfs_result[fcfs_result["CPU KV cache usage"] > 0].describe()
    return


@app.cell
def __(plt):
    value = 0.999136
    nums = []
    values = []
    for n in range(28):
        nums.append(n)
        values.append(value * ((1 + n * value**(n + 1) - (n + 1) *
                                (value**n)) / ((1 - value)**2)))
    print(values[-1])
    plt.plot(nums, values)
    plt.show()
    return n, nums, value, values


@app.cell
def __(mo):
    mo.md(r"""# Detailed Analysis""")
    return


@app.cell
def __(base_dir, os):
    _date = "20240729"
    _counters = [0, 747, 748]
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
                _detailed_result_df = pd.read_csv(
                    os.path.join(_dir_name, _file))
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
    detailed_mean_result = pd.concat([sjf_mean, fcfs_mean, tfittradeoff_mean],
                                     axis=1)
    detailed_mean_result.columns = ["SJF", "FCFS", "TFITTradeoff"]
    return detailed_mean_result,


@app.cell
def __(detailed_mean_result, plt):
    detailed_mean_result.plot(kind="bar")
    plt.yscale("log")
    plt.show()
    return


@app.cell
def __(detailed_result_dfs, plt, sns):
    swap_times = []
    swap_block_nums = []

    for df in detailed_result_dfs.values():
        swap_times.extend(df["swap time"])
        swap_block_nums.extend(df["swap out block num"] +
                               df["swap in block num"])

    plt.figure(figsize=(10, 5))

    sns.lineplot(x=swap_block_nums, y=swap_times, label="Swap Time")
    plt.title("Swap Time")
    plt.xlabel("Swap Block Num")
    plt.ylabel("Time")
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
