import os
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score

sns.set_theme(style="whitegrid")

COMBINED_RESULTS_DIR = "./analysis_results_combined"
os.makedirs(COMBINED_RESULTS_DIR, exist_ok=True)


# ==========================================
# HELPER & PLOTTING FUNCTIONS
# ==========================================
def shorten_category(cat):
    """Shortens category names to save horizontal space on the Y-axis."""
    mapping = {
        "Completely subjective": "Subjective",
        "Question with unknown answer": "Unknown Answer",
        "No scientific consensus": "No Consensus",
        "Too many variables": "Many Variables"
    }
    return mapping.get(cat, cat)


def fmt_val(v):
    """Helper to format percentages: strips .0 if the value is exactly 100%."""
    if pd.isna(v): return "N/A"
    return "100" if round(v, 1) == 100.0 else f"{v:.1f}"


def fmt_pct(num, den):
    """Formats numbers into a two-line string for standard tables."""
    if pd.isna(den) or den == 0:
        return "N/A"
    pct = (num / den) * 100
    return f"{fmt_val(pct)}%\n({int(num)}/{int(den)})"


def plot_combined_heatmap(model_dfs, title, filename, cmap="Reds", vmin=0, vmax=100, annot_fontsize=16):
    """
    Plots a consolidated multi-model figure.
    """
    valid_models = [(name, ndf, adf) for name, ndf, adf in model_dfs if not ndf.empty]
    n_models = len(valid_models)
    if n_models == 0:
        return

    n_rows = len(valid_models[0][1])
    # Multiplier (1.35) comfortably fits 3 lines of text per cell
    dynamic_height = max(9.0, n_rows * 1.35)

    fig, axes = plt.subplots(
        1, n_models,
        # 5.5 width multiplier keeps cells narrow while still fitting the text
        # figsize=(6.8 * n_models, dynamic_height) #use for attack
        figsize = (5.8 * n_models, dynamic_height),
        sharey=True,
        layout="constrained"
    )
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, num_df, annot_df) in enumerate(valid_models):
        ax = axes[idx]

        sns.heatmap(
            num_df.astype(float),
            annot=annot_df.values,
            fmt="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": annot_fontsize, "fontweight": "normal", "linespacing": 1.3},
            linewidths=0.75,
            linecolor="white"
        )

        # Draw a thick horizontal line before the OVERALL summary row
        if "OVERALL" in num_df.index:
            hline_pos = num_df.index.get_loc("OVERALL")
            ax.axhline(hline_pos, color='black', linewidth=3.5)

        # Subplot Titles & Axis Labels
        ax.set_title(f"Model: {model_name}", fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel("Model Size / Scale", fontsize=16, fontweight="bold", labelpad=10)

        if idx == 0:
            #ax.set_ylabel("Target Category / Summary", fontsize=16, fontweight="bold", labelpad=10)
            ax.tick_params(axis="y", labelsize=16)
            # Rotate y-axis labels 45 degrees and align them to the right to save horizontal space
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", labelsize=16)

    save_path = os.path.join(COMBINED_RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f">> Saved consolidated heatmap: {save_path}")


# ==========================================
# ADVANCED TABLE GENERATORS
# ==========================================
def generate_rate_table(df, group_cols, unstack_col, target_col, col_order):
    """Standard table generator with num/den."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if not group_cols:
        df_temp = df.assign(Group="Overall")
        grp_cols = ["Group"]
    else:
        df_temp = df
        grp_cols = group_cols

    sum_df = df_temp.groupby(grp_cols + [unstack_col])[target_col].sum().unstack().reindex(columns=col_order)
    count_df = df_temp.groupby(grp_cols + [unstack_col])[target_col].count().unstack().reindex(columns=col_order)

    num_df = (sum_df / count_df) * 100
    annot_df = pd.DataFrame(index=num_df.index, columns=num_df.columns)

    for c in num_df.columns:
        for i in num_df.index:
            s_val = sum_df.loc[i, c] if (c in sum_df.columns and i in sum_df.index) else 0
            c_val = count_df.loc[i, c] if (c in count_df.columns and i in count_df.index) else 0
            annot_df.loc[i, c] = fmt_pct(s_val, c_val)

    return num_df, annot_df


def generate_advanced_rate_table(df, unstack_col, target_col, col_order):
    """
    Advanced table generator specifically for Student Hallucination Rates (Exp A).
    Uses ultra-compact prefixes (E/S) to save space.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    langs = sorted(df['lang'].dropna().unique())
    categories = sorted(df['category'].dropna().unique())

    cat_overall = df.groupby(["category", unstack_col])[target_col].mean() * 100
    cat_lang = df.groupby(["category", "lang", unstack_col])[target_col].mean() * 100
    overall_all = df.groupby([unstack_col])[target_col].mean() * 100
    lang_all = df.groupby(["lang", unstack_col])[target_col].mean() * 100

    deviations = []
    for cat in categories:
        for col in col_order:
            ov = cat_overall.get((cat, col), np.nan)
            for l in langs:
                lv = cat_lang.get((cat, l, col), np.nan)
                if pd.notna(ov) and pd.notna(lv):
                    deviations.append(abs(lv - ov))

    avg_dev = np.mean(deviations) if deviations else 0

    index_order = categories + ["OVERALL"]
    num_df = pd.DataFrame(index=index_order, columns=col_order, dtype=float)
    annot_df = pd.DataFrame(index=index_order, columns=col_order, dtype=object)

    for col in col_order:
        for cat in categories:
            ov = cat_overall.get((cat, col), np.nan)
            num_df.loc[cat, col] = ov

            if pd.isna(ov):
                annot_df.loc[cat, col] = "N/A"
            else:
                lines = [f"{fmt_val(ov)}%"]

                for l in langs:
                    lv = cat_lang.get((cat, l, col), np.nan)
                    prefix = "E" if l == "en" else ("S" if l == "sl" else l)

                    if pd.isna(lv):
                        lines.append(f"{prefix} N/A")
                    else:
                        if abs(lv - ov) > avg_dev:
                            lines.append(f"$\\mathbf{{{prefix}\\ {fmt_val(lv)}\\%}}$")
                        else:
                            lines.append(f"{prefix} {fmt_val(lv)}%")

                annot_df.loc[cat, col] = "\n".join(lines)

        ov_all = overall_all.get(col, np.nan)
        num_df.loc["OVERALL", col] = ov_all

        if pd.isna(ov_all):
            annot_df.loc["OVERALL", col] = "N/A"
        else:
            lines = [f"$\\mathit{{{fmt_val(ov_all)}\\%}}$"]
            for l in langs:
                lv_all = lang_all.get((l, col), np.nan)
                prefix = "E" if l == "en" else ("S" if l == "sl" else l)

                if pd.isna(lv_all):
                    lines.append(f"{prefix} N/A")
                else:
                    lines.append(f"{prefix} {fmt_val(lv_all)}%")
            annot_df.loc["OVERALL", col] = "\n".join(lines)

    return num_df, annot_df


def generate_exp_b_advanced_table(df, unstack_col, target_col, valid_col, col_order):
    """
    Advanced table generator for Exp B (Attack Strength + Adherence).
    3-line stacked layout with E/S prefixes.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    langs = sorted(df['lang'].dropna().unique())
    categories = sorted(df['category'].dropna().unique())

    atk_cat_overall = df.groupby(["category", unstack_col])[target_col].mean() * 100
    atk_cat_lang = df.groupby(["category", "lang", unstack_col])[target_col].mean() * 100
    atk_overall_all = df.groupby([unstack_col])[target_col].mean() * 100
    atk_lang_all = df.groupby(["lang", unstack_col])[target_col].mean() * 100

    val_cat_overall = df.groupby(["category", unstack_col])[valid_col].mean() * 100
    val_cat_lang = df.groupby(["category", "lang", unstack_col])[valid_col].mean() * 100
    val_overall_all = df.groupby([unstack_col])[valid_col].mean() * 100
    val_lang_all = df.groupby(["lang", unstack_col])[valid_col].mean() * 100

    deviations = []
    for cat in categories:
        for col in col_order:
            ov = atk_cat_overall.get((cat, col), np.nan)
            for l in langs:
                lv = atk_cat_lang.get((cat, l, col), np.nan)
                if pd.notna(ov) and pd.notna(lv):
                    deviations.append(abs(lv - ov))
    avg_dev = np.mean(deviations) if deviations else 0

    index_order = categories + ["OVERALL"]
    num_df = pd.DataFrame(index=index_order, columns=col_order, dtype=float)
    annot_df = pd.DataFrame(index=index_order, columns=col_order, dtype=object)

    for col in col_order:
        for cat in categories:
            atk_ov = atk_cat_overall.get((cat, col), np.nan)
            val_ov = val_cat_overall.get((cat, col), np.nan)
            num_df.loc[cat, col] = atk_ov

            if pd.isna(atk_ov):
                annot_df.loc[cat, col] = "N/A"
            else:
                lines = [f"{fmt_val(atk_ov)}% ({fmt_val(val_ov)}%)"]

                for l in langs:
                    atk_lv = atk_cat_lang.get((cat, l, col), np.nan)
                    val_lv = val_cat_lang.get((cat, l, col), np.nan)
                    prefix = "E" if l == "en" else ("S" if l == "sl" else l)

                    if pd.isna(atk_lv):
                        lines.append(f"{prefix} N/A")
                    else:
                        v_str = f" ({fmt_val(val_lv)}%)"
                        if abs(atk_lv - atk_ov) > avg_dev:
                            lines.append(f"$\\mathbf{{{prefix}\\ {fmt_val(atk_lv)}\\%}}${v_str}")
                        else:
                            lines.append(f"{prefix} {fmt_val(atk_lv)}%{v_str}")

                annot_df.loc[cat, col] = "\n".join(lines)

        atk_ov_all = atk_overall_all.get(col, np.nan)
        val_ov_all = val_overall_all.get(col, np.nan)
        num_df.loc["OVERALL", col] = atk_ov_all

        if pd.isna(atk_ov_all):
            annot_df.loc["OVERALL", col] = "N/A"
        else:
            lines = [f"$\\mathit{{{fmt_val(atk_ov_all)}\\%}}$ ({fmt_val(val_ov_all)}%)"]
            for l in langs:
                atk_lv_all = atk_lang_all.get((l, col), np.nan)
                val_lv_all = val_lang_all.get((l, col), np.nan)
                prefix = "E" if l == "en" else ("S" if l == "sl" else l)

                if pd.isna(atk_lv_all):
                    lines.append(f"{prefix} N/A")
                else:
                    lines.append(f"{prefix} {fmt_val(atk_lv_all)}% ({fmt_val(val_lv_all)}%)")
            annot_df.loc["OVERALL", col] = "\n".join(lines)

    return num_df, annot_df


def generate_exp_c_advanced_table(pivot_df, size, target_col_prefix="has_error"):
    """
    Advanced table generator for Exp C (Judge Agreement).
    Shows Match Rate % and Cohen's Kappa (K: 0.xx) in stacked format.
    """
    if pivot_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    langs = sorted(pivot_df['lang'].dropna().unique())
    categories = sorted(pivot_df['category'].dropna().unique())

    gt_j = size[-1]
    eval_judges = size[:-1]
    if not eval_judges: return pd.DataFrame(), pd.DataFrame()

    mr_dict, k_dict = {}, {}

    for cat in categories + ["OVERALL"]:
        for j in eval_judges:
            for l in [None] + langs:
                if cat == "OVERALL":
                    mask = pd.Series(True, index=pivot_df.index) if l is None else pivot_df["lang"] == l
                else:
                    mask = pivot_df["category"] == cat if l is None else (pivot_df["category"] == cat) & (
                                pivot_df["lang"] == l)

                y_true = pivot_df[mask][f"{target_col_prefix}_J{gt_j}"].astype(bool)
                y_pred = pivot_df[mask][f"{target_col_prefix}_J{j}"].astype(bool)

                if len(y_true) > 0:
                    mr_dict[(cat, l, j)] = np.mean(y_true == y_pred) * 100
                    kappa = cohen_kappa_score(y_true, y_pred)
                    k_dict[(cat, l, j)] = 0.0 if pd.isna(kappa) else kappa

    deviations = []
    for cat in categories:
        for j in eval_judges:
            ov = mr_dict.get((cat, None, j), np.nan)
            for l in langs:
                lv = mr_dict.get((cat, l, j), np.nan)
                if pd.notna(ov) and pd.notna(lv):
                    deviations.append(abs(lv - ov))
    avg_dev = np.mean(deviations) if deviations else 0

    index_order = categories + ["OVERALL"]
    num_df = pd.DataFrame(index=index_order, columns=eval_judges, dtype=float)
    annot_df = pd.DataFrame(index=index_order, columns=eval_judges, dtype=object)

    for j in eval_judges:
        for cat in categories:
            ov_mr = mr_dict.get((cat, None, j))
            ov_k = k_dict.get((cat, None, j))
            num_df.loc[cat, j] = ov_mr

            if ov_mr is None:
                annot_df.loc[cat, j] = "N/A"
            else:
                lines = [f"{fmt_val(ov_mr)}% (K: {ov_k:.2f})"]
                for l in langs:
                    l_mr = mr_dict.get((cat, l, j))
                    l_k = k_dict.get((cat, l, j))
                    prefix = "E" if l == "en" else ("S" if l == "sl" else l)

                    if l_mr is None:
                        lines.append(f"{prefix} N/A")
                    else:
                        k_str = f" (K: {l_k:.2f})"
                        if abs(l_mr - ov_mr) > avg_dev:
                            lines.append(f"$\\mathbf{{{prefix}\\ {fmt_val(l_mr)}\\%}}${k_str}")
                        else:
                            lines.append(f"{prefix} {fmt_val(l_mr)}%{k_str}")
                annot_df.loc[cat, j] = "\n".join(lines)

        ov_mr_all = mr_dict.get(("OVERALL", None, j))
        ov_k_all = k_dict.get(("OVERALL", None, j))
        num_df.loc["OVERALL", j] = ov_mr_all

        if ov_mr_all is None:
            annot_df.loc["OVERALL", j] = "N/A"
        else:
            lines = [f"$\\mathit{{{fmt_val(ov_mr_all)}\\%}}$ (K: {ov_k_all:.2f})"]
            for l in langs:
                l_mr_all = mr_dict.get(("OVERALL", l, j))
                l_k_all = k_dict.get(("OVERALL", l, j))
                prefix = "E" if l == "en" else ("S" if l == "sl" else l)

                if l_mr_all is None:
                    lines.append(f"{prefix} N/A")
                else:
                    lines.append(f"{prefix} {fmt_val(l_mr_all)}% (K: {l_k_all:.2f})")
            annot_df.loc["OVERALL", j] = "\n".join(lines)

    return num_df, annot_df


def generate_multiturn_advanced_table(df, turn_err_cols, valid_filter, table_type="standard"):
    """
    Advanced table generator for Multi-turn Degradation.
    table_type: "standard" outputs normal rate. "conditional" outputs P(Error_T | Error_{T-1}).
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    langs = sorted(df['lang'].dropna().unique())
    categories = sorted(df['category'].dropna().unique())

    # Determine which columns to process based on table type
    if table_type == "conditional":
        # Drop Turn 1 since there is no prior turn
        cols_to_process = turn_err_cols[1:]
    else:
        cols_to_process = turn_err_cols

    data_cat_overall = {}
    data_cat_lang = {}
    data_overall_all = {}
    data_lang_all = {}

    for i, col in enumerate(cols_to_process):
        v_col = col.replace("_error", "_valid")
        col_df = df if valid_filter is None else df[df[v_col] == valid_filter].copy()

        if table_type == "conditional":
            # Find the original index of this column in the full list
            orig_idx = turn_err_cols.index(col)
            prev_col = turn_err_cols[orig_idx - 1]
            calc_col = f"{col}_cond"
            col_df[calc_col] = np.where(col_df[prev_col] == True, col_df[col].astype(float), np.nan)
        else:
            calc_col = col

        data_cat_overall[col] = col_df.groupby("category")[calc_col].mean() * 100
        data_cat_lang[col] = col_df.groupby(["category", "lang"])[calc_col].mean() * 100
        data_overall_all[col] = col_df[calc_col].mean() * 100
        data_lang_all[col] = col_df.groupby("lang")[calc_col].mean() * 100

    clean_cols = [c.replace("_error", "").replace("_", " ") for c in cols_to_process]

    deviations = []
    for col in cols_to_process:
        for cat in categories:
            ov = data_cat_overall[col].get(cat, np.nan)
            for l in langs:
                lv = data_cat_lang[col].get((cat, l), np.nan)
                if pd.notna(ov) and pd.notna(lv):
                    deviations.append(abs(lv - ov))
    avg_dev = np.mean(deviations) if deviations else 0

    index_order = categories + ["OVERALL"]
    num_df = pd.DataFrame(index=index_order, columns=clean_cols, dtype=float)
    annot_df = pd.DataFrame(index=index_order, columns=clean_cols, dtype=object)

    for col, clean_col in zip(cols_to_process, clean_cols):
        for cat in categories:
            ov = data_cat_overall[col].get(cat, np.nan)
            num_df.loc[cat, clean_col] = ov

            if pd.isna(ov):
                annot_df.loc[cat, clean_col] = "N/A"
            else:
                lines = [f"{fmt_val(ov)}%"]
                for l in langs:
                    lv = data_cat_lang[col].get((cat, l), np.nan)
                    prefix = "E" if l == "en" else ("S" if l == "sl" else l)
                    if pd.isna(lv):
                        lines.append(f"{prefix} N/A")
                    else:
                        if abs(lv - ov) > avg_dev:
                            lines.append(f"$\\mathbf{{{prefix}\\ {fmt_val(lv)}\\%}}$")
                        else:
                            lines.append(f"{prefix} {fmt_val(lv)}%")
                annot_df.loc[cat, clean_col] = "\n".join(lines)

        ov_all = data_overall_all[col]
        num_df.loc["OVERALL", clean_col] = ov_all

        if pd.isna(ov_all):
            annot_df.loc["OVERALL", clean_col] = "N/A"
        else:
            lines = [f"$\\mathit{{{fmt_val(ov_all)}\\%}}$"]
            for l in langs:
                lv_all = data_lang_all[col].get(l, np.nan)
                prefix = "E" if l == "en" else ("S" if l == "sl" else l)
                if pd.isna(lv_all):
                    lines.append(f"{prefix} N/A")
                else:
                    lines.append(f"{prefix} {fmt_val(lv_all)}%")
            annot_df.loc["OVERALL", clean_col] = "\n".join(lines)

    return num_df, annot_df


# ==========================================
# DATA LOADING
# ==========================================
def load_phase3_data(output_dir, model_name):
    files = glob.glob(os.path.join(output_dir, "judged_G*_S*_J*.jsonl"))
    data = []
    for file in files:
        basename = os.path.basename(file).replace(".jsonl", "")
        parts = basename.split("_")
        G, S, J = parts[1][1:], parts[2][1:], parts[3][1:]

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    eval_data = row.get("eval_t1", {})
                    data.append({
                        "id": row["id"],
                        "family": model_name,
                        "generator": G,
                        "student": S,
                        "judge": J,
                        "category": shorten_category(row["target_category"]),
                        "lang": row["lang"],
                        "question": row["question_t1"],
                        "q_length_words": len(row["question_t1"].split()),
                        "is_valid": bool(eval_data.get("is_question_valid", False)),
                        "has_error": bool(eval_data.get("has_error", False)),
                    })
                except Exception:
                    continue
    return pd.DataFrame(data)


def load_multiturn_data(output_dir, model_name, target_size):
    selected_file = None
    for turn in [4, 3, 2]:
        candidate = os.path.join(output_dir, f"multiturn_{target_size}_T{turn}.jsonl")
        if os.path.exists(candidate):
            selected_file = candidate
            break

    if not selected_file:
        pattern = os.path.join(output_dir, "*multiturn*.jsonl")
        candidates = glob.glob(pattern)
        if candidates:
            def turn_rank(f):
                match = re.search(r"T(\d+)", f, re.IGNORECASE)
                return int(match.group(1)) if match else 0

            candidates.sort(key=turn_rank, reverse=True)
            selected_file = candidates[0]

    if not selected_file:
        return pd.DataFrame()

    data = []
    with open(selected_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                history = row.get("turn_history", [])
                turns = {
                    "id": row["id"],
                    "family": model_name,
                    "category": shorten_category(row["target_category"]),
                    "lang": row["lang"]
                }
                for i, t_data in enumerate(history):
                    eval_data = t_data.get("eval", {})
                    turns[f"Turn_{i + 1}_valid"] = bool(eval_data.get("is_question_valid", False))
                    turns[f"Turn_{i + 1}_error"] = bool(eval_data.get("has_error", False))
                data.append(turns)
            except Exception:
                continue
    return pd.DataFrame(data)


# ==========================================
# EXPERIMENT RUNNERS
# ==========================================
def run_experiment_a_all_models(all_data):
    modes = [("All_Data", None), ("Valid_Only", True)]
    for mode_name, valid_filter in modes:
        print("\n" + "=" * 90)
        print(f"EXPERIMENT A: STUDENT VULNERABILITY [{mode_name.replace('_', ' ').upper()}]")
        print("=" * 90)

        combined_panels = []
        for model_name, df, size in all_data:
            subset = df[(df["generator"] == size[-1]) & (df["judge"] == size[-1])].copy()
            if valid_filter is not None:
                subset = subset[subset["is_valid"] == valid_filter]
            if subset.empty: continue

            print(f"\n>>> Model Family: {model_name} ({mode_name}) <<<")
            num_df, annot_df = generate_advanced_rate_table(subset, "student", "has_error", size)
            print(annot_df.to_markdown())
            combined_panels.append((model_name, num_df, annot_df))

        plot_combined_heatmap(
            combined_panels,
            title="",
            filename=f"ExpA_Student_Vulnerability_{mode_name}.png",
            cmap="Reds",
            annot_fontsize=18
        )


def run_experiment_b_all_models(all_data):
    modes = [("All_Data", None), ("Valid_Only", True)]
    for mode_name, valid_filter in modes:
        print("\n" + "=" * 90)
        print(f"EXPERIMENT B: PROBING / ATTACK STRENGTH [{mode_name.replace('_', ' ').upper()}]")
        print("=" * 90)

        combined_panels = []
        for model_name, df, size in all_data:
            subset = df[(df["student"] == size[-1]) & (df["judge"] == size[-1])].copy()
            if valid_filter is not None:
                subset = subset[subset["is_valid"] == valid_filter]
            if subset.empty: continue

            print(f"\n>>> Model Family: {model_name} ({mode_name}) <<<")
            num_df, annot_df = generate_exp_b_advanced_table(
                subset, "generator", "has_error", "is_valid", size
            )
            print(annot_df.to_markdown())
            combined_panels.append((model_name, num_df, annot_df))

        plot_combined_heatmap(
            combined_panels,
            title="",
            filename=f"ExpB_Attack_Strength_{mode_name}.png",
            cmap="Reds",
            annot_fontsize=16
        )


def run_experiment_c_all_models(all_data):
    modes = [("All_Data", None), ("Valid_Only", True)]
    for mode_name, valid_filter in modes:
        print("\n" + "=" * 90)
        print(f"EXPERIMENT C: JUDGE AGREEMENT [{mode_name.replace('_', ' ').upper()}]")
        print("=" * 90)

        combined_panels = []
        for model_name, df, size in all_data:
            subset = df[(df["generator"] == size[-1]) & (df["student"] == size[-1])].copy()
            if subset.empty: continue

            pivot = subset.pivot(index="id", columns="judge", values=["has_error", "is_valid"])
            pivot.columns = [f"{col[0]}_J{col[1]}" for col in pivot.columns]
            pivot = pivot.reset_index().merge(subset[['id', 'category', 'lang']].drop_duplicates(), on='id').dropna()

            if valid_filter is not None:
                pivot = pivot[pivot[f"is_valid_J{size[-1]}"] == valid_filter]
            if pivot.empty: continue

            print(f"\n>>> Model Family: {model_name} ({mode_name}) <<<")
            num_df, annot_df = generate_exp_c_advanced_table(pivot, size, target_col_prefix="has_error")
            print(annot_df.to_markdown())

            if not num_df.empty:
                combined_panels.append((model_name, num_df, annot_df))

        plot_combined_heatmap(
            combined_panels,
            title="",
            filename=f"ExpC_Judge_Agreement_{mode_name}.png",
            cmap="Greens",
            vmin=0,
            vmax=100,
            annot_fontsize=16
        )


def run_multiturn_all_models(all_mt_data):
    """
    MULTI-TURN DEGRADATION:
    Outputs two distinct heatmaps per mode:
    1. Standard Hallucination Rate
    2. Conditional Hallucination Rate P(Error | Prior Error)
    """
    modes = [("All_Data", None), ("Valid_Only", True)]
    for mode_name, valid_filter in modes:
        print("\n" + "=" * 90)
        print(f"MULTI-TURN DEGRADATION [{mode_name.replace('_', ' ').upper()}]")
        print("=" * 90)

        # --- 1. Standard Hallucination Rate ---
        combined_panels_standard = []
        for model_name, mt_df, _ in all_mt_data:
            if mt_df.empty: continue
            turn_err_cols = sorted([c for c in mt_df.columns if c.endswith("_error")])

            print(f"\n>>> Model Family: {model_name} (Standard - {mode_name}) <<<")
            num_df, annot_df = generate_multiturn_advanced_table(
                mt_df.copy(), turn_err_cols, valid_filter, table_type="standard"
            )
            print(annot_df.to_markdown())
            if not num_df.empty:
                combined_panels_standard.append((model_name, num_df, annot_df))

        plot_combined_heatmap(
            combined_panels_standard,
            title="",
            filename=f"MultiTurn_Degradation_Standard_{mode_name}.png",
            cmap="Oranges",
            annot_fontsize=16
        )

        # --- 2. Conditional Hallucination Rate ---
        combined_panels_cond = []
        for model_name, mt_df, _ in all_mt_data:
            if mt_df.empty: continue
            turn_err_cols = sorted([c for c in mt_df.columns if c.endswith("_error")])

            print(f"\n>>> Model Family: {model_name} (Conditional - {mode_name}) <<<")
            num_df, annot_df = generate_multiturn_advanced_table(
                mt_df.copy(), turn_err_cols, valid_filter, table_type="conditional"
            )
            print(annot_df.to_markdown())
            if not num_df.empty:
                combined_panels_cond.append((model_name, num_df, annot_df))

        plot_combined_heatmap(
            combined_panels_cond,
            title="",
            filename=f"MultiTurn_Degradation_Conditional_{mode_name}.png",
            cmap="Oranges",
            annot_fontsize=16
        )


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_CONFIGS = [
        {"name": "Qwen", "dir": "./hallucination_datasets_qwen", "size": ["8B", "14B", "32B"]},
        {"name": "Qwen Thinking", "dir": "./hallucination_datasets_qwen_thinking", "size": ["8B", "14B", "32B"]},
        {"name": "Gemma", "dir": "./hallucination_datasets_gemma", "size": ["4B", "12B", "27B"]},
    ]

    all_phase3_data = []
    all_multiturn_data = []

    print("Loading data across all models...")
    for cfg in MODEL_CONFIGS:
        df_p3 = load_phase3_data(cfg["dir"], cfg["name"])
        df_mt = load_multiturn_data(cfg["dir"], cfg["name"], target_size=cfg["size"][-1])
        if not df_p3.empty: all_phase3_data.append((cfg["name"], df_p3, cfg["size"]))
        if not df_mt.empty: all_multiturn_data.append((cfg["name"], df_mt, cfg["size"]))

    # 1. Experiment A
    run_experiment_a_all_models(all_phase3_data)

    # 2. Experiment B
    run_experiment_b_all_models(all_phase3_data)

    # 3. Experiment C
    run_experiment_c_all_models(all_phase3_data)

    # 4. Multi-Turn Degradation
    run_multiturn_all_models(all_multiturn_data)

    print(f"\nAll analyses complete! Consolidated multi-model heatmaps saved to: '{COMBINED_RESULTS_DIR}/'")
