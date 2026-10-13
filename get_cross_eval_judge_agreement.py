import json
import os
import math

try:
    from sklearn.metrics import cohen_kappa_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn is not installed. Pairwise Cohen's Kappa will be skipped.")
    print("To enable, run: pip install scikit-learn\n")

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "./data/cross_eval_merged.jsonl"


def shorten_category(cat):
    """Shortens category names to save horizontal space."""
    mapping = {
        "Completely subjective": "Subjective",
        "Question with unknown answer": "Unknown Answer",
        "No scientific consensus": "No Consensus",
        "Too many variables": "Many Variables"
    }
    return mapping.get(cat, cat)


def fleiss_kappa_binary(ratings):
    """Computes Fleiss' Kappa for binary (True/False) ratings."""
    N = len(ratings)
    n = 3  # number of raters
    if N == 0: return 0.0

    total_true = sum(sum(row) for row in ratings)
    total_false = (N * n) - total_true

    p_true = total_true / (N * n)
    p_false = total_false / (N * n)
    P_e_bar = p_true ** 2 + p_false ** 2

    P_i_sum = 0
    for row in ratings:
        n_true = sum(row)
        n_false = n - n_true
        P_i = (n_true ** 2 - n_true + n_false ** 2 - n_false) / (n * (n - 1))
        P_i_sum += P_i

    P_bar = P_i_sum / N

    if P_e_bar == 1.0: return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def calc_all_match(list1, list2, list3):
    """Calculates the percentage where all 3 judges perfectly agree."""
    if len(list1) == 0: return 0.0
    matches = sum(1 for a, b, c in zip(list1, list2, list3) if a == b == c)
    return (matches / len(list1)) * 100


def calc_pairwise(list1, list2):
    """Returns (accuracy_percentage, cohens_kappa)"""
    if len(list1) == 0: return 0.0, 0.0

    matches = sum(1 for a, b in zip(list1, list2) if a == b)
    acc = (matches / len(list1)) * 100

    if SKLEARN_AVAILABLE:
        kappa = cohen_kappa_score(list1, list2)
        if math.isnan(kappa): kappa = 0.0
    else:
        kappa = None

    return acc, kappa


def get_metrics(subset, keys):
    """Calculates Fleiss, All-3 Match, and 3 Pairwise metrics for a given subset of data."""
    k1, k2, k3 = keys
    if not subset:
        return 0.0, 0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    ratings = [(r[k1], r[k2], r[k3]) for r in subset]
    f_k = fleiss_kappa_binary(ratings)

    l1, l2, l3 = [r[k1] for r in subset], [r[k2] for r in subset], [r[k3] for r in subset]

    all_match = calc_all_match(l1, l2, l3)
    p1 = calc_pairwise(l1, l2)  # Gemma vs Qwen NT
    p2 = calc_pairwise(l1, l3)  # Gemma vs Qwen T
    p3 = calc_pairwise(l2, l3)  # Qwen NT vs Qwen T

    return f_k, all_match, p1, p2, p3


def format_pair(pair, is_highest=False, is_lowest=False):
    """Formats (Match%, Kappa) into a LaTeX safe string, bolding highest, italicizing lowest."""
    match, kappa = pair
    if kappa is None:
        s = f"{match:.1f}\\% (N/A)"
    else:
        s = f"{match:.1f}\\% ({kappa:.2f})"

    # If all three are tied (highest and lowest are the same), leave unformatted
    if is_highest and is_lowest:
        return s
    if is_highest:
        return f"\\textbf{{{s}}}"
    if is_lowest:
        return f"\\textit{{{s}}}"

    return s


def generate_combined_latex_table(title, label, data, keys_v, keys_e):
    """
    Generates a combined LaTeX table with Validity (Val.) and Error (Err.) side-by-side.
    Includes an "All 3 Match" column.
    Bolds the highest pairwise match rate and italicizes the lowest in each row.
    """
    langs = sorted(list(set(row['lang'] for row in data if row.get('lang'))))
    cats = sorted(list(set(row['category'] for row in data if row.get('category'))))

    def format_combined_row(name, m_v, m_e):
        f_v, a_v, p1_v, p2_v, p3_v = m_v
        f_e, a_e, p1_e, p2_e, p3_e = m_e

        # Find min and max match rates for Validity
        v_matches = [p1_v[0], p2_v[0], p3_v[0]]
        min_v, max_v = min(v_matches), max(v_matches)

        # Find min and max match rates for Error
        e_matches = [p1_e[0], p2_e[0], p3_e[0]]
        min_e, max_e = min(e_matches), max(e_matches)

        s_p1_v = format_pair(p1_v, is_highest=(p1_v[0] == max_v), is_lowest=(p1_v[0] == min_v))
        s_p2_v = format_pair(p2_v, is_highest=(p2_v[0] == max_v), is_lowest=(p2_v[0] == min_v))
        s_p3_v = format_pair(p3_v, is_highest=(p3_v[0] == max_v), is_lowest=(p3_v[0] == min_v))

        s_p1_e = format_pair(p1_e, is_highest=(p1_e[0] == max_e), is_lowest=(p1_e[0] == min_e))
        s_p2_e = format_pair(p2_e, is_highest=(p2_e[0] == max_e), is_lowest=(p2_e[0] == min_e))
        s_p3_e = format_pair(p3_e, is_highest=(p3_e[0] == max_e), is_lowest=(p3_e[0] == min_e))

        return (f"{name} & {f_v:.3f} & {f_e:.3f} & "
                f"{a_v:.1f}\\% & {a_e:.1f}\\% & "
                f"{s_p1_v} & {s_p1_e} & "
                f"{s_p2_v} & {s_p2_e} & "
                f"{s_p3_v} & {s_p3_e} \\\\")

    latex = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l cc cc cc cc cc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Fleiss $\\kappa$}} "
        "& \\multicolumn{2}{c}{\\textbf{All 3 Match}} "
        "& \\multicolumn{2}{c}{\\textbf{G27B vs Q32B-NT}} "
        "& \\multicolumn{2}{c}{\\textbf{G27B vs Q32B-T}} "
        "& \\multicolumn{2}{c}{\\textbf{Q32B-NT vs Q32B-T}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}",
        "\\textbf{Subset} & \\textbf{Val.} & \\textbf{Err.} & \\textbf{Val.} & \\textbf{Err.} & \\textbf{Val.} & \\textbf{Err.} & \\textbf{Val.} & \\textbf{Err.} & \\textbf{Val.} & \\textbf{Err.} \\\\",
        "\\midrule"
    ]

    # OVERALL
    m_v = get_metrics(data, keys_v)
    m_e = get_metrics(data, keys_e)
    latex.append(format_combined_row("\\textbf{OVERALL}", m_v, m_e))
    latex.append("\\midrule")

    # BY LANGUAGE
    latex.append("\\textit{By Language} & & & & & & & & & & \\\\")
    for l in langs:
        subset = [r for r in data if r.get('lang') == l]
        m_v = get_metrics(subset, keys_v)
        m_e = get_metrics(subset, keys_e)
        latex.append(format_combined_row(f"\\hspace{{1em}} {l.upper()}", m_v, m_e))
    latex.append("\\midrule")

    # BY CATEGORY
    latex.append("\\textit{By Category} & & & & & & & & & & \\\\")
    for c in cats:
        subset = [r for r in data if r.get('category') == c]
        m_v = get_metrics(subset, keys_v)
        m_e = get_metrics(subset, keys_e)
        latex.append(format_combined_row(f"\\hspace{{1em}} {c}", m_v, m_e))

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\caption{{{title}}}",
        f"\\label{{{label}}}",
        "\\end{table*}"
    ])

    return "\n".join(latex)


def generate_single_latex_table(title, label, data, keys):
    """
    Generates a single-metric LaTeX table (used for Valid-Questions Only).
    Includes an "All 3 Match" column.
    Bolds the highest and italicizes the lowest pairwise match rate in each row.
    """
    langs = sorted(list(set(row['lang'] for row in data if row.get('lang'))))
    cats = sorted(list(set(row['category'] for row in data if row.get('category'))))

    def format_row(name, m):
        f_k, a_match, p1, p2, p3 = m

        # Find min and max match rate
        matches = [p1[0], p2[0], p3[0]]
        min_m, max_m = min(matches), max(matches)

        s_p1 = format_pair(p1, is_highest=(p1[0] == max_m), is_lowest=(p1[0] == min_m))
        s_p2 = format_pair(p2, is_highest=(p2[0] == max_m), is_lowest=(p2[0] == min_m))
        s_p3 = format_pair(p3, is_highest=(p3[0] == max_m), is_lowest=(p3[0] == min_m))

        return f"{name} & {f_k:.3f} & {a_match:.1f}\\% & {s_p1} & {s_p2} & {s_p3} \\\\"

    latex = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l c c c c c}",
        "\\toprule",
        "\\textbf{Subset} & \\textbf{Fleiss $\\kappa$} & \\textbf{All 3 Match} & \\textbf{G27B vs Q32B-NT} & \\textbf{G27B vs Q32B-T} & \\textbf{Q32B-NT vs Q32B-T} \\\\",
        "\\midrule"
    ]

    # OVERALL
    m = get_metrics(data, keys)
    latex.append(format_row("\\textbf{OVERALL}", m))
    latex.append("\\midrule")

    # BY LANGUAGE
    latex.append("\\textit{By Language} & & & & & \\\\")
    for l in langs:
        subset = [r for r in data if r.get('lang') == l]
        latex.append(format_row(f"\\hspace{{1em}} {l.upper()}", get_metrics(subset, keys)))
    latex.append("\\midrule")

    # BY CATEGORY
    latex.append("\\textit{By Category} & & & & & \\\\")
    for c in cats:
        subset = [r for r in data if r.get('category') == c]
        latex.append(format_row(f"\\hspace{{1em}} {c}", get_metrics(subset, keys)))

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\caption{{{title}}}",
        f"\\label{{{label}}}",
        "\\end{table*}"
    ])

    return "\n".join(latex)


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Did you run the cross-evaluation script first?")
        return

    all_data = []
    skipped_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except:
                continue

            if not all(k in item for k in ['eval_gemma', 'eval_qwen_nothink', 'eval_qwen_think']):
                continue

            e_g = item['eval_gemma']
            e_qnt = item['eval_qwen_nothink']
            e_qt = item['eval_qwen_think']

            if (e_g.get('error_description') == "Failed to parse" or
                    e_qnt.get('error_description') == "Failed to parse" or
                    e_qt.get('error_description') == "Failed to parse"):
                skipped_count += 1
                continue

            row = {
                'id': item.get('id'),
                'lang': item.get('lang', 'Unknown'),
                'category': shorten_category(item.get('target_category', 'Unknown')),

                'v_g': bool(e_g.get('is_question_valid', False)),
                'v_qnt': bool(e_qnt.get('is_question_valid', False)),
                'v_qt': bool(e_qt.get('is_question_valid', False)),

                'err_g': bool(e_g.get('has_error', False)),
                'err_qnt': bool(e_qnt.get('has_error', False)),
                'err_qt': bool(e_qt.get('has_error', False)),
            }
            all_data.append(row)

    print("=" * 70)
    print(f"INTER-RATER AGREEMENT: GENERATING LATEX TABLES")
    print(f"Successfully processed records : {len(all_data)}")
    print(f"Records skipped (JSON errors)  : {skipped_count}")
    print("Note: Make sure to include \\usepackage{booktabs} in your LaTeX preamble.")
    print("=" * 70)
    print("\nCopy and paste the LaTeX blocks below into your paper.\n")

    # ---------------------------------------------------------
    # TABLE 1: Combined Validity & Error (All Data)
    # ---------------------------------------------------------
    keys_v = ('v_g', 'v_qnt', 'v_qt')
    keys_e = ('err_g', 'err_qnt', 'err_qt')

    table1 = generate_combined_latex_table(
        title="Inter-Rater Agreement: Question Validity (Val.) vs Hallucination Detection (Err.). Bolded values indicate the highest pairwise agreement in each row, while italicized values indicate the lowest.",
        label="tab:agreement_combined_all",
        data=all_data,
        keys_v=keys_v,
        keys_e=keys_e
    )
    print(table1)
    print("\n" + "% " + "-" * 60 + "\n")


if __name__ == "__main__":
    main()