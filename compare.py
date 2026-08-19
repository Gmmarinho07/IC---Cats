import json
import csv
from pathlib import Path

from evaluation.comparator import compare


# =====================================================
# LOAD PREDICTIONS
# =====================================================

def load_predictions(results_folder):

    predictions = []

    results_folder = Path(results_folder)

    for file in sorted(results_folder.glob("*.json")):

        with open(file, "r", encoding="utf-8") as f:

            data = json.load(f)

        predictions.append(
            {
                "paper": file.stem,
                "catalysts": data.get("catalysts", [])
            }
        )

    print(f"Loaded {len(predictions)} prediction files from {results_folder}")

    return predictions


# =====================================================
# LOAD GROUND TRUTH
# =====================================================

with open(
    "benchmark/ground_truth.json",
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)

print(f"Loaded {len(ground_truth)} ground truth entries.")


# =====================================================
# RUN COMPARATOR
# =====================================================

gpt_predictions = load_predictions(
    "benchmark/results/gpt"
)

claude_predictions = load_predictions(
    "benchmark/results/claude"
)
gpt_summary, gpt_results = compare(
    gpt_predictions,
    ground_truth
)

claude_summary, claude_results = compare(
    claude_predictions,
    ground_truth
)

print(f"Compared {len(gpt_results)} papers successfully.")


# =====================================================
# SAVE JSON
# =====================================================

for model_name, summary, results in [

    ("gpt", gpt_summary, gpt_results),

    ("claude", claude_summary, claude_results)

]:

    comparison = {

        "summary": summary,

        "results": results

    }

    with open(

        f"benchmark/comparison_{model_name}.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            comparison,

            f,

            indent=4,

            ensure_ascii=False

        )
# =====================================================
# EXPORT METRICS CSV
# =====================================================

for model_name, results in [

    ("gpt", gpt_results),

    ("claude", claude_results)

]:

    with open(
        f"benchmark/metrics_{model_name}.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "paper",
            "tp",
            "fp",
            "fn",
            "predicted",
            "expected",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ])

        for r in results:

            writer.writerow([

                r["paper"],

                r["tp"],

                r["fp"],

                r["fn"],

                r["tp"] + r["fp"],

                r["tp"] + r["fn"],

                r["accuracy"],

                r["precision"],

                r["recall"],

                r["f1"]

            ])

# =====================================================
# EXPORT SUMMARY CSV
# =====================================================

for model_name, summary in [

    ("gpt", gpt_summary),

    ("claude", claude_summary)

]:

    with open(
        f"benchmark/summary_{model_name}.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Value"
        ])

        writer.writerow([
            "Total Papers",
            summary["total_papers"]
        ])

        writer.writerow([
            "Similarity Threshold",
            summary["similarity_threshold"]
        ])

        writer.writerow([])

        # -------------------------
        # MICRO AVERAGE
        # -------------------------

        writer.writerow([
            "Micro Accuracy",
            summary["micro_average"]["accuracy"]
        ])

        writer.writerow([
            "Micro Precision",
            summary["micro_average"]["precision"]
        ])

        writer.writerow([
            "Micro Recall",
            summary["micro_average"]["recall"]
        ])

        writer.writerow([
            "Micro F1",
            summary["micro_average"]["f1"]
        ])

        writer.writerow([])

        # -------------------------
        # MACRO AVERAGE
        # -------------------------

        writer.writerow([
            "Macro Accuracy",
            summary["macro_average"]["accuracy"]
        ])

        writer.writerow([
            "Macro Precision",
            summary["macro_average"]["precision"]
        ])

        writer.writerow([
            "Macro Recall",
            summary["macro_average"]["recall"]
        ])

        writer.writerow([
            "Macro F1",
            summary["macro_average"]["f1"]
        ])

        writer.writerow([])

        # -------------------------
        # CONFUSION
        # -------------------------

        writer.writerow([
            "True Positives",
            summary["confusion"]["tp"]
        ])

        writer.writerow([
            "False Positives",
            summary["confusion"]["fp"]
        ])

        writer.writerow([
            "False Negatives",
            summary["confusion"]["fn"]
        ])

# =====================================================
# PRINT FINAL
# =====================================================

print("\n==============================")
print("BENCHMARK COMPLETED")
print("==============================")

print(
    json.dumps(
        gpt_summary,
        indent=4,
        ensure_ascii=False
    )
)

print("\nGenerated files:")

print("benchmark/metrics_gpt.csv")
print("benchmark/metrics_claude.csv")
print("benchmark/summary_gpt.csv")
print("benchmark/summary_claude.csv")