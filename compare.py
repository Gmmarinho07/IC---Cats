import json
import csv
from pathlib import Path

from evaluation.comparator import compare


# =====================================================
# LOAD PREDICTIONS
# =====================================================

results_folder = Path("benchmark/results")

predictions = []

for file in sorted(results_folder.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:

        data = json.load(f)

    predictions.append(
        {
            "paper": file.stem,
            "catalysts": data.get("catalysts", [])
        }
    )

print(f"\nLoaded {len(predictions)} prediction files.")


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

summary, results = compare(
    predictions,
    ground_truth
)

print(f"Compared {len(results)} papers successfully.")


# =====================================================
# SAVE JSON
# =====================================================

comparison = {
    "summary": summary,
    "results": results
}

with open(
    "benchmark/comparison_results.json",
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

with open(
    "benchmark/metrics.csv",
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

with open(
    "benchmark/summary.csv",
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
        summary,
        indent=4,
        ensure_ascii=False
    )
)

print("\nGenerated files:")

print("benchmark/comparison_results.json")
print("benchmark/metrics.csv")
print("benchmark/summary.csv")