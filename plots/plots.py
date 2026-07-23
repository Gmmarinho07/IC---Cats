import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plots_folder = Path("plots")

plots_folder.mkdir(exist_ok=True)

df = pd.read_csv(
    "benchmark/metrics.csv"
)

df.head()

plt.figure(figsize=(12,6))

plt.bar(
    df["paper"],
    df["accuracy"]
)

plt.title("Accuracy per Paper")

plt.xlabel("Paper")

plt.ylabel("Accuracy (%)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    plots_folder / "accuracy.png",
    dpi=300
)

plt.close()

print(
    "Accuracy plot saved!"
)

def plot_precision(df, plots_folder):

    plt.figure(figsize=(12,6))

    plt.bar(
        df["paper"],
        df["precision"]
    )

    plt.title("Precision per Paper")

    plt.xlabel("Paper")

    plt.ylabel("Precision (%)")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        plots_folder / "precision.png",
        dpi=300
    )

    plt.close()

    print("Precision plot saved!")


def plot_recall(df, plots_folder):

    plt.figure(figsize=(12,6))

    plt.bar(
        df["paper"],
        df["recall"]
    )

    plt.title("Recall per Paper")

    plt.xlabel("Paper")

    plt.ylabel("Recall (%)")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        plots_folder / "recall.png",
        dpi=300
    )

    plt.close()

    print("Recall plot saved!")

def plot_f1(df, plots_folder):

    plt.figure(figsize=(12,6))

    plt.bar(
        df["paper"],
        df["f1"]
    )

    plt.title("F1 Score per Paper")

    plt.xlabel("Paper")

    plt.ylabel("F1 Score (%)")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        plots_folder / "f1.png",
        dpi=300
    )

    plt.close()

    print("F1 plot saved!")

def plot_confusion(df, plots_folder):

    plt.figure(figsize=(14,6))

    x = range(len(df))

    width = 0.25

    plt.bar(
        [i-width for i in x],
        df["tp"],
        width,
        label="TP"
    )

    plt.bar(
        x,
        df["fp"],
        width,
        label="FP"
    )

    plt.bar(
        [i+width for i in x],
        df["fn"],
        width,
        label="FN"
    )

    plt.xticks(
        x,
        df["paper"],
        rotation=90
    )

    plt.ylabel("Count")

    plt.title("TP / FP / FN per Paper")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plots_folder / "tp_fp_fn.png",
        dpi=300
    )

    plt.close()

    print("Confusion plot saved!")

def plot_histogram(df, plots_folder):

    plt.figure(figsize=(8,6))

    plt.hist(
        df["f1"],
        bins=10
    )

    plt.xlabel("F1 Score")

    plt.ylabel("Number of Papers")

    plt.title("Distribution of F1 Scores")

    plt.tight_layout()

    plt.savefig(
        plots_folder / "histogram_f1.png",
        dpi=300
    )

    plt.close()

    print("Histogram saved!")

def plot_boxplot(df, plots_folder):

    plt.figure(figsize=(8,6))

    plt.boxplot([
        df["accuracy"],
        df["precision"],
        df["recall"],
        df["f1"]
    ])

    plt.xticks(
        [1,2,3,4],
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    )

    plt.ylabel("%")

    plt.title("Metrics Distribution")

    plt.tight_layout()

    plt.savefig(
        plots_folder / "metrics_boxplot.png",
        dpi=300
    )

    plt.close()

    print("Boxplot saved!")

summary_df = pd.read_csv(
    "benchmark/summary.csv"
)

def plot_radar(summary_df, plots_folder):

    accuracy = float(
        summary_df.loc[
            summary_df["Metric"] == "Micro Accuracy",
            "Value"
        ].values[0]
    )

    precision = float(
        summary_df.loc[
            summary_df["Metric"] == "Micro Precision",
            "Value"
        ].values[0]
    )

    recall = float(
        summary_df.loc[
            summary_df["Metric"] == "Micro Recall",
            "Value"
        ].values[0]
    )

    f1 = float(
        summary_df.loc[
            summary_df["Metric"] == "Micro F1",
            "Value"
        ].values[0]
    )

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    values = [
        accuracy,
        precision,
        recall,
        f1
    ]

    values += values[:1]

    angles = np.linspace(
        0,
        2*np.pi,
        len(labels),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    plt.figure(figsize=(7,7))

    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values,
        linewidth=2
    )

    ax.fill(
        angles,
        values,
        alpha=0.25
    )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        100
    )

    plt.title(
        "Overall Benchmark Performance"
    )

    plt.savefig(
        plots_folder/"radar_chart.png",
        dpi=300
    )

    plt.close()

    print("Radar chart saved!")



plot_precision(df, plots_folder)

plot_recall(df, plots_folder)

plot_f1(df, plots_folder)

plot_confusion(df, plots_folder)

plot_histogram(df, plots_folder)

plot_boxplot(df, plots_folder)

plot_radar(
    summary_df,
    plots_folder
)
,