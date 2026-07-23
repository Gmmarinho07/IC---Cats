"""
evaluation/comparator.py

Comparator V3

Responsabilidades:
- Normalização
- Similaridade
- Matching
- Cálculo de métricas

Não gera CSV nem gráficos.
"""

from rapidfuzz import fuzz


# =====================================================
# CONFIGURATION
# =====================================================

SIMILARITY_THRESHOLD = 80


# =====================================================
# NORMALIZATION
# =====================================================

def normalize(text):
    """
    Normalize catalyst names before comparison.
    """

    if text is None:
        return ""

    text = str(text).lower()

    replacements = [
        "-", "_", "/", "\\",
        "(", ")", "[", "]",
        "{", "}", ",", ".",
        ";", ":", " "
    ]

    for char in replacements:
        text = text.replace(char, "")

    return text.strip()


# =====================================================
# STRING SIMILARITY
# =====================================================

def similarity(a, b):
    """
    Uses three RapidFuzz metrics and returns
    the highest similarity score.
    """

    a = normalize(a)
    b = normalize(b)

    return max(

        fuzz.ratio(a, b),

        fuzz.partial_ratio(a, b),

        fuzz.token_set_ratio(a, b)

    )


# =====================================================
# GROUND TRUTH PARSER
# =====================================================

def extract_gt_names(gt):
    """
    Extract catalyst names from Ground Truth.

    Compatible with:

    {
        "catalysts":[
            {
                "catalyst":"..."
            }
        ]
    }
    """

    names = []

    for item in gt.get("catalysts", []):

        if isinstance(item, str):

            names.append(item)

            continue

        if not isinstance(item, dict):
            continue

        if item.get("catalyst"):

            names.append(item["catalyst"])

        elif item.get("normalized_name"):

            names.append(item["normalized_name"])

        elif item.get("name"):

            names.append(item["name"])

    return names


# =====================================================
# BEST MATCH
# =====================================================

def best_match(prediction, gt_names, used):
    """
    Finds the best unused Ground Truth catalyst.
    """

    best_index = None
    best_score = -1

    for i, gt in enumerate(gt_names):

        if i in used:
            continue

        score = similarity(
            prediction,
            gt
        )

        if score > best_score:

            best_score = score
            best_index = i

    return best_index, best_score


# =====================================================
# MATCH PREDICTIONS
# =====================================================

def match_predictions(predictions, gt_names):
    """
    Performs one-to-one matching.

    Returns:

    TP
    FP
    FN
    """

    used = set()

    tp = 0

    for pred in predictions:

        idx, score = best_match(
            pred,
            gt_names,
            used
        )

        if idx is not None and score >= SIMILARITY_THRESHOLD:

            tp += 1

            used.add(idx)

    fp = max(
        0,
        len(predictions) - tp
    )

    fn = max(
        0,
        len(gt_names) - tp
    )

    return tp, fp, fn


# =====================================================
# METRICS
# =====================================================

def calculate_metrics(tp, fp, fn):
    """
    Calculates benchmark metrics.
    """

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall)
        else 0
    )

    accuracy = (
        tp / (tp + fp + fn)
        if (tp + fp + fn)
        else 0
    )

    return {

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "precision": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall * 100,
            2
        ),

        "f1": round(
            f1 * 100,
            2
        )

    }

# =====================================================
# MAIN COMPARATOR
# =====================================================

def compare(predictions, ground_truth):
    """
    Compare predictions against the Ground Truth.
    """

    # ---------------------------------------------
    # Ground Truth Dictionary
    # ---------------------------------------------

    gt_dict = {
        normalize(item["paper"]): item
        for item in ground_truth
    }

    # ---------------------------------------------
    # Global Counters
    # ---------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # ---------------------------------------------
    # Macro Metrics
    # ---------------------------------------------

    macro_accuracy = []

    macro_precision = []

    macro_recall = []

    macro_f1 = []

    # ---------------------------------------------
    # Results
    # ---------------------------------------------

    results = []

    # ---------------------------------------------
    # Iterate over every prediction
    # ---------------------------------------------

    for prediction in predictions:

        paper = normalize(
            prediction["paper"]
        )

        if paper not in gt_dict:

            print(
                f"Skipping {prediction['paper']} "
                "(no ground truth)"
            )

            continue

        gt = gt_dict[paper]

        pred_names = prediction.get(
            "catalysts",
            []
        )

        gt_names = extract_gt_names(gt)

        tp, fp, fn = match_predictions(
            pred_names,
            gt_names
        )

        metrics = calculate_metrics(
            tp,
            fp,
            fn
        )

        # -----------------------------------------
        # Global counters
        # -----------------------------------------

        total_tp += tp

        total_fp += fp

        total_fn += fn

        # -----------------------------------------
        # Macro metrics
        # -----------------------------------------

        macro_accuracy.append(
            metrics["accuracy"]
        )

        macro_precision.append(
            metrics["precision"]
        )

        macro_recall.append(
            metrics["recall"]
        )

        macro_f1.append(
            metrics["f1"]
        )

        # -----------------------------------------
        # Save paper result
        # -----------------------------------------

        results.append({

            "paper": prediction["paper"],

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "accuracy": metrics["accuracy"],

            "precision": metrics["precision"],

            "recall": metrics["recall"],

            "f1": metrics["f1"]

        })

    # ---------------------------------------------
    # Sort by F1 (best first)
    # ---------------------------------------------

    results.sort(
        key=lambda x: x["f1"],
        reverse=True
    )

    # ---------------------------------------------
    # Micro Average
    # ---------------------------------------------

    micro_metrics = calculate_metrics(

        total_tp,

        total_fp,

        total_fn

    )

    # ---------------------------------------------
    # Macro Average
    # ---------------------------------------------

    macro_metrics = {

        "accuracy":

            round(
                sum(macro_accuracy) /
                len(macro_accuracy),
                2
            ) if macro_accuracy else 0,

        "precision":

            round(
                sum(macro_precision) /
                len(macro_precision),
                2
            ) if macro_precision else 0,

        "recall":

            round(
                sum(macro_recall) /
                len(macro_recall),
                2
            ) if macro_recall else 0,

        "f1":

            round(
                sum(macro_f1) /
                len(macro_f1),
                2
            ) if macro_f1 else 0

    }

    # ---------------------------------------------
    # Summary
    # ---------------------------------------------

    summary = {

        "total_papers": len(results),

        "similarity_threshold": SIMILARITY_THRESHOLD,

        # Global confusion matrix

        "confusion": {

            "tp": total_tp,

            "fp": total_fp,

            "fn": total_fn

        },

        # Micro Average
        # (computed from global TP/FP/FN)

        "micro_average": micro_metrics,

        # Macro Average
        # (average of paper metrics)

        "macro_average": macro_metrics

    }

    return summary, results