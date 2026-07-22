"""
evaluation/comparator.py

Comparator V2
"""

from rapidfuzz import fuzz

SIMILARITY_THRESHOLD = 80


def normalize(text):
    if text is None:
        return ""
    return (
        str(text)
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace(" ", "")
        .strip()
    )


def extract_gt_names(gt):
    names = []

    for item in gt.get("catalysts", []):

        if isinstance(item, str):
            names.append(item)

        elif isinstance(item, dict):

            if item.get("catalyst"):
                names.append(item["catalyst"])

            elif item.get("normalized_name"):
                names.append(item["normalized_name"])

            elif item.get("name"):
                names.append(item["name"])

    return names


def best_match(pred, gt_list, used):

    best_index = None
    best_score = -1

    for i, gt in enumerate(gt_list):

        if i in used:
            continue

        score = fuzz.ratio(
            normalize(pred),
            normalize(gt)
        )

        if score > best_score:
            best_score = score
            best_index = i

    return best_index, best_score


def compare(predictions, ground_truth):

    gt_dict = {
        normalize(item["paper"]): item
        for item in ground_truth
    }

    total_tp = total_fp = total_fn = 0
    results = []

    for pred in predictions:

        paper = normalize(pred["paper"])

        if paper not in gt_dict:
            print(f"Skipping {pred['paper']} (no ground truth)")
            continue

        gt = gt_dict[paper]

        pred_names = pred.get("catalysts", [])
        gt_names = extract_gt_names(gt)

        used = set()
        tp = 0

        for p in pred_names:

            idx, score = best_match(p, gt_names, used)

            if idx is not None and score >= SIMILARITY_THRESHOLD:
                tp += 1
                used.add(idx)

        fp = max(0, len(pred_names) - tp)
        fn = max(0, len(gt_names) - tp)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        accuracy = tp / len(gt_names) if gt_names else 0

        results.append({
            "paper": pred["paper"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2)
        })

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    total_gt = total_tp + total_fn
    accuracy = total_tp / total_gt if total_gt else 0

    summary = {
        "total_papers": len(results),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "catalyst": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2)
        }
    }

    return summary, results
