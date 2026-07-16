# =====================================================
# ACCURACY
# =====================================================

def accuracy(hits, total):

    if total == 0:
        return 0.0

    return round(
        hits / total * 100,
        2
    )


# =====================================================
# PRECISION
# =====================================================

def precision(tp, fp):

    if tp + fp == 0:
        return 0.0

    return round(
        tp / (tp + fp) * 100,
        2
    )


# =====================================================
# RECALL
# =====================================================

def recall(tp, fn):

    if tp + fn == 0:
        return 0.0

    return round(
        tp / (tp + fn) * 100,
        2
    )


# =====================================================
# F1
# =====================================================

def f1(tp, fp, fn):

    p = precision(tp, fp)
    r = recall(tp, fn)

    if p + r == 0:
        return 0.0

    return round(
        2 * p * r / (p + r),
        2
    )


# =====================================================
# BUILD SUMMARY
# =====================================================

def build_summary(
    total,
    threshold,
    metrics
):

    summary = {

        "total_papers": total,

        "similarity_threshold": threshold

    }

    # ==========================================
    # GPT / CLAUDE / GEMINI ...
    # ==========================================

    for llm_name, entities in metrics.items():

        summary[llm_name] = {}

        # --------------------------------------

        for entity_name, values in entities.items():

            summary[llm_name][entity_name] = {

                "accuracy": accuracy(
                    values["hits"],
                    total
                ),

                "precision": precision(
                    values["tp"],
                    values["fp"]
                ),

                "recall": recall(
                    values["tp"],
                    values["fn"]
                ),

                "f1": f1(
                    values["tp"],
                    values["fp"],
                    values["fn"]
                )

            }

    return summary