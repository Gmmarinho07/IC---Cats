# =====================================================
# ACCURACY
# =====================================================

def accuracy(hits, total):
    """
    Calcula a acurácia (%).

    Parameters
    ----------
    hits : int
        Número de acertos.

    total : int
        Número total de exemplos.
    """

    if total == 0:
        return 0.0

    return round((hits / total) * 100, 2)


# =====================================================
# PRECISION
# =====================================================

def precision(tp, fp):
    """
    Precision = TP / (TP + FP)
    """

    if (tp + fp) == 0:
        return 0.0

    return round((tp / (tp + fp)) * 100, 2)


# =====================================================
# RECALL
# =====================================================

def recall(tp, fn):
    """
    Recall = TP / (TP + FN)
    """

    if (tp + fn) == 0:
        return 0.0

    return round((tp / (tp + fn)) * 100, 2)


# =====================================================
# F1 SCORE
# =====================================================

def f1_score(tp, fp, fn):
    """
    F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    p = precision(tp, fp)
    r = recall(tp, fn)

    if (p + r) == 0:
        return 0.0

    p /= 100
    r /= 100

    return round(
        (2 * p * r / (p + r)) * 100,
        2
    )


# =====================================================
# BUILD SUMMARY
# =====================================================

def build_summary(
    total,
    threshold,
    gpt_catalyst_hits,
    claude_catalyst_hits,
    gpt_metal_hits,
    claude_metal_hits,
    gpt_support_hits,
    claude_support_hits
):
    """
    Resumo geral do benchmark.
    """

    return {

        "total_papers": total,

        "similarity_threshold": threshold,

        "gpt_catalyst_accuracy":
            accuracy(gpt_catalyst_hits, total),

        "claude_catalyst_accuracy":
            accuracy(claude_catalyst_hits, total),

        "gpt_metal_accuracy":
            accuracy(gpt_metal_hits, total),

        "claude_metal_accuracy":
            accuracy(claude_metal_hits, total),

        "gpt_support_accuracy":
            accuracy(gpt_support_hits, total),

        "claude_support_accuracy":
            accuracy(claude_support_hits, total)

    }