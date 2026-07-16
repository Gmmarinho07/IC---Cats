from normalization import (
    normalize,
    normalize_list
)

from evaluation.similarity import (
    SIMILARITY_THRESHOLD,
    match_entities,
    best_similarity,
    optional_text_similarity,
    optional_list_similarity,
    is_match
)

from evaluation.metrics import (
    build_summary
)


# =====================================================
# GROUND TRUTH
# =====================================================

def build_truth_dict(ground_truth):

    truth_dict = {}

    for item in ground_truth:

        truth_dict[item["paper"]] = item

    return truth_dict


# =====================================================
# COMPARATOR
# =====================================================

def compare(dataset, ground_truth):

    truth_dict = build_truth_dict(
        ground_truth
    )

    # ==========================================
    # ARTICLE LEVEL
    # ==========================================

    gpt_catalyst_hits = 0
    claude_catalyst_hits = 0

    gpt_metal_hits = 0
    claude_metal_hits = 0

    gpt_support_hits = 0
    claude_support_hits = 0

    # ==========================================
    # ENTITY LEVEL - GPT
    # (Precision / Recall / F1)
    # ==========================================

    gpt_catalyst_tp = 0
    gpt_catalyst_fp = 0
    gpt_catalyst_fn = 0

    gpt_metal_tp = 0
    gpt_metal_fp = 0
    gpt_metal_fn = 0

    gpt_support_tp = 0
    gpt_support_fp = 0
    gpt_support_fn = 0

    # ==========================================
    # ENTITY LEVEL - CLAUDE
    # (Precision / Recall / F1)
    # ==========================================

    claude_catalyst_tp = 0
    claude_catalyst_fp = 0
    claude_catalyst_fn = 0

    claude_metal_tp = 0
    claude_metal_fp = 0
    claude_metal_fn = 0

    claude_support_tp = 0
    claude_support_fp = 0
    claude_support_fn = 0

    # ==========================================

    results = []

    # ==========================================

    for item in dataset:

        paper = item["paper"]

        truth = truth_dict.get(
            paper
        )

        if truth is None:
            continue

        # ======================================
        # GROUND TRUTH
        # ======================================

        expected_catalysts = normalize_list(
            truth["expected_catalysts"]
        )

        expected_metal = normalize_list(
            truth["expected_metal"]
        )

        expected_support = normalize(
            truth["expected_support"]
        )
        # =====================================================
        # GPT - CATALYST
        # =====================================================

        gpt_catalyst = normalize_list(
            item["gpt"]["catalyst"]["catalysts"]
        )

        gpt_stats = match_entities(
            predicted=gpt_catalyst,
            expected=expected_catalysts
        )

        gpt_catalyst_tp += gpt_stats["tp"]
        gpt_catalyst_fp += gpt_stats["fp"]
        gpt_catalyst_fn += gpt_stats["fn"]

        gpt_score = best_similarity(
            gpt_catalyst,
            expected_catalysts
        )

        gpt_match = is_match(
            gpt_score
        )

        if gpt_match:
            gpt_catalyst_hits += 1


        # =====================================================
        # CLAUDE - CATALYST
        # =====================================================

        claude_catalyst = normalize_list(
            item["claude"]["catalyst"]["catalysts"]
        )

        claude_stats = match_entities(
            predicted=claude_catalyst,
            expected=expected_catalysts
        )

        claude_catalyst_tp += claude_stats["tp"]
        claude_catalyst_fp += claude_stats["fp"]
        claude_catalyst_fn += claude_stats["fn"]

        claude_score = best_similarity(
            claude_catalyst,
            expected_catalysts
        )

        claude_match = is_match(
            claude_score
        )

        if claude_match:
            claude_catalyst_hits += 1
        # =====================================================
        # GPT - METAL
        # =====================================================

        gpt_metal = normalize_list(
            item["gpt"]["metal_support"]["metal"]
        )

        gpt_metal_stats = match_entities(
            predicted=gpt_metal,
            expected=expected_metal
        )

        gpt_metal_tp += gpt_metal_stats["tp"]
        gpt_metal_fp += gpt_metal_stats["fp"]
        gpt_metal_fn += gpt_metal_stats["fn"]

        gpt_metal_score = optional_list_similarity(
            gpt_metal,
            expected_metal
        )

        gpt_metal_match = is_match(
            gpt_metal_score
        )

        if gpt_metal_match:
            gpt_metal_hits += 1


        # =====================================================
        # CLAUDE - METAL
        # =====================================================

        claude_metal = normalize_list(
            item["claude"]["metal_support"]["metal"]
        )

        claude_metal_stats = match_entities(
            predicted=claude_metal,
            expected=expected_metal
        )

        claude_metal_tp += claude_metal_stats["tp"]
        claude_metal_fp += claude_metal_stats["fp"]
        claude_metal_fn += claude_metal_stats["fn"]

        claude_metal_score = optional_list_similarity(
            claude_metal,
            expected_metal
        )

        claude_metal_match = is_match(
            claude_metal_score
        )

        if claude_metal_match:
            claude_metal_hits += 1


        # =====================================================
        # GPT - SUPPORT
        # =====================================================

        gpt_support = normalize(
            item["gpt"]["metal_support"]["support"]
        )

        gpt_support_stats = match_entities(
            predicted=[gpt_support] if gpt_support else [],
            expected=[expected_support] if expected_support else []
        )

        gpt_support_tp += gpt_support_stats["tp"]
        gpt_support_fp += gpt_support_stats["fp"]
        gpt_support_fn += gpt_support_stats["fn"]

        gpt_support_score = optional_text_similarity(
            gpt_support,
            expected_support
        )

        gpt_support_match = is_match(
            gpt_support_score
        )

        if gpt_support_match:
            gpt_support_hits += 1


        # =====================================================
        # CLAUDE - SUPPORT
        # =====================================================

        claude_support = normalize(
            item["claude"]["metal_support"]["support"]
        )

        claude_support_stats = match_entities(
            predicted=[claude_support] if claude_support else [],
            expected=[expected_support] if expected_support else []
        )

        claude_support_tp += claude_support_stats["tp"]
        claude_support_fp += claude_support_stats["fp"]
        claude_support_fn += claude_support_stats["fn"]

        claude_support_score = optional_text_similarity(
            claude_support,
            expected_support
        )

        claude_support_match = is_match(
            claude_support_score
        )

        if claude_support_match:
            claude_support_hits += 1


        # =====================================================
        # RESULTADOS DO PAPER
        # =====================================================

        results.append({

            "paper": paper,

            # ----------------------------------
            # Catalyst
            # ----------------------------------

            "gpt_catalyst_similarity":
                round(gpt_score, 2),

            "gpt_catalyst_match":
                gpt_match,

            "claude_catalyst_similarity":
                round(claude_score, 2),

            "claude_catalyst_match":
                claude_match,

            # ----------------------------------
            # Metal
            # ----------------------------------

            "gpt_metal_similarity":
                round(gpt_metal_score, 2),

            "gpt_metal_match":
                gpt_metal_match,

            "claude_metal_similarity":
                round(claude_metal_score, 2),

            "claude_metal_match":
                claude_metal_match,

            # ----------------------------------
            # Support
            # ----------------------------------

            "gpt_support_similarity":
                round(gpt_support_score, 2),

            "gpt_support_match":
                gpt_support_match,

            "claude_support_similarity":
                round(claude_support_score, 2),

            "claude_support_match":
                claude_support_match

        })
    # ==========================================
    # SUMMARY
    # ==========================================

    total = len(results)

    metrics = {

        "gpt": {

            "catalyst": {

                "hits": gpt_catalyst_hits,

                "tp": gpt_catalyst_tp,
                "fp": gpt_catalyst_fp,
                "fn": gpt_catalyst_fn

            },

            "metal": {

                "hits": gpt_metal_hits,

                "tp": gpt_metal_tp,
                "fp": gpt_metal_fp,
                "fn": gpt_metal_fn

            },

            "support": {

                "hits": gpt_support_hits,

                "tp": gpt_support_tp,
                "fp": gpt_support_fp,
                "fn": gpt_support_fn

            }

        },

        "claude": {

            "catalyst": {

                "hits": claude_catalyst_hits,

                "tp": claude_catalyst_tp,
                "fp": claude_catalyst_fp,
                "fn": claude_catalyst_fn

            },

            "metal": {

                "hits": claude_metal_hits,

                "tp": claude_metal_tp,
                "fp": claude_metal_fp,
                "fn": claude_metal_fn

            },

            "support": {

                "hits": claude_support_hits,

                "tp": claude_support_tp,
                "fp": claude_support_fp,
                "fn": claude_support_fn

            }

        }

    }

    summary = build_summary(

        total=total,

        threshold=SIMILARITY_THRESHOLD,

        metrics=metrics

    )

    # ==========================================
    # RETURN
    # ==========================================

    return summary, results