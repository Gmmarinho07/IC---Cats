import json

from normalization import (
    normalize,
    normalize_list
)

from evaluation.similarity import (
    SIMILARITY_THRESHOLD,
    best_similarity,
    optional_list_similarity,
    optional_text_similarity,
    is_match
)

from evaluation.metrics import (
    build_summary
)


# =====================================================
# COMPARATOR
# =====================================================

def compare(dataset, ground_truth):

    # =============================================
    # INDEXAR GROUND TRUTH
    # =============================================

    truth_dict = {}

    for item in ground_truth:

        if item.get("skip_benchmark", False):
            continue

        truth_dict[item["paper"]] = item

    # =============================================
    # CONTADORES
    # =============================================

    gpt_catalyst_hits = 0
    claude_catalyst_hits = 0

    gpt_metal_hits = 0
    claude_metal_hits = 0

    gpt_support_hits = 0
    claude_support_hits = 0

    results = []

    # =============================================
    # LOOP PRINCIPAL
    # =============================================

    for item in dataset:

        paper = item["paper"]

        truth = truth_dict.get(paper)

        if truth is None:
            continue

        # -----------------------------------------
        # GROUND TRUTH NORMALIZADO
        # -----------------------------------------

        expected_catalysts = normalize_list(
            truth["expected_catalysts"]
        )

        expected_metal = normalize_list(
            truth["expected_metal"]
        )

        expected_support = normalize(
            truth["expected_support"]
        )
        # ======================================
        # GPT - CATALYST
        # ======================================

        gpt_catalyst = normalize_list(
            item["gpt"]["catalyst"]["catalysts"]
        )

        gpt_catalyst_score = best_similarity(
            gpt_catalyst,
            expected_catalysts
        )

        gpt_catalyst_match = is_match(
            gpt_catalyst_score
        )

        if gpt_catalyst_match:
            gpt_catalyst_hits += 1


        # ======================================
        # CLAUDE - CATALYST
        # ======================================

        claude_catalyst = normalize_list(
            item["claude"]["catalyst"]["catalysts"]
        )

        claude_catalyst_score = best_similarity(
            claude_catalyst,
            expected_catalysts
        )

        claude_catalyst_match = is_match(
            claude_catalyst_score
        )

        if claude_catalyst_match:
            claude_catalyst_hits += 1


        # ======================================
        # GPT - METAL
        # ======================================

        gpt_metal = normalize_list(
            item["gpt"]["metal_support"]["metal"]
        )

        gpt_metal_score = optional_list_similarity(
            gpt_metal,
            expected_metal
        )

        gpt_metal_match = is_match(
            gpt_metal_score
        )

        if gpt_metal_match:
            gpt_metal_hits += 1


        # ======================================
        # CLAUDE - METAL
        # ======================================

        claude_metal = normalize_list(
            item["claude"]["metal_support"]["metal"]
        )

        claude_metal_score = optional_list_similarity(
            claude_metal,
            expected_metal
        )

        claude_metal_match = is_match(
            claude_metal_score
        )

        if claude_metal_match:
            claude_metal_hits += 1


        # ======================================
        # GPT - SUPPORT
        # ======================================

        gpt_support = normalize(
            item["gpt"]["metal_support"]["support"]
        )

        gpt_support_score = optional_text_similarity(
            gpt_support,
            expected_support
        )

        gpt_support_match = is_match(
            gpt_support_score
        )

        if gpt_support_match:
            gpt_support_hits += 1


        # ======================================
        # CLAUDE - SUPPORT
        # ======================================

        claude_support = normalize(
            item["claude"]["metal_support"]["support"]
        )

        claude_support_score = optional_text_similarity(
            claude_support,
            expected_support
        )

        claude_support_match = is_match(
            claude_support_score
        )

        if claude_support_match:
            claude_support_hits += 1


        # ======================================
        # RESULTADO DO PAPER
        # ======================================

        results.append({

            "paper": paper,

            "gpt_catalyst_similarity":
                round(gpt_catalyst_score, 2),

            "gpt_catalyst_match":
                gpt_catalyst_match,

            "claude_catalyst_similarity":
                round(claude_catalyst_score, 2),

            "claude_catalyst_match":
                claude_catalyst_match,

            "gpt_metal_similarity":
                round(gpt_metal_score, 2),

            "gpt_metal_match":
                gpt_metal_match,

            "claude_metal_similarity":
                round(claude_metal_score, 2),

            "claude_metal_match":
                claude_metal_match,

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

    summary = build_summary(

        total=total,

        threshold=SIMILARITY_THRESHOLD,

        gpt_catalyst_hits=gpt_catalyst_hits,

        claude_catalyst_hits=claude_catalyst_hits,

        gpt_metal_hits=gpt_metal_hits,

        claude_metal_hits=claude_metal_hits,

        gpt_support_hits=gpt_support_hits,

        claude_support_hits=claude_support_hits

    )

    # ==========================================
    # RETORNO
    # ==========================================

    return summary, results