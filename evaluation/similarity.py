from rapidfuzz import fuzz

# =====================================================
# CONFIGURAÇÃO
# =====================================================

SIMILARITY_THRESHOLD = 80


# =====================================================
# BEST SIMILARITY ENTRE DUAS LISTAS
# =====================================================

def best_similarity(agent_list, expected_list):

    if not agent_list or not expected_list:
        return 0

    best_score = 0

    for expected in expected_list:

        for agent in agent_list:

            score = fuzz.token_set_ratio(
                str(expected).lower(),
                str(agent).lower()
            )

            best_score = max(
                best_score,
                score
            )

    return best_score


# =====================================================
# SIMILARIDADE ENTRE STRINGS
# =====================================================

def text_similarity(text_a, text_b):

    if text_a is None or text_b is None:
        return 0

    return fuzz.token_set_ratio(
        str(text_a).lower(),
        str(text_b).lower()
    )


# =====================================================
# OPTIONALS
# =====================================================

def optional_text_similarity(predicted, expected):

    if predicted is None and expected is None:
        return 100

    return text_similarity(
        predicted,
        expected
    )


def optional_list_similarity(predicted, expected):

    if not predicted and not expected:
        return 100

    return best_similarity(
        predicted,
        expected
    )


# =====================================================
# MATCH
# =====================================================

def is_match(
    score,
    threshold=SIMILARITY_THRESHOLD
):

    return score >= threshold


# =====================================================
# ENTITY MATCHING (NOVO)
# =====================================================

def match_entities(
    predicted,
    expected,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Realiza o matching guloso entre listas de entidades.

    Retorna:
    {
        "tp": ...,
        "fp": ...,
        "fn": ...,
        "matches": [
            {
                "expected": "...",
                "predicted": "...",
                "score": ...
            }
        ]
    }
    """

    predicted = predicted or []
    expected = expected or []

    used_predictions = set()

    tp = 0
    fp = 0
    fn = 0

    matches = []

    # ==========================================
    # PARA CADA ENTIDADE ESPERADA
    # ==========================================

    for expected_entity in expected:

        best_index = None
        best_score = 0

        for i, predicted_entity in enumerate(predicted):

            if i in used_predictions:
                continue

            score = fuzz.token_set_ratio(
                str(expected_entity).lower(),
                str(predicted_entity).lower()
            )

            if score > best_score:

                best_score = score
                best_index = i

        # --------------------------------------

        if (
            best_index is not None
            and best_score >= threshold
        ):

            tp += 1

            used_predictions.add(
                best_index
            )

            matches.append({

                "expected":
                    expected_entity,

                "predicted":
                    predicted[best_index],

                "score":
                    round(best_score, 2)

            })

        else:

            fn += 1

    # ==========================================
    # SOBROU PREDIÇÃO = FP
    # ==========================================

    fp = len(predicted) - len(used_predictions)

    # ==========================================
    # RETORNO
    # ==========================================

    return {

        "tp": tp,

        "fp": fp,

        "fn": fn,

        "matches": matches

    }