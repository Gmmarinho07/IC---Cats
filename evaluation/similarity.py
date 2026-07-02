from rapidfuzz import fuzz


# =====================================================
# CONFIGURAÇÃO
# =====================================================

SIMILARITY_THRESHOLD = 80


# =====================================================
# MELHOR SIMILARIDADE ENTRE DUAS LISTAS
# =====================================================

def best_similarity(agent_list, expected_list):
    """
    Calcula a maior similaridade entre os itens de duas listas.
    Retorna um valor entre 0 e 100.
    """

    if not agent_list or not expected_list:
        return 0

    best_score = 0

    for expected in expected_list:

        for agent in agent_list:

            score = fuzz.token_set_ratio(
                str(expected).lower(),
                str(agent).lower()
            )

            if score > best_score:
                best_score = score

    return best_score


# =====================================================
# SIMILARIDADE ENTRE STRINGS
# =====================================================

def text_similarity(text_a, text_b):
    """
    Calcula a similaridade entre duas strings.
    Utilizado principalmente para Support.
    """

    if text_a is None or text_b is None:
        return 0

    return fuzz.token_set_ratio(
        str(text_a).lower(),
        str(text_b).lower()
    )


# =====================================================
# VERIFICA MATCH
# =====================================================

def is_match(score, threshold=SIMILARITY_THRESHOLD):
    """
    Retorna True se a similaridade atingir o limiar.
    """

    return score >= threshold


# =====================================================
# COMPARAÇÃO PARA CAMPOS OPCIONAIS
# =====================================================

def optional_text_similarity(predicted, expected):
    """
    Trata corretamente casos onde ambos são None.
    """

    if predicted is None and expected is None:
        return 100

    return text_similarity(predicted, expected)


def optional_list_similarity(predicted, expected):
    """
    Trata corretamente listas opcionais.
    """

    if not predicted and not expected:
        return 100

    return best_similarity(predicted, expected)