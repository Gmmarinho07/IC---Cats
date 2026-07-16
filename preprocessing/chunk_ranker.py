"""
chunk_ranker.py

Domain-Aware Chunk Ranker

Versão 1.0
"""

import re


# =====================================================
# SECTION BONUS
# =====================================================

SECTION_BONUS = {

    "abstract": 10,

    "results": 9,

    "experimental": 8,

    "introduction": 3,

    "conclusion": 2,

    "references": -100

}


# =====================================================
# POSITIVE KEYWORDS
# =====================================================

KEYWORDS = {

    "catalyst": 5,
    "catalysts": 5,

    "support": 4,
    "supported": 4,

    "metal": 4,

    "active phase": 6,

    "activity": 3,

    "conversion": 3,

    "selectivity": 3,

    "yield": 3,

    "reaction": 2,

    "ethanol": 1,

    "prepared": 4,

    "preparation": 4,

    "impregnation": 5,

    "calcination": 4,

    "reduction": 3,

    "hydrogenation": 2,

    "dehydrogenation": 2,

    "co-precipitation": 5,

    "characterization": 2,

    "xrd": 2,

    "tem": 2,

    "sem": 2

}


# =====================================================
# NEGATIVE KEYWORDS
# =====================================================

NEGATIVE_KEYWORDS = {

    "credit authorship": -80,

    "author contributions": -80,

    "acknowledgements": -80,

    "acknowledgments": -80,

    "funding": -60,

    "conflict of interest": -100,

    "supplementary information": -100,

    "supporting information": -100,

    "supplementary data": -100,

    "appendix": -100,

    "references": -150

}


# =====================================================
# FIGURE / TABLE WORDS
# =====================================================

FIGURE_WORDS = [

    "figure",

    "fig.",

    "fig ",

    "table",

    "scheme"

]


# =====================================================
# REGEX PATTERNS
# =====================================================

PATTERNS = {

    # Ru/MgO
    # Pd/C
    # Ni-HAP
    # Pt/CeO2

    "supported_catalyst": (

        r"\b[A-Z][a-z]?(?:[-/][A-Za-z0-9().+\-]+)+\b",

        10

    ),

    # Fe2O3
    # Co3O4
    # CeO2

    "oxide_formula": (

        r"\b[A-Z][a-z]?\d*O\d+\b",

        5

    ),

    # wt%

    "wt_percent": (

        r"\b\d+(?:\.\d+)?\s*wt%",

        2

    ),

    # °C

    "temperature": (

        r"\b\d+(?:\.\d+)?\s*°?\s*C\b",

        2

    ),

    # %

    "percentage": (

        r"\b\d+(?:\.\d+)?\s*%",

        1

    )

}


# =====================================================
# KEYWORD SCORE
# =====================================================

def keyword_score(text):

    score = 0

    lower = text.lower()

    for keyword, weight in KEYWORDS.items():

        occurrences = lower.count(keyword)

        score += occurrences * weight

    return score


# =====================================================
# REGEX SCORE
# =====================================================

def regex_score(text):

    score = 0

    for pattern, weight in PATTERNS.values():

        matches = re.findall(pattern, text)

        score += len(matches) * weight

    return score


# =====================================================
# FIGURE SCORE
# =====================================================

def figure_score(text):

    score = 0

    lower = text.lower()

    for word in FIGURE_WORDS:

        score += lower.count(word) * 3

    return score


# =====================================================
# NEGATIVE SCORE
# =====================================================

def negative_score(text):

    score = 0

    lower = text.lower()

    for keyword, penalty in NEGATIVE_KEYWORDS.items():

        if keyword in lower:

            score += penalty

    return score


# =====================================================
# SECTION BONUS
# =====================================================

def section_score(section):

    return SECTION_BONUS.get(

        section.lower(),

        0

    )
# =====================================================
# CALCULATE SCORE
# =====================================================

def calculate_score(chunk):
    """
    Calcula a pontuação final de um chunk.
    """

    text = chunk["text"]
    section = chunk["section"]

    scores = {

        "section_bonus": section_score(section),

        "keywords": keyword_score(text),

        "regex": regex_score(text),

        "figures": figure_score(text),

        "negative": negative_score(text)

    }

    total = sum(scores.values())

    chunk["score"] = total
    chunk["score_breakdown"] = scores

    return chunk


# =====================================================
# RANK CHUNKS
# =====================================================

def rank_chunks(chunks):
    """
    Calcula o score de todos os chunks
    e retorna ordenados do maior para o menor.
    """

    ranked = []

    for chunk in chunks:

        ranked.append(
            calculate_score(chunk)
        )

    ranked.sort(

        key=lambda x: x["score"],

        reverse=True

    )

    return ranked
# =====================================================
# SELECT CHUNKS
# =====================================================

def select_chunks(
    ranked_chunks,
    selection_policy
):
    """
    Exemplo:

    selection_policy = {

        "abstract":1,

        "experimental":2,

        "results":2

    }
    """

    selected = []

    # Agrupa por seção
    grouped = {}

    for chunk in ranked_chunks:

        section = chunk["section"]

        grouped.setdefault(
            section,
            []
        ).append(chunk)

    # Ordena cada grupo
    for section in grouped:

        grouped[section].sort(

            key=lambda x: x["score"],

            reverse=True

        )

    # Seleciona conforme política

    for section, limit in selection_policy.items():

        if section not in grouped:

            continue

        selected.extend(

            grouped[section][:limit]

        )

    return selected
# =====================================================
# PREVIEW
# =====================================================

def preview_ranking(
    ranked_chunks,
    n=10
):

    print()

    print("=" * 70)
    print("CHUNK RANKING")
    print("=" * 70)

    for chunk in ranked_chunks[:n]:

        print()

        print(

            f"[{chunk['section'].upper()}] "

            f"Chunk {chunk['chunk_id']}"

        )

        print(

            f"Score: {chunk['score']}"

        )

        print(

            chunk["score_breakdown"]

        )

        print("-" * 70)

        print(

            chunk["text"][:300]

        )

        print()