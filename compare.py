import json

from rapidfuzz import fuzz

from normalization import (
    normalize_list
)

SIMILARITY_THRESHOLD = 80


# =====================================
# FUNÇÕES
# =====================================

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


# =====================================
# CARREGAR ARQUIVOS
# =====================================

with open(
    "dataset.json",
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


with open(
    "ground_truth.json",
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)


# =====================================
# INDEXAR GROUND TRUTH
# =====================================

truth_dict = {}

for item in ground_truth:

    truth_dict[item["paper"]] = item


# =====================================
# CONTADORES
# =====================================

gpt_hits = 0
claude_hits = 0

results = []


# =====================================
# LOOP PRINCIPAL
# =====================================

for item in dataset:

    paper = item["paper"]

    truth = truth_dict.get(paper)

    if not truth:
        continue

    # ----------------------------
    # GROUND TRUTH NORMALIZADO
    # ----------------------------

    expected_catalysts = normalize_list(
        truth["expected_catalysts"]
    )

    # ----------------------------
    # GPT
    # ----------------------------

    gpt_list = normalize_list(
        item["gpt"]["catalysts"]
    )

    gpt_score = best_similarity(
        gpt_list,
        expected_catalysts
    )

    gpt_match = (
        gpt_score >= SIMILARITY_THRESHOLD
    )

    if gpt_match:
        gpt_hits += 1

    # ----------------------------
    # CLAUDE
    # ----------------------------

    claude_list = normalize_list(
        item["claude"]["catalysts"]
    )

    claude_score = best_similarity(
        claude_list,
        expected_catalysts
    )

    claude_match = (
        claude_score >= SIMILARITY_THRESHOLD
    )

    if claude_match:
        claude_hits += 1

    # ----------------------------
    # RESULTADO INDIVIDUAL
    # ----------------------------

    results.append({

        "paper": paper,

        "gpt_similarity":
            round(gpt_score, 2),

        "gpt_match":
            gpt_match,

        "claude_similarity":
            round(claude_score, 2),

        "claude_match":
            claude_match

    })


# =====================================
# RESUMO
# =====================================

total = len(results)

summary = {

    "total_papers":
        total,

    "similarity_threshold":
        SIMILARITY_THRESHOLD,

    "gpt_accuracy":
        round(
            gpt_hits / total * 100,
            2
        ),

    "claude_accuracy":
        round(
            claude_hits / total * 100,
            2
        )
}


# =====================================
# SALVAR RESULTADOS
# =====================================

with open(
    "comparison_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "summary": summary,
            "results": results
        },
        f,
        indent=4,
        ensure_ascii=False
    )


# =====================================
# PRINT FINAL
# =====================================

print("\n==============================")
print("COMPARISON COMPLETED")
print("==============================")

print(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    )
)