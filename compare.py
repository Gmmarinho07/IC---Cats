import json
from rapidfuzz import fuzz

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


def support_similarity(agent_support,
                       expected_support):

    if not agent_support or not expected_support:
        return 0

    return fuzz.token_set_ratio(
        str(agent_support).lower(),
        str(expected_support).lower()
    )

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

agent_1_hits = 0
agent_2_hits = 0

metal_hits = 0
support_hits = 0

results = []

# =====================================
# LOOP
# =====================================

for item in dataset:

    paper = item["paper"]

    truth = truth_dict.get(paper)

    if not truth:
        continue

    expected_catalysts = truth["expected_catalysts"]

    expected_metal = truth["expected_metal"]

    expected_support = truth["expected_support"]

    # --------------------
    # AGENT 1
    # --------------------

    agent_1 = item["agent_1"]["catalysts"]

    score_1 = best_similarity(
        agent_1,
        expected_catalysts
    )

    match_1 = score_1 >= SIMILARITY_THRESHOLD

    if match_1:
        agent_1_hits += 1

    # --------------------
    # AGENT 2
    # --------------------

    agent_2 = item["agent_2"]["catalysts"]

    score_2 = best_similarity(
        agent_2,
        expected_catalysts
    )

    match_2 = score_2 >= SIMILARITY_THRESHOLD

    if match_2:
        agent_2_hits += 1

    # --------------------
    # AGENT 3 METAL
    # --------------------

    agent_metal = (
        item["agent_3"]["metal"]
        if item["agent_3"]["metal"]
        else []
    )

    metal_score = best_similarity(
        agent_metal,
        expected_metal
    )

    metal_match = (
        metal_score >= SIMILARITY_THRESHOLD
        if expected_metal
        else True
    )

    if metal_match:
        metal_hits += 1

    # --------------------
    # AGENT 3 SUPPORT
    # --------------------

    agent_support = item["agent_3"]["support"]

    support_score = support_similarity(
        agent_support,
        expected_support
    )

    support_match = (
        support_score >= SIMILARITY_THRESHOLD
    )

    if support_match:
        support_hits += 1

    # --------------------
    # RESULTADOS
    # --------------------

    results.append({

        "paper": paper,

        "agent_1_similarity":
            round(score_1, 2),

        "agent_1_match":
            match_1,

        "agent_2_similarity":
            round(score_2, 2),

        "agent_2_match":
            match_2,

        "metal_similarity":
            round(metal_score, 2),

        "metal_match":
            metal_match,

        "support_similarity":
            round(support_score, 2),

        "support_match":
            support_match
    })

# =====================================
# RESUMO
# =====================================

total = len(results)

summary = {

    "total_papers": total,

    "agent_1_accuracy":
        round(
            agent_1_hits / total * 100,
            2
        ),

    "agent_2_accuracy":
        round(
            agent_2_hits / total * 100,
            2
        ),

    "metal_accuracy":
        round(
            metal_hits / total * 100,
            2
        ),

    "support_accuracy":
        round(
            support_hits / total * 100,
            2
        )
}

# =====================================
# SALVAR
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

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(json.dumps(
    summary,
    indent=4,
    ensure_ascii=False
))