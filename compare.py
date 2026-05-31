import json
from rapidfuzz import fuzz

# =====================================
# CONFIGURAÇÃO
# =====================================

SIMILARITY_THRESHOLD = 80

# =====================================
# FUNÇÃO DE COMPARAÇÃO
# =====================================

def best_similarity(agent_list, expected_list):

    best_score = 0

    for expected in expected_list:

        for agent in agent_list:

            score = fuzz.token_set_ratio(
                expected.lower(),
                agent.lower()
            )

            if score > best_score:
                best_score = score

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

    truth_dict[item["paper"]] = item["expected_catalysts"]


# =====================================
# RESULTADOS
# =====================================

results = []

agent_1_hits = 0
agent_2_hits = 0

# =====================================
# LOOP
# =====================================

for item in dataset:

    paper = item["paper"]

    expected = truth_dict.get(paper, [])

    agent_1 = item["agent_1"]["catalysts"]

    agent_2 = item["agent_2"]["catalysts"]

    # -----------------------------
    # Similaridade
    # -----------------------------

    agent_1_score = best_similarity(
        agent_1,
        expected
    )

    agent_2_score = best_similarity(
        agent_2,
        expected
    )

    # -----------------------------
    # Match
    # -----------------------------

    agent_1_match = (
        agent_1_score >= SIMILARITY_THRESHOLD
    )

    agent_2_match = (
        agent_2_score >= SIMILARITY_THRESHOLD
    )

    if agent_1_match:
        agent_1_hits += 1

    if agent_2_match:
        agent_2_hits += 1

    # -----------------------------
    # Salvar resultado
    # -----------------------------

    results.append({

        "paper": paper,

        "expected": expected,

        "agent_1": agent_1,

        "agent_1_similarity":
            round(agent_1_score, 2),

        "agent_1_match":
            agent_1_match,

        "agent_2": agent_2,

        "agent_2_similarity":
            round(agent_2_score, 2),

        "agent_2_match":
            agent_2_match
    })

    # -----------------------------
    # Mostrar no terminal
    # -----------------------------

    print("\n====================")

    print(f"Paper: {paper}")

    print(f"Expected: {expected}")

    print(
        f"Agent 1 similarity: "
        f"{agent_1_score:.2f}"
    )

    print(
        f"Agent 2 similarity: "
        f"{agent_2_score:.2f}"
    )


# =====================================
# ESTATÍSTICAS
# =====================================

total = len(dataset)

summary = {

    "total_papers": total,

    "similarity_threshold":
        SIMILARITY_THRESHOLD,

    "agent_1_accuracy":
        round(
            (agent_1_hits / total) * 100,
            2
        ),

    "agent_2_accuracy":
        round(
            (agent_2_hits / total) * 100,
            2
        )
}

# =====================================
# SALVAR JSON
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
# RESUMO
# =====================================

print("\n================================")
print("COMPARISON COMPLETED")
print("================================")
print(json.dumps(
    summary,
    indent=4,
    ensure_ascii=False
))