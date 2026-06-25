import json

from rapidfuzz import fuzz

from normalization import (
    normalize,
    normalize_list
)

SIMILARITY_THRESHOLD = 80


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
            best_score = max(best_score, score)

    return best_score


def support_similarity(agent_support, expected_support):
    if expected_support is None and agent_support is None:
        return 100

    if expected_support is None or agent_support is None:
        return 0

    return fuzz.token_set_ratio(
        str(agent_support).lower(),
        str(expected_support).lower()
    )


with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

with open("ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

truth_dict = {item["paper"]: item for item in ground_truth}

results = []

gpt_cat_hits = 0
claude_cat_hits = 0

gpt_metal_hits = 0
claude_metal_hits = 0

gpt_support_hits = 0
claude_support_hits = 0

total = 0

for item in dataset:

    paper = item["paper"]

    truth = truth_dict.get(paper)

    if truth is None:
        continue

    if truth.get("skip_benchmark", False):
        print(f"Skipping {paper}")
        continue

    total += 1

    expected_catalysts = normalize_list(
        truth["expected_catalysts"]
    )

    expected_metal = normalize_list(
        truth["expected_metal"]
    )

    expected_support = normalize(
        truth["expected_support"]
    )

    # GPT
    gpt_cat = normalize_list(
        item["gpt"]["catalyst"]["catalysts"]
    )

    gpt_metal = normalize_list(
        item["gpt"]["metal_support"]["metal"]
    )

    gpt_support = normalize(
        item["gpt"]["metal_support"]["support"]
    )

    # Claude
    claude_cat = normalize_list(
        item["claude"]["catalyst"]["catalysts"]
    )

    claude_metal = normalize_list(
        item["claude"]["metal_support"]["metal"]
    )

    claude_support = normalize(
        item["claude"]["metal_support"]["support"]
    )

    # Catalyst
    gpt_cat_score = best_similarity(gpt_cat, expected_catalysts)
    claude_cat_score = best_similarity(claude_cat, expected_catalysts)

    gpt_cat_match = gpt_cat_score >= SIMILARITY_THRESHOLD
    claude_cat_match = claude_cat_score >= SIMILARITY_THRESHOLD

    if gpt_cat_match:
        gpt_cat_hits += 1

    if claude_cat_match:
        claude_cat_hits += 1

    # Metal
    if not expected_metal and not gpt_metal:
        gpt_metal_score = 100
    else:
        gpt_metal_score = best_similarity(gpt_metal, expected_metal)

    if not expected_metal and not claude_metal:
        claude_metal_score = 100
    else:
        claude_metal_score = best_similarity(claude_metal, expected_metal)

    gpt_metal_match = gpt_metal_score >= SIMILARITY_THRESHOLD
    claude_metal_match = claude_metal_score >= SIMILARITY_THRESHOLD

    if gpt_metal_match:
        gpt_metal_hits += 1

    if claude_metal_match:
        claude_metal_hits += 1

    # Support
    gpt_support_score = support_similarity(
        gpt_support,
        expected_support
    )

    claude_support_score = support_similarity(
        claude_support,
        expected_support
    )

    gpt_support_match = gpt_support_score >= SIMILARITY_THRESHOLD
    claude_support_match = claude_support_score >= SIMILARITY_THRESHOLD

    if gpt_support_match:
        gpt_support_hits += 1

    if claude_support_match:
        claude_support_hits += 1

    results.append({

        "paper": paper,

        "gpt_catalyst_similarity": round(gpt_cat_score,2),
        "gpt_catalyst_match": gpt_cat_match,

        "claude_catalyst_similarity": round(claude_cat_score,2),
        "claude_catalyst_match": claude_cat_match,

        "gpt_metal_similarity": round(gpt_metal_score,2),
        "gpt_metal_match": gpt_metal_match,

        "claude_metal_similarity": round(claude_metal_score,2),
        "claude_metal_match": claude_metal_match,

        "gpt_support_similarity": round(gpt_support_score,2),
        "gpt_support_match": gpt_support_match,

        "claude_support_similarity": round(claude_support_score,2),
        "claude_support_match": claude_support_match

    })

summary = {

    "total_papers": total,

    "similarity_threshold": SIMILARITY_THRESHOLD,

    "gpt_catalyst_accuracy":
        round(gpt_cat_hits/total*100,2),

    "claude_catalyst_accuracy":
        round(claude_cat_hits/total*100,2),

    "gpt_metal_accuracy":
        round(gpt_metal_hits/total*100,2),

    "claude_metal_accuracy":
        round(claude_metal_hits/total*100,2),

    "gpt_support_accuracy":
        round(gpt_support_hits/total*100,2),

    "claude_support_accuracy":
        round(claude_support_hits/total*100,2)

}

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

print(json.dumps(summary, indent=4, ensure_ascii=False))

