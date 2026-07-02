import json

from evaluation.comparator import compare


# =====================================================
# CARREGAR DATASET
# =====================================================

with open(
    "benchmark/dataset.json",
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


# =====================================================
# CARREGAR GROUND TRUTH
# =====================================================

with open(
    "benchmark/ground_truth.json",
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)


# =====================================================
# EXECUTAR COMPARAÇÃO
# =====================================================

summary, results = compare(
    dataset,
    ground_truth
)


# =====================================================
# SALVAR RESULTADOS
# =====================================================

comparison = {

    "summary": summary,

    "results": results

}

with open(
    "benchmark/comparison_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        indent=4,
        ensure_ascii=False
    )


# =====================================================
# PRINT FINAL
# =====================================================

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

print("\nResults saved to:")
print("benchmark/comparison_results.json")