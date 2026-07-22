import json
from pathlib import Path

from evaluation.comparator import compare


# =====================================================
# CARREGAR RESULTADOS DOS AGENTES
# =====================================================

results_folder = Path("benchmark/results")

predictions = []

for file in sorted(results_folder.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:

        data = json.load(f)

    predictions.append(
        {
            "paper": file.stem,
            "catalysts": data.get("catalysts", [])
        }
    )

print(f"\nLoaded {len(predictions)} prediction files.")


# =====================================================
# CARREGAR GROUND TRUTH
# =====================================================

with open(
    "benchmark/ground_truth.json",
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)

print(f"Loaded {len(ground_truth)} ground truth entries.")


# =====================================================
# DEBUG (PODE REMOVER DEPOIS)
# =====================================================

print("\nExample prediction:")
print(json.dumps(predictions[0], indent=4, ensure_ascii=False))

print("\nExample ground truth:")
print(json.dumps(ground_truth[0], indent=4, ensure_ascii=False))


# =====================================================
# EXECUTAR COMPARAÇÃO
# =====================================================

summary, results = compare(
    predictions,
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