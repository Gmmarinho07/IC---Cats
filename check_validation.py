
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================
# CONFIGURAÇÕES
# =====================================================

BASE = Path("benchmark/validation/cross")

OUTPUT = BASE / "plots"
OUTPUT.mkdir(parents=True, exist_ok=True)

DECISIONS = [
    "correct",
    "partial",
    "incorrect",
    "not_supported",
]

LABELS = [
    "Corretas",
    "Parciais",
    "Incorretas",
    "Não sustentadas",
]

EXPERIMENTS = {
    "GPT → Claude": {
        "folder": "gpt_to_claude",
        "extraction": "gpt",
        "judge": "claude",
    },
    "Claude → GPT": {
        "folder": "claude_to_gpt",
        "extraction": "claude",
        "judge": "gpt",
    },
}


# =====================================================
# LEITURA DOS RESULTADOS
# =====================================================

def load_experiment(config):
    folder = BASE / config["folder"]
    counts = Counter()
    articles = 0
    errors = 0

    if not folder.exists():
        print(f"[ERRO] Pasta não encontrada: {folder}")
        return counts, articles, errors

    files = sorted(folder.glob("*.json"))

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Confirma que o arquivo pertence ao experimento.
            if data.get("extraction_model") != config["extraction"]:
                continue

            if data.get("judge_model") != config["judge"]:
                continue

            result = data.get("validation", {})
            items = result.get("validation", [])

            if not isinstance(items, list):
                raise ValueError("Campo validation não é uma lista")

            articles += 1

            for item in items:
                decision = item.get("decision")

                if decision in DECISIONS:
                    counts[decision] += 1

        except Exception as exc:
            errors += 1
            print(f"[ERRO] {path.name}: {exc}")

    return counts, articles, errors


# =====================================================
# GRÁFICOS INDIVIDUAIS
# =====================================================

def plot_individual(name, counts, articles):
    total = sum(counts.values())

    if total == 0:
        print(f"[AVISO] Nenhuma avaliação para {name}")
        return

    values = [counts.get(d, 0) for d in DECISIONS]
    percentages = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(LABELS, values)

    for bar, value, pct in zip(bars, values, percentages):
        ax.annotate(
            f"{value}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_title(f"Validação cruzada — {name}")
    ax.set_ylabel("Quantidade de entidades")
    ax.set_xlabel("Categoria da avaliação")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.18 + 1)

    fig.text(
        0.5, 0.01,
        f"Artigos: {articles} | Avaliações: {total}",
        ha="center",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    filename = name.lower().replace(" → ", "_to_").replace(" ", "")
    fig.savefig(OUTPUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# COMPARAÇÃO
# =====================================================

def plot_comparison(results):
    names = list(EXPERIMENTS.keys())
    percentages = {}

    for name in names:
        counts = results[name]["counts"]
        total = sum(counts.values())

        percentages[name] = [
            counts.get(d, 0) / total * 100 if total else 0
            for d in DECISIONS
        ]

    x = np.arange(len(LABELS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6.5))

    bars1 = ax.bar(
        x - width / 2,
        percentages[names[0]],
        width,
        label=names[0],
    )
    bars2 = ax.bar(
        x + width / 2,
        percentages[names[1]],
        width,
        label=names[1],
    )

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax.set_title("Comparação da validação cruzada")
    ax.set_ylabel("Percentual das avaliações (%)")
    ax.set_xlabel("Categoria da avaliação")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "validation_cross_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# =====================================================
# EXECUÇÃO
# =====================================================

def main():
    results = {}
    csv_rows = []

    for name, config in EXPERIMENTS.items():
        counts, articles, errors = load_experiment(config)
        total = sum(counts.values())

        results[name] = {
            "counts": counts,
            "articles": articles,
        }

        print(f"\n{name}")
        print(f"Artigos encontrados: {articles}")
        print(f"Total de avaliações: {total}")
        print(f"Arquivos com erro de leitura: {errors}")

        for decision, label in zip(DECISIONS, LABELS):
            value = counts.get(decision, 0)
            pct = value / total * 100 if total else 0
            print(f"  {label}: {value} ({pct:.1f}%)")

            csv_rows.append({
                "experimento": name,
                "modelo_extracao": config["extraction"],
                "modelo_juiz": config["judge"],
                "categoria": decision,
                "quantidade": value,
                "percentual": round(pct, 2),
                "artigos": articles,
                "total_avaliacoes": total,
            })

        plot_individual(name, counts, articles)

    if all(
        sum(results[name]["counts"].values()) > 0
        for name in EXPERIMENTS
    ):
        plot_comparison(results)
    else:
        print(
            "\n[AVISO] Comparativo não gerado: "
            "um dos experimentos não tem avaliações."
        )

    pd.DataFrame(csv_rows).to_csv(
        OUTPUT / "validation_cross_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArquivos gerados em: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
