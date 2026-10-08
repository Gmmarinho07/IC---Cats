"""
plot_cross_validation.py

Analisa os resultados da validação cruzada:

    GPT extraction    -> Claude judge
    Claude extraction -> GPT judge

IMPORTANTE:
Os arquivos antigos GPT -> GPT também podem estar presentes
em benchmark/validation/gpt/.

Por isso, o script NÃO usa apenas a pasta.
Ele verifica:
    extraction_model
    judge_model

para identificar corretamente cada experimento.
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


# =====================================================
# CONFIGURAÇÕES
# =====================================================

VALIDATION_FOLDER = Path("benchmark/validation")

OUTPUT_FOLDER = Path("benchmark/validation")


DECISIONS = [
    "correct",
    "partial",
    "incorrect",
    "not_supported"
]


LABELS = [
    "Corretas",
    "Parciais",
    "Incorretas",
    "Não sustentadas"
]


# =====================================================
# EXPERIMENTOS
# =====================================================

EXPERIMENTS = {
    "GPT → Claude": {
        "extraction_model": "gpt",
        "judge_model": "claude"
    },

    "Claude → GPT": {
        "extraction_model": "claude",
        "judge_model": "gpt"
    }
}


# =====================================================
# LEITURA
# =====================================================

def load_experiment(
    extraction_model,
    judge_model
):
    """
    Lê todos os JSONs e mantém somente aqueles
    pertencentes ao experimento especificado.
    """

    counts = Counter()

    articles = 0
    evaluations = 0
    errors = 0

    # Procuramos em ambas as pastas porque os resultados
    # podem estar organizados de acordo com o modelo
    # de extração.
    for folder_model in ["gpt", "claude"]:

        folder = (
            VALIDATION_FOLDER
            / folder_model
        )

        if not folder.exists():
            continue

        for file_path in folder.glob("*.json"):

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

                # -----------------------------------------
                # FILTRO DO EXPERIMENTO
                # -----------------------------------------

                if data.get("extraction_model") != extraction_model:
                    continue

                if data.get("judge_model") != judge_model:
                    continue

                # -----------------------------------------
                # VALIDAÇÃO
                # -----------------------------------------

                validation_data = data.get(
                    "validation",
                    {}
                )

                items = validation_data.get(
                    "validation",
                    []
                )

                if not isinstance(items, list):
                    continue

                articles += 1

                # -----------------------------------------
                # CONTAGEM
                # -----------------------------------------

                for item in items:

                    if not isinstance(item, dict):
                        continue

                    decision = item.get(
                        "decision"
                    )

                    if decision in DECISIONS:

                        counts[decision] += 1
                        evaluations += 1

            except (
                OSError,
                json.JSONDecodeError,
                TypeError,
                AttributeError
            ) as error:

                errors += 1

                print(
                    f"[ERRO] {file_path.name}: "
                    f"{error}"
                )

    return (
        counts,
        articles,
        evaluations,
        errors
    )


# =====================================================
# RESUMO
# =====================================================

def print_summary(
    experiment,
    counts,
    articles,
    evaluations
):

    print("\n" + "=" * 65)

    print(
        f"EXPERIMENTO: {experiment}"
    )

    print("=" * 65)

    print(
        f"Artigos: {articles}"
    )

    print(
        f"Entidades avaliadas: {evaluations}"
    )

    print()

    for decision, label in zip(
        DECISIONS,
        LABELS
    ):

        value = counts.get(
            decision,
            0
        )

        percentage = (
            value / evaluations * 100
            if evaluations > 0
            else 0
        )

        print(
            f"{label}: "
            f"{value} "
            f"({percentage:.1f}%)"
        )


# =====================================================
# GRÁFICO INDIVIDUAL
# =====================================================

def generate_individual_chart(
    experiment,
    counts,
    articles,
    evaluations
):

    values = [
        counts.get(
            decision,
            0
        )
        for decision in DECISIONS
    ]

    if evaluations == 0:

        print(
            f"[AVISO] Nenhum resultado para "
            f"{experiment}"
        )

        return

    percentages = [
        value / evaluations * 100
        for value in values
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        LABELS,
        values
    )

    for bar, value, percentage in zip(
        bars,
        values,
        percentages
    ):

        ax.annotate(
            f"{value}\n({percentage:.1f}%)",

            xy=(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height()
            ),

            xytext=(0, 6),

            textcoords="offset points",

            ha="center",

            va="bottom",

            fontsize=10
        )

    ax.set_title(
        f"Validação cruzada — {experiment}",
        fontsize=15,
        pad=18
    )

    ax.set_xlabel(
        "Categoria da avaliação"
    )

    ax.set_ylabel(
        "Quantidade de entidades"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)

    ax.set_ylim(
        0,
        max(values) * 1.18 + 1
    )

    fig.text(
        0.5,
        0.015,
        (
            f"Artigos: {articles} | "
            f"Entidades avaliadas: {evaluations}"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    filename = (
        experiment
        .replace(" → ", "_")
        .replace(" ", "")
        .lower()
    )

    output = (
        OUTPUT_FOLDER
        / f"validation_{filename}.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"[OK] Gráfico salvo: {output}"
    )

    plt.close(fig)


# =====================================================
# GRÁFICO COMPARATIVO
# =====================================================

def generate_comparison_chart(
    experiment_results
):

    experiments = list(
        experiment_results.keys()
    )

    # ---------------------------------------------
    # Percentuais
    # ---------------------------------------------

    percentages = {}

    for experiment in experiments:

        counts = experiment_results[
            experiment
        ]["counts"]

        total = sum(
            counts.values()
        )

        percentages[experiment] = [
            (
                counts.get(
                    decision,
                    0
                ) / total * 100
            )
            if total > 0
            else 0

            for decision in DECISIONS
        ]

    # ---------------------------------------------
    # POSIÇÕES
    # ---------------------------------------------

    import numpy as np

    x = np.arange(
        len(LABELS)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars1 = ax.bar(
        x - width / 2,
        percentages["GPT → Claude"],
        width,
        label="GPT → Claude"
    )

    bars2 = ax.bar(
        x + width / 2,
        percentages["Claude → GPT"],
        width,
        label="Claude → GPT"
    )

    # ---------------------------------------------
    # RÓTULOS
    # ---------------------------------------------

    for bars in [bars1, bars2]:

        for bar in bars:

            height = bar.get_height()

            ax.annotate(
                f"{height:.1f}%",

                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,

                    height
                ),

                xytext=(0, 4),

                textcoords="offset points",

                ha="center",

                va="bottom",

                fontsize=9
            )

    ax.set_title(
        "Comparação da validação cruzada",
        fontsize=16,
        pad=18
    )

    ax.set_ylabel(
        "Percentual das avaliações (%)"
    )

    ax.set_xlabel(
        "Categoria da avaliação"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        LABELS
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)

    ax.legend()

    plt.tight_layout()

    output = (
        OUTPUT_FOLDER
        / "validation_cross_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\n[OK] Gráfico comparativo salvo:"
    )

    print(output)

    plt.close(fig)


# =====================================================
# MAIN
# =====================================================

def main():

    print(
        "\n"
        + "=" * 65
    )

    print(
        "ANÁLISE DA VALIDAÇÃO CRUZADA"
    )

    print(
        "=" * 65
    )

    experiment_results = {}

    # ---------------------------------------------
    # PROCESSA OS DOIS EXPERIMENTOS
    # ---------------------------------------------

    for experiment, config in EXPERIMENTS.items():

        counts, articles, evaluations, errors = (
            load_experiment(
                config["extraction_model"],
                config["judge_model"]
            )
        )

        experiment_results[experiment] = {
            "counts": counts,
            "articles": articles,
            "evaluations": evaluations
        }

        print_summary(
            experiment,
            counts,
            articles,
            evaluations
        )

        generate_individual_chart(
            experiment,
            counts,
            articles,
            evaluations
        )

    # ---------------------------------------------
    # COMPARATIVO
    # ---------------------------------------------

    generate_comparison_chart(
        experiment_results
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "ANÁLISE CONCLUÍDA"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":

    main()