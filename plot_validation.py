
"""
plot_validation.py

Gera gráficos das avaliações do Agente 3 (GPT)
do projeto IC-CATS.

Gráfico 1:
    Distribuição geral das classificações.

Gráfico 2:
    Distribuição das classificações por artigo,
    em barras horizontais empilhadas, ordenadas
    pela quantidade de avaliações não corretas.

Entrada:
    benchmark/validation/gpt/*.json

Saídas:
    benchmark/validation/validation_gpt.png
    benchmark/validation/validation_gpt_por_artigo.png
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


# =====================================================
# CONFIGURAÇÕES
# =====================================================

VALIDATION_FOLDER = Path("benchmark/validation/gpt")

OUTPUT_FOLDER = Path("benchmark/validation")

OUTPUT_GENERAL = OUTPUT_FOLDER / "validation_gpt.png"

OUTPUT_ARTICLES = (
    OUTPUT_FOLDER / "validation_gpt_por_artigo.png"
)

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
# LEITURA DOS RESULTADOS
# =====================================================

def load_validations():
    """
    Lê os JSONs de validação do GPT e contabiliza
    as classificações gerais e por artigo.

    Retorna:
        counts: contador geral das decisões.
        article_counts: contador de decisões por artigo.
        files_processed: quantidade de arquivos lidos.
        files_with_errors: quantidade de arquivos com erro.
    """

    counts = Counter()
    article_counts = {}

    files_processed = 0
    files_with_errors = 0

    if not VALIDATION_FOLDER.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {VALIDATION_FOLDER}"
        )

    json_files = sorted(
        VALIDATION_FOLDER.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"Nenhum JSON encontrado em {VALIDATION_FOLDER}"
        )

    print(f"\n{len(json_files)} arquivos encontrados.")

    for file_path in json_files:

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            # Estrutura esperada:
            # {
            #     "paper": "...",
            #     "extraction_model": "gpt",
            #     "judge_model": "gpt",
            #     "validation": {
            #         "validation": [...],
            #         "summary": {...}
            #     }
            # }

            validation_data = data.get("validation", {})

            if not isinstance(validation_data, dict):
                raise ValueError(
                    "Estrutura inválida na chave 'validation'."
                )

            items = validation_data.get("validation", [])

            if not isinstance(items, list):
                raise ValueError(
                    "A lista de avaliações está inválida."
                )

            paper_name = data.get(
                "paper",
                file_path.stem
            )

            article_counts[paper_name] = Counter()

            for item in items:

                if not isinstance(item, dict):
                    print(
                        f"[AVISO] Item inválido em "
                        f"{file_path.name}"
                    )
                    continue

                decision = item.get("decision")

                if decision not in DECISIONS:
                    print(
                        f"[AVISO] Decisão desconhecida em "
                        f"{file_path.name}: {decision}"
                    )
                    continue

                # Contabilização geral.
                counts[decision] += 1

                # Contabilização por artigo.
                article_counts[paper_name][decision] += 1

            files_processed += 1

            print(f"[OK] {paper_name}")

        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            AttributeError,
            TypeError
        ) as error:

            files_with_errors += 1

            print(
                f"[ERRO] Falha ao ler "
                f"{file_path.name}: {error}"
            )

    return (
        counts,
        article_counts,
        files_processed,
        files_with_errors
    )


# =====================================================
# GRÁFICO 1: DISTRIBUIÇÃO GERAL
# =====================================================

def generate_general_chart(counts, files_processed):
    """
    Gera um gráfico de barras com a quantidade
    e o percentual de cada classificação.
    """

    values = [
        counts.get(decision, 0)
        for decision in DECISIONS
    ]

    total = sum(values)

    if total == 0:
        print(
            "\n[AVISO] Nenhuma avaliação válida "
            "para gerar o gráfico geral."
        )
        return

    percentages = [
        (value / total) * 100
        for value in values
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.bar(
        LABELS,
        values
    )

    # Quantidade e percentual sobre cada barra.
    for bar, value, percentage in zip(
        bars,
        values,
        percentages
    ):

        ax.annotate(
            f"{value}\n({percentage:.1f}%)",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(
        "Distribuição das avaliações do Agente 3 (GPT)",
        fontsize=15,
        pad=18
    )

    ax.set_xlabel(
        "Categoria da avaliação",
        fontsize=11
    )

    ax.set_ylabel(
        "Quantidade de entidades",
        fontsize=11
    )

    ax.set_ylim(
        0,
        max(values) * 1.2 + 1
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.015,
        (
            f"Artigos analisados: {files_processed} | "
            f"Total de avaliações: {total}"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_GENERAL,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\n[OK] Gráfico geral salvo em: "
        f"{OUTPUT_GENERAL}"
    )

    plt.show()
    plt.close(fig)


# =====================================================
# GRÁFICO 2: DISTRIBUIÇÃO POR ARTIGO
# =====================================================

def generate_article_chart(article_counts, files_processed):
    """
    Gera um gráfico de barras horizontais empilhadas,
    mostrando a distribuição das classificações por
    artigo.

    Os artigos são ordenados pela quantidade de
    avaliações não corretas, em ordem decrescente.
    """

    if not article_counts:
        print(
            "\n[AVISO] Nenhum dado por artigo "
            "para gerar o segundo gráfico."
        )
        return

    # Ordena pelos artigos com mais avaliações
    # parciais, incorretas e não sustentadas.
    articles = sorted(
        article_counts.keys(),
        key=lambda article: (
            sum(
                article_counts[article].get(
                    decision,
                    0
                )
                for decision in (
                    "partial",
                    "incorrect",
                    "not_supported"
                )
            ),
            article_counts[article].get(
                "correct",
                0
            )
        ),
        reverse=True
    )

    # Quantidades por categoria e artigo.
    values = {
        decision: [
            article_counts[article].get(
                decision,
                0
            )
            for article in articles
        ]
        for decision in DECISIONS
    }

    # Altura dinâmica para manter os nomes legíveis.
    fig_height = max(
        8,
        len(articles) * 0.30
    )

    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    left = [0] * len(articles)

    # Barras horizontais empilhadas.
    for decision, label in zip(
        DECISIONS,
        LABELS
    ):

        ax.barh(
            articles,
            values[decision],
            left=left,
            label=label
        )

        left = [
            previous + current
            for previous, current in zip(
                left,
                values[decision]
            )
        ]

    ax.set_title(
        "Distribuição das avaliações do Agente 3 por artigo",
        fontsize=15,
        pad=18
    )

    ax.set_xlabel(
        "Quantidade de entidades avaliadas",
        fontsize=11
    )

    ax.set_ylabel(
        "Artigo",
        fontsize=11
    )

    # Artigos com mais avaliações não corretas no topo.
    ax.invert_yaxis()

    ax.legend(
        title="Classificação",
        loc="lower right"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        (
            f"Artigos analisados: {files_processed} | "
            "Ordenação: avaliações não corretas "
            "(decrescente)"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.025, 1, 1]
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_ARTICLES,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\n[OK] Gráfico por artigo salvo em: "
        f"{OUTPUT_ARTICLES}"
    )

    plt.show()
    plt.close(fig)


# =====================================================
# RESUMO DOS RESULTADOS
# =====================================================

def print_summary(
    counts,
    files_processed,
    files_with_errors
):
    """
    Exibe no terminal o resumo das avaliações.
    """

    total = sum(counts.values())

    print("\n" + "=" * 55)
    print("RESUMO DA VALIDAÇÃO GPT")
    print("=" * 55)

    for decision, label in zip(
        DECISIONS,
        LABELS
    ):

        value = counts.get(
            decision,
            0
        )

        percentage = (
            (value / total) * 100
            if total > 0
            else 0
        )

        print(
            f"{label}: {value} "
            f"({percentage:.1f}%)"
        )

    print("-" * 55)
    print(f"Artigos lidos: {files_processed}")
    print(f"Arquivos com erro: {files_with_errors}")
    print(f"Total de avaliações: {total}")
    print("=" * 55)


# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================

def main():

    print(
        "\nLendo os resultados da validação GPT..."
    )

    (
        counts,
        article_counts,
        files_processed,
        files_with_errors
    ) = load_validations()

    # Exibe o resumo.
    print_summary(
        counts,
        files_processed,
        files_with_errors
    )

    # Gera o gráfico geral.
    generate_general_chart(
        counts,
        files_processed
    )

    # Gera o gráfico por artigo.
    generate_article_chart(
        article_counts,
        files_processed
    )

    print("\nProcessamento dos gráficos concluído.")


if __name__ == "__main__":
    main()
