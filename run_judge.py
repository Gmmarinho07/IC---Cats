"""
run_judge.py

Executa o Agente 3 em validação cruzada.

Experimentos:

    GPT extraction    -> Claude judge
    Claude extraction -> GPT judge

Os resultados antigos GPT -> GPT são preservados.

Estrutura:

benchmark/
├── results/
│   ├── gpt/
│   └── claude/
│
├── contexts/
│
└── validation/
    ├── gpt/
    │   └── resultados antigos GPT -> GPT
    │
    └── cross/
        ├── gpt_to_claude/
        └── claude_to_gpt/
"""

import json
from pathlib import Path

from agents.validation_judge import validate


# =====================================================
# CONFIGURAÇÕES
# =====================================================

RESULTS_FOLDER = Path("benchmark/results")

CONTEXT_FOLDER = Path("benchmark/contexts")

VALIDATION_FOLDER = Path(
    "benchmark/validation/cross"
)


# =====================================================
# VALIDAÇÃO CRUZADA
# =====================================================

EXPERIMENTS = {
    "gpt_to_claude": {
        "extraction_model": "gpt",
        "judge_model": "claude"
    },

    "claude_to_gpt": {
        "extraction_model": "claude",
        "judge_model": "gpt"
    }
}


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(data, file_path):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


# =====================================================
# PROCESSAMENTO DE UM ARTIGO
# =====================================================

def process_file(
    result_path,
    extraction_model,
    judge_model,
    output_folder
):

    paper_name = result_path.stem

    context_path = (
        CONTEXT_FOLDER
        / f"{paper_name}.txt"
    )

    output_path = (
        output_folder
        / f"{paper_name}.json"
    )

    print("\n" + "=" * 70)

    print(
        f"Artigo: {paper_name}"
    )

    print(
        f"Extração: {extraction_model.upper()}"
    )

    print(
        f"Juiz: {judge_model.upper()}"
    )

    print("=" * 70)

    # -------------------------------------------------
    # EVITA REPETIÇÃO
    # -------------------------------------------------

    if output_path.exists():

        print(
            f"[SKIP] Já existe: "
            f"{output_path}"
        )

        return "skipped"

    # -------------------------------------------------
    # VERIFICA CONTEXTO
    # -------------------------------------------------

    if not context_path.exists():

        print(
            f"[ERRO] Contexto não encontrado: "
            f"{context_path}"
        )

        return "error"

    try:

        # -------------------------------------------------
        # CARREGA EXTRAÇÃO
        # -------------------------------------------------

        extraction = load_json(
            result_path
        )

        # -------------------------------------------------
        # CARREGA CONTEXTO
        # -------------------------------------------------

        context = context_path.read_text(
            encoding="utf-8"
        )

        # -------------------------------------------------
        # EXECUTA AGENTE 3
        # -------------------------------------------------

        validation_result = validate(
            context=context,
            extraction=extraction,
            model=judge_model
        )

        # -------------------------------------------------
        # ORGANIZA RESULTADO
        # -------------------------------------------------

        output = {

            "paper": paper_name,

            "extraction_model":
                extraction_model,

            "judge_model":
                judge_model,

            "validation":
                validation_result
        }

        # -------------------------------------------------
        # SALVA
        # -------------------------------------------------

        save_json(
            output,
            output_path
        )

        print(
            f"[OK] Validação salva em:"
        )

        print(
            output_path
        )

        return "processed"

    except Exception as error:

        print(
            f"[ERRO] Falha em "
            f"{paper_name}: {error}"
        )

    return "errors"


# =====================================================
# EXECUTA UM EXPERIMENTO
# =====================================================

def run_experiment(
    experiment_name,
    extraction_model,
    judge_model
):

    print("\n")
    print("#" * 70)

    print(
        f"EXPERIMENTO: "
        f"{experiment_name}"
    )

    print(
        f"Extração: "
        f"{extraction_model.upper()}"
    )

    print(
        f"Juiz: "
        f"{judge_model.upper()}"
    )

    print("#" * 70)

    # -------------------------------------------------
    # PASTA DAS EXTRAÇÕES
    # -------------------------------------------------

    results_folder = (
        RESULTS_FOLDER
        / extraction_model
    )

    if not results_folder.exists():

        print(
            f"[ERRO] Pasta não encontrada: "
            f"{results_folder}"
        )

        return {
            "found": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0
        }

    # -------------------------------------------------
    # PASTA DE SAÍDA
    # -------------------------------------------------

    output_folder = (
        VALIDATION_FOLDER
        / experiment_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # LOCALIZA JSONS
    # -------------------------------------------------

    result_files = sorted(
        results_folder.glob("*.json")
    )

    print(
        f"Artigos encontrados: "
        f"{len(result_files)}"
    )

    stats = {
        "found": len(result_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0
    }

    # -------------------------------------------------
    # PROCESSAMENTO
    # -------------------------------------------------

    for result_path in result_files:

        status = process_file(
            result_path=result_path,

            extraction_model=
                extraction_model,

            judge_model=
                judge_model,

            output_folder=
                output_folder
        )

        stats[status] += 1

    return stats


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 70)

    print(
        "VALIDAÇÃO CRUZADA — IC-CATS"
    )

    print("=" * 70)

    total = {
        "found": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 0
    }

    # =================================================
    # GPT → CLAUDE
    # =================================================

    stats_gpt_claude = run_experiment(
        experiment_name="gpt_to_claude",

        extraction_model="gpt",

        judge_model="claude"
    )

    # =================================================
    # CLAUDE → GPT
    # =================================================

    stats_claude_gpt = run_experiment(
        experiment_name="claude_to_gpt",

        extraction_model="claude",

        judge_model="gpt"
    )

    # =================================================
    # SOMA
    # =================================================

    for stats in [
        stats_gpt_claude,
        stats_claude_gpt
    ]:

        for key in total:

            total[key] += stats[key]

    # =================================================
    # RESUMO
    # =================================================

    print("\n" + "=" * 70)

    print(
        "PROCESSAMENTO CONCLUÍDO"
    )

    print("=" * 70)

    print(
        f"Arquivos encontrados: "
        f"{total['found']}"
    )

    print(
        f"Arquivos processados: "
        f"{total['processed']}"
    )

    print(
        f"Arquivos ignorados: "
        f"{total['skipped']}"
    )

    print(
        f"Arquivos com erro: "
        f"{total['errors']}"
    )

    print("\nExperimentos:")

    print(
        "  GPT    → Claude"
    )

    print(
        "  Claude → GPT"
    )

    print("\nResultados em:")

    print(
        VALIDATION_FOLDER
    )

    print("=" * 70)


# =====================================================
# EXECUÇÃO
# =====================================================

if __name__ == "__main__":

    main()