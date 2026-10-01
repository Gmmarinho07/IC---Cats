
import json
from pathlib import Path

from agents.validation_judge import validate


# =========================
# CONFIGURAÇÕES
# =========================

RESULTS_FOLDER = Path("benchmark/results")
CONTEXT_FOLDER = Path("benchmark/contexts")
VALIDATION_FOLDER = Path("benchmark/validation")

# Modelos cujas extrações serão avaliadas.
EXTRACTION_MODELS = ["gpt"]

# Por enquanto, todas as validações serão feitas pelo GPT.
JUDGE_MODEL = "gpt"


# =========================
# FUNÇÕES AUXILIARES
# =========================

def load_json(file_path):
    """Carrega um arquivo JSON."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):
    """Salva os resultados da validação em JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# =========================
# PROCESSAMENTO
# =========================

def process_file(model, result_path):
    """Valida a extração de um artigo usando o GPT."""

    paper_name = result_path.stem

    context_path = CONTEXT_FOLDER / f"{paper_name}.txt"
    output_path = (
        VALIDATION_FOLDER
        / model
        / f"{paper_name}.json"
    )

    print(f"\n{'=' * 60}")
    print(f"Artigo: {paper_name}")
    print(f"Modelo de extração: {model}")
    print(f"Modelo juiz: {JUDGE_MODEL}")
    print(f"{'=' * 60}")

    # Evita reprocessar resultados já validados.
    if output_path.exists():
        print(f"[SKIP] Validação já existe: {output_path}")
        return

    # Verifica se o contexto do artigo existe.
    if not context_path.exists():
        print(f"[ERRO] Contexto não encontrado: {context_path}")
        return

    try:
        # Carrega a extração produzida pelo modelo.
        extraction = load_json(result_path)

        # Carrega o contexto textual do artigo.
        context = context_path.read_text(encoding="utf-8")

        # Executa o agente juiz.
        validation_result = validate(
            context=context,
            extraction=extraction,
            model=JUDGE_MODEL
        )

        # Organiza o resultado final.
        output = {
            "paper": paper_name,
            "extraction_model": model,
            "judge_model": JUDGE_MODEL,
            "validation": validation_result
        }

        # Salva o JSON de validação.
        save_json(output, output_path)

        print(f"[OK] Validação salva em: {output_path}")

    except Exception as error:
        print(f"[ERRO] Falha ao validar {paper_name}: {error}")


# =========================
# EXECUÇÃO PRINCIPAL
# =========================

def main():
    """Percorre os resultados e executa as validações."""

    total_files = 0

    for model in EXTRACTION_MODELS:
        model_folder = RESULTS_FOLDER / model

        if not model_folder.exists():
            print(f"[AVISO] Pasta não encontrada: {model_folder}")
            continue

        result_files = sorted(model_folder.glob("*.json"))

        if not result_files:
            print(f"[AVISO] Nenhum JSON encontrado em {model_folder}")
            continue

        print(f"\nModelo de extração: {model}")
        print(f"Artigos encontrados: {len(result_files)}")

        for result_path in result_files:
            total_files += 1
            process_file(model, result_path)

    print(f"\n{'=' * 60}")
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"Arquivos encontrados: {total_files}")
    print(f"Resultados em: {VALIDATION_FOLDER}")
    print(f"Modelo juiz utilizado: {JUDGE_MODEL}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
