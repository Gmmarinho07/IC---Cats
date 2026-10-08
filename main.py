"""
main.py

Pipeline principal do IC-CATS

Fluxo:
PDF -> Extração de texto -> Seções -> Chunks -> Ranking
-> Seleção de contexto -> Agentes de extração -> JSON

A validação pelo Agente 3 é executada separadamente
através do arquivo run_judge.py.
"""

import json
import os

from extractor import extract_text

from preprocessing.section_extractor import extract_sections
from preprocessing.chunk_builder import build_chunks
from preprocessing.chunk_ranker import (
    rank_chunks,
    select_chunks,
    preview_ranking
)
from preprocessing.context_builder import (
    build_context,
    preview_context,
    context_statistics
)

from agents.catalyst import extract as extract_catalyst
from agents.metal_support import extract as extract_metal_support


# =====================================================
# CONFIGURAÇÕES
# =====================================================

PDF_FOLDER = "Papers"

OUTPUT_FOLDER = "benchmark/results"

CONTEXT_FOLDER = "benchmark/contexts"

MODELS = [
    "gpt",
    "claude"
]

SELECTION_POLICY = {
    "abstract": 1,
    "experimental": 2,
    "results": 4
}


# =====================================================
# PROCESSAMENTO DE UM PDF
# =====================================================

def process_pdf(pdf_path):

    print("\n" + "=" * 70)
    print(f"Processing: {os.path.basename(pdf_path)}")
    print("=" * 70)

    # -------------------------------------------------
    # EXTRAÇÃO DO TEXTO
    # -------------------------------------------------

    text = extract_text(pdf_path)

    # -------------------------------------------------
    # EXTRAÇÃO DAS SEÇÕES
    # -------------------------------------------------

    sections = extract_sections(text)

    # Remove seções que não serão utilizadas

    sections.pop("references", None)
    sections.pop("introduction", None)
    sections.pop("conclusion", None)

    # -------------------------------------------------
    # GERAÇÃO DOS CHUNKS
    # -------------------------------------------------

    chunks = build_chunks(sections)

    print(f"\nChunks gerados: {len(chunks)}")

    # -------------------------------------------------
    # RANKING DOS CHUNKS
    # -------------------------------------------------

    ranked_chunks = rank_chunks(chunks)

    preview_ranking(ranked_chunks, n=5)

    # -------------------------------------------------
    # SELEÇÃO DOS CHUNKS
    # -------------------------------------------------

    selected_chunks = select_chunks(
        ranked_chunks,
        SELECTION_POLICY
    )

    # -------------------------------------------------
    # CONSTRUÇÃO DO CONTEXTO
    # -------------------------------------------------

    context = build_context(selected_chunks)

    preview_context(context)

    print()
    print(context_statistics(context))

    # -------------------------------------------------
    # AGENTES DE EXTRAÇÃO
    # -------------------------------------------------

    results = {}

    for model in MODELS:

        print("\n" + "-" * 70)
        print(f"Running extraction with {model.upper()}...")
        print("-" * 70)

        # ---------------------------------------------
        # AGENTE 1: CATALISADORES
        # ---------------------------------------------

        catalyst_result = extract_catalyst(
            context,
            model
        )

        # ---------------------------------------------
        # AGENTE 2: METAIS E SUPORTES
        # ---------------------------------------------

        metal_support_result = extract_metal_support(
            context,
            model
        )

        # ---------------------------------------------
        # RESULTADO DOS AGENTES DE EXTRAÇÃO
        # ---------------------------------------------

        results[model] = {
            "catalyst": catalyst_result,
            "metal_support": metal_support_result
        }

        print(
            f"\nExtraction completed for {model.upper()}."
        )

    return results, context


# =====================================================
# PROCESSAMENTO DE TODOS OS PDFs
# =====================================================

def process_all():

    # -------------------------------------------------
    # CRIAÇÃO DAS PASTAS
    # -------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        CONTEXT_FOLDER,
        exist_ok=True
    )

    # -------------------------------------------------
    # LOCALIZAÇÃO DOS PDFs
    # -------------------------------------------------

    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(
            f"Pasta de PDFs não encontrada: {PDF_FOLDER}"
        )

    pdfs = sorted(
        [
            pdf
            for pdf in os.listdir(PDF_FOLDER)
            if pdf.lower().endswith(".pdf")
        ]
    )

    print(f"\n{len(pdfs)} PDFs encontrados.")

    # -------------------------------------------------
    # PROCESSAMENTO DOS PDFs
    # -------------------------------------------------

    for pdf in pdfs:

        pdf_path = os.path.join(
            PDF_FOLDER,
            pdf
        )

        # Executa os agentes de extração

        results, context = process_pdf(pdf_path)

        # ---------------------------------------------
        # SALVAMENTO DO CONTEXTO
        # ---------------------------------------------

        context_filename = (
            os.path.splitext(pdf)[0] + ".txt"
        )

        context_path = os.path.join(
            CONTEXT_FOLDER,
            context_filename
        )

        with open(
            context_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(context)

        print("\nContexto salvo em:")
        print(context_path)

        # ---------------------------------------------
        # SALVAMENTO DOS RESULTADOS
        # ---------------------------------------------

        for model, result in results.items():

            model_folder = os.path.join(
                OUTPUT_FOLDER,
                model
            )

            os.makedirs(
                model_folder,
                exist_ok=True
            )

            # Nome do arquivo JSON

            output_filename = (
                os.path.splitext(pdf)[0] + ".json"
            )

            output_path = os.path.join(
                model_folder,
                output_filename
            )

            # Salva o resultado da extração

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                f"\nResultado {model.upper()} salvo em:"
            )

            print(output_path)


# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================

if __name__ == "__main__":

    process_all()