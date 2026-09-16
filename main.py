"""
main.py

Pipeline principal do IC-CATS
"""

import json
import os

from matplotlib.style import context

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
    # SEÇÕES
    # -------------------------------------------------

    sections = extract_sections(text)

    # Remove seções que não serão utilizadas

    sections.pop("references", None)
    sections.pop("introduction", None)
    sections.pop("conclusion", None)

    # -------------------------------------------------
    # CHUNKS
    # -------------------------------------------------

    chunks = build_chunks(sections)

    print(f"\nChunks gerados: {len(chunks)}")

    # -------------------------------------------------
    # RANKING
    # -------------------------------------------------

    ranked_chunks = rank_chunks(chunks)

    preview_ranking(ranked_chunks, n=5)

    # -------------------------------------------------
    # SELEÇÃO
    # -------------------------------------------------

    selected_chunks = select_chunks(

        ranked_chunks,

        SELECTION_POLICY

    )

    # -------------------------------------------------
    # CONTEXTO
    # -------------------------------------------------

    context = build_context(selected_chunks)

    preview_context(context)

    print()

    print(context_statistics(context))

    # -------------------------------------------------
    # LLM
    # -------------------------------------------------

    results = {}

    for model in MODELS:

        print(f"\nRunning {model.upper()}...")

        catalyst_result = extract_catalyst(
            context,
            model
        )

        metal_support_result = extract_metal_support(
            context,
            model
        )

        results[model] = {
            "catalyst": catalyst_result,
            "metal_support": metal_support_result
        }

    return results


# =====================================================
# TODOS OS PDFs
# =====================================================

def process_all():

    os.makedirs(

        OUTPUT_FOLDER,

        exist_ok=True

    )

    pdfs = sorted(

        [

            pdf

            for pdf in os.listdir(PDF_FOLDER)

            if pdf.lower().endswith(".pdf")

        ]

    )

    print(f"\n{len(pdfs)} PDFs encontrados.")

    for pdf in pdfs:

        pdf_path = os.path.join(

            PDF_FOLDER,

            pdf

        )

        # Executa TODOS os modelos
        results = process_pdf(pdf_path)

        # Salva o resultado de cada modelo
        for model, result in results.items():

            model_folder = os.path.join(

                OUTPUT_FOLDER,

                model

            )

            os.makedirs(

                model_folder,

                exist_ok=True

            )

            output_path = os.path.join(

                model_folder,

                pdf.replace(".pdf", ".json")

            )

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

            print(f"\nResultado {model.upper()} salvo em:")

            print(output_path)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    process_all()