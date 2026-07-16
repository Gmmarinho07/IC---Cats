"""
main.py

Pipeline principal do IC-CATS
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

from agents.catalyst import extract


# =====================================================
# CONFIGURAÇÕES
# =====================================================

PDF_FOLDER = "Papers"

OUTPUT_FOLDER = "benchmark/results"

MODEL = "gpt"
# MODEL = "claude"

SELECTION_POLICY = {

    "abstract": 1,

    "experimental": 2,

    "results": 2

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

    result = extract(

        context,

        MODEL

    )

    return result


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

        result = process_pdf(pdf_path)

        output_path = os.path.join(

            OUTPUT_FOLDER,

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

        print(f"\nResultado salvo em:")

        print(output_path)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    process_all()