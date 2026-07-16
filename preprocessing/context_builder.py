"""
context_builder.py

Monta o contexto final que será enviado ao LLM.
"""

from collections import defaultdict


# =====================================================
# BUILD CONTEXT
# =====================================================

def build_context(selected_chunks):

    grouped = defaultdict(list)

    # Agrupa por seção
    for chunk in selected_chunks:

        grouped[chunk["section"]].append(chunk)

    # Ordem desejada
    section_order = [

        "abstract",

        "experimental",

        "results",

        "introduction",

        "conclusion"

    ]

    context = ""

    for section in section_order:

        if section not in grouped:

            continue

        context += f"\n========== {section.upper()} ==========\n\n"

        # Mantém ordem dos chunks
        grouped[section].sort(
            key=lambda x: x["chunk_id"]
        )

        for chunk in grouped[section]:

            context += chunk["text"]

            context += "\n\n"

    return context.strip()


# =====================================================
# ESTATÍSTICAS
# =====================================================

def context_statistics(context):

    return {

        "characters": len(context),

        "words": len(context.split())

    }


# =====================================================
# DEBUG
# =====================================================

def preview_context(context, chars=1200):

    print()

    print("=" * 70)

    print("CONTEXT PREVIEW")

    print("=" * 70)

    print()

    print(context[:chars])

    print()

    print("=" * 70)