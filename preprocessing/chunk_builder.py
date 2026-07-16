"""
chunk_builder.py

Constrói chunks a partir das seções extraídas do artigo.

Entrada:
{
    "abstract": "...",
    "experimental": "...",
    "results": "...",
    ...
}

Saída:
[
    {
        "section": "abstract",
        "chunk_id": 1,
        "text": "..."
    },
    ...
]
"""

import re

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_OVERLAP = 300


# =====================================================
# LIMPEZA
# =====================================================

def clean_text(text):

    text = text.replace("\r", "")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =====================================================
# BUILD CHUNKS DE UMA SEÇÃO
# =====================================================

def build_section_chunks(
    section_name,
    section_text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):

    section_text = clean_text(section_text)

    chunks = []

    start = 0

    chunk_id = 1

    while start < len(section_text):

        end = start + chunk_size

        chunks.append({

            "section": section_name,

            "chunk_id": chunk_id,

            "text": section_text[start:end]

        })

        chunk_id += 1

        if end >= len(section_text):
            break

        start += chunk_size - overlap

    return chunks


# =====================================================
# BUILD CHUNKS DE TODAS AS SEÇÕES
# =====================================================

def build_chunks(
    sections,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):

    all_chunks = []

    for section_name, section_text in sections.items():

        if len(section_text.strip()) == 0:
            continue

        section_chunks = build_section_chunks(

            section_name,

            section_text,

            chunk_size,

            overlap

        )

        all_chunks.extend(section_chunks)

    return all_chunks


# =====================================================
# ESTATÍSTICAS
# =====================================================

def chunk_statistics(chunks):

    if not chunks:

        return {

            "total_chunks": 0,

            "average_size": 0

        }

    sizes = [

        len(chunk["text"])

        for chunk in chunks

    ]

    return {

        "total_chunks": len(chunks),

        "average_size": round(

            sum(sizes)/len(sizes),

            2

        ),

        "max_size": max(sizes),

        "min_size": min(sizes)

    }


# =====================================================
# PREVIEW
# =====================================================

def preview_chunks(chunks, n=10):

    print("\n==============================")
    print("CHUNK PREVIEW")
    print("==============================")

    for chunk in chunks[:n]:

        print()

        print(
            f"[{chunk['section'].upper()}]"
            f" Chunk {chunk['chunk_id']}"
        )

        print("-"*60)

        print(chunk["text"][:500])

        print("\n...")