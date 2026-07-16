"""
section_extractor.py

Extrai automaticamente as principais seções de um artigo científico.

Objetivo:
- Detectar os cabeçalhos das seções.
- Funcionar em artigos de diferentes revistas.
- Retornar um dicionário com o conteúdo de cada seção.
"""

import re


# =====================================================
# ALIASES DAS SEÇÕES
# =====================================================

SECTION_PATTERNS = {

    "abstract": [

        r"\babstract\b",
        r"\bsummary\b"

    ],

    "introduction": [

        r"\bintroduction\b"

    ],

    "experimental": [

        r"\bexperimental\b",
        r"\bexperimental section\b",
        r"\bmaterials and methods\b",
        r"\bmethodology\b",
        r"\bcatalyst preparation\b",
        r"\bexperimental methods\b"

    ],

    "results": [

        r"\bresults\b",
        r"\bresults and discussion\b",
        r"\bdiscussion\b"

    ],

    "conclusion": [

        r"\bconclusion\b",
        r"\bconclusions\b"

    ],

    "references": [

        r"\breferences\b",
        r"\bbibliography\b"

    ]

}


# =====================================================
# LIMPEZA
# =====================================================

def clean_text(text):

    text = text.replace("\r", "")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =====================================================
# LOCALIZA UMA SEÇÃO
# =====================================================

def find_section_positions(text):

    lower = text.lower()

    positions = []

    for section_name, patterns in SECTION_PATTERNS.items():

        for pattern in patterns:

            match = re.search(pattern, lower)

            if match:

                positions.append({

                    "name": section_name,

                    "start": match.start()

                })

                break

    positions.sort(key=lambda x: x["start"])

    return positions


# =====================================================
# EXTRAI AS SEÇÕES
# =====================================================

def extract_sections(text):

    text = clean_text(text)

    positions = find_section_positions(text)

    sections = {}

    if not positions:

        return {

            "full_text": text

        }

    for i, current in enumerate(positions):

        start = current["start"]

        if i < len(positions) - 1:

            end = positions[i + 1]["start"]

        else:

            end = len(text)

        sections[current["name"]] = text[start:end].strip()

    return sections


# =====================================================
# ESTATÍSTICAS
# =====================================================

def section_statistics(sections):

    stats = {}

    for name, content in sections.items():

        stats[name] = {

            "characters": len(content),

            "words": len(content.split())

        }

    return stats


# =====================================================
# DEBUG
# =====================================================

def preview_sections(sections):

    print("\n==============================")
    print("SECTIONS FOUND")
    print("==============================")

    for name, content in sections.items():

        print(f"\n[{name.upper()}]")

        print("-" * 50)

        print(content[:500])

        print("\n...")