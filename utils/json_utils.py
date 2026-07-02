import json
import re


# =====================================================
# REMOVE MARKDOWN
# =====================================================

def remove_markdown(content):

    content = content.replace("```json", "")
    content = content.replace("```", "")

    return content.strip()


# =====================================================
# EXTRAI O PRIMEIRO JSON VÁLIDO
# =====================================================

def extract_first_json(content):

    decoder = json.JSONDecoder()

    match = re.search(r"\{", content)

    if not match:
        raise ValueError("Nenhum objeto JSON encontrado.")

    start = match.start()

    obj, _ = decoder.raw_decode(content[start:])

    return obj


# =====================================================
# LIMPEZA COMPLETA
# =====================================================

def clean_json(content):

    content = remove_markdown(content)

    return extract_first_json(content)