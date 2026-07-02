def build_catalyst_prompt(text):

    return f"""
Extract catalyst names explicitly mentioned in the abstract.

Rules

- Use ONLY information explicitly present in the abstract.
- Do NOT infer.
- Do NOT guess.
- Do NOT include catalytic sites.
- Do NOT include reaction intermediates.
- Do NOT include products.
- Return ONLY catalyst names.
- Return exactly one JSON object.
- Do not explain your answer.
- Stop immediately after the JSON.

Format

{{
    "catalysts": []
}}

Abstract

{text}
"""