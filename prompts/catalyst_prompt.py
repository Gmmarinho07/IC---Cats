def build_catalyst_prompt(text):

    return f"""
Extract catalyst names explicitly mentioned in the provided text.

For each catalyst, preserve the original catalyst name and identify
any separate identifier, label, code, abbreviation, or alternative
designation explicitly assigned by the authors to that catalyst.

Definitions

Catalyst

The catalyst name or chemical composition explicitly mentioned by the authors.

Raw Name

The catalyst name exactly as it appears in the provided text.

Preserve the original spelling, capitalization, numbers, symbols,
chemical notation, hyphens, percentages, and other relevant details.

Article Identifier

A separate identifier explicitly assigned by the authors to refer to
a catalyst that already has another name or chemical description.

Examples:

Alumina → A
Silica-alumina → SA
H-ferrierite → FER
H-ZSM-5 → MFI
H-faujasite → USY
Calcined hydrotalcite → MgA

Other examples:

CAT-1
NC-10
N1
M-3
Catalyst A
Sample 1

Rules

- Use ONLY information explicitly present in the provided text.
- Do NOT infer.
- Do NOT guess.
- Do NOT invent an article identifier.
- Only include an article identifier when the authors explicitly use
  a separate identifier to refer to the catalyst.
- Preserve every article identifier exactly as written by the authors.
- Preserve capitalization, numbers, symbols, hyphens, and chemical notation.
- Do NOT normalize article identifiers.
- Do NOT translate article identifiers.
- Do NOT simplify article identifiers.
- Do NOT treat the complete catalyst name as an article identifier
  when it is simply the catalyst name itself.
- Do NOT treat a chemical formula as an article identifier unless the
  authors explicitly use it as an alternative identifier.
- Do NOT treat the metal loading as an article identifier.
- Do NOT treat the support name as an article identifier.
- Do NOT treat a complete catalyst composition as an article identifier
  unless the authors explicitly use it as an alternative designation.
- If the catalyst has no separate article identifier, return an empty list.
- If multiple separate identifiers explicitly refer to the same catalyst,
  preserve all of them in "article_identifiers".
- Do NOT create separate catalysts when an identifier is only another
  reference to an already identified catalyst.
- The "catalyst" field must preserve the catalyst name as currently
  extracted by the system.
- The "raw_name" field must contain the catalyst name exactly as written
  in the provided text.
- Do NOT include catalytic sites.
- Do NOT include reaction intermediates.
- Do NOT include products.
- Return ONLY one valid JSON object.
- Do not explain your answer.
- Do not revise your answer.
- Stop immediately after the JSON.

Format

{{
    "catalysts": [
        {{
            "catalyst": "",
            "raw_name": "",
            "article_identifiers": []
        }}
    ]
}}

Article Text

{text}
"""