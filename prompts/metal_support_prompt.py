def build_metal_support_prompt(text):

    return f"""
Extract the active metal(s) and catalyst support explicitly mentioned
in the provided text.

For each metal and support, preserve the original representation and
identify any separate identifier, label, code, abbreviation, or
alternative designation explicitly assigned by the authors.

Definitions

Active Metal

The catalytically active metallic element explicitly mentioned by
the authors.

Examples:

Ru
Cu
Ni
Pd
Pt
Ag
Co
Fe

Raw Name

The metal representation exactly as it appears in the provided text.

Preserve the original spelling, capitalization, numbers, symbols,
chemical notation, and other relevant details.

Support

The material supporting the active metal.

Examples:

MgO
Al2O3
SiO2
TiO2
Hydroxyapatite
MgAl-LDO

Raw Name

The support representation exactly as it appears in the provided text.

Preserve the original spelling, capitalization, numbers, symbols,
chemical notation, hyphens, and other relevant details.

Article Identifier

A separate identifier explicitly assigned by the authors to refer to
a metal or support that already has another name or chemical
description.

Examples:

Ruthenium → R
Alumina → A
Silica-alumina → SA
Support 1 → S1
Metal A → M-A

Important:

An article identifier is NOT simply an abbreviation, chemical formula,
or shortened name.

For example:

Ru
MgO
Al2O3
Alumina
Ruthenium

must NOT automatically be considered article identifiers.

They are article identifiers only when the authors explicitly use them
as a separate designation for the corresponding metal or support.

Rules

- Use ONLY information explicitly present in the provided text.
- Do NOT infer.
- Do NOT guess.
- Do NOT invent an article identifier.
- Only include an article identifier when the authors explicitly use
  a separate identifier to refer to the corresponding metal or support.
- Preserve every article identifier exactly as written by the authors.
- Preserve capitalization, numbers, symbols, hyphens, and chemical notation.
- Do NOT normalize article identifiers.
- Do NOT translate article identifiers.
- Do NOT simplify article identifiers.
- Do NOT automatically treat a chemical formula as an article identifier.
- Do NOT automatically treat an abbreviation as an article identifier.
- Do NOT treat metal loading as an article identifier.
- Do NOT treat the catalyst composition as an article identifier.
- Do NOT assign a catalyst identifier to a metal or support unless the
  authors explicitly establish that relationship.
- If multiple active metals exist, return all of them.
- If multiple identifiers explicitly refer to the same metal or support,
  preserve all of them in "article_identifiers".
- If no active metal is explicitly mentioned, return [].
- If no support is explicitly identifiable, return null.
- Do NOT include catalytic sites.
- Do NOT include reaction intermediates.
- Do NOT include products.
- Return ONLY one valid JSON object.
- Do not explain your answer.
- Do not revise your answer.
- Stop immediately after the JSON.

Format

{{
    "metal": [
        {{
            "name": "",
            "raw_name": "",
            "article_identifiers": []
        }}
    ],
    "support": {{
        "name": "",
        "raw_name": "",
        "article_identifiers": []
    }}
}}

Provided Text

{text}
"""