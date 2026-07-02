def build_metal_support_prompt(text):

    return f"""
Extract the active metal(s) and catalyst support explicitly mentioned in the abstract.

Definitions

Active Metal

The catalytically active metallic element.

Examples:

Ru
Cu
Ni
Pd
Pt
Ag
Co
Fe

Support

The material supporting the active metal.

Examples:

MgO
Al2O3
SiO2
TiO2
Hydroxyapatite
MgAl-LDO

Rules

- Use ONLY information explicitly present in the abstract.
- Do NOT infer.
- Do NOT guess.
- If the catalyst appears as Ru/MgO:

metal:
["Ru"]

support:
"MgO"

- If multiple active metals exist, return all of them.
- If no active metal is explicitly mentioned, return [].
- If no support is explicitly identifiable, return null.
- Return ONLY one valid JSON object.
- Do not explain your answer.
- Do not revise your answer.
- Stop immediately after the JSON.

Format

{{
    "metal": [],
    "support": null
}}

Abstract

{text}
"""