# Agent 1 - Catalyst

## Prompt v1
.Extract catalyst names explicitly mentioned in the abstract.

Rules:
- Use only information present in the text.
- Do not infer.
- Do not guess.
- Return ONLY valid JSON.

Format:

{{
    "catalysts": []
}}

Abstract:

{text}
"""


Accuracy:
GPT = 90%
Claude = 90%

## Prompt v2
Extract catalyst names explicitly mentioned in the abstract.

Rules:
- Use only information present in the text.
- Do not infer.
- Do not guess.
- Return ONLY valid JSON.

Format:

{{
    "catalysts": []
}}

Abstract:

{text}
"""


Accuracy:
GPT = 100%
Claude = 90%

Observações:
...