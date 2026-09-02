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

Cada módulo possui uma responsabilidade específica.

## agents/

Responsável pela lógica dos agentes de extração.

Atualmente:

* catalyst.py
* metal_support.py

---

## preprocessing/

Responsável pelo processamento e seleção do texto extraído dos artigos.

Principais componentes:

* section_extractor.py
* chunk_builder.py
* chunk_ranker.py
* context_builder.py

---

## evaluation/

Responsável pela avaliação das extrações.

Principais componentes:

* similarity.py
* metrics.py
* comparator.py

---

## benchmark/

Responsável pela execução e armazenamento dos resultados do benchmark.

Estrutura:

```text
benchmark/
    results/
        gpt/
        claude/

    comparison_gpt.json
    comparison_claude.json

    metrics_gpt.csv
    metrics_claude.csv

    summary_gpt.csv
    summary_claude.csv