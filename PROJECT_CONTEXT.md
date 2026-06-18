# PROJECT_CONTEXT.md

# Projeto

Extração automática de informações de catalisadores a partir de artigos científicos para construção de datasets destinados a Machine Learning em catálise.

# Objetivo

Construir um pipeline baseado em LLMs para extrair informações estruturadas de artigos científicos e gerar um dataset para treinamento e análise de modelos de ML.

# Estrutura Atual

```
Projeto/
│
├── Papers/
│     ├── 039MCN.pdf
│     ├── 032MCN.pdf
│     ├── ...
│
├── extractor.py
├── llm_extractor.py
├── main.py
├── compare.py
├── normalization.py
├── dataset.json
├── ground_truth.json
├── comparison_results.json
├── logs.json
├── .env
├── .gitignore
└── PROJECT_CONTEXT.md
```

# APIs

OpenAI:

* gpt-4o-mini

Anthropic:

* claude-sonnet-4-6

Google:

* gemini-2.5-flash
* temporariamente desativado devido às cotas gratuitas

# Organização Atual

## extractor.py

Responsável por extrair texto dos PDFs utilizando PyMuPDF (fitz).

Função principal:

```python
extract_text(pdf_path)
```

## llm_extractor.py

Possui:

* build_prompt()
* clean_json()
* gpt_catalyst()
* claude_catalyst()

Gemini removido temporariamente.

Prompt atual:

"Extract catalyst names explicitly mentioned in the abstract."

Restrições:

* Use only information present in the text.
* Do not infer.
* Do not guess.
* Return ONLY valid JSON.

Formato:

{
"catalysts": []
}

## main.py

Fluxo:

1. Ler PDFs.
2. Extrair texto.
3. Selecionar Abstract.
4. Enviar para GPT.
5. Enviar para Claude.
6. Salvar dataset.json.
7. Salvar logs.json.

Dataset possui estrutura:

{
"paper": "...",
"gpt": {
"catalysts": [...]
},
"claude": {
"catalysts": [...]
}
}

## ground_truth.json

Estrutura:

{
"paper": "...",
"expected_catalysts": [...]
}

Ground truth construída manualmente.

## compare.py

Compara:

* GPT x Ground Truth
* Claude x Ground Truth

Utiliza:

RapidFuzz
token_set_ratio

Threshold:

80%

Gera:

comparison_results.json

## normalization.py

Responsável por:

normalize()

normalize_list()

# Decisões importantes

* Utilizar somente informações explícitas do abstract.
* Não inferir catalisadores.
* JSON obrigatório.
* Temperature = 0.
* Comparação por similaridade usando RapidFuzz.
* Threshold de 80%.

# Próximos Passos

## Agent 1

Extrair catalisadores.

Saída:

{
"catalysts":[]
}

## Agent 2

Extrair metais ativos.

Saída:

{
"metal":[]
}

## Agent 3

Extrair suportes.

Saída:

{
"support":""
}

## Agent 4

Extrair método de síntese.

## Agent 5

Extrair temperatura de reação.

## Agent 6

Extrair pressão.

## Agent 7

Extrair rendimento ou conversão.

## Agent 8

Extrair seletividade.

# Objetivo Final

Construir um dataset estruturado contendo:

* catalisador
* metal
* suporte
* método de síntese
* temperatura
* pressão
* conversão
* seletividade

para utilização em Machine Learning aplicado à catálise de etanol.
