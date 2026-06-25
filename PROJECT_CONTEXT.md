# PROJECT_CONTEXT.md

# Projeto

Extração automática de informações catalíticas de artigos científicos utilizando Large Language Models (LLMs) para construção de datasets destinados a aplicações de Machine Learning em Catálise.

---

# Objetivo

Automatizar a leitura de artigos científicos e extrair informações estruturadas de catalisadores, reduzindo o tempo gasto na construção manual de bases de dados.

---

# Arquitetura Atual

## Modelos

* GPT-4o-mini
* Claude Sonnet 4

Gemini foi testado, porém removido temporariamente devido às limitações de cota da API gratuita.

---

# Agentes Implementados

## Agent 1 — Catalyst

Objetivo:

Extrair os catalisadores explicitamente mencionados no abstract.

Saída:

```json
{
    "catalysts":[]
}
```

Modelos:

* GPT
* Claude

---

## Agent 2 — Metal + Support

Objetivo:

Extrair:

* metais ativos
* suporte catalítico

Saída:

```json
{
    "metal": [],
    "support": null
}
```

Modelos:

* GPT
* Claude

---

# Organização Atual do Dataset

```json
{
    "paper": "...",

    "gpt":{

        "catalyst":{

            "catalysts":[]
        },

        "metal_support":{

            "metal":[],
            "support":null
        }
    },

    "claude":{

        "catalyst":{

            "catalysts":[]
        },

        "metal_support":{

            "metal":[],
            "support":null
        }
    }
}
```

---

# Pipeline

PDF

↓

extractor.py

↓

Abstract

↓

Agent 1

↓

Agent 2

↓

dataset.json

↓

compare.py

↓

comparison_results.json

---

# Benchmark

Ground Truth:

* expected_catalysts
* expected_metal
* expected_support

Métrica:

RapidFuzz Token Set Ratio

Threshold:

80%

Artigos com:

```json
"skip_benchmark": true
```

são ignorados automaticamente.

---

# Correções Implementadas

## JSON Parser

O Claude ocasionalmente retornava múltiplos blocos JSON acompanhados de explicações.

Foi implementado um parser baseado em:

```python
json.JSONDecoder().raw_decode()
```

permitindo extrair apenas o primeiro objeto JSON válido.

---

# Próximos Agentes

Agent 3

Temperature

Agent 4

Pressure

Agent 5

Synthesis Method

Agent 6

Conversion

Agent 7

Selectivity

---

# Arquitetura Futura

Planejamento:

```
agents/

catalyst.py

metal_support.py

temperature.py

pressure.py

conversion.py

selectivity.py
```

O arquivo `llm_extractor.py` ficará responsável apenas pela comunicação com os modelos (GPT, Claude e futuramente Gemini).
