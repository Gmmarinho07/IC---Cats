# RESULTS.md

# Última Atualização

Data:

Junho/2026

---

# Agent 1 — Catalyst

Modelos:

* GPT-4o-mini
* Claude Sonnet 4

Resultado:

GPT:

90%

Claude:

90%

Observação:

Desconsiderando o artigo de revisão marcado com:

```json
"skip_benchmark": true
```

ambos apresentam 100% de acerto no conjunto de benchmark.

---

# Agent 2 — Metal + Support

Status:

Implementado.

Benchmark em fase de validação.

---

# Melhorias Implementadas

* Novo prompt para extração de metais ativos.
* Novo prompt para extração de suporte catalítico.
* Novo parser robusto para respostas do Claude.
* Estrutura do dataset reorganizada para múltiplos agentes.
* compare.py atualizado para múltiplas tarefas.

---

# Próximos Objetivos

* Validar benchmark do Agent 2.
* Melhorar a normalização química.
* Integrar Gemini utilizando a nova SDK (`google-genai`).
* Separar agentes em módulos independentes.
* Construir dataset final para Machine Learning.
