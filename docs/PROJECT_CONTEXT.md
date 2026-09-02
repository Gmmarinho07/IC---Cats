# PROJECT_CONTEXT.md

# Projeto

Extração automática de informações catalíticas de artigos científicos
utilizando Large Language Models (LLMs) para construção de datasets
destinados a aplicações de Machine Learning em Catálise.

---

# Objetivo

Automatizar a leitura de artigos científicos e extrair informações
estruturadas de catalisadores, reduzindo o tempo gasto na construção
manual de bases de dados.

O projeto busca desenvolver um pipeline capaz de selecionar as partes
mais relevantes dos artigos e utilizar LLMs para transformar informações
não estruturadas em dados estruturados.

---

# Arquitetura Atual

## Modelos

* GPT-4o-mini
* Claude Sonnet 4.6

Gemini foi testado anteriormente, porém está temporariamente fora do
benchmark devido às limitações de cota da API gratuita.

---

# Organização do Projeto

A arquitetura atual é dividida em módulos independentes:

```text
agents/
preprocessing/
evaluation/
benchmark/
prompts/
llms/

extractor.py
main.py
compare.py