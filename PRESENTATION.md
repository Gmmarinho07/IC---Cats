# PRESENTATION_2026_06.md

# Title

Zeolite-Catalyzed Ethanol Dehydration to Ethylene — Progress Report

Automated extraction of catalytic data from scientific articles using LLMs.

Presenter:
Gabriel Maia Marinho

---

# Research Timeline

April 2026

* Understanding chemistry concepts.
* Understanding the bottleneck caused by human limitations.
* Understanding problems related to unstructured data.

May 2026

* Prompt engineering for catalyst names.
* Dataset formalization.

Current stage

* Testing catalyst extraction.

Planned

* Expansion to other parameters and code optimization.

Goal:

Create an automated pipeline capable of extracting catalytic information directly from scientific articles.

---

# Initial Architecture

Input:

PDF articles

Extraction:

Abstract only

Tools:

* Python
* OpenAI
* GitHub
* VSCode
* PyMuPDF (fitz)

---

# First Test

Goal:

Extract catalyst names from the abstract.

Output format:

{
"catalysts":[]
}

---

# First Agents

Agent 1

Catalyst extraction

Generated:

* dataset.json
* logs.json

Agent 2

Alternative extraction approach

Generated:

* dataset.json
* logs.json

---

# Current Pipeline

PDF

↓

Abstract

↓

LLMs

↓

JSON

↓

Dataset

↓

Benchmark

---

# Benchmark

Files:

* dataset.json
* ground_truth.json
* comparison_results.json

Metric:

RapidFuzz Token Set Ratio

Match criterion:

Similarity >= 80

Accuracy:

matches / total papers

Limitation:

Text similarity does not guarantee chemical equivalence.

---

# Agent 3

Objective:

Extract:

* metal
* support

Motivation:

Evaluate structure decomposition and benchmark quality.

---

# Problems Found

## Extraction and abbreviation

Examples:

* HAP
* MgAl-LDO

Need for normalization.

## Catalytic structure

Examples:

* Ru/MgAl-LDO
* Ru on Mg-Al mixed oxide

Exact comparison is insufficient.

## Ground Truth

Still requires manual validation.

---

# Normalization

normalize.py

Purpose:

Reduce problems caused by:

* abbreviations
* different representations
* catalyst structure

Limitation:

Needs updating for new papers.

Execution:

normalize.py

↓

compare.py

---

# Multiple Agents

Models tested:

* GPT-4o-mini
* Claude Sonnet 4
* Gemini 2.5 Flash

Gemini limitation:

RPM quota.

---

# Comparison

Metric:

RapidFuzz Token Set Ratio

Threshold:

80

Result:

GPT:

90%

Claude:

90%

Observation:

Same evaluation metric used for both models.

---

# Next Steps

Short term

* Fix metal/support benchmark.
* Validate 10 papers.
* Improve prompts.
* Add verbose mode.

Medium term

Extract:

* temperature
* pressure
* conversion
* synthesis method
* selectivity

Long term

Create a normalized catalytic dataset.

Apply automation and machine learning.

Final objective:

Catalytic performance prediction.
