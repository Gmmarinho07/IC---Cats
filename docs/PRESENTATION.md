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
* Improving text selection from scientific articles.
* Benchmarking GPT and Claude.
* Improving the representation of catalyst names and article-specific identifiers.

Planned

* Expansion to other catalytic parameters.
* Prompt optimization.
* Retrieval and ranking improvements.
* Code optimization.

Goal:

Create an automated pipeline capable of extracting catalytic information directly from scientific articles.

---

# Initial Architecture

Input:

PDF articles

Extraction:

Full article text

Processing:

* Section extraction
* Text chunking
* Chunk ranking
* Context selection

Tools:

* Python
* OpenAI
* Anthropic
* GitHub
* VSCode
* PyMuPDF (fitz)

---

# First Test

Goal:

Extract catalyst names from the abstract.

Initial output format:

{
    "catalysts":[]
}

Result:

The first tests showed that catalyst extraction is possible, but
different representations and abbreviations make direct comparison
difficult.

---

# Initial Agents

Agent 1

Catalyst extraction

Agent 2

Alternative extraction approach

Initial outputs:

* dataset.json
* logs.json

These experiments were used to understand the extraction problem
and evaluate different approaches.

---

# Current Pipeline

PDF

↓

Text Extraction

↓

Section Extraction

↓

Text Chunks

↓

Chunk Ranking

↓

Context Selection

↓

LLMs

↓

JSON

↓

Benchmark

↓

Metrics

---

# Chunk Selection

Objective:

Select the most relevant parts of the article before sending them
to the LLM.

Sections considered:

* Abstract
* Experimental
* Results
* Introduction
* Conclusion

Ranking considers:

* Catalyst-related keywords
* Experimental terminology
* Chemical formulas
* Catalyst structures
* Preparation methods
* Characterization terms
* Section importance

Current selection policy:

Abstract → 1 chunk

Experimental → 2 chunks

Results → 4 chunks

Purpose:

Increase the amount of relevant information available to the LLM
without sending the entire article.

---

# Current Extraction

The extraction focuses on catalyst information.

For each catalyst, the system can extract:

* catalyst
* metal
* support
* raw_name
* article_identifiers

Example:

{
    "catalyst": "H-ZSM-5",
    "metal": null,
    "support": null,
    "raw_name": "H-ZSM-5",
    "article_identifiers": ["MFI"]
}

---

# Raw Name and Article Identifiers

Raw Name

Preserves the catalyst name exactly as it appears in the article.

Article Identifiers

Stores separate labels, codes, abbreviations or alternative
designations explicitly assigned by the authors.

Examples:

* Alumina → A
* Silica-alumina → SA
* H-ferrierite → FER
* H-ZSM-5 → MFI
* H-faujasite → USY

Important:

Article identifiers are not used by the current benchmark.

They are preserved for future dataset construction and machine
learning applications.

---

# Ground Truth

Ground truth was standardized to a common structure.

Example:

{
    "paper": "...",
    "title": "...",
    "skip_benchmark": false,
    "catalysts": [
        {
            "catalyst": "...",
            "metal": null,
            "support": null
        }
    ]
}

Reviews can be excluded from the benchmark using:

"skip_benchmark": true

Purpose:

Ensure that all papers are evaluated using the same structure.

---

# Benchmark

The benchmark compares model extraction against the ground truth.

Current models:

* GPT-4o-mini
* Claude Sonnet 4.6

Comparison:

* Catalyst names are normalized.
* Similarity is calculated using RapidFuzz.
* Multiple similarity functions are considered.
* One-to-one matching is used.

Match criterion:

Similarity >= 80

Metrics:

* TP — True Positives
* FP — False Positives
* FN — False Negatives
* Precision
* Recall
* F1

---

# Current Results

Current benchmark shows that Claude has higher overall performance
than GPT under the current configuration.

GPT:

Precision ≈ 61.94%

Recall ≈ 64.86%

F1 ≈ 63.37%

Claude:

Precision ≈ 66.07%

Recall ≈ 75.00%

F1 ≈ 70.25%

Observation:

Claude currently shows better recall and F1, while both models still
produce false positives and false negatives.

---

# Problems Found

## Extraction and abbreviation

Examples:

* HAP
* MgAl-LDO
* MgO
* H-ZSM-5

Different representations can refer to the same catalyst.

---

## Catalyst structure

Examples:

* Ru/MgAl-LDO
* Ru on Mg-Al mixed oxide

Text similarity alone may not recognize that different expressions
represent chemically equivalent catalysts.

---

## Ground Truth

Ground truth still requires manual validation.

Incomplete ground truth can incorrectly classify valid model
extractions as false positives.

---

## Context Selection

Increasing the amount of retrieved context does not always improve
performance.

Example:

Some papers improved when more Results chunks were selected,
while others became worse.

Current baseline:

Abstract → 1

Experimental → 2

Results → 4

---

# Normalization

normalize.py

Purpose:

Reduce problems caused by:

* abbreviations
* different representations
* catalyst structure

Current limitation:

Text normalization does not guarantee chemical equivalence.

Example:

MgO

and

magnesium oxide

may represent the same catalyst but require chemical-aware
normalization.

---

# Project Architecture

Objective:

Improve the organization of the project and allow the pipeline
to scale.

Main modules:

* agents/
* preprocessing/
* evaluation/
* benchmark/
* prompts/
* extractor.py
* main.py

Responsibilities are separated between:

* PDF extraction
* text preprocessing
* context selection
* LLM extraction
* evaluation
* benchmark execution

Benefits:

* More organized code.
* Easier inclusion of new models.
* Easier inclusion of new extraction agents.
* Independent evaluation components.
* More scalable benchmark.
* Easier experimentation with retrieval strategies.

---

# Multiple Models

Models currently evaluated:

* GPT-4o-mini
* Claude Sonnet 4.6

Gemini:

Temporarily removed from the current benchmark because of API
rate-limit restrictions.

Future:

Reintegrate Gemini when the API limitations allow reliable testing.

---

# Current Research Focus

The main challenge is no longer simply extracting catalyst names.

The current focus is understanding:

* Which parts of an article contain the relevant information.
* How much context should be provided to the LLM.
* Why false positives occur.
* Why false negatives occur.
* How catalyst representations should be normalized.
* How article-specific identifiers can be preserved.

Goal:

Improve extraction quality without unnecessarily increasing the
amount of context sent to the models.

---

# Next Steps

Short term

* Continue prompt improvements.
* Investigate false positives and false negatives.
* Validate the ground truth.
* Test article-specific identifiers.
* Evaluate the current chunk-selection strategy.

Medium term

Extract additional parameters:

* temperature
* pressure
* conversion
* synthesis method
* selectivity
* reaction conditions

Improve:

* chemical normalization
* context selection
* benchmark quality

Long term

Create a normalized catalytic dataset.

Apply automation and machine learning.

Final objective:

Catalytic performance prediction.