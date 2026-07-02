import json
import os

from extractor import extract_text

from agents.catalyst import extract as catalyst
from agents.metal_support import extract as metal_support


# =====================================================
# PAPERS
# =====================================================

papers = [
    "Papers/039MCN.pdf",
    "Papers/032MCN.pdf",
    "Papers/034MCN.pdf",
    "Papers/018MCN.pdf",
    "Papers/020CAT.pdf",
    "Papers/028CAT.pdf",
    "Papers/030CAT.pdf",
    "Papers/031RVW.pdf",
    "Papers/036CAT.pdf",
    "Papers/038CAT.pdf"
]

# =====================================================
# DATASET E LOGS
# =====================================================

dataset = []
logs = []

# =====================================================
# LOOP DOS PAPERS
# =====================================================

for pdf_path in papers:

    print(f"\nProcessando: {pdf_path}")

    text = extract_text(pdf_path)

    # =============================================
    # PEGAR APENAS O ABSTRACT
    # =============================================

    start = text.find("Abstract")

    if start != -1:
        text = text[start:start + 1500]
    else:
        text = text[:1500]

    # =============================================
    # AGENT 1 - CATALYST
    # =============================================

    gpt_catalyst = catalyst(
        text,
        "gpt"
    )

    claude_catalyst = catalyst(
        text,
        "claude"
    )

    # =============================================
    # AGENT 2 - METAL / SUPPORT
    # =============================================

    gpt_metal_support = metal_support(
        text,
        "gpt"
    )

    claude_metal_support = metal_support(
        text,
        "claude"
    )

    # =============================================
    # PRINT
    # =============================================

    print("\n==============================")
    print("GPT - Catalyst")
    print("==============================")
    print(gpt_catalyst)

    print("\n==============================")
    print("Claude - Catalyst")
    print("==============================")
    print(claude_catalyst)

    print("\n==============================")
    print("GPT - Metal / Support")
    print("==============================")
    print(gpt_metal_support)

    print("\n==============================")
    print("Claude - Metal / Support")
    print("==============================")
    print(claude_metal_support)

    # =============================================
    # DATASET
    # =============================================

    dataset.append({

        "paper": os.path.basename(pdf_path),

        "gpt": {

            "catalyst": gpt_catalyst,

            "metal_support": gpt_metal_support

        },

        "claude": {

            "catalyst": claude_catalyst,

            "metal_support": claude_metal_support

        }

    })

    # =============================================
    # LOGS
    # =============================================

    logs.append({

        "paper": os.path.basename(pdf_path),

        "input_length": len(text),

        "gpt": {

            "catalyst": gpt_catalyst,

            "metal_support": gpt_metal_support

        },

        "claude": {

            "catalyst": claude_catalyst,

            "metal_support": claude_metal_support

        }

    })

# =====================================================
# SALVAR DATASET
# =====================================================

os.makedirs("benchmark", exist_ok=True)

with open(
    "benchmark/dataset.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================================
# SALVAR LOGS
# =====================================================

with open(
    "benchmark/logs.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        logs,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================================
# FINAL
# =====================================================

print("\n===================================")
print("Pipeline executado com sucesso.")
print("benchmark/dataset.json atualizado.")
print("benchmark/logs.json atualizado.")
print("===================================")