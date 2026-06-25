import json
import os

from extractor import extract_text

from llm_extractor import (
    gpt_catalyst,
    claude_catalyst,
    gpt_metal_support,
    claude_metal_support
)

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
    # GPT - AGENT 1
    # =============================================

    gpt_catalyst_result = gpt_catalyst(text)

    # =============================================
    # GPT - AGENT 2
    # =============================================

    gpt_metal_support_result = gpt_metal_support(text)

    # =============================================
    # CLAUDE - AGENT 1
    # =============================================

    claude_catalyst_result = claude_catalyst(text)

    # =============================================
    # CLAUDE - AGENT 2
    # =============================================

    claude_metal_support_result = claude_metal_support(text)

    # =============================================
    # PRINT
    # =============================================

    print("\n==============================")
    print("GPT - Catalyst")
    print("==============================")
    print(gpt_catalyst_result)

    print("\n==============================")
    print("GPT - Metal / Support")
    print("==============================")
    print(gpt_metal_support_result)

    print("\n==============================")
    print("Claude - Catalyst")
    print("==============================")
    print(claude_catalyst_result)

    print("\n==============================")
    print("Claude - Metal / Support")
    print("==============================")
    print(claude_metal_support_result)

    # =============================================
    # DATASET
    # =============================================

    dataset.append({

        "paper": os.path.basename(pdf_path),

        "gpt": {

            "catalyst": gpt_catalyst_result,

            "metal_support": gpt_metal_support_result

        },

        "claude": {

            "catalyst": claude_catalyst_result,

            "metal_support": claude_metal_support_result

        }

    })

    # =============================================
    # LOGS
    # =============================================

    logs.append({

        "paper": os.path.basename(pdf_path),

        "input_length": len(text),

        "gpt_catalyst": gpt_catalyst_result,

        "gpt_metal_support": gpt_metal_support_result,

        "claude_catalyst": claude_catalyst_result,

        "claude_metal_support": claude_metal_support_result

    })

# =====================================================
# SALVAR DATASET
# =====================================================

with open(
    "dataset.json",
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
    "logs.json",
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
print("Teste concluído.")
print("dataset.json atualizado.")
print("logs.json atualizado.")
print("===================================")