import json
import os

from extractor import extract_text

from llm_extractor import (
    agent_1_catalyst,
    agent_2_catalyst,
    agent_3_metal_support
)

# =====================================================
# PAPERS
# =====================================================

papers = [
    "Papers/039MCN.pdf",
    "Papers/032MCN.pdf",
    "Papers/034MCN.pdf",
    "Papers/018MCN.pdf",
    "Papers/020CAT.pdf"
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
    # PEGAR SOMENTE O ABSTRACT
    # =============================================

    start = text.find("Abstract")

    if start != -1:
        text = text[start:start + 1500]
    else:
        text = text[:1500]

    # =============================================
    # AGENTE 1
    # =============================================

    result_1 = agent_1_catalyst(text)

    # =============================================
    # AGENTE 2
    # =============================================

    result_2 = agent_2_catalyst(text)

    # =============================================
    # AGENTE 3
    # =============================================

    result_3 = agent_3_metal_support(text)

    # =============================================
    # PRINT
    # =============================================

    print("\nAgent 1:")
    print(result_1)

    print("\nAgent 2:")
    print(result_2)

    print("\nAgent 3:")
    print(result_3)

    # =============================================
    # DATASET
    # =============================================

    dataset.append({

        "paper": os.path.basename(pdf_path),

        "agent_1": result_1,

        "agent_2": result_2,

        "agent_3": result_3
    })

    # =============================================
    # LOGS
    # =============================================

    logs.append({

        "paper": os.path.basename(pdf_path),

        "input_length": len(text),

        "agent_1_output": result_1,

        "agent_2_output": result_2,

        "agent_3_output": result_3
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