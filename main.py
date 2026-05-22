import json
import os

from extractor import extract_text
from llm_extractor import extract_with_llm

pdf_path = "Papers/039MCN.pdf"

# =====================================
# Extrair texto do PDF
# =====================================

text = extract_text(pdf_path)

# =====================================
# Selecionar trecho
# =====================================

text = text[:4000]

# =====================================
# Extração com OpenAI
# =====================================

data = extract_with_llm(text)

# =====================================
# Adicionar nome do paper
# =====================================

data["paper_name"] = os.path.basename(pdf_path)

# =====================================
# Ler dataset existente
# =====================================

dataset_path = "dataset.json"

if os.path.exists(dataset_path):

    with open(dataset_path, "r", encoding="utf-8") as f:

        try:
            dataset = json.load(f)

        except:
            dataset = []

else:

    dataset = []

# =====================================
# Adicionar novo dado
# =====================================

dataset.append(data)

# =====================================
# Salvar dataset atualizado
# =====================================

with open(dataset_path, "w", encoding="utf-8") as f:

    json.dump(
        dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================
# Mostrar resultado
# =====================================

print("\nDataset atualizado:\n")

print(json.dumps(data, indent=4, ensure_ascii=False))