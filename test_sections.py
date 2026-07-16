from extractor import extract_text

from preprocessing.section_extractor import (
    extract_sections,
    preview_sections,
    section_statistics
)

pdf = "Papers/020CAT.pdf"

text = extract_text(pdf)

sections = extract_sections(text)

preview_sections(sections)

print()

print(section_statistics(sections))