from extractor import extract_text

from preprocessing.section_extractor import extract_sections

from preprocessing.chunk_builder import build_chunks

from preprocessing.chunk_ranker import (
    rank_chunks,
    top_chunks_by_section
)

from preprocessing.context_builder import (
    build_context,
    preview_context,
    context_statistics
)

text = extract_text("Papers/020CAT.pdf")

sections = extract_sections(text)

sections.pop("references", None)

chunks = build_chunks(sections)

ranked = rank_chunks(chunks)

selected = top_chunks_by_section(
    ranked,
    top_k=2
)

context = build_context(selected)

preview_context(context)

print(context_statistics(context))