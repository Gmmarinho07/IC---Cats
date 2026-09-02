import fitz


def extract_text(pdf_path):

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text("text")

    # Remove caracteres de controle que podem aparecer
    # durante a extração de PDFs.
    text = "".join(
        char for char in text
        if char in "\n\r\t" or ord(char) >= 32
    )

    return text