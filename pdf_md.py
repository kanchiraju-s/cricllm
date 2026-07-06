from pathlib import Path
from docling.document_converter import DocumentConverter

input_pdf = "ilovepdf_merged.pdf"
output_md = "icc_rulebook.md"

converter = DocumentConverter()

result = converter.convert(input_pdf)

markdown = result.document.export_to_markdown()

Path(output_md).write_text(markdown, encoding="utf-8")

print(f"Saved Markdown to {output_md}")