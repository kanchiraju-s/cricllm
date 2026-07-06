# Data directory

Place the ICC Laws of Cricket Markdown rulebook here, e.g.:

```
data/icc_rulebook.md
```

(`pdf_md.py` at the project root converts the official ICC Laws PDF into this
Markdown file, if you're starting from the PDF.)

The pipeline also accepts a directory of `.md` files:

```bash
python scripts/run_ingestion.py --input data/
```
