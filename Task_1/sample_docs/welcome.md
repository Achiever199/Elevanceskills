# Welcome

This is a sample document used to verify that the file loader, chunker,
and vector store are working end to end.

Replace or add files in this `sample_docs/` folder (or point the `docs_folder`
source in `config.yaml` at a different directory) with the material you
actually want the chatbot to know about. Supported formats: .txt, .md, .pdf.

Every time the scheduler runs (or you run `python main.py update-now`),
new or changed files here will automatically be re-chunked, re-embedded,
and upserted into the vector database, while unchanged files are skipped
for efficiency.
