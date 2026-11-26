# Finity Data Extractor

Streamlit app that turns financial documents (PDF, scanned PDF, Excel, CSV) into structured data and a RAG-ready JSON output.

## How it works

1. Upload a PDF, XLSX/XLS or CSV.
2. Text PDFs go through `pdfplumber`/`PyMuPDF`/`camelot`; scanned pages go through OCR (Tesseract + OpenCV).
3. Tables are cleaned, typed and tagged; notes and sections are split for retrieval.
4. Results are shown in the UI and exportable as JSON.

The chatbot layer uses LangChain, FAISS and Google Gemini embeddings/LLM.

## Run

System packages (also listed in `packages.txt`): `tesseract-ocr`, `libgl1`, `ghostscript`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: Streamlit UI
- `enhanced_financial_data_extractor.py`: extraction pipeline (`extract_financial_data_rag`)
