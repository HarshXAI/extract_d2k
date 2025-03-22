import os
import json
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime
import io
from typing import Dict, List, Any, Union, Tuple, Optional
import warnings
import subprocess
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# Dictionary for standardizing column names
COLUMN_NAME_MAPPING = {
    # Revenue related
    "rev.": "Revenue",
    "revenue": "Revenue",
    "sales": "Revenue",
    "total revenue": "Revenue",
    "gross revenue": "Revenue",
    "turnover": "Revenue",
    
    # Profit related
    "net income": "Net Profit",
    "net profit": "Net Profit",
    "profit": "Net Profit",
    "earnings": "Net Profit",
    "net earnings": "Net Profit",
    "net profit after tax": "Net Profit",
    "profit after tax": "Net Profit",
    
    # EBITDA related
    "ebitda": "EBITDA",
    "earnings before interest, tax, depreciation and amortization": "EBITDA",
    
    # Gross profit related
    "gross profit": "Gross Profit",
    "gross margin": "Gross Profit",
    
    # Assets related
    "total assets": "Assets",
    "assets": "Assets",
    "current assets": "Current Assets",
    "non-current assets": "Non-Current Assets",
    "fixed assets": "Fixed Assets",
    
    # Liabilities related
    "total liabilities": "Liabilities",
    "liabilities": "Liabilities",
    "current liabilities": "Current Liabilities",
    "non-current liabilities": "Non-Current Liabilities",
    "long term liabilities": "Long-Term Liabilities",
    
    # Equity related
    "total equity": "Equity",
    "shareholder's equity": "Equity",
    "stockholder's equity": "Equity",
    "equity": "Equity",
    "share capital": "Share Capital",
    
    # Expenses related
    "total expenses": "Expenses",
    "expenses": "Expenses",
    "operating expenses": "Operating Expenses",
    "sgna": "SG&A",
    "selling, general and administrative": "SG&A",
    
    # Cash Flow related
    "cash flow": "Cash Flow",
    "operating cash flow": "Operating Cash Flow",
    "free cash flow": "Free Cash Flow",
    "cash from operations": "Operating Cash Flow",
}

# Financial terms to tag sections
FINANCIAL_TERMS = {
    "profitability": ["margin", "profit", "earnings", "ebitda", "ebit", "income", "return", "roi", "roe", "roa"],
    "revenue": ["revenue", "sales", "turnover", "income", "proceeds"],
    "expenses": ["expense", "cost", "expenditure", "overhead", "outlay", "spending"],
    "assets": ["asset", "property", "equipment", "inventory", "receivable", "investment"],
    "liabilities": ["liability", "debt", "loan", "payable", "obligation", "borrowing"],
    "equity": ["equity", "capital", "stock", "share", "retained earnings", "reserves"],
    "cash_flow": ["cash flow", "liquidity", "solvency", "financing", "investing"],
    "taxes": ["tax", "taxation", "duty", "levy", "tariff"],
    "growth": ["growth", "increase", "expansion", "rise", "gain", "improvement"],
    "decline": ["decline", "decrease", "reduction", "fall", "drop", "loss", "deterioration"]
}

def check_ghostscript_installation() -> bool:
    """Check if Ghostscript is properly installed and accessible"""
    try:
        result = subprocess.run(['gs', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Ghostscript version {result.stdout.strip()} found.")
            return True
    except FileNotFoundError:
        pass
    
    print("WARNING: Ghostscript not found in PATH.")
    print("Camelot requires Ghostscript to extract tables from PDFs.")
    print("Installation instructions:")
    print("  - macOS: brew install ghostscript")
    print("  - Ubuntu/Debian: apt-get install ghostscript")
    print("  - Windows: Download from https://ghostscript.com/releases/")
    print("After installing, ensure the 'gs' command is in your PATH.")
    
    return False

# Import camelot only after checking Ghostscript
has_ghostscript = check_ghostscript_installation()
camelot = None
try:
    import camelot
except ImportError as e:
    print(f"WARNING: Could not import camelot: {e}")
except Exception as e:
    print(f"WARNING: Error initializing camelot: {e}")

def extract_financial_data_rag(file_path: str) -> Dict[str, Any]:
    """
    Extract financial data from a document (PDF or spreadsheet) optimized for RAG systems
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict: Structured JSON with extracted financial data, contextual text, and metadata
    """
    # Basic validation
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Initialize result structure
    result = {
        "metadata": {
            "source_file": os.path.basename(file_path),
            "extraction_timestamp": datetime.now().isoformat(),
        },
        "financial_data": [],
        "contextual_text": [],
        "notes": []
    }
    
    try:
        # Process based on file type
        if ext in ['.csv', '.xlsx', '.xls']:
            result["metadata"]["document_type"] = "Spreadsheet"
            result["metadata"]["extraction_method"] = "pandas"
            result = process_spreadsheet_rag(file_path, result)
        elif ext == '.pdf':
            result["metadata"]["document_type"] = "PDF"
            result = process_pdf_rag(file_path, result)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        return result
    
    except Exception as e:
        # In case of error, return what we have with error info
        result["metadata"]["error"] = str(e)
        return result

def process_spreadsheet_rag(file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Process spreadsheet files for RAG system"""
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            # For CSV, add a single page indicator
            page_num = 1
            
            # Standardize column names
            df = standardize_column_names(df)
            
            # Determine table name based on content
            table_name = determine_table_type(df)
            
            table_data = {
                "type": "table",
                "table_name": table_name,
                "columns": df.columns.tolist(),
                "rows": df.to_dict('records'),
                "page": page_num
            }
            
            result["financial_data"].append(table_data)
            
            # Add contextual information from the table
            context = extract_context_from_table(df, table_name)
            if context:
                result["contextual_text"].append({
                    "type": "text",
                    "content": context,
                    "page": page_num,
                    "tags": derive_tags_from_content(context)
                })
            
        else:
            # Read all sheets for Excel files
            sheets = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_idx, (sheet_name, df) in enumerate(sheets.items()):
                # Standardize column names
                df = standardize_column_names(df)
                
                # Determine table name based on content
                derived_name = determine_table_type(df)
                table_name = f"{derived_name} ({sheet_name})" if derived_name != "Financial Data" else sheet_name
                
                # Convert DataFrame to structured format
                table_data = {
                    "type": "table",
                    "table_name": table_name,
                    "columns": df.columns.tolist(),
                    "rows": df.to_dict('records'),
                    "page": sheet_idx + 1  # Use sheet index as "page" number
                }
                
                result["financial_data"].append(table_data)
                
                # Add contextual information from the table
                context = extract_context_from_table(df, table_name)
                if context:
                    result["contextual_text"].append({
                        "type": "text",
                        "content": context,
                        "page": sheet_idx + 1,
                        "tags": derive_tags_from_content(context)
                    })
            
            # Add metadata about number of sheets
            result["metadata"]["sheets_processed"] = len(sheets)
    
    except Exception as e:
        result["metadata"]["error"] = f"Error processing spreadsheet: {str(e)}"
    
    return result

def process_pdf_rag(file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Process PDF files optimized for RAG system"""
    try:
        # Open the PDF to get total number of pages
        with pdfplumber.open(file_path) as pdf:
            result["metadata"]["pages_processed"] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                # Check if this page is text-based or scanned
                text = page.extract_text() or ""
                is_text_based = len(text) > 50  # Simple heuristic
                
                if is_text_based:
                    # Process text-based page
                    process_text_based_page(page, page_num + 1, result)
                else:
                    # Process scanned page with OCR
                    process_scanned_page(file_path, page_num, result)
                    
                # Look for footnotes or notes
                notes = extract_notes(text)
                if notes:
                    result["notes"].append({
                        "type": "footnote",
                        "content": notes,
                        "page": page_num + 1
                    })
        
        # Extract tables with Camelot if available for the entire document
        if camelot is not None:
            try:
                # Extract tables using Camelot from the entire document
                tables = camelot.read_pdf(file_path, flavor='lattice', pages='all')
                
                for i, table in enumerate(tables):
                    df = table.df
                    page = table.parsing_report['page']
                    
                    # Clean and standardize table
                    df = clean_table(df)
                    df = standardize_column_names(df)
                    
                    # Determine the type of financial table
                    table_name = determine_table_type(df)
                    
                    table_data = {
                        "type": "table",
                        "table_name": f"{table_name} {i+1}" if i > 0 else table_name,
                        "columns": df.columns.tolist(),
                        "rows": df.to_dict('records'),
                        "page": int(page)
                    }
                    
                    result["financial_data"].append(table_data)
                    
                    # Extract contextual information from the table
                    context = extract_context_from_table(df, table_name)
                    if context:
                        result["contextual_text"].append({
                            "type": "text",
                            "content": context,
                            "page": int(page),
                            "tags": derive_tags_from_content(context)
                        })
                        
            except Exception as e:
                result["metadata"]["warning"] = f"Error using Camelot: {str(e)}. Some tables may be missed."
    
    except Exception as e:
        result["metadata"]["error"] = f"Error processing PDF: {str(e)}"
    
    return result

def process_text_based_page(page, page_num: int, result: Dict[str, Any]) -> None:
    """Process a text-based PDF page"""
    # Extract text
    text = page.extract_text() or ""
    
    # Extract tables using pdfplumber
    tables = page.extract_tables()
    
    # Process tables
    for i, table_data in enumerate(tables):
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        
        # Try to use the first row as header if it seems appropriate
        if len(df) > 1 and not df.empty:
            # Check if first row looks like a header
            if all(isinstance(x, str) for x in df.iloc[0].values):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
        
        # Clean and standardize
        df = clean_table(df)
        df = standardize_column_names(df)
        
        # Determine table type
        table_name = determine_table_type(df)
        
        # Create table data structure
        if not df.empty:
            table_dict = {
                "type": "table",
                "table_name": f"{table_name} {i+1}" if i > 0 else table_name,
                "columns": df.columns.tolist(),
                "rows": df.to_dict('records'),
                "page": page_num
            }
            
            result["financial_data"].append(table_dict)
    
    # Process remaining text as contextual text
    if text:
        # Remove text that is already in tables to avoid duplication
        for table in tables:
            for row in table:
                for cell in row:
                    if cell:
                        text = text.replace(cell, "")
                        
        # Clean and split the text into logical sections
        sections = split_text_to_sections(text)
        
        for i, section in enumerate(sections):
            if section.strip():
                # Clean the section text
                cleaned_section = clean_text(section)
                
                # Derive tags for the section
                tags = derive_tags_from_content(cleaned_section)
                
                # Add to contextual text
                result["contextual_text"].append({
                    "type": "text",
                    "content": cleaned_section,
                    "page": page_num,
                    "section": i + 1,
                    "tags": tags
                })

def process_scanned_page(file_path: str, page_num: int, result: Dict[str, Any]) -> None:
    """Process a scanned PDF page using OCR"""
    # Open the PDF
    pdf_document = fitz.open(file_path)
    
    # Get the page
    page = pdf_document[page_num]
    
    # Convert page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    # Convert to grayscale if needed
    if img.shape[2] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # Apply adaptive thresholding for better OCR
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Run OCR
    text = pytesseract.image_to_string(thresh)
    
    # Extract tables from OCR output
    tables = extract_tables_from_ocr(text)
    
    for i, table_df in enumerate(tables):
        # Standardize column names
        table_df = standardize_column_names(table_df)
        
        # Determine the type of financial table
        table_name = determine_table_type(table_df)
        
        table_data = {
            "type": "table",
            "table_name": f"{table_name} {i+1}" if i > 0 else table_name,
            "columns": table_df.columns.tolist(),
            "rows": table_df.to_dict('records'),
            "page": page_num + 1
        }
        
        result["financial_data"].append(table_data)
        
        # Extract contextual information from the table
        context = extract_context_from_table(table_df, table_name)
        if context:
            result["contextual_text"].append({
                "type": "text",
                "content": context,
                "page": page_num + 1,
                "tags": derive_tags_from_content(context)
            })
    
    # Process remaining text as contextual text
    # Remove text that is in tables
    for table_df in tables:
        for col in table_df.columns:
            for value in table_df[col].astype(str).values:
                text = text.replace(value, "")
    
    # Clean and split the text into logical sections
    sections = split_text_to_sections(text)
    
    for i, section in enumerate(sections):
        if section.strip():
            # Clean the section text
            cleaned_section = clean_text(section)
            
            # Derive tags for the section
            tags = derive_tags_from_content(cleaned_section)
            
            # Add to contextual text
            result["contextual_text"].append({
                "type": "text",
                "content": cleaned_section,
                "page": page_num + 1,
                "section": i + 1,
                "tags": tags
            })
    
    # Look for notes
    notes = extract_notes(text)
    if notes:
        result["notes"].append({
            "type": "footnote",
            "content": notes,
            "page": page_num + 1
        })
    
    pdf_document.close()

def extract_tables_from_ocr(text: str) -> List[pd.DataFrame]:
    """Extract tables from OCR text using regex patterns"""
    tables = []
    
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Find potential table regions
    table_lines = []
    current_table = []
    
    # Pattern to detect table rows (having multiple number sequences or currency values)
    number_pattern = r'(\$?[\d,]+\.?\d*)'
    
    for line in lines:
        # If the line has at least 2 numbers or currency values, consider it part of a table
        numbers = re.findall(number_pattern, line)
        if len(numbers) >= 2:
            current_table.append(line)
        elif current_table and line.strip() == "":
            # Empty line after a table
            if len(current_table) > 2:  # At least 3 rows for a valid table
                table_lines.append(current_table)
            current_table = []
    
    # Handle the last table if exists
    if len(current_table) > 2:
        table_lines.append(current_table)
    
    # Convert detected table regions to DataFrames
    for table in table_lines:
        # For each detected table, try to determine columns
        
        # Split on multiple spaces or tabs
        rows = []
        for line in table:
            # Replace multiple spaces with a single space
            line = re.sub(r'\s+', ' ', line).strip()
            # Split by spaces
            cells = line.split(' ')
            # Remove empty cells
            cells = [cell for cell in cells if cell.strip()]
            rows.append(cells)
        
        # Create a DataFrame
        if rows:
            # Try to detect header row
            if len(rows[0]) == len(rows[1]):  # Same number of columns
                header = rows[0]
                data = rows[1:]
            else:
                header = None
                data = rows
            
            # Create DataFrame
            try:
                df = pd.DataFrame(data, columns=header)
                # Convert numeric columns
                df = convert_numeric_columns(df)
                tables.append(df)
            except:
                # If DataFrame creation fails, skip this table
                continue
    
    return tables

def extract_notes(text: str) -> str:
    """Extract notes and footnotes from text"""
    notes = []
    
    # Look for sections that might contain notes
    # Common patterns for notes and footnotes
    note_patterns = [
        r'Note\s+\d+[:.](.*?)(?=Note\s+\d+[:.:]|$)',  # Note X: content
        r'Notes to[^:]*:(.*?)(?=\n\n|\Z)',  # Notes to financial statements: content
        r'Footnote[s]?[^:]*:(.*?)(?=\n\n|\Z)',  # Footnotes: content
        r'\*\s*(.*?)(?=\n\*|\Z)'  # * content (footnote indicator)
    ]
    
    for pattern in note_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            note_text = match.group(1).strip()
            if note_text:
                notes.append(note_text)
    
    # Join all found notes
    return "\n".join(notes)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using the mapping dictionary"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    new_columns = []
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in COLUMN_NAME_MAPPING:
            new_columns.append(COLUMN_NAME_MAPPING[col_lower])
        else:
            new_columns.append(col)
    
    df.columns = new_columns
    return df

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean table data by removing empty rows and handling merged cells"""
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Handle merged cells (cells with empty values in the middle of the table)
    for col in df.columns:
        # Forward fill column header cells
        if col == 0 or isinstance(col, int):
            continue
            
        # If a column has NaN values and some previous value, fill forward
        if df[col].isna().any() and not df[col].isna().all():
            df[col] = df[col].fillna(method='ffill')
    
    # Validate numeric data
    validate_numeric_data(df)
    
    return df

def determine_table_type(df: pd.DataFrame) -> str:
    """Determine the type of financial table based on column names"""
    columns = [str(col).lower() for col in df.columns]
    
    # Check for common financial table types
    if any(term in ' '.join(columns) for term in ["balance", "assets", "liabilities", "equity"]):
        return "Balance Sheet"
    elif any(term in ' '.join(columns) for term in ["income", "profit", "loss", "revenue", "earnings"]):
        return "Income Statement"
    elif any(term in ' '.join(columns) for term in ["cash flow", "operating activities"]):
        return "Cash Flow Statement"
    elif any(term in ' '.join(columns) for term in ["ratio", "metrics", "performance"]):
        return "Financial Ratios"
    else:
        return "Financial Data"

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns with numbers to numeric types"""
    for col in df.columns:
        # Skip columns that are already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Try to convert to numeric, preserving non-numeric rows
        try:
            # Remove currency symbols and commas
            temp_col = df[col].astype(str).str.replace('$', '').str.replace(',', '')
            df[col] = pd.to_numeric(temp_col, errors='coerce')
        except:
            # Keep as is if conversion fails
            pass
    
    return df

def validate_numeric_data(df: pd.DataFrame) -> None:
    """Validate numerical data in the table and fix inconsistencies"""
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # If this is a financial statement, check if totals equal sum of items
    # Look for "Total" rows
    for i, row in df.iterrows():
        row_vals = row.astype(str).str.lower()
        if any("total" in str(val) for val in row_vals):
            # This might be a total row, check if totals match
            for col in numeric_cols:
                # Find non-total rows for this column
                other_vals = df[col].drop(i).dropna()
                
                # If we have a total and multiple individual values
                if not pd.isna(row[col]) and len(other_vals) >= 2:
                    total = float(row[col])
                    sum_of_others = float(other_vals.sum())
                    
                    # If total doesn't match sum (allowing for small rounding errors)
                    if abs(total - sum_of_others) > 0.01 * abs(total):
                        # Add a warning but keep the data as is
                        # We could also adjust the total or add a note
                        df.loc[i, col] = total  # Keep original, but could set to sum_of_others

def clean_text(text: str) -> str:
    """Clean text by removing noise and irrelevant information"""
    # Remove multiple spaces, tabs, and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common noise patterns often found in OCR output
    text = re.sub(r'[^\w\s.,;:$%()[\]{}\-\'\"]+', '', text)
    
    # Remove very short lines that might be noise
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 5]
    
    return '\n'.join(cleaned_lines).strip()

def split_text_to_sections(text: str) -> List[str]:
    """Split text into logical sections based on headings or paragraph breaks"""
    # Try to identify sections based on common heading patterns
    section_headers = [
        r'([A-Z][A-Z\s]{2,}\n)',  # ALL CAPS HEADERS
        r'(\d+\.\s+[A-Z][a-z]+.*?\n)',  # Numbered sections
        r'([A-Z][a-z]+\s+[A-Z][a-z]+.*?:)',  # Title Case Headers with colon
    ]
    
    sections = []
    current_section = ""
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    for paragraph in paragraphs:
        # Check if this paragraph is a header
        is_header = False
        for pattern in section_headers:
            if re.match(pattern, paragraph):
                # If we already have content in the current section, save it
                if current_section.strip():
                    sections.append(current_section.strip())
                    current_section = ""
                
                # Start a new section with this header
                current_section = paragraph
                is_header = True
                break
        
        if not is_header:
            # Add to the current section
            current_section += "\n" + paragraph
    
    # Add the last section if it has content
    if current_section.strip():
        sections.append(current_section.strip())
    
    # If no sections were identified, return the whole text as one section
    if not sections:
        sections = [text]
    
    return sections

def derive_tags_from_content(text: str) -> List[str]:
    """Identify relevant financial terms in the text and tag accordingly"""
    text_lower = text.lower()
    tags = []
    
    for category, terms in FINANCIAL_TERMS.items():
        for term in terms:
            if term in text_lower:
                tags.append(category)
                break
    
    return tags

def extract_context_from_table(df: pd.DataFrame, table_name: str) -> str:
    """Extract contextual information from financial tables"""
    context = []
    
    # Add table type information
    context.append(f"This table represents {table_name}.")
    
    # Handle different table types
    if "Balance Sheet" in table_name:
        # Extract key balance sheet metrics
        try:
            total_assets = df.loc[df.astype(str).apply(lambda x: x.str.contains("Total Assets", case=False)).any(axis=1)]
            total_liabilities = df.loc[df.astype(str).apply(lambda x: x.str.contains("Total Liabilities", case=False)).any(axis=1)]
            equity = df.loc[df.astype(str).apply(lambda x: x.str.contains("Equity", case=False)).any(axis=1)]
            
            for series in [total_assets, total_liabilities, equity]:
                if not series.empty:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            context.append(f"{series.index[0]} is {series[col].values[0]}.")
        except:
            pass
            
    elif "Income Statement" in table_name:
        # Extract key income statement metrics
        try:
            revenue = df.loc[df.astype(str).apply(lambda x: x.str.contains("Revenue", case=False)).any(axis=1)]
            net_profit = df.loc[df.astype(str).apply(lambda x: x.str.contains("Net Profit|Net Income", case=False)).any(axis=1)]
            
            for series in [revenue, net_profit]:
                if not series.empty:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            context.append(f"{series.index[0]} is {series[col].values[0]}.")
        except:
            pass
            
    elif "Cash Flow" in table_name:
        # Extract key cash flow metrics
        try:
            operating = df.loc[df.astype(str).apply(lambda x: x.str.contains("Operating", case=False)).any(axis=1)]
            investing = df.loc[df.astype(str).apply(lambda x: x.str.contains("Investing", case=False)).any(axis=1)]
            financing = df.loc[df.astype(str).apply(lambda x: x.str.contains("Financing", case=False)).any(axis=1)]
            
            for series in [operating, investing, financing]:
                if not series.empty:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            context.append(f"{series.index[0]} is {series[col].values[0]}.")
        except:
            pass
    
    return " ".join(context)

def extract_and_save_financial_data_rag(file_path: str, output_path: str = None) -> str:
    """
    Extract financial data from a file and save it as JSON optimized for RAG
    
    Args:
        file_path: Path to the input file
        output_path: Path to save the output JSON (optional)
        
    Returns:
        str: Path to the saved JSON file
    """
    result = extract_financial_data_rag(file_path)
    
    if output_path is None:
        # Create output filename based on input file
        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(file_path), f"{file_name}_rag_extracted.json")
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_financial_data_extractor.py <file_path> [output_path]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result_path = extract_and_save_financial_data_rag(file_path, output_path)
    print(f"Extraction complete. Results saved to: {result_path}")
