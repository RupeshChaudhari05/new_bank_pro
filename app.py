import os
import shutil
import uuid
import time
import logging
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pdfplumber
import pandas as pd
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to store temp files
TEMP_DIR = "/app/temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# FastAPI app setup
app = FastAPI(
    title="Advanced PDF Table Extractor API",
    description="Extract tables from PDFs using traditional methods + OCR fallback. Supports Excel, CSV, JSON, and Tally XML outputs.",
    version="3.0.0-prod"
)

# Allowed frontend domains
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:8000").split(",")

def check_origin(request: Request):
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        raise HTTPException(status_code=403, detail="No origin header.")
    if not any(origin.startswith(allowed) for allowed in ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail=f"Origin not allowed: {origin}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background cleanup job
CLEANUP_INTERVAL = 600  # 10 minutes
FILE_LIFETIME = 600     # 10 minutes

def cleanup_temp_files():
    """Delete files older than FILE_LIFETIME"""
    logger.info("Running temp file cleanup")
    now = time.time()
    for folder in os.listdir(TEMP_DIR):
        folder_path = os.path.join(TEMP_DIR, folder)
        if os.path.isdir(folder_path):
            mtime = os.path.getmtime(folder_path)
            if now - mtime > FILE_LIFETIME:
                try:
                    shutil.rmtree(folder_path)
                    logger.info(f"Deleted expired folder: {folder}")
                except Exception as e:
                    logger.error(f"Cleanup failed for {folder}: {str(e)}")

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_temp_files, 'interval', seconds=CLEANUP_INTERVAL)
scheduler.start()

# Helper: OCR-based table extraction with image enhancement
def extract_tables_with_ocr(pdf_bytes, enhance_images=True):
    """Fallback table extraction using PaddleOCR when no tables found"""
    try:
        from pdf2image import convert_from_bytes
        from paddleocr import PPStructure, draw_structure_result
        from PIL import Image, ImageEnhance
    except ImportError as e:
        logger.error("OCR dependencies not installed: " + str(e))
        return []

    tables = []
    try:
        # Convert PDF to images (use 300 DPI for better OCR accuracy)
        images = convert_from_bytes(
            pdf_bytes, 
            dpi=300,
            thread_count=4,
            poppler_path=os.getenv("POPPLER_PATH", "/usr/bin")
        )
        
        # Initialize OCR engine with optimized settings
        table_engine = PPStructure(
            show_log=False,
            ocr=True,  # Enable OCR for text recognition
            layout=True,  # Enable layout analysis
            table=True,  # Enable table recognition
            lang='en'  # English language
        )
        
        for page_num, image in enumerate(images, 1):
            try:
                # Enhance low-quality images
                if enhance_images:
                    # Increase contrast
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)
                    
                    # Increase sharpness
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(2.0)
                
                # Convert to OpenCV format
                img_np = np.array(image)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Process with PaddleOCR
                result = table_engine(img_cv)
                
                for region in result:
                    if region['type'].lower() == 'table':
                        html_str = region['res']['html']
                        try:
                            # Parse HTML table to DataFrame
                            dfs = pd.read_html(html_str)
                            if dfs:
                                df = dfs[0]
                                
                                # Clean up OCR artifacts
                                df = df.replace(r'^\s*$', np.nan, regex=True)
                                df = df.dropna(how='all').reset_index(drop=True)
                                
                                tables.append({
                                    "page": page_num,
                                    "data": df,
                                    "method": "ocr"
                                })
                        except Exception as e:
                            logger.warning(f"Table parsing failed on page {page_num}: {str(e)}")
            except Exception as page_e:
                logger.error(f"OCR processing failed on page {page_num}: {str(page_e)}")
        
        return tables
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        return []

# Supported output formats
SUPPORTED_FORMATS = ["html", "excel", "csv", "json", "tallyxml"]

def extract_balances(tables):
    """Extract opening and closing balances from financial tables"""
    if not tables:
        return None, None
    
    # Try to find a table with balance information
    for table in tables:
        df = table['data']
        if df.empty:
            continue
            
        # Find potential balance column
        balance_col = None
        for col in df.columns:
            if col and any(keyword in str(col).lower() for keyword in ['balance', 'bal']):
                balance_col = col
                break
                
        if balance_col:
            try:
                # Clean balance values (remove non-numeric characters)
                df[balance_col] = df[balance_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[balance_col] = pd.to_numeric(df[balance_col], errors='coerce')
                
                # Find first and last valid balances
                valid_balances = df[balance_col].dropna()
                if len(valid_balances) > 1:
                    return valid_balances.iloc[0], valid_balances.iloc[-1]
            except Exception:
                continue
                
    return None, None

def to_tally_xml(tables):
    """Convert table data to Tally XML format"""
    if not tables:
        return ""
    
    # Use the first table that looks like a financial statement
    for table in tables:
        df = table['data']
        if df.empty:
            continue
            
        # Identify columns by common financial headers
        col_mapping = {}
        for col in df.columns:
            if not col:
                continue
            lcol = str(col).lower()
            if 'date' in lcol:
                col_mapping['date'] = col
            elif 'desc' in lcol or 'particular' in lcol or 'narration' in lcol:
                col_mapping['desc'] = col
            elif 'debit' in lcol:
                col_mapping['debit'] = col
            elif 'credit' in lcol:
                col_mapping['credit'] = col
            elif 'balance' in lcol:
                col_mapping['balance'] = col
                
        # Require at least date and description
        if 'date' in col_mapping and 'desc' in col_mapping:
            break
    else:
        # No suitable table found
        return ""
    
    # Build XML structure
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ENVELOPE>',
        ' <HEADER>',
        '  <TALLYREQUEST>Import Data</TALLYREQUEST>',
        ' </HEADER>',
        ' <BODY>',
        '  <IMPORTDATA>',
        '   <REQUESTDESC>',
        '    <REPORTNAME>Vouchers</REPORTNAME>',
        '   </REQUESTDESC>',
        '   <REQUESTDATA>',
    ]
    
    # Add voucher entries
    for _, row in df.iterrows():
        date_val = str(row[col_mapping['date']]) if 'date' in col_mapping else ''
        desc_val = str(row[col_mapping['desc']]) if 'desc' in col_mapping else ''
        debit_val = str(row[col_mapping['debit']]) if 'debit' in col_mapping else '0'
        credit_val = str(row[col_mapping['credit']]) if 'credit' in col_mapping else '0'
        balance_val = str(row[col_mapping['balance']]) if 'balance' in col_mapping else ''
        
        # Clean numeric values
        for val in [debit_val, credit_val, balance_val]:
            val = ''.join(filter(lambda x: x.isdigit() or x in ['.', '-'], val))
        
        xml_lines += [
            '    <TALLYMESSAGE>',
            '     <VOUCHER VCHTYPE="Bank Statement" ACTION="Create">',
            f'      <DATE>{date_val}</DATE>',
            f'      <NARRATION>{desc_val}</NARRATION>',
        ]
        
        if debit_val and float(debit_val) > 0:
            xml_lines.append(f'      <DEBIT>{debit_val}</DEBIT>')
        if credit_val and float(credit_val) > 0:
            xml_lines.append(f'      <CREDIT>{credit_val}</CREDIT>')
        if balance_val:
            xml_lines.append(f'      <BALANCE>{balance_val}</BALANCE>')
            
        xml_lines += [
            '     </VOUCHER>',
            '    </TALLYMESSAGE>'
        ]
    
    xml_lines += [
        '   </REQUESTDATA>',
        '  </IMPORTDATA>',
        ' </BODY>',
        '</ENVELOPE>'
    ]
    
    return '\n'.join(xml_lines)

def extract_and_save(pdf_bytes, out_dir, password=None, file_map=None):
    """Main extraction function with fallback to OCR"""
    tables = []
    unique_tables = {}
    non_blank_pages = set()
    extraction_method = "pdfplumber"
    ocr_fallback = False
    
    # First attempt: Traditional PDF extraction
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes), password=password) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    found_table = False
                    for table in page.find_tables():
                        data = table.extract()
                        if data and len(data) > 1:
                            df = pd.DataFrame(data[1:], columns=data[0])
                            
                            # Basic cleaning
                            df = df.replace(r'^\s*$', np.nan, regex=True)
                            df = df.dropna(how='all').reset_index(drop=True)
                            
                            if not df.empty:
                                tables.append({
                                    "page": page_num,
                                    "data": df,
                                    "method": "traditional"
                                })
                                non_blank_pages.add(page_num)
                                found_table = True
                                
                                # Track for merged CSV
                                headers_key = tuple(df.columns)
                                if headers_key not in unique_tables:
                                    unique_tables[headers_key] = []
                                unique_tables[headers_key].append(df)
                    
                    if found_table:
                        non_blank_pages.add(page_num)
                except Exception as page_e:
                    logger.error(f"Page {page_num} processing error: {str(page_e)}")
    except Exception as e:
        logger.warning(f"PDF extraction failed: {str(e)}")
    
    # Fallback to OCR if no tables found
    if not tables:
        logger.info("No tables found traditionally. Attempting OCR extraction...")
        ocr_tables = extract_tables_with_ocr(pdf_bytes, enhance_images=True)
        if ocr_tables:
            tables = ocr_tables
            extraction_method = "paddleocr"
            ocr_fallback = True
            
            # Process OCR-extracted tables
            for table in tables:
                page_num = table["page"]
                df = table["data"]
                non_blank_pages.add(page_num)
                
                # Track for merged CSV
                headers_key = tuple(df.columns)
                if headers_key not in unique_tables:
                    unique_tables[headers_key] = []
                unique_tables[headers_key].append(df)
    
    if not tables:
        return 0, 0, None, None, extraction_method, False
    
    # Use file_map for output names if provided
    if file_map is None:
        file_map = {
            "html": "tables.html",
            "excel": "tables.xlsx",
            "csv": "tables.csv",
            "json": "tables.json",
            "tallyxml": "tables_tally.xml"
        }
    
    # Save HTML
    html = ""
    for i, t in enumerate(tables):
        html += f"<h3>Page {t['page']} - Table {i+1}</h3>"
        html += t['data'].to_html(index=False, border=1, classes="table table-striped")
    with open(os.path.join(out_dir, file_map["html"]), "w", encoding="utf-8") as f:
        f.write(f"<html><body>{html}</body></html>")
    
    # Save Excel
    with pd.ExcelWriter(os.path.join(out_dir, file_map["excel"]), engine='xlsxwriter') as writer:
        for i, t in enumerate(tables):
            sheet_name = f"Pg{t['page']}_T{i+1}"[:31]  # Excel sheet name limit
            t['data'].to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Save CSV (merge tables with same headers)
    with open(os.path.join(out_dir, file_map["csv"]), "w", encoding="utf-8") as f:
        for headers, dfs in unique_tables.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(f, index=False)
            f.write("\n\n")
    
    # Save JSON
    json_data = []
    for i, t in enumerate(tables):
        json_data.append({
            "table": i+1,
            "page": t['page'],
            "method": t.get("method", "unknown"),
            "columns": list(t['data'].columns),
            "rows": t['data'].to_dict(orient='records')
        })
    import json
    with open(os.path.join(out_dir, file_map["json"]), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # Save Tally XML
    tally_xml = to_tally_xml(tables)
    with open(os.path.join(out_dir, file_map["tallyxml"]), "w", encoding="utf-8") as f:
        f.write(tally_xml)
    
    # Extract balances
    opening, closing = extract_balances(tables)
    
    return len(tables), len(non_blank_pages), opening, closing, extraction_method, ocr_fallback

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(None),
    request: Request = None,
    _: None = Depends(check_origin)
):
    """Handle PDF uploads and process extraction"""
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "error_code": "INVALID_FILE_TYPE",
                "message": "Only PDF files are allowed."
            }
        )
    
    try:
        pdf_bytes = await file.read()
        file_id = str(uuid.uuid4())
        out_dir = os.path.join(TEMP_DIR, file_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # Create output file names based on original filename
        base_name = os.path.splitext(file.filename)[0].replace(" ", "_")
        file_map = {
            "html": f"{base_name}.html",
            "excel": f"{base_name}.xlsx", 
            "csv": f"{base_name}.csv",
            "json": f"{base_name}.json",
            "tallyxml": f"{base_name}_tally.xml"
        }
        
        # Save original PDF
        with open(os.path.join(out_dir, "original.pdf"), "wb") as f:
            f.write(pdf_bytes)
        
        # Process PDF
        try:
            tables_found, pages_count, opening_balance, closing_balance, method, ocr_fallback = extract_and_save(
                pdf_bytes, out_dir, password=password, file_map=file_map
            )
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            shutil.rmtree(out_dir)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": "PROCESSING_ERROR",
                    "message": "Failed to process PDF",
                    "details": str(e)
                }
            )
        
        if tables_found == 0:
            shutil.rmtree(out_dir)
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error_code": "NO_TABLES_FOUND",
                    "message": "No tables found in the PDF.",
                    "pages_count": pages_count
                }
            )
        
        # Create download links
        links = {fmt: f"/download/{file_id}/{fmt}" for fmt in SUPPORTED_FORMATS}
        
        return {
            "success": True,
            "tables_found": tables_found,
            "pages_count": pages_count,
            "file_id": file_id,
            "extraction_method": method,
            "ocr_fallback": ocr_fallback,
            "download_links": links,
            "output_file_names": file_map,
            "opening_balance": str(opening_balance) if opening_balance else None,
            "closing_balance": str(closing_balance) if closing_balance else None
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "SERVER_ERROR",
                "message": "Internal server error"
            }
        )

@app.get("/download/{file_id}/{fmt}")
def download_file(file_id: str, fmt: str):
    """Serve extracted files for download"""
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid format.")
    
    # Prevent path traversal
    safe_id = file_id.replace("..", "").replace("/", "")
    out_dir = os.path.join(TEMP_DIR, safe_id)
    
    if not os.path.exists(out_dir):
        raise HTTPException(status_code=404, detail="File not found or expired.")
    
    # Find matching file
    ext_map = {
        "html": ".html",
        "excel": ".xlsx", 
        "csv": ".csv",
        "json": ".json",
        "tallyxml": "_tally.xml"
    }
    
    target_file = None
    for file in os.listdir(out_dir):
        if file.endswith(ext_map[fmt]):
            target_file = file
            break
    
    if not target_file:
        raise HTTPException(status_code=404, detail="Requested format not found.")
    
    file_path = os.path.join(out_dir, target_file)
    
    # Set appropriate media types
    media_types = {
        "html": "text/html",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "json": "application/json",
        "tallyxml": "application/xml"
    }
    
    return FileResponse(
        file_path,
        media_type=media_types[fmt],
        filename=target_file
    )

@app.get("/health")
def health_check():
    """Endpoint for health checks"""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/")
def root():
    """API information endpoint"""
    return {
        "message": "Advanced PDF Table Extractor API",
        "version": app.version,
        "endpoints": {
            "POST /upload": "Upload PDF for table extraction",
            "GET /download/{file_id}/{format}": "Download extracted tables",
            "GET /health": "Service health check"
        }
    }
