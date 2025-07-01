from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse
from paddleocr import PaddleOCR
import pandas as pd
import os, time, shutil, fitz, tempfile
from datetime import datetime
from threading import Thread

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='en')

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Cleanup thread
def cleanup_temp_files():
    while True:
        now = time.time()
        for filename in os.listdir(TEMP_DIR):
            path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(path) and now - os.path.getmtime(path) > 600:
                os.remove(path)
        time.sleep(600)  # Run every 10 minutes

Thread(target=cleanup_temp_files, daemon=True).start()

def extract_text_from_pdf(pdf_path: str, password: str = ""):
    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        if not doc.authenticate(password):
            raise Exception("Incorrect password")

    pages_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(tempfile.gettempdir(), f"{page.number}.png")
        pix.save(img_path)
        result = ocr.ocr(img_path, cls=True)
        text_data = []
        for line in result[0]:
            text_data.append([x[0] for x in line[0]])  # bounding box
            text_data.append(line[1][0])  # text
        pages_text.append((page.number + 1, result))
    return pages_text

def extract_tables_from_ocr(ocr_results):
    page_tables = {}
    merged_df = pd.DataFrame()
    for page_num, result in ocr_results:
        page_data = [line[1][0] for line in result[0]]
        rows = [row.split() for row in page_data if len(row.split()) > 1]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        header = tuple(df.iloc[0])
        df.columns = header
        df = df[1:]  # remove header row from data
        page_tables[page_num] = df
        if merged_df.empty:
            merged_df = df
        elif tuple(merged_df.columns) == header:
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    return page_tables, merged_df

@app.post("/extract-tables/")
async def extract_tables(background_tasks: BackgroundTasks, file: UploadFile = File(...), password: str = Form("")):
    try:
        # Save PDF
        temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{int(time.time())}_{file.filename}")
        with open(temp_pdf_path, "wb") as f:
            f.write(await file.read())

        # OCR and Table Extraction
        ocr_results = extract_text_from_pdf(temp_pdf_path, password)
        page_tables, merged_table = extract_tables_from_ocr(ocr_results)

        # Save to Excel
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        excel_path = os.path.join(TEMP_DIR, f"tables_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_path) as writer:
            for page_num, df in page_tables.items():
                df.to_excel(writer, sheet_name=f"Page_{page_num}", index=False)
            if not merged_table.empty:
                merged_table.to_excel(writer, sheet_name="Merged_Tables", index=False)

        background_tasks.add_task(lambda path=excel_path: time.sleep(600) or os.remove(path))

        return {"message": "Tables extracted", "download_link": f"/download/{os.path.basename(excel_path)}"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename)
    return {"error": "File not found"}
