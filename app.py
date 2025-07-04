from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os
import subprocess

app = FastAPI()

@app.post("/extract-tables")
async def extract_tables(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    # Save uploaded file
    input_filename = f"{uuid.uuid4()}.pdf"
    input_path = f"/tmp/{input_filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_dir = f"/tmp/output_{uuid.uuid4()}/"
    os.makedirs(output_dir, exist_ok=True)

    # Call your script with CLI arguments
    cmd = [
        "python3",
        "demo.py",
        "--input_path", os.path.dirname(input_path) + '/',
        "--output_path", output_dir
    ]
    subprocess.run(cmd, check=True)

    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    return {
        "message": "Extraction successful",
        "csv_files": csv_files,
        "output_dir": output_dir
    }
