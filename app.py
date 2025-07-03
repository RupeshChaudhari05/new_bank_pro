from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import subprocess
import uuid

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "PaddleOCR Table API is running."}

@app.post("/predict")
async def predict_table(file: UploadFile = File(...)):
    # Save uploaded image
    image_id = str(uuid.uuid4())
    image_path = f"uploads/{image_id}.png"
    os.makedirs("uploads", exist_ok=True)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run PaddleOCR on the image
    output_dir = f"output/{image_id}"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python3", "PaddleOCR/ppstructure/table/predict_table.py",
        "--det_model_dir", "PaddleOCR/ppstructure/inference/en_ppocr_mobile_v2.0_table_det_infer",
        "--rec_model_dir", "PaddleOCR/ppstructure/inference/en_ppocr_mobile_v2.0_table_rec_infer",
        "--table_model_dir", "PaddleOCR/ppstructure/inference/en_ppocr_mobile_v2.0_table_structure_infer",
        "--image_dir", image_path,
        "--rec_char_dict_path", "PaddleOCR/ppocr/utils/dict/table_dict.txt",
        "--table_char_dict_path", "PaddleOCR/ppocr/utils/dict/table_structure_dict.txt",
        "--output", output_dir
    ]

    subprocess.run(command)

    result_json = os.path.join(output_dir, f"{image_id}.png.xlsx.json")
    if not os.path.exists(result_json):
        return JSONResponse(content={"error": "Failed to process image"}, status_code=500)

    with open(result_json, "r") as f:
        return JSONResponse(content=f.read())

