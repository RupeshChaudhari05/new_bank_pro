from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import subprocess
import uuid
import json
import traceback

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "PaddleOCR Table API is running."}


@app.post("/predict")
async def predict_table(file: UploadFile = File(...)):
    try:
        # Create upload and output directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        # Save uploaded image
        image_id = str(uuid.uuid4())
        image_path = f"uploads/{image_id}.png"
        output_dir = f"output/{image_id}"
        os.makedirs(output_dir, exist_ok=True)

        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Construct PaddleOCR table prediction command
        command = [
            "python3", "PaddleOCR/ppstructure/table/predict_table.py",
            "--det_model_dir", "PaddleOCR/ppstructure/inference/en_ppocr_mobile_v2.0_table_det_infer",
            "--rec_model_dir", "PaddleOCR/ppstructure/inference/en_ppocr_mobile_v2.0_table_rec_infer",
            "--table_model_dir", "PaddleOCR/ppstructure/inference/en_ppocr_mobile_v2.0_table_structure_infer",
            "--image_dir", image_path,
            "--rec_char_dict_path", "PaddleOCR/ppocr/utils/dict/table_dict.txt",
            "--table_char_dict_path", "PaddleOCR/ppocr/utils/dict/table_structure_dict.txt",
            "--output", output_dir,
            "--use_gpu=False"  # <== force CPU mode
        ]

        # Run the OCR command and capture output
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "OCR process failed",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }
            )

        # Locate result JSON file
        result_json_path = os.path.join(output_dir, f"{image_id}.png.xlsx.json")
        if not os.path.exists(result_json_path):
            return JSONResponse(
                status_code=500,
                content={"error": "OCR result not found"}
            )

        with open(result_json_path, "r") as f:
            data = json.load(f)
            return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": traceback.format_exc()
            }
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
