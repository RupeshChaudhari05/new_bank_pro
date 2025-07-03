from fastapi import FastAPI, UploadFile, File
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi.responses import JSONResponse

app = FastAPI()
model = ocr_predictor(pretrained=True)  # You can customize archs

@app.post("/ocr")
async def ocr(files: list[UploadFile] = File(...)):
    docs = []
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith(".pdf"):
            doc = DocumentFile.from_pdf(content)
        else:
            doc = DocumentFile.from_images(content)
        docs.append(doc)
    result = model(docs)  # `model` can process list of DocumentFile
    return JSONResponse(content=result.export())
