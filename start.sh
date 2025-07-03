#!/bin/bash

set -e

# Clone PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# Install dependencies
pip install --upgrade pip
pip install paddleocr paddlepaddle fastapi uvicorn lmdb

# Download models
mkdir -p ppstructure/inference
cd ppstructure/inference

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar

cd ../../..

# Run API
uvicorn app:app --host 0.0.0.0 --port 10000
