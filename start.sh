#!/bin/bash

set -e

# Clone PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# Install dependencies
pip install --upgrade pip
pip install paddleocr paddlepaddle lmdb

# Download models
mkdir -p ppstructure/inference
cd ppstructure/inference

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar

cd ..

# Run the table OCR
python3 table/predict_table.py \
  --det_model_dir=inference/en_ppocr_mobile_v2.0_table_det_infer \
  --rec_model_dir=inference/en_ppocr_mobile_v2.0_table_rec_infer \
  --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer \
  --image_dir=../../table_2.png \
  --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt \
  --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
  --output ./output/table

echo "✅ OCR Done. Output saved to PaddleOCR/ppstructure/output/table"
